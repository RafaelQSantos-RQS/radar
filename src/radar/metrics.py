"""Métricas financeiras para avaliação de estratégias.

Todas as funções são **puras** — recebem Series/listas numéricas e
retornam valores calculados, sem efeitos colaterais e sem I/O.

Desenhado para minicontratos (WDO$, WIN$), mas agnóstico de ativo.
"""

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "annualization_periods",
    "calmar_ratio",
    "expected_payoff",
    "maximum_drawdown",
    "profit_factor",
    "recovery_factor",
    "sharpe_ratio",
    "sortino_ratio",
]


# ── Anualização ──────────────────────────────────────────


def annualization_periods(index: pd.Index) -> int:
    """Períodos por ano inferidos do índice temporal dos dados.

    Computa a mediana de barras por dia e multiplica por 252 dias de
    negociação. Assim a anualização do Sharpe fica correta para qualquer
    timeframe (M1 ≈ 1440·252, não 252 fixo — bug de ~38× em M1).
    """
    if len(index) < 2:
        return 252
    idx = pd.DatetimeIndex(index)
    per_day = pd.Series(idx).groupby(idx.normalize()).size()
    return round(per_day.median() * 252)


# ── Sharpe Ratio ────────────────────────────────────────────


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int | None = None,
) -> float:
    """Sharpe Ratio anualizado.

    ``Sharpe = (E[R] - Rf) / sigma * sqrt(periods_per_year)``

    Se ``periods_per_year`` for None, infere do índice de ``returns``
    (veja :func:`annualization_periods`). Para minicontratos intradiários
    ``risk_free_rate=0`` é aceitável.

    Returns:
        Sharpe anualizado. 0.0 se volatilidade for zero.
    """
    if len(returns) < 2:
        return 0.0

    excess = returns - risk_free_rate
    std = excess.std()

    if std == 0 or pd.isna(std):
        return 0.0

    if periods_per_year is None:
        periods_per_year = annualization_periods(returns.index)

    return float((excess.mean() / std) * np.sqrt(periods_per_year))


# ── Sortino Ratio ────────────────────────────────────────────


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int | None = None,
) -> float:
    """Sortino Ratio anualizado — só penaliza o downside.

    Returns:
        Sortino anualizado. 0.0 se downside for zero.
    """
    if len(returns) < 2:
        return 0.0

    excess = returns - risk_free_rate
    downside = returns[returns < risk_free_rate]

    if len(downside) < 2:
        return 0.0

    downside_std = downside.std()
    if downside_std == 0 or pd.isna(downside_std):
        return 0.0

    if periods_per_year is None:
        periods_per_year = annualization_periods(returns.index)

    return float((excess.mean() / downside_std) * np.sqrt(periods_per_year))


# ── Maximum Drawdown ─────────────────────────────────────────


def maximum_drawdown(equity_curve: pd.Series) -> dict[str, Any]:
    """Maximum Drawdown (queda do pico ao vale).

    Args:
        equity_curve: Série temporal do patrimônio (equity), indexada por data/hora.

    Returns:
        Dict com ``max_drawdown_pct``, ``max_drawdown_abs``, ``peak_time``,
        ``valley_time``, ``recovery_time`` e ``duration_bars``.
    """
    empty = {
        "max_drawdown_pct": 0.0,
        "max_drawdown_abs": 0.0,
        "peak_time": None,
        "valley_time": None,
        "recovery_time": None,
        "duration_bars": 0,
    }
    if len(equity_curve) < 2:
        return empty

    rolling_max = equity_curve.cummax()

    drawdown_abs = rolling_max - equity_curve
    drawdown_pct = drawdown_abs / rolling_max

    max_dd_idx = drawdown_pct.idxmax()
    max_dd_pct = float(drawdown_pct.max())
    max_dd_abs = float(drawdown_abs.loc[max_dd_idx])

    # Encontra o pico: última barra ANTES do vale em que o equity
    # efetivamente tocou o máximo (corrige: pegar o último índice do
    # rolling max aponta para a própria barra do vale em caso de platô).
    at_peak = equity_curve.loc[:max_dd_idx] == rolling_max.loc[:max_dd_idx]
    peak_idx = at_peak[at_peak].index[-1]

    recovery_idx = None
    post_valley = equity_curve.loc[max_dd_idx:]
    recovered = post_valley[post_valley >= rolling_max.loc[peak_idx]]
    if not recovered.empty:
        recovery_idx = recovered.index[0]

    if recovery_idx is not None:
        duration = len(equity_curve.loc[peak_idx:recovery_idx]) - 1
    else:
        duration = len(equity_curve.loc[peak_idx:]) - 1

    return {
        "max_drawdown_pct": round(max_dd_pct, 6),
        "max_drawdown_abs": round(max_dd_abs, 2),
        "peak_time": peak_idx,
        "valley_time": max_dd_idx,
        "recovery_time": recovery_idx,
        "duration_bars": duration,
    }


# ── Calmar Ratio ──────────────────────────────────────────────


def calmar_ratio(
    total_return_pct: float,
    max_drawdown_pct: float,
) -> float:
    """Calmar Ratio — retorno total / max drawdown (valor absoluto)."""
    denom = abs(max_drawdown_pct)
    if denom == 0:
        return 0.0
    return round(total_return_pct / denom, 4)


# ── Profit Factor ─────────────────────────────────────────────


def profit_factor(trades: list[dict]) -> float:
    """Profit Factor — soma dos lucros / soma das perdas.

    Trades são dicts com chave ``pnl``. Retorna 99.999 se não houver
    perdas (e houver lucros), 0.0 se não houver trades.
    """
    if not trades:
        return 0.0

    gross_profit = sum(t["pnl"] for t in trades if t.get("pnl", 0) > 0)
    gross_loss = sum(abs(t["pnl"]) for t in trades if t.get("pnl", 0) < 0)

    if gross_loss == 0:
        return 99.999 if gross_profit > 0 else 0.0

    return round(gross_profit / gross_loss, 4)


# ── Recovery Factor ───────────────────────────────────────────


def recovery_factor(
    total_profit: float,
    max_drawdown_abs: float,
) -> float:
    """Recovery Factor — lucro líquido / max drawdown absoluto."""
    if max_drawdown_abs == 0:
        return 0.0
    return round(total_profit / max_drawdown_abs, 4)


# ── Expected Payoff (Expectativa Matemática) ───────────────────


def expected_payoff(trades: list[dict]) -> dict[str, float]:
    """Expectativa matemática por operação: E = wr·avg_win - (1-wr)·|avg_loss|."""
    empty = {
        "expected_payoff": 0.0,
        "win_rate": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "total_trades": 0,
    }
    if not trades:
        return empty

    winners = [t["pnl"] for t in trades if t.get("pnl", 0) > 0]
    losers = [t["pnl"] for t in trades if t.get("pnl", 0) < 0]
    total = len(trades)

    wr = len(winners) / total
    avg_w = float(np.mean(winners)) if winners else 0.0
    avg_l = float(np.mean(losers)) if losers else 0.0

    expectation = (wr * avg_w) - ((1 - wr) * abs(avg_l))

    return {
        "expected_payoff": round(expectation, 4),
        "win_rate": round(wr, 4),
        "avg_win": round(avg_w, 2),
        "avg_loss": round(avg_l, 2),
        "total_trades": total,
    }
