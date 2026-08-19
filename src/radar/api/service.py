"""Serviço — orquestra motor + dados para os endpoints.

Separa a lógica de negócio (avaliar sinais, rodar backtest) das rotas
HTTP, mantendo as funções puras e testáveis.

Estratégia única: QuantScore (implícita — não há campo ``strategy``).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from radar.backtest.costs import CostParams
from radar.backtest.engine import TradeSim
from radar.benchmarks import benchmark_returns
from radar.config import DB_PATH
from radar.data.ingest import get_candles
from radar.indicators import compute_all
from radar.metrics import maximum_drawdown, sharpe_ratio
from radar.strategy.quantscore import (
    QuantScoreParams,
    compute_score,
    quantscore_strategy,
)

__all__ = ["evaluate_signal", "run_backtest"]


def _prepare_frame(candles: list[dict] | None, symbol: str, timeframe: str, db_path) -> pd.DataFrame:
    """Monta DataFrame com indicadores: candles inline ou leitura do banco.

    O motor espera colunas ``time, open, high, low, close, volume``.
    """
    if candles:
        df = pd.DataFrame(candles)
        df = df.sort_values("ts")
    else:
        df = get_candles(symbol, timeframe, db_path)
        if df.empty:
            raise ValueError(f"sem candles em {symbol} {timeframe} — ingira dados primeiro")

    df = df.rename(columns={"ts": "time"})
    missing = {"time", "open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"colunas OHLC/ts ausentes: {sorted(missing)}")
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    if df.empty:
        raise ValueError("nenhuma barra válida após normalização")

    return compute_all(df)


def _resolve_db(db_path) -> Path:
    """Resolve o caminho do banco em runtime (permite monkeypatch nos testes)."""
    return db_path if db_path is not None else DB_PATH


def _validate_params(params: QuantScoreParams) -> None:
    """Valida faixas plausíveis dos parâmetros (spec api/backtest)."""
    if params.stop_inicial <= 0:
        raise ValueError("stop_inicial deve ser > 0")
    if params.take_inicial <= 0:
        raise ValueError("take_inicial deve ser > 0")
    for name, value in (
        ("min_score_compra", params.min_score_compra),
        ("max_score_venda", params.max_score_venda),
    ):
        if not (0 <= value <= 100):
            raise ValueError(f"{name} deve estar em [0, 100]")
    for name, value in (
        ("ang_min_compra", params.ang_min_compra),
        ("ang_max_compra", params.ang_max_compra),
        ("ang_min_venda", params.ang_min_venda),
        ("ang_max_venda", params.ang_max_venda),
    ):
        if not (0 <= value <= 90):
            raise ValueError(f"{name} deve estar em [0, 90]")


def evaluate_signal(
    symbol: str,
    timeframe: str,
    params: dict | None = None,
    candles: list[dict] | None = None,
    db_path=None,
) -> dict:
    """Avalia o QuantScore na última barra.

    Returns:
        Dict com ``signal`` (COMPRA|VENDA|NEUTRO), ``score``,
        ``components``, ``confidence``, ``justification`` e ``at``.
    """
    p = QuantScoreParams.from_dict(params)
    _validate_params(p)

    df = _prepare_frame(candles, symbol, timeframe, _resolve_db(db_path))
    if df.empty:
        raise ValueError("nenhuma barra disponível para avaliar")

    last = df.iloc[-1]
    comp = compute_score(df, lookback=p.lookback)
    score = comp["score_final"]
    angulo = comp["angulo"]

    # IBOV não é alvo de entrada
    if symbol.upper() in p.ibov_symbols:
        return {
            "signal": "NEUTRO",
            "score": score,
            "components": comp,
            "confidence": 0.5,
            "justification": "IBOV não é alvo de entrada (benchmark)",
            "at": last["time"],
        }

    if score == -1.0:
        return {
            "signal": "NEUTRO",
            "score": score,
            "components": comp,
            "confidence": 0.5,
            "justification": "evento corporativo detectado (variação > 35%) — entrada bloqueada",
            "at": last["time"],
        }

    if p.ang_min_compra <= angulo <= p.ang_max_compra and score >= p.min_score_compra:
        return {
            "signal": "COMPRA",
            "score": score,
            "components": comp,
            "confidence": round(min(score / 100.0, 1.0), 4),
            "justification": (
                f"ângulo {angulo:.1f}° em [{p.ang_min_compra}, {p.ang_max_compra}] "
                f"e score {score:.1f} >= {p.min_score_compra}"
            ),
            "at": last["time"],
        }

    if -p.ang_max_venda <= angulo <= -p.ang_min_venda and score <= p.max_score_venda:
        return {
            "signal": "VENDA",
            "score": score,
            "components": comp,
            "confidence": round(min((100 - score) / 100.0, 1.0), 4),
            "justification": (
                f"ângulo {angulo:.1f}° em [{-p.ang_max_venda}, {-p.ang_min_venda}] "
                f"e score {score:.1f} <= {p.max_score_venda}"
            ),
            "at": last["time"],
        }

    return {
        "signal": "NEUTRO",
        "score": score,
        "components": comp,
        "confidence": 0.5,
        "justification": f"ângulo {angulo:.1f}° fora dos intervalos ou score {score:.1f} fora das faixas",
        "at": last["time"],
    }


def run_backtest(
    symbol: str,
    timeframe: str,
    start: datetime | None = None,
    end: datetime | None = None,
    params: dict | None = None,
    costs_params: dict | None = None,
    db_path=None,
) -> dict:
    """Roda o TradeSim com QuantScore e devolve métricas + benchmarks.

    Returns:
        Dict com ``symbol``, ``timeframe``, ``trades``, ``net_profit``,
        ``sharpe``, ``max_drawdown``, ``win_rate``, ``profit_factor``,
        ``total_costs``, ``equity_curve``, ``benchmarks`` e ``params``.
    """
    p = QuantScoreParams.from_dict(params)
    _validate_params(p)
    cost_params = CostParams.from_dict(costs_params)

    df = get_candles(symbol, timeframe, _resolve_db(db_path), start=start, end=end)
    if df.empty:
        raise ValueError(f"sem candles em {symbol} {timeframe} — ingira dados primeiro")

    df = df.rename(columns={"ts": "time"})
    df = compute_all(df)

    strategy_fn = quantscore_strategy(p)
    sim = TradeSim(
        initial_balance=10_000.0,
        symbol=symbol,
        use_costs=True,
        cost_params=cost_params,
    )
    result = sim.run(df, strategy_fn, warmup=p.lookback)

    summary = result["summary"]
    eq = result["equity_curve"]

    # Sharpe e drawdown a partir da equity_curve (índice temporal).
    sharpe = 0.0
    max_dd = 0.0
    if not eq.empty:
        equity = eq.set_index("time")["equity"]
        returns = equity.pct_change().dropna()
        if len(returns) > 1:
            sharpe = round(sharpe_ratio(returns), 4)
        max_dd = round(maximum_drawdown(equity)["max_drawdown_pct"], 4)

    benchmarks = benchmark_returns(start, end)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "trades": result["trades"],
        "net_profit": summary["net_profit"],
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": summary["win_rate"],
        "profit_factor": summary["profit_factor"],
        "total_costs": summary["total_costs"],
        "equity_curve": eq["equity"].round(2).tolist() if not eq.empty else [],
        "benchmarks": benchmarks,
        "params": p.__dict__,
    }