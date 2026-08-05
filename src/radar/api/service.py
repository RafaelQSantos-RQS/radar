"""Serviço — orquestra motor + dados para os endpoints.

Separa a lógica de negócio (avaliar sinais, rodar backtest) das rotas
HTTP, mantendo as funções puras e testáveis.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from radar.backtest.runner import run_trade_backtest
from radar.config import DB_PATH
from radar.data.ingest import get_candles
from radar.indicators import compute_all
from radar.metrics import maximum_drawdown, sharpe_ratio
from radar.strategy.defaults import BUILT_IN_STRATEGIES, get_strategy

__all__ = ["evaluate_signal", "list_strategies", "run_backtest"]


def list_strategies() -> list[dict]:
    """Retorna as estratégias built-in no formato da API."""
    return [
        {
            "id": s["name"],
            "name": s["name"],
            "description": s["description"],
            "rules": s["params"]["rules"],
        }
        for s in BUILT_IN_STRATEGIES
    ]


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


def evaluate_signal(
    symbol: str,
    timeframe: str,
    strategy: str,
    candles: list[dict] | None = None,
    db_path=None,
) -> dict:
    """Avalia as regras da estratégia na última barra.

    Returns:
        Dict com ``signal`` (COMPRA|VENDA|NEUTRO), ``confidence``,
        ``justification`` e ``at``.
    """
    s = get_strategy(strategy)
    df = _prepare_frame(candles, symbol, timeframe, _resolve_db(db_path))

    if df.empty:
        raise ValueError("nenhuma barra disponível para avaliar")

    from radar.strategy.engine import evaluate_rules

    last = df.iloc[-1]
    signals = evaluate_rules(df, s["params"]["rules"])

    # Prioriza o primeiro sinal direcional (buy/sell); senão NEUTRO.
    directional = next((sig for sig in signals if sig["direction"] in ("buy", "sell")), None)

    if directional is None:
        return {
            "signal": "NEUTRO",
            "confidence": 0.5,
            "justification": signals[0]["reason"] if signals else "sem condições disparadas",
            "at": last["time"],
        }

    return {
        "signal": "COMPRA" if directional["direction"] == "buy" else "VENDA",
        "confidence": directional["confidence"],
        "justification": directional["reason"],
        "at": last["time"],
    }


def run_backtest(
    strategy: str,
    symbol: str,
    timeframe: str,
    start: datetime | None = None,
    end: datetime | None = None,
    costs: bool = True,
    db_path=None,
) -> dict:
    """Roda o TradeSim com a estratégia e devolve métricas.

    Returns:
        Dict com ``strategy``, ``symbol``, ``timeframe``, ``trades``,
        ``net_profit``, ``sharpe``, ``max_drawdown``, ``win_rate`` e
        ``equity_curve`` (lista de valores).
    """
    s = get_strategy(strategy)
    df = get_candles(symbol, timeframe, _resolve_db(db_path), start=start, end=end)
    if df.empty:
        raise ValueError(f"sem candles em {symbol} {timeframe} — ingira dados primeiro")

    df = df.rename(columns={"ts": "time"})
    df = compute_all(df)
    result = run_trade_backtest(df, s["params"]["rules"], symbol=symbol, use_costs=costs)

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

    return {
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": timeframe,
        "trades": summary["total_trades"],
        "net_profit": summary["net_profit"],
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": summary["win_rate"],
        "equity_curve": eq["equity"].round(2).tolist() if not eq.empty else [],
    }
