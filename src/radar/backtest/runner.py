"""Runner — adapta as estratégias rule-based para o TradeSim.

Converte regras (formato ``strategy/defaults.py``) em função
``(idx, df) -> signal`` compatível com o TradeSim, calculando
SL/TP automáticos baseados em ATR quando não explicitamente definidos.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from radar.backtest.engine import TradeSim
from radar.strategy.engine import evaluate_rule

__all__ = ["build_strategy_fn", "run_trade_backtest"]


def _calc_sl_tp(
    row: pd.Series,
    direction: str,
    atr_mult_sl: float = 2.0,
    atr_mult_tp: float = 3.0,
    sl_fixed: float = 0.0,
    tp_fixed: float = 0.0,
) -> tuple[float, float]:
    """Calcula SL e TP — prioridade para valores fixos, fallback para ATR.

    Returns:
        Tupla ``(sl, tp)`` em pontos.
    """
    close = float(row["close"])

    if sl_fixed > 0:
        sl = close - sl_fixed if direction == "buy" else close + sl_fixed
    elif atr_mult_sl > 0:
        atr = row.get("atr_14", None)
        if pd.notna(atr) and atr is not None:
            sl = close - (float(atr) * atr_mult_sl) if direction == "buy" else close + (float(atr) * atr_mult_sl)
        else:
            sl = 0.0
    else:
        sl = 0.0

    if tp_fixed > 0:
        tp = close + tp_fixed if direction == "buy" else close - tp_fixed
    elif atr_mult_tp > 0:
        atr = row.get("atr_14", None)
        if pd.notna(atr) and atr is not None:
            tp = close + (float(atr) * atr_mult_tp) if direction == "buy" else close - (float(atr) * atr_mult_tp)
        else:
            tp = 0.0
    else:
        tp = 0.0

    return sl, tp


def build_strategy_fn(
    rules: list[dict],
    atr_mult_sl: float = 2.0,
    atr_mult_tp: float = 3.0,
    sl_fixed: float = 0.0,
    tp_fixed: float = 0.0,
    volume: float = 1.0,
) -> Callable[[int, pd.DataFrame], dict]:
    """Cria uma função de estratégia compatível com TradeSim.

    A função resultante avalia as regras na barra atual e retorna um
    sinal com SL/TP calculados por ATR.

    Returns:
        Função ``(idx, df) -> dict`` para usar no ``TradeSim.run()``.
    """
    def strategy_fn(idx: int, df: pd.DataFrame) -> dict:
        if idx < 0 or idx >= len(df):
            return {"direction": "hold"}

        row = df.iloc[idx]

        for rule in rules:
            if not evaluate_rule(row, rule):
                continue

            direction = rule.get("then", "hold")
            if direction == "hold":
                continue

            sl, tp = _calc_sl_tp(row, direction, atr_mult_sl, atr_mult_tp, sl_fixed, tp_fixed)

            return {
                "direction": direction,
                "sl": sl,
                "tp": tp,
                "volume": volume,
            }

        return {"direction": "hold"}

    return strategy_fn


def run_trade_backtest(
    df: pd.DataFrame,
    rules: list[dict],
    symbol: str = "WDO$",
    initial_balance: float = 10_000.0,
    warmup: int = 60,
    atr_mult_sl: float = 2.0,
    atr_mult_tp: float = 3.0,
    sl_fixed: float = 0.0,
    tp_fixed: float = 0.0,
    volume: float = 1.0,
    trailing_distance: float = 0.0,
    use_costs: bool = True,
    max_positions: int = 1,
) -> dict[str, Any]:
    """Atalho: cria TradeSim + strategy_fn e executa.

    Returns:
        Dict com resultado do TradeSim.
    """
    strategy_fn = build_strategy_fn(
        rules=rules,
        atr_mult_sl=atr_mult_sl,
        atr_mult_tp=atr_mult_tp,
        sl_fixed=sl_fixed,
        tp_fixed=tp_fixed,
        volume=volume,
    )

    sim = TradeSim(
        initial_balance=initial_balance,
        symbol=symbol,
        trailing_distance=trailing_distance,
        use_costs=use_costs,
        max_positions=max_positions,
    )

    return sim.run(df, strategy_fn, warmup=warmup)
