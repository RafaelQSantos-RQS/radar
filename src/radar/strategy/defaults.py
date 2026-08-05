"""Estratégias built-in do Radar.

Dados puros — sem efeitos colaterais. Cada estratégia é um dict com
``name``, ``description`` e ``params.rules`` (lista de regras).

Cada regra é um dict com ``name``, ``conditions``, ``logic``, ``then``
e ``confidence``. Cada condição é um dict com ``indicator``, ``op``,
``value`` (literal) e/ou ``indicator_ref`` (comparação entre colunas).
"""

from typing import Any

__all__ = ["BUILT_IN_STRATEGIES", "get_strategy"]

BUILT_IN_STRATEGIES: list[dict[str, Any]] = [
    {
        "name": "rsi_ma",
        "description": "RSI + EMA(9) — oversold/overbought with trend filter",
        "params": {
            "rules": [
                {
                    "name": "rsi_oversold_bounce",
                    "conditions": [
                        {"indicator": "rsi_14", "op": "<", "value": 30},
                        {"indicator": "close", "op": ">", "indicator_ref": "ema_9"},
                    ],
                    "logic": "AND",
                    "then": "buy",
                    "confidence": 0.65,
                },
                {
                    "name": "rsi_overbought_reject",
                    "conditions": [
                        {"indicator": "rsi_14", "op": ">", "value": 70},
                        {"indicator": "close", "op": "<", "indicator_ref": "ema_9"},
                    ],
                    "logic": "AND",
                    "then": "sell",
                    "confidence": 0.65,
                },
            ],
        },
    },
    {
        "name": "macd",
        "description": "MACD line vs Signal line crossover",
        "params": {
            "rules": [
                {
                    "name": "macd_cross_up",
                    "conditions": [
                        {"indicator": "macd", "op": ">", "indicator_ref": "macd_signal"},
                        {"indicator": "macd_hist", "op": ">=", "value": 0},
                    ],
                    "logic": "AND",
                    "then": "buy",
                    "confidence": 0.7,
                },
                {
                    "name": "macd_cross_down",
                    "conditions": [
                        {"indicator": "macd", "op": "<", "indicator_ref": "macd_signal"},
                        {"indicator": "macd_hist", "op": "<=", "value": 0},
                    ],
                    "logic": "AND",
                    "then": "sell",
                    "confidence": 0.7,
                },
            ],
        },
    },
    {
        "name": "bb_breakout",
        "description": "Bollinger Bands touch + RSI confirmation",
        "params": {
            "rules": [
                {
                    "name": "bb_oversold_bounce",
                    "conditions": [
                        {"indicator": "close", "op": "<=", "indicator_ref": "bb_lower"},
                        {"indicator": "rsi_14", "op": "<", "value": 35},
                    ],
                    "logic": "AND",
                    "then": "buy",
                    "confidence": 0.7,
                },
                {
                    "name": "bb_overbought_reject",
                    "conditions": [
                        {"indicator": "close", "op": ">=", "indicator_ref": "bb_upper"},
                        {"indicator": "rsi_14", "op": ">", "value": 65},
                    ],
                    "logic": "AND",
                    "then": "sell",
                    "confidence": 0.7,
                },
            ],
        },
    },
    {
        "name": "ma_cross",
        "description": "EMA(9) vs SMA(50) — golden cross / death cross",
        "params": {
            "rules": [
                {
                    "name": "golden_cross",
                    "conditions": [
                        {"indicator": "ema_9", "op": ">", "indicator_ref": "sma_50"},
                    ],
                    "logic": "AND",
                    "then": "buy",
                    "confidence": 0.65,
                },
                {
                    "name": "death_cross",
                    "conditions": [
                        {"indicator": "ema_9", "op": "<", "indicator_ref": "sma_50"},
                    ],
                    "logic": "AND",
                    "then": "sell",
                    "confidence": 0.65,
                },
            ],
        },
    },
]


def get_strategy(name: str) -> dict[str, Any]:
    """Retorna a estratégia built-in pelo nome. KeyError se não existir."""
    for strategy in BUILT_IN_STRATEGIES:
        if strategy["name"] == name:
            return strategy
    raise KeyError(f"Estratégia desconhecida: {name}")
