"""Estratégias do Radar — estratégia única QuantScore."""

from radar.strategy.quantscore import (
    DEFAULT_PARAMS,
    QuantScoreParams,
    compute_score,
    quantscore_strategy,
)

__all__ = [
    "DEFAULT_PARAMS",
    "QuantScoreParams",
    "compute_score",
    "quantscore_strategy",
]