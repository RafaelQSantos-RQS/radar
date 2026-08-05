"""Estratégias do Radar."""

from radar.strategy.defaults import BUILT_IN_STRATEGIES, get_strategy
from radar.strategy.engine import evaluate_rule, evaluate_rules

__all__ = [
    "BUILT_IN_STRATEGIES",
    "evaluate_rule",
    "evaluate_rules",
    "get_strategy",
]
