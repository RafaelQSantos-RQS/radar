"""Backtest — simulação de trades barra a barra com custos B3."""

from radar.backtest.account import AccountInfo
from radar.backtest.costs import (
    COST_MODELS,
    DEFAULT_COST_PARAMS,
    CostParams,
    compute_entry_costs,
    compute_exit_costs,
)
from radar.backtest.engine import Order, OrderStatus, OrderType, TradeSim

__all__ = [
    "COST_MODELS",
    "DEFAULT_COST_PARAMS",
    "AccountInfo",
    "CostParams",
    "Order",
    "OrderStatus",
    "OrderType",
    "TradeSim",
    "compute_entry_costs",
    "compute_exit_costs",
]