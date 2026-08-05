"""Backtest — simulação de trades barra a barra com custos B3."""

from radar.backtest.account import AccountInfo
from radar.backtest.costs import COST_MODELS, compute_costs
from radar.backtest.engine import Order, OrderStatus, OrderType, TradeSim
from radar.backtest.runner import build_strategy_fn, run_trade_backtest

__all__ = [
    "COST_MODELS",
    "AccountInfo",
    "Order",
    "OrderStatus",
    "OrderType",
    "TradeSim",
    "build_strategy_fn",
    "compute_costs",
    "run_trade_backtest",
]
