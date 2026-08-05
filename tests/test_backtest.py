"""Testes do TradeSim e modelo de custos.

Inclui regressões dos bugs corrigidos na migração do TraderBot:
    - Bug 1: custos de saída (commission + b3_fee) não eram debitados
      na ordem, inflando win rate / profit factor.
    - Bug 2: sinal contrário nunca fechava posição (branch morto).
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from radar.backtest.costs import COST_MODELS, compute_costs
from radar.backtest.engine import OrderType, TradeSim


def _df(prices: list[float], start: datetime | None = None) -> pd.DataFrame:
    """DataFrame OHLCV sintético (high=low=close para barras sem pavio)."""
    start = start or datetime(2025, 1, 2, 9, 0)
    times = [start + timedelta(minutes=1) * i for i in range(len(prices))]
    return pd.DataFrame({
        "time": times,
        "open": prices,
        "high": prices,
        "low": prices,
        "close": prices,
        "volume": [1] * len(prices),
    })


# ── Costos ────────────────────────────────────────────────


def test_compute_costs_wdo():
    costs = compute_costs("WDO$", 5000.0, volume=1)
    assert costs == {
        "commission": 0.90,
        "b3_fee": 1.20,
        "slippage": 5.00,   # 1 tick × R$ 5,00
        "total": 7.10,
    }


def test_compute_costs_win():
    costs = compute_costs("WIN$", 120000.0, volume=2)
    assert costs["b3_fee"] == 0.60     # 0,30 × 2 contratos
    assert costs["slippage"] == 2.00   # 1 tick × R$ 1,00 × 2
    assert costs["total"] == pytest.approx(0.90 + 0.60 + 2.00)


def test_compute_costs_unknown_symbol_falls_back_to_wdo():
    costs = compute_costs("XXX$", 1.0)
    assert costs["total"] == pytest.approx(7.10)
    assert costs["commission"] == 0.90
    assert COST_MODELS["WDO$"].tick_value == 5.00


# ── Bug 1 (regressão): custos de saída na ordem ───────────


def test_bug1_exit_costs_charged_to_order():
    sim = TradeSim(initial_balance=10_000.0, symbol="WDO$", use_costs=True)
    ticket = sim.open_order(OrderType.BUY, price=5000.0, timestamp=datetime(2025, 1, 2, 9, 1))
    sim.close_order(ticket, price=5001.0, timestamp=datetime(2025, 1, 2, 9, 2), reason="tp")

    order = sim.closed_orders[0]
    per_side = compute_costs("WDO$", 5000.0, volume=1)  # entrada E saída

    # 1 tick de lucro = R$ 5,00 bruto; custos de ida E volta debitados
    assert order.gross_pnl == 5.00
    assert order.commission == pytest.approx(per_side["commission"] * 2)
    assert order.b3_fee == pytest.approx(per_side["b3_fee"] * 2)
    assert order.slippage == pytest.approx(per_side["slippage"] * 2)
    assert order.net_pnl == pytest.approx(5.00 - per_side["total"] * 2)


def test_bug1_sum_trades_net_pnl_equals_account_net_profit():
    sim = TradeSim(initial_balance=10_000.0, symbol="WDO$", use_costs=True)
    t = datetime(2025, 1, 2, 9, 1)
    for i, (open_px, close_px) in enumerate([(5000.0, 5001.0), (5002.0, 4999.0)]):
        ticket = sim.open_order(OrderType.BUY, price=open_px, timestamp=t + timedelta(minutes=i * 2))
        sim.close_order(ticket, price=close_px, timestamp=t + timedelta(minutes=i * 2 + 1))

    sum_net = sum(o.net_pnl for o in sim.closed_orders)
    assert sim.account.net_profit == pytest.approx(sum_net)
    assert sim.account.balance == pytest.approx(10_000.0 + sum_net)


def test_bug1_win_rate_not_inflated_by_costs():
    # Sem a correção, um trade de 1 tick parecia lucro (ignorava custos)
    sim = TradeSim(initial_balance=10_000.0, symbol="WDO$", use_costs=True)
    ticket = sim.open_order(OrderType.BUY, price=5000.0, timestamp=datetime(2025, 1, 2, 9, 1))
    sim.close_order(ticket, price=5001.0, timestamp=datetime(2025, 1, 2, 9, 2))

    # 1 tick de lucro < custos totais (R$ 14,20) → trade líquido é PREJUÍZO
    assert sim.closed_orders[0].net_pnl < 0
    assert sim.account.net_profit < 0


def test_costs_disabled_when_use_costs_false():
    sim = TradeSim(initial_balance=10_000.0, symbol="WDO$", use_costs=False)
    ticket = sim.open_order(OrderType.BUY, price=5000.0, timestamp=datetime(2025, 1, 2, 9, 1))
    sim.close_order(ticket, price=5001.0, timestamp=datetime(2025, 1, 2, 9, 2))
    assert sim.closed_orders[0].net_pnl == 5.00


# ── Bug 2 (regressão): sinal contrário fecha posição ──────


def test_bug2_opposite_signal_closes_position():
    df = _df([5000.0, 5001.0, 5002.0])
    signals = iter([
        {"direction": "buy"},    # idx 0: abre
        {"direction": "hold"},   # idx 1: mantém
        {"direction": "sell"},   # idx 2: fecha por sinal
    ])

    def strategy_fn(idx, _df):
        return next(signals)

    sim = TradeSim(initial_balance=10_000.0, symbol="WDO$", use_costs=False)
    result = sim.run(df, strategy_fn, warmup=0)

    assert len(result["trades"]) == 1
    trade = result["trades"][0]
    assert trade["close_reason"] == "signal"
    assert trade["direction"] == "buy"
    # Fechou no último close (5002), não no primeiro (5001)
    assert trade["close_price"] == 5002.0


def test_bug2_same_direction_signal_does_not_close():
    df = _df([5000.0, 5001.0, 5002.0])
    signals = iter([
        {"direction": "buy"},
        {"direction": "buy"},   # mesmo sinal: não fecha
        {"direction": "buy"},
    ])

    def strategy_fn(idx, _df):
        return next(signals)

    sim = TradeSim(initial_balance=10_000.0, symbol="WDO$", use_costs=False)
    result = sim.run(df, strategy_fn, warmup=0)

    # Nenhuma posição fechada por sinal; aberta no último candle fecha EOD
    assert len(result["trades"]) == 1
    assert result["trades"][0]["close_reason"] == "eod"


def test_bug2_hold_keeps_position_until_eod():
    df = _df([5000.0, 5001.0, 5002.0, 5003.0])
    signals = iter([
        {"direction": "buy"},
        {"direction": "hold"},
        {"direction": "hold"},
        {"direction": "hold"},
    ])

    def strategy_fn(idx, _df):
        return next(signals)

    sim = TradeSim(initial_balance=10_000.0, symbol="WDO$", use_costs=False)
    result = sim.run(df, strategy_fn, warmup=0)

    assert len(result["trades"]) == 1
    assert result["trades"][0]["close_reason"] == "eod"
    assert result["trades"][0]["close_price"] == 5003.0


# ── SL / TP ───────────────────────────────────────────────


def test_stop_loss_hit():
    df = _df([5000.0, 4995.0])
    df.loc[1, "low"] = 4994.0  # pavio toca o SL

    def strategy_fn(idx, _df):
        return {"direction": "buy", "sl": 4995.0} if idx == 0 else {"direction": "hold"}

    sim = TradeSim(initial_balance=10_000.0, symbol="WDO$", use_costs=False)
    result = sim.run(df, strategy_fn, warmup=0)

    assert result["trades"][0]["close_reason"] == "sl"
    assert result["trades"][0]["close_price"] == 4995.0
    assert result["trades"][0]["net_pnl"] < 0


def test_take_profit_hit():
    df = _df([5000.0, 5010.0])

    def strategy_fn(idx, _df):
        return {"direction": "buy", "tp": 5005.0} if idx == 0 else {"direction": "hold"}

    sim = TradeSim(initial_balance=10_000.0, symbol="WDO$", use_costs=False)
    result = sim.run(df, strategy_fn, warmup=0)

    assert result["trades"][0]["close_reason"] == "tp"
    assert result["trades"][0]["close_price"] == 5005.0
    assert result["trades"][0]["net_pnl"] == 25.0  # 5 pts × R$ 5,00


def test_no_costs_summary_matches_account():
    sim = TradeSim(initial_balance=10_000.0, symbol="WDO$", use_costs=True)
    df = _df([5000.0, 5001.0, 5002.0])
    signals = iter([{"direction": "buy"}, {"direction": "hold"}, {"direction": "sell"}])

    def strategy_fn(idx, _df):
        return next(signals)

    result = sim.run(df, strategy_fn, warmup=0)
    summary = result["summary"]
    assert summary["total_trades"] == 1
    assert summary["final_balance"] == pytest.approx(10_000.0 + result["trades"][0]["net_pnl"])
    assert summary["total_costs"] == pytest.approx(summary["total_commission"] + summary["total_b3_fee"] + summary["total_slippage"])


def test_warmup_skips_trading():
    df = _df([5000.0] * 5)
    called = []

    def strategy_fn(idx, _df):
        called.append(idx)
        return {"direction": "buy"}

    sim = TradeSim(initial_balance=10_000.0, symbol="WDO$", use_costs=False)
    sim.run(df, strategy_fn, warmup=3)
    assert called == [3, 4]  # estratégia não é chamada durante warmup


def test_empty_df_returns_error():
    sim = TradeSim()
    result = sim.run(pd.DataFrame(), lambda idx, df: {"direction": "hold"})
    assert result["error"] == "DataFrame vazio."
    assert result["summary"]["total_trades"] == 0
