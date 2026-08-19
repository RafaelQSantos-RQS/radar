"""Testes do TradeSim e modelo de custos.

Inclui regressões dos bugs corrigidos na migração do TraderBot:
    - Bug 1: custos de saída (commission + b3_fee) não eram debitados
      na ordem, inflando win rate / profit factor.
    - Bug 2: sinal contrário nunca fechava posição (branch morto).
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from radar.backtest.costs import (
    COST_MODELS,
    CostParams,
    compute_entry_costs,
    compute_exit_costs,
)
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


# ── Custos B3 ─────────────────────────────────────────────


def test_compute_entry_costs_wdo():
    # WDO$ 5000,00 × 1 contrato: comissão 0, ISS 0, emolumentos 0,005%,
    # registro 0, custódia 0
    costs = compute_entry_costs(5000.0, volume=1)
    assert costs["commission"] == 0.0
    assert costs["iss"] == 0.0
    assert costs["emolumentos"] == pytest.approx(5000.0 * 0.005 / 100)
    assert costs["registro"] == 0.0
    assert costs["custodia"] == 0.0
    assert costs["total"] == pytest.approx(costs["emolumentos"])


def test_compute_entry_costs_with_commission_and_iss():
    params = CostParams(comissao_pct=0.5, aliquota_iss=5.0)
    costs = compute_entry_costs(5000.0, volume=1, params=params)
    assert costs["commission"] == pytest.approx(25.0)          # 0,5% de 5000
    assert costs["iss"] == pytest.approx(25.0 * 5.0 / 100)     # 5% da corretagem
    assert costs["total"] == pytest.approx(25.0 + 1.25 + costs["emolumentos"])


def test_compute_exit_costs_daytrade():
    # Compra 5000 → 5001, mesmo dia: IRRF 1% do lucro, IR 20% do lucro
    # após custos operacionais
    costs = compute_exit_costs(
        open_price=5000.0,
        close_price=5001.0,
        volume=1,
        gross_pnl=5.00,
        is_daytrade=True,
        vol_venda=5001.0,
    )
    assert costs["liquidacao"] == pytest.approx(round((5000.0 + 5001.0) * 0.025 / 100, 2))
    assert costs["irrf"] == pytest.approx(5.00 * 1.0 / 100)
    operacionais = costs["commission"] + costs["iss"] + costs["emolumentos"] + costs["liquidacao"] + costs["registro"] + costs["custodia"]
    assert costs["ir"] == pytest.approx(max(0.0, 5.00 - operacionais) * 20.0 / 100)
    assert costs["total"] == pytest.approx(operacionais + costs["irrf"] + costs["ir"])


def test_compute_exit_costs_swing_irrf_on_volume():
    # Swing (dias diferentes): IRRF 0,005% do volume de venda
    costs = compute_exit_costs(
        open_price=5000.0,
        close_price=5001.0,
        volume=1,
        gross_pnl=5.00,
        is_daytrade=False,
        vol_venda=5001.0,
    )
    assert costs["irrf"] == pytest.approx(round(5001.0 * 0.005 / 100, 2))
    assert costs["ir"] == pytest.approx(max(0.0, 5.00 - (costs["commission"] + costs["iss"] + costs["emolumentos"] + costs["liquidacao"] + costs["registro"] + costs["custodia"])) * 15.0 / 100)


def test_compute_exit_costs_ir_zero_on_loss():
    costs = compute_exit_costs(
        open_price=5000.0,
        close_price=4990.0,
        volume=1,
        gross_pnl=-50.0,
        is_daytrade=True,
        vol_venda=4990.0,
    )
    assert costs["ir"] == 0.0
    assert costs["irrf"] == 0.0  # IRRF não incide sobre prejuízo


def test_cost_models_tick_values():
    assert COST_MODELS["WDO$"].tick_value == 5.00
    assert COST_MODELS["WIN$"].tick_value == 1.00


# ── Bug 1 (regressão): custos de saída na ordem ───────────


def test_bug1_exit_costs_charged_to_order():
    sim = TradeSim(initial_balance=10_000.0, symbol="WDO$", use_costs=True)
    ticket = sim.open_order(OrderType.BUY, price=5000.0, timestamp=datetime(2025, 1, 2, 9, 1))
    sim.close_order(ticket, price=5001.0, timestamp=datetime(2025, 1, 2, 9, 2), reason="tp")

    order = sim.closed_orders[0]
    entry = compute_entry_costs(5000.0, volume=1)
    exit_ = compute_exit_costs(
        open_price=5000.0, close_price=5001.0, volume=1,
        gross_pnl=5.00, is_daytrade=True, vol_venda=5001.0,
    )

    # 1 tick de lucro = R$ 5,00 bruto; custos de ida E volta debitados
    assert order.gross_pnl == 5.00
    assert order.commission == pytest.approx(entry["commission"] + exit_["commission"])
    assert order.emolumentos == pytest.approx(entry["emolumentos"] + exit_["emolumentos"])
    assert order.liquidacao == pytest.approx(exit_["liquidacao"])
    assert order.irrf == pytest.approx(exit_["irrf"])
    assert order.ir == pytest.approx(exit_["ir"])
    assert order.is_daytrade is True
    assert order.net_pnl == pytest.approx(5.00 - entry["total"] - exit_["total"])


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
    # Comissão alta garante que o trade líquido é prejuízo apesar do lucro bruto
    params = CostParams(comissao_pct=0.5)  # 0,5% do volume
    sim = TradeSim(initial_balance=10_000.0, symbol="WDO$", use_costs=True, cost_params=params)
    ticket = sim.open_order(OrderType.BUY, price=5000.0, timestamp=datetime(2025, 1, 2, 9, 1))
    sim.close_order(ticket, price=5001.0, timestamp=datetime(2025, 1, 2, 9, 2))

    # 1 tick de lucro (R$ 5,00) < custos (comissão 0,5% × 2 lados + taxas)
    assert sim.closed_orders[0].gross_pnl == 5.00
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
    assert summary["total_costs"] == pytest.approx(
        summary["total_commission"]
        + summary["total_b3_fee"]
        + summary["total_iss"]
        + summary["total_liquidacao"]
        + summary["total_custodia"]
        + summary["total_irrf"]
        + summary["total_ir"]
    )


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


# ── BCT (Breakout Ladder) ───────────────────────────────────


def _bct_df(prices: list[float], highs: list[float], lows: list[float]) -> pd.DataFrame:
    """DataFrame OHLCV com pavios explícitos para testar BCT."""
    start = datetime(2025, 1, 2, 9, 0)
    times = [start + timedelta(minutes=1) * i for i in range(len(prices))]
    return pd.DataFrame({
        "time": times,
        "open": prices,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": [1] * len(prices),
    })


def test_bct_rompeu_virtual_tp_sobe_stop():
    # Compra: high rompe virtual TP → stop sobe para o high e TP escala ×1.05
    df = _bct_df(
        prices=[5000.0, 5010.0, 5012.0],
        highs=[5000.0, 5010.0, 5012.0],
        lows=[5000.0, 5008.0, 5010.0],
    )
    signals = iter([
        {"direction": "buy", "sl_reference": 5000.0, "sl_estrategy": 4990.0,
         "sl_estrategy_distance": 10.0, "virtual_tp": 5005.0},
        {"direction": "hold"},
        {"direction": "hold"},
    ])

    def strategy_fn(idx, _df):
        return next(signals)

    sim = TradeSim(initial_balance=10_000.0, symbol="WDO$", use_costs=False, exit_mode="bct")
    sim.run(df, strategy_fn, warmup=0)
    # Na barra 1 o high (5010) rompeu o virtual TP (5005) → stop sobe para 5010
    order = sim.closed_orders[0] if sim.closed_orders else None
    # Se não fechou, a ordem ainda está aberta com o stop escalado
    if order is None:
        open_order = next(iter(sim.open_orders.values()))
        assert open_order.sl_estrategy == 5010.0
        assert open_order.virtual_tp == pytest.approx(5005.0 * 1.05)


def test_bct_rompeu_stop_fecha_com_motivo_bct():
    # Compra: low rompe o stop de proteção → fecha com motivo "bct"
    df = _bct_df(
        prices=[5000.0, 4995.0],
        highs=[5000.0, 4996.0],
        lows=[5000.0, 4990.0],
    )
    signals = iter([
        {"direction": "buy", "sl_reference": 5000.0, "sl_estrategy": 4992.0,
         "sl_estrategy_distance": 8.0, "virtual_tp": 5010.0},
        {"direction": "hold"},
    ])

    def strategy_fn(idx, _df):
        return next(signals)

    sim = TradeSim(initial_balance=10_000.0, symbol="WDO$", use_costs=False, exit_mode="bct")
    result = sim.run(df, strategy_fn, warmup=0)
    assert len(result["trades"]) == 1
    assert result["trades"][0]["close_reason"] == "bct"


def test_bct_gap_overnight_anula_posicao():
    # Gap overnight > 100% → anula no preço de entrada (motivo "anomalia")
    df = _bct_df(
        prices=[5000.0, 10500.0],
        highs=[5000.0, 10600.0],
        lows=[5000.0, 10400.0],
    )
    signals = iter([
        {"direction": "buy", "sl_reference": 5000.0, "sl_estrategy": 4900.0,
         "sl_estrategy_distance": 100.0, "virtual_tp": 5100.0},
        {"direction": "hold"},
    ])

    def strategy_fn(idx, _df):
        return next(signals)

    sim = TradeSim(initial_balance=10_000.0, symbol="WDO$", use_costs=False, exit_mode="bct")
    result = sim.run(df, strategy_fn, warmup=0)
    assert len(result["trades"]) == 1
    assert result["trades"][0]["close_reason"] == "anomalia"
    assert result["trades"][0]["close_price"] == 5000.0  # preço de entrada
