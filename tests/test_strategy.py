"""Testes da estratégia QuantScore (score, filtro corporativo, regras de entrada)."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from radar.strategy.quantscore import (
    DEFAULT_PARAMS,
    QuantScoreParams,
    compute_score,
    quantscore_strategy,
)


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
        "volume": [100] * len(prices),
    })


def _uptrend(n: int = 30, step: float = 1.0, base: float = 100.0) -> list[float]:
    return [base + step * i for i in range(n)]


def _downtrend(n: int = 30, step: float = 1.0, base: float = 130.0) -> list[float]:
    return [base - step * i for i in range(n)]


# ── Score ───────────────────────────────────────────────────


def test_score_uptrend_above_50():
    df = _df(_uptrend())
    comp = compute_score(df)
    assert comp["score_final"] > 50
    assert comp["momentum"] > 0
    assert comp["preco_vs_ema"] > 0


def test_score_downtrend_below_50():
    df = _df(_downtrend())
    comp = compute_score(df)
    assert comp["score_final"] < 50
    assert comp["momentum"] < 0


def test_score_clamped_to_100():
    # Uptrend forte + RSI neutro + preço acima da EMA → score alto, nunca > 100
    df = _df(_uptrend(n=60, step=5.0))
    comp = compute_score(df)
    assert 0.0 <= comp["score_final"] <= 100.0


def test_score_components_present():
    df = _df(_uptrend())
    comp = compute_score(df)
    for key in ("momentum", "fr_ibov", "rsi", "preco_vs_ema", "atr_pct", "angulo", "score_final"):
        assert key in comp


def test_score_with_ibov_relative_strength():
    df = _df(_uptrend(step=1.0))
    ibov_strong = _df(_uptrend(step=3.0))  # IBOV sobe mais → força relativa negativa
    comp = compute_score(df, ibov_df=ibov_strong)
    assert comp["fr_ibov"] < 0


def test_score_empty_df_raises():
    with pytest.raises(ValueError):
        compute_score(pd.DataFrame())


# ── Filtro anti-evento corporativo ──────────────────────────


def test_corporate_event_sets_score_minus_one():
    # Salto de 50% em um dia → evento corporativo → score -1.0
    prices = [100.0] * 20 + [150.0] + [150.0] * 10
    df = _df(prices)
    comp = compute_score(df)
    assert comp["score_final"] == -1.0
    assert comp["teve_evento_corporativo"] is True


def test_no_corporate_event_normal_score():
    df = _df(_uptrend())
    comp = compute_score(df)
    assert comp["teve_evento_corporativo"] is False
    assert comp["score_final"] >= 0.0


# ── Regras de entrada ───────────────────────────────────────


def test_strategy_buys_on_uptrend():
    # Tendência íngreme (step 2.0 → ângulo ~63°) para passar no filtro de ângulo
    df = _df(_uptrend(n=10, step=2.5))
    strategy_fn = quantscore_strategy(DEFAULT_PARAMS)
    signal = strategy_fn(len(df) - 1, df)
    assert signal["direction"] == "buy"
    assert signal["sl"] < signal["tp"]
    assert signal["volume"] >= 1


def test_strategy_sells_on_downtrend():
    df = _df(_downtrend(n=10, step=2.5))
    strategy_fn = quantscore_strategy(DEFAULT_PARAMS)
    signal = strategy_fn(len(df) - 1, df)
    assert signal["direction"] == "sell"
    assert signal["sl"] > signal["tp"]


def test_strategy_hold_on_flat():
    df = _df([100.0] * 40)
    strategy_fn = quantscore_strategy(DEFAULT_PARAMS)
    signal = strategy_fn(len(df) - 1, df)
    assert signal["direction"] == "hold"


def test_strategy_hold_on_corporate_event():
    prices = [100.0] * 20 + [150.0] + [150.0] * 10
    df = _df(prices)
    strategy_fn = quantscore_strategy(DEFAULT_PARAMS)
    signal = strategy_fn(len(df) - 1, df)
    assert signal["direction"] == "hold"


def test_strategy_early_idx_holds():
    df = _df(_uptrend(n=40))
    strategy_fn = quantscore_strategy(DEFAULT_PARAMS)
    assert strategy_fn(0, df)["direction"] == "hold"


def test_strategy_custom_params_change_behavior():
    # min_score_compra alto demais → nunca compra
    df = _df(_uptrend(n=10, step=2.5))
    strategy_fn = quantscore_strategy(QuantScoreParams(min_score_compra=99.0))
    assert strategy_fn(len(df) - 1, df)["direction"] == "hold"


def test_strategy_volume_from_capital():
    df = _df(_uptrend(n=10, step=2.5, base=100.0))
    strategy_fn = quantscore_strategy(QuantScoreParams(capital_por_ordem=1000.0))
    signal = strategy_fn(len(df) - 1, df)
    # capital 1000 / preço ~178 → volume ~5
    assert signal["volume"] == max(1, int(1000.0 / df.iloc[-1]["open"]))