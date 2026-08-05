"""Testes do motor de indicadores."""

import numpy as np
import pandas as pd
import pytest

from radar.indicators import (
    compute_atr,
    compute_bollinger,
    compute_ema,
    compute_macd,
    compute_rsi,
    compute_sma,
)


def _series(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": values, "high": values, "low": values})


def test_sma():
    df = _series([1, 2, 3, 4, 5])
    result = compute_sma(df, period=3)
    assert np.isnan(result.iloc[0]) and np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_ema_span_weights_recent_values_more():
    df = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = compute_ema(df, period=3)
    assert result.iloc[-1] > compute_sma(df, period=3).iloc[-1]


def test_rsi_bounds():
    # Uptrend constante → RSI alto; downtrend → RSI baixo
    up = _series([float(i) for i in range(1, 40)])
    down = _series([float(i) for i in range(40, 1, -1)])
    rsi_up = compute_rsi(up).iloc[-1]
    rsi_down = compute_rsi(down).iloc[-1]
    assert rsi_up > 70
    assert rsi_down < 30


def test_rsi_range():
    rng = np.random.default_rng(42)
    df = _series(list(rng.normal(100, 1, 200)))
    rsi = compute_rsi(df).dropna()
    assert rsi.between(0, 100).all()


def test_macd_columns():
    df = _series([float(i) for i in range(1, 60)])
    result = compute_macd(df)
    assert list(result.columns) == ["macd", "signal", "histogram"]
    assert result["histogram"].iloc[-1] == pytest.approx(
        result["macd"].iloc[-1] - result["signal"].iloc[-1]
    )


def test_atr_constant_price_is_zero():
    df = _series([10.0] * 30)
    atr = compute_atr(df)
    assert atr.dropna().iloc[-1] == pytest.approx(0.0)  # preço constante → TR=0


def test_bollinger_ordering():
    df = _series([float(i) for i in range(1, 40)])
    bb = compute_bollinger(df)
    last = bb.iloc[-1]
    assert last["bb_upper"] >= last["bb_middle"] >= last["bb_lower"]
