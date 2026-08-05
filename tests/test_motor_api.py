"""Testes dos endpoints de motor (strategies, signal, backtest)."""

from datetime import datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from radar.api.app import create_app
from radar.api.service import evaluate_signal, list_strategies, run_backtest
from radar.data.ingest import store_candles

client = TestClient(create_app())


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Isola o banco em tmp_path para não vazar estado entre testes."""
    db = tmp_path / "radar.duckdb"
    monkeypatch.setattr("radar.api.service.DB_PATH", db)
    return db


def _trending_candles(n: int = 120) -> list[dict]:
    """Uptrend sintética: close cresce monotonicamente (RSI alto, golden cross)."""
    start = datetime(2025, 1, 2, 9, 0)
    candles = []
    price = 5000.0
    for i in range(n):
        ts = start + timedelta(minutes=i)
        candles.append({
            "ts": ts.isoformat(),
            "open": price,
            "high": price + 2.0,
            "low": price - 2.0,
            "close": price,
            "volume": 10,
        })
        price += 0.5
    return candles


def _seed_db(db, n: int = 200) -> None:
    """Grava candles sintéticos no banco isolado."""
    df = pd.DataFrame(_trending_candles(n))
    df.columns = ["ts", "open", "high", "low", "close", "volume"]
    store_candles(df, "WDO$", "M1", db)


# ── /strategies ───────────────────────────────────────────


def test_list_strategies():
    resp = client.get("/v1/strategies")
    assert resp.status_code == 200
    names = {s["id"] for s in resp.json()}
    assert names == {"rsi_ma", "macd", "bb_breakout", "ma_cross"}
    assert all(s["description"] for s in resp.json())
    assert all(s["rules"] for s in resp.json())


# ── /signal ───────────────────────────────────────────────


def test_signal_with_inline_candles():
    resp = client.post("/v1/signal", json={
        "symbol": "WDO$",
        "timeframe": "M1",
        "strategy": "ma_cross",
        "candles": _trending_candles(),
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["signal"] in ("COMPRA", "VENDA", "NEUTRO")
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["justification"]


def test_signal_uptrend_buys():
    # Uptrend → ema_9 > sma_50 → golden cross → COMPRA
    result = evaluate_signal("WDO$", "M1", "ma_cross", candles=_trending_candles())
    assert result["signal"] == "COMPRA"


def test_signal_unknown_strategy_404():
    resp = client.post("/v1/signal", json={
        "symbol": "WDO$", "timeframe": "M1", "strategy": "nao_existe", "candles": _trending_candles(),
    })
    assert resp.status_code == 404


def test_signal_without_candles_and_db_400():
    resp = client.post("/v1/signal", json={
        "symbol": "WDO$", "timeframe": "M1", "strategy": "ma_cross",
    })
    assert resp.status_code == 400


def test_signal_invalid_candles_400():
    resp = client.post("/v1/signal", json={
        "symbol": "WDO$", "timeframe": "M1", "strategy": "ma_cross",
        "candles": [{"ts": "x", "open": "y"}],
    })
    assert resp.status_code == 400


# ── /backtest ─────────────────────────────────────────────


def test_backtest_runs_without_db_400():
    resp = client.post("/v1/backtest", json={
        "strategy": "ma_cross", "symbol": "WDO$", "timeframe": "M1",
    })
    assert resp.status_code == 400


def test_run_backtest_with_candles(isolated_db):
    _seed_db(isolated_db)
    result = run_backtest("ma_cross", "WDO$", "M1", db_path=isolated_db)
    assert result["strategy"] == "ma_cross"
    assert result["trades"] >= 0
    assert result["net_profit"] is not None
    assert result["sharpe"] >= 0 or result["sharpe"] == 0.0
    assert result["max_drawdown"] >= 0
    assert 0.0 <= result["win_rate"] <= 1.0


def test_backtest_uptrend_is_profitable(isolated_db):
    _seed_db(isolated_db)
    result = run_backtest("ma_cross", "WDO$", "M1", db_path=isolated_db, costs=False)
    assert result["trades"] > 0
    assert result["net_profit"] > 0


def test_backtest_via_api(isolated_db):
    _seed_db(isolated_db)
    resp = client.post("/v1/backtest", json={
        "strategy": "ma_cross", "symbol": "WDO$", "timeframe": "M1",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "ma_cross"
    assert body["trades"] >= 0


def test_list_strategies_service_shape():
    strategies = list_strategies()
    assert all({"id", "name", "description", "rules"} <= set(s) for s in strategies)
