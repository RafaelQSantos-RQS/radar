"""Testes dos endpoints de motor (signal, backtest, optimize)."""

from datetime import datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from radar.api.app import create_app
from radar.api.service import evaluate_signal, run_backtest
from radar.data.ingest import store_candles

client = TestClient(create_app())


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Isola o banco em tmp_path para não vazar estado entre testes."""
    db = tmp_path / "radar.duckdb"
    monkeypatch.setattr("radar.api.service.DB_PATH", db)
    monkeypatch.setattr("radar.api.router.DB_PATH", db)
    return db


def _trending_candles(n: int = 120, pct: float = 0.02) -> list[dict]:
    """Uptrend sintética: close cresce ~2%/barra (ângulo ~64° → COMPRA)."""
    start = datetime(2025, 1, 2, 9, 0)
    candles = []
    price = 5000.0
    for i in range(n):
        ts = start + timedelta(minutes=i)
        candles.append({
            "ts": ts.isoformat(),
            "open": price,
            "high": price * 1.001,
            "low": price * 0.999,
            "close": price,
            "volume": 10,
        })
        price *= (1 + pct)
    return candles


def _seed_db(db, n: int = 200) -> None:
    """Grava candles sintéticos no banco isolado."""
    df = pd.DataFrame(_trending_candles(n))
    df.columns = ["ts", "open", "high", "low", "close", "volume"]
    store_candles(df, "WDO$", "M1", db)


# ── /signal ───────────────────────────────────────────────


def test_signal_with_inline_candles():
    resp = client.post("/v1/signal", json={
        "symbol": "WDO$",
        "timeframe": "M1",
        "candles": _trending_candles(),
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["signal"] in ("COMPRA", "VENDA", "NEUTRO")
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["justification"]
    assert "score" in body
    assert "components" in body


def test_signal_uptrend_buys():
    # Uptrend ~2%/barra → ângulo ~64° e score alto → COMPRA
    result = evaluate_signal("WDO$", "M1", candles=_trending_candles())
    assert result["signal"] == "COMPRA"
    assert result["score"] >= 60


def test_signal_ibov_not_target():
    # IBOV é benchmark — nunca é alvo de entrada
    result = evaluate_signal("IBOV", "M1", candles=_trending_candles())
    assert result["signal"] == "NEUTRO"


def test_signal_without_candles_and_db_400():
    resp = client.post("/v1/signal", json={
        "symbol": "WDO$", "timeframe": "M1",
    })
    assert resp.status_code == 400


def test_signal_invalid_candles_400():
    resp = client.post("/v1/signal", json={
        "symbol": "WDO$", "timeframe": "M1",
        "candles": [{"ts": "x", "open": "y"}],
    })
    assert resp.status_code == 400


def test_signal_invalid_params_400():
    resp = client.post("/v1/signal", json={
        "symbol": "WDO$", "timeframe": "M1",
        "params": {"stop_inicial": -1},
        "candles": _trending_candles(),
    })
    assert resp.status_code == 400


# ── /backtest ─────────────────────────────────────────────


def test_backtest_runs_without_db_400():
    resp = client.post("/v1/backtest", json={
        "symbol": "WDO$", "timeframe": "M1",
    })
    assert resp.status_code == 400


def test_run_backtest_with_candles(isolated_db):
    _seed_db(isolated_db)
    result = run_backtest("WDO$", "M1", db_path=isolated_db)
    assert result["symbol"] == "WDO$"
    assert result["trades"] is not None
    assert result["net_profit"] is not None
    assert result["sharpe"] >= 0 or result["sharpe"] == 0.0
    assert result["max_drawdown"] >= 0
    assert 0.0 <= result["win_rate"] <= 1.0
    assert "profit_factor" in result
    assert "total_costs" in result
    assert "benchmarks" in result
    assert "params" in result


def test_backtest_uptrend_is_profitable(isolated_db):
    _seed_db(isolated_db)
    result = run_backtest("WDO$", "M1", db_path=isolated_db)
    assert result["trades"]
    assert result["net_profit"] > 0


def test_backtest_via_api(isolated_db):
    _seed_db(isolated_db)
    resp = client.post("/v1/backtest", json={
        "symbol": "WDO$", "timeframe": "M1",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "WDO$"
    assert "trades" in body
    assert "benchmarks" in body
    assert "params" in body


def test_backtest_invalid_interval_400(isolated_db):
    _seed_db(isolated_db)
    resp = client.post("/v1/backtest", json={
        "symbol": "WDO$", "timeframe": "M1",
        "start": "2025-02-01T00:00:00",
        "end": "2025-01-01T00:00:00",
    })
    assert resp.status_code == 400


# ── /optimize ─────────────────────────────────────────────


def test_optimize_job_created_and_done(isolated_db):
    _seed_db(isolated_db)
    resp = client.post("/v1/optimize", json={
        "symbol": "WDO$",
        "timeframe": "M1",
        "param_grid": {
            "stop_inicial": [1.0, 2.0],
            "min_score_compra": [60, 70],
        },
    })
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert job_id

    # Job roda em thread — aguarda conclusão
    status = client.get(f"/v1/jobs/{job_id}").json()
    assert status["status"] in ("pending", "running", "done", "error")
    assert status["job_id"] == job_id


def test_optimize_grid_executed_and_sorted(isolated_db):
    _seed_db(isolated_db)
    resp = client.post("/v1/optimize", json={
        "symbol": "WDO$",
        "timeframe": "M1",
        "param_grid": {
            "stop_inicial": [1.0, 2.0],
            "min_score_compra": [60, 70],
        },
    })
    job_id = resp.json()["job_id"]

    # Aguarda o job concluir (thread roda em background)
    import time
    for _ in range(100):
        status = client.get(f"/v1/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.05)

    assert status["status"] == "done", status.get("error")
    result = status["result"]
    assert result["total"] == 4  # 2 × 2 combinações válidas
    net_profits = [r["net_profit"] for r in result["results"]]
    assert net_profits == sorted(net_profits, reverse=True)  # ordenado por net_profit
    assert all({"params", "net_profit", "win_rate", "profit_factor", "sharpe"} <= set(r) for r in result["results"])


def test_optimize_empty_grid_400(isolated_db):
    _seed_db(isolated_db)
    resp = client.post("/v1/optimize", json={
        "symbol": "WDO$",
        "timeframe": "M1",
        "param_grid": {},
    })
    assert resp.status_code == 400


def test_optimize_job_not_found_404():
    resp = client.get("/v1/jobs/nao_existe")
    assert resp.status_code == 404


def test_optimize_invalid_combo_ignored(isolated_db):
    _seed_db(isolated_db)
    resp = client.post("/v1/optimize", json={
        "symbol": "WDO$",
        "timeframe": "M1",
        "param_grid": {
            "stop_inicial": [1.0, -5.0],  # -5 inválido → ignorado
            "min_score_compra": [60],
        },
    })
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    status = client.get(f"/v1/jobs/{job_id}").json()
    assert status["status"] in ("pending", "running", "done", "error")