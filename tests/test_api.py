"""Testes do esqueleto da API."""

import pytest
from fastapi.testclient import TestClient

from radar.api.app import create_app


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Isola o banco para o teste não depender do DB_PATH de produção."""
    monkeypatch.setattr("radar.api.service.DB_PATH", tmp_path / "radar.duckdb")
    monkeypatch.setattr("radar.api.router.DB_PATH", tmp_path / "radar.duckdb")


def test_health_ok() -> None:
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_signal_sem_candles_retorna_400() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/v1/signal",
        json={"symbol": "WDO$", "timeframe": "M1"},
    )
    assert response.status_code == 400


def test_strategies_removido() -> None:
    client = TestClient(create_app())
    response = client.get("/v1/strategies")
    assert response.status_code == 404