"""Testes da camada de ingestão de dados."""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from radar.api.app import create_app

MT5_CSV = """2025.10.21 09:00,5213.0,5214.0,5212.0,5213.0,123
2025.10.21 09:01,5213.0,5215.0,5211.0,5214.0,98
2025.10.21 09:02,5214.0,5216.0,5213.0,5215.0,150
"""

HEADER_CSV = """date,time,open,high,low,close,volume
2025.10.21,09:00,5213.0,5214.0,5212.0,5213.0,123
2025.10.21,09:01,5213.0,5215.0,5211.0,5214.0,98
"""

DUPLICATE_CSV = MT5_CSV + "2025.10.21 09:00,5213.0,5214.0,5212.0,5213.0,123\n"

# Formato exportado pelo script scripts/ExportTicks1s.mq5 (grão 1s).
SCRIPT_CSV = """date;time;open;high;low;close;volume
2025.01.02;09:00:01;5234.5;5235.0;5234.0;5234.5;12
2025.01.02;09:00:02;5234.5;5236.0;5234.5;5236.0;8
"""


@pytest.fixture
def db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "test.duckdb"
    monkeypatch.setattr("radar.api.router.DB_PATH", path)
    return path


def test_parse_mt5_sem_cabecalho() -> None:
    from radar.data.ingest import parse_csv

    df = parse_csv(MT5_CSV)
    assert len(df) == 3
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert df["open"].iloc[0] == 5213.0
    assert df["volume"].iloc[0] == 123


def test_parse_com_cabecalho() -> None:
    from radar.data.ingest import parse_csv

    df = parse_csv(HEADER_CSV)
    assert len(df) == 2
    assert df["close"].iloc[-1] == 5214.0


def test_parse_formato_script_mql5_1s() -> None:
    """O CSV do ExportTicks1s.mq5 deve ser ingerido sem ajustes."""
    from radar.data.ingest import parse_csv

    df = parse_csv(SCRIPT_CSV)
    assert len(df) == 2
    assert df["ts"].iloc[0] == pd.Timestamp("2025-01-02 09:00:01")
    assert df["open"].iloc[0] == 5234.5
    assert df["volume"].iloc[1] == 8


def test_parse_rejeita_arquivo_vazio() -> None:
    from radar.data.ingest import parse_csv

    with pytest.raises(ValueError):
        parse_csv("")


def test_parse_rejeita_linhas_invalidas() -> None:
    from radar.data.ingest import parse_csv

    with pytest.raises(ValueError):
        parse_csv("abc,def,ghi\n1,2,3\n")


def test_ingest_endpoint_grava_e_deduplica(db_path: Path) -> None:
    client = TestClient(create_app())

    # 1ª subida: arquivo com duplicata interna → parse dedupe, 3 únicas
    response = client.post(
        "/v1/ingest?symbol=WDO%24&timeframe=M1",
        files={"file": ("candles.csv", DUPLICATE_CSV, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "WDO$"
    assert data["timeframe"] == "M1"
    assert data["rows_ingested"] == 3
    assert data["rows_skipped"] == 0  # duplicata interna removida no parse

    # 2ª subida: mesmo arquivo → dedupe contra o banco (PK)
    response = client.post(
        "/v1/ingest?symbol=WDO%24&timeframe=M1",
        files={"file": ("candles.csv", MT5_CSV, "text/csv")},
    )
    data = response.json()
    assert data["rows_ingested"] == 0
    assert data["rows_skipped"] == 3


def test_ingest_endpoint_rejeita_csv_invalido(db_path: Path) -> None:
    client = TestClient(create_app())
    response = client.post(
        "/v1/ingest?symbol=WDO%24&timeframe=M1",
        files={"file": ("bad.csv", "nada,de,valido\n", "text/csv")},
    )
    assert response.status_code == 400
