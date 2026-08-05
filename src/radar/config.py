"""Configuração central do Radar."""

from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path("data/radar.duckdb")


@dataclass(frozen=True)
class Settings:
    """Configuração da aplicação. Valores mudam só por variável de ambiente no futuro."""

    app_name: str = "radar"
    version: str = "0.1.0"
    db_path: Path = DB_PATH


__all__ = ["DB_PATH", "Settings"]
