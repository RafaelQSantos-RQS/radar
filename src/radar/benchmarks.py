"""Benchmarks de mercado (SELIC, IPCA, Ibovespa) para comparação de backtests.

Carrega séries mensais de retorno de dados seed em ``data/benchmarks/``
e devolve o retorno acumulado para o período do backtest.

Se não houver dados para o período, devolve ``None`` (sem erro) — o
consumidor (API) trata como benchmark indisponível.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

__all__ = ["benchmark_returns", "load_benchmarks"]

# Caminho padrão dos dados seed (relativo à raiz do projeto).
DEFAULT_SEED_PATH = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "benchmarks.json"


def _parse_date(value: str) -> date:
    """Aceita 'YYYY-MM' ou 'YYYY-MM-DD'."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()  # noqa: DTZ007 — data pura, sem fuso
    except ValueError:
        return datetime.strptime(value, "%Y-%m").date()  # noqa: DTZ007 — data pura, sem fuso


def load_benchmarks(seed_path: Path | str | None = None) -> dict[str, list[dict[str, Any]]]:
    """Carrega as séries de benchmarks do arquivo seed.

    Returns:
        Dict ``{"selic": [{"date": ..., "return": ...}], "ipca": [...], "ibov": [...]}``.
        Vazio se o arquivo não existir ou estiver malformado.
    """
    path = Path(seed_path) if seed_path else DEFAULT_SEED_PATH
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    series: dict[str, list[dict[str, Any]]] = {}
    for key in ("selic", "ipca", "ibov"):
        items = raw.get(key, [])
        parsed = []
        for item in items:
            try:
                parsed.append({
                    "date": _parse_date(item["date"]),
                    "return": float(item["return"]),
                })
            except (KeyError, ValueError, TypeError):
                continue
        series[key] = sorted(parsed, key=lambda x: x["date"])
    return series


def _accumulated(series: list[dict[str, Any]], start: date, end: date) -> float | None:
    """Retorno acumulado (composto) dos retornos mensais no intervalo [start, end]."""
    relevant = [s["return"] for s in series if start <= s["date"] <= end]
    if not relevant:
        return None
    acc = 1.0
    for r in relevant:
        acc *= 1.0 + r
    return acc - 1.0


def benchmark_returns(
    start: date | datetime | None,
    end: date | datetime | None,
    seed_path: Path | str | None = None,
) -> dict[str, float | None]:
    """Retornos acumulados de SELIC, IPCA e Ibovespa para o período.

    Args:
        start: Início do período (inclusive). None = sem limite inferior.
        end: Fim do período (inclusive). None = sem limite superior.
        seed_path: Caminho alternativo do arquivo seed.

    Returns:
        Dict ``{"selic": float|None, "ipca": float|None, "ibov": float|None}``.
        ``None`` quando não há dados para o período.
    """
    series = load_benchmarks(seed_path)
    if not series:
        return {"selic": None, "ipca": None, "ibov": None}

    start_d = start.date() if isinstance(start, datetime) else start
    end_d = end.date() if isinstance(end, datetime) else end
    if start_d is None:
        start_d = date.min
    if end_d is None:
        end_d = date.max

    return {
        key: _accumulated(series[key], start_d, end_d)
        for key in ("selic", "ipca", "ibov")
    }