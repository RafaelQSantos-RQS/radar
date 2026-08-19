"""Grid de parâmetros — produto cartesiano com validação.

Gera todas as combinações do ``param_grid`` (dict parâmetro → lista de
valores) e valida cada combinação contra as faixas plausíveis do
QuantScore (spec api/optimize).
"""

from __future__ import annotations

import itertools
from typing import Any

from radar.strategy.quantscore import QuantScoreParams

__all__ = ["build_grid", "validate_params"]

# Faixas plausíveis por parâmetro (spec api/backtest: stop_inicial > 0,
# scores em [0, 100], ângulos em [0, 90]).
_RANGES: dict[str, tuple[float, float]] = {
    "stop_inicial": (0.0, 100.0),
    "take_inicial": (0.0, 100.0),
    "min_score_compra": (0.0, 100.0),
    "max_score_venda": (0.0, 100.0),
    "capital_por_ordem": (1.0, 1_000_000.0),
    "ang_min_compra": (0.0, 90.0),
    "ang_max_compra": (0.0, 90.0),
    "ang_min_venda": (0.0, 90.0),
    "ang_max_venda": (0.0, 90.0),
    "lookback": (2.0, 500.0),
}


def validate_params(params: dict[str, Any]) -> str | None:
    """Valida uma combinação de parâmetros.

    Returns:
        Mensagem de erro, ou None se válida.
    """
    for key, value in params.items():
        if key not in _RANGES:
            continue  # parâmetros desconhecidos são ignorados
        try:
            num = float(value)
        except (TypeError, ValueError):
            return f"parâmetro {key} não é numérico: {value!r}"
        lo, hi = _RANGES[key]
        if not (lo <= num <= hi):
            return f"parâmetro {key}={value} fora da faixa [{lo}, {hi}]"

    # Regras de consistência
    if params.get("ang_min_compra", 0) > params.get("ang_max_compra", 90):
        return "ang_min_compra > ang_max_compra"
    if params.get("ang_min_venda", 0) > params.get("ang_max_venda", 90):
        return "ang_min_venda > ang_max_venda"
    return None


def build_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Produto cartesiano do grid, descartando combinações inválidas.

    Returns:
        Lista de combinações válidas (dicts parâmetro → valor).
    """
    if not param_grid:
        return []

    keys = list(param_grid)
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*[param_grid[k] for k in keys]):
        combo = dict(zip(keys, values))
        if validate_params(combo) is None:
            combos.append(combo)
    return combos


def params_to_quantscore(combo: dict[str, Any]) -> QuantScoreParams:
    """Converte uma combinação do grid em QuantScoreParams (defaults p/ ausentes)."""
    return QuantScoreParams.from_dict(combo)