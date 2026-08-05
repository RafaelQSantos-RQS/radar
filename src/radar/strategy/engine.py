"""Motor de estratégias — avaliação pura de regras.

Lê definições de regras (dicts JSON-compatible) e dados de
indicadores, retorna sinais buy/sell/hold. Sem I/O e sem efeitos
colaterais, facilmente testável.

Formato de condição:
    {"indicator": "rsi_14", "op": "<", "value": 30}
    {"indicator": "close", "op": ">", "indicator_ref": "ema_9"}

Formato de regra:
    {
        "name": "rsi_oversold_bounce",
        "conditions": [<condition>, ...],
        "logic": "AND" | "OR",
        "then": "buy" | "sell" | "hold",
        "confidence": 0.0–1.0,
    }
"""

from typing import Any

import pandas as pd

__all__ = ["evaluate_rule", "evaluate_rules"]


# Operadores de comparação suportados.
_OPS: dict[str, Any] = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: abs(a - b) < 1e-8,
}


def _get_value(row: pd.Series, ref: str) -> float | None:
    """Resolve um valor da linha pelo nome da coluna. None se ausente/NaN."""
    val = row.get(ref)
    return float(val) if pd.notna(val) else None


def _evaluate_condition(row: pd.Series, cond: dict) -> bool:
    """Avalia uma condição contra a linha. Exige ``value`` ou ``indicator_ref``."""
    left = _get_value(row, cond["indicator"])
    if left is None:
        return False

    if "value" in cond:
        right = float(cond["value"])
    elif "indicator_ref" in cond:
        right = _get_value(row, cond["indicator_ref"])
    else:
        return False

    if right is None:
        return False

    op_fn = _OPS.get(cond["op"])
    if op_fn is None:
        return False
    return op_fn(left, right)


def evaluate_rule(row: pd.Series, rule: dict) -> bool:
    """Avalia uma regra contra a linha. True se as condições forem atendidas."""
    conditions = rule.get("conditions", [])
    results = [_evaluate_condition(row, c) for c in conditions]

    if not results:
        return False

    logic = rule.get("logic", "AND")
    return all(results) if logic == "AND" else any(results)


def _format_match(row: pd.Series, conditions: list[dict]) -> str:
    """Formata descrição legível das condições atendidas."""
    parts: list[str] = []
    for c in conditions:
        left = c["indicator"]
        op = c["op"]
        right = c.get("value", c.get("indicator_ref", "?"))
        val = _get_value(row, left)
        parts.append(f"{left}={val:.1f} {op} {right}")
    return " & ".join(parts)


def evaluate_rules(
    df: pd.DataFrame,
    rules: list[dict],
) -> list[dict]:
    """Avalia todas as regras contra a última barra + indicadores.

    Returns:
        Lista de sinais (um por regra disparada, mais um HOLD se nenhuma
        regra disparar). Cada sinal tem ``direction``, ``confidence``,
        ``price``, ``time`` e ``reason``.
    """
    if df.empty:
        return []

    last = df.iloc[-1]
    signals: list[dict] = []

    for rule in rules:
        if evaluate_rule(last, rule):
            signals.append({
                "direction": rule.get("then", "hold"),
                "confidence": float(rule.get("confidence", 0.5)),
                "price": float(last["close"]),
                "time": last["time"],
                "reason": f"{rule.get('name', 'rule')}: {_format_match(last, rule.get('conditions', []))}",
            })

    if not signals:
        signals.append({
            "direction": "hold",
            "confidence": 0.5,
            "price": float(last["close"]),
            "time": last["time"],
            "reason": "No conditions triggered",
        })

    return signals
