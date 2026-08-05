"""Testes do motor de estratégia (avaliação de regras)."""

import pandas as pd

from radar.strategy.defaults import BUILT_IN_STRATEGIES, get_strategy
from radar.strategy.engine import evaluate_rule, evaluate_rules


def _row(**values) -> pd.Series:
    return pd.Series(values)


def test_evaluate_rule_and():
    rule = {
        "conditions": [
            {"indicator": "rsi_14", "op": "<", "value": 30},
            {"indicator": "close", "op": ">", "indicator_ref": "ema_9"},
        ],
        "logic": "AND",
    }
    row = _row(rsi_14=25, close=100.0, ema_9=98.0)
    assert evaluate_rule(row, rule)

    row_bad_rsi = _row(rsi_14=50, close=100.0, ema_9=98.0)
    assert not evaluate_rule(row_bad_rsi, rule)


def test_evaluate_rule_or():
    rule = {
        "conditions": [
            {"indicator": "rsi_14", "op": "<", "value": 30},
            {"indicator": "rsi_14", "op": ">", "value": 70},
        ],
        "logic": "OR",
    }
    assert evaluate_rule(_row(rsi_14=25), rule)
    assert evaluate_rule(_row(rsi_14=80), rule)
    assert not evaluate_rule(_row(rsi_14=50), rule)


def test_evaluate_rule_missing_indicator_is_false():
    rule = {"conditions": [{"indicator": "atr_14", "op": "<", "value": 5}]}
    assert not evaluate_rule(_row(close=100.0), rule)  # atr_14 ausente


def test_evaluate_rule_empty_conditions_is_false():
    assert not evaluate_rule(_row(close=1.0), {"conditions": []})


def test_evaluate_rules_signals_and_hold_fallback():
    rules = [
        {"name": "buy_rule", "conditions": [{"indicator": "close", "op": ">", "value": 100}], "then": "buy", "confidence": 0.8},
        {"name": "sell_rule", "conditions": [{"indicator": "close", "op": "<", "value": 50}], "then": "sell", "confidence": 0.6},
    ]
    df = pd.DataFrame({
        "time": pd.to_datetime(["2025-01-01 09:00"]),
        "close": [120.0],
    })

    signals = evaluate_rules(df, rules)
    assert len(signals) == 1
    assert signals[0]["direction"] == "buy"
    assert signals[0]["confidence"] == 0.8
    assert "buy_rule" in signals[0]["reason"]

    df_hold = pd.DataFrame({
        "time": pd.to_datetime(["2025-01-01 09:00"]),
        "close": [75.0],
    })
    signals_hold = evaluate_rules(df_hold, rules)
    assert signals_hold[0]["direction"] == "hold"


def test_builtin_strategies_are_valid():
    for strategy in BUILT_IN_STRATEGIES:
        assert strategy["name"]
        assert strategy["description"]
        rules = strategy["params"]["rules"]
        assert rules
        for rule in rules:
            assert rule["then"] in ("buy", "sell", "hold")
            assert rule["logic"] in ("AND", "OR")
            assert 0.0 <= rule["confidence"] <= 1.0


def test_get_strategy_unknown_raises():
    try:
        get_strategy("nao_existe")
    except KeyError:
        return
    raise AssertionError("deveria levantar KeyError")
