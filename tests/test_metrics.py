"""Testes das métricas de desempenho."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from radar.benchmarks import benchmark_returns, load_benchmarks
from radar.metrics import (
    annualization_periods,
    calmar_ratio,
    expected_payoff,
    maximum_drawdown,
    profit_factor,
    recovery_factor,
    sharpe_ratio,
    sortino_ratio,
)


def test_annualization_periods_inferred_from_index():
    idx = pd.date_range("2025-01-02", periods=1440, freq="min")  # 1 dia de M1
    assert annualization_periods(idx) == 1440 * 252


def test_sharpe_ratio_constant_returns_is_zero():
    returns = pd.Series([0.001] * 50)
    assert sharpe_ratio(returns) == 0.0


def test_sharpe_ratio_positive_for_steady_gains():
    returns = pd.Series([0.01, 0.02, 0.01, 0.03, 0.02] * 20)
    assert sharpe_ratio(returns) > 0


def test_sharpe_ratio_anualization_scale():
    # Mesmos retornos, mais períodos/ano → Sharpe maior (anualizado)
    returns = pd.Series([0.001, -0.0005, 0.001] * 30)
    daily = sharpe_ratio(returns, periods_per_year=252)
    m1 = sharpe_ratio(returns, periods_per_year=252 * 1440)
    assert m1 > daily


def test_sortino_ratio_penalizes_downside():
    # Série com downside violento → Sortino menor que série com downside suave
    # (ambas com média positiva, para o sortino ser positivo)
    wild = pd.Series([0.06, -0.05, 0.07, -0.06, 0.06, -0.05] * 20)
    calm = pd.Series([0.02, -0.005, 0.02, -0.01, 0.02, -0.008] * 20)
    assert sortino_ratio(wild) > 0
    assert sortino_ratio(wild) < sortino_ratio(calm)


def test_maximum_drawdown():
    equity = pd.Series([100.0, 110.0, 105.0, 95.0, 115.0],
                       index=pd.date_range("2025-01-02", periods=5, freq="min"))
    mdd = maximum_drawdown(equity)
    assert mdd["max_drawdown_pct"] == pytest.approx(0.136364, rel=1e-3)  # 105 → 95
    assert mdd["peak_time"] == equity.index[1]
    assert mdd["valley_time"] == equity.index[3]
    assert mdd["recovery_time"] == equity.index[4]
    assert mdd["duration_bars"] == 3


def test_maximum_drawdown_flat_curve():
    equity = pd.Series([100.0] * 5)
    mdd = maximum_drawdown(equity)
    assert mdd["max_drawdown_pct"] == 0.0


def test_calmar_ratio():
    assert calmar_ratio(0.20, -0.10) == 2.0
    assert calmar_ratio(0.20, 0.0) == 0.0


def test_profit_factor():
    trades = [{"pnl": 100}, {"pnl": -40}, {"pnl": 60}, {"pnl": 0}]
    assert profit_factor(trades) == pytest.approx(160 / 40, rel=1e-3)
    assert profit_factor([]) == 0.0
    assert profit_factor([{"pnl": 10}, {"pnl": 5}]) == 99.999  # sem perdas


def test_recovery_factor():
    assert recovery_factor(500.0, 250.0) == 2.0
    assert recovery_factor(100.0, 0.0) == 0.0


def test_expected_payoff():
    trades = [{"pnl": 100}, {"pnl": -50}, {"pnl": 100}]
    result = expected_payoff(trades)
    assert result["win_rate"] == pytest.approx(2 / 3, rel=1e-2)  # arredondado p/ 4 casas
    assert result["avg_win"] == pytest.approx(100.0)
    assert result["avg_loss"] == pytest.approx(-50.0)
    assert result["expected_payoff"] == pytest.approx((2 / 3 * 100) - (1 / 3 * 50), rel=1e-2)
    assert expected_payoff([])["total_trades"] == 0


# ── Benchmarks ────────────────────────────────────────────


def test_benchmark_returns_available():
    # Período coberto pelos dados seed (2024-2025)
    result = benchmark_returns(date(2024, 1, 1), date(2024, 12, 31))
    assert result["selic"] is not None
    assert result["ipca"] is not None
    assert result["ibov"] is not None
    # SELIC 2024 = 12 × 1,05% composto
    assert result["selic"] == pytest.approx((1.0105 ** 12) - 1, rel=1e-3)


def test_benchmark_returns_unavailable():
    # Período sem dados (2020) → None sem erro
    result = benchmark_returns(date(2020, 1, 1), date(2020, 12, 31))
    assert result == {"selic": None, "ipca": None, "ibov": None}


def test_benchmark_returns_missing_seed_file():
    result = benchmark_returns(
        date(2024, 1, 1), date(2024, 12, 31),
        seed_path=Path("/tmp/nao-existe-benchmarks.json"),
    )
    assert result == {"selic": None, "ipca": None, "ibov": None}


def test_load_benchmarks_parses_series():
    series = load_benchmarks()
    assert set(series.keys()) == {"selic", "ipca", "ibov"}
    assert len(series["selic"]) == 24  # 2024 + 2025
    assert series["selic"][0]["date"] == date(2024, 1, 1)
