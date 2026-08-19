"""Serviço do optimizer — orquestra grid + backtest + jobs.

Cria um job assíncrono que roda o produto cartesiano do ``param_grid``,
executando um backtest QuantScore por combinação e ordenando os
resultados por ``net_profit`` (spec api/optimize).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from radar.backtest.costs import CostParams
from radar.backtest.engine import TradeSim
from radar.config import DB_PATH
from radar.data.ingest import get_candles
from radar.indicators import compute_all
from radar.metrics import maximum_drawdown, sharpe_ratio
from radar.optimizer.grid import build_grid, params_to_quantscore
from radar.optimizer.jobs import JobStore
from radar.strategy.quantscore import quantscore_strategy

__all__ = ["OptimizerService", "create_optimizer_service"]

# Instância global do JobStore (estado em memória da aplicação).
_job_store = JobStore()


def create_optimizer_service() -> OptimizerService:
    return OptimizerService(_job_store)


class OptimizerService:
    """Orquestra jobs de otimização por grid."""

    def __init__(self, store: JobStore) -> None:
        self._store = store

    def create_job(
        self,
        symbol: str,
        timeframe: str,
        param_grid: dict[str, list[Any]],
        start: datetime | None = None,
        end: datetime | None = None,
        costs_params: dict[str, Any] | None = None,
        db_path: Path | None = None,
    ) -> str:
        """Cria um job de otimização e devolve o job_id.

        Raises:
            ValueError: se o grid não tem combinações válidas.
        """
        combos = build_grid(param_grid)
        if not combos:
            raise ValueError("param_grid vazio ou sem combinações válidas")

        job_id = self._store.create({
            "symbol": symbol,
            "timeframe": timeframe,
            "combos": combos,
            "start": start,
            "end": end,
            "costs_params": costs_params,
            "db_path": db_path,
        })
        self._store.start(job_id, lambda: self._run_grid(job_id))
        return job_id

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Devolve o estado do job (None se não existir)."""
        return self._store.get(job_id)

    def _run_grid(self, job_id: str) -> dict[str, Any]:
        spec = self._store._jobs[job_id]["spec"]
        symbol = spec["symbol"]
        timeframe = spec["timeframe"]
        db_path = spec["db_path"] or DB_PATH

        df = get_candles(symbol, timeframe, db_path, start=spec["start"], end=spec["end"])
        if df.empty:
            raise ValueError(f"sem candles em {symbol} {timeframe} — ingira dados primeiro")
        df = df.rename(columns={"ts": "time"})
        df = compute_all(df)

        cost_params = CostParams.from_dict(spec["costs_params"])
        results: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for combo in spec["combos"]:
            try:
                result = self._run_one(df, symbol, combo, cost_params)
                results.append(result)
            except Exception as exc:  # noqa: BLE001 — combinação inválida não aborta o job
                errors.append({"params": combo, "error": str(exc)})

        results.sort(key=lambda r: r["net_profit"], reverse=True)
        return {"results": results, "errors": errors, "total": len(results)}

    def _run_one(
        self,
        df,
        symbol: str,
        combo: dict[str, Any],
        cost_params: CostParams,
    ) -> dict[str, Any]:
        params = params_to_quantscore(combo)
        strategy_fn = quantscore_strategy(params)
        sim = TradeSim(
            initial_balance=10_000.0,
            symbol=symbol,
            use_costs=True,
            cost_params=cost_params,
        )
        result = sim.run(df, strategy_fn, warmup=params.lookback)

        summary = result["summary"]
        eq = result["equity_curve"]
        sharpe = 0.0
        max_dd = 0.0
        if not eq.empty:
            equity = eq.set_index("time")["equity"]
            returns = equity.pct_change().dropna()
            if len(returns) > 1:
                sharpe = round(sharpe_ratio(returns), 4)
            max_dd = round(maximum_drawdown(equity)["max_drawdown_pct"], 4)

        return {
            "params": combo,
            "net_profit": summary["net_profit"],
            "win_rate": summary["win_rate"],
            "profit_factor": summary["profit_factor"],
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "total_costs": summary["total_costs"],
            "total_trades": summary["total_trades"],
        }