"""Contratos (schemas) da API v1."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

SIGNALS = ("COMPRA", "VENDA", "NEUTRO")


class SignalRequest(BaseModel):
    symbol: str = Field(examples=["WDO$"])
    timeframe: str = Field(examples=["M1"])
    params: dict[str, Any] | None = Field(
        default=None,
        description="Parâmetros do QuantScore (opcional — usa defaults).",
    )
    candles: list[dict[str, Any]] | None = Field(
        default=None,
        description="Candles inline (últimos N) — alternativa à leitura do banco.",
    )


class SignalResponse(BaseModel):
    symbol: str
    timeframe: str
    signal: str  # COMPRA | VENDA | NEUTRO
    score: float = 0.0
    components: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str = ""
    at: datetime


class BacktestRequest(BaseModel):
    symbol: str = Field(examples=["WDO$"])
    timeframe: str = Field(examples=["M1"])
    start: datetime | None = None
    end: datetime | None = None
    params: dict[str, Any] | None = Field(
        default=None,
        description="Parâmetros do QuantScore (opcional — usa defaults).",
    )
    costs_params: dict[str, Any] | None = Field(
        default=None,
        description="Parâmetros de custos B3 (opcional — usa defaults).",
    )


class BacktestResponse(BaseModel):
    symbol: str
    timeframe: str
    trades: list[dict[str, Any]] = Field(default_factory=list)
    net_profit: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_costs: float = 0.0
    equity_curve: list[float] = Field(default_factory=list)
    benchmarks: dict[str, float | None] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)


class OptimizeRequest(BaseModel):
    symbol: str = Field(examples=["WDO$"])
    timeframe: str = Field(examples=["M1"])
    param_grid: dict[str, list[Any]] = Field(
        examples=[{"stop_inicial": [1.0, 2.0], "min_score_compra": [60, 70]}]
    )
    start: datetime | None = None
    end: datetime | None = None
    costs_params: dict[str, Any] | None = None


class OptimizeJob(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending | running | done | error
    result: dict[str, Any] | None = None
    error: str | None = None


class IngestResponse(BaseModel):
    symbol: str
    timeframe: str
    rows_ingested: int
    rows_skipped: int


__all__ = [
    "SIGNALS",
    "BacktestRequest",
    "BacktestResponse",
    "IngestResponse",
    "JobStatus",
    "OptimizeJob",
    "OptimizeRequest",
    "SignalRequest",
    "SignalResponse",
]