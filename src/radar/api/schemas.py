"""Contratos (schemas) da API v1."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

SIGNALS = ("COMPRA", "VENDA", "NEUTRO")


class SignalRequest(BaseModel):
    symbol: str = Field(examples=["WDO$"])
    timeframe: str = Field(examples=["M1"])
    strategy: str = Field(examples=["bb_breakout"])
    candles: list[dict[str, Any]] | None = Field(
        default=None,
        description="Candles inline (últimos N) — alternativa à leitura do banco.",
    )


class SignalResponse(BaseModel):
    symbol: str
    timeframe: str
    strategy: str
    signal: str  # COMPRA | VENDA | NEUTRO
    confidence: float = Field(ge=0.0, le=1.0)
    justification: str = ""
    at: datetime


class StrategyInfo(BaseModel):
    id: str
    name: str
    description: str = ""
    rules: list[dict] = Field(default_factory=list)


class BacktestRequest(BaseModel):
    strategy: str = Field(examples=["bb_breakout"])
    symbol: str = Field(examples=["WDO$"])
    timeframe: str = Field(examples=["M1"])
    start: datetime | None = None
    end: datetime | None = None
    costs: bool = True


class BacktestResponse(BaseModel):
    strategy: str
    symbol: str
    timeframe: str
    trades: int = 0
    net_profit: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    equity_curve: list[float] = Field(default_factory=list)


class OptimizeRequest(BaseModel):
    strategy: str = Field(examples=["bb_breakout"])
    symbol: str = Field(examples=["WDO$"])
    timeframe: str = Field(examples=["M1"])
    param_grid: dict[str, list[Any]] = Field(
        examples=[{"lookback": [20, 40], "entry_mult": [1.5, 2.0]}]
    )


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
    "StrategyInfo",
]
