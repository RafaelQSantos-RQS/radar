"""Rotas da API v1."""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from radar.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    IngestResponse,
    JobStatus,
    OptimizeJob,
    OptimizeRequest,
    SignalRequest,
    SignalResponse,
    StrategyInfo,
)
from radar.api.service import evaluate_signal, list_strategies, run_backtest
from radar.config import DB_PATH
from radar.data.ingest import parse_csv, store_candles

router = APIRouter(prefix="/v1")


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    file: Annotated[UploadFile, File(description="CSV exportado do MetaTrader 5 (ou OHLCV genérico)")],
    symbol: str = ...,
    timeframe: str = ...,
) -> IngestResponse:
    """Sobe uma base de candles (CSV) e grava no DuckDB, com dedupe por timestamp."""
    try:
        text = file.file.read().decode("utf-8", errors="replace")
        candles = parse_csv(text)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"CSV inválido: {exc}") from exc

    inserted, skipped = store_candles(candles, symbol, timeframe, DB_PATH)
    return IngestResponse(
        symbol=symbol, timeframe=timeframe, rows_ingested=inserted, rows_skipped=skipped
    )


@router.get("/strategies", response_model=list[StrategyInfo])
def strategies() -> list[StrategyInfo]:
    """Lista as estratégias rule-based disponíveis."""
    return [StrategyInfo(**s) for s in list_strategies()]


@router.post("/signal", response_model=SignalResponse)
def signal(request: SignalRequest) -> SignalResponse:
    """Avalia as regras da estratégia na última barra."""
    try:
        result = evaluate_signal(
            symbol=request.symbol,
            timeframe=request.timeframe,
            strategy=request.strategy,
            candles=request.candles,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SignalResponse(
        symbol=request.symbol,
        timeframe=request.timeframe,
        strategy=request.strategy,
        signal=result["signal"],
        confidence=result["confidence"],
        justification=result["justification"],
        at=result["at"],
    )


@router.post("/backtest", response_model=BacktestResponse)
def backtest(request: BacktestRequest) -> BacktestResponse:
    """Roda o backtest da estratégia com custos B3 e devolve métricas."""
    try:
        result = run_backtest(
            strategy=request.strategy,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start=request.start,
            end=request.end,
            costs=request.costs,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BacktestResponse(**result)


@router.post("/optimize", response_model=OptimizeJob)
def optimize(_request: OptimizeRequest) -> OptimizeJob:
    raise HTTPException(status_code=501, detail="Validação ainda não implementada")


@router.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(_job_id: str) -> JobStatus:
    raise HTTPException(status_code=501, detail="Validação ainda não implementada")


__all__ = ["router"]
