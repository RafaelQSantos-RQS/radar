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
)
from radar.api.service import evaluate_signal, run_backtest
from radar.config import DB_PATH
from radar.data.ingest import parse_csv, store_candles
from radar.optimizer.service import create_optimizer_service

router = APIRouter(prefix="/v1")

optimizer = create_optimizer_service()


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


@router.post("/signal", response_model=SignalResponse)
def signal(request: SignalRequest) -> SignalResponse:
    """Avalia o QuantScore na última barra."""
    try:
        result = evaluate_signal(
            symbol=request.symbol,
            timeframe=request.timeframe,
            params=request.params,
            candles=request.candles,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return SignalResponse(symbol=request.symbol, timeframe=request.timeframe, **result)


@router.post("/backtest", response_model=BacktestResponse)
def backtest(request: BacktestRequest) -> BacktestResponse:
    """Roda o backtest QuantScore com custos B3 e devolve métricas."""
    try:
        result = run_backtest(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start=request.start,
            end=request.end,
            params=request.params,
            costs_params=request.costs_params,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return BacktestResponse(**result)


@router.post("/optimize", response_model=OptimizeJob)
def optimize(request: OptimizeRequest) -> OptimizeJob:
    """Cria um job de otimização por grid paramétrico do QuantScore."""
    try:
        job_id = optimizer.create_job(
            symbol=request.symbol,
            timeframe=request.timeframe,
            param_grid=request.param_grid,
            start=request.start,
            end=request.end,
            costs_params=request.costs_params,
            db_path=DB_PATH,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return OptimizeJob(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    """Consulta o status de um job de otimização."""
    job = optimizer.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job {job_id} não encontrado")
    return JobStatus(**job)


__all__ = ["router"]