"""Endpoint de health check (fora do versionamento — usado por infra)."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


__all__ = ["router"]
