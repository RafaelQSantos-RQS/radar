"""Fábrica da aplicação FastAPI."""

from fastapi import FastAPI

from radar import __version__
from radar.api.health import router as health_router
from radar.api.router import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Radar",
        description="Motor de apoio à decisão para day trading na B3.",
        version=__version__,
    )
    app.include_router(health_router)
    app.include_router(router)
    return app


def run() -> None:
    import uvicorn

    uvicorn.run("radar.api.app:create_app", factory=True, host="0.0.0.0", port=8000)


__all__ = ["create_app", "run"]
