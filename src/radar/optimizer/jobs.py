"""Registro de jobs de otimização em memória.

Mantém o estado dos jobs (pending/running/done/error) e executa o
trabalho em thread, preservando o resultado após a conclusão
(spec api/optimize — persistência em memória).
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from typing import Any

__all__ = ["JobStore"]


class JobStore:
    """Armazena jobs em memória com execução assíncrona em thread."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, spec: dict[str, Any]) -> str:
        """Cria um job pending e devolve o job_id."""
        job_id = uuid.uuid4().hex[:8]
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "result": None,
                "error": None,
                "spec": spec,
            }
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Devolve o job (sem o spec interno) ou None se não existir."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                "job_id": job["job_id"],
                "status": job["status"],
                "result": job["result"],
                "error": job["error"],
            }

    def start(self, job_id: str, fn: Callable[[], Any]) -> None:
        """Executa ``fn`` em thread e atualiza o estado do job."""
        def _run() -> None:
            with self._lock:
                self._jobs[job_id]["status"] = "running"
            try:
                result = fn()
                with self._lock:
                    self._jobs[job_id]["result"] = result
                    self._jobs[job_id]["status"] = "done"
            except Exception as exc:  # noqa: BLE001 — erro vai para o job
                with self._lock:
                    self._jobs[job_id]["error"] = str(exc)
                    self._jobs[job_id]["status"] = "error"

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()