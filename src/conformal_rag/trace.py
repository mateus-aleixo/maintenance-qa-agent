"""JSONL tracing: one line per event, append-only, grep-able.

Observability without a service: latency, token counts, gate decisions and guard
flags land in traces/YYYY-MM-DD.jsonl. A dashboard can come later; the data
format will not have to change.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, trace_dir: Path | str):
        self.dir = Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self) -> Path:
        return self.dir / f"{date.today().isoformat()}.jsonl"

    def emit(self, kind: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "kind": kind,
            **fields,
        }
        with self._path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    @contextmanager
    def span(self, kind: str, **fields: Any):
        t0 = time.perf_counter()
        err: str | None = None
        try:
            yield
        except Exception as e:  # re-raised; the trace still records it
            err = repr(e)
            raise
        finally:
            self.emit(kind, latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                      error=err, **fields)


class NullTracer(Tracer):
    def __init__(self):  # no directory, no writes
        self.dir = Path(".")

    def emit(self, kind: str, **fields: Any) -> None:
        pass
