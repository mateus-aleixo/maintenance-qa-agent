"""SQLite persistence: chunks, FTS5 for BM25, embeddings as float32 blobs.

One file on disk. FTS5 ships inside CPython's bundled SQLite; vectors are searched
brute-force in NumPy (see retrieve.py) because the corpus is thousands of chunks and
exact search in milliseconds beats an approximate index with failure modes.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .ingest import Chunk

_TOKEN = re.compile(r"[A-Za-z0-9]+")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id      INTEGER PRIMARY KEY,
    doc     TEXT NOT NULL,
    page    INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    text    TEXT NOT NULL,
    UNIQUE (doc, ordinal)
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id),
    dim      INTEGER NOT NULL,
    vec      BLOB NOT NULL
);
"""


class Store:
    """SQLite-backed chunk store.

    The connection is **thread-local**. sqlite3 connections may not cross threads,
    and FastAPI runs sync endpoints in a worker threadpool, so a single shared
    connection raises "SQLite objects created in a thread can only be used in that
    same thread" on the second request that lands on a different worker. Each thread
    lazily opens its own; the schema is CREATE IF NOT EXISTS, so re-running it per
    connection is idempotent.

    `:memory:` is the exception: an in-memory database belongs to its connection, so
    a per-thread one would be empty. There the single connection is shared with
    check_same_thread disabled, which is what tests use and what they need.
    """

    def __init__(self, path: Path | str, read_only: bool = False):
        path = Path(path)
        self._path = str(path)
        self._memory = self._path == ":memory:"
        self._read_only = read_only and not self._memory
        if path.parent and not self._memory and not self._read_only:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._shared: sqlite3.Connection | None = None
        # Touch the property so the creating thread opens its connection and
        # applies the schema once, before any worker thread arrives.
        _ = self.conn

    def _connect(self) -> sqlite3.Connection:
        if self._read_only:
            # Serving an already-built index from a container: the filesystem is
            # read-only outside /tmp on Lambda, and BOTH of the writes below fail
            # there. WAL is the non-obvious one -- it creates `-wal` and `-shm`
            # files NEXT TO the database, so merely opening the connection raises
            # "unable to open database file" even though nothing has been written.
            # immutable=1 is the part that actually matters. journal_mode=WAL is
            # persisted in the database HEADER, so a WAL database opened read-only
            # still wants to create `-wal` and `-shm` beside itself, and fails on a
            # read-only filesystem with SQLITE_CANTOPEN. immutable promises the file
            # cannot change, which lets SQLite skip the WAL machinery entirely. That
            # promise is true here: the index is baked into the container image.
            conn = sqlite3.connect(f"file:{self._path}?mode=ro&immutable=1", uri=True)
            # Opening read-only is not enough. An FTS5 MATCH wants scratch space,
            # and SQLite tries to create it on disk, so the query fails with the
            # same SQLITE_CANTOPEN ("unable to open database file") even though the
            # database opened cleanly. That is why /gates worked and /retrieve did
            # not. temp_store=MEMORY keeps scratch off the filesystem entirely.
            conn.execute("PRAGMA temp_store=MEMORY")
            return conn
        conn = sqlite3.connect(self._path, check_same_thread=not self._memory)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        if self._memory:
            if self._shared is None:
                self._shared = self._connect()
            return self._shared
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close this thread's connection (and the shared one for :memory:).

        Needed because a live connection holds a lock: checkpointing or replacing
        the file underneath an open Store fails with "database is locked".
        """
        conn = self._shared if self._memory else getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            if self._memory:
                self._shared = None
            else:
                self._local.conn = None

    # -- write ---------------------------------------------------------------

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        cur = self.conn.executemany(
            "INSERT OR IGNORE INTO chunks (doc, page, ordinal, text) VALUES (?,?,?,?)",
            [(c.doc, c.page, c.ordinal, c.text) for c in chunks],
        )
        # Standalone FTS (not external-content: there, `SELECT rowid FROM fts`
        # reads through to the content table and the index silently stays empty).
        # chunk ids are monotone, so index everything past the FTS high-water mark.
        self.conn.execute(
            "INSERT INTO chunks_fts (rowid, text) SELECT id, text FROM chunks "
            "WHERE id > (SELECT COALESCE(MAX(rowid), 0) FROM chunks_fts)"
        )
        self.conn.commit()
        return cur.rowcount

    def add_embeddings(self, ids: Sequence[int], vecs: np.ndarray) -> None:
        vecs = np.asarray(vecs, dtype=np.float32)
        self.conn.executemany(
            "INSERT OR REPLACE INTO embeddings (chunk_id, dim, vec) VALUES (?,?,?)",
            [(int(i), vecs.shape[1], v.tobytes()) for i, v in zip(ids, vecs, strict=True)],
        )
        self.conn.commit()

    # -- read ----------------------------------------------------------------

    def bm25(self, query: str, k: int) -> list[tuple[int, float]]:
        """FTS5 BM25. Lower rank value = better; return as (id, score) with
        score negated so that, everywhere downstream, bigger is better.

        Every token is double-quoted (user text can contain FTS5 operators —
        a bare `AND` at end-of-query is a syntax error) and tokens are joined
        with OR: natural-language questions carry words no chunk contains, and
        FTS5's implicit AND would zero out recall. BM25 still ranks by term
        weight, so OR costs precision-at-1 nothing measurable at this scale."""
        tokens = _TOKEN.findall(query)
        if not tokens:
            return []
        sanitized = " OR ".join(f'"{t}"' for t in tokens)
        rows = self.conn.execute(
            "SELECT rowid, rank FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY rank LIMIT ?",
            (sanitized, k),
        ).fetchall()
        return [(int(r), -float(s)) for r, s in rows]

    def all_embeddings(self) -> tuple[np.ndarray, np.ndarray]:
        rows = self.conn.execute(
            "SELECT chunk_id, dim, vec FROM embeddings ORDER BY chunk_id"
        ).fetchall()
        if not rows:
            return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        dim = rows[0][1]
        mat = np.frombuffer(b"".join(r[2] for r in rows), dtype=np.float32)
        return ids, mat.reshape(len(rows), dim)

    def get_chunks(self, ids: Sequence[int]) -> list[dict]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, doc, page, ordinal, text FROM chunks WHERE id IN ({marks})",
            [int(i) for i in ids],
        ).fetchall()
        by_id = {r[0]: r for r in rows}
        return [
            {"id": r[0], "doc": r[1], "page": r[2], "ordinal": r[3], "text": r[4]}
            for i in ids
            if (r := by_id.get(int(i)))
        ]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

