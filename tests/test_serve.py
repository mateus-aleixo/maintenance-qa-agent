"""The serving layer, on a synthetic registry.

No corpus download, no encoder download, no network: a tiny SQLite index is built
with the deterministic HashEmbedder and paired with a hand-written gate manifest.
The ONNX encoder is exercised separately by scripts/export_embedder.py's parity
check, which needs torch and therefore does not belong in CI.
"""

import json
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conformal_rag.embed import HashEmbedder
from conformal_rag.store import Chunk, Store

GATE_MANIFEST = {
    "source_artifact": "gate_v3_14b.json",
    "generator": "qwen2.5:14b-instruct",
    "score": "support_v1",
    "gate": {
        "alpha": 0.2,
        "min_group": 30,
        "global_threshold": 0.17,
        "group_thresholds": {},
    },
    "held_out": {"risk": 0.05, "answer_rate": 0.74, "n_answered": 20, "n_total": 27},
    "ungated_risk": 0.296,
    "n_calibration": 27,
}

CHUNKS = [
    ("manual.pdf", 12, 0, "Low oil pressure at idle indicates worn bearings or a failing pump."),
    ("manual.pdf", 13, 1, "Torque the cylinder head bolts in sequence to 90 newton metres."),
    ("manual.pdf", 14, 2, "The cooling system is pressurised; do not open it hot."),
]


@pytest.fixture(scope="module")
def monkeymodule():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeymodule):
    root = tmp_path_factory.mktemp("rag_serving")

    store = Store(root / "index.db")
    store.add_chunks([Chunk(doc=d, page=p, ordinal=o, text=t) for d, p, o, t in CHUNKS])
    embedder = HashEmbedder()
    ids = [c["id"] for c in store.get_chunks([1, 2, 3])]
    store.add_embeddings(ids, embedder.encode([t for _, _, _, t in CHUNKS]))

    (root / "gate.json").write_text(json.dumps(GATE_MANIFEST))

    from conformal_rag.serve import app as app_module

    # The real bundle loads an ONNX encoder; substitute the deterministic one so
    # CI needs no 135 MB download. Everything else is the production path.
    class TestBundle:
        def __init__(self):
            self.manifest = GATE_MANIFEST
            from conformal_rag.conformal import ConformalGate

            self.gate = ConformalGate.from_dict(GATE_MANIFEST["gate"])
            self.store = Store(root / "index.db")
            self.embedder = embedder

    monkeymodule.setattr(app_module, "bundle", lambda: TestBundle())
    yield TestClient(app_module.app)


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_gates_reports_the_trade_not_just_the_bound(client):
    """A guarantee quoted without its coverage is half a result."""
    g = client.get("/gates").json()
    assert g["alpha"] == 0.2 and g["threshold"] == 0.17
    assert g["ungated_risk"] > g["held_out_risk"]      # the gate bought something
    assert g["held_out_risk"] <= g["alpha"]            # and honoured its bound
    assert 0.0 < g["coverage"] <= 1.0                  # at a stated price
    # The gate is only valid for the generator and score it was fitted against,
    # so both have to be on the wire.
    assert g["generator"] and g["score"]


def test_retrieve_returns_ranked_hits(client):
    r = client.get("/retrieve", params={"q": "low oil pressure at idle", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "low oil pressure at idle"
    assert 1 <= len(body["hits"]) <= 2
    top = body["hits"][0]
    assert "oil pressure" in top["text"].lower()       # BM25 finds the right chunk
    assert set(top["sources"]) <= {"bm25", "vec"}
    scores = [h["score"] for h in body["hits"]]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_validates_k_and_query(client):
    assert client.get("/retrieve", params={"q": "x", "k": 5}).status_code == 422
    assert client.get("/retrieve", params={"q": "valid query", "k": 0}).status_code == 422
    assert client.get("/retrieve", params={"q": "valid query", "k": 99}).status_code == 422


def test_gate_decision_is_the_threshold_comparison(client):
    below = client.get("/gate", params={"score": 0.05}).json()
    at = client.get("/gate", params={"score": 0.17}).json()
    above = client.get("/gate", params={"score": 0.9}).json()

    assert below["decision"] == "abstain" and below["answer"] is False
    assert at["decision"] == "answer"      # inclusive: score >= threshold
    assert above["decision"] == "answer"
    for d in (below, at, above):
        assert d["threshold"] == 0.17 and d["alpha"] == 0.2
        assert d["held_out_risk"] == 0.05  # the measured risk rides along


def test_gate_rejects_scores_outside_the_unit_interval(client):
    assert client.get("/gate", params={"score": -0.1}).status_code == 422
    assert client.get("/gate", params={"score": 1.5}).status_code == 422


def test_ask_is_503_and_explains_why(client):
    """Generation is not hosted. That must be a clear answer, not a crash."""
    r = client.post("/ask", params={"q": "what indicates low oil pressure?"})
    assert r.status_code == 503
    detail = r.json()["detail"].lower()
    assert "llm_base_url" in detail          # says how to enable it
    assert "retrieve" in detail              # says what IS served


def test_store_survives_use_from_another_thread(tmp_path):
    """sqlite3 connections are thread-bound and FastAPI runs sync endpoints in a
    worker threadpool, so a shared connection raises on the second request that
    lands elsewhere. Store keeps one connection per thread."""
    from concurrent.futures import ThreadPoolExecutor

    store = Store(tmp_path / "t.db")
    store.add_chunks([Chunk(doc="d", page=1, ordinal=0, text="pressure relief valve")])

    def query():
        return len(store.bm25("pressure", 5))

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: query(), range(8)))
    assert all(r >= 1 for r in results)


def test_in_memory_store_is_shared_across_threads(tmp_path):
    """A per-thread connection to :memory: would see an empty database."""
    from concurrent.futures import ThreadPoolExecutor

    store = Store(":memory:")
    store.add_chunks([Chunk(doc="d", page=1, ordinal=0, text="gasket sealant")])
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: len(store.bm25("gasket", 5)), range(4)))
    assert all(r == 1 for r in results)


def test_read_only_store_opens_without_writing(tmp_path):
    """Lambda mounts the image read-only outside /tmp.

    The trap is WAL: `PRAGMA journal_mode=WAL` creates `-wal` and `-shm` files
    NEXT TO the database, so a read-only directory makes merely *opening* the
    connection raise "unable to open database file", long before any query. This
    passed locally and 500'd on Lambda until Store grew a read-only mode.
    """
    from conformal_rag.registry import _checkpoint_wal

    db = tmp_path / "index.db"
    writable = Store(db)
    writable.add_chunks([Chunk(doc="d", page=1, ordinal=0, text="bearing clearance")])

    # The served copy must not be left in WAL mode: whether an immutable open of a
    # WAL database succeeds depends on whether the filesystem still allows the
    # sidecar files, which is exactly the difference between a laptop and Lambda.
    # The registry removes the question at build time.
    assert writable.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    writable.close()  # a live connection holds a lock; checkpointing would fail
    _checkpoint_wal(db)
    assert sqlite3.connect(str(db)).execute(
        "PRAGMA journal_mode").fetchone()[0] == "delete"
    assert not db.with_name(db.name + "-wal").exists()

    ro = Store(db, read_only=True)
    assert len(ro.bm25("bearing", 5)) == 1
    with pytest.raises(Exception):
        ro.add_chunks([Chunk(doc="d", page=2, ordinal=1, text="nope")])


def test_read_only_is_ignored_for_memory_databases(tmp_path):
    """:memory: has nothing to open read-only; the flag must not break it."""
    s = Store(":memory:", read_only=True)
    s.add_chunks([Chunk(doc="d", page=1, ordinal=0, text="thermostat housing")])
    assert len(s.bm25("thermostat", 5)) == 1


def test_read_only_store_can_run_fts_queries(tmp_path):
    """Opening read-only is not the whole problem.

    FTS5 MATCH needs scratch space and SQLite reaches for the filesystem to get
    it, so on a read-only container the *query* raises SQLITE_CANTOPEN even
    though the connection opened fine. /gates worked and /retrieve 500'd on
    Lambda for exactly this reason.
    """
    db = tmp_path / "index.db"
    w = Store(db)
    w.add_chunks([
        Chunk(doc="d", page=1, ordinal=0, text="crankshaft main bearing journal"),
        Chunk(doc="d", page=2, ordinal=1, text="coolant thermostat opens at 82 degrees"),
    ])
    from conformal_rag.registry import _checkpoint_wal

    w.close()
    _checkpoint_wal(db)
    ro = Store(db, read_only=True)
    assert len(ro.bm25("crankshaft", 5)) == 1
    assert len(ro.bm25("thermostat", 5)) == 1
    assert ro.conn.execute("PRAGMA temp_store").fetchone()[0] == 2  # 2 == MEMORY
