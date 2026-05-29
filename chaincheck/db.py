"""
Database abstraction layer — SQLite (dev) or PostgreSQL (prod).

Reads DATABASE_URL from the environment:
  sqlite:///./chaincheck_history.db   ← default (no external dep)
  postgresql://user:pass@host:5432/db ← production

All public functions are synchronous so they can be called from
FastAPI endpoints without an async driver; connection pooling is
handled by SQLAlchemy's QueuePool (Postgres) or StaticPool (SQLite).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import (
    Column,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine

_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./chaincheck_history.db",
)

# asyncpg uses postgresql+asyncpg:// but SQLAlchemy sync uses postgresql://
# normalise the URL so either prefix works
if _DATABASE_URL.startswith("postgres://"):
    _DATABASE_URL = _DATABASE_URL.replace("postgres://", "postgresql://", 1)

_IS_POSTGRES = _DATABASE_URL.startswith("postgresql")

_CONNECT_ARGS: dict = {}
if not _IS_POSTGRES:
    # SQLite: allow cross-thread use (FastAPI background tasks run in threads)
    _CONNECT_ARGS = {"check_same_thread": False}

_engine: Engine | None = None
_metadata = MetaData()

# ── Table definitions ──────────────────────────────────────────────────────────

results_table = Table(
    "results",
    _metadata,
    Column("request_id",      String,  primary_key=True),
    Column("created_at",      Float,   nullable=False),
    Column("response_preview", Text),
    Column("aggregate_score",  Float),
    Column("risk_level",      String),
    Column("total_latency_ms", Float),
    Column("methods",         Text),
    Column("payload",         Text),
)

Index("idx_results_ts", results_table.c.created_at)

feedback_table = Table(
    "feedback",
    _metadata,
    Column("id",         Integer, primary_key=True, autoincrement=True),
    Column("request_id", String,  nullable=False),
    Column("correct",    Integer, nullable=False),
    Column("note",       Text),
    Column("created_at", Float,   nullable=False),
)


# ── Engine bootstrap ───────────────────────────────────────────────────────────

def get_engine() -> Engine:
    """Return the shared SQLAlchemy engine, creating it on first call."""
    global _engine
    if _engine is None:
        kwargs: dict = {"connect_args": _CONNECT_ARGS}
        if _IS_POSTGRES:
            # Production pool: 5–10 connections, 30-second recycling
            kwargs.update({"pool_size": 5, "max_overflow": 10, "pool_recycle": 30})
        _engine = create_engine(_DATABASE_URL, **kwargs)
    return _engine


@contextmanager
def get_conn() -> Generator:
    """Yield a SQLAlchemy connection from the pool."""
    with get_engine().connect() as conn:
        yield conn


# ── Schema management ──────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables (and indexes) if they don't exist. Idempotent."""
    _metadata.create_all(get_engine())


# ── Write helpers ──────────────────────────────────────────────────────────────

def save_result(
    request_id: str,
    created_at: float,
    response_preview: str,
    aggregate_score: float,
    risk_level: str,
    total_latency_ms: float | None,
    methods: list[str],
    payload_json: str,
) -> None:
    """Upsert a detection result row. Best-effort — never raises."""
    try:
        upsert_sql: str
        if _IS_POSTGRES:
            upsert_sql = """
                INSERT INTO results
                    (request_id, created_at, response_preview, aggregate_score,
                     risk_level, total_latency_ms, methods, payload)
                VALUES
                    (:request_id, :created_at, :response_preview, :aggregate_score,
                     :risk_level, :total_latency_ms, :methods, :payload)
                ON CONFLICT (request_id) DO UPDATE SET
                    created_at       = EXCLUDED.created_at,
                    response_preview = EXCLUDED.response_preview,
                    aggregate_score  = EXCLUDED.aggregate_score,
                    risk_level       = EXCLUDED.risk_level,
                    total_latency_ms = EXCLUDED.total_latency_ms,
                    methods          = EXCLUDED.methods,
                    payload          = EXCLUDED.payload
            """
        else:
            upsert_sql = """
                INSERT OR REPLACE INTO results
                    (request_id, created_at, response_preview, aggregate_score,
                     risk_level, total_latency_ms, methods, payload)
                VALUES
                    (:request_id, :created_at, :response_preview, :aggregate_score,
                     :risk_level, :total_latency_ms, :methods, :payload)
            """
        params = {
            "request_id":      request_id,
            "created_at":      created_at,
            "response_preview": response_preview,
            "aggregate_score": aggregate_score,
            "risk_level":      risk_level,
            "total_latency_ms": total_latency_ms,
            "methods":         json.dumps(sorted(methods)),
            "payload":         payload_json,
        }
        with get_engine().begin() as conn:
            conn.execute(text(upsert_sql), params)
    except Exception:
        pass


def save_feedback(request_id: str, correct: bool, note: str) -> None:
    """Insert a feedback row. Best-effort — never raises."""
    try:
        with get_engine().begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO feedback (request_id, correct, note, created_at)"
                    " VALUES (:rid, :correct, :note, :ts)"
                ),
                {"rid": request_id, "correct": int(correct), "note": note, "ts": time.time()},
            )
    except Exception:
        pass


# ── Read helpers ───────────────────────────────────────────────────────────────

def fetch_history(limit: int = 20) -> list[dict]:
    """Return the most recent detection result summaries."""
    with get_conn() as conn:
        rows = conn.execute(
            text(
                "SELECT request_id, created_at, response_preview,"
                "       aggregate_score, risk_level, total_latency_ms, methods"
                " FROM results ORDER BY created_at DESC LIMIT :lim"
            ),
            {"lim": limit},
        ).fetchall()
    return [
        {
            "request_id":      r[0],
            "created_at":      r[1],
            "response_preview": r[2],
            "aggregate_score": r[3],
            "risk_level":      r[4],
            "total_latency_ms": r[5],
            "methods":         json.loads(r[6]) if r[6] else [],
        }
        for r in rows
    ]


def fetch_result_payload(request_id: str) -> str | None:
    """Return the raw JSON payload for a single result, or None if not found."""
    with get_conn() as conn:
        row = conn.execute(
            text("SELECT payload FROM results WHERE request_id = :rid"),
            {"rid": request_id},
        ).fetchone()
    return row[0] if row else None


def fetch_analytics() -> dict:
    """Aggregate detection history into charts-ready stats."""
    with get_conn() as conn:
        risk_rows = conn.execute(
            text("SELECT risk_level, COUNT(*) FROM results GROUP BY risk_level")
        ).fetchall()
        trend_rows = conn.execute(
            text(
                "SELECT created_at, aggregate_score, risk_level"
                " FROM results ORDER BY created_at DESC LIMIT 60"
            )
        ).fetchall()
        total = conn.execute(text("SELECT COUNT(*) FROM results")).fetchone()[0]
        avg_latency = conn.execute(
            text("SELECT AVG(total_latency_ms) FROM results")
        ).fetchone()[0]
        fb_rows = conn.execute(
            text(
                "SELECT r.risk_level, f.correct, COUNT(*)"
                " FROM feedback f"
                " JOIN results r ON f.request_id = r.request_id"
                " GROUP BY r.risk_level, f.correct"
            )
        ).fetchall()
        method_rows = conn.execute(
            text("SELECT methods FROM results WHERE methods IS NOT NULL")
        ).fetchall()

    risk_dist = {r[0]: r[1] for r in risk_rows}
    score_trend = [
        {"ts": r[0], "score": round(r[1], 4), "risk": r[2]}
        for r in reversed(trend_rows)
    ]
    method_counts: dict[str, int] = {}
    for (methods_json,) in method_rows:
        for m in json.loads(methods_json):
            method_counts[m] = method_counts.get(m, 0) + 1
    feedback_accuracy: dict[str, dict] = {}
    for risk_level, correct, count in fb_rows:
        if risk_level not in feedback_accuracy:
            feedback_accuracy[risk_level] = {"correct": 0, "incorrect": 0}
        key = "correct" if correct else "incorrect"
        feedback_accuracy[risk_level][key] += count

    return {
        "total_scans":       total,
        "avg_latency_ms":    round(avg_latency or 0, 1),
        "risk_distribution": risk_dist,
        "score_trend":       score_trend,
        "method_usage":      method_counts,
        "feedback_accuracy": feedback_accuracy,
    }
