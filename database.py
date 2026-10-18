import asyncio
import json
import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT NOT NULL,
    submission_text TEXT NOT NULL,
    overall_eqi REAL NOT NULL,
    tone_json TEXT NOT NULL,
    impact_json TEXT NOT NULL,
    risk_json TEXT NOT NULL,
    latency_seconds REAL NOT NULL,
    feishu_synced INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evaluations_student
    ON evaluations (student_id, id DESC);
"""


def _db_path() -> str:
    return os.getenv("DATABASE_PATH", "prismeval.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "student_id": row["student_id"],
        "submission_text": row["submission_text"],
        "overall_eqi": row["overall_eqi"],
        "tone_score": json.loads(row["tone_json"]),
        "impact_score": json.loads(row["impact_json"]),
        "risk_score": json.loads(row["risk_json"]),
        "latency_seconds": row["latency_seconds"],
        "feishu_synced": bool(row["feishu_synced"]),
        "created_at": row["created_at"],
    }


# ----------------------------------------------------------------------
# Synchronous implementations (run in a worker thread)
# ----------------------------------------------------------------------
def _init_db_sync() -> None:
    directory = os.path.dirname(os.path.abspath(_db_path()))
    os.makedirs(directory, exist_ok=True)
    with closing(_connect()) as conn:
        conn.executescript(_SCHEMA)


def _save_sync(
    student_id: str,
    submission_text: str,
    overall_eqi: float,
    tone: Dict[str, Any],
    impact: Dict[str, Any],
    risk: Dict[str, Any],
    latency_seconds: float,
    feishu_synced: bool,
) -> int:
    with closing(_connect()) as conn:
        with conn:  # commits on success, rolls back on error
            cursor = conn.execute(
                """
                INSERT INTO evaluations (
                    student_id, submission_text, overall_eqi,
                    tone_json, impact_json, risk_json,
                    latency_seconds, feishu_synced, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    submission_text,
                    overall_eqi,
                    json.dumps(tone),
                    json.dumps(impact),
                    json.dumps(risk),
                    latency_seconds,
                    1 if feishu_synced else 0,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            return int(cursor.lastrowid)


def _list_sync(student_id: str, limit: int, offset: int) -> List[Dict[str, Any]]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM evaluations WHERE student_id = ? "
            "ORDER BY id DESC LIMIT ? OFFSET ?",
            (student_id, limit, offset),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def _get_sync(evaluation_id: int) -> Optional[Dict[str, Any]]:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


# ----------------------------------------------------------------------
# Async API
# ----------------------------------------------------------------------
async def init_db() -> None:
    await asyncio.to_thread(_init_db_sync)
    logger.info("Database ready at %s", _db_path())


async def save_evaluation(
    student_id: str,
    submission_text: str,
    overall_eqi: float,
    tone: Dict[str, Any],
    impact: Dict[str, Any],
    risk: Dict[str, Any],
    latency_seconds: float,
    feishu_synced: bool,
) -> int:
    return await asyncio.to_thread(
        _save_sync,
        student_id, submission_text, overall_eqi,
        tone, impact, risk, latency_seconds, feishu_synced,
    )


async def list_evaluations(
    student_id: str, limit: int = 20, offset: int = 0
) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(_list_sync, student_id, limit, offset)


async def get_evaluation(evaluation_id: int) -> Optional[Dict[str, Any]]:
    return await asyncio.to_thread(_get_sync, evaluation_id)