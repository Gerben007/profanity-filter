import json
import uuid
from datetime import datetime, timezone

import aiosqlite


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def init_db(db_path: str) -> None:
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id     TEXT PRIMARY KEY,
                file_path  TEXT NOT NULL,
                status     TEXT NOT NULL DEFAULT 'pending',
                error_msg  TEXT,
                matches    TEXT,
                segments   TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await conn.commit()


async def create_job(db_path: str, file_path: str) -> str:
    job_id = str(uuid.uuid4())
    now = _now()
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "INSERT INTO jobs (job_id, file_path, status, created_at, updated_at) VALUES (?, ?, 'pending', ?, ?)",
            (job_id, file_path, now, now),
        )
        await conn.commit()
    return job_id


async def update_job(
    db_path: str,
    job_id: str,
    status: str,
    error_msg: str | None = None,
    matches: list[dict] | None = None,
    segments: list[tuple[float, float]] | None = None,
) -> None:
    now = _now()
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """UPDATE jobs SET status=?, error_msg=?, matches=?, segments=?, updated_at=?
               WHERE job_id=?""",
            (
                status,
                error_msg,
                json.dumps(matches) if matches is not None else None,
                json.dumps(segments) if segments is not None else None,
                now,
                job_id,
            ),
        )
        await conn.commit()


def _deserialize(row: dict) -> dict:
    for key in ("matches", "segments"):
        if row.get(key):
            row[key] = json.loads(row[key])
    return row


async def get_job(db_path: str, job_id: str) -> dict | None:
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)) as cur:
            row = await cur.fetchone()
    if row is None:
        return None
    return _deserialize(dict(row))


async def list_jobs(db_path: str, limit: int = 200) -> list[dict]:
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
    return [_deserialize(dict(r)) for r in rows]


async def delete_job(db_path: str, job_id: str) -> bool:
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
        await conn.commit()
    return cur.rowcount > 0


async def recover_stale_jobs(db_path: str) -> list[str]:
    """Reset jobs stuck in 'processing' from a previous run and return their file paths."""
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT job_id, file_path FROM jobs WHERE status='processing'"
        ) as cur:
            stale = await cur.fetchall()
        if stale:
            await conn.execute(
                "UPDATE jobs SET status='pending', updated_at=? WHERE status='processing'",
                (_now(),),
            )
            await conn.commit()
    return [dict(r)["file_path"] for r in stale]
