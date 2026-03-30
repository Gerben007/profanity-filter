import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import aiosqlite


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@asynccontextmanager
async def _connect(db_path: str):
    """Open a connection with WAL mode and a generous busy timeout."""
    async with aiosqlite.connect(db_path, timeout=30) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=10000")
        yield conn


async def init_db(db_path: str) -> None:
    async with _connect(db_path) as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id     TEXT PRIMARY KEY,
                file_path  TEXT NOT NULL,
                status     TEXT NOT NULL DEFAULT 'pending',
                priority   INTEGER NOT NULL DEFAULT 1,
                error_msg  TEXT,
                matches    TEXT,
                segments   TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        # Migrate existing DB: add priority column if missing
        await conn.execute("""
            ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 1
        """).close() if False else None
        try:
            await conn.execute("ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 1")
        except Exception:
            pass  # column already exists
        try:
            await conn.execute("ALTER TABLE jobs ADD COLUMN progress INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass  # column already exists
        try:
            await conn.execute("ALTER TABLE jobs ADD COLUMN ignored_words TEXT")
        except Exception:
            pass  # column already exists
        await conn.commit()


async def create_job(db_path: str, file_path: str, priority: int = 1) -> str:
    job_id = str(uuid.uuid4())
    now = _now()
    async with _connect(db_path) as conn:
        await conn.execute(
            "INSERT INTO jobs (job_id, file_path, status, priority, progress, created_at, updated_at) VALUES (?, ?, 'pending', ?, 0, ?, ?)",
            (job_id, file_path, priority, now, now),
        )
        await conn.commit()
    return job_id


async def update_progress(db_path: str, job_id: str, progress: int) -> None:
    async with _connect(db_path) as conn:
        await conn.execute(
            "UPDATE jobs SET progress=?, updated_at=? WHERE job_id=?",
            (progress, _now(), job_id),
        )
        await conn.commit()


async def update_job(
    db_path: str,
    job_id: str,
    status: str,
    error_msg: str | None = None,
    matches: list[dict] | None = None,
    segments: list[tuple[float, float]] | None = None,
) -> None:
    now = _now()
    async with _connect(db_path) as conn:
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
    if row.get("ignored_words"):
        row["ignored_words"] = json.loads(row["ignored_words"])
    return row


async def get_job(db_path: str, job_id: str) -> dict | None:
    async with _connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)) as cur:
            row = await cur.fetchone()
    if row is None:
        return None
    return _deserialize(dict(row))


async def list_jobs(
    db_path: str,
    limit: int = 100,
    offset: int = 0,
    status: str | None = None,
) -> dict:
    """Returns {items, total, limit, offset}."""
    async with _connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row

        if status in ("pending", "processing"):
            where = "WHERE status=?"
            order = "ORDER BY COALESCE(priority,1) ASC, created_at ASC"
            count_params = (status,)
            data_params  = (status, limit, offset)
        elif status:
            where = "WHERE status=?"
            order = "ORDER BY created_at DESC"
            count_params = (status,)
            data_params  = (status, limit, offset)
        else:
            where = ""
            order = "ORDER BY created_at DESC"
            count_params = ()
            data_params  = (limit, offset)

        async with conn.execute(f"SELECT COUNT(*) FROM jobs {where}", count_params) as cur:
            total = (await cur.fetchone())[0]

        async with conn.execute(
            f"SELECT * FROM jobs {where} {order} LIMIT ? OFFSET ?", data_params
        ) as cur:
            rows = await cur.fetchall()

    return {
        "items":  [_deserialize(dict(r)) for r in rows],
        "total":  total,
        "limit":  limit,
        "offset": offset,
    }


async def get_queue_positions(db_path: str) -> dict[str, int]:
    """Return {job_id: position} for all pending jobs, ordered by priority then created_at."""
    async with _connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT job_id FROM jobs WHERE status='pending' ORDER BY COALESCE(priority,1) ASC, created_at ASC"
        ) as cur:
            rows = await cur.fetchall()
    return {dict(r)["job_id"]: i + 1 for i, r in enumerate(rows)}


async def set_ignored_words(db_path: str, job_id: str, words: list[str]) -> bool:
    """Persist the per-job ignored-words list. Returns False if job not found."""
    async with _connect(db_path) as conn:
        cur = await conn.execute(
            "UPDATE jobs SET ignored_words=?, updated_at=? WHERE job_id=?",
            (json.dumps(words) if words else None, _now(), job_id),
        )
        await conn.commit()
    return cur.rowcount > 0


async def reset_job_for_reprocess(db_path: str, job_id: str) -> bool:
    """Reset a job to pending so it gets re-transcribed. Keeps ignored_words intact."""
    async with _connect(db_path) as conn:
        cur = await conn.execute(
            """UPDATE jobs
               SET status='pending', progress=0, matches=NULL, segments=NULL,
                   error_msg=NULL, updated_at=?
               WHERE job_id=? AND status != 'processing'""",
            (_now(), job_id),
        )
        await conn.commit()
    return cur.rowcount > 0


async def delete_job(db_path: str, job_id: str) -> bool:
    async with _connect(db_path) as conn:
        cur = await conn.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
        await conn.commit()
    return cur.rowcount > 0


async def recover_stale_jobs(db_path: str) -> list[dict]:
    """Reset jobs stuck in 'processing' from a previous run.
    Returns list of {job_id, file_path, priority} so callers can re-enqueue correctly."""
    async with _connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT job_id, file_path, COALESCE(priority, 1) as priority FROM jobs WHERE status='processing'"
        ) as cur:
            stale = await cur.fetchall()
        if stale:
            await conn.execute(
                "UPDATE jobs SET status='pending', updated_at=? WHERE status='processing'",
                (_now(),),
            )
            await conn.commit()
    return [dict(r) for r in stale]
