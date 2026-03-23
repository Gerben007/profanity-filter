import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import db
import transcriber
import watcher
import worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (all overridable via environment variables)
# ---------------------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "jobs.db")
BADWORDS_PATH = os.getenv("BADWORDS_PATH", str(Path(__file__).parent / "badwords.txt"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WATCH_FOLDER = os.getenv("WATCH_FOLDER", "/media")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
MUTE_PADDING = float(os.getenv("MUTE_PADDING", "0.1"))

# ---------------------------------------------------------------------------
# Word-list helpers (preserved from original main.py)
# ---------------------------------------------------------------------------

def load_words(path: str) -> list[str]:
    words = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        word = line.split("|")[0].strip()
        if word:
            words.append(word)
    return sorted(set(words), key=len, reverse=True)


def build_pattern(words: list[str]) -> re.Pattern:
    escaped = [re.escape(w) for w in words]
    return re.compile(r"(?<![a-zA-Z])(" + "|".join(escaped) + r")(?![a-zA-Z])", re.IGNORECASE)


MEDIA_EXTENSIONS = watcher.MEDIA_EXTENSIONS


async def _scan_existing(watch_folder: str, queue: asyncio.Queue) -> None:
    """Queue media files already in *watch_folder* that have no DB record yet."""
    existing = await db.list_jobs(DB_PATH)
    processed = {j["file_path"] for j in existing}

    found = 0
    for path in Path(watch_folder).rglob("*"):
        if path.suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        if path.name.startswith("tmp"):
            continue
        str_path = str(path)
        if str_path not in processed:
            queue.put_nowait(str_path)
            found += 1

    if found:
        logger.info("Startup scan: queued %d unprocessed file(s).", found)
    else:
        logger.info("Startup scan: all files already processed.")


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
_state: dict = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Init DB
    await db.init_db(DB_PATH)

    # 2. Load Whisper model (CPU-bound — run in thread)
    logger.info("Loading Whisper model '%s' on %s ...", WHISPER_MODEL, WHISPER_DEVICE)
    model = await asyncio.to_thread(
        transcriber.load_model, WHISPER_MODEL, WHISPER_DEVICE
    )
    logger.info("Whisper model ready.")

    # 3. Load word list
    words = load_words(BADWORDS_PATH)
    pattern = build_pattern(words)
    logger.info("Loaded %d bad words.", len(words))

    # 4. Job queue
    queue: asyncio.Queue = asyncio.Queue()

    # 5. Start watchdog first so watch_handler exists before the worker starts
    loop = asyncio.get_running_loop()
    observer, watch_handler = watcher.start_watcher(WATCH_FOLDER, loop, queue)

    # 6. Worker (passes mark_processed so it suppresses re-watch after muting)
    worker_task = asyncio.create_task(
        worker.run_worker(queue, DB_PATH, model, pattern, MUTE_PADDING, FFMPEG_BIN,
                          mark_processed=watch_handler.mark_processed)
    )

    # 7. Recover jobs that were processing when the service last crashed
    stale_paths = await db.recover_stale_jobs(DB_PATH)
    if stale_paths:
        logger.info("Re-enqueueing %d stale job(s).", len(stale_paths))
        for path in stale_paths:
            queue.put_nowait(path)

    # 8. Scan watch folder for existing files that have never been processed
    await _scan_existing(WATCH_FOLDER, queue)

    _state.update(
        queue=queue,
        worker_task=worker_task,
        observer=observer,
        pattern=pattern,
        words=words,
    )

    yield  # ← app serves requests here

    # Shutdown
    observer.stop()
    observer.join()
    worker_task.cancel()
    await asyncio.gather(worker_task, return_exceptions=True)
    logger.info("Shutdown complete.")


app = FastAPI(title="Profanity Audio Muter", version="2.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SubmitJobRequest(BaseModel):
    file_path: str


class SubmitJobResponse(BaseModel):
    job_id: str
    file_path: str


class ReloadResponse(BaseModel):
    word_count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    queue: asyncio.Queue = _state.get("queue")
    worker_task: asyncio.Task = _state.get("worker_task")
    return {
        "status": "ok",
        "queue_depth": queue.qsize() if queue else 0,
        "worker_alive": worker_task is not None and not worker_task.done(),
        "word_count": len(_state.get("words", [])),
    }


@app.get("/jobs")
async def list_jobs():
    return await db.list_jobs(DB_PATH)


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = await db.get_job(DB_PATH, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/jobs", status_code=202, response_model=SubmitJobResponse)
async def submit_job(body: SubmitJobRequest):
    path = body.file_path
    if not Path(path).is_file():
        raise HTTPException(status_code=400, detail=f"File not found: {path}")
    # Create the DB record first so the caller has an ID to poll immediately.
    # Pass the id along in the queue so the worker reuses it instead of
    # creating a duplicate.
    job_id = await db.create_job(DB_PATH, path)
    queue: asyncio.Queue = _state["queue"]
    queue.put_nowait((path, job_id))
    return SubmitJobResponse(job_id=job_id, file_path=path)


@app.post("/reload", response_model=ReloadResponse)
async def reload_badwords():
    words = load_words(BADWORDS_PATH)
    pattern = build_pattern(words)
    _state["words"] = words
    _state["pattern"] = pattern
    logger.info("Word list reloaded: %d words.", len(words))
    return ReloadResponse(word_count=len(words))


@app.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str):
    deleted = await db.delete_job(DB_PATH, job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")
