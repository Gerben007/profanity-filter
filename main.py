import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db
import transcriber
from transcriber import build_pattern
import watcher
import worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy access log entries for high-frequency polling endpoints
class _SuppressPolling(logging.Filter):
    _SKIP = ("/health", "/jobs?status=", "/jobs/queue-positions")
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(s in msg for s in self._SKIP)

logging.getLogger("uvicorn.access").addFilter(_SuppressPolling())

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


MEDIA_EXTENSIONS = watcher.MEDIA_EXTENSIONS


async def _scan_existing(watch_folder: str, queue: asyncio.Queue) -> None:
    """Queue media files already in *watch_folder* that have no DB record yet."""
    existing = await db.list_jobs(DB_PATH, limit=100_000)
    processed = {j["file_path"] for j in existing["items"]}

    found = 0
    for path in Path(watch_folder).rglob("*"):
        if path.suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        if path.name.startswith("tmp"):
            continue
        str_path = str(path)
        if str_path not in processed:
            _enqueue(queue, 1, str_path, None)
            found += 1

    if found:
        logger.info("Startup scan: queued %d unprocessed file(s).", found)
    else:
        logger.info("Startup scan: all files already processed.")


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------
_state: dict = {}
_enqueue_counter = 0  # monotonic tiebreaker so PriorityQueue never compares job_id/None


def _enqueue(queue: asyncio.PriorityQueue, priority: int, path: str, job_id) -> None:
    global _enqueue_counter
    _enqueue_counter += 1
    queue.put_nowait((priority, _enqueue_counter, path, job_id))


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

    # 4. Job queue (priority queue: lower number = processed first)
    queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

    # 5. Start watchdog first so watch_handler exists before the worker starts
    loop = asyncio.get_running_loop()
    observer, watch_handler = watcher.start_watcher(WATCH_FOLDER, loop, queue)

    # 6. Worker (passes mark_processed so it suppresses re-watch after muting)
    worker_task = asyncio.create_task(
        worker.run_worker(queue, DB_PATH, model, pattern, MUTE_PADDING, FFMPEG_BIN,
                          mark_processed=watch_handler.mark_processed,
                          get_words=lambda: _state.get("words", []))
    )

    # 7. Recover jobs that were processing when the service last crashed
    stale_jobs = await db.recover_stale_jobs(DB_PATH)
    if stale_jobs:
        logger.info("Re-enqueueing %d stale job(s).", len(stale_jobs))
        for j in stale_jobs:
            _enqueue(queue, j["priority"], j["file_path"], j["job_id"])

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

_static_dir = Path(__file__).parent / "static"
if _static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def ui():
    return FileResponse(str(_static_dir / "index.html"))


@app.get("/browse")
async def browse(path: str = "/media"):
    """List directory contents for the file browser UI."""
    target = Path(path).resolve()
    # Safety: only allow browsing under WATCH_FOLDER or /media
    watch_root = Path(WATCH_FOLDER).resolve()
    if target != watch_root and watch_root not in target.parents and target not in watch_root.parents:
        raise HTTPException(status_code=403, detail="Path outside allowed root")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Not a directory")

    entries = []
    for entry in sorted(target.iterdir(), key=lambda e: (e.is_file(), e.name.lower())):
        if entry.name.startswith(".") or entry.name.startswith("tmp"):
            continue
        entries.append({
            "name": entry.name,
            "path": str(entry),
            "is_dir": entry.is_dir(),
            "is_media": entry.is_file() and entry.suffix.lower() in MEDIA_EXTENSIONS,
        })

    return {
        "current": str(target),
        "parent": str(target.parent) if target != watch_root else None,
        "entries": entries,
    }


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
async def list_jobs(status: str | None = None, offset: int = 0):
    return await db.list_jobs(DB_PATH, limit=100, offset=offset, status=status)


@app.get("/jobs/queue-positions")
async def queue_positions():
    return await db.get_queue_positions(DB_PATH)


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
    # Priority 0 = high (processed next); watcher/scan jobs use priority 1.
    job_id = await db.create_job(DB_PATH, path, priority=0)
    queue: asyncio.PriorityQueue = _state["queue"]
    _enqueue(queue, 0, path, job_id)
    return SubmitJobResponse(job_id=job_id, file_path=path)


class FolderRequest(BaseModel):
    folder_path: str


@app.post("/jobs/folder", status_code=202)
async def submit_folder(body: FolderRequest):
    """Queue all unprocessed media files in a folder (non-recursive) at high priority."""
    folder = Path(body.folder_path)
    watch_root = Path(WATCH_FOLDER).resolve()
    target = folder.resolve()
    if target != watch_root and watch_root not in target.parents and target not in watch_root.parents:
        raise HTTPException(status_code=403, detail="Path outside allowed root")
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {body.folder_path}")

    existing = await db.list_jobs(DB_PATH, limit=100_000)
    already_queued = {j["file_path"] for j in existing["items"] if j["status"] in ("pending", "processing", "done")}

    queue: asyncio.PriorityQueue = _state["queue"]
    queued = []
    for entry in sorted(folder.iterdir(), key=lambda e: e.name.lower()):
        if entry.suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        if entry.name.startswith("tmp"):
            continue
        str_path = str(entry)
        if str_path in already_queued:
            continue
        job_id = await db.create_job(DB_PATH, str_path, priority=0)
        _enqueue(queue, 0, str_path, job_id)
        queued.append(str_path)

    return {"queued": len(queued), "files": queued}


@app.post("/reload", response_model=ReloadResponse)
async def reload_badwords():
    words = load_words(BADWORDS_PATH)
    pattern = build_pattern(words)
    _state["words"] = words
    _state["pattern"] = pattern
    logger.info("Word list reloaded: %d words.", len(words))
    return ReloadResponse(word_count=len(words))


@app.post("/jobs/{job_id}/reprocess", status_code=202)
async def reprocess_job(job_id: str):
    job = await db.get_job(DB_PATH, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] == "processing":
        raise HTTPException(status_code=409, detail="Job is currently processing")
    reset = await db.reset_job_for_reprocess(DB_PATH, job_id)
    if not reset:
        raise HTTPException(status_code=409, detail="Job could not be reset")
    queue: asyncio.PriorityQueue = _state["queue"]
    _enqueue(queue, 0, job["file_path"], job_id)
    return {"job_id": job_id, "status": "pending"}


@app.post("/jobs/{job_id}/skip", status_code=204)
async def skip_job(job_id: str):
    job = await db.get_job(DB_PATH, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("pending",):
        raise HTTPException(status_code=409, detail="Only pending jobs can be skipped")
    await db.update_job(DB_PATH, job_id, status="skipped")


class IgnoredWordsRequest(BaseModel):
    ignored_words: list[str]


@app.put("/jobs/{job_id}/ignored-words", status_code=200)
async def set_ignored_words(job_id: str, body: IgnoredWordsRequest):
    job = await db.get_job(DB_PATH, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    words = [w.strip().lower() for w in body.ignored_words if w.strip()]
    saved = await db.set_ignored_words(DB_PATH, job_id, words)
    if not saved:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "ignored_words": words}


@app.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str):
    deleted = await db.delete_job(DB_PATH, job_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found")


# ---------------------------------------------------------------------------
# Bad-words management
# ---------------------------------------------------------------------------

@app.get("/words")
async def list_words():
    return {"words": sorted(_state.get("words", []))}


class WordsRequest(BaseModel):
    words: list[str]


def _save_words(words: list[str]) -> None:
    Path(BADWORDS_PATH).write_text(
        "\n".join(sorted(set(words), key=str.lower)) + "\n", encoding="utf-8"
    )


@app.post("/words", status_code=201)
async def add_words(body: WordsRequest):
    current = list(_state.get("words", []))
    new_entries = [w.strip().lower() for w in body.words if w.strip()]
    merged = list(set(current) | set(new_entries))
    _save_words(merged)
    pattern = build_pattern(merged)
    _state["words"] = merged
    _state["pattern"] = pattern
    return {"word_count": len(merged)}


@app.delete("/words", status_code=200)
async def remove_words(body: WordsRequest):
    current = set(_state.get("words", []))
    to_remove = {w.strip().lower() for w in body.words if w.strip()}
    updated = list(current - to_remove)
    _save_words(updated)
    pattern = build_pattern(updated)
    _state["words"] = updated
    _state["pattern"] = pattern
    return {"word_count": len(updated)}
