import asyncio
import logging
import re
from typing import Callable

from faster_whisper import WhisperModel

import db
import transcriber
import processor

logger = logging.getLogger(__name__)


async def run_worker(
    queue: asyncio.Queue,
    db_path: str,
    model: WhisperModel,
    pattern: re.Pattern,
    padding: float = 0.1,
    ffmpeg_bin: str = "ffmpeg",
    mark_processed=None,  # optional callable(path) to suppress re-watch
    get_words: Callable[[], list[str]] | None = None,  # returns current global word list
) -> None:
    """
    Single async worker that consumes (file_path, job_id | None) tuples from
    *queue* and processes them one at a time.  When job_id is None (watcher-
    enqueued files) a new DB record is created; when it is already set
    (manually submitted via the API) the existing record is reused.
    Runs until cancelled.
    """
    logger.info("Worker started.")
    while True:
        item = await queue.get()
        # Queue items: (priority, seq, file_path, job_id | None)
        if isinstance(item, tuple) and len(item) == 4:
            _priority, _seq, file_path, job_id = item
        elif isinstance(item, tuple) and len(item) == 2:
            file_path, job_id = item
        else:
            file_path, job_id = item, None

        logger.info("Processing: %s", file_path)

        # Fetch current job state to check for skip and ignored_words
        existing = await db.get_job(db_path, job_id) if job_id else None
        if existing and existing.get("status") == "skipped":
            logger.info("Skipping already-skipped job %s", job_id)
            queue.task_done()
            continue

        if job_id is None:
            job_id = await db.create_job(db_path, file_path)
        await db.update_job(db_path, job_id, status="processing")
        await db.update_progress(db_path, job_id, 0)

        # Build per-job pattern: subtract any ignored words from the global list
        ignored = (existing or {}).get("ignored_words") or []
        if ignored and get_words is not None:
            effective_words = [w for w in get_words() if w not in ignored]
            job_pattern = transcriber.build_pattern(effective_words) if effective_words else pattern
        else:
            job_pattern = pattern

        loop = asyncio.get_running_loop()

        def progress_cb(pct: int):
            asyncio.run_coroutine_threadsafe(
                db.update_progress(db_path, job_id, pct), loop
            )

        try:
            hits, segments = await asyncio.to_thread(
                transcriber.transcribe, model, file_path, job_pattern, padding,
                progress_cb,
            )
            logger.info(
                "Transcription done: %d hit(s), %d mute segment(s)",
                len(hits),
                len(segments),
            )

            matches = [
                {"word": h.word, "raw": h.raw, "start": h.start, "end": h.end}
                for h in hits
            ]
            # Store matches now so UI shows count while muting is in progress
            await db.update_job(db_path, job_id, status="processing", matches=matches, segments=segments)
            await db.update_progress(db_path, job_id, 95)

            await asyncio.to_thread(
                processor.mute_file, file_path, segments, ffmpeg_bin
            )

            await db.update_progress(db_path, job_id, 100)
            await db.update_job(
                db_path, job_id, status="done", matches=matches, segments=segments
            )
            logger.info("Job %s done (%d match(es)).", job_id, len(matches))
            if mark_processed:
                mark_processed(file_path)

        except asyncio.CancelledError:
            # Reset to pending so recover_stale_jobs re-queues it on next start
            await db.update_job(db_path, job_id, status="pending")
            raise

        except Exception as exc:
            logger.exception("Job %s failed: %s", job_id, exc)
            await db.update_job(
                db_path, job_id, status="failed", error_msg=str(exc)
            )

        finally:
            queue.task_done()
