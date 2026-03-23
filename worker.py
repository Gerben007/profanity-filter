import asyncio
import logging
import re

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
        # Support both bare strings (legacy / stale-job recovery) and tuples
        if isinstance(item, tuple):
            file_path, job_id = item
        else:
            file_path, job_id = item, None

        logger.info("Processing: %s", file_path)

        if job_id is None:
            job_id = await db.create_job(db_path, file_path)
        await db.update_job(db_path, job_id, status="processing")

        try:
            hits, segments = await asyncio.to_thread(
                transcriber.transcribe, model, file_path, pattern, padding
            )
            logger.info(
                "Transcription done: %d hit(s), %d mute segment(s)",
                len(hits),
                len(segments),
            )

            await asyncio.to_thread(
                processor.mute_file, file_path, segments, ffmpeg_bin
            )

            matches = [
                {"word": h.word, "raw": h.raw, "start": h.start, "end": h.end}
                for h in hits
            ]
            await db.update_job(
                db_path, job_id, status="done", matches=matches, segments=segments
            )
            logger.info("Job %s done (%d match(es)).", job_id, len(matches))
            if mark_processed:
                mark_processed(file_path)

        except asyncio.CancelledError:
            await db.update_job(
                db_path, job_id, status="failed", error_msg="Worker cancelled"
            )
            raise

        except Exception as exc:
            logger.exception("Job %s failed: %s", job_id, exc)
            await db.update_job(
                db_path, job_id, status="failed", error_msg=str(exc)
            )

        finally:
            queue.task_done()
