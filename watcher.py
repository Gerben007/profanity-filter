import asyncio
import logging
import os
import threading
import time
from pathlib import Path

from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

MEDIA_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".ts", ".wmv"}
STABILITY_POLL_INTERVAL = 2.0   # seconds between size polls
STABILITY_DURATION = 5.0        # seconds of unchanged size before enqueuing


class _MediaEventHandler(FileSystemEventHandler):
    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue) -> None:
        super().__init__()
        self._loop = loop
        self._queue = queue
        self._seen: set[str] = set()
        self._lock = threading.Lock()

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory:
            self._handle(event.src_path)

    def on_modified(self, event: FileModifiedEvent) -> None:
        if not event.is_directory:
            self._handle(event.src_path)

    def _handle(self, path: str) -> None:
        p = Path(path)
        if p.suffix.lower() not in MEDIA_EXTENSIONS:
            return
        # Ignore temp files written by processor.py (e.g. tmp7vzn_88h.mkv)
        if p.name.startswith("tmp"):
            return
        with self._lock:
            if path in self._seen:
                return
            self._seen.add(path)
        logger.info("New media file detected: %s — waiting for stability.", path)
        t = threading.Thread(
            target=_wait_for_stable,
            args=(path, self._loop, self._queue, self._seen, self._lock),
            daemon=True,
        )
        t.start()


def _wait_for_stable(
    path: str,
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue,
    seen: set[str],
    lock: threading.Lock,
) -> None:
    prev_size: int = -1
    stable_since: float = time.monotonic()

    while True:
        try:
            size = os.path.getsize(path)
        except FileNotFoundError:
            logger.warning("File disappeared before processing: %s", path)
            with lock:
                seen.discard(path)
            return

        now = time.monotonic()

        if size != prev_size:
            prev_size = size
            stable_since = now
        elif now - stable_since >= STABILITY_DURATION:
            logger.info("File stable, enqueueing: %s", path)
            loop.call_soon_threadsafe(queue.put_nowait, path)
            return

        time.sleep(STABILITY_POLL_INTERVAL)


def start_watcher(
    watch_folder: str,
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue,
) -> Observer:
    handler = _MediaEventHandler(loop, queue)
    observer = Observer()
    observer.schedule(handler, watch_folder, recursive=True)
    observer.start()
    logger.info("Watching folder: %s", watch_folder)
    return observer
