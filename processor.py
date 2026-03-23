import os
import subprocess
import tempfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_volume_filter(segments: list[tuple[float, float]]) -> str:
    """
    Build an FFmpeg `volume` audio filter expression that silences every
    interval in *segments* and leaves the rest at full volume.

    Example output for two segments:
        volume='if(between(t,1.2,1.8)+between(t,45.0,45.7),0,1)'
    """
    if not segments:
        return ""
    conditions = "+".join(
        f"between(t,{start:.6f},{end:.6f})" for start, end in segments
    )
    return f"volume='if({conditions},0,1)'"


def mute_file(
    input_path: str,
    segments: list[tuple[float, float]],
    ffmpeg_bin: str = "ffmpeg",
) -> None:
    """
    Mute *segments* in *input_path* in-place using FFmpeg.
    Video and subtitle streams are stream-copied; only audio is re-encoded.
    Does nothing if *segments* is empty.
    """
    if not segments:
        logger.info("No profanity segments found in %s — skipping FFmpeg.", input_path)
        return

    af = build_volume_filter(segments)
    suffix = Path(input_path).suffix
    parent = Path(input_path).parent

    # Write temp file to the same directory so os.replace() is atomic
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=parent)
    os.close(fd)

    try:
        cmd = [
            ffmpeg_bin, "-y",
            "-i", input_path,
            "-af", af,
            "-c:v", "copy",
            "-c:a", "aac",
            "-c:s", "copy",
            tmp_path,
        ]
        logger.info("Running FFmpeg on %s (%d segments)", input_path, len(segments))
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            # Retry without subtitle copy — some containers reject -c:s copy
            logger.warning(
                "FFmpeg failed with subtitle copy, retrying without: %s",
                result.stderr[-500:],
            )
            cmd_no_sub = [
                ffmpeg_bin, "-y",
                "-i", input_path,
                "-af", af,
                "-c:v", "copy",
                "-c:a", "aac",
                tmp_path,
            ]
            result2 = subprocess.run(cmd_no_sub, capture_output=True, text=True)
            if result2.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg failed:\n{result2.stderr[-1000:]}"
                )

        # os.replace() can fail on Windows if the destination is locked.
        # Fall back to: delete original, then rename temp into place.
        try:
            os.replace(tmp_path, input_path)
        except PermissionError:
            os.unlink(input_path)
            os.rename(tmp_path, input_path)
        logger.info("Muted %s successfully.", input_path)

    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
