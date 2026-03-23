import json
import os
import subprocess
import tempfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _build_filter_complex(
    segments: list[tuple[float, float]],
) -> str:
    """
    Build a filter_complex that splits the audio timeline around every mute
    segment using atrim+volume, then concatenates back.  This approach avoids
    any comma-in-expression issues entirely.

    Returns (filter_complex_string, extra_map_flags).
    """
    # Build the ordered list of intervals:
    # [normal, muted, normal, muted, ..., normal]
    segs = sorted(segments)
    parts: list[tuple[float | None, float | None, bool]] = []  # (start, end, is_muted)

    cursor = 0.0
    for s, e in segs:
        if cursor < s:
            parts.append((cursor, s, False))   # normal segment before this mute
        parts.append((s, e, True))              # muted segment
        cursor = e
    parts.append((cursor, None, False))         # tail (to end of file)

    lines = []
    seg_labels = []

    for i, (start, end, muted) in enumerate(parts):
        label = f"seg{i}"
        seg_labels.append(f"[{label}]")

        # atrim options — no function calls, no commas in expressions
        trim_opts = f"start={start:.6f}"
        if end is not None:
            trim_opts += f":end={end:.6f}"

        chain = f"[0:a]atrim={trim_opts},asetpts=PTS-STARTPTS"
        if muted:
            chain += ",volume=0"
        chain += f"[{label}]"
        lines.append(chain)

    n = len(parts)
    concat_inputs = "".join(seg_labels)
    lines.append(f"{concat_inputs}concat=n={n}:v=0:a=1[aout]")

    return ";".join(lines)


def _probe_audio(input_path: str, ffmpeg_bin: str) -> tuple[str, int, int]:
    """Return (codec_name, channels, bit_rate) for the first audio stream."""
    ffprobe = Path(ffmpeg_bin).parent / "ffprobe"
    cmd = [
        str(ffprobe), "-v", "quiet",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,channels,bit_rate",
        "-of", "json",
        input_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        codec = stream.get("codec_name", "aac")
        channels = int(stream.get("channels", 2))
        bitrate = int(stream.get("bit_rate", 0)) or _default_bitrate(codec, channels)
        return codec, channels, bitrate
    except Exception as e:
        logger.warning("ffprobe failed, using AAC fallback: %s", e)
        return "aac", 2, 192000


def _default_bitrate(codec: str, channels: int) -> int:
    per_channel = {"ac3": 96000, "eac3": 96000, "dts": 128000, "aac": 96000, "mp3": 128000}
    return per_channel.get(codec, 96000) * channels


_ENCODABLE = {"aac", "ac3", "eac3", "mp3", "flac", "opus", "vorbis", "pcm_s16le"}


def mute_file(
    input_path: str,
    segments: list[tuple[float, float]],
    ffmpeg_bin: str = "ffmpeg",
) -> None:
    """
    Mute *segments* in *input_path* in-place using FFmpeg.
    Uses atrim+concat filter_complex to avoid comma-escaping issues.
    Video and subtitle streams are stream-copied.
    """
    if not segments:
        logger.info("No profanity segments found in %s — skipping FFmpeg.", input_path)
        return

    orig_codec, channels, bitrate = _probe_audio(input_path, ffmpeg_bin)
    out_codec = orig_codec if orig_codec in _ENCODABLE else "aac"
    logger.info(
        "Audio: codec=%s → %s, channels=%d, bitrate=%d",
        orig_codec, out_codec, channels, bitrate,
    )

    filter_complex = _build_filter_complex(segments)
    logger.debug("filter_complex: %s", filter_complex)

    suffix = Path(input_path).suffix
    parent = Path(input_path).parent
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=parent)
    os.close(fd)

    def _run(extra_flags: list[str]) -> subprocess.CompletedProcess:
        cmd = [
            ffmpeg_bin, "-y",
            "-i", input_path,
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", out_codec,
            "-ac", str(channels),
            "-b:a", str(bitrate),
            *extra_flags,
            tmp_path,
        ]
        return subprocess.run(cmd, capture_output=True, text=True)

    try:
        logger.info("Running FFmpeg on %s (%d segments)", input_path, len(segments))
        result = _run(["-c:s", "copy"])

        if result.returncode != 0:
            logger.warning(
                "FFmpeg failed with subtitle copy, retrying without.\n%s",
                result.stderr[-500:],
            )
            result = _run([])
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed:\n{result.stderr[-1000:]}")

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
