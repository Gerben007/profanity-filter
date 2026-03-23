import json
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
    """
    if not segments:
        return ""
    # Commas inside FFmpeg filter expressions must be escaped with \,
    # otherwise FFmpeg's filter graph parser treats them as filter separators.
    conditions = "+".join(
        f"between(t\\,{start:.6f}\\,{end:.6f})" for start, end in segments
    )
    return f"volume=if({conditions}\\,0\\,1)"


def _probe_audio(input_path: str, ffmpeg_bin: str) -> tuple[str, int, int]:
    """
    Return (codec_name, channels, bit_rate) for the first audio stream.
    Falls back to ("aac", 2, 192000) if probing fails.
    """
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
    """Sensible default bitrate when ffprobe doesn't report one."""
    per_channel = {"ac3": 96000, "eac3": 96000, "dts": 128000, "aac": 96000, "mp3": 128000}
    return per_channel.get(codec, 96000) * channels


# Codecs that FFmpeg can encode to (not all input codecs have encoders)
_ENCODABLE = {"aac", "ac3", "eac3", "mp3", "flac", "opus", "vorbis", "pcm_s16le"}


def mute_file(
    input_path: str,
    segments: list[tuple[float, float]],
    ffmpeg_bin: str = "ffmpeg",
) -> None:
    """
    Mute *segments* in *input_path* in-place using FFmpeg.
    Video and subtitle streams are stream-copied.
    Audio is re-encoded to match the original codec, channels, and bitrate.
    Does nothing if *segments* is empty.
    """
    if not segments:
        logger.info("No profanity segments found in %s — skipping FFmpeg.", input_path)
        return

    af = build_volume_filter(segments)
    suffix = Path(input_path).suffix
    parent = Path(input_path).parent

    # Detect original audio properties so we preserve quality
    orig_codec, channels, bitrate = _probe_audio(input_path, ffmpeg_bin)
    # Use original codec if encodable, else fall back to AAC
    out_codec = orig_codec if orig_codec in _ENCODABLE else "aac"
    logger.info(
        "Audio: codec=%s → %s, channels=%d, bitrate=%d",
        orig_codec, out_codec, channels, bitrate,
    )

    # Write temp file to the same directory so rename is atomic
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=parent)
    os.close(fd)

    def _run(extra_flags: list[str]) -> subprocess.CompletedProcess:
        cmd = [
            ffmpeg_bin, "-y",
            "-i", input_path,
            "-af", af,
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
            # Retry without subtitle copy — some containers reject -c:s copy
            logger.warning("FFmpeg failed with subtitle copy, retrying without.")
            result = _run([])
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed:\n{result.stderr[-1000:]}")

        # os.replace() can fail on Windows if the destination is locked —
        # fall back to delete-then-rename.
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
