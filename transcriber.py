import re
from dataclasses import dataclass

from faster_whisper import WhisperModel


@dataclass
class WordHit:
    word: str    # cleaned word that matched the bad-words pattern
    raw: str     # original word token from Whisper (may include punctuation)
    start: float # seconds into the media file
    end: float   # seconds into the media file


def load_model(model_size: str = "base", device: str = "cpu") -> WhisperModel:
    return WhisperModel(model_size, device=device, compute_type="int8")


def merge_segments(
    segments: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge overlapping or adjacent mute intervals."""
    if not segments:
        return []
    sorted_segs = sorted(segments, key=lambda s: s[0])
    merged = [sorted_segs[0]]
    for start, end in sorted_segs[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


_CLEAN_RE = re.compile(r"[^a-zA-Z']")


def transcribe(
    model: WhisperModel,
    file_path: str,
    pattern: re.Pattern,
    padding: float = 0.1,
) -> tuple[list[WordHit], list[tuple[float, float]]]:
    """
    Transcribe *file_path* and return every word that matches *pattern*,
    together with the merged mute segments (padded by *padding* seconds).
    """
    segments_gen, _info = model.transcribe(
        file_path,
        word_timestamps=True,
        language="en",
    )

    hits: list[WordHit] = []
    raw_segments: list[tuple[float, float]] = []

    # Fully consume the generator — faster-whisper is lazy
    for segment in list(segments_gen):
        if segment.words is None:
            continue
        for word_obj in segment.words:
            cleaned = _CLEAN_RE.sub("", word_obj.word).lower()
            if not cleaned:
                continue
            if pattern.fullmatch(cleaned):
                hits.append(
                    WordHit(
                        word=cleaned,
                        raw=word_obj.word,
                        start=word_obj.start,
                        end=word_obj.end,
                    )
                )
                raw_segments.append(
                    (max(0.0, word_obj.start - padding), word_obj.end + padding)
                )

    return hits, merge_segments(raw_segments)
