import re
from dataclasses import dataclass
from typing import Iterable

from faster_whisper import WhisperModel


def build_pattern(words: Iterable[str]) -> re.Pattern:
    escaped = [re.escape(w) for w in words]
    return re.compile(r"(?<![a-zA-Z])(" + "|".join(escaped) + r")(?![a-zA-Z])", re.IGNORECASE)


@dataclass
class WordHit:
    word: str    # cleaned phrase/word that matched the bad-words pattern
    raw: str     # original token(s) from Whisper (may include punctuation)
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

# Longest possible phrase in the bad-words list (e.g. "as god is my witness" = 6)
_MAX_PHRASE_WORDS = 6


def transcribe(
    model: WhisperModel,
    file_path: str,
    pattern: re.Pattern,
    padding: float = 0.1,
    progress_cb=None,  # optional callable(pct: int) called during transcription
) -> tuple[list[WordHit], list[tuple[float, float]]]:
    """
    Transcribe *file_path* and return every word/phrase that matches *pattern*,
    together with the merged mute segments (padded by *padding* seconds).

    Uses a sliding window so multi-word phrases (e.g. "oh my god") are caught
    in addition to single bad words.
    """
    segments_gen, info = model.transcribe(
        file_path,
        word_timestamps=True,
        language="en",
    )
    total_duration = info.duration or 1.0

    # Collect all words as (cleaned_text, raw_text, start, end)
    all_words: list[tuple[str, str, float, float]] = []
    last_reported = -1
    for segment in segments_gen:
        if progress_cb and total_duration:
            pct = min(int(segment.end / total_duration * 90), 90)
            if pct != last_reported:
                progress_cb(pct)
                last_reported = pct
        if segment.words is None:
            continue
        for word_obj in segment.words:
            cleaned = _CLEAN_RE.sub("", word_obj.word).lower()
            if cleaned:
                all_words.append((cleaned, word_obj.word, word_obj.start, word_obj.end))

    hits: list[WordHit] = []
    raw_segments: list[tuple[float, float]] = []
    skip_until: int = 0  # index: skip words already consumed by a phrase match

    for i in range(len(all_words)):
        if i < skip_until:
            continue

        # Try longest windows first so phrases take priority over single words
        max_n = min(_MAX_PHRASE_WORDS, len(all_words) - i)
        for n in range(max_n, 0, -1):
            phrase = " ".join(w for w, _, _, _ in all_words[i:i + n])
            if pattern.fullmatch(phrase):
                start = all_words[i][2]
                end = all_words[i + n - 1][3]
                raw = " ".join(r for _, r, _, _ in all_words[i:i + n])
                hits.append(WordHit(word=phrase, raw=raw, start=start, end=end))
                raw_segments.append((max(0.0, start - padding), end + padding))
                skip_until = i + n  # don't re-match words inside this phrase
                break

    return hits, merge_segments(raw_segments)
