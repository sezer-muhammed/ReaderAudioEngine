from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple


_EDGE_PUNCT_RE = re.compile(r"^[^\w']+|[^\w']+$", re.UNICODE)


def normalize_token(token: str) -> str:
    token = token.strip().lower()
    token = token.replace("’", "'").replace("`", "'")
    token = _EDGE_PUNCT_RE.sub("", token)
    token = re.sub(r"[^\w']+", "", token)
    return token


@dataclass(frozen=True)
class WordTiming:
    start: float
    end: float


def _clamp_monotonic(timings: List[WordTiming], epsilon: float = 0.02) -> List[WordTiming]:
    if not timings:
        return timings

    clamped: List[WordTiming] = []
    prev_end = 0.0
    for t in timings:
        start = max(t.start, prev_end - epsilon)
        end = max(t.end, start)
        clamped.append(WordTiming(start=start, end=end))
        prev_end = end
    return clamped


def align_expected_tokens_to_words(
    *,
    expected_tokens: Sequence[str],
    recognized_words: Sequence[Dict],
    total_duration_s: float,
) -> List[Dict]:
    expected_norm = [normalize_token(t) for t in expected_tokens]
    recog_norm = [normalize_token(w.get("word", "")) for w in recognized_words]

    matcher = SequenceMatcher(a=expected_norm, b=recog_norm, autojunk=False)

    mapped: List[Optional[WordTiming]] = [None] * len(expected_tokens)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                rw = recognized_words[j1 + k]
                mapped[i1 + k] = WordTiming(float(rw["start"]), float(rw["end"]))
            continue

        # For non-equal spans, interpolate timings over a reasonable window.
        span_len = i2 - i1
        if span_len <= 0:
            continue

        prev_end: Optional[float] = None
        for p in range(i1 - 1, -1, -1):
            if mapped[p] is not None:
                prev_end = mapped[p].end
                break

        next_start: Optional[float] = None
        for n in range(i2, len(mapped)):
            if mapped[n] is not None:
                next_start = mapped[n].start
                break

        window_start = prev_end
        window_end = next_start

        if window_start is None:
            if j1 < len(recognized_words):
                window_start = float(recognized_words[j1].get("start", 0.0))
            else:
                window_start = 0.0

        if window_end is None:
            if j2 > 0 and (j2 - 1) < len(recognized_words):
                window_end = float(recognized_words[j2 - 1].get("end", window_start))
            else:
                window_end = total_duration_s

        if window_end < window_start:
            window_end = window_start

        step = (window_end - window_start) / float(span_len) if span_len else 0.0
        for k in range(span_len):
            start = window_start + (k * step)
            end = window_start + ((k + 1) * step)
            mapped[i1 + k] = WordTiming(start=start, end=end)

    # Handle pure punctuation tokens (normalize -> ""), by snapping to neighbors when possible.
    for idx, tok in enumerate(expected_tokens):
        if expected_norm[idx] != "":
            continue
        prev_end = 0.0
        for p in range(idx - 1, -1, -1):
            if mapped[p] is not None:
                prev_end = mapped[p].end
                break
        mapped[idx] = WordTiming(start=prev_end, end=prev_end)

    timings = _clamp_monotonic([t or WordTiming(0.0, 0.0) for t in mapped])
    return [
        {"word": expected_tokens[i], "start": round(timings[i].start, 3), "end": round(timings[i].end, 3)}
        for i in range(len(expected_tokens))
    ]

