from __future__ import annotations

from typing import Dict, List


def estimate_word_timestamps(text: str, total_duration_s: float) -> List[Dict]:
    """
    Punctuation-aware heuristic timing. Used as a fallback when an aligner/ASR isn't available.
    """
    words = text.split()
    if not words:
        return []

    words = [w for w in words if w.strip()]

    weights = []
    for w in words:
        weight = len(w)
        if w.endswith((".", "!", "?")):
            weight += 4
        elif w.endswith((",", ";", ":")):
            weight += 2
        weights.append(weight)

    total_weight = sum(weights)
    if total_weight <= 0:
        return []

    gap_total = total_duration_s * 0.20
    speak_total = max(total_duration_s - gap_total, 0.0)

    time_per_weight = speak_total / total_weight if total_weight else 0.0
    gap_per_word = gap_total / len(words) if words else 0.0

    timestamps: List[Dict] = []
    current_time = 0.05

    for i, word in enumerate(words):
        duration = weights[i] * time_per_weight
        start_time = current_time
        end_time = start_time + duration

        timestamps.append(
            {
                "word": word,
                "start": round(start_time, 3),
                "end": round(end_time, 3),
            }
        )

        gap = gap_per_word
        if word.endswith((".", "!", "?")):
            gap *= 3.0
        elif word.endswith((",", ";")):
            gap *= 2.0

        current_time = end_time + gap

    return timestamps

