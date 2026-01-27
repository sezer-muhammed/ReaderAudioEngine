from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class PcmAudio:
    pcm_s16le: bytes
    sample_rate: int
    duration_s: float


def float_to_pcm16le(audio: np.ndarray) -> bytes:
    if audio.dtype != np.float32 and audio.dtype != np.float64:
        audio = audio.astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio * 32767.0).astype(np.int16)
    return audio_i16.tobytes()


def resample_audio(audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if sample_rate == target_sample_rate:
        return audio

    try:
        from scipy.signal import resample_poly
    except Exception as e:  # pragma: no cover
        raise RuntimeError("scipy is required for Vosk resampling") from e

    ratio = Fraction(target_sample_rate, sample_rate).limit_denominator()
    up, down = ratio.numerator, ratio.denominator
    return resample_poly(audio, up, down).astype(np.float32)


def normalize_audio_for_vosk(
    audio: np.ndarray, sample_rate: int, target_sample_rate: int = 16000
) -> PcmAudio:
    if audio.ndim != 1:
        audio = audio.reshape(-1)

    audio_16k = resample_audio(audio.astype(np.float32), sample_rate, target_sample_rate)
    duration_s = float(len(audio_16k)) / float(target_sample_rate) if len(audio_16k) else 0.0
    return PcmAudio(
        pcm_s16le=float_to_pcm16le(audio_16k),
        sample_rate=target_sample_rate,
        duration_s=duration_s,
    )

