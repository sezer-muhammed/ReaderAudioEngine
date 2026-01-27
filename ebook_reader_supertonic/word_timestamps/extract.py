from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

from .estimate import estimate_word_timestamps
from .vosk import VoskWordTimestampExtractor
from .model_cache import VoskModelError, default_vosk_model_for_lang, ensure_vosk_model


@lru_cache(maxsize=4)
def _get_vosk_extractor(model_path: str) -> VoskWordTimestampExtractor:
    return VoskWordTimestampExtractor(model_path=model_path)


def resolve_vosk_model_path(explicit_model_path: Optional[str] = None) -> Optional[str]:
    if explicit_model_path:
        return explicit_model_path
    return (
        os.environ.get("EBOOK_READER_VOSK_MODEL_PATH")
        or os.environ.get("VOSK_MODEL_PATH")
        or None
    )

def _auto_download_enabled() -> bool:
    # Only used when backend is 'auto' or when backend is explicitly 'vosk' but model path isn't provided.
    return os.environ.get("EBOOK_READER_VOSK_AUTO_DOWNLOAD", "1") != "0"


def extract_word_timestamps(
    *,
    audio: np.ndarray,
    sample_rate: int,
    text: str,
    backend: str = "estimate",
    lang: Optional[str] = None,
    vosk_model_path: Optional[str] = None,
    fallback_to_estimate: bool = True,
) -> List[Dict]:
    backend = (backend or "estimate").lower()
    total_duration_s = float(len(audio)) / float(sample_rate) if len(audio) else 0.0

    if backend == "estimate":
        return estimate_word_timestamps(text, total_duration_s)

    if backend == "auto":
        resolved = resolve_vosk_model_path(vosk_model_path)
        if resolved is None:
            if _auto_download_enabled():
                spec = default_vosk_model_for_lang(lang)
                if spec is None:
                    return estimate_word_timestamps(text, total_duration_s)
                try:
                    resolved = str(ensure_vosk_model(spec))
                except VoskModelError:
                    return estimate_word_timestamps(text, total_duration_s)
            else:
                return estimate_word_timestamps(text, total_duration_s)
        backend = "vosk"
        vosk_model_path = resolved

    if backend == "vosk":
        try:
            resolved = resolve_vosk_model_path(vosk_model_path)
            if not resolved:
                if _auto_download_enabled():
                    spec = default_vosk_model_for_lang(lang)
                    if spec is None:
                        raise ValueError(
                            "Vosk model path is required for backend='vosk' (no default model for this language). "
                            "Set VOSK_MODEL_PATH or pass vosk_model_path."
                        )
                    resolved = str(ensure_vosk_model(spec))
                else:
                    raise ValueError(
                        "Vosk model path is required for backend='vosk'. Set VOSK_MODEL_PATH or pass vosk_model_path."
                    )
            extractor = _get_vosk_extractor(resolved)
            return extractor.extract(audio=audio, sample_rate=sample_rate, text=text, lang=lang)
        except Exception:
            if fallback_to_estimate:
                return estimate_word_timestamps(text, total_duration_s)
            raise

    raise ValueError(f"Unknown timestamps backend: {backend!r}")
