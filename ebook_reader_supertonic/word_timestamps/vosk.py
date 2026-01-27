from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from .align import align_expected_tokens_to_words
from .audio import normalize_audio_for_vosk


class VoskWordTimestampExtractor:
    """
    Extract word-level timestamps by recognizing the audio with Vosk,
    then aligning timings back onto the original tokenization (text.split()).
    """

    def __init__(
        self,
        *,
        model_path: Optional[str],
        target_sample_rate: int = 16000,
        chunk_frames: int = 4000,
    ):
        if not model_path:
            raise ValueError(
                "Vosk model path is required. Set EBOOK_READER_VOSK_MODEL_PATH or VOSK_MODEL_PATH."
            )

        try:
            from vosk import Model  # type: ignore
        except Exception as e:
            raise ImportError(
                "vosk is not installed. Install with: pip install vosk"
            ) from e

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found at: {model_path}")

        self._Model = Model
        self.model = Model(model_path)
        self.target_sample_rate = int(target_sample_rate)
        self.chunk_frames = int(chunk_frames)

    def _recognize_words(self, audio: np.ndarray, sample_rate: int) -> List[Dict]:
        try:
            from vosk import KaldiRecognizer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("vosk is not installed. Install with: pip install vosk") from e

        normalized = normalize_audio_for_vosk(
            audio=audio, sample_rate=sample_rate, target_sample_rate=self.target_sample_rate
        )
        rec = KaldiRecognizer(self.model, normalized.sample_rate)
        rec.SetWords(True)

        results: List[Dict] = []
        chunk_size_bytes = self.chunk_frames * 2  # s16le mono
        pcm = normalized.pcm_s16le
        for i in range(0, len(pcm), chunk_size_bytes):
            chunk = pcm[i : i + chunk_size_bytes]
            if rec.AcceptWaveform(chunk):
                results.append(json.loads(rec.Result()))
        results.append(json.loads(rec.FinalResult()))

        words: List[Dict] = []
        for r in results:
            if "result" in r and isinstance(r["result"], list):
                for w in r["result"]:
                    if "word" in w and "start" in w and "end" in w:
                        words.append(
                            {
                                "word": w["word"],
                                "start": float(w["start"]),
                                "end": float(w["end"]),
                                **({"conf": float(w["conf"])} if "conf" in w else {}),
                            }
                        )

        # Ensure monotonic order
        words.sort(key=lambda d: (d["start"], d["end"]))
        return words

    def extract(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
        text: str,
        lang: Optional[str] = None,
    ) -> List[Dict]:
        recognized_words = self._recognize_words(audio=audio, sample_rate=sample_rate)
        expected_tokens = text.split()
        total_duration_s = float(len(audio)) / float(sample_rate) if len(audio) else 0.0

        if not expected_tokens:
            return []
        if not recognized_words:
            return [
                {"word": w, "start": 0.0, "end": 0.0}
                for w in expected_tokens
            ]

        return align_expected_tokens_to_words(
            expected_tokens=expected_tokens,
            recognized_words=recognized_words,
            total_duration_s=total_duration_s,
        )

