"""
GPU-accelerated word timestamp extraction using OpenAI Whisper.
"""
from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional

import numpy as np


class WhisperWordTimestampExtractor:
    """
    Extract word-level timestamps using OpenAI Whisper with GPU acceleration.
    """

    def __init__(
        self,
        *,
        model_size: str = "base",
        device: Optional[str] = None,
        compute_type: str = "float16",
    ):
        """
        Initialize Whisper timestamp extractor.

        :param model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        :param device: Device to use ('cuda', 'cpu', or None for auto)
        :param compute_type: Compute type ('float16', 'int8', 'float32')
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper is not installed. "
                "Install with: pip install faster-whisper"
            ) from e

        # Auto-detect device
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        self.device = device
        self.model_size = model_size
        self.compute_type = compute_type

        # Download and load model
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

    def extract(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
        text: str,
        lang: Optional[str] = None,
    ) -> List[Dict]:
        """
        Extract word-level timestamps from audio.

        :param audio: Audio data as numpy array
        :param sample_rate: Sample rate of the audio
        :param text: Expected text (for alignment)
        :param lang: Language code (optional)
        :return: List of word timestamps with 'word', 'start', 'end' keys
        """
        # Convert to float32 and normalize if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            from scipy import signal
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, num_samples)

        # Transcribe with word timestamps
        segments, info = self.model.transcribe(
            audio,
            language=lang,
            word_timestamps=True,
            condition_on_previous_text=False,
        )

        # Extract word-level timestamps
        words = []
        for segment in segments:
            if segment.words:
                for word_info in segment.words:
                    words.append({
                        "word": word_info.word.strip(),
                        "start": float(word_info.start),
                        "end": float(word_info.end),
                    })

        # If no words found, return empty list
        if not words:
            return []

        # Align with expected text
        expected_tokens = text.split()
        if not expected_tokens:
            return words

        # Simple alignment: match recognized words to expected tokens
        return self._align_words(words, expected_tokens)

    def _align_words(
        self,
        recognized_words: List[Dict],
        expected_tokens: List[str],
    ) -> List[Dict]:
        """
        Align recognized words with expected tokens.
        """
        import re

        def normalize(text: str) -> str:
            return re.sub(r"[^\w]", "", text.lower())

        aligned = []
        rec_idx = 0
        exp_idx = 0

        while exp_idx < len(expected_tokens) and rec_idx < len(recognized_words):
            expected = expected_tokens[exp_idx]
            recognized = recognized_words[rec_idx]

            exp_norm = normalize(expected)
            rec_norm = normalize(recognized["word"])

            if exp_norm == rec_norm:
                aligned.append({
                    "word": expected,
                    "start": recognized["start"],
                    "end": recognized["end"],
                })
                rec_idx += 1
                exp_idx += 1
            elif len(exp_norm) < len(rec_norm) and rec_norm.startswith(exp_norm):
                # Partial match - expected is prefix of recognized
                # Split the recognized word timing proportionally
                ratio = len(exp_norm) / len(rec_norm)
                duration = recognized["end"] - recognized["start"]
                split_time = recognized["start"] + duration * ratio

                aligned.append({
                    "word": expected,
                    "start": recognized["start"],
                    "end": split_time,
                })
                exp_idx += 1
                # Keep recognized word for next expected token
            else:
                # No match, advance the one with smaller normalized text
                if len(exp_norm) <= len(rec_norm):
                    # Use estimated timing based on position
                    if aligned:
                        last_end = aligned[-1]["end"]
                    elif rec_idx > 0:
                        last_end = recognized_words[rec_idx - 1]["end"]
                    else:
                        last_end = 0.0

                    # Estimate duration based on word length
                    est_duration = len(expected) * 0.08  # ~80ms per char
                    aligned.append({
                        "word": expected,
                        "start": last_end,
                        "end": last_end + est_duration,
                    })
                    exp_idx += 1
                else:
                    rec_idx += 1

        # Handle remaining expected tokens
        while exp_idx < len(expected_tokens):
            expected = expected_tokens[exp_idx]
            if aligned:
                last_end = aligned[-1]["end"]
            else:
                last_end = 0.0

            est_duration = len(expected) * 0.08
            aligned.append({
                "word": expected,
                "start": last_end,
                "end": last_end + est_duration,
            })
            exp_idx += 1

        return aligned


@lru_cache(maxsize=2)
def _get_whisper_extractor(
    model_size: str = "base",
    device: Optional[str] = None,
    compute_type: str = "float16",
) -> WhisperWordTimestampExtractor:
    return WhisperWordTimestampExtractor(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )


# Need to import here for the decorator
from functools import lru_cache
