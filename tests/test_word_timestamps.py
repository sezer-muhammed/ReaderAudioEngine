import os
import sys
import unittest

import numpy as np

# Add parent directory to path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ebook_reader_supertonic.word_timestamps.model_cache import (
    VoskModelError,
    ensure_vosk_en_us_lgraph_0_22,
)
from ebook_reader_supertonic.word_timestamps.align import align_expected_tokens_to_words
from ebook_reader_supertonic.word_timestamps.estimate import estimate_word_timestamps
from ebook_reader_supertonic.word_timestamps.extract import extract_word_timestamps


class TestWordTimestamps(unittest.TestCase):
    def test_estimate_word_timestamps_shape(self):
        text = "Hello world."
        ts = estimate_word_timestamps(text, total_duration_s=1.0)
        self.assertEqual(len(ts), 2)
        self.assertIn("word", ts[0])
        self.assertIn("start", ts[0])
        self.assertIn("end", ts[0])

    def test_align_expected_tokens_to_words_equal(self):
        expected = ["Hello", "world."]
        recognized = [
            {"word": "hello", "start": 0.1, "end": 0.3},
            {"word": "world", "start": 0.31, "end": 0.6},
        ]
        aligned = align_expected_tokens_to_words(
            expected_tokens=expected, recognized_words=recognized, total_duration_s=1.0
        )
        self.assertEqual([a["word"] for a in aligned], expected)
        self.assertAlmostEqual(aligned[0]["start"], 0.1, places=2)
        self.assertAlmostEqual(aligned[1]["end"], 0.6, places=2)

    def test_align_expected_tokens_to_words_missing_word_interpolates(self):
        expected = ["one", "two", "three"]
        recognized = [
            {"word": "one", "start": 0.0, "end": 0.2},
            {"word": "three", "start": 0.6, "end": 0.8},
        ]
        aligned = align_expected_tokens_to_words(
            expected_tokens=expected, recognized_words=recognized, total_duration_s=1.0
        )
        self.assertEqual(len(aligned), 3)
        self.assertTrue(aligned[0]["end"] <= aligned[1]["start"] + 0.05)
        self.assertTrue(aligned[1]["end"] <= aligned[2]["start"] + 0.05)

    def test_extract_word_timestamps_auto_falls_back(self):
        audio = np.zeros(44100, dtype=np.float32)
        # Ensure env isn't set for this test
        os.environ.pop("EBOOK_READER_VOSK_MODEL_PATH", None)
        os.environ.pop("VOSK_MODEL_PATH", None)
        os.environ["EBOOK_READER_VOSK_AUTO_DOWNLOAD"] = "0"
        try:
            ts = extract_word_timestamps(
                audio=audio,
                sample_rate=44100,
                text="Hello world.",
                backend="auto",
            )
        finally:
            os.environ.pop("EBOOK_READER_VOSK_AUTO_DOWNLOAD", None)
        self.assertEqual(len(ts), 2)

    def test_ensure_vosk_offline_raises_when_missing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            os.environ["VOSK_CACHE_DIR"] = d
            os.environ["VOSK_OFFLINE"] = "1"
            try:
                with self.assertRaises(VoskModelError):
                    ensure_vosk_en_us_lgraph_0_22()
            finally:
                os.environ.pop("VOSK_OFFLINE", None)
                os.environ.pop("VOSK_CACHE_DIR", None)

    def test_ensure_vosk_returns_cached_dir_without_network(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            os.environ["VOSK_CACHE_DIR"] = d
            os.environ["VOSK_OFFLINE"] = "1"
            try:
                model_dir = Path(d) / "models" / "en-us" / "vosk-model-en-us-0.22-lgraph"
                (model_dir / "am").mkdir(parents=True, exist_ok=True)
                (model_dir / "conf").mkdir(parents=True, exist_ok=True)
                (model_dir / "graph").mkdir(parents=True, exist_ok=True)
                self.assertEqual(ensure_vosk_en_us_lgraph_0_22(), model_dir)
            finally:
                os.environ.pop("VOSK_OFFLINE", None)
                os.environ.pop("VOSK_CACHE_DIR", None)


if __name__ == "__main__":
    unittest.main()
