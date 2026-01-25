import os
import sys
import unittest
import numpy as np

# Add parent directory to path to import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supertonic_tts import SupertonicTTS, VOICE_STYLES, MIN_SPEED, MAX_SPEED, MIN_STEPS, MAX_STEPS

class TestSupertonicTTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = SupertonicTTS()

    def test_constants_visibility(self):
        self.assertTrue(len(VOICE_STYLES) == 10)
        self.assertEqual(MIN_SPEED, 0.9)
        self.assertEqual(MAX_SPEED, 1.4)
        self.assertEqual(MIN_STEPS, 3)
        self.assertEqual(MAX_STEPS, 14)

    def test_synthesis_basic(self):
        text = "Hello world."
        audio, sr, tokens = self.engine.synthesize(text, voice='M3')
        
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(sr, 44100) # Based on tts.json
        self.assertTrue(len(audio) > 0)
        self.assertIsInstance(tokens, list)
        if tokens:
            self.assertIn("word", tokens[0])
            self.assertIn("start", tokens[0])
            self.assertIn("end", tokens[0])

    def test_speed_variation(self):
        text = "This is a longer sentence to test speed variations accurately."
        audio_slow, _, _ = self.engine.synthesize(text, speed=0.9, steps=3)
        audio_fast, _, _ = self.engine.synthesize(text, speed=1.4, steps=3)
        # Slower speed should result in more samples (longer duration)
        self.assertTrue(len(audio_slow) > len(audio_fast))

    def test_step_variation_quality(self):
        text = "Step test."
        # Higher steps should theoretically be distinct or at least not fail
        audio_low, _, _ = self.engine.synthesize(text, steps=MIN_STEPS)
        audio_high, _, _ = self.engine.synthesize(text, steps=MAX_STEPS)
        self.assertTrue(len(audio_low) > 0)
        self.assertTrue(len(audio_high) > 0)

    def test_voices(self):
        # Test a few different voices
        voices = ['F1', 'M5']
        for voice in voices:
            audio, sr, _ = self.engine.synthesize("Test voice.", voice=voice)
            self.assertTrue(len(audio) > 0)

    def test_languages(self):
        # Test language detection/forcing
        texts = {
            "Hello": "en",
            "안녕하세요": "ko",
            "Hola mundo": "es"
        }
        for text, lang in texts.items():
            audio, sr, _ = self.engine.synthesize(text, lang=lang)
            self.assertTrue(len(audio) > 0)

if __name__ == '__main__':
    unittest.main()
