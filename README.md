# Supertonic TTS

A high-quality Flow-Matching based Text-to-Speech library using ONNX. This is a Python port of the Supertonic-2 web implementation.

## Features
- **10 Unique Voice Styles**: Professional male and female voices.
- **Auto-Downloader**: Automatically fetches models from HuggingFace to a global cache (`~/.cache/supertonic_tts`).
- **Word Timestamps**: Precise estimation of word-level timing.
- **Adjustable Parameters**: Control speed (0.9 - 1.4) and diffusion steps (3 - 14).
- **Lightweight Inference**: Runs on CPU/GPU via ONNX Runtime.

## Installation

```bash
pip install .
```

## Quick Start

```python
from supertonic_tts import SupertonicTTS, VOICE_STYLES, MIN_SPEED, MAX_SPEED

# 1. Initialize engine
# Models are automatically cached in ~/.cache/supertonic_tts
engine = SupertonicTTS()

# 2. Synthesize
# Returns:
# - audio: np.ndarray (float32, normalized -1 to 1)
# - sample_rate: int (44100)
# - word_timestamps: List[Dict] -> [{'word': str, 'start': float, 'end': float}]
audio, sr, word_timestamps = engine.synthesize(
    text="Hello! Welcome to Supertonic TTS.", 
    voice='F5', 
    speed=1.0, 
    steps=10
)

# 3. Calculate Total Duration
duration = len(audio) / sr
print(f"Generated {duration:.2f}s of audio")

# 4. Access Word Timing
for segment in word_timestamps:
    print(f"{segment['word']}: {segment['start']}s -> {segment['end']}s")

# 5. Save to file
engine.save_wav(audio, "output.wav")
```

## API Reference

### `SupertonicTTS.synthesize(text, voice='M3', steps=10, speed=1.0, lang=None)`
- **Parameters**:
  - `text` (str): Text to synthesize.
  - `voice` (str): Voice ID (`F1-F5`, `M1-M5`).
  - `steps` (int): Diffusion steps (`MIN_STEPS=3` to `MAX_STEPS=14`).
  - `speed` (float): Speed factor (`MIN_SPEED=0.9` to `MAX_SPEED=1.4`).
  - `lang` (str): Manual language override (e.g., 'en', 'ko'). Auto-detects if None.
- **Returns**: `(audio_data, sample_rate, word_timestamps)`

### `VOICE_STYLES`
A list of Pydantic models containing voice metadata:
```python
voice = VOICE_STYLES[0]
print(voice.id)          # 'F1'
print(voice.gender)      # 'female'
print(voice.description) # 'Correct and natural...'
```

## Author
Izzet Sezer <sezer@imsezer.com>
