from supertonic_tts import SupertonicTTS, VOICE_STYLES, MIN_SPEED, MAX_SPEED, MIN_STEPS, MAX_STEPS
import numpy as np

def main():
    """
    This example demonstrates how to use the SupertonicTTS package with various settings.
    """
    # 1. Initialize the engine
    # This will automatically download models to ~/.cache/supertonic_tts if not present.
    engine = SupertonicTTS()

    # 2. Basic Synthesis parameters
    text = "Supertonic TTS is a high-quality flow-matching based speech synthesis system."
    
    # Selecting a voice style (you can use VOICE_STYLES[i].id or just strings like 'F1', 'M3')
    voice_choice = VOICE_STYLES[4] # F5: Professional and attractive
    
    print(f"Synthesizing using: {voice_choice.name} ({voice_choice.description})")

    # 3. Call synthesize
    # Parameters:
    # - text: The string to speak
    # - voice: Voice ID (F1-F5, M1-M5)
    # - steps: Number of diffusion steps (Higher = better quality, Lower = faster)
    # - speed: Speech rate (lower is slower, higher is faster)
    audio, sample_rate, word_timestamps = engine.synthesize(
        text=text,
        voice=voice_choice.id,
        steps=MAX_STEPS,  # Using 14 steps for best quality
        speed=1.0         # Normal speed
    )

    # 4. Understanding the Output
    # - audio: A NumPy array (float32) of normalized audio samples (-1 to 1).
    # - sample_rate: The sampling rate (always 44100 Hz for Supertonic).
    # - word_timestamps: A list of dicts: [{'word': str, 'start': float, 'end': float}]
    
    duration = len(audio) / sample_rate
    print(f"\n--- Output Info ---")
    print(f"Audio Buffer Type: {type(audio)}")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Total Duration: {duration:.2f} seconds")
    
    # 5. Extracting Word Metadata
    print("\n--- Word-by-Word Timestamps ---")
    for ts in word_timestamps:
        word = ts['word']
        start = ts['start']
        end = ts['end']
        word_duration = end - start
        print(f"[{start:5.2f}s -> {end:5.2f}s] {word:12} (Length: {word_duration:.2f}s)")

    # 6. Saving the output
    output_file = "comprehensive_example.wav"
    engine.save_wav(audio, output_file)
    print(f"\nSaved audio to: {output_file}")

    # 7. Speed limits example
    print(f"\nRecommended speed range: {MIN_SPEED} to {MAX_SPEED}")
    print(f"Recommended steps range: {MIN_STEPS} to {MAX_STEPS}")

if __name__ == "__main__":
    main()
