import os
import json
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from .utils import UnicodeProcessor, preprocess_text, detect_language

class SupertonicTTS:
    def __init__(self, device='cpu'):
        """
        Initialize SupertonicTTS with the directory containing ONNX models and config.
        
        :param device: 'cpu' or 'cuda' (for GPU support)
        """
        # Set up global user cache directory
        user_home = os.path.expanduser("~")
        self.global_cache_dir = os.path.join(user_home, '.cache', 'supertonic_tts')
        self.assets_dir = os.path.join(self.global_cache_dir, 'assets', 'onnx')
        self.voices_dir = os.path.join(self.global_cache_dir, 'assets', 'voice_styles')

        # Ensure directories exist
        os.makedirs(self.assets_dir, exist_ok=True)
        os.makedirs(self.voices_dir, exist_ok=True)

        self._ensure_models()
            
        self.config_path = os.path.join(self.assets_dir, 'tts.json')
        self.indexer_path = os.path.join(self.assets_dir, 'unicode_indexer.json')
        
        # Load configuration
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.cfgs = json.load(f)
            
        # Load unicode indexer
        with open(self.indexer_path, 'r', encoding='utf-8') as f:
            self.indexer = json.load(f)
            
        self.processor = UnicodeProcessor(self.indexer)
        
        # ONNX Runtime session options
        self.opts = ort.SessionOptions()
        # self.opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # Default is usually good
        
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            providers = ['CUDAExecutionProvider'] + providers
        
        # Load models
        print(f"Loading models from {self.assets_dir}...")
        self.dp_sess = ort.InferenceSession(os.path.join(self.assets_dir, 'duration_predictor.onnx'), self.opts, providers=providers)
        self.text_enc_sess = ort.InferenceSession(os.path.join(self.assets_dir, 'text_encoder.onnx'), self.opts, providers=providers)
        self.vector_est_sess = ort.InferenceSession(os.path.join(self.assets_dir, 'vector_estimator.onnx'), self.opts, providers=providers)
        self.vocoder_sess = ort.InferenceSession(os.path.join(self.assets_dir, 'vocoder.onnx'), self.opts, providers=providers)
        
        self.sample_rate = self.cfgs['ae']['sample_rate']
        self.voice_cache = {}

    def _ensure_models(self):
        repo_id = "Supertone/supertonic-2"
        repo_type = "space"
        
        # List of required ONNX assets
        onnx_files = [
            "duration_predictor.onnx",
            "text_encoder.onnx",
            "vector_estimator.onnx",
            "vocoder.onnx",
            "tts.json",
            "unicode_indexer.json"
        ]
        
        # List of required voice styles
        voice_files = [
            f"{v}{i}.json" for v in ['F', 'M'] for i in range(1, 6)
        ]

        print("Checking/Downloading models from HuggingFace...")
        
        for f in onnx_files:
            target = os.path.join(self.assets_dir, f)
            if not os.path.exists(target):
                print(f"Downloading {f}...")
                path = hf_hub_download(
                    repo_id=repo_id, 
                    filename=f"assets/onnx/{f}", 
                    repo_type=repo_type, 
                    local_dir=self.global_cache_dir
                )
        
        for f in voice_files:
            target = os.path.join(self.voices_dir, f)
            if not os.path.exists(target):
                print(f"Downloading {f}...")
                path = hf_hub_download(
                    repo_id=repo_id, 
                    filename=f"assets/voice_styles/{f}", 
                    repo_type=repo_type, 
                    local_dir=self.global_cache_dir
                )

    def _load_voice_style(self, voice_name_or_path):
        if voice_name_or_path in self.voice_cache:
            return self.voice_cache[voice_name_or_path]
        
        if os.path.exists(voice_name_or_path):
            path = voice_name_or_path
        else:
            # Assume it's in .cache/voice_styles/
            path = os.path.join(self.voices_dir, f"{voice_name_or_path}.json")
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"Voice style file not found: {path}. Available voices: F1-F5, M1-M5")
            
        with open(path, 'r') as f:
            data = json.load(f)
            
        style_ttl = np.array(data['style_ttl']['data'], dtype=np.float32)
        style_dp = np.array(data['style_dp']['data'], dtype=np.float32)
        
        # Reshape if necessary (should match expected ONNX inputs)
        if 'dims' in data['style_ttl']:
            style_ttl = style_ttl.reshape(data['style_ttl']['dims'])
        if 'dims' in data['style_dp']:
            style_dp = style_dp.reshape(data['style_dp']['dims'])
            
        style = {'style_ttl': style_ttl, 'style_dp': style_dp}
        self.voice_cache[voice_name_or_path] = style
        return style

    def sample_noisy_latent(self, duration, bsz=1):
        base_chunk_size = self.cfgs['ae']['base_chunk_size']
        chunk_compress_factor = self.cfgs['ttl']['chunk_compress_factor']
        ldim = self.cfgs['ttl']['latent_dim']

        wav_lengths = [(int(d * self.sample_rate)) for d in duration]
        wav_len_max = max(wav_lengths)
        
        chunk_size = base_chunk_size * chunk_compress_factor
        latent_len = (wav_len_max + chunk_size - 1) // chunk_size
        latent_dim = ldim * chunk_compress_factor

        noisy_latent = np.random.normal(size=(bsz, latent_dim, latent_len)).astype(np.float32)
        
        latent_lengths = [(wl + chunk_size - 1) // chunk_size for wl in wav_lengths]
        
        # Manual masking
        latent_mask = np.zeros((bsz, 1, latent_len), dtype=np.float32)
        for i, length in enumerate(latent_lengths):
            latent_mask[i, 0, :length] = 1.0
            
        noisy_latent *= latent_mask
        return noisy_latent, latent_mask

    def synthesize(self, text, voice='M3', lang=None, steps=10, speed=1.0):
        """
        Synthesize speech from text.
        
        :param text: Text to synthesize
        :param voice: Voice name (e.g., 'F1', 'M3') or path to style JSON
        :param lang: Language code ('en', 'ko', etc.) or None for auto-detect
        :param steps: Denoising steps (default 10)
        :param speed: Speech speed (default 1.0)
        :return: (audio_data, sample_rate, word_timestamps)
        """
        if lang is None:
            lang = detect_language(text)
            
        # Get voice style
        style = self._load_voice_style(voice)
        style_ttl = style['style_ttl']
        style_dp = style['style_dp']
        
        # Preprocess and Tokenize
        encoded = self.processor([text], lang=lang)
        text_ids = encoded['text_ids']
        text_mask = encoded['text_mask']
        processed_text = encoded['processed_texts'][0]
        original_text = encoded['original_texts'][0]
        
        bsz = 1 # We process one text at a time for now
        
        # 1. Predict duration
        dp_inputs = {
            'text_ids': text_ids,
            'style_dp': style_dp,
            'text_mask': text_mask
        }
        duration = self.dp_sess.run(None, dp_inputs)[0] # JS output 'duration'
        
        # Apply speed (duration factor)
        # JS uses: speedToDurationFactor = 1 / (speed + 0.05)
        duration_factor = 1.0 / (speed + 0.05)
        duration_val = float(duration[0]) * duration_factor
        
        # Estimate word timestamps (rough estimation)
        # We know the total duration and the input text.
        # We can estimate word boundaries based on character lengths.
        word_timestamps = self._estimate_word_timestamps(original_text, duration_val)

        # 2. Encode text
        enc_inputs = {
            'text_ids': text_ids,
            'style_ttl': style_ttl,
            'text_mask': text_mask
        }
        text_emb = self.text_enc_sess.run(None, enc_inputs)[0] # JS output 'text_emb'
        
        # 3. Diffusion loop
        noisy_latent, latent_mask = self.sample_noisy_latent([duration_val], bsz=bsz)
        
        total_steps_arr = np.array([steps] * bsz, dtype=np.float32)
        
        current_latent = noisy_latent
        for step in range(steps):
            current_step_arr = np.array([step] * bsz, dtype=np.float32)
            
            est_inputs = {
                'noisy_latent': current_latent,
                'text_emb': text_emb,
                'style_ttl': style_ttl,
                'text_mask': text_mask,
                'latent_mask': latent_mask,
                'total_step': total_steps_arr,
                'current_step': current_step_arr
            }
            # Vector estimator returns 'denoised_latent'
            current_latent = self.vector_est_sess.run(None, est_inputs)[0]
            
        # 4. Vocode
        voc_inputs = {
            'latent': current_latent
        }
        wav = self.vocoder_sess.run(None, voc_inputs)[0] # JS output 'wav_tts'
        
        # Trim wav to predicted duration
        wav_len = int(duration_val * self.sample_rate)
        wav = wav.flatten()[:wav_len]
        
        return wav, self.sample_rate, word_timestamps

    def _estimate_word_timestamps(self, text, total_duration):
        """
        Roughly estimate word timestamps based on character counts.
        """
        words = text.split()
        if not words:
            return []
            
        # Filter out empty strings if any
        words = [w for w in words if w.strip()]
        
        total_chars = sum(len(w) for w in words)
        if total_chars == 0:
            return []
            
        timestamps = []
        current_time = 0.0
        
        # Add a small buffer for spaces/pauses
        space_ratio = 0.1 # 10% of time for spacing
        char_time_total = total_duration * (1 - space_ratio)
        space_time_total = total_duration * space_ratio
        
        avg_char_time = char_time_total / total_chars
        avg_space_time = space_time_total / max(1, len(words))
        
        for word in words:
            word_duration = len(word) * avg_char_time
            start_time = current_time
            end_time = start_time + word_duration
            
            timestamps.append({
                "word": word,
                "start": round(start_time, 3),
                "end": round(end_time, 3)
            })
            
            current_time = end_time + avg_space_time
            
        return timestamps

    def save_wav(self, audio_data, path):
        import scipy.io.wavfile as wavfile
        # Normalize to 16-bit PCM
        audio_norm = np.clip(audio_data, -1, 1)
        audio_int = (audio_norm * 32767).astype(np.int16)
        wavfile.write(path, self.sample_rate, audio_int)
