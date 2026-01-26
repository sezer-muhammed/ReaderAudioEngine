import os
import sys
import subprocess
import json
import site
import ctypes
import numpy as np

def patch_nvidia_paths():
    """
    On Windows, ensures that NVIDIA libraries (CUDA, cuDNN) installed via pip 
    are properly found by ONNX Runtime.
    """
    if sys.platform != "win32":
        return
    
    # Get all potential site-packages directories
    packages_dirs = site.getsitepackages()
    if hasattr(site, 'getusersitepackages'):
        user_site = site.getusersitepackages()
        if user_site not in packages_dirs:
            packages_dirs.append(user_site)
        
    # We want to add ALL nvidia subfolders' bin directories to the path
    # and to the DLL directory search list.
    for base_path in packages_dirs:
        nvidia_path = os.path.join(base_path, "nvidia")
        if not os.path.exists(nvidia_path):
            continue
            
        for sub in os.listdir(nvidia_path):
            bin_path = os.path.join(nvidia_path, sub, "bin")
            if os.path.exists(bin_path):
                try:
                    os.add_dll_directory(bin_path)
                except (OSError, AttributeError):
                    pass
                
                if bin_path not in os.environ["PATH"]:
                    os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]

    # Pre-loading critical DLLs can help ORT find them
    # Order: CUDA runtime -> cuBLas -> cuDNN
    critical_dlls = [
        "cudart64_12.dll", 
        "cublas64_12.dll", 
        "cublasLt64_12.dll", 
        "cudnn64_9.dll"
    ]
    
    for dll in critical_dlls:
        try:
            ctypes.WinDLL(dll)
        except Exception:
            pass

# Run patch BEFORE importing onnxruntime
patch_nvidia_paths()

import onnxruntime as ort
from huggingface_hub import hf_hub_download
from .utils import UnicodeProcessor, preprocess_text, detect_language

class SupertonicTTS:
    def __init__(self, device=None):
        """
        Initialize SupertonicTTS.
        
        :param device: Best available will be chosen if None. 
                      Options: 'cuda', 'cpu', 'DirectMLExecutionProvider', etc.
        """
        # Set up global user cache directory
        user_home = os.path.expanduser("~")
        self.global_cache_dir = os.path.join(user_home, '.cache', 'ebook_reader_supertonic')
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
        self.opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Automatically determine best providers
        available = ort.get_available_providers()
        print(f"Detected ONNX providers: {available}")
        
        providers = []
        # Only add TensorRT if it is available and nvinfer DLLs are present
        def trt_usable():
            if 'TensorrtExecutionProvider' not in available:
                return False
            # Check for nvinfer DLL (Windows) or libnvinfer.so (Linux)
            if sys.platform == 'win32':
                for p in os.environ['PATH'].split(os.pathsep):
                    if os.path.exists(os.path.join(p, 'nvinfer.dll')) or os.path.exists(os.path.join(p, 'nvinfer_10.dll')):
                        return True
                return False
            else:
                for p in os.environ['PATH'].split(os.pathsep):
                    if os.path.exists(os.path.join(p, 'libnvinfer.so')):
                        return True
                return False

        if device is None or device == 'cuda':
            if trt_usable():
                providers.append(('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_max_workspace_size': 2147483648,
                    'trt_fp16_enable': True,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': os.path.join(self.global_cache_dir, 'trt_cache')
                }))
            if 'CUDAExecutionProvider' in available:
                providers.append('CUDAExecutionProvider')
        elif device in available:
            providers.append(device)

        # Always add CPU as fallback
        if 'CPUExecutionProvider' not in [p[0] if isinstance(p, tuple) else p for p in providers]:
            providers.append('CPUExecutionProvider')
        
        # Load models
        print(f"Initializing models with providers: {providers}")
        try:
            self.dp_sess = ort.InferenceSession(os.path.join(self.assets_dir, 'duration_predictor.onnx'), self.opts, providers=providers)
            self.text_enc_sess = ort.InferenceSession(os.path.join(self.assets_dir, 'text_encoder.onnx'), self.opts, providers=providers)
            self.vector_est_sess = ort.InferenceSession(os.path.join(self.assets_dir, 'vector_estimator.onnx'), self.opts, providers=providers)
            self.vocoder_sess = ort.InferenceSession(os.path.join(self.assets_dir, 'vocoder.onnx'), self.opts, providers=providers)
            
            # Print which provider is actually being used
            actual_providers = self.dp_sess.get_providers()
            print(f"Active session providers: {actual_providers}")
        except Exception as e:
            print(f"Warning: Failed to initialize with preferred providers. Error: {e}")
            print("Falling back to CPU...")
            fallback_providers = ['CPUExecutionProvider']
            self.dp_sess = ort.InferenceSession(os.path.join(self.assets_dir, 'duration_predictor.onnx'), self.opts, providers=fallback_providers)
            self.text_enc_sess = ort.InferenceSession(os.path.join(self.assets_dir, 'text_encoder.onnx'), self.opts, providers=fallback_providers)
            self.vector_est_sess = ort.InferenceSession(os.path.join(self.assets_dir, 'vector_estimator.onnx'), self.opts, providers=fallback_providers)
            self.vocoder_sess = ort.InferenceSession(os.path.join(self.assets_dir, 'vocoder.onnx'), self.opts, providers=fallback_providers)
        
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
        Estimate word timestamps with punctuation-aware weighting.
        """
        words = text.split()
        if not words:
            return []
            
        words = [w for w in words if w.strip()]
        
        # Calculate weights based on characters + punctuation bonus
        # Punctuation at the end of a word increases the pause after it
        weights = []
        for w in words:
            weight = len(w)
            if w.endswith(('.', '!', '?')): weight += 4
            elif w.endswith((',', ';', ':')): weight += 2
            weights.append(weight)
        
        total_weight = sum(weights)
        if total_weight == 0: return []
        
        # We'll use 80% of time for actual speaking, 20% for gaps
        gap_total = total_duration * 0.20
        speak_total = total_duration * 0.80
        
        time_per_weight = speak_total / total_weight
        gap_per_word = gap_total / len(words)
        
        timestamps = []
        current_time = 0.05 # Tiny initial buffer
        
        for i, word in enumerate(words):
            duration = weights[i] * time_per_weight
            start_time = current_time
            end_time = start_time + duration
            
            timestamps.append({
                "word": word,
                "start": round(start_time, 3),
                "end": round(end_time, 3)
            })
            
            # Gap after word is proportional to punctuation
            gap = gap_per_word
            if word.endswith(('.', '!', '?')): gap *= 3.0
            elif word.endswith((',', ';')): gap *= 2.0
            
            current_time = end_time + gap
            
        return timestamps

    def save_wav(self, audio_data, path):
        import scipy.io.wavfile as wavfile
        # Normalize to 16-bit PCM
        audio_norm = np.clip(audio_data, -1, 1)
        audio_int = (audio_norm * 32767).astype(np.int16)
        wavfile.write(path, self.sample_rate, audio_int)

    def save_ogg_optimised(self, audio_data, path, bitrate='32k'):
        """
        Save audio as OGG Opus with optimized settings for speech using ffmpeg.
        """
        # Normalize to 16-bit PCM for input to ffmpeg pipe
        audio_norm = np.clip(audio_data, -1, 1)
        audio_int = (audio_norm * 32767).astype(np.int16)
        
        try:
            # Construct ffmpeg command
            # Input: s16le, 1 channel (mono), sample rate self.sample_rate
            # Output: libopus, 24000Hz sample rate, bitrate 32k
            
            cmd = [
                'ffmpeg',
                '-y', # Overwrite output
                '-f', 's16le', # Input format: signed 16-bit little endian
                '-ar', str(self.sample_rate), # Input sample rate
                '-ac', '1', # Input channels (assuming mono for TTS)
                '-i', 'pipe:0', # Input from stdin
                '-c:a', 'libopus', # Codec
                '-b:a', bitrate, # Bitrate
                '-ar', '24000', # Output sample rate (resample to 24k as requested)
                '-ac', '1', # Output channels
                path
            ]
            
            # Execute with pipe
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            out, err = process.communicate(input=audio_int.tobytes())
            
            if process.returncode != 0:
                print(f"FFmpeg error: {err.decode()}")
                # Fallback to WAV if ffmpeg fails
                wav_path = path.replace('.ogg', '.wav')
                print(f"Falling back to WAV: {wav_path}")
                self.save_wav(audio_data, wav_path)
                
        except Exception as e:
            print(f"Error saving optimized OGG: {e}")
            # Fallback
            wav_path = path.replace('.ogg', '.wav')
            print(f"Falling back to WAV: {wav_path}")
            self.save_wav(audio_data, wav_path)
