import numpy as np
import torch
import torchaudio
import signal
import random
import threading, queue, signal
from queue import Queue

import torch.nn.functional as F

from einops import rearrange
import time
import math

import datetime
import numpy as np
from collections import deque

import jack
import threading
from threading import Lock, Event

from pythonosc import dispatcher, osc_server

import asyncio

import json
import os
from typing import Dict, Optional, Union
from prefigure.prefigure import get_all_args, push_wandb_config
from stable_audio_tools.data.dataset import create_dataloader_from_config, fast_scandir
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.inference.utils import prepare_audio

from stable_audio_tools.models.transformer import ContinuousTransformer
from mappers.mapper_models import flatten_and_offset_latents, unflatten_and_unoffset_token_by_token, sample_ar_w_temp_cond_kv, ARMapper

from pybela import Streamer




# -----------------------------
# Playback gating & monitoring
# -----------------------------
playback_primed = Event()   # set only after first non-zero audio write
RING_MONITOR_INTERVAL = 0.25  # seconds

START_TIME = time.time()
SECTION_2_START_TIME = START_TIME + 30000
SECTION_3_START_TIME = SECTION_2_START_TIME + 30000

# Stats
ring_samples_written = 0
decode_writes = 0
tokens_ingested = 0

phrase_detector = None

stream_duration_multiplier = 2.0

def osc_set_multiplier(address, *args):
    global stream_duration_multiplier
    try:
        val = float(args[0])
        stream_duration_multiplier = max(0.1, val)  # clamp to sensible minimum
        print(f"[OSC] Updated stream_duration_multiplier = {stream_duration_multiplier}")
    except Exception as e:
        print(f"[OSC] Bad multiplier message: {args} ({e})")

# dispatcher = dispatcher.Dispatcher()
# dispatcher.map("/stream/multiplier", osc_set_multiplier)


# -----------------------------
# Persistent Bela Streamer
# -----------------------------
bela_streamer = Streamer(ip="192.168.30.1") # changed ip to allow wireless streaming from Bela AccessPoint
bela_channels = ["emg1", "emg2", "emg3", "emg4", "emg5", "emg6"]
bela_connected = False

from scipy.signal import iirnotch, filtfilt
def notch_filter(data, fs=44100, freq=50.0, Q=30.0):
    """
    Apply a 50 Hz notch filter to EMG data.
    data: np.ndarray [channels, samples]
    fs: sampling rate of EMG (Hz)
    freq: notch frequency (Hz)
    Q: quality factor (controls notch width)
    """
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, data, axis=-1)

def trim_to_shortest(a, b):
    """Trim the longer of two tensors to the length of the shorter one."""
    if a.shape[-1] > b.shape[-1]:
        return a[:,:,:b.shape[-1]], b
    elif b.shape[-1] > a.shape[-1]:
        return a, b[:,:,:a.shape[-1]]
    return a, b

if torch.cuda.is_available():
    print("cuda is available")
else:
    print("cuda is NOT available")
device = "cuda" if torch.cuda.is_available() else "cpu"
print('gpu device:', device)


with open('./rvq_testing_ckpts/stable_audio_2_0_vae_audio_rvq.json') as f:
    model_config = json.load(f)

# load audio ckpt:
audio_enc_pth = './modality_branches/audio_unwrapped_monday.ckpt'#./rvq_testing_ckpts/audio_rvq_unwrapped.ckpt'

audio_enc = create_model_from_config(model_config)
print('audio_enc architecture set')

audio_checkpoint = torch.load(audio_enc_pth, map_location="cpu")
print('audio_checkpoint loaded')

try:
    state_dict = audio_checkpoint["state_dict"]
    print("Loaded wrapped audio_enc audio_checkpoint (with 'state_dict').")
except (KeyError, TypeError):
    state_dict = audio_checkpoint
    print("Loaded raw checkpoint (direct state_dict).")

audio_enc.load_state_dict(state_dict)
print('audio_enc weights set from', audio_enc_pth)

for param in audio_enc.parameters():
    param.requires_grad = False

audio_enc.to(device)
print('audio_enc on', device)
audio_enc.eval()
print('evaluation mode')


#load emg ckpt:

with open('./rvq_testing_ckpts/stable_audio_2_0_vae_emg_rvq.json') as f:
    emg_config = json.load(f)

emg_pth = './modality_branches/emg_unwrapped_monday.ckpt'#'./rvq_testing_ckpts/emg_rvq_unwrapped.ckpt'

emg_enc = create_model_from_config(emg_config)
print('emg architecture set')

emg_checkpoint = torch.load(emg_pth, map_location="cpu")
print('checkpoint loaded')

try:
    emg_state_dict = emg_checkpoint["state_dict"]
    print("Loaded wrapped checkpoint (with 'state_dict').")
except (KeyError, TypeError):
    emg_state_dict = emg_checkpoint
    print("Loaded raw checkpoint (direct state_dict).")

emg_enc.load_state_dict(emg_state_dict)
print('model weights set from', emg_pth)

for param in emg_enc.parameters():
    param.requires_grad = False

emg_enc.to(device)
print('model on', device)
emg_enc.eval()
print('evaluation mode')


# load transformer ckpt:

#mapper = ARMapper(vocab_size=8192, d_model=384, n_layers=6, n_heads=6).to(device)
mapper = ARMapper(vocab_size=8192, d_model=384, n_layers=6, n_heads=6).to(device)

#mapper_pth = './modality_branches/rvq_audio_transformer_causal_cross_8dec2025_43.pt'
mapper_pth = './modality_branches/rvq_audio_transformer_causal_cross_8dec2025_100.pt'

mapper_ckpt = torch.load(mapper_pth, map_location="cpu")
print('checkpoint loaded')

try:
    state_dict = mapper_ckpt["state_dict"]
    print("Loaded wrapped checkpoint (with 'state_dict').")
except (KeyError, TypeError):
    state_dict = mapper_ckpt
    print("Loaded raw checkpoint (direct state_dict).")

mapper.load_state_dict(state_dict, strict=False)
#mapper.load_state_dict(state_dict)
print('model weights set from', emg_pth)

for param in mapper.parameters():
    param.requires_grad = False

mapper.to(device)
print('model on', device)
mapper.eval()
print('evaluation mode')


print('+++++++++++++++++++++')

def design_notch(fs=1000, freq=50.0, Q=30.0, device="cpu"):
    """
    Design IIR notch filter coefficients (biquad).
    Returns (b, a) as torch tensors on the given device.
    """
    w0 = 2.0 * torch.pi * torch.tensor(freq / fs, device=device)
    alpha = torch.sin(w0) / (2.0 * Q)

    b0 = 1.0
    b1 = -2.0 * torch.cos(w0)
    b2 = 1.0
    a0 = 1.0 + alpha
    a1 = -2.0 * torch.cos(w0)
    a2 = 1.0 - alpha

    b = torch.tensor([b0, b1, b2], dtype=torch.float32, device=device) / a0
    a = torch.tensor([1.0, a1 / a0, a2 / a0], dtype=torch.float32, device=device)
    return b, a

def apply_iir_notch(x, b, a):
    """
    Apply notch filter to EMG data using direct-form II.
    x: [batch, channels, samples] tensor
    """
    B, C, N = x.shape
    y = torch.zeros_like(x)
    for c in range(C):
        xv = torch.zeros(3, device=x.device)
        yv = torch.zeros(3, device=x.device)
        for n in range(N):
            xv[0] = x[0, c, n]
            yv[0] = b[0]*xv[0] + b[1]*xv[1] + b[2]*xv[2] - a[1]*yv[1] - a[2]*yv[2]
            y[0, c, n] = yv[0]
            xv[2], xv[1] = xv[1], xv[0]
            yv[2], yv[1] = yv[1], yv[0]
    return y
# -----------------------------
# JACK Audio Input Configuration
# -----------------------------
JACK_CLIENT_NAME = "rvq_transformer_output_streamer"
JACK_INPUT_SOURCE = "REAPER:out3"
JACK_TARGET_PORT = "REAPER:in3"
AUDIO_INPUT_DURATION_SEC = 0.1
AUDIO_SAMPLE_RATE = 44100

# -----------------------------
# Audio input capture buffer
# -----------------------------
audio_input_lock = Lock()
audio_input_buffer = deque(maxlen=int(AUDIO_SAMPLE_RATE * AUDIO_INPUT_DURATION_SEC * 2))

FADE_OUT_SAMPLES = 0#4410
FADE_IN_SAMPLES = 0#4410
fade_in_ramp = torch.linspace(0, 1, FADE_IN_SAMPLES, device=device).view(1, 1, -1)
fade_out_ramp = torch.linspace(1, 0, FADE_OUT_SAMPLES, device=device).view(1, 1, -1)

# def apply_fade(audio):
#     """Apply pre-computed fade in/out to audio."""
#     audio[:, :, :FADE_IN_SAMPLES].mul_(fade_in_ramp)
#     audio[:, :, -FADE_OUT_SAMPLES:].mul_(fade_out_ramp)
#     return audio

def apply_fade(segment, sample_rate, fade_in_duration=0.01, fade_out_duration=0.01):
    """Apply crossfades that will overlap-add correctly"""
    fade_in_samples = int(sample_rate * fade_in_duration)
    fade_out_samples = int(sample_rate * fade_out_duration)
    
    result = segment.copy()
    
    # Fade in at start
    if fade_in_samples > 0 and len(result) > fade_in_samples:
        fade_in_curve = np.linspace(0, 1, fade_in_samples)
        result[:fade_in_samples] *= fade_in_curve
    
    # Fade out at end
    if fade_out_samples > 0 and len(result) > fade_out_samples:
        fade_out_curve = np.linspace(1, 0, fade_out_samples)
        result[-fade_out_samples:] *= fade_out_curve
    
    return result, fade_in_samples, fade_out_samples


def concatenate_with_overlap(segments_with_fades):
    """Properly overlap-add segments instead of concatenating"""
    if not segments_with_fades:
        return np.array([])
    
    # Start with first segment
    result = segments_with_fades[0]['audio'].copy()
    
    for i in range(1, len(segments_with_fades)):
        current = segments_with_fades[i]['audio']
        overlap_samples = segments_with_fades[i]['fade_in_samples']
        
        if overlap_samples > 0:
            # Overlap-add the crossfade region
            result[-overlap_samples:] += current[:overlap_samples]
            # Append the rest
            result = np.concatenate([result, current[overlap_samples:]])
        else:
            # No overlap, just concatenate
            result = np.concatenate([result, current])
    
    return result

# -----------------------------
# Phrase Detector (silence -> phrase start)
# -----------------------------


class PhraseStartDetector:
    def __init__(self, samplerate=44100, hop_size=512,
                 silence_db=-50.0, min_silence_ms=300,
                 callback_phrase_start=None):
        self.sr = samplerate
        self.hop = hop_size
        self.silence_lin = 10 ** (silence_db / 20.0)
        self.min_silence = min_silence_ms / 1000.0
        self.cb_start = callback_phrase_start or (lambda t: None)

        self.in_phrase = False
        self.last_silence_t = None


    def _now(self):
        return time.time()

    def process_block(self, block):
        """block: np.ndarray of shape (frames,) or (frames, channels)"""
        t = self._now()
        x = block
        if x.ndim == 2:
            x = np.mean(x, axis=1)

        rms = np.sqrt(np.mean(x**2) + 1e-12)

        if rms > self.silence_lin:
            # Audio is active
            if not self.in_phrase:
                # Check if we were silent long enough
                if self.last_silence_t is None or (t - self.last_silence_t) >= self.min_silence:
                    self.in_phrase = True
                    self.cb_start(t)
        else:
            # Audio is silent
            if self.in_phrase:
                self.in_phrase = False
            self.last_silence_t = t

# -----------------------------
# JACK setup
# -----------------------------
client = jack.Client(JACK_CLIENT_NAME)
outport = client.outports.register("output_1")
inport = client.inports.register("input_1")

# Shared state for JACK callback
audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=16)
current_audio: np.ndarray | None = None
cursor: int = 0

# Statistics counters
queue_get_success = 0
queue_get_empty = 0

audio_lock = Lock()




def calibrate_noise_floor(duration_sec=2.0, margin_db=10.0, mode="ambient"):
    """
    Calibrate silence threshold.
    mode="ambient" → measure room tone only
    mode="playback" → generate audio and measure leakage during playback
    """
    if mode == "ambient":
        print(f"[CALIBRATION] Ambient mode: measuring {duration_sec}s of room tone...")
        time.sleep(duration_sec)
        with audio_input_lock:
            captured = np.array(list(audio_input_buffer), dtype=np.float32)

        if len(captured) == 0:
            print("[CALIBRATION] No samples captured, using default -50 dB")
            return -50.0

        rms = np.sqrt(np.mean(captured**2) + 1e-12)
        rms_db = 20 * np.log10(rms)
        threshold_db = rms_db + margin_db
        print(f"[CALIBRATION] Ambient RMS={rms_db:.1f} dBFS → threshold={threshold_db:.1f} dBFS")
        return threshold_db

    elif mode == "playback":
        print(f"[CALIBRATION] Playback mode: generating audio for {duration_sec}s...")
        # Random start tokens
        B, T = 1, 60
        start_tokens = torch.randint(low=0, high=2048, size=(B, T), device=device)

        with torch.no_grad():
            latents = unflatten_and_unoffset_token_by_token(
                start_tokens,
                T=None,
                num_quantizers=pipeline_state.num_quantizers,
                codebook_size=2048
            )
            audio_chunk = audio_enc.decode_tokens(latents)
            audio_np = audio_chunk.squeeze().cpu().numpy().astype(np.float32)


        
        # Loop playback until duration_sec is covered
        samples_needed = int(AUDIO_SAMPLE_RATE * duration_sec)
        looped = np.tile(audio_np, int(np.ceil(samples_needed / len(audio_np))))
        looped = looped[:samples_needed]

        with ring_lock:
            audio_ring_buffer.extend(looped)
            playback_primed.set()

        # Capture mic input while playback runs
        time.sleep(duration_sec)
        with audio_input_lock:
            captured = np.array(list(audio_input_buffer), dtype=np.float32)

        if len(captured) == 0:
            print("[CALIBRATION] No samples captured, using default -50 dB")
            return -50.0

        rms = np.sqrt(np.mean(captured**2) + 1e-12)
        rms_db = 20 * np.log10(rms)
        threshold_db = rms_db + margin_db
        print(f"[CALIBRATION] Leakage RMS={rms_db:.1f} dBFS → threshold={threshold_db:.1f} dBFS")
        return threshold_db

    else:
        raise ValueError(f"Unknown calibration mode: {mode}")



def gated_callback(t, silence_mode="generation"): # output generation combined
    if not bela_streamer.is_streaming() and jack_silence(mode=silence_mode):
        pipeline_active.set()


# -----------------------------
# Ring Buffer for Continuous Audio
# -----------------------------
from collections import deque

# Globals (near ring buffer)
play_blocks = 0
play_samples_total = 0
ring_samples_written = 0
last_out_last_sample = 0.0


class AdditiveRingBuffer:
    def __init__(self, capacity: int):
        self.buf = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.write_idx = 0
        self.read_idx = 0

    def __len__(self):
        if self.write_idx >= self.read_idx:
            return self.write_idx - self.read_idx
        else:
            return self.capacity - (self.read_idx - self.write_idx)

    def clear(self):
        """Reset buffer to empty state."""
        self.buf[:] = 0.0
        self.write_idx = 0
        self.read_idx = 0

    def add_at_old(self, samples: np.ndarray, start_cursor: int):
        """Accumulate samples into buffer starting at absolute sample cursor."""
        idx = start_cursor % self.capacity
        n = len(samples)
        end = idx + n
        if end <= self.capacity:
            self.buf[idx:end] += samples
        else:
            first = self.capacity - idx
            self.buf[idx:] += samples[:first]
            self.buf[:end - self.capacity] += samples[first:]

    def add_at(self, samples: np.ndarray, start_cursor: int):
        """Accumulate samples into buffer starting at absolute sample cursor."""
        idx = start_cursor % self.capacity
        n = len(samples)

        print(f"[RING ADD_AT] start_cursor={start_cursor}, idx={idx}, n={n}, max_val={np.max(np.abs(samples)):.4f}")

        end = idx + n
        
        # Accumulate samples
        if end <= self.capacity:
            before = self.buf[idx:end].copy()
            self.buf[idx:end] += samples
            after = self.buf[idx:end].copy()
            print(f"[RING ADD_AT] Overlap region had {np.count_nonzero(before > 1e-6)} non-zero samples before")
            print(f"[RING ADD_AT] After add: {np.count_nonzero(after > 1e-6)} non-zero samples")
        else:
            first = self.capacity - idx
            self.buf[idx:] += samples[:first]
            self.buf[:end - self.capacity] += samples[first:]
        
        # Track furthest written position
        expected_end = (start_cursor + n) % self.capacity
        # Only advance write_idx forward, never bckward
        if self.write_idx < start_cursor:
            self.write_idx = expected_end
        else:
            self.write_idx = max(self.write_idx, expected_end)

    def extend(self, samples: np.ndarray):
        #Append samples sequentially
        n = len(samples)
        end = self.write_idx + n
        if end <= self.capacity:
            self.buf[self.write_idx:end] = samples
        else:
            first = self.capacity - self.write_idx
            self.buf[self.write_idx:] = samples[:first]
            self.buf[:end - self.capacity] = samples[first:]
        self.write_idx = (self.write_idx + n) % self.capacity

    def read(self, n: int) -> np.ndarray:
        #Read n samples sequentially (zero‑fill if not enough).
        end = self.read_idx + n
        if end <= self.capacity:
            out = self.buf[self.read_idx:end].copy()
        else:
            first = self.capacity - self.read_idx
            out = np.concatenate([
                self.buf[self.read_idx:].copy(),
                self.buf[:end - self.capacity].copy()
            ])
        self.read_idx = (self.read_idx + n) % self.capacity
        return out
    
    def peek(self, n: int) -> np.ndarray:
        #Look at next n samples without advancing index
        end = self.read_idx + n
        if end <= self.capacity:
            return self.buf[self.read_idx:end].copy()
        else:
            first = self.capacity - self.read_idx
            return np.concatenate([
                self.buf[self.read_idx:].copy(),
                self.buf[:end - self.capacity].copy()
            ])





RING_BUFFER_SIZE = AUDIO_SAMPLE_RATE * 30  # 30 seconds of audio
audio_ring_buffer = AdditiveRingBuffer(RING_BUFFER_SIZE)
ring_lock = Lock()

last_out_array = np.zeros(0, dtype=np.float32) 

def process(frames: int):

    global last_out_array

    out_buf = outport.get_buffer()
    out_array = np.frombuffer(out_buf, dtype=np.float32)
    #last_out_array = out_array.copy()

    if not playback_primed.is_set():
        out_array[:] = 0.0
    else:
        with ring_lock:
            out_chunk = audio_ring_buffer.read(frames)
            if len(out_chunk) < frames:
                out_array[:len(out_chunk)] = out_chunk
                out_array[len(out_chunk):] = 0.0
            else:
                out_array[:] = out_chunk

    last_out_array = out_array.copy()
    in_buf = inport.get_buffer()
    in_array = np.frombuffer(in_buf, dtype=np.float32).copy()
    with audio_input_lock:
        audio_input_buffer.extend(in_array)

     # gate phrase detection externally
    if phrase_detector is not None:
        phrase_detector.process_block(in_array)




client.set_process_callback(process)

def upsample(src_latent):
    src_len = src_latent.size(2)
    tgt_len = src_len * 2
    
    if src_len == tgt_len:
        return src_latent
    
    if src_len < tgt_len:
        src_latent = F.interpolate(src_latent, size=tgt_len, mode='linear', align_corners=True)
    else:
        print('strangeness with samplerate')
    
    return src_latent

def ring_monitor_thread():
    #Periodically report ring buffer level and primed state
    while True:
        time.sleep(RING_MONITOR_INTERVAL)
        with ring_lock:
            level = len(audio_ring_buffer)
        primed = playback_primed.is_set()
        #print(f"[RING] primed={primed}, level={level} samples, writes={decode_writes}, tokens={tokens_ingested}")


def capture_audio_input(duration_sec=AUDIO_INPUT_DURATION_SEC):
    #Capture audio from JACK input buffer
    target_samples = int(AUDIO_SAMPLE_RATE * duration_sec)
    
    with audio_input_lock:
        audio_input_buffer.clear()
    
    #print(f"[AUDIO INPUT] Capturing {duration_sec}s of audio...")
    time.sleep(duration_sec + 0.1)
    
    with audio_input_lock:
        captured = np.array(list(audio_input_buffer), dtype=np.float32)
    
    if len(captured) < target_samples:
        captured = np.pad(captured, (0, target_samples - len(captured)), mode='constant')
    elif len(captured) > target_samples:
        captured = captured[:target_samples]
    
    #print(f"[AUDIO INPUT] Captured {len(captured)} samples, min/max={captured.min():.4f}/{captured.max():.4f}")
    
    audio_tensor = torch.tensor(captured, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    return audio_tensor


def create_start_tokens_from_input():
    #Capture audio from JACK input and create start_tokens
    reals = capture_audio_input(duration_sec=AUDIO_INPUT_DURATION_SEC)
    
    with torch.no_grad():
        _, info = audio_enc.encode(reals[:, 0:1], return_info=True)
        latents_seed = info["quantizer_indices"]
        B, T, L = latents_seed.shape
        audio_tokens = flatten_and_offset_latents(latents_seed, codebook_size=2048)
        
        prefix_len = 25#60
        start_tokens = audio_tokens[:, :prefix_len]
    
    #print(f"[START TOKENS] Created start_tokens with shape {start_tokens.shape}, L={L}")
    return start_tokens, audio_tokens, latents_seed


EMG_INPUT_DURATION_SEC = AUDIO_INPUT_DURATION_SEC 
EMG_SAMPLE_RATE = 22050       

def capture_emg_from_bela(duration_sec=EMG_INPUT_DURATION_SEC):

    target_samples = int(EMG_SAMPLE_RATE * duration_sec)
 
    channels = [f"emg{i}" for i in range(1, 7)]
   
    stacked = []
    for ch in channels:
        buf = emg_buffers.get(ch, [])
        if len(buf) < target_samples:
            arr = np.pad(np.asarray(buf, dtype=np.float32), (0, target_samples - len(buf)), mode="constant")
        else:
            arr = np.asarray(buf[:target_samples], dtype=np.float32)
        stacked.append(arr)
    captured = np.stack(stacked, axis=0)  # [C, T]
    emg_tensor = torch.tensor(captured, dtype=torch.float32, device=device).unsqueeze(0)  # [B=1, C, T]
    return emg_tensor

def create_emg_start_conditioning(prefix_len=60):

    reals = capture_emg_from_bela(duration_sec=EMG_INPUT_DURATION_SEC)
    with torch.no_grad():
        reals = upsample(reals) 
        reals = reals - reals.mean(dim=-1, keepdim=True) # DC removal
        _, info = emg_enc.encode(reals, return_info=True)
        latents = info["quantizer_indices"]               # [B, T_cond, L]
        emg_tokens = flatten_and_offset_latents(latents, codebook_size=2048)
        emg_embeds = mapper.embed(emg_tokens)             # [B, T_cond, d_model]
        start_cond = emg_embeds[:, :prefix_len, :]
    return start_cond, emg_embeds, latents


# -----------------------------
# Pipeline Queues
# -----------------------------
EMG_SAMPLES_PER_ENCODE = 2048

# EMG data → conditioning embeddings
emg_queue = Queue(maxsize=64)  # Raw EMG chunks

# Conditioning embeddings → tokens
conditioning_queue = Queue(maxsize=32)  # Conditioning embeddings

# Tokens → audio samples
token_queue = Queue(maxsize=32)  # Generated tokens to decode

# Control
pipeline_active = Event()
generation_complete = Event()


# -----------------------------
# Pipeline State
# -----------------------------
class PipelineState:
    def __init__(self):
        self.lock = Lock()
        self.start_tokens = None
        self.num_quantizers = 4
        self.generated_token_count = 0
        self.target_token_count = 500
        self.all_conditioning = None  # accumulated conditioning embeddings
        self.token_block_samples = 480#2048
        
        
    def reset(self):
        with self.lock:
            self.start_tokens = None
            self.generated_token_count = 0
            self.target_token_count = 500
            self.all_conditioning = None

pipeline_state = PipelineState()


# -----------------------------
# Thread 1: EMG Encoding
# -----------------------------
def emg_encoding_thread():
    """Continuously encode EMG chunks as they arrive."""
    #print("[EMG THREAD] Started")
    
    while True:
        #b, a = design_notch(fs=1000, freq=50.0, Q=30.0, device=device)
        if not pipeline_active.is_set():
            time.sleep(0.01)
            continue
        
        try:
            # Get EMG chunk from queue
            emg_chunk = emg_queue.get(timeout=0.1)
            
            t0 = time.time()
            
            # Convert to tensor
            chunk_tensor = torch.tensor(np.array(emg_chunk), dtype=torch.float32, device=device)
            chunk_tensor = upsample(chunk_tensor.unsqueeze(0))
            #chunk_tensor = apply_iir_notch(chunk_tensor, b, a)
            chunk_tensor = chunk_tensor - chunk_tensor.mean(dim=-1, keepdim=True)
            
            # Encode
            with torch.no_grad():
                _, chunk_info = emg_enc.encode(chunk_tensor, return_info=True)
                chunk_latents = chunk_info["quantizer_indices"]
                chunk_tokens = flatten_and_offset_latents(chunk_latents, codebook_size=2048)
                print("EMG chunk_tokens:", int(chunk_tokens.min()), int(chunk_tokens.max()))

                chunk_emb = mapper.embed(chunk_tokens)
            
            # Accumulate conditioning
            with pipeline_state.lock:
                if pipeline_state.all_conditioning is None:
                    pipeline_state.all_conditioning = chunk_emb
                else:
                    pipeline_state.all_conditioning = torch.cat([pipeline_state.all_conditioning, chunk_emb], dim=1)
            
            num_cond_tokens = chunk_emb.size(1)
            
            # Send to token generation thread (one signal per conditioning token)
            for _ in range(num_cond_tokens):
                try:
                    conditioning_queue.put("generate", block=False)
                except queue.Full:
                    #print("[EMG THREAD] Conditioning queue full")
                    break
            
            t1 = time.time()
            print(f"[EMG THREAD] Encoded → {num_cond_tokens} cond tokens in {(t1-t0)*1000:.1f}ms")
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[EMG THREAD] Error: {e}")
            import traceback
            traceback.print_exc()




def token_generation_thread():
    past_kvs = None
    past_kvs_cross = None
    tokens = None
    DEBUG_KV = False#True

    while True:
        if not pipeline_active.is_set():
            time.sleep(0.01)
            past_kvs = None
            past_kvs_cross = None
            tokens = None
            continue

        try:
            got_tick = True
            try:
                _ = conditioning_queue.get(timeout=0.05)
            except queue.Empty:
                got_tick = False

            with pipeline_state.lock:
                if pipeline_state.generated_token_count >= pipeline_state.target_token_count:
                    continue

                # Initialize tokens if needed
                if tokens is None:
                    print('\n' + '='*80)
                    print('INITIALIZING TOKEN GENERATION')
                    print('='*80)
                    tokens = pipeline_state.start_tokens.clone().to(device)
                    print(f"Start tokens: shape={tokens.shape}, range=[{tokens.min().item()}, {tokens.max().item()}]")
                    
                    # cache with start tokens
                    cond_full = pipeline_state.all_conditioning
                    if cond_full is None:
                        cond_full = torch.zeros((1, tokens.shape[1], mapper.d_model), device=device)
                        print(f"WARNING: No conditioning available, using zeros")
                    else:
                        cond_full = cond_full[:, :tokens.shape[1], :]
                    
                    print(f"Initial conditioning: shape={cond_full.shape}")
                    
                    with torch.no_grad():
                        logits, past_kvs, past_kvs_cross = mapper(
                            tokens, 
                            cond_full, 
                            use_cache=True, 
                            start_pos=0,
                            debug_kv=DEBUG_KV
                        )
                    
                    print(f"Priming complete: logits.shape={logits.shape}")
                    print(f"KV cache initialized: {len(past_kvs)} layers")
                    if past_kvs[0] is not None:
                        print(f"Layer 0 KV: K.shape={past_kvs[0][0].shape}, V.shape={past_kvs[0][1].shape}")
                    print('='*80 + '\n')

                # Generate tokens
                steps = 4 if got_tick else 2
                
                for step_i in range(steps):
                    # Get conditioning
                    cond_full = pipeline_state.all_conditioning
                    
                    if cond_full is None:
                        cond_full = torch.zeros((1, tokens.shape[1], mapper.d_model), device=device, dtype=tokens.dtype)
                    elif cond_full.shape[1] < tokens.shape[1]:
                        pad_len = tokens.shape[1] - cond_full.shape[1]
                        padding = torch.zeros((1, pad_len, cond_full.shape[2]), device=device, dtype=cond_full.dtype)
                        cond_full = torch.cat([cond_full, padding], dim=1)
                    
                    if DEBUG_KV or pipeline_state.generated_token_count % 10 == 0:
                        print(f"\n[TOKEN THREAD Step {step_i}] Count={pipeline_state.generated_token_count}, tokens.shape={tokens.shape}, cond.shape={cond_full.shape}")

                    next_token, (past_kvs, past_kvs_cross) = sample_ar_w_temp_cond_kv(
                        mapper,
                        tokens,
                        cond_full=cond_full,
                        past_kvs=past_kvs,
                        past_kvs_cross=past_kvs_cross,
                        temperature=0.7,#1.2,
                        top_k=None,#50,
                        top_p=None,#0.95,
                        repetition_penalty=0.,#1.2,
                        device=device,
                        debug_kv=DEBUG_KV and pipeline_state.generated_token_count < 5  # Debug first 5 tokens
                    )

                    tokens = torch.cat([tokens, next_token], dim=1)
                    pipeline_state.generated_token_count += 1

                    try:
                        token_queue.put(next_token.clone(), block=False)
                    except queue.Full:
                        print("[TOKEN THREAD] Token queue full, dropping token")

                    if pipeline_state.generated_token_count >= pipeline_state.target_token_count:
                        print("~~~~~ Token generation complete ~~~~~")
                        generation_complete.set()
                        break

        except Exception as e:
            print(f"[TOKEN THREAD] Error: {e}")
            import traceback
            traceback.print_exc()

class ConditioningKVBuffer:

    def __init__(self, window=None):
        self.cached_len = 0
        self.past_kvs_cross = None
        self.window = window  # e.g., 4096 to cap memory, None to keep all

    def reset(self):
        self.cached_len = 0
        self.past_kvs_cross = None

    def prime(self, cond_full):
        if cond_full is None:
            self.cached_len = 0
            return None
        self.cached_len = cond_full.shape[1]
        return cond_full

    def delta(self, cond_full):
        if cond_full is None:
            return None
        total_len = cond_full.shape[1]
        if total_len <= self.cached_len:
            return cond_full[:, 0:0, :]  # empty delta
        delta = cond_full[:, self.cached_len:total_len, :]
        self.cached_len = total_len
        return delta

    def prime_tail(self, cond_full):
        # ?
        assert self.window is not None, "prime_tail requires window to be set"
        if cond_full is None:
            self.reset()
            return None
        tail = cond_full[:, -self.window:, :]
        self.reset()
        self.cached_len = tail.shape[1]
        return tail
 

# -----------------------------
# Thread 3: Audio Decoding
# -----------------------------
def audio_decoding_thread():

    print("[DECODE THREAD] Started")

    all_tokens = []
    last_decode_end_token_idx = 0
    start_tokens_processed = False
    bootstrap_done = False
    measured_samples_per_token = None

    # Tunables
    DECODE_CHUNK_TOKENS = 64#16
    DECODE_STRIDE_TOKENS = 32#8  # 50% overlap TODO: experiment
    DISCARD_SAMPLES_HEAD = 0
    DISCARD_SAMPLES_TAIL = 0
    BOOTSTRAP_TOKENS = 64#16


    STRIDE_SAMPLES = None

    window_cache = {}
    
    def get_sqrt_hann_window(n: int) -> np.ndarray:

        w = window_cache.get(n)
        if w is None:
            hann = np.hanning(n)
            w = np.sqrt(hann).astype(np.float32)
            window_cache[n] = w
        return w

    def decode_tokens_to_audio(decode_tokens: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            latents = unflatten_and_unoffset_token_by_token(
                decode_tokens,
                T=None,
                num_quantizers=pipeline_state.num_quantizers,
                codebook_size=2048
            )
            audio_chunk = audio_enc.decode_tokens(latents)
            audio = audio_chunk.squeeze().detach().cpu().numpy().astype(np.float32)
            return audio

    def ring_add(samples: np.ndarray, start_cursor: int):
        with ring_lock:
            audio_ring_buffer.add_at(samples, start_cursor)
            if np.count_nonzero(samples) > 0:
                playback_primed.set()

    write_cursor_samples = 0

    def try_prime_with_start_tokens():
        nonlocal write_cursor_samples, start_tokens_processed, measured_samples_per_token, STRIDE_SAMPLES
        if pipeline_state.start_tokens is not None and not start_tokens_processed:
            audio_np = decode_tokens_to_audio(pipeline_state.start_tokens)
            n = len(audio_np)
            
            # Measure samples per token from start_tokens
            if measured_samples_per_token is None:
                measured_samples_per_token = n / pipeline_state.start_tokens.shape[1]
                STRIDE_SAMPLES = int(DECODE_STRIDE_TOKENS * measured_samples_per_token)
                print(f"[DECODE] Measured {measured_samples_per_token:.1f} samples/token, stride={STRIDE_SAMPLES}")
            
            if n > 0:
                w = get_sqrt_hann_window(n)
                audio_np = audio_np * w
                ring_add(audio_np, write_cursor_samples)
                write_cursor_samples += STRIDE_SAMPLES
                start_tokens_processed = True
                print(f"[DECODE] Primed with start_tokens → {len(audio_np)} samples")

    while True:
        if not pipeline_active.is_set():
            time.sleep(0.01)
            all_tokens.clear()
            last_decode_end_token_idx = 0
            start_tokens_processed = False
            bootstrap_done = False
            write_cursor_samples = 0
            measured_samples_per_token = None
            STRIDE_SAMPLES = None
            continue

        try:
            token = token_queue.get(timeout=0.1)
            all_tokens.append(token)

    
            if len(all_tokens) % 8 == 0:
                print(f"[DECODE] Tokens buffered={len(all_tokens)}")

            # prime playback once
            if not playback_primed.is_set():
                try_prime_with_start_tokens()

            # bootstrap decode
            if not bootstrap_done and len(all_tokens) >= BOOTSTRAP_TOKENS:
                print('[DECODE] ===== BOOTSTRAPPING =====')
                
                if pipeline_state.start_tokens is not None and not start_tokens_processed:
                    combined_tokens = torch.cat([pipeline_state.start_tokens] + all_tokens[:BOOTSTRAP_TOKENS], dim=1)
                    audio_np = decode_tokens_to_audio(combined_tokens)
                    start_tokens_processed = True
                else:
                    tokens_to_decode = torch.cat(all_tokens[:BOOTSTRAP_TOKENS], dim=1)
                    audio_np = decode_tokens_to_audio(tokens_to_decode)

                n = len(audio_np)
                
                # Measure samples per token if not already done
                if measured_samples_per_token is None:
                    num_tokens = combined_tokens.shape[1] if pipeline_state.start_tokens is not None else BOOTSTRAP_TOKENS
                    measured_samples_per_token = n / num_tokens
                    STRIDE_SAMPLES = int(DECODE_STRIDE_TOKENS * measured_samples_per_token)
                    print(f"[DECODE] Measured {measured_samples_per_token:.1f} samples/token, stride={STRIDE_SAMPLES}")

                # Apply window
                w = get_sqrt_hann_window(n)
                windowed = audio_np * w

                ring_add(windowed, write_cursor_samples)
                write_cursor_samples += STRIDE_SAMPLES
                last_decode_end_token_idx = DECODE_STRIDE_TOKENS
                bootstrap_done = True
                print(f"[DECODE] Bootstrap complete: {n} samples, cursor now at {write_cursor_samples}")
                continue

            # Streaming with overlap-add
            if bootstrap_done and STRIDE_SAMPLES is not None:
                tokens_available = len(all_tokens)
                next_decode_end = last_decode_end_token_idx + DECODE_STRIDE_TOKENS
                need = next_decode_end + (DECODE_CHUNK_TOKENS - DECODE_STRIDE_TOKENS)
                
                if tokens_available >= need:
                    start_idx = last_decode_end_token_idx
                    end_idx = start_idx + DECODE_CHUNK_TOKENS

                    decode_tokens = torch.cat(all_tokens[start_idx:end_idx], dim=1)
                    audio_np = decode_tokens_to_audio(decode_tokens)
                    
                    # Verify expected length
                    expected_len = int(DECODE_CHUNK_TOKENS * measured_samples_per_token)
                    actual_len = len(audio_np)
                    if abs(actual_len - expected_len) > 10:
                        print(f"[DECODE] ⚠️ Length mismatch: expected {expected_len}, got {actual_len}")
                    
                    # Apply sqrt-Hann window for perfect 50% overlap
                    w = get_sqrt_hann_window(len(audio_np))
                    windowed = audio_np * w

                    ring_add(windowed, write_cursor_samples)
                    write_cursor_samples += STRIDE_SAMPLES
                    last_decode_end_token_idx = next_decode_end
                    
                    print(f"[DECODE] Stream: {len(windowed)} samples at {write_cursor_samples - STRIDE_SAMPLES}, tokens [{start_idx}:{end_idx}]")

        except queue.Empty:
            if generation_complete.is_set():
                print('[DECODE] Flushing remaining tokens')
                
                # Flush remaining tokens
                if bootstrap_done and last_decode_end_token_idx < len(all_tokens):
                    decode_tokens = torch.cat(all_tokens[last_decode_end_token_idx:], dim=1)
                    audio_np = decode_tokens_to_audio(decode_tokens)
                    
                    n = len(audio_np)
                    w = get_sqrt_hann_window(n)
                    windowed = audio_np * w
                    
                    ring_add(windowed, write_cursor_samples)
                    print(f"[DECODE] Flushed {len(windowed)} final samples")

                # add silence tail for clean shutdown
                with ring_lock:
                    silence = np.zeros(int(0.25 * AUDIO_SAMPLE_RATE), dtype=np.float32)
                    audio_ring_buffer.extend(silence)
                
                pipeline_active.clear()
                print('[DECODE] Shutdown complete')
            continue
            
        except Exception as e:
            print(f"[DECODE THREAD] Error: {e}")
            import traceback
            traceback.print_exc()


# -----------------------------
# EMG Callback (feeds pipeline)
# -----------------------------
def emg_block_callback(block, _):
    """Called for each block of EMG data. Feeds the pipeline."""
    if not pipeline_active.is_set():
        return
    
    print(f"[CALLBACK] Received block with {len(block)} buffers")
    
    # Accumulate incoming data
    for buffer in block:
        var = buffer["name"]
        data = buffer["buffer"]["data"]
        if var not in emg_buffers:
            emg_buffers[var] = []
        emg_buffers[var].extend(data)
        print(f"[CALLBACK] {var}: added {len(data)} samples, total={len(emg_buffers[var])}")
    
    # Check if we have enough for one encode
    min_samples = min(len(emg_buffers[ch]) for ch in emg_buffers)
    print(f"[CALLBACK] min_samples={min_samples}, need={EMG_SAMPLES_PER_ENCODE}")
    
    while min_samples >= EMG_SAMPLES_PER_ENCODE:
        # Extract chunk
        chunk_data = []
        for ch in [f"emg{i}" for i in range(1, 7)]:
            chunk = emg_buffers[ch][:EMG_SAMPLES_PER_ENCODE]
            chunk_data.append(np.array(chunk, dtype=np.float32))
            emg_buffers[ch] = emg_buffers[ch][EMG_SAMPLES_PER_ENCODE:]
        
        # Send to encoding thread
        try:
            emg_queue.put(chunk_data, block=False)
        except queue.Full:
            print("[CALLBACK] EMG queue full, dropping chunk")
        
        min_samples = min(len(emg_buffers[ch]) for ch in emg_buffers)

# EMG buffer (used in callback)
emg_buffers = {f"emg{i}": [] for i in range(1, 7)}


# -----------------------------
# Generation Control
# -----------------------------
# STREAM_DURATION_SEC = 5.0

def start_generation():
    """Start a generation cycle with prefill and playback gating."""
    global bela_connected, emg_buffers, stream_duration_multiplier

    STREAM_DURATION_SEC = random.randint(2, 3) * stream_duration_multiplier
    print('STREAM_DURATION_SEC', STREAM_DURATION_SEC)
    print('stream_duration_multiplier', stream_duration_multiplier)

    # --- Capture audio start tokens ---
    start_tokens, _, latents_seed = create_start_tokens_from_input()

    # --- Capture EMG start conditioning at the same time ---
    emg_prefix_len = start_tokens.shape[1]
    start_cond, all_cond_init, _ = create_emg_start_conditioning(prefix_len=emg_prefix_len)

    # --- Reset pipeline state ---
    pipeline_state.reset()
    with pipeline_state.lock:
        pipeline_state.start_tokens = start_tokens
        pipeline_state.num_quantizers = latents_seed.shape[2]
        pipeline_state.target_token_count = STREAM_DURATION_SEC * 100
        pipeline_state.all_conditioning = all_cond_init  # seed conditioning immediately

    # --- Clear queues ---
    for q in (emg_queue, conditioning_queue, token_queue):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    # --- Clear EMG buffers ---
    for key in emg_buffers:
        emg_buffers[key] = []

    # --- Clear ring buffer and prefill a small cushion ---
    with ring_lock:
        audio_ring_buffer.clear()
        silence_duration = 0.2
        silence_samples = int(AUDIO_SAMPLE_RATE * silence_duration)
        audio_ring_buffer.extend(np.zeros(silence_samples, dtype=np.float32))

    # --- Reset gating and stats ---
    playback_primed.clear()
    global ring_samples_written, decode_writes, tokens_ingested
    ring_samples_written = 0
    decode_writes = 0
    tokens_ingested = 0

    # --- Activate pipeline AFTER seeding both audio + EMG start tokens ---
    generation_complete.clear()
    pipeline_active.set()
    print("[TRIGGER] Pipeline activated")

    # --- Start Bela streaming ---
    try:
        if not bela_connected:
            bela_streamer.connect()
            bela_connected = True

        bela_streamer.start_streaming(
            variables=bela_channels,
            saving_enabled=False,
            on_block_callback=emg_block_callback,
            callback_args=(None,)
        )

        # Kick token thread with as many ticks as the EMG prefix length
        for _ in range(emg_prefix_len):
            try:
                conditioning_queue.put("generate", block=False)
            except queue.Full:
                break

        bela_streamer.wait(STREAM_DURATION_SEC)
        bela_streamer.stop_streaming()

    except Exception as e:
        print(f"[TRIGGER] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        pipeline_active.clear()
        print("[TRIGGER] Pipeline deactivated")




# -----------------------------
# Continuous Generation Loop
# -----------------------------
def continuous_generation_loop():
    """Continuously trigger generation cycles."""
    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'='*50}")
        print(f"[LOOP] Starting iteration {iteration}")
        print(f"{'='*50}\n")
        
        try:
            start_generation()
            time.sleep(0.5)
        except Exception as e:
            print(f"[LOOP] Error in iteration {iteration}: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1.0)


def jack_output_is_silent(threshold_db=-60.0):
    """Check if the last JACK output buffer was effectively silent."""
    global last_out_array
    if last_out_array.size == 0:
        return True
    rms = np.sqrt(np.mean(last_out_array**2) + 1e-12)
    rms_db = 20 * np.log10(rms)
    return rms_db < threshold_db


def jack_generation_is_silent(threshold_db=-60.0, window_samples=1024):
    """Check if generation has stopped feeding audio and buffer contains only silence."""
    with ring_lock:
        if len(audio_ring_buffer) == 0:
            return True  # nothing queued at all

        # Peek at next window of samples
        n = min(window_samples, len(audio_ring_buffer))
        recent = audio_ring_buffer.peek(n)  # requires peek() method
    rms = np.sqrt(np.mean(recent**2) + 1e-12)
    rms_db = 20 * np.log10(rms)
    return rms_db < threshold_db

def jack_silence(mode="output", threshold_db=-60.0):

    if mode == "output":
        return jack_output_is_silent(threshold_db)
    elif mode == "generation":
        return jack_generation_is_silent(threshold_db)
    elif mode == "combined":
        return jack_generation_is_silent(threshold_db) and jack_output_is_silent(threshold_db)
    else:
        raise ValueError(f"Unknown silence mode: {mode}")







# -----------------------------
# Main
# -----------------------------
def main():
    print("[MAIN] Starting main function...")

    global phrase_detector

    # Activate JACK
    try:
        client.activate()
        print("JACK client activated")
    except Exception as e:
        print("JACK activation failed:", e)
        return

    # Connect input
    try:
        client.connect(JACK_INPUT_SOURCE, f"{JACK_CLIENT_NAME}:input_1")
        print(f"Connected {JACK_INPUT_SOURCE} -> {JACK_CLIENT_NAME}:input_1")
    except jack.JackError as e:
        print("Input connection failed:", e)

    print("JACK block size:", client.blocksize)
    print("JACK sample rate:", client.samplerate)

    # Connect output
    try:
        client.connect(f"{JACK_CLIENT_NAME}:output_1", JACK_TARGET_PORT)
        print(f"Connected {JACK_CLIENT_NAME}:output_1 -> {JACK_TARGET_PORT}")
    except jack.JackError as e:
        print("Output connection failed:", e)

    #calibrated_db = calibrate_noise_floor(duration_sec=5.0, margin_db=10.0)
    calibrated_db = calibrate_noise_floor(mode="playback") # ambient playback

    

    phrase_detector = PhraseStartDetector(
    samplerate=AUDIO_SAMPLE_RATE,
    hop_size=512,
    silence_db=calibrated_db,
    min_silence_ms=300,
    callback_phrase_start=gated_callback
    )

    osc_dispatcher = dispatcher.Dispatcher()
    osc_dispatcher.map("/stream/multiplier", osc_set_multiplier)
    server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 5005), osc_dispatcher)
    print("[OSC] Serving on 127.0.0.1:5005")

    t_osc = threading.Thread(target=server.serve_forever,    name="OSC",    daemon=True)
    t_emg   = threading.Thread(target=emg_encoding_thread,    name="EMG",    daemon=True)
    t_token = threading.Thread(target=token_generation_thread, name="TOKEN", daemon=True)
    t_decode= threading.Thread(target=audio_decoding_thread,  name="DECODE", daemon=True)
    t_ring  = threading.Thread(target=ring_monitor_thread,     name="RING",  daemon=True)

    t_osc.start();  print("[MAIN] OSC started, alive?", t_osc.is_alive())
    t_emg.start();   print("[MAIN] EMG started, alive?", t_emg.is_alive())
    t_token.start(); print("[MAIN] TOKEN started, alive?", t_token.is_alive())
    t_decode.start();print("[MAIN] DECODE started, alive?", t_decode.is_alive())
    t_ring.start();  print("[MAIN] RING started, alive?", t_ring.is_alive())
    

    print("Pipeline running... Ctrl+C to stop")
    try:
        while True:
            # Block until phrase detector fires
            pipeline_active.wait()   # waits until set()
            pipeline_active.clear()  # reset for next phrase
            print("[MAIN] Trigger received → starting generation")
            start_generation()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        try:
            if bela_connected:
                bela_streamer.stop_streaming()
        except:
            pass
        client.deactivate()
        client.close()
        print("Shutdown complete.")


if __name__ == "__main__":
    print("[SCRIPT] Running main...")
    main()
    print("[SCRIPT] Main completed")
