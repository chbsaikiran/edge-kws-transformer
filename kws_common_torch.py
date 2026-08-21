"""
Shared config, dataset, and model definition for the PyTorch keyword spotter.

Imported by train.py, convert_to_onnx.py, and run_onnx_mac.py so all three
stages (train / quantize / verify) use exactly the same preprocessing and
architecture -- mirrors how kws_common.py backs the TensorFlow pipeline.

This is the original decoder-only Transformer architecture (unchanged), just
factored out of the monolithic train.py so it can be reused.

Dataset download/listing intentionally does NOT use
torchaudio.datasets.SPEECHCOMMANDS: current torchaudio routes all file
loading through TorchCodec, which hard-requires a system FFmpeg install to
even import (no fallback to soundfile/sox any more). Since we only ever deal
with plain 16-bit PCM WAV files, download+split logic is reused from
kws_common.py (the TF side, which has the same requirement for a different
reason) and files are read with the stdlib `wave` module instead --
torchaudio is only used here for its MelSpectrogram/FrequencyMasking/
TimeMasking *transforms*, which operate on tensors already in memory and
never touch TorchCodec.
"""

import math
import tarfile
import urllib.request
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking

SPEECH_COMMANDS_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_RATE  = 16_000
N_MELS       = 80
N_FFT        = 512
WIN_LENGTH   = 400     # 25 ms
HOP_LENGTH   = 160     # 10 ms
MAX_FRAMES   = 101     # frames for 1-second clip

D_MODEL      = 256
N_HEADS      = 4
N_LAYERS     = 4
D_FF         = 512
DROPOUT      = 0.1

BATCH_SIZE   = 256
LR           = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 40
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT    = "./data"
CKPT_DIR     = "./checkpoints"
CKPT_PATH    = f"{CKPT_DIR}/best_model.pt"

KEYWORDS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila",
    "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero",
]
LABEL2IDX  = {w: i for i, w in enumerate(KEYWORDS)}
NUM_CLASSES = len(KEYWORDS)  # 35


# ── Dataset download (Google Speech Commands v0.02) ─────────────────────────────
# Same download/split logic as kws_common.py (the TF side), so both frameworks
# reuse one ./data download and stay split-for-split identical.

def _ensure_downloaded(root: str) -> Path:
    base_dir = Path(root) / "SpeechCommands" / "speech_commands_v0.02"
    marker = base_dir / "validation_list.txt"
    if marker.exists():
        return base_dir

    base_dir.mkdir(parents=True, exist_ok=True)
    archive_path = Path(root) / "speech_commands_v0.02.tar.gz"
    if not archive_path.exists():
        print(f"Downloading {SPEECH_COMMANDS_URL} (~2.4 GB, first run only)...")
        urllib.request.urlretrieve(SPEECH_COMMANDS_URL, archive_path)

    print("Extracting archive...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(base_dir)

    return base_dir


def _load_list(base_dir: Path, filename: str) -> set:
    with open(base_dir / filename) as f:
        return {line.strip() for line in f if line.strip()}


def list_split(root: str, subset: str):
    """Return [(wav_path, label_idx), ...] for subset in {"training","validation","testing"}."""
    base_dir = _ensure_downloaded(root)
    testing = _load_list(base_dir, "testing_list.txt")
    validation = _load_list(base_dir, "validation_list.txt")

    pairs = []
    for label_dir in sorted(base_dir.iterdir()):
        if not label_dir.is_dir() or label_dir.name not in LABEL2IDX:
            continue  # skips _background_noise_ and any non-keyword folder
        label = label_dir.name
        for wav_path in sorted(label_dir.glob("*.wav")):
            rel = f"{label}/{wav_path.name}"
            if subset == "testing":
                keep = rel in testing
            elif subset == "validation":
                keep = rel in validation
            else:  # "training"
                keep = rel not in testing and rel not in validation
            if keep:
                pairs.append((str(wav_path), LABEL2IDX[label]))
    return pairs


# ── WAV I/O ───────────────────────────────────────────────────────────────────
# Plain stdlib `wave` reader for the 16-bit PCM mono WAVs Speech Commands
# ships -- see the module docstring for why this doesn't use torchaudio.load.

def read_wav(path: str) -> torch.Tensor:
    """Returns a [1, num_samples] float32 waveform in [-1, 1], like torchaudio.load."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(f"{path}: expected 16-bit PCM WAV, got sampwidth={sampwidth}")
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: sample rate {sr} != {SAMPLE_RATE}")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    return torch.from_numpy(audio).unsqueeze(0)  # [1, num_samples]


# ── Dataset ───────────────────────────────────────────────────────────────────

class SpeechCommandsDataset(Dataset):
    def __init__(self, root: str, subset: str, augment: bool = False):
        self._pairs = list_split(root, subset)
        self.augment = augment
        self.mel_transform = MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT,
            win_length=WIN_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS,
        )
        self.freq_mask = FrequencyMasking(freq_mask_param=10)
        self.time_mask = TimeMasking(time_mask_param=20)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int):
        path, label_idx = self._pairs[idx]
        waveform = read_wav(path)

        target_len = SAMPLE_RATE
        if waveform.shape[-1] < target_len:
            waveform = F.pad(waveform, (0, target_len - waveform.shape[-1]))
        else:
            waveform = waveform[..., :target_len]

        mel = self.mel_transform(waveform)     # [1, n_mels, T]
        mel = (mel + 1e-6).log()
        mel = (mel - mel.mean()) / (mel.std() + 1e-5)

        if self.augment:
            mel = self.freq_mask(mel)
            mel = self.time_mask(mel)

        mel = mel.squeeze(0).T                 # [T, n_mels]
        return mel, label_idx


def collate_fn(batch):
    mels, labels = zip(*batch)
    return torch.stack(mels), torch.tensor(labels, dtype=torch.long)


def load_wav_as_mel(path: str, augment: bool = False) -> torch.Tensor:
    """Single-file preprocessing path used by run_onnx_mac.py's --wav mode."""
    waveform = read_wav(path)

    target_len = SAMPLE_RATE
    if waveform.shape[-1] < target_len:
        waveform = F.pad(waveform, (0, target_len - waveform.shape[-1]))
    else:
        waveform = waveform[..., :target_len]

    mel_transform = MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT,
        win_length=WIN_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    mel = mel_transform(waveform)
    mel = (mel + 1e-6).log()
    mel = (mel - mel.mean()) / (mel.std() + 1e-5)
    if augment:
        mel = FrequencyMasking(freq_mask_param=10)(mel)
        mel = TimeMasking(time_mask_param=20)(mel)
    return mel.squeeze(0).T   # [T, n_mels]


# ── Model ─────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = math.sqrt(self.head_dim)
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)     # [3, B, H, T, D]
        q, k, v = qkv.unbind(0)               # each [B, H, T, D]

        attn = (q @ k.transpose(-2, -1)) / self.scale   # [B, H, T, T]
        causal = torch.triu(
            torch.full((T, T), -1e4, device=x.device), diagonal=1
        )
        attn = attn + causal
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # net.0 and net.3 are the two Linear layers targeted by SmoothQuant/AWQ
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class KeywordSpottingTransformer(nn.Module):
    """
    Decoder-only transformer for keyword spotting.
    Input:  [B, T, n_mels]  log-mel spectrogram frames
    Output: [B, num_classes] logits

    A learnable [CLS] token is appended at the END of the sequence.
    In causal attention the last position attends to all previous frames,
    making it a natural pooling point for classification.
    """

    def __init__(
        self,
        n_mels:      int = N_MELS,
        d_model:     int = D_MODEL,
        n_heads:     int = N_HEADS,
        n_layers:    int = N_LAYERS,
        d_ff:        int = D_FF,
        dropout:     float = DROPOUT,
        num_classes: int = NUM_CLASSES,
        max_len:     int = MAX_FRAMES,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_mels, d_model)
        self.pos_emb    = nn.Embedding(max_len + 1, d_model)  # +1 for CLS
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks     = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.cls_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x   = self.input_proj(x)                          # [B, T, D]
        cls = self.cls_token.expand(B, -1, -1)            # [B, 1, D]
        x   = torch.cat([x, cls], dim=1)                  # [B, T+1, D]
        pos = torch.arange(T + 1, device=x.device)
        x   = x + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x   = self.norm(x)
        return self.head(x[:, -1, :])                     # CLS token output
