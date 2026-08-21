"""
Shared config, dataset pipeline, and model definition for the TensorFlow
port of the decoder-only Transformer keyword spotter.

Imported by train_tf.py, convert_to_tflite.py, and run_tflite_mac.py so all
three stages (train / quantize / verify) use *exactly* the same preprocessing
and architecture.
"""

import math
import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# ── Config ────────────────────────────────────────────────────────────────────
# Identical values to the PyTorch train.py so checkpoints/results are comparable.

SAMPLE_RATE = 16_000
N_MELS = 80
N_FFT = 512
WIN_LENGTH = 400  # 25 ms (kept for reference; see note in _log_mel_spectrogram)
HOP_LENGTH = 160  # 10 ms
MAX_FRAMES = 101  # frames for a 1-second clip

D_MODEL = 256
N_HEADS = 4
N_LAYERS = 4
D_FF = 512
DROPOUT = 0.1

BATCH_SIZE = 256
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 40
DATA_ROOT = "./data"
CKPT_DIR = "./checkpoints_tf"
WEIGHTS_PATH = f"{CKPT_DIR}/best_model.weights.h5"
CONFIG_PATH = f"{CKPT_DIR}/best_model_config.json"

KEYWORDS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila",
    "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero",
]
LABEL2IDX = {w: i for i, w in enumerate(KEYWORDS)}
NUM_CLASSES = len(KEYWORDS)  # 35

SPEECH_COMMANDS_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


# ── Dataset download (Google Speech Commands v0.02) ─────────────────────────────
# Uses the same on-disk layout torchaudio's SPEECHCOMMANDS dataset uses
# (<root>/SpeechCommands/speech_commands_v0.02/...), so if you already
# downloaded the data for the PyTorch script, this will reuse it instead of
# downloading ~2.4 GB again.

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
    path = base_dir / filename
    with open(path) as f:
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


# ── Preprocessing: waveform -> log-mel spectrogram ─────────────────────────────

def _load_wav(path):
    audio_binary = tf.io.read_file(path)
    wav, _sr = tf.audio.decode_wav(audio_binary, desired_channels=1)
    return tf.squeeze(wav, axis=-1)  # [samples]  (source files are already 16 kHz)


def _pad_or_trim(wav):
    length = tf.shape(wav)[0]
    wav = tf.cond(
        length < SAMPLE_RATE,
        lambda: tf.pad(wav, [[0, SAMPLE_RATE - length]]),
        lambda: wav[:SAMPLE_RATE],
    )
    wav.set_shape([SAMPLE_RATE])
    return wav


def _log_mel_spectrogram(wav):
    # torchaudio's MelSpectrogram (center=True) pads the waveform by n_fft//2
    # on each side and frames it with the FFT window itself; we replicate that
    # centering with a reflect-pad so the frame count lands exactly on
    # MAX_FRAMES=101 for a 1s clip. We use a Hann window of length N_FFT
    # (rather than zero-padding a shorter WIN_LENGTH window inside N_FFT, as
    # torch does) -- numerically slightly different, functionally equivalent.
    pad = N_FFT // 2
    padded = tf.pad(wav, [[pad, pad]], mode="REFLECT")
    stft = tf.signal.stft(
        padded, frame_length=N_FFT, frame_step=HOP_LENGTH, fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
    )
    power = tf.math.real(stft * tf.math.conj(stft))  # |STFT|^2 -> [T, N_FFT//2+1]
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=N_MELS, num_spectrogram_bins=N_FFT // 2 + 1,
        sample_rate=SAMPLE_RATE, lower_edge_hertz=0.0, upper_edge_hertz=SAMPLE_RATE / 2,
    )
    mel = tf.matmul(power, mel_weights)  # [T, n_mels]
    mel = tf.math.log(mel + 1e-6)
    mean = tf.reduce_mean(mel)
    std = tf.math.reduce_std(mel)
    mel = (mel - mean) / (std + 1e-5)
    mel.set_shape([MAX_FRAMES, N_MELS])
    return mel


def _freq_mask(mel, param=10):
    F = tf.shape(mel)[1]
    f = tf.random.uniform((), 0, param + 1, dtype=tf.int32)
    f0 = tf.random.uniform((), 0, tf.maximum(F - f, 1), dtype=tf.int32)
    idx = tf.range(F)
    mask = tf.cast(tf.logical_or(idx < f0, idx >= f0 + f), mel.dtype)
    return mel * mask[tf.newaxis, :]


def _time_mask(mel, param=20):
    T = tf.shape(mel)[0]
    t = tf.random.uniform((), 0, param + 1, dtype=tf.int32)
    t0 = tf.random.uniform((), 0, tf.maximum(T - t, 1), dtype=tf.int32)
    idx = tf.range(T)
    mask = tf.cast(tf.logical_or(idx < t0, idx >= t0 + t), mel.dtype)
    return mel * mask[:, tf.newaxis]


def load_and_process(path, augment: bool):
    wav = _load_wav(path)
    wav = _pad_or_trim(wav)
    mel = _log_mel_spectrogram(wav)
    if augment:
        mel = _freq_mask(mel, 10)
        mel = _time_mask(mel, 20)
    return mel


def make_dataset(pairs, batch_size: int, shuffle: bool, augment: bool):
    paths = [p for p, _ in pairs]
    labels = [l for _, l in pairs]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    def _map(path, label):
        return load_and_process(path, augment), label

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ── Model ─────────────────────────────────────────────────────────────────────

_TRUNC_NORMAL = tf.keras.initializers.TruncatedNormal(stddev=0.02)


class CausalSelfAttention(layers.Layer):
    def __init__(self, d_model, n_heads, dropout, seq_len, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)
        self.qkv = layers.Dense(3 * d_model, use_bias=True, kernel_initializer=_TRUNC_NORMAL)
        self.out_proj = layers.Dense(d_model, use_bias=True, kernel_initializer=_TRUNC_NORMAL)
        self.attn_drop = layers.Dropout(dropout)
        # Mask magnitude matters for int8 static quantization even though it's a
        # no-op for float32: the ADD op combining this mask with the real attention
        # logits gets a SINGLE int8 quantization range spanning both. With -1e4,
        # that range is ~[-1e4, +27] -- ~39 units/level, i.e. the real logit spread
        # (empirically ~[-27, +27] on the trained model) collapses into roughly one
        # quantization level, destroying attention resolution (confirmed: this
        # alone took validation accuracy from 0.95 float32 to 0.26 int8). -100 is
        # still >>3x past the observed logit extremes (still ~0 softmax weight for
        # masked positions) while keeping the ADD's int8 range tight (~[-127, +27],
        # ~0.6 units/level -- roughly 65x better precision).
        causal = np.triu(np.full((seq_len, seq_len), -100.0, dtype=np.float32), k=1)
        self.causal_bias = tf.constant(causal[np.newaxis, np.newaxis, :, :])  # [1,1,T,T]

    def call(self, x, training=False):
        B = tf.shape(x)[0]
        T = x.shape[1]  # static (fixed sequence length)
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B, T, 3, self.n_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # [3,B,H,T,Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = tf.matmul(q, k, transpose_b=True) / self.scale  # [B,H,T,T]
        attn = attn + self.causal_bias
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        out = tf.matmul(attn, v)                       # [B,H,T,Dh]
        out = tf.transpose(out, [0, 2, 1, 3])           # [B,T,H,Dh]
        out = tf.reshape(out, [B, T, self.n_heads * self.head_dim])
        return self.out_proj(out)


class FeedForward(layers.Layer):
    def __init__(self, d_model, d_ff, dropout, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(d_ff, kernel_initializer=_TRUNC_NORMAL)
        self.act = layers.Activation(tf.keras.activations.gelu)  # exact (erf) GELU, matches nn.GELU()
        self.drop1 = layers.Dropout(dropout)
        self.fc2 = layers.Dense(d_model, kernel_initializer=_TRUNC_NORMAL)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x


class DecoderBlock(layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, seq_len)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def call(self, x, training=False):
        x = x + self.attn(self.norm1(x), training=training)
        x = x + self.ff(self.norm2(x), training=training)
        return x


class ClsPosEmbed(layers.Layer):
    """Appends a learnable [CLS] token and adds learned positional embeddings."""

    def __init__(self, d_model, max_len, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        self.pos_emb = layers.Embedding(max_len + 1, d_model, embeddings_initializer=_TRUNC_NORMAL)

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token", shape=(1, 1, self.d_model), initializer="zeros", trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        B = tf.shape(x)[0]
        T = x.shape[1]
        cls = tf.tile(self.cls_token, [B, 1, 1])
        x = tf.concat([x, cls], axis=1)          # [B, T+1, D]
        pos_ids = tf.range(T + 1)
        x = x + self.pos_emb(pos_ids)
        return x


def build_model(
    n_mels: int = N_MELS,
    d_model: int = D_MODEL,
    n_heads: int = N_HEADS,
    n_layers: int = N_LAYERS,
    d_ff: int = D_FF,
    dropout: float = DROPOUT,
    num_classes: int = NUM_CLASSES,
    max_len: int = MAX_FRAMES,
    batch_size=None,
) -> tf.keras.Model:
    """Functional-API model so it traces cleanly for SavedModel/TFLite export.

    `batch_size` should be left None for training (dynamic batches) and set
    to a fixed int (e.g. 1) when building the model for TFLite conversion.
    """
    inputs = layers.Input(shape=(max_len, n_mels), batch_size=batch_size, name="mel_input")
    x = layers.Dense(d_model, kernel_initializer=_TRUNC_NORMAL, name="input_proj")(inputs)
    x = ClsPosEmbed(d_model, max_len, name="cls_pos_embed")(x)
    seq_len = max_len + 1
    for i in range(n_layers):
        x = DecoderBlock(d_model, n_heads, d_ff, dropout, seq_len, name=f"decoder_block_{i}")(x)
    x = layers.LayerNormalization(epsilon=1e-5, name="final_norm")(x)
    cls_out = layers.Lambda(lambda t: t[:, -1, :], name="take_cls")(x)
    logits = layers.Dense(num_classes, kernel_initializer=_TRUNC_NORMAL, name="head")(cls_out)
    return tf.keras.Model(inputs, logits, name="KeywordSpottingTransformer")
