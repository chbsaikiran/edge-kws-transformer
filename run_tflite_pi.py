"""
Standalone Raspberry Pi inference script for the statically quantized
keyword-spotting TFLite model. Deliberately self-contained: only `numpy` and
a lightweight TFLite interpreter package are required -- no `tensorflow`, so
it's a light install on a Pi (mirrors run_onnx_pi.py's role for the ONNX
side). Copy this file + `keyword_spotting_int8.tflite` to the device and run:

    pip install ai-edge-litert numpy    # or: pip install tflite-runtime numpy
    python run_tflite_pi.py --wav some_clip.wav
    python run_tflite_pi.py --wav some_clip.wav --tflite /path/to/keyword_spotting_int8.tflite

IMPORTANT: the mel-spectrogram math here reimplements kws_common.py's TF-side
preprocessing specifically -- NOT the torchaudio-based math in run_onnx_pi.py.
The two pipelines deliberately use different windowing conventions (see
kws_common.py's docstring: a full N_FFT-length Hann window, rather than
torchaudio's WIN_LENGTH-length window zero-padded/centered inside N_FFT), so
reusing run_onnx_pi.py's preprocessing here would silently mismatch what this
TF-trained model actually saw during training. Verified numerically against
real kws_common.py output before shipping (see README.md); a mismatched
preprocessing path would silently hurt accuracy even though the code runs
without error.
"""

import argparse
import time
import wave

import numpy as np

# --- Raspberry Pi: use ai-edge-litert, Google's actively-maintained
#     replacement for tflite_runtime (confirmed: `pip install tflite-runtime`
#     has no wheel at all for several current platform/Python combos --
#     ai-edge-litert is the one verified installable and working here).
from ai_edge_litert.interpreter import Interpreter, load_delegate
# --- Alternative: comment the line above and uncomment these instead, if
#     your device already has tflite_runtime and you'd rather not switch:
# import tflite_runtime.interpreter as tflite
# Interpreter = tflite.Interpreter
# load_delegate = tflite.load_delegate

# ── Config (mirrors kws_common.py -- duplicated here so this file has zero
#    project-module dependencies) ────────────────────────────────────────────

SAMPLE_RATE = 16_000
N_MELS = 80
N_FFT = 512
HOP_LENGTH = 160
MAX_FRAMES = 101

KEYWORDS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila",
    "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero",
]


# ── WAV I/O (stdlib `wave`, no torchaudio/soundfile needed) ─────────────────

def read_wav(path: str) -> np.ndarray:
    """Reads a 16-bit PCM mono WAV file (Speech Commands' format) into a
    float32 waveform in [-1, 1]. Raises if the file isn't 16-bit PCM."""
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError(f"{path}: expected 16-bit PCM WAV, got sampwidth={sampwidth}")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)  # downmix to mono

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"{path}: sample rate {sr} != {SAMPLE_RATE}. Resample before "
            f"running (e.g. `ffmpeg -i in.wav -ar 16000 -ac 1 out.wav`) -- "
            f"kept dependency-free here rather than pulling in a resampler."
        )
    return audio


# ── Log-mel spectrogram (pure NumPy reimplementation of kws_common.py's TF
#    pipeline: reflect-pad center=True, htk mel scale, N_FFT-length window) ──

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(sample_rate=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
                     f_min=0.0, f_max=None) -> np.ndarray:
    """[n_fft//2+1, n_mels] triangular filterbank, matching
    tf.signal.linear_to_mel_weight_matrix's htk-scale, unnormalized filters."""
    if f_max is None:
        f_max = sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    all_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)

    m_pts = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    f_pts = _mel_to_hz(m_pts)                       # [n_mels+2]
    f_diff = np.diff(f_pts)                          # [n_mels+1]

    slopes = f_pts[np.newaxis, :] - all_freqs[:, np.newaxis]  # [n_freqs, n_mels+2]
    down_slopes = -slopes[:, :-2] / f_diff[:-1]        # [n_freqs, n_mels]
    up_slopes = slopes[:, 2:] / f_diff[1:]             # [n_freqs, n_mels]
    fb = np.maximum(0.0, np.minimum(down_slopes, up_slopes))
    return fb.astype(np.float32)


_MEL_FB = _mel_filterbank()  # built once at import time


def _pad_or_trim(waveform: np.ndarray) -> np.ndarray:
    if len(waveform) < SAMPLE_RATE:
        return np.pad(waveform, (0, SAMPLE_RATE - len(waveform)))
    return waveform[:SAMPLE_RATE]


def log_mel_spectrogram(waveform: np.ndarray) -> np.ndarray:
    """[SAMPLE_RATE] float32 waveform -> [MAX_FRAMES, N_MELS] log-mel spectrogram."""
    waveform = _pad_or_trim(waveform).astype(np.float32)

    # center=True: reflect-pad by n_fft//2 each side, like tf.signal.stft's
    # equivalent framing convention.
    pad = N_FFT // 2
    padded = np.pad(waveform, (pad, pad), mode="reflect")

    # kws_common.py uses a full N_FFT-length Hann window directly (NOT a
    # shorter WIN_LENGTH window zero-padded/centered inside N_FFT, unlike
    # run_onnx_pi.py's torchaudio-matching version) -- see that file's
    # docstring for why. Must match here, not the ONNX side's convention.
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)  # periodic Hann, length N_FFT

    n_frames = 1 + (len(padded) - N_FFT) // HOP_LENGTH  # == MAX_FRAMES == 101
    frames = np.stack([
        padded[i * HOP_LENGTH: i * HOP_LENGTH + N_FFT] * window
        for i in range(n_frames)
    ])                                                    # [n_frames, N_FFT]

    spectrum = np.fft.rfft(frames, axis=-1)                # [n_frames, N_FFT//2+1]
    power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)

    mel = power @ _MEL_FB                                  # [n_frames, N_MELS]
    mel = np.log(mel + 1e-6)
    mel = (mel - mel.mean()) / (mel.std() + 1e-5)
    return mel.astype(np.float32)


def load_wav_as_mel(path: str) -> np.ndarray:
    return log_mel_spectrogram(read_wav(path))


# ── Delegation: DSP/NPU -> GPU -> CPU (same pattern as run_tflite_mac.py,
#    but with the Coral Edge TPU as the realistic RPi accelerator -- USB/PCIe
#    Coral accelerators are the common real-world TFLite+RPi accelerator
#    story, unlike a Hexagon DSP or a mobile GPU delegate, which generally
#    aren't available on this hardware). Each candidate's shared library is
#    attempted and caught individually, same as the Mac script, since
#    tflite_runtime's Python API has no built-in provider-list fallback the
#    way ONNX Runtime does. ───────────────────────────────────────────────

_DELEGATE_CANDIDATES = [
    ("Coral Edge TPU", "libedgetpu.so.1", {}),
    ("DSP (Hexagon)", "libhexagon_delegate.so", {}),
    ("GPU", "libtensorflowlite_gpu_delegate.so", {}),
]


def load_delegates():
    """Try Edge TPU, DSP, then GPU, returning the first that loads. Falls
    back to no explicit delegate (TFLite's default XNNPACK CPU path) if
    none do -- expected unless a Coral accelerator is actually plugged in."""
    for name, lib, options in _DELEGATE_CANDIDATES:
        try:
            delegate = load_delegate(lib, options)
            print(f"Delegate: {name} ({lib})")
            return [delegate], name
        except (ValueError, OSError) as e:
            print(f"Delegate unavailable: {name} ({lib}) -- {e.__class__.__name__}")
    print("Delegate: CPU (XNNPACK, default -- no accelerator delegate available)")
    return [], "CPU"


# ── TFLite inference ──────────────────────────────────────────────────────

def quantize(x_float, quant_params, dtype):
    scale, zero_point = quant_params
    if dtype in (np.int8, np.uint8) and scale != 0:
        q = np.round(x_float / scale + zero_point)
        info = np.iinfo(dtype)
        q = np.clip(q, info.min, info.max)
        return q.astype(dtype)
    return x_float.astype(dtype)


def dequantize(x_q, quant_params, dtype):
    scale, zero_point = quant_params
    if dtype in (np.int8, np.uint8) and scale != 0:
        return (x_q.astype(np.float32) - zero_point) * scale
    return x_q.astype(np.float32)


def load_interpreter(tflite_path):
    delegates, active = load_delegates()
    interp = Interpreter(model_path=tflite_path, experimental_delegates=delegates)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    print("Input :", in_det["shape"].tolist(), in_det["dtype"], "quant(scale,zp)=", in_det["quantization"])
    print("Output:", out_det["shape"].tolist(), out_det["dtype"], "quant(scale,zp)=", out_det["quantization"])
    return interp, in_det, out_det


def predict(interp, in_det, out_det, mel: np.ndarray):
    x = mel[np.newaxis, ...]
    x_q = quantize(x, in_det["quantization"], in_det["dtype"])
    interp.set_tensor(in_det["index"], x_q)

    t0 = time.perf_counter()
    interp.invoke()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    out_q = interp.get_tensor(out_det["index"])
    logits = dequantize(out_q, out_det["quantization"], out_det["dtype"])[0]

    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return probs, latency_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite", default="keyword_spotting_int8.tflite")
    parser.add_argument("--wav", required=True)
    args = parser.parse_args()

    interp, in_det, out_det = load_interpreter(args.tflite)
    mel = load_wav_as_mel(args.wav)
    probs, latency_ms = predict(interp, in_det, out_det, mel)
    top5 = np.argsort(probs)[::-1][:5]

    print(f"\n{args.wav}")
    print(f"Latency: {latency_ms:.2f} ms")
    print("Top-5 predictions:")
    for idx in top5:
        print(f"  {KEYWORDS[idx]:<10s} {probs[idx]:.4f}")


if __name__ == "__main__":
    main()
