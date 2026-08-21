"""
Standalone Raspberry Pi inference script for the statically quantized
keyword-spotting ONNX model. Deliberately self-contained: only `numpy` and
`onnxruntime` are required -- no torch/torchaudio, so it's a light install on
a Pi. Copy this file + `keyword_spotting_int8.onnx` to the device and run:

    pip install onnxruntime numpy
    python run_onnx_pi.py --wav some_clip.wav
    python run_onnx_pi.py --wav some_clip.wav --onnx /path/to/keyword_spotting_int8.onnx

The mel-spectrogram math here reimplements torchaudio's MelSpectrogram
(center=True, htk mel scale, no filterbank-area normalization) in plain
NumPy so it matches what the model was actually trained on -- this is
verified numerically against real torchaudio output before shipping (see
README.md); a mismatched preprocessing path would silently hurt accuracy
even though the code runs without error.
"""

import argparse
import time
import wave

import numpy as np
import onnxruntime as ort

# ── Config (mirrors kws_common_torch.py -- duplicated here so this file has
#    zero project-module dependencies) ──────────────────────────────────────

SAMPLE_RATE = 16_000
N_MELS = 80
N_FFT = 512
WIN_LENGTH = 400
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


# ── Log-mel spectrogram (pure NumPy reimplementation of torchaudio's
#    MelSpectrogram: center=True/reflect-pad, htk mel scale, norm=None) ─────

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(sample_rate=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS,
                     f_min=0.0, f_max=None) -> np.ndarray:
    """[n_fft//2+1, n_mels] triangular filterbank, matching torchaudio's
    melscale_fbanks(..., norm=None, mel_scale='htk')."""
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

    # center=True: reflect-pad by n_fft//2 each side, like torch.stft.
    pad = N_FFT // 2
    padded = np.pad(waveform, (pad, pad), mode="reflect")

    # Periodic Hann window of length WIN_LENGTH, zero-padded/centered inside
    # an N_FFT-length buffer -- matches torch.stft(win_length < n_fft).
    hann = np.hanning(WIN_LENGTH + 1)[:-1].astype(np.float32)  # periodic, not symmetric
    window = np.zeros(N_FFT, dtype=np.float32)
    offset = (N_FFT - WIN_LENGTH) // 2
    window[offset:offset + WIN_LENGTH] = hann

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


# ── ONNX Runtime inference ───────────────────────────────────────────────────

def load_session(onnx_path: str) -> ort.InferenceSession:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_info = sess.get_inputs()[0]
    out_info = sess.get_outputs()[0]
    print(f"Input : {in_info.name} {in_info.shape} {in_info.type}")
    print(f"Output: {out_info.name} {out_info.shape} {out_info.type}")
    return sess


def predict(sess: ort.InferenceSession, mel: np.ndarray):
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    x = mel[np.newaxis, ...].astype(np.float32)  # [1, MAX_FRAMES, N_MELS]

    t0 = time.perf_counter()
    (logits,) = sess.run([output_name], {input_name: x})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    logits = logits[0]
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return probs, latency_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="keyword_spotting_int8.onnx")
    parser.add_argument("--wav", required=True)
    args = parser.parse_args()

    sess = load_session(args.onnx)
    mel = load_wav_as_mel(args.wav)
    probs, latency_ms = predict(sess, mel)
    top5 = np.argsort(probs)[::-1][:5]

    print(f"\n{args.wav}")
    print(f"Latency: {latency_ms:.2f} ms")
    print("Top-5 predictions:")
    for idx in top5:
        print(f"  {KEYWORDS[idx]:<10s} {probs[idx]:.4f}")


if __name__ == "__main__":
    main()
