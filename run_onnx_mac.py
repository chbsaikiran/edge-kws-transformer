"""
Run and verify the statically quantized ONNX keyword-spotting model on Mac,
using ONNX Runtime, before taking it to the Raspberry Pi.

Two modes:
  1. Full validation-set sweep -> accuracy + average/p95 latency
        python run_onnx_mac.py
  2. Single WAV file smoke test -> top-5 predicted keywords
        python run_onnx_mac.py --wav path/to/clip.wav

This script deliberately reuses kws_common_torch.py (torch/torchaudio) for
preprocessing, since it's meant to run on Mac where you already have those
installed for training -- the full validation-set sweep needs the labeled
dataset anyway. For the Raspberry Pi, use run_onnx_pi.py instead: it's a
separate, self-contained script with its own NumPy-only mel-spectrogram
implementation and no torch/torchaudio dependency at all -- only
`onnxruntime` + `numpy`, which is what you actually want on a Pi. The
inference call itself (`onnxruntime.InferenceSession` + `.run(...)`) is
identical in both scripts; only the preprocessing path differs.
"""

import argparse
import time

import numpy as np
import onnxruntime as ort

import kws_common_torch as kws
from kws_common_torch import SpeechCommandsDataset, load_wav_as_mel

# Delegation priority: DSP -> GPU -> CPU. Unlike TFLite's Python API (which
# has no built-in fallback and requires catching a load failure per
# delegate), ONNX Runtime does this natively: InferenceSession tries
# `providers` in list order and silently skips any provider that isn't
# compiled into the installed onnxruntime package, always falling through to
# CPUExecutionProvider, which ships in every build. Confirmed on this Mac
# (`ort.get_available_providers()`): CoreMLExecutionProvider (Apple GPU/
# Neural Engine) and CPUExecutionProvider are present; QNNExecutionProvider
# (Qualcomm Hexagon DSP) is not, since this hardware doesn't have it -- so
# here this resolves to CoreML, with CPU as the guaranteed fallback. On a
# Qualcomm-powered device with the QNN provider installed, the same list
# would pick that up first instead, no code change needed.
PROVIDER_PRIORITY = [
    "QNNExecutionProvider",     # DSP/NPU (Qualcomm Hexagon)
    "CoreMLExecutionProvider",  # GPU/Neural Engine (Apple Silicon)
    "CUDAExecutionProvider",    # GPU (NVIDIA, only if onnxruntime-gpu is installed)
    "CPUExecutionProvider",     # always available -- guaranteed fallback
]


def load_session(onnx_path: str) -> ort.InferenceSession:
    available = set(ort.get_available_providers())
    providers = [p for p in PROVIDER_PRIORITY if p in available]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    print(f"Execution provider: {sess.get_providers()[0]}  (tried, in order: {providers})")
    in_info = sess.get_inputs()[0]
    out_info = sess.get_outputs()[0]
    print(f"Input : {in_info.name} {in_info.shape} {in_info.type}")
    print(f"Output: {out_info.name} {out_info.shape} {out_info.type}")
    return sess


def run_one(sess: ort.InferenceSession, mel_float: np.ndarray):
    """mel_float: np.float32 array shaped [MAX_FRAMES, N_MELS]."""
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    x = mel_float[np.newaxis, ...].astype(np.float32)  # [1, MAX_FRAMES, N_MELS]

    t0 = time.perf_counter()
    (logits,) = sess.run([output_name], {input_name: x})
    latency_ms = (time.perf_counter() - t0) * 1000.0

    return logits[0], latency_ms


def eval_validation_set(sess, data_root, num_samples):
    val_ds = SpeechCommandsDataset(data_root, "validation", augment=False)
    n = min(num_samples, len(val_ds)) if num_samples else len(val_ds)
    print(f"Evaluating on {n} validation clips...")

    correct = 0
    latencies = []
    for i in range(n):
        mel, label = val_ds[i]
        logits, latency_ms = run_one(sess, mel.numpy())
        pred = int(np.argmax(logits))
        correct += int(pred == label)
        latencies.append(latency_ms)

    acc = correct / n
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies)) - 1]
    print(f"\nAccuracy:        {acc:.4f}  ({correct}/{n})")
    print(f"Avg latency:     {avg_latency:.2f} ms/sample")
    print(f"p95 latency:     {p95_latency:.2f} ms/sample")


def run_single_wav(sess, wav_path):
    mel = load_wav_as_mel(wav_path, augment=False).numpy()
    logits, latency_ms = run_one(sess, mel)
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    top5 = np.argsort(probs)[::-1][:5]

    print(f"\n{wav_path}")
    print(f"Latency: {latency_ms:.2f} ms")
    print("Top-5 predictions:")
    for idx in top5:
        print(f"  {kws.KEYWORDS[idx]:<10s} {probs[idx]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="keyword_spotting_int8.onnx")
    parser.add_argument("--data-root", default=kws.DATA_ROOT)
    parser.add_argument("--num-samples", type=int, default=None,
                         help="Limit validation-set eval to N samples (default: all)")
    parser.add_argument("--wav", default=None, help="Run a single WAV file instead of the full sweep")
    args = parser.parse_args()

    sess = load_session(args.onnx)

    if args.wav:
        run_single_wav(sess, args.wav)
    else:
        eval_validation_set(sess, args.data_root, args.num_samples)


if __name__ == "__main__":
    main()
