"""
Run and verify the statically quantized TFLite keyword-spotting model on Mac,
before taking it to the Raspberry Pi.

Two modes:
  1. Full validation-set sweep -> accuracy + average latency
        python run_tflite_mac.py
  2. Single WAV file smoke test -> predicted keyword + confidence
        python run_tflite_mac.py --wav path/to/clip.wav

Uses `tf.lite.Interpreter` (part of the `tensorflow` package, which is what
you'll already have installed for training/conversion). On the Raspberry Pi
you generally do NOT want the full `tensorflow` package -- swap the two
import lines marked below for `tflite_runtime` (or the newer `ai-edge-litert`
package) and the rest of this script runs unchanged, since the Interpreter
API is identical.
"""

import argparse
import time

import numpy as np

# --- Mac (this script): use tf.lite.Interpreter from the full tensorflow package.
import tensorflow as tf
Interpreter = tf.lite.Interpreter
# --- Raspberry Pi: comment the two lines above and uncomment ONE of these:
# import tflite_runtime.interpreter as tflite
# Interpreter = tflite.Interpreter
# from ai_edge_litert.interpreter import Interpreter

import kws_common as kws


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
    interp = Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    print("Input :", in_det["shape"].tolist(), in_det["dtype"], "quant(scale,zp)=", in_det["quantization"])
    print("Output:", out_det["shape"].tolist(), out_det["dtype"], "quant(scale,zp)=", out_det["quantization"])
    return interp, in_det, out_det


def run_one(interp, in_det, out_det, mel_float):
    """mel_float: np.float32 array shaped [MAX_FRAMES, N_MELS]."""
    x = mel_float[np.newaxis, ...]
    x_q = quantize(x, in_det["quantization"], in_det["dtype"])
    interp.set_tensor(in_det["index"], x_q)

    t0 = time.perf_counter()
    interp.invoke()
    latency_ms = (time.perf_counter() - t0) * 1000.0

    out_q = interp.get_tensor(out_det["index"])
    logits = dequantize(out_q, out_det["quantization"], out_det["dtype"])[0]
    return logits, latency_ms


def eval_validation_set(interp, in_det, out_det, data_root, num_samples):
    pairs = kws.list_split(data_root, "validation")
    if num_samples:
        pairs = pairs[:num_samples]
    print(f"Evaluating on {len(pairs)} validation clips...")

    correct = 0
    latencies = []
    for path, label in pairs:
        mel = kws.load_and_process(path, augment=False).numpy()
        logits, latency_ms = run_one(interp, in_det, out_det, mel)
        pred = int(np.argmax(logits))
        correct += int(pred == label)
        latencies.append(latency_ms)

    acc = correct / len(pairs)
    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies)) - 1]
    print(f"\nAccuracy:        {acc:.4f}  ({correct}/{len(pairs)})")
    print(f"Avg latency:     {avg_latency:.2f} ms/sample")
    print(f"p95 latency:     {p95_latency:.2f} ms/sample")


def run_single_wav(interp, in_det, out_det, wav_path):
    mel = kws.load_and_process(wav_path, augment=False).numpy()
    logits, latency_ms = run_one(interp, in_det, out_det, mel)
    probs = tf.nn.softmax(logits).numpy()
    top5 = np.argsort(probs)[::-1][:5]

    print(f"\n{wav_path}")
    print(f"Latency: {latency_ms:.2f} ms")
    print("Top-5 predictions:")
    for idx in top5:
        print(f"  {kws.KEYWORDS[idx]:<10s} {probs[idx]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite", default="keyword_spotting_int8.tflite")
    parser.add_argument("--data-root", default=kws.DATA_ROOT)
    parser.add_argument("--num-samples", type=int, default=None,
                         help="Limit validation-set eval to N samples (default: all)")
    parser.add_argument("--wav", default=None, help="Run a single WAV file instead of the full sweep")
    args = parser.parse_args()

    interp, in_det, out_det = load_interpreter(args.tflite)

    if args.wav:
        run_single_wav(interp, in_det, out_det, args.wav)
    else:
        eval_validation_set(interp, in_det, out_det, args.data_root, args.num_samples)


if __name__ == "__main__":
    main()
