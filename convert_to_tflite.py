"""
Convert the trained Keras keyword-spotting model to a statically
(full-integer, post-training) quantized TFLite model.

This is the TF equivalent of PyTorch static quantization: weights AND
activations are calibrated and quantized to int8 using a representative
dataset (real, unaugmented mel-spectrograms drawn from the validation set).

Usage:
    python convert_to_tflite.py
    python convert_to_tflite.py --io-type float32   # keep float32 in/out, int8 internals
"""

import argparse
import json

import numpy as np
import tensorflow as tf

import kws_common as kws


def representative_dataset_gen(pairs, num_samples):
    """Yields real, unaugmented mel-spectrograms for calibration."""
    for path, _label in pairs[:num_samples]:
        mel = kws.load_and_process(path, augment=False).numpy()  # [MAX_FRAMES, N_MELS]
        mel = mel[np.newaxis, ...].astype(np.float32)             # [1, MAX_FRAMES, N_MELS]
        yield [mel]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=kws.WEIGHTS_PATH)
    parser.add_argument("--config", default=kws.CONFIG_PATH)
    parser.add_argument("--data-root", default=kws.DATA_ROOT)
    parser.add_argument("--out", default="keyword_spotting_int8.tflite")
    parser.add_argument("--num-calibration", type=int, default=200,
                         help="Number of representative samples for calibration")
    parser.add_argument("--io-type", choices=["int8", "float32"], default="int8",
                         help="int8: fully integer I/O (best for RPi/microcontrollers). "
                              "float32: float I/O, int8 math internally (easier to feed raw floats).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    # Drop training-run bookkeeping keys that aren't build_model() kwargs.
    model_kwargs = {k: v for k, v in config.items() if k not in ("epoch", "val_acc")}

    print(f"Building model with config: {model_kwargs}")
    model = kws.build_model(**model_kwargs, batch_size=1)
    model.load_weights(args.weights)

    print("Gathering calibration data from the validation split...")
    calib_pairs = kws.list_split(args.data_root, "validation")
    if len(calib_pairs) < args.num_calibration:
        raise ValueError(
            f"Only {len(calib_pairs)} validation samples available, "
            f"need at least {args.num_calibration}."
        )

    def rep_gen():
        yield from representative_dataset_gen(calib_pairs, args.num_calibration)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if args.io_type == "int8":
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    print("Converting (this runs calibration over the representative dataset)...")
    tflite_model = converter.convert()

    with open(args.out, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"\nSaved statically quantized model: {args.out} ({size_kb:.1f} KB)")
    print(f"I/O dtype: {args.io_type}")
    print("Next: run run_tflite_mac.py to verify accuracy/latency locally before deploying to the Raspberry Pi.")


if __name__ == "__main__":
    main()
