"""
Export the trained PyTorch keyword-spotting model to ONNX, then statically
(post-training, full-integer) quantize it with ONNX Runtime -- the ONNX
equivalent of PyTorch's own static quantization APIs, using ONNX Runtime's
QDQ (QuantizeLinear/DequantizeLinear) quantizer instead.

Two stages:
  1. torch.onnx.export -> a float32 .onnx graph with a fixed [1, 101, 80]
     input shape (matches the TFLite side's batch=1 fixed-shape export).
  2. onnxruntime.quantization.quantize_static -> int8 weights + int8
     activations, calibrated on real, unaugmented validation-set spectrograms
     (same idea as convert_to_tflite.py's representative_dataset).

Usage:
    python convert_to_onnx.py
    python convert_to_onnx.py --num-calibration 300
"""

import argparse

import numpy as np
import torch
from onnxruntime.quantization import (
    CalibrationDataReader, QuantFormat, QuantType, quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

import kws_common_torch as kws
from kws_common_torch import KeywordSpottingTransformer, SpeechCommandsDataset


class MelCalibrationDataReader(CalibrationDataReader):
    """Feeds real, unaugmented validation-set mel-spectrograms to the ORT
    calibrator, one [1, MAX_FRAMES, N_MELS] sample at a time."""

    def __init__(self, val_ds: SpeechCommandsDataset, input_name: str, num_samples: int):
        self.val_ds = val_ds
        self.input_name = input_name
        self.n = min(num_samples, len(val_ds))
        self.idx = 0

    def get_next(self):
        if self.idx >= self.n:
            return None
        mel, _label = self.val_ds[self.idx]
        self.idx += 1
        mel_np = mel.unsqueeze(0).numpy().astype(np.float32)  # [1, MAX_FRAMES, N_MELS]
        return {self.input_name: mel_np}

    def rewind(self):
        self.idx = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=kws.CKPT_PATH)
    parser.add_argument("--data-root", default=kws.DATA_ROOT)
    parser.add_argument("--fp32-out", default="keyword_spotting_fp32.onnx")
    parser.add_argument("--out", default="keyword_spotting_int8.onnx")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--num-calibration", type=int, default=200,
                         help="Number of representative samples for calibration")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    print(f"  epoch={ckpt.get('epoch')}  val_acc={ckpt.get('val_acc'):.4f}  config={config}")

    model = KeywordSpottingTransformer(**config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 1. Export FP32 ONNX with a fixed [1, MAX_FRAMES, N_MELS] input shape --
    #    no dynamic_axes, matching the fixed batch=1 shape used on the TFLite
    #    side. Keeps the graph simple and calibration-friendly.
    dummy = torch.zeros(1, kws.MAX_FRAMES, kws.N_MELS, dtype=torch.float32)
    print(f"Exporting FP32 ONNX (opset {args.opset}) -> {args.fp32_out}")
    torch.onnx.export(
        model, dummy, args.fp32_out,
        input_names=["mel_input"], output_names=["logits"],
        opset_version=args.opset, do_constant_folding=True,
    )

    # 2. Pre-process: symbolic shape inference + light graph optimization.
    #    ORT's quantizer relies on this to see shapes for every node.
    preprocessed_path = args.fp32_out.replace(".onnx", "_preprocessed.onnx")
    print(f"Pre-processing (shape inference) -> {preprocessed_path}")
    quant_pre_process(args.fp32_out, preprocessed_path, skip_optimization=False)

    # 3. Calibrate + statically quantize (QDQ, int8 weights + int8 activations).
    print("Gathering calibration data from the validation split...")
    val_ds = SpeechCommandsDataset(args.data_root, "validation", augment=False)
    reader = MelCalibrationDataReader(val_ds, input_name="mel_input", num_samples=args.num_calibration)

    print(f"Calibrating over {reader.n} samples and quantizing -> {args.out}")
    quantize_static(
        model_input=preprocessed_path,
        model_output=args.out,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    print(f"\nSaved statically quantized model: {args.out}")
    print("Next: run run_onnx_mac.py to verify accuracy/latency locally before deploying to the Raspberry Pi.")


if __name__ == "__main__":
    main()
