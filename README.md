# Keyword spotting: decoder-only Transformer, two deployment pipelines

A decoder-only Transformer keyword spotter, trained and statically (int8) quantized two ways — **TensorFlow → TFLite** and **PyTorch → ONNX** — each verified on Mac before deploying to a Raspberry Pi. Both pipelines implement the exact same architecture (see [Model architecture](#model-architecture)) and share one downloaded dataset, so their results are directly comparable.

### TensorFlow → TFLite

| File | Purpose |
|---|---|
| [kws_common.py](kws_common.py) | Config, dataset download/pipeline, model architecture — shared by everything below so there's no drift between training and inference. |
| [train_tf.py](train_tf.py) | Trains the model (custom loop mirroring the PyTorch script: OneCycle LR, grad clipping, label smoothing, best-val checkpointing). |
| [convert_to_tflite.py](convert_to_tflite.py) | Post-training **static** (full-integer) quantization to `.tflite`, calibrated on real validation-set spectrograms. |
| [run_tflite_mac.py](run_tflite_mac.py) | Runs the quantized model on Mac — full validation sweep (accuracy + latency) or a single WAV smoke test — before touching the Pi. |

### PyTorch → ONNX

| File | Purpose |
|---|---|
| [kws_common_torch.py](kws_common_torch.py) | Config, dataset download/pipeline, model architecture — the PyTorch-side equivalent of `kws_common.py`, shared by everything below. |
| [train.py](train.py) | Trains the model (OneCycleLR, grad clipping, label-smoothed cross-entropy, best-val checkpointing). |
| [convert_to_onnx.py](convert_to_onnx.py) | Exports to ONNX, then post-training **static** (int8 weights + int8 activations, QDQ format) quantization via ONNX Runtime, calibrated on real validation-set spectrograms. |
| [run_onnx_mac.py](run_onnx_mac.py) | Runs the quantized model on Mac via ONNX Runtime — full validation sweep or a single WAV smoke test — before touching the Pi. |
| [run_onnx_pi.py](run_onnx_pi.py) | **Standalone** Raspberry Pi inference script — only `numpy` + `onnxruntime`, no torch/torchaudio. Has its own NumPy mel-spectrogram implementation (verified to match torchaudio's output to float32 precision — see [Notes](#notes)). |

## Model architecture

The task: classify a 1-second, 16 kHz audio clip into one of 35 keywords. The model is a **decoder-only Transformer** (GPT-style — causal self-attention, not the bidirectional attention a BERT-style encoder uses), repurposed for classification by reading out a trailing `[CLS]` token instead of generating text. This section describes the architecture in framework-agnostic terms — `kws_common.py` (TensorFlow) and `kws_common_torch.py` (PyTorch) both implement it identically, layer for layer.

### 1. Front end: waveform → log-mel spectrogram

Before the network ever sees the audio, it's converted to a log-mel spectrogram:

```
waveform [16,000 samples @ 16 kHz]
   │  reflect-pad by N_FFT//2 = 256 each side (centers the framing, à la torchaudio's center=True)
   ▼
STFT: 512-point FFT, 400-sample Hann window, 160-sample hop  →  257 freq bins × 101 frames
   │  power = |STFT|²
   ▼
Mel filterbank: 257 linear bins → 80 mel bins (0–8000 Hz)
   ▼
log(mel + 1e-6), then per-clip normalize: (x − mean) / (std + 1e-5)
   ▼
log-mel spectrogram [101 frames, 80 mel bins]
```

101 frames is exact and load-bearing: it's baked into the model's fixed positional-embedding table (`MAX_FRAMES = 101`), so every clip is padded/trimmed to exactly 1 second before this transform runs. During training, `FrequencyMasking`/`TimeMasking`-style SpecAugment (random block-outs along the mel and time axes) is applied to this spectrogram for regularization; at inference it's skipped.

### 2. Tokenizing frames + the `[CLS]` token

```
log-mel [101, 80]
   │  Dense(80 → 256)                      "input_proj": each frame becomes a 256-dim vector
   ▼
frame embeddings [101, 256]
   │  append a learnable [CLS] token (zero-initialized) at the END → [102, 256]
   │  add learned positional embeddings (Embedding[102, 256], one per position 0..101)
   ▼
sequence [102, 256]   (101 audio frames, then 1 CLS token at position 101)
```

Putting `[CLS]` at the **end**, combined with causal masking (next section), is the key trick: with causal attention, a position can only attend to itself and earlier positions. The last position is therefore the *only* one that can see the entire clip — making it a natural, forced summarization point. (A BERT-style encoder would instead put `[CLS]` first and rely on full bidirectional attention to let it see everything; a decoder-only model can't do that, so the token has to go last.)

### 3. Decoder blocks (× 4)

Each of the 4 identical blocks is a standard pre-norm Transformer block:

```
x  →  LayerNorm  →  Causal Self-Attention (4 heads, 64-dim each)  →  + x  (residual)
   →  LayerNorm  →  FeedForward (256 → 512 → 256, GELU)           →  + …  (residual)
```

**Causal self-attention**: queries/keys/values are a single fused `Dense(256 → 768)` split into 4 heads of 64 dims each. Attention scores are `QKᵀ / √64`, masked with an additive `-1e4` upper-triangular bias before softmax so position *i* can only attend to positions `≤ i` — this is what forces the CLS token (last position) into the "sees everything" role above. It's the same causal mask used in GPT-style language models, just applied to audio frames instead of text tokens.

**FeedForward**: two dense layers with an exact (erf-based) GELU in between and a 4× expansion (256 → 512 → 256) — the standard Transformer MLP block.

### 4. Classification head

```
[102, 256]  →  final LayerNorm  →  take position -1 (the CLS token)  →  Dense(256 → 35)  →  logits
```

Only the CLS token's final representation feeds the classifier — the other 101 positions exist purely to be attended *into* the CLS token along the way.

### Config & parameter count

| | |
|---|---|
| Sample rate / clip length | 16 kHz / 1.0 s (16,000 samples) |
| Mel bins / frames | 80 / 101 |
| `d_model` / heads / layers / `d_ff` | 256 / 4 / 4 / 512 |
| Dropout | 0.1 (train only) |
| Classes | 35 keywords |
| **Total parameters** | **2,165,027** (~8.3 MB in float32) |

| Component | Params |
|---|---|
| `input_proj` (Dense 80→256) | 20,736 |
| `cls_pos_embed` (CLS token + positional embeddings) | 26,368 |
| 4 × decoder block (attention + feedforward + 2 LayerNorms) | 527,104 each → 2,108,416 |
| `final_norm` (LayerNorm) | 512 |
| `head` (Dense 256→35) | 8,995 |

## TensorFlow → TFLite: how to run

### 1. Setup

```bash
uv sync                              # or: pip install -r requirements_tf.txt
```

Pinned to `tensorflow==2.18.1` and deliberately **not** using `tensorflow-metal` — Apple's Metal GPU backend was found to compute numerically incorrect gradients for this model (see [Notes](#notes)), so `train_tf.py` force-disables the GPU and always trains on CPU.

### 2. Train

```bash
uv run python train_tf.py
```

Downloads the Google Speech Commands v0.02 dataset (~2.4 GB, first run only) to `./data/`, trains for 40 epochs, and saves the best checkpoint (by validation accuracy) to:
- `checkpoints_tf/best_model.weights.h5` — the weights
- `checkpoints_tf/best_model_config.json` — model config + which epoch/val_acc produced it

CPU-only, ~230s/epoch at batch size 256 on Apple Silicon — budget a few hours for the full 40 epochs. Progress prints per-epoch train/val loss and accuracy; only checkpoints that improve validation accuracy are saved.

### 3. Static quantization → TFLite

```bash
uv run python convert_to_tflite.py
```

This is the TF equivalent of PyTorch static quantization: it loads the trained checkpoint, feeds 200 real (unaugmented) validation-set spectrograms through the converter as a **representative dataset** to calibrate int8 activation ranges, then quantizes both weights and activations to int8. Output: `keyword_spotting_int8.tflite`.

Useful flags:
```bash
--weights PATH          # default: checkpoints_tf/best_model.weights.h5
--out PATH               # default: keyword_spotting_int8.tflite
--num-calibration N       # default: 200 representative samples
--io-type {int8,float32}  # default: int8 (fully integer I/O, best for RPi);
                           # float32 keeps float in/out with int8 math internally
```

### 4. Verify the quantized model on Mac

```bash
uv run python run_tflite_mac.py                       # full validation-set sweep: accuracy + latency
uv run python run_tflite_mac.py --num-samples 500      # faster partial sweep
uv run python run_tflite_mac.py --wav some_clip.wav    # single-file smoke test: top-5 predictions
```

The full sweep reports overall accuracy plus average and p95 per-sample latency — this is the check to run before trusting the model on-device. Only once this looks right should you copy `keyword_spotting_int8.tflite` (plus `run_tflite_mac.py`/`kws_common.py` for the preprocessing code) over to the Raspberry Pi.

### 5. Deploy to Raspberry Pi

In `run_tflite_mac.py`, swap the "Mac" import block for the "Raspberry Pi" block (both are present, one commented out) — the `Interpreter.allocate_tensors/set_tensor/invoke/get_tensor` API is identical either way, so no other code changes are needed. On the Pi, install `tflite-runtime` or `ai-edge-litert` instead of full `tensorflow` (see [requirements_tf.txt](requirements_tf.txt)) — the converted model uses only built-in TFLite ops (no Flex/Select-TF ops), so the lightweight runtime is sufficient.

## PyTorch → ONNX: how to run

### 1. Setup

```bash
uv sync --extra torch                # or: pip install -r requirements_torch.txt
```

Installs `torch`, `torchaudio` (transforms only — see [Notes](#notes)), `onnx`, `onnxruntime`, `onnxscript` (required by `torch.onnx.export`'s current exporter).

### 2. Train

```bash
uv run python train.py
```

Reuses the same `./data/SpeechCommands/speech_commands_v0.02/` download the TF pipeline uses (no re-download), trains for 40 epochs, and saves the best checkpoint (by validation accuracy) to `checkpoints/best_model.pt` — a single file containing weights, config, epoch, and val_acc, exactly like the original script.

Runs on Apple Silicon GPU (MPS) automatically when available (`kws_common_torch.py`'s `DEVICE`), falling back to CUDA then CPU. Unlike `tensorflow-metal` (see [Notes](#notes)), PyTorch's MPS backend was verified safe for this model before being enabled by default — see the MPS note below if you ever want the receipts or need to fall back to CPU.

### 3. Static quantization → ONNX

```bash
uv run python convert_to_onnx.py
```

Two stages: (1) `torch.onnx.export` to a float32 ONNX graph with a fixed `[1, 101, 80]` input shape (no dynamic axes, matching the TFLite side's fixed batch=1 export), then a `quant_pre_process` shape-inference pass; (2) ONNX Runtime's `quantize_static` — the ONNX equivalent of PyTorch's own static-quantization APIs — calibrates int8 ranges on 200 real (unaugmented) validation-set spectrograms and produces a QDQ-format int8 model (int8 weights, per-channel; int8 activations). Output: `keyword_spotting_int8.onnx`.

Useful flags:
```bash
--checkpoint PATH        # default: checkpoints/best_model.pt
--out PATH                # default: keyword_spotting_int8.onnx
--num-calibration N        # default: 200 representative samples
--opset N                  # default: 18
```

### 4. Verify the quantized model on Mac

```bash
uv run python run_onnx_mac.py                       # full validation-set sweep: accuracy + latency
uv run python run_onnx_mac.py --num-samples 500      # faster partial sweep
uv run python run_onnx_mac.py --wav some_clip.wav    # single-file smoke test: top-5 predictions
```

Same shape as the TFLite verification step: this is the check to run before trusting the model on-device.

### 5. Deploy to Raspberry Pi

```bash
pip install onnxruntime numpy   # on the Pi — no torch/torchaudio needed
python run_onnx_pi.py --wav some_clip.wav
```

Copy `keyword_spotting_int8.onnx` and **`run_onnx_pi.py`** (not `run_onnx_mac.py`) to the Pi — it's a separate, self-contained script with its own NumPy log-mel implementation and zero torch/torchaudio dependency (verified in isolation, see [Notes](#notes)). The inference call itself (`onnxruntime.InferenceSession` + `.run(...)`) is identical to what runs on Mac; only the preprocessing path differs.

## Quantized model I/O: why TFLite's boundary is int8 and ONNX's is float32

Run both verification scripts and the printed I/O dtypes differ:

```
$ uv run python run_tflite_mac.py
Input : [1, 101, 80] <class 'numpy.int8'> quant(scale,zp)= (0.03343628719449043, -26)

$ uv run python run_onnx_mac.py
Input : mel_input [1, 101, 80] tensor(float)
```

This is **not** a difference between what TFLite and ONNX Runtime are *capable* of — it's a difference between the conversion flags this repo happens to use for each. Both formats support both boundary types; the sections below explain the actual mechanism in each, then exactly what changes to switch each one to the other. Every claim and code snippet below was run against the real files in this repo, not written from memory.

### 1. TFLite: the converter can strip or keep the boundary Q/DQ ops

`convert_to_tflite.py` calls `TFLiteConverter.convert()` with full-graph int8 calibration active either way (`converter.optimizations = [tf.lite.Optimize.DEFAULT]`, `target_spec.supported_ops = [TFLITE_BUILTINS_INT8]`). What changes I/O dtype is two extra lines, gated on `--io-type`:

```python
# convert_to_tflite.py
if args.io_type == "int8":
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
```

Internally, the converter *always* computes where a boundary `Quantize`/`Dequantize` pair would go (using the representative-dataset calibration, same as every other op). `inference_input_type`/`inference_output_type` decide what happens to that pair:
- **Left at the default (`tf.float32`)**: the boundary `Quantize`/`Dequantize` ops are kept as real ops inside the flatbuffer graph. The model accepts/returns float32, and quantization happens as the first/last thing the graph itself does.
- **Set to `tf.int8`**: the converter *removes* those boundary ops and exposes their `(scale, zero_point)` as tensor metadata instead — retrievable via `interpreter.get_input_details()[0]['quantization']`. The graph's first real op now consumes int8 directly; nothing inside the flatbuffer does the float→int8 conversion, because the boundary op that used to do it was deleted.

That's why `int8` mode needs caller-side code and `float32` mode wouldn't: `run_tflite_mac.py`'s `quantize()`/`dequantize()` functions are reimplementing, in Python, exactly the op the converter stripped out of the graph:

```python
# run_tflite_mac.py
def quantize(x_float, quant_params, dtype):
    scale, zero_point = quant_params
    if dtype in (np.int8, np.uint8) and scale != 0:
        q = np.round(x_float / scale + zero_point)
        info = np.iinfo(dtype)
        q = np.clip(q, info.min, info.max)
        return q.astype(dtype)
    return x_float.astype(dtype)   # already float32 -- no-op
```
i.e. standard affine/asymmetric quantization: `q = clip(round(x/scale + zero_point), -128, 127)`, `x = (q - zero_point) * scale`.

### 2. ONNX Runtime (QDQ format): the Q/DQ ops always live *inside* the graph

`convert_to_onnx.py` calls `quantize_static(..., quant_format=QuantFormat.QDQ, ...)`. Critically, **`quant_format` controls internal op representation, not the graph's external I/O dtype** — there is no ORT equivalent of `inference_input_type`. `graph.input`/`graph.output` keep whatever dtype `torch.onnx.export` gave them (float32), regardless of `QuantFormat.QDQ` vs `QuantFormat.QOperator`.

Tracing the real exported graph (`onnx.load("keyword_spotting_int8.onnx")`, following node producers/consumers from `mel_input`) shows exactly what QDQ actually inserted:

```
mel_input (FLOAT)
  --[Reshape "gemm_input_reshape"]-->  gemm_input_reshape_arg
  --[QuantizeLinear "gemm_input_reshape_arg_QuantizeLinear"]-->  ..._QuantizeLinear_Output   (INT8)
  --[DequantizeLinear "gemm_input_reshape_arg_DequantizeLinear"]-->  ..._DequantizeLinear_Output  (FLOAT)
  --[Gemm "node_MatMul_1/MatMulAddFusion"]-->  ...   (input_proj Linear layer)
```
with the real calibrated parameters: `gemm_input_reshape_arg_scale = 0.033006`, `gemm_input_reshape_arg_zero_point = -29` (int8). Symmetrically, the graph's *last* op is a `DequantizeLinear` (`logits_DequantizeLinear`, `scale=0.034553`, `zero_point=-63`) converting the int8-computed logits back to the float32 `logits` output.

Per the ONNX spec, `QuantizeLinear` computes `y = saturate(round(x / y_scale) + y_zero_point)` and `DequantizeLinear` computes `y = (x - x_zero_point) * x_scale` — the same affine scheme as TFLite, just expressed as graph nodes instead of tensor metadata. (One real subtlety if you're chasing bit-exactness: ONNX's `QuantizeLinear` rounds half-to-even per spec; TFLite's rounding-tie behavior isn't guaranteed identical. Doesn't matter for accuracy at these scales, but worth knowing before assuming int8 quantized values are portable byte-for-byte between the two.)

This is also *why* `run_onnx_mac.py`/`run_onnx_pi.py` have no `quantize()`/`dequantize()` functions at all: that logic isn't missing, it's just living inside the graph as `QuantizeLinear`/`DequantizeLinear` nodes that execute automatically on every `session.run()`, instead of as Python code the caller has to invoke. Separately: the *on-disk* graph shape above (Reshape→Quantize→Dequantize→Gemm, i.e. the Gemm still nominally runs on dequantized float32 values) is what QDQ format always looks like as saved — ONNX Runtime's graph optimizer is free to fuse `Quantize→Op→Dequantize` patterns into genuine int8 kernels at session-load time for whichever execution provider is active (this is very likely why the measured latency was ~1.3ms — much faster than a real all-float32 forward pass at this size would be), but that fusion is a runtime decision, not something reflected in the saved `.onnx` file itself.

### 3. Making TFLite accept float32 input instead of int8

Already fully implemented — no code changes needed, just a flag:

```bash
uv run python convert_to_tflite.py --io-type float32
```

This skips the `converter.inference_input_type = tf.int8` / `inference_output_type = tf.int8` lines, so both stay at the converter's default (`tf.float32`), which keeps the boundary `Quantize`/`Dequantize` ops inside the graph — structurally the same shape as the ONNX QDQ graph in §2.

`run_tflite_mac.py` doesn't need touching either: its `quantize()`/`dequantize()` functions already branch on `in_det["dtype"]` — for a `float32`-I/O model that branch is false, so they fall through to `.astype(dtype)` (a no-op cast) and pass the array straight through. The same script already handles both `--io-type` variants transparently; this was true before this section was written, just not exercised or documented until now.

### 4. Making ONNX accept int8 input instead of float32

Unlike TFLite, **`onnxruntime.quantization` has no built-in flag for this** — `quant_format=QuantFormat.QOperator` changes how ops get fused/named internally, but neither `QOperator` nor `QDQ` ever touches `graph.input`/`graph.output`'s declared dtype. Getting genuine int8 boundary I/O requires manual graph surgery on the already-quantized `.onnx` file: delete the boundary `QuantizeLinear`/`DequantizeLinear` nodes, rewire their neighbors, and change the graph's declared input/output element type. This is not officially documented/supported by ONNX Runtime's quantization tools, so treat it as a recipe, not an API — the following was written against this repo's actual `keyword_spotting_int8.onnx` and verified (below), but it depends on the specific node names ORT's quantizer happened to generate.

```python
import onnx
from onnx import TensorProto, numpy_helper, shape_inference

model = onnx.load("keyword_spotting_int8.onnx")
graph = model.graph
init_by_name = {i.name: i for i in graph.initializer}

def get_scale_zp(node):
    scale = float(numpy_helper.to_array(init_by_name[node.input[1]]))
    zero_point = int(numpy_helper.to_array(init_by_name[node.input[2]]))
    return scale, zero_point

# --- Input side: delete the QuantizeLinear right after mel_input's Reshape,
#     rewire its consumers to read its (now-int8) input tensor directly, and
#     declare the graph input itself as int8.
q_node = next(n for n in graph.node if n.name == "gemm_input_reshape_arg_QuantizeLinear")
in_scale, in_zero_point = get_scale_zp(q_node)
produced, consumed = q_node.output[0], q_node.input[0]
for n in graph.node:
    n.input[:] = [consumed if x == produced else x for x in n.input]
graph.node.remove(q_node)
graph.input[0].type.tensor_type.elem_type = TensorProto.INT8

# --- Output side: delete the final DequantizeLinear, and make the graph's
#     declared output the (int8) tensor that used to feed it.
dq_node = next(n for n in graph.node if n.name == "logits_DequantizeLinear")
out_scale, out_zero_point = get_scale_zp(dq_node)
graph.output[0].name = dq_node.input[0]
graph.node.remove(dq_node)
graph.output[0].type.tensor_type.elem_type = TensorProto.INT8

# The intermediate value_info entries between the old boundary ops and the
# rest of the graph still declare float32 from before this edit -- onnx's
# checker doesn't catch the resulting mismatch, but ONNX Runtime's loader
# does (confirmed: "Type Error: Type (tensor(float)) of output arg ... does
# not match expected type (tensor(int8))" without this). Clear and recompute.
del graph.value_info[:]
model = shape_inference.infer_shapes(model)

onnx.checker.check_model(model)
onnx.save(model, "keyword_spotting_int8_intio.onnx")
print(f"input:  scale={in_scale}  zero_point={in_zero_point}")
print(f"output: scale={out_scale}  zero_point={out_zero_point}")
```

**Verified, not just written**: ran this against the real `keyword_spotting_int8.onnx`, loaded the result in `onnxruntime.InferenceSession` (confirmed `input: mel_input [1,101,80] tensor(int8)`, `output: [1,35] tensor(int8)`), and compared predictions against the original float-I/O model on 20 real validation clips using the printed `in_scale`/`in_zero_point`/`out_scale`/`out_zero_point` to manually quantize/dequantize (i.e. `run_tflite_mac.py`'s `quantize()`/`dequantize()` functions, reused as-is) — **20/20 matched**. The first attempt (without the `value_info` clear + `shape_inference.infer_shapes()` step) passed `onnx.checker.check_model()` but failed to load in `onnxruntime.InferenceSession` with the type-mismatch error quoted in the comment above — a real gap between what the ONNX checker validates and what the ORT loader requires, worth knowing if you extend this.

To actually use an int8-I/O ONNX model, `run_onnx_mac.py`/`run_onnx_pi.py` would need their own `quantize()`/`dequantize()` functions added (the same two functions already in `run_tflite_mac.py` would work verbatim, since both formats use the identical affine int8 scheme) — not done in this repo since QDQ's float-I/O + internal fusion already gets the same ~1.3ms latency without that extra caller-side bookkeeping.

## Notes

### TensorFlow → TFLite

- **Causal mask magnitude (`-100`, not `-1e4`) — this one actually mattered.** After real training (val_acc 0.9531), the first int8 conversion scored only **0.2602** on the same validation set. Root cause, confirmed with the trained model's real numbers: the causal mask was added to attention logits as a `-1e4` constant. That's invisible in float32 (`exp(-1e4)` and `exp(-100)` both underflow to a 0 softmax weight), but the mask-add op's *output* gets a single int8 quantization range spanning both the mask and the real logits. With `-1e4` that range is ~[-10013, +27] → ~39 units per quantization level, while the real (trained) attention-logit spread is only ~[-27, +27] — i.e. the entire meaningful signal was collapsing into about *one* int8 bucket. Verified directly in the converted `.tflite`: every `causal_self_attention_N/add` tensor had `scale≈39.3`. Changed the mask to `-100` (still >>3x past the observed logit extremes, so masking is unaffected) — range becomes ~[-127, +27], ~0.6 units/level, ~65x better precision. Since the mask isn't a trainable parameter, the existing trained weights didn't need retraining — just rebuild + reconvert. Re-measured: **0.9521**, a normal ~0.1-point quantization gap from the 0.9531 float32 baseline. The same fix was applied to `kws_common_torch.py` proactively, before the ONNX pipeline hits the identical issue.
- **Dataset reuse**: the TF loader downloads to `./data/SpeechCommands/speech_commands_v0.02/`, the same layout torchaudio uses — if you already ran the PyTorch script against `./data`, this reuses it instead of re-downloading ~2.4 GB.
- **Frame count**: the log-mel pipeline reflect-pads by `N_FFT//2` before framing so a 1 s clip always yields exactly `MAX_FRAMES=101` frames, matching torchaudio's `center=True` behavior. The window is `N_FFT`-length Hann rather than a zero-padded `WIN_LENGTH` window inside `N_FFT` (torch's exact approach) — numerically slightly different, functionally equivalent; not worth chasing bit-exactness for a from-scratch TF training run.
- **Verified locally** (this session, TF 2.21 on Apple Silicon): model builds and traces correctly, mel spectrogram yields shape `(101, 80)`, static int8 conversion succeeds using **only built-in TFLite ops** (`FULLY_CONNECTED`, `BATCH_MATMUL`, `SOFTMAX`, `GELU`, `MEAN`, …) — no Flex/Select-TF ops, no custom ops — so the same `.tflite` file will load with lightweight `tflite_runtime`/`ai-edge-litert` on the Raspberry Pi, not just the full `tensorflow` package.
- **LayerNorm quantization islands**: a handful of tiny float32 tensors remain around each `LayerNormalization`'s mean computation (`DEQUANTIZE → NEG/MEAN → QUANTIZE`) — this is standard, expected TFLite behavior for LayerNorm (no native quantized mean-subtraction kernel), not a conversion bug, and adds negligible overhead.
- **I/O dtype**: `convert_to_tflite.py --io-type int8` (default) gives fully integer input/output, best for microcontroller-style RPi deployment. `--io-type float32` keeps float32 in/out with int8 math internally, if you'd rather feed raw float mel arrays without manual quantization.
- **CPU-only training, deliberately**: `train_tf.py` force-disables the GPU (`tf.config.set_visible_devices([], "GPU")`) and the project does not depend on `tensorflow-metal`. Apple's Metal GPU PluggableDevice was tried first (it does run, ~2.9x faster per step) but was found to compute numerically **wrong** gradients for this model: with the identical forward pass and the same random init, CPU gives a normal global gradient norm (~2.5) while Metal gives ~1e5–1e6, with some individual weight-gradient norms in the *billions* — inconsistent with the (small, correct) activation gradients feeding them by many orders of magnitude, which is not "instability," it's an incorrect backward pass. That corruption is exactly what caused the `loss nan` / `acc 0.0000` you'd see a few epochs in. Revisit GPU once Apple ships a `tensorflow-metal` fix for this op combination (batched multi-head attention + causal masking + LayerNorm is the likely culprit, though the exact faulty op wasn't isolated further).

### PyTorch → ONNX

- **Apple Silicon GPU (MPS) — checked for correctness, not just checked that it runs.** Given `tensorflow-metal`'s confirmed gradient-correctness bug on this exact architecture (see above), PyTorch's MPS backend was not trusted by default either. Two checks: (1) a single-batch CPU-vs-MPS gradient comparison on freshly-initialized weights showed a real per-element discrepancy — worst case ~104% relative difference, concentrated in `blocks.0.norm1.weight` (LayerNorm again, the same op category that broke on Metal/TF) — clearly more than float32 rounding noise, but with the overall gradient *global norm* staying close (16.31 CPU vs 16.28 MPS), unlike TF/Metal's 5-6-order-of-magnitude blowup; (2) what actually matters — a 120-step real-data training run on MPS — converged cleanly: loss 3.59→1.88, accuracy 0.8%→58.6%, zero NaN/Inf, and a 25-step CPU-vs-MPS comparison with identical init/data order tracked step-for-step within ~0.01-0.05 loss. Conclusion: real but non-fatal cross-backend numerical noise, not a correctness bug — training on MPS is enabled by default (`kws_common_torch.py`'s `DEVICE`). If a future PyTorch/macOS update ever produces `loss nan` here, force CPU first (same first move as the TF/Metal incident) and re-run the gradient comparison before assuming it's your data or model.
- **Causal mask magnitude is `-100`, not `-1e4` — applied proactively.** The TF side hit this for real (see above): a `-1e4` additive causal mask is fine in float32 but collapses the real attention-logit range into ~1 int8 quantization level once the mask-add op gets a single calibrated int8 range, tanking accuracy from 0.95 to 0.26. `kws_common_torch.py`'s `CausalSelfAttention` was fixed to `-100` before ever running real training here, to avoid hitting the identical failure once `train.py` + `convert_to_onnx.py` are actually run to completion.
- **`torchaudio.load` is avoided entirely.** Current `torchaudio` routes *all* file decoding through TorchCodec, which hard-requires a system FFmpeg install just to `import` — there's no fallback to the old soundfile/sox backends any more (confirmed directly: `torchaudio.load(path, backend="soundfile")` still raises, with `backend=` silently ignored). Rather than push a Homebrew FFmpeg install on you for reading plain 16-bit PCM WAV files, `kws_common_torch.py` reuses `kws_common.py`'s download/split logic and reads WAVs with the stdlib `wave` module instead. `torchaudio` is kept only for `MelSpectrogram`/`FrequencyMasking`/`TimeMasking`, which operate on in-memory tensors and never touch TorchCodec.
- **The NumPy Pi preprocessing was numerically verified, not assumed.** `run_onnx_pi.py`'s log-mel implementation reimplements torchaudio's `MelSpectrogram` (htk mel scale, `norm=None`, centered/reflect-padded framing) in plain NumPy. Compared directly against real `torchaudio.MelSpectrogram` output on 20 real validation clips: correlation ≥ 0.9999999999974, max absolute difference 0.0002 — i.e. float32 FFT rounding noise, not a preprocessing mismatch. This matters more here than on the TF side, because training (`kws_common_torch.py`) uses torchaudio's *real* MelSpectrogram, so a sloppy NumPy reimplementation for inference would create a train/inference feature mismatch — a real accuracy risk, not just a style choice.
- **`run_onnx_pi.py`'s "zero torch dependency" claim was verified, not assumed.** Ran it end-to-end against a real quantized model and real WAV file in a virtualenv containing *only* `numpy` + `onnxruntime` (no torch, torchaudio, or onnx) — it worked without modification.
- **Fixed shape, opset 18**: like the TFLite side, `convert_to_onnx.py` exports with a fixed `[1, 101, 80]` input (no dynamic axes) for a simpler, more calibration-friendly graph. `torch.onnx.export`'s current exporter needs `onnxscript` and will auto-upgrade a requested opset < 18 with a warning, hence the default of 18.
- **QDQ, int8/int8, per-channel weights**: `quantize_static(..., quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True)` — ONNX Runtime's documented default recommendation for broad hardware support (including ARM/Raspberry Pi), analogous to the TFLite side's int8/int8 setup. As with TFLite's LayerNorm islands, expect a few ops (e.g. around LayerNorm) to stay in float rather than get fully quantized — this is normal, not a bug, and the export step logs benign `Axis 1 is out-of-range for weight '...norm...' with rank 1` warnings for exactly this reason (LayerNorm's 1-D gamma isn't a per-channel-quantizable weight the way a Linear/Conv kernel is).
- **Verified locally** (this session, torch 2.13.0 / torchaudio 2.11.0 / onnxruntime 1.29.0 on Apple Silicon): model builds, forward pass and dataset loading both correct (train/val split sizes match the TF side exactly: 84,843 / 9,981), and the full export → quantize → run pipeline was run end-to-end (using a freshly-initialized, not yet trained, checkpoint purely to validate the mechanics — real accuracy still requires running `train.py` to completion).
