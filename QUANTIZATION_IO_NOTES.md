# Static Quantization I/O: TFLite vs ONNX — Interview Notes

Grounded in a real project: a decoder-only Transformer keyword spotter, statically
quantized to int8 two ways (TFLite converter, ONNX Runtime `quantize_static`).
All numbers below are real values pulled from the actual converted models, not
made up for illustration.

## TL;DR

- **Calibration** (offline, run once at conversion time) is what *computes*
  `(scale, zero_point)` for every tensor that gets quantized — internal
  activations, weights, **and** the model's input/output boundary. Same
  process, no special case for the boundary.
- **Inference** (every `session.run()` / `interpreter.invoke()` call) never
  computes `(scale, zero_point)` — it only *applies* the fixed pair that
  calibration already determined. Quantize and Dequantize are just fixed
  affine formulas at that point, not adaptive/learned per-call.
- The only real difference between "TFLite int8 I/O" and "ONNX QDQ float32
  I/O" is *where* that fixed `(scale, zero_point)` pair physically lives:
  **baked into a graph node** (executes automatically) vs **exposed as
  tensor metadata** (caller must replicate the same math in Python).

---

## 1. What calibration actually computes

Post-training static quantization needs to know the real value range every
tensor takes on, so it can pick an 8-bit grid that covers it. That's what a
representative/calibration dataset is for — you feed it real (unaugmented)
inputs, and the calibrator watches the min/max (or a more advanced statistic,
depending on `CalibrationMethod`) each tensor actually reaches.

For standard asymmetric (affine) int8 quantization, given an observed range
`[rmin, rmax]`:

```
scale      = (rmax - rmin) / (qmax - qmin)      # qmax-qmin = 255 for int8
zero_point = round(qmin - rmin / scale)          # clipped into [-128, 127]
```

This runs **once**, inside the conversion script:

```python
# convert_to_tflite.py — representative_dataset feeds real calibration data
def representative_dataset_gen(pairs, num_samples):
    for path, _label in pairs[:num_samples]:
        mel = kws.load_and_process(path, augment=False).numpy()
        yield [mel[np.newaxis, ...].astype(np.float32)]

converter.representative_dataset = rep_gen   # TFLiteConverter calibrates using this
```

```python
# convert_to_onnx.py — same idea, ONNX Runtime's CalibrationDataReader
class MelCalibrationDataReader(CalibrationDataReader):
    def get_next(self):
        mel, _label = self.val_ds[self.idx]
        return {self.input_name: mel.unsqueeze(0).numpy().astype(np.float32)}

quantize_static(..., calibration_data_reader=reader, ...)
```

The result — a `(scale, zero_point)` pair per quantized tensor — gets written
into the model file as fixed constants. **Nothing about them changes at
inference time**, no matter how many times or what data you run through the
model afterward.

---

## 2. The actual math: QuantizeLinear / DequantizeLinear

Both TFLite and ONNX use the identical affine scheme, just expressed
differently (tensor metadata vs. explicit graph ops):

```
Quantize:    q = clip(round(x / scale) + zero_point, qmin, qmax)
Dequantize:  x' = (q - zero_point) * scale
```

Real example, using this project's actual calibrated constants
(`scale = 0.03300607576966286`, `zero_point = -29`, both int8):

```python
>>> x = 0.5
>>> q = round(x / scale) + zero_point
>>> q = clip(q, -128, 127)          # = -14
>>> x_reconstructed = (q - zero_point) * scale
>>> x_reconstructed                  # = 0.495091
>>> abs(x - x_reconstructed)         # = 0.004909  (bounded by scale/2 = 0.0165)
```

`x' != x` — quantization snaps every value to the nearest of 256 evenly
spaced grid points, `scale` apart. That rounding error is the actual "cost"
of static quantization; it's why a quantized model's accuracy is close to,
but not identical to, the float32 original.

---

## 3. TFLite: the converter can strip the boundary Q/DQ, or keep it

```python
# convert_to_tflite.py
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
if args.io_type == "int8":
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
```

The converter *always* calibrates a boundary Quantize/Dequantize pair — these
two lines only decide what happens to it:

| `inference_input_type` | Boundary Q/DQ | Model accepts | Who quantizes? |
|---|---|---|---|
| `tf.float32` (default) | kept as a real graph node | float32 | the graph, automatically |
| `tf.int8` | **removed**, `(scale, zp)` exposed as tensor metadata | int8 | **the caller, in Python** |

With `tf.int8`, the caller has to do by hand exactly what the deleted node
used to do:

```python
# run_tflite_mac.py
def quantize(x_float, quant_params, dtype):
    scale, zero_point = quant_params
    if dtype in (np.int8, np.uint8) and scale != 0:
        q = np.round(x_float / scale + zero_point)
        info = np.iinfo(dtype)
        return np.clip(q, info.min, info.max).astype(dtype)
    return x_float.astype(dtype)          # float32 model -> no-op passthrough

def dequantize(x_q, quant_params, dtype):
    scale, zero_point = quant_params
    if dtype in (np.int8, np.uint8) and scale != 0:
        return (x_q.astype(np.float32) - zero_point) * scale
    return x_q.astype(np.float32)

# The (scale, zero_point) are read straight off the model's own metadata --
# NOT recomputed, just the same numbers calibration already produced:
in_det = interpreter.get_input_details()[0]
print(in_det["quantization"])   # (0.03343628719449043, -26) -- real, from this project
```

---

## 4. ONNX (QDQ format): the boundary Q/DQ always lives *inside* the graph

```python
# convert_to_onnx.py
quantize_static(
    model_input=preprocessed_path, model_output=args.out,
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,          # <- inserts QuantizeLinear/DequantizeLinear NODES
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
)
```

Key fact: **`quant_format` never changes `graph.input`/`graph.output`'s
dtype** — there's no ONNX equivalent of `inference_input_type`. The graph
keeps whatever dtype `torch.onnx.export` gave it (float32), and QDQ instead
inserts explicit node pairs *around* quantized ops throughout the graph.

Tracing the real exported graph (`onnx.load(...)`, following node
producer/consumer edges from the input) shows this directly:

```
mel_input (FLOAT)
  --[Reshape]-->  gemm_input_reshape_arg
  --[QuantizeLinear, scale=0.033006, zero_point=-29]-->  (INT8)
  --[DequantizeLinear]-->  (FLOAT, reconstructed w/ rounding error)
  --[Gemm]-->  ...   # input_proj Linear layer runs on the reconstructed float
```

and symmetrically at the output: the graph's *last* node is a
`DequantizeLinear` (`scale=0.034553`, `zero_point=-63`) converting the
int8-computed logits back to float32 for the declared output. This is why
`run_onnx_mac.py` never calls a `quantize()`/`dequantize()` function anywhere
— that logic isn't missing, it's living inside the graph as real nodes that
execute automatically on every `session.run()`:

```python
# run_onnx_mac.py — no manual quantization needed
sess = ort.InferenceSession("keyword_spotting_int8.onnx")
logits = sess.run(["logits"], {"mel_input": mel_float32})[0]   # float in, float out
```

(One subtlety: the graph *as saved* still shows Reshape→Q→DQ→Gemm, i.e. Gemm
nominally runs on reconstructed float32 values. Whether ONNX Runtime actually
fuses `Q→Op→DQ` into a genuine int8 kernel at runtime is a session-load-time
optimization decision by the execution provider, not something visible in the
`.onnx` file itself.)

---

## 5. Making each one do what the other does by default

**TFLite → float32 I/O** (trivial, already a flag):
```bash
python convert_to_tflite.py --io-type float32
```
Skips the two `inference_*_type = tf.int8` lines, so the boundary Q/DQ stays
in the graph — structurally identical to ONNX's QDQ shape. No caller code
changes needed: `quantize()`/`dequantize()` already branch on `dtype`, so
they fall through to a no-op cast for a float32-I/O model.

**ONNX → int8 I/O** (nontrivial — no built-in flag, requires graph surgery):
`onnxruntime.quantization` has nothing equivalent to `inference_input_type`.
`QuantFormat.QOperator` vs `QuantFormat.QDQ` changes internal op fusion, not
the graph's declared I/O dtype. To get genuine int8 boundary tensors, you
have to manually delete the boundary Quantize/Dequantize nodes and rewire —
verified working against this project's real model:

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

# Input side: delete the boundary QuantizeLinear, rewire its consumers to its
# own input tensor, declare the graph input itself as int8.
q_node = next(n for n in graph.node if n.name == "gemm_input_reshape_arg_QuantizeLinear")
in_scale, in_zero_point = get_scale_zp(q_node)
produced, consumed = q_node.output[0], q_node.input[0]
for n in graph.node:
    n.input[:] = [consumed if x == produced else x for x in n.input]
graph.node.remove(q_node)
graph.input[0].type.tensor_type.elem_type = TensorProto.INT8

# Output side: delete the final DequantizeLinear, make the graph output the
# int8 tensor that used to feed it.
dq_node = next(n for n in graph.node if n.name == "logits_DequantizeLinear")
out_scale, out_zero_point = get_scale_zp(dq_node)
graph.output[0].name = dq_node.input[0]
graph.node.remove(dq_node)
graph.output[0].type.tensor_type.elem_type = TensorProto.INT8

# Stale value_info entries (float32, from before this edit) fail ONNX
# Runtime's loader even though onnx.checker.check_model() doesn't catch it --
# confirmed: "Type Error: ... does not match expected type (tensor(int8))".
del graph.value_info[:]
model = shape_inference.infer_shapes(model)
onnx.checker.check_model(model)
onnx.save(model, "keyword_spotting_int8_intio.onnx")
```

Verified end to end: the surgically-edited model loads with genuine
`tensor(int8)` input/output, and produces predictions matching the original
float-I/O model on 20/20 real validation clips (using the extracted
`in_scale`/`in_zero_point`/`out_scale`/`out_zero_point` to quantize/dequantize
manually — the exact same `quantize()`/`dequantize()` functions already
written for the TFLite side work here verbatim, since both formats use the
identical affine int8 scheme).

---

## 6. Quick comparison table

| | TFLite (`--io-type int8`) | ONNX (QDQ, default) |
|---|---|---|
| Declared model I/O dtype | int8 | float32 |
| Boundary quantize/dequantize | Python code, caller-side | graph nodes, automatic |
| `(scale, zero_point)` source | calibration (same either way) | calibration (same either way) |
| `(scale, zero_point)` location | tensor metadata (`get_input_details()['quantization']`) | node initializer constants |
| To flip the default | `--io-type float32` flag (built-in) | manual graph surgery (no built-in flag) |
| Math when flipped | identical affine formula either way | identical affine formula either way |

## 7. Likely interview follow-ups

- **"Does Quantize 'learn' the scale?"** No — that's the calibration step,
  and it's a statistical observation (min/max or similar) over a
  representative dataset, done once, offline. Static quantization has no
  learned/adaptive component at inference time (that would be closer to
  quantization-aware training, a different technique).
- **"Why does accuracy sometimes collapse after quantization?"** A specific,
  real example from this project: a causal attention mask implemented as an
  additive `-1e4` constant is invisible in float32, but forces the
  mask-add op's calibrated int8 range to span both the mask and the real
  logits (`~[-10000, +27]` instead of `~[-27, +27]`), collapsing ~39 real
  logit values worth of precision into ~1 quantization level. Fixed by using
  a smaller-magnitude mask (`-100`) that's still far past the real logit
  range. Moral: quantization range is determined by the *widest* value that
  flows through a tensor, including constants, not just "real" signal.
- **"Is Q→DQ ever a no-op?"** No — `round()` always introduces up to
  `±scale/2` error unless the original value already sat exactly on the
  quantization grid.
- **"Why would you want float32 I/O with int8 internals at all?"** Easier to
  feed raw data without writing quantize/dequantize code yourself, at the
  cost of a tiny bit of host-side (Python-side, if TFLite) conversion
  overhead per call. Int8 I/O is preferred for microcontroller-class targets
  where you may not want any float32 math anywhere, including at the
  boundary.
