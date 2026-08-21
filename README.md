# TensorFlow port + TFLite static quantization pipeline

TensorFlow port of a PyTorch decoder-only Transformer keyword spotter, plus the pipeline to statically (int8) quantize it to TFLite and verify it on Mac before deploying to a Raspberry Pi.

| File | Purpose |
|---|---|
| [kws_common.py](kws_common.py) | Config, dataset download/pipeline, model architecture — shared by everything below so there's no drift between training and inference. |
| [train_tf.py](train_tf.py) | Trains the model (custom loop mirroring the PyTorch script: OneCycle LR, grad clipping, label smoothing, best-val checkpointing). |
| [convert_to_tflite.py](convert_to_tflite.py) | Post-training **static** (full-integer) quantization to `.tflite`, calibrated on real validation-set spectrograms. |
| [run_tflite_mac.py](run_tflite_mac.py) | Runs the quantized model on Mac — full validation sweep (accuracy + latency) or a single WAV smoke test — before touching the Pi. |

## Model architecture

The task: classify a 1-second, 16 kHz audio clip into one of 35 keywords. The model is a **decoder-only Transformer** (GPT-style — causal self-attention, not the bidirectional attention a BERT-style encoder uses), repurposed for classification by reading out a trailing `[CLS]` token instead of generating text.

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

## How to run

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

## Notes

- **Dataset reuse**: the TF loader downloads to `./data/SpeechCommands/speech_commands_v0.02/`, the same layout torchaudio uses — if you already ran the PyTorch script against `./data`, this reuses it instead of re-downloading ~2.4 GB.
- **Frame count**: the log-mel pipeline reflect-pads by `N_FFT//2` before framing so a 1 s clip always yields exactly `MAX_FRAMES=101` frames, matching torchaudio's `center=True` behavior. The window is `N_FFT`-length Hann rather than a zero-padded `WIN_LENGTH` window inside `N_FFT` (torch's exact approach) — numerically slightly different, functionally equivalent; not worth chasing bit-exactness for a from-scratch TF training run.
- **Verified locally** (this session, TF 2.21 on Apple Silicon): model builds and traces correctly, mel spectrogram yields shape `(101, 80)`, static int8 conversion succeeds using **only built-in TFLite ops** (`FULLY_CONNECTED`, `BATCH_MATMUL`, `SOFTMAX`, `GELU`, `MEAN`, …) — no Flex/Select-TF ops, no custom ops — so the same `.tflite` file will load with lightweight `tflite_runtime`/`ai-edge-litert` on the Raspberry Pi, not just the full `tensorflow` package.
- **LayerNorm quantization islands**: a handful of tiny float32 tensors remain around each `LayerNormalization`'s mean computation (`DEQUANTIZE → NEG/MEAN → QUANTIZE`) — this is standard, expected TFLite behavior for LayerNorm (no native quantized mean-subtraction kernel), not a conversion bug, and adds negligible overhead.
- **I/O dtype**: `convert_to_tflite.py --io-type int8` (default) gives fully integer input/output, best for microcontroller-style RPi deployment. `--io-type float32` keeps float32 in/out with int8 math internally, if you'd rather feed raw float mel arrays without manual quantization.
- **CPU-only training, deliberately**: `train_tf.py` force-disables the GPU (`tf.config.set_visible_devices([], "GPU")`) and the project does not depend on `tensorflow-metal`. Apple's Metal GPU PluggableDevice was tried first (it does run, ~2.9x faster per step) but was found to compute numerically **wrong** gradients for this model: with the identical forward pass and the same random init, CPU gives a normal global gradient norm (~2.5) while Metal gives ~1e5–1e6, with some individual weight-gradient norms in the *billions* — inconsistent with the (small, correct) activation gradients feeding them by many orders of magnitude, which is not "instability," it's an incorrect backward pass. That corruption is exactly what caused the `loss nan` / `acc 0.0000` you'd see a few epochs in. Revisit GPU once Apple ships a `tensorflow-metal` fix for this op combination (batched multi-head attention + causal masking + LayerNorm is the likely culprit, though the exact faulty op wasn't isolated further).
