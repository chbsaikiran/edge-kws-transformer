# ONNX Runtime as a Hardware Abstraction Layer

*How `session.run()` reaches a GPU/NPU/DSP without you writing driver code — and what it takes to build that bridge yourself for a new accelerator.*

---

## 1. The three layers

| Layer | What happens | Who writes it |
|---|---|---|
| Model graph | You define/export the network | You / the framework (PyTorch, TF) |
| Which device runs which node | Graph partitioning across devices | The runtime + Execution Provider |
| Kernel implementation, machine code, memory, DMA | Actual math on actual silicon | Vendor runtime/driver developer |

The mental model worth keeping, especially coming from DSP/embedded work, is not *"ONNX Runtime replaces DSP programming"* but:

```
┌─────────────────────────────────────┐
│ Python / C++ application             │
├─────────────────────────────────────┤
│ ONNX Runtime / TFLite                │  ← you usually work here
├─────────────────────────────────────┤
│ Execution Provider / Delegate        │
├─────────────────────────────────────┤
│ Vendor ML runtime                    │
├─────────────────────────────────────┤
│ Driver / OS                          │
├─────────────────────────────────────┤
│ GPU / NPU / DSP                      │  ← the vendor works here
├─────────────────────────────────────┤
│ Hardware memory / DMA / accelerator  │
└─────────────────────────────────────┘
```

ONNX Runtime is an **abstraction layer that sits on top of an existing hardware software stack**. It doesn't remove the stack — it hides it behind a uniform interface, as long as somebody has already built the bridge for that piece of hardware.

---

## 2. Traditional DSP programming (the layer you already know)

```
Your C/C++ code
      ↓
DSP compiler
      ↓
DSP machine code / binary
      ↓
Load binary into DSP memory
      ↓
DSP executes instructions
      ↓
You manage buffers / DMA / memory
```

Writing something like an FIR filter for a DSP means dealing directly with DSP intrinsics, DSP memory regions, DMA, cache, alignment, shared memory, and synchronization. This is the layer ONNX Runtime is built *on top of* — it doesn't replace it, it wraps it.

---

## 3. ML inference with ONNX Runtime — you don't usually touch the layer above

```python
import onnxruntime as ort

session = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
output = session.run(None, {"input": x})
```

Nothing here writes GPU machine code or copies a binary into device memory by hand. ONNX Runtime does this through **Execution Providers (EPs)**: an EP declares which ops/subgraphs it can run, and the runtime partitions the graph across whichever EPs are registered, in priority order.

```
                model.onnx
                    │
                    ▼
          ┌──────────────────┐
          │  ONNX Runtime    │
          └────────┬─────────┘
                    │
              graph analysis
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     GPU-supported        CPU-supported
        nodes                nodes
          │                   │
          ▼                   ▼
     GPU Execution        CPU Execution
       Provider             Provider
```

Example partition:

```
Conv1 ───────► GPU
ReLU ────────► GPU
Conv2 ───────► GPU
CustomOp ────► CPU   (no GPU kernel registered for it)
Softmax ─────► GPU
```

You never write that mapping by hand — the EP's `GetCapability()` call and the runtime's fallback logic decide it for you.

---

## 4. `pip install onnxruntime-gpu` doesn't install hardware — it installs the bridge

```
pip install
     │
     ▼
Python package
     │
     ├── ONNX Runtime core
     ├── GPU Execution Provider
     └── supporting libraries
             │
             ▼
       GPU driver/runtime (already on your system)
             │
             ▼
            GPU
```

`providers=["CUDAExecutionProvider", "CPUExecutionProvider"]` is a **priority list** — CUDA gets first refusal on every node; anything it can't run falls back to CPU. On mobile/embedded targets the same pattern shows up as NNAPI, CoreML, QNN, or OpenVINO EPs, each wrapping a vendor's own runtime for CPU/GPU/NPU.

Memory movement follows the same rule: you don't write the `cudaMemcpy` calls. The EP copies inputs to device memory on `Run()` if they aren't already there, and copies outputs back to host memory afterward — unless you opt into **I/O Binding** to keep tensors resident on the device and skip the copies.

### Who handles what

| Concern | Handled by |
|---|---|
| Model graph | You / framework |
| Node → device assignment | Runtime + Execution Provider |
| Kernel implementation | Vendor/runtime developer |
| GPU/NPU machine code | Compiler / vendor runtime |
| Device memory allocation | Runtime / EP |
| Host ↔ device transfers | Runtime / EP (or you, via I/O Binding) |
| DMA / cache details | Runtime / driver / vendor software |
| Hardware programming | Driver / vendor runtime |
| Hardware itself | CPU / GPU / NPU / DSP |

The exception: **if no EP exists for your hardware, none of this happens for free.** Someone — possibly you — has to build the bridge. That's the rest of this document.

---

## 5. Building the bridge yourself: an Execution Provider for a DSP

Take a concrete case that matches the kind of hardware this project already touches (`run_tflite_pi.py`, `run_tflite_mac.py` probe for a Hexagon `.so` delegate before falling back to XNNPACK): a **Qualcomm Hexagon-class DSP**. When no EP exists yet, or you're extending one, the stack you build looks like this, top to bottom:

```
┌───────────────────────────────────────────────────┐
│ Application (session.run())                        │
├───────────────────────────────────────────────────┤
│ ONNX Runtime core                                   │
├───────────────────────────────────────────────────┤
│ ★ HexagonExecutionProvider  (you write this)        │
│   - GetCapability(): which nodes it claims          │
│   - Compile(): builds a callable per fused subgraph │
├───────────────────────────────────────────────────┤
│ ★ Host-side stub  (generated + you write glue)       │
│   libdsp_stub.so — marshals calls over FastRPC      │
├───────────────────────────────────────────────────┤
│ adsprpc kernel driver (/dev/adsprpc)  — vendor-owned │
│   FastRPC transport, shared memory (ION), IPC        │
├───────────────────────────────────────────────────┤
│ ★ Device-side skel  (generated + you write kernels)  │
│   libdsp_skel.so running under QuRT on the DSP       │
│   Conv2D / MatMul / Relu kernels using HVX intrinsics│
├───────────────────────────────────────────────────┤
│ Hexagon DSP hardware (HVX vector unit, TCM, DMA)     │
└───────────────────────────────────────────────────┘
```

The ★ pieces are what *you* build. Everything else — the driver, the RTOS, the FastRPC transport, the toolchain — is vendor-supplied infrastructure you build against, the same relationship your existing DSP work already has with a vendor SDK.

### Step 1 — Driver & transport (usually already there)

On Qualcomm platforms this is `adsprpc`, the kernel driver exposing `/dev/adsprpc`, plus `libadsprpc.so` in userspace. It gives you:

- `remote_handle64_open()` / `remote_handle64_invoke()` / `remote_handle64_close()` — open a session with a DSP-side library and call into it
- Shared memory (ION-backed buffers) so host and DSP can see the same physical memory instead of copying through FastRPC for every call

You don't write this layer. You consume its API.

### Step 2 — Define the interface (IDL) and generate stubs

Qualcomm's Hexagon SDK uses an IDL compiler (`qaic`) that reads an interface file and generates matching host (stub) and device (skel) C code:

```idl
// dsp_ops.idl
interface dsp_ops {
  long conv2d(in uint8_t[] input, in uint8_t[] weights,
              in int32_t[] bias, rout uint8_t[] output,
              in int32_t params[8]);
  long matmul(in uint8_t[] a, in uint8_t[] b, rout uint8_t[] out,
              in int32_t m, in int32_t n, in int32_t k);
};
```

```bash
qaic dsp_ops.idl --stub-only   # → dsp_ops_stub.c  (links into the host .so)
qaic dsp_ops.idl --skel-only   # → dsp_ops_skel.c  (links into the device .so)
```

### Step 3 — Device-side kernels (`libdsp_skel.so`, runs under QuRT on the DSP)

This is where your DSP/DMA/vectorization background does the real work — writing the actual quantized-int8 Conv2D/MatMul kernels using HVX intrinsics, managing TCM (tightly-coupled memory) and DMA transfers explicitly:

```c
// conv2d_hvx.c — compiled with hexagon-clang, linked into libdsp_skel.so
#include "hexagon_types.h"

int conv2d(const uint8_t *input, const uint8_t *weights,
           const int32_t *bias, uint8_t *output, const int32_t params[8]) {
    // params: {in_h, in_w, in_c, out_c, k_h, k_w, stride, pad}
    HVX_Vector *in_vp  = (HVX_Vector *)input;
    HVX_Vector *w_vp   = (HVX_Vector *)weights;

    // DMA the working tile into TCM before compute, per output row
    hexagon_dma_start(input, tcm_buffer, tile_bytes);
    hexagon_dma_wait();

    // 128-byte-wide HVX vector MACs over the quantized tile
    for (int oc = 0; oc < params[3]; oc += 32) {
        HVX_Vector acc = Q6_V_vzero();
        acc = Q6_Vuw_vmpyacc_VubVub(acc, in_vp[0], w_vp[oc]);
        // ... accumulate across kernel window, requantize, store
    }
    return 0; // QRT/AEE error code in real code
}
```

This is exactly the layer described in §2 — DSP compiler, DSP memory, DMA, intrinsics — except it now exposes a fixed entry point (`conv2d`) that FastRPC can call remotely instead of running as a standalone firmware image.

### Step 4 — Host-side runtime wrapper (`libdsp_stub.so`)

A thin C++ wrapper around the generated stub, opening the remote handle once and reusing it:

```cpp
// dsp_runtime.h
class DspRuntime {
 public:
  bool Open() {
    return dsp_ops_open("file:///libdsp_skel.so?entry&_modver=1.0",
                         &handle_) == 0;
  }
  bool Conv2D(const uint8_t* in, const uint8_t* w, const int32_t* bias,
              uint8_t* out, const int32_t params[8]) {
    return dsp_ops_conv2d(handle_, in, in_len_, w, w_len_,
                           bias, bias_len_, out, out_len_, params, 8) == 0;
  }
 private:
  remote_handle64 handle_;
};
```

This is the layer that, in the mainstream stack, ships as a "vendor ML runtime" (e.g. Qualcomm's QNN or SNPE) — the thing an Execution Provider links against instead of talking to FastRPC directly. Building your own means you're writing a minimal version of that.

### Step 5 — The Execution Provider itself

ONNX Runtime's EP interface (`onnxruntime/core/framework/execution_provider.h`) needs two things at minimum: which nodes you claim, and what to do with them.

```cpp
// hexagon_execution_provider.cc
class HexagonExecutionProvider : public IExecutionProvider {
 public:
  HexagonExecutionProvider() : IExecutionProvider{"HexagonExecutionProvider"} {
    dsp_.Open();
  }

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const GraphViewer& graph, const IKernelLookup&) const override {
    std::vector<std::unique_ptr<ComputeCapability>> result;
    for (const auto& node : graph.Nodes()) {
      if (IsSupported(node)) {                 // op type + dtype + shape checks
        auto sub = std::make_unique<IndexedSubGraph>();
        sub->nodes.push_back(node.Index());
        result.push_back(std::make_unique<ComputeCapability>(std::move(sub)));
      }
    }
    return result;                              // unclaimed nodes fall back to CPU EP
  }

  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused,
                          std::vector<NodeComputeInfo>& node_compute_funcs) override {
    for (const auto& node_graph : fused) {
      NodeComputeInfo info;
      info.compute_func = [this](FunctionState, const OrtApi*, OrtKernelContext* ctx) {
        // pull input tensors, call dsp_.Conv2D(...)/MatMul(...), write outputs
        return Status::OK();
      };
      node_compute_funcs.push_back(std::move(info));
    }
    return Status::OK();
  }

 private:
  bool IsSupported(const Node& node) const {
    static const std::unordered_set<std::string> kOps = {"Conv", "MatMul", "Relu"};
    return kOps.count(node.OpType()) > 0;       // real EPs also check dtype/quant params
  }
  DspRuntime dsp_;
};
```

`GetCapability()` claims the nodes your device-side kernels actually cover (Step 3). `Compile()` produces a closure ORT calls at inference time, which routes tensors through the host stub (Step 4) into the DSP kernels.

### Step 6 — Register it and use it exactly like a built-in EP

```cpp
// registration (C++ side, or exposed via a small C API for Python)
Ort::SessionOptions so;
so.AppendExecutionProvider("Hexagon", /* provider options */ {});
Ort::Session session(env, "model.onnx", so);
```

```python
# once wrapped, it's indistinguishable from a stock provider
session = ort.InferenceSession(
    "model.onnx",
    providers=["HexagonExecutionProvider", "CPUExecutionProvider"],
)
```

From here the runtime behaves exactly as in §3–4: it partitions the graph, routes Conv/MatMul/Relu to Hexagon, falls back the rest to CPU, and (unless you wire up I/O binding) copies tensors across the FastRPC boundary automatically via your `Compile()` closure.

### What's genuinely hard here

The mechanical wiring above is the easy 20%. The real cost is:

- **Op & dtype coverage** — every ONNX op your model uses needs a matching HVX kernel, correctly matching ONNX's quantization semantics (zero-point, per-tensor vs per-channel scale) or you'll silently diverge from the CPU reference output.
- **Graph partitioning correctness** — `GetCapability()` must not claim a node whose *neighbors* it can't also efficiently handle, or you pay FastRPC round-trip latency for every tiny fragment (this is exactly the CoreML "42-way partition" overhead noted in this project's `run_onnx_mac.py`/`run_tflite_mac.py` work — more partitions means more cross-device handoffs, which can beat a device with lower nominal FLOPs).
- **Memory strategy** — naive per-call FastRPC copies are slow; production EPs pre-allocate ION/shared buffers and reuse them across `Run()` calls.
- **Numerical validation** — every kernel needs to be checked against the CPU EP's output on real data, not just unit shapes.

---

## 6. Takeaway

- ONNX Runtime doesn't replace DSP/driver work — it's a **hardware abstraction layer** that lets application code stay hardware-agnostic *once* someone has written the Execution Provider, the vendor runtime, and the driver underneath it.
- For mainstream hardware (CUDA, CoreML, NNAPI, OpenVINO, QNN) that bridge already exists — `pip install` and a `providers=[...]` list is enough.
- For a new or proprietary DSP/NPU, the low-level skills (kernel writing, DMA, memory layout, cross-compilation, RTOS-side code) are still exactly what's needed — they just get packaged behind `GetCapability()` / `Compile()` so the rest of the stack can treat the device like any other provider.

---

### References

- ONNX Runtime, [Execution Providers overview](https://onnxruntime.ai/docs/execution-providers/)
- ONNX Runtime, [Android NNAPI Execution Provider](https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html)
- ONNX Runtime, [I/O Binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)
- ONNX Runtime, [Intel OpenVINO Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)
- ONNX Runtime, [Add a new execution provider](https://onnxruntime.ai/docs/execution-providers/add-execution-provider.html)

*Note on §5: the IDL/FastRPC/HVX code samples are illustrative of the real Hexagon SDK pattern (`qaic`, `remote_handle64_*`, HVX intrinsics, `adsprpc`) to show the shape of the bridge — not copy-pasteable against a specific SDK version, since exact signatures vary by Hexagon SDK release and are under Qualcomm's own documentation/licensing.*
