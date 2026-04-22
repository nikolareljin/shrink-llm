# ShrinkLLM — Architecture Blueprint

---

## 1. PROJECT OVERVIEW

### Problem Definition

Production AI models (vision transformers, large language models, audio classifiers) achieve state-of-the-art accuracy but require gigabytes of RAM and powerful GPUs. Smartphones — the primary computing device for 6+ billion people — have strict constraints:

- 4–8 GB shared RAM (2–4 GB available to apps)
- ARM Cortex-A CPUs at 2–3 GHz
- Dedicated NPUs (Apple Neural Engine, Qualcomm Hexagon, Google Edge TPU)
- 5–10 W power envelope

ShrinkLLM provides a reproducible, modular pipeline to compress any capable model and deploy it on Android/iOS without sacrificing acceptable accuracy.

### Target Use Cases

| Use Case | Description | Accuracy Target | Latency Target |
|---|---|---|---|
| OCR | Extract text from photos of documents, receipts, handwritten notes | ≥ 95% CER | < 200 ms/page |
| Legal Document Reasoning | Flag risky clauses, summarize contracts, answer legal questions | ≥ 85% F1 | < 2 s/doc |
| Audio Classification | Classify cry type (hunger, pain, discomfort) from 5-second audio clip | ≥ 90% accuracy | < 100 ms |

### Smartphone Hardware Constraints

| Constraint | Android (Mid-range) | iOS (iPhone 12+) |
|---|---|---|
| RAM available to app | 2–3 GB | 2–4 GB |
| Model max size (recommended) | 200 MB | 200 MB |
| Compute | Snapdragon 7xx NPU | Apple Neural Engine |
| Supported runtimes | TFLite, ONNX Runtime, GGUF | CoreML, ONNX Runtime |
| INT8 acceleration | Yes (via NNAPI/Hexagon) | Yes (via ANE) |
| INT4 support | Partial (GGUF via llama.cpp) | Partial (CoreML) |
| Battery budget | < 5% per inference session | < 5% per inference session |

### Success Criteria & KPIs

| Metric | Target |
|---|---|
| Model size reduction | ≥ 70% vs teacher |
| Accuracy drop (vs teacher) | ≤ 5% absolute |
| Inference latency (cold start) | < 500 ms |
| Inference latency (warm) | < 200 ms |
| RAM usage at inference | < 512 MB |
| Battery drain per 100 inferences | < 2% |
| ONNX export success rate | 100% |
| TFLite / CoreML conversion success | 100% |

---

## 2. MODEL SELECTION

### 2.1 OCR

| Role | Model | Size | Why |
|---|---|---|---|
| Teacher | microsoft/trocr-large-printed | ~1.3 GB | Best-in-class printed OCR; encoder-decoder ViT + GPT2 |
| Teacher | microsoft/trocr-large-handwritten | ~1.3 GB | Covers handwritten text |
| Student | microsoft/trocr-base-printed | ~340 MB | Same architecture, 4× smaller |
| Student | microsoft/trocr-small-printed | ~90 MB | Aggressive size target |
| Student | apple/mobilevit-small + CTC head | ~22 MB | Fully mobile-native encoder |

**Tradeoffs**: TrOCR-small loses ~3% CER vs large on IIIT-5k. MobileViT-CTC loses ~5% but fits in 22 MB.

### 2.2 Legal Document Reasoning

| Role | Model | Size | Why |
|---|---|---|---|
| Teacher | mistralai/Mistral-7B-Instruct-v0.3 | ~14 GB | Strong instruction following + legal reasoning |
| Teacher | microsoft/Phi-3-medium-4k-instruct | ~7.6 GB | Smaller but highly capable for long documents |
| Student | microsoft/Phi-3-mini-4k-instruct | ~2.3 GB (FP16) | Best small model for reasoning tasks |
| Student | google/gemma-2b-it | ~5 GB (FP16) → ~1.3 GB (INT4) | Strong instruction model, quantizes well |
| Student | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | ~1.1 GB | Extreme size target, lower accuracy |

**Tradeoffs**: Phi-3-mini at INT4 (~700 MB) achieves ~82% F1 on CUAD legal benchmark vs ~91% for Mistral-7B.

### 2.3 Audio Classification

| Role | Model | Size | Why |
|---|---|---|---|
| Teacher | facebook/wav2vec2-large-960h | ~1.2 GB | Best audio feature extractor |
| Teacher | openai/whisper-medium | ~769 MB | Robust audio encoder |
| Student | google/mobilenet_v3_small (mel-spec) | ~2.5 MB | Tiny CNN on mel spectrogram |
| Student | facebook/wav2vec2-base | ~360 MB → ~90 MB INT8 | Balanced teacher-student gap |
| Student | MIT/ast-finetuned-audioset-10-10-0.4593 | ~86 MB | Audio Spectrogram Transformer, prunable |

**Tradeoffs**: MobileNet-v3 on mel spectrograms reaches ~88% accuracy at 2.5 MB. AST-pruned reaches ~92% at 12 MB.

---

## 3. REPOSITORY STRUCTURE

```
shrink-llm/
├── models/
│   ├── teacher/                    # Downloaded/cached teacher models
│   │   └── .gitkeep
│   └── student/                    # Output compressed models
│       └── .gitkeep
│
├── compression/
│   ├── __init__.py
│   ├── quantization/
│   │   ├── __init__.py
│   │   ├── onnx_quantizer.py       # ONNX quantization (onnxruntime)
│   │   ├── torch_quantizer.py      # PyTorch static/dynamic quant
│   │   └── gptq_quantizer.py       # GPTQ 4-bit for LLMs
│   ├── pruning/
│   │   ├── __init__.py
│   │   ├── attention_pruner.py     # Head importance + removal
│   │   ├── mlp_pruner.py           # FFN block pruning
│   │   └── magnitude_pruner.py     # Weight magnitude pruning
│   └── distillation/
│       ├── __init__.py
│       ├── trainer.py              # KD training loop
│       ├── losses.py               # KL divergence, MSE, cosine losses
│       └── callbacks.py            # Checkpointing, early stopping
│
├── datasets/
│   ├── __init__.py
│   ├── ocr/
│   │   ├── download.py             # IIIT-5k, IAM, FUNSD downloaders
│   │   └── preprocess.py
│   ├── legal/
│   │   ├── download.py             # CUAD, ContractNLI
│   │   └── preprocess.py
│   └── audio/
│       ├── download.py             # Donate-a-cry corpus
│       └── preprocess.py
│
├── mobile_deployment/
│   ├── android/
│   │   ├── tflite_packager.py      # TFLite model packaging
│   │   ├── onnx_mobile_packager.py
│   │   └── README.md
│   ├── ios/
│   │   ├── coreml_packager.py      # CoreML .mlpackage builder
│   │   └── README.md
│   └── onnx_mobile/
│       ├── optimizer.py            # ONNX graph optimizations
│       └── README.md
│
├── scripts/
│   ├── export_to_onnx.py
│   ├── quantize.py
│   ├── prune.py
│   ├── distill.py
│   ├── benchmark.py
│   ├── convert_to_tflite.py
│   ├── convert_to_coreml.py
│   ├── convert_to_onnx_mobile.py
│   └── run_pipeline.py             # Orchestrates full pipeline
│
├── benchmarks/
│   ├── runner.py                   # Device + accuracy benchmarks
│   ├── metrics.py                  # CER, F1, accuracy, latency
│   ├── datasets/                   # Small eval subsets
│   └── results/                    # JSON + Markdown outputs
│       └── .gitkeep
│
├── configs/
│   ├── ocr_pipeline.yaml
│   ├── legal_pipeline.yaml
│   └── audio_pipeline.yaml
│
├── notebooks/
│   ├── 01_model_exploration.ipynb
│   ├── 02_compression_analysis.ipynb
│   └── 03_benchmark_visualization.ipynb
│
├── tests/
│   ├── test_quantization.py
│   ├── test_pruning.py
│   ├── test_distillation.py
│   ├── test_export.py
│   └── test_benchmarks.py
│
├── docs/
│   ├── architecture.md             # This file
│   ├── compression_pipeline.md
│   ├── mobile_deployment.md
│   ├── benchmarking.md
│   ├── roadmap.md
│   └── contributing.md
│
├── pyproject.toml
├── AGENTS.md
├── README.md
├── LICENSE
└── .gitignore
```

---

## 4. FULL COMPRESSION PIPELINE

### Stage A: ONNX Export

**Goal**: Produce a portable, framework-agnostic graph that all downstream tools can consume.

**Algorithm**:
1. Load model in PyTorch (HuggingFace `AutoModel`)
2. Create dummy input tensors matching model's expected input shape
3. Call `torch.onnx.export()` with `opset_version=17`, `dynamic_axes` for variable-length inputs
4. Run `onnx.checker.check_model()` and `onnxruntime` inference validation

**Expected improvements**: None in size yet; this is the foundation.

**Risks**:
- Custom ops not supported by ONNX → use `custom_opsets` or rewrite ops
- Dynamic control flow (loops in decoders) → use `torch.jit.script` first
- Attention mask shapes → set `dynamic_axes` carefully

**Mitigation**: Always validate exported ONNX with a sample input before proceeding.

---

### Stage B: Quantization

**Goal**: Reduce model weight precision to shrink size and accelerate inference on NPU.

#### B1. Dynamic INT8 Quantization (fastest, CPU-friendly)
- Weights quantized offline; activations quantized at runtime
- Tools: `onnxruntime.quantization.quantize_dynamic`
- Size reduction: ~4× (FP32 → INT8)
- Accuracy drop: < 1% for encoder models, 1–3% for LLMs

#### B2. Static INT8 Quantization (best accuracy/speed tradeoff)
- Requires calibration dataset (100–1000 samples)
- Tools: `onnxruntime.quantization.quantize_static` with `CalibrationDataReader`
- Size reduction: ~4×
- Accuracy drop: < 0.5%

#### B3. INT4 GPTQ (for LLMs)
- Group-wise quantization with reconstruction error minimization
- Tools: `auto-gptq`, `bitsandbytes`
- Size reduction: ~8× (FP32 → INT4)
- Accuracy drop: 2–5% on reasoning benchmarks

#### B4. Mixed Precision
- First + last layers stay FP16; middle layers INT8/INT4
- Protects accuracy at critical layers
- Tools: `onnxruntime` `TensorrtExecutionProvider` or custom layer-by-layer config

**Expected improvements**: 4–8× size reduction, 2–4× latency improvement on NPU.

**Risks**: Quantization of softmax/LayerNorm can cause NaN → use per-channel quantization and skip sensitive ops.

---

### Stage C: Structured Pruning

**Goal**: Remove entire attention heads and MLP blocks that contribute least to output quality, reducing computation permanently.

#### C1. Attention Head Pruning
1. Compute head importance scores (Taylor expansion or gradient × activation magnitude)
2. Rank heads globally across all layers
3. Zero out bottom-K% heads, retrain for 1–3 epochs
4. Tools: `nn_pruning`, custom `transformers` hooks

#### C2. MLP Block Pruning
1. Measure neuron activation frequency on calibration set
2. Remove neurons with < threshold activation rate
3. Reconstruct weight matrices without pruned neurons

#### C3. Layer Dropping
- For encoder models: drop middle transformer layers (layers 4–8 of 12 in BERT-style)
- For decoder LLMs: prune via DistilBERT-style even/odd layer selection

**Expected improvements**: 20–40% parameter reduction, 15–30% latency reduction.

**Risks**: Over-pruning causes irreversible accuracy collapse. Always prune iteratively with validation after each step.

---

### Stage D: Knowledge Distillation

**Goal**: Train a smaller student model to mimic the teacher's behavior, not just its outputs.

**Loss Function**:
```
L_total = α × L_CE(student_logits, labels)         # Hard labels
         + β × L_KL(student_logits, teacher_logits) # Soft labels (temperature T)
         + γ × L_MSE(student_hidden, teacher_hidden) # Intermediate representation
```

**Hyperparameters**:
- Temperature T = 4–8 (softer probability distributions transfer more knowledge)
- α = 0.1, β = 0.9 for pure distillation; adjust for labeled data availability
- Layers to match: every K-th teacher layer to each student layer

**Training**:
1. Freeze teacher, train student end-to-end
2. Use mixed dataset: original task data + unlabeled in-domain data
3. Learning rate warmup + cosine decay; AdamW optimizer

**Expected improvements**: Student reaches 90–95% of teacher accuracy at 30–50% of teacher size.

**Risks**: If teacher is too large relative to student (capacity gap), distillation fails. Solution: use intermediate teacher (teacher assistant) or progressive distillation.

---

### Stage E: Post-Compression Fine-tuning

**Goal**: Recover accuracy lost during quantization/pruning via short targeted training.

**Protocol**:
1. Load compressed (quantized + pruned) model
2. Fine-tune on task-specific labeled data for 2–5 epochs
3. Use QAT (Quantization-Aware Training) for INT8 if accuracy gap > 2%
4. Learning rate: 10× smaller than original training

**Tools**: HuggingFace `Trainer`, `torch.ao.quantization` for QAT

---

### Stage F: Validation + Regression Testing

**Goal**: Ensure no stage introduced regressions beyond acceptable thresholds.

**Tests per stage**:
- ONNX export: input/output shape match, numerical diff < 1e-4
- Quantization: accuracy delta ≤ 2%, latency improvement ≥ 2×
- Pruning: accuracy delta ≤ 3%, model size reduced ≥ 15%
- Distillation: student accuracy ≥ 90% of teacher
- Mobile conversion: on-device output matches ONNX output within tolerance

---

## 5. SCRIPTS SPECIFICATION

### `scripts/export_to_onnx.py`

**Purpose**: Export any HuggingFace model to ONNX format.

**CLI**:
```
python export_to_onnx.py \
  --model <hf_model_id_or_path> \
  --task <ocr|legal|audio|classification|seq2seq> \
  --output <path/to/output.onnx> \
  --opset 17 \
  --dynamic-axes           # enable variable-length inputs
  --validate               # run inference check post-export
  --device <cpu|cuda>
```

**Inputs**: HuggingFace model ID or local path
**Outputs**: `.onnx` file + `model_config.json` (input shapes, tokenizer info)

**Internal modules**:
- `load_model(model_id, task)` — loads model + tokenizer/processor
- `build_dummy_inputs(task, model_config)` — creates sample tensors
- `export(model, inputs, output_path, opset, dynamic_axes)` — wraps `torch.onnx.export`
- `validate_onnx(onnx_path, inputs)` — runs ORT and checks output shapes

---

### `scripts/quantize.py`

**Purpose**: Quantize an ONNX model to INT8 or INT4.

**CLI**:
```
python quantize.py \
  --input <model.onnx> \
  --output <model_quantized.onnx> \
  --precision <int8|int4|fp16|mixed> \
  --mode <dynamic|static|gptq> \
  --calibration-data <path/to/calib_dataset/> \  # required for static
  --calibration-samples 512 \
  --skip-ops <op1,op2>                           # ops to keep in FP32
```

**Internal modules**:
- `DynamicQuantizer` — wraps `onnxruntime.quantization.quantize_dynamic`
- `StaticQuantizer` — wraps `quantize_static` with custom `CalibrationDataReader`
- `GPTQQuantizer` — wraps `auto-gptq` for INT4 LLM quantization
- `MixedPrecisionQuantizer` — layer-by-layer precision assignment

---

### `scripts/prune.py`

**Purpose**: Apply structured pruning to a PyTorch model.

**CLI**:
```
python prune.py \
  --model <hf_model_id_or_path> \
  --task <ocr|legal|audio> \
  --method <attention_heads|mlp|layers|magnitude> \
  --sparsity 0.3 \                              # fraction to prune
  --calibration-data <path/> \
  --output-dir <models/student/pruned/> \
  --finetune-epochs 3 \
  --device <cpu|cuda>
```

**Internal modules**:
- `HeadImportanceScorer` — Taylor/gradient scoring of attention heads
- `MLPPruner` — removes low-activation neurons from FFN layers
- `LayerDropper` — removes entire transformer layers
- `PruningTrainer` — short fine-tune loop post-pruning

---

### `scripts/distill.py`

**Purpose**: Train a student model using knowledge distillation from a teacher.

**CLI**:
```
python distill.py \
  --teacher <hf_model_id_or_path> \
  --student <hf_model_id_or_path> \
  --task <legal> \
  --dataset <path/to/dataset/> \
  --output-dir <models/student/distilled/> \
  --temperature 6.0 \
  --alpha 0.1 \
  --beta 0.9 \
  --gamma 0.1 \
  --epochs 10 \
  --batch-size 16 \
  --lr 5e-5 \
  --device <cpu|cuda> \
  --fp16
```

**Internal modules**:
- `DistillationLoss` — combined CE + KL + MSE loss
- `TeacherStudentTrainer` — HuggingFace `Trainer` subclass
- `HiddenStateProjector` — aligns hidden dims between teacher/student

---

### `scripts/benchmark.py`

**Purpose**: Measure accuracy, latency, memory, and model size.

**CLI**:
```
python benchmark.py \
  --model <model.onnx|model.tflite|model.mlpackage> \
  --task <ocr|legal|audio> \
  --dataset <path/to/eval_dataset/> \
  --runtime <onnxruntime|tflite|coreml> \
  --device <cpu|gpu|npu> \
  --warmup-runs 10 \
  --benchmark-runs 100 \
  --output-json <benchmarks/results/result.json> \
  --output-md <benchmarks/results/result.md>
```

**Internal modules**:
- `AccuracyEvaluator` — task-specific metrics (CER, F1, accuracy)
- `LatencyProfiler` — wall-clock timing with warmup
- `MemoryProfiler` — peak RSS measurement
- `SizeReporter` — file size + parameter count
- `ReportGenerator` — JSON + Markdown output

---

### `scripts/convert_to_tflite.py`

**Purpose**: Convert ONNX model to TFLite for Android deployment.

**CLI**:
```
python convert_to_tflite.py \
  --input <model.onnx> \
  --output <model.tflite> \
  --quantization <none|int8|fp16> \
  --representative-dataset <path/> \  # for full-int8
  --optimize-for <latency|size>
```

**Internal modules**:
- `ONNXToTFConverter` — onnx2tf or onnx-tf bridge
- `TFLiteConverter` — wraps `tf.lite.TFLiteConverter`
- `TFLiteValidator` — runs inference and checks output parity

---

### `scripts/convert_to_coreml.py`

**Purpose**: Convert ONNX model to CoreML for iOS deployment.

**CLI**:
```
python convert_to_coreml.py \
  --input <model.onnx> \
  --output <model.mlpackage> \
  --minimum-deployment-target <iOS16|iOS17> \
  --compute-units <ALL|CPU_AND_NE|CPU_ONLY> \
  --quantization <none|fp16|int8>
```

**Internal modules**:
- `CoreMLConverter` — wraps `coremltools.convert()`
- `ANEOptimizer` — applies palettization and activation compression for ANE
- `CoreMLValidator` — runs `coremltools.models.MLModel` prediction check

---

### `scripts/convert_to_onnx_mobile.py`

**Purpose**: Optimize ONNX graph for mobile ONNX Runtime.

**CLI**:
```
python convert_to_onnx_mobile.py \
  --input <model.onnx> \
  --output <model_mobile.onnx> \
  --optimization-level <basic|extended|all> \
  --target <android|ios> \
  --enable-nhwc              # layout optimization for mobile
```

**Internal modules**:
- `ONNXGraphOptimizer` — `onnxruntime.transformers.optimizer`
- `MobileLayoutConverter` — NCHW → NHWC conversion
- `OPSetDowngrader` — ensures compatibility with mobile ORT version

---

### `scripts/run_pipeline.py`

**Purpose**: Orchestrate the full end-to-end compression pipeline from a YAML config.

**CLI**:
```
python run_pipeline.py \
  --config configs/ocr_pipeline.yaml \
  --stages export,quantize,prune,distill,benchmark \
  --output-dir models/student/ \
  --dry-run
```

---

## 6. MOBILE DEPLOYMENT BLUEPRINT

### Android Deployment

#### Option A: TFLite (recommended for CNN/vision models)
1. Convert via `convert_to_tflite.py`
2. Place `.tflite` file in `assets/` of Android project
3. Use `org.tensorflow:tensorflow-lite:2.x` Gradle dependency
4. Enable NNAPI delegate: `NnApiDelegate()` for hardware acceleration
5. Enable GPU delegate: `GpuDelegate()` as fallback

```kotlin
val model = FileUtil.loadMappedFile(context, "model.tflite")
val interpreter = Interpreter(model, Interpreter.Options().apply {
    addDelegate(NnApiDelegate())
})
```

#### Option B: ONNX Runtime Mobile (recommended for transformer models)
1. Convert via `convert_to_onnx_mobile.py`
2. Use `com.microsoft.onnxruntime:onnxruntime-android` dependency
3. Enable QNN (Qualcomm Neural Network) execution provider for NPU

```kotlin
val session = OrtEnvironment.getEnvironment().createSession(modelBytes,
    OrtSession.SessionOptions().apply {
        addNnapi()
    })
```

#### Option C: GGUF / llama.cpp (for LLMs)
1. Convert to GGUF with `llama.cpp convert-hf-to-gguf.py`
2. Quantize to Q4_K_M or Q5_K_M
3. Use `llama.cpp` Android bindings via JNI
4. Target: Phi-3-mini Q4_K_M ≈ 2.2 GB, TinyLlama Q4 ≈ 670 MB

### iOS Deployment

#### Option A: CoreML (recommended, best ANE utilization)
1. Convert via `convert_to_coreml.py`
2. Add `.mlpackage` to Xcode project
3. Use `CoreML` framework with `MLModelConfiguration`

```swift
let config = MLModelConfiguration()
config.computeUnits = .all  // enables Neural Engine
let model = try MyModel(configuration: config)
let prediction = try model.prediction(input: modelInput)
```

#### Option B: ONNX Runtime for iOS
1. Use `pod 'onnxruntime-objc'` or Swift package
2. Supports CPU + Metal (GPU) execution providers

### Model Packaging

| Target | Format | Tool | Max recommended size |
|---|---|---|---|
| Android TFLite | `.tflite` | `tf.lite.TFLiteConverter` | 200 MB |
| Android ONNX | `.onnx` + `.with_runtime_opt.ort` | ORT tools | 200 MB |
| Android LLM | `.gguf` | llama.cpp | 4 GB (with mmap) |
| iOS CoreML | `.mlpackage` | `coremltools` | 200 MB |
| iOS ONNX | `.onnx` | ORT iOS | 200 MB |

### Inference Efficiency Tips

- Use memory-mapped model loading (no full RAM copy)
- Batch inputs where possible (reduce overhead)
- Cache tokenizer/processor outside inference loop
- Use async inference for UI responsiveness
- Profile with Android GPU Inspector / Instruments (Xcode)

---

## 7. BENCHMARKING SUITE

### Metrics

| Metric | OCR | Legal | Audio | How Measured |
|---|---|---|---|---|
| Accuracy | CER (↓), WER (↓) | F1 (↑), Exact Match (↑) | Accuracy (↑) | Dataset evaluation |
| Model Size | MB | MB | MB | `os.path.getsize()` |
| Parameters | Count | Count | Count | `sum(p.numel())` |
| Latency (cold) | ms | ms | ms | `time.perf_counter()` |
| Latency (warm) | ms | ms | ms | Mean of N runs |
| P95 Latency | ms | ms | ms | Percentile of distribution |
| Peak RAM | MB | MB | MB | `/proc/self/status` or `tracemalloc` |
| Energy (on-device) | mWh | mWh | mWh | Android BatteryManager API |

### Test Datasets

| Task | Dataset | Samples | Source |
|---|---|---|---|
| OCR | IIIT-5K-Words | 3,000 | Academic |
| OCR | IAM Handwriting | 1,539 pages | Academic |
| Legal | CUAD (Contract Understanding) | 510 contracts | Academic |
| Legal | ContractNLI | 607 contracts | Academic |
| Audio | Donate-a-cry corpus | 457 recordings | Open source |
| Audio | ESC-50 (audio classification) | 2,000 clips | Open source |

### JSON Output Format

```json
{
  "run_id": "ocr_trocr_int8_20240315_143022",
  "timestamp": "2024-03-15T14:30:22Z",
  "model": {
    "name": "trocr-base-printed-int8",
    "path": "models/student/trocr_int8.onnx",
    "size_mb": 84.3,
    "parameters": 44700000,
    "format": "onnx",
    "compression": ["int8_static"]
  },
  "teacher": {
    "name": "trocr-large-printed",
    "size_mb": 1340.0
  },
  "hardware": {
    "device": "Samsung Galaxy S23",
    "cpu": "Snapdragon 8 Gen 2",
    "ram_gb": 8,
    "os": "Android 13"
  },
  "accuracy": {
    "cer": 0.043,
    "wer": 0.089,
    "teacher_cer": 0.031,
    "accuracy_drop_pct": 1.2
  },
  "latency_ms": {
    "cold_start": 312.4,
    "warm_mean": 87.3,
    "warm_p50": 85.1,
    "warm_p95": 103.2,
    "warm_p99": 119.8
  },
  "memory_mb": {
    "peak_rss": 284.6,
    "model_loaded": 192.1
  },
  "size_reduction_pct": 93.7,
  "latency_speedup_vs_teacher": 4.2
}
```

### Markdown Output Format

Results are auto-generated to `benchmarks/results/<run_id>.md` with tables comparing teacher vs student across all metrics.

---

## 8. ROADMAP & MILESTONES

See [docs/roadmap.md](roadmap.md) for full details.

| Phase | Duration | Deliverables |
|---|---|---|
| Phase 1 | Week 1–2 | Model selection, ONNX export pipeline |
| Phase 2 | Week 3–5 | Quantization (INT8/INT4) + pruning |
| Phase 3 | Week 6–9 | Knowledge distillation training |
| Phase 4 | Week 10–12 | TFLite + CoreML conversion |
| Phase 5 | Week 13–15 | Full benchmark suite |
| Phase 6 | Week 16 | v1.0 release |

---

## 9. GITHUB ISSUE GENERATION FORMAT

Each TODO item follows this machine-readable format to enable automated GitHub issue creation:

```yaml
# TODO Item Template
- id: SHRINK-001
  title: "Implement ONNX export script for encoder-decoder models"
  description: |
    Create scripts/export_to_onnx.py with support for seq2seq models (TrOCR, T5)
    and encoder-only models (BERT, ViT). Must handle dynamic axes for variable-length
    inputs and validate exported model with ORT inference.
  acceptance_criteria:
    - TrOCR-base exports successfully to ONNX opset 17
    - Dynamic axes set correctly for batch_size and sequence_length
    - Exported model validated with sample inputs; max numerical diff < 1e-4
    - CLI help text complete and all args documented
    - Unit test in tests/test_export.py passes
  dependencies: []
  labels: [pipeline, onnx, priority-high, phase-1]
  milestone: "Phase 1: Model Selection + ONNX Export"
  estimate: "3d"

- id: SHRINK-002
  title: "Implement static INT8 quantization with calibration"
  description: |
    Extend scripts/quantize.py to support static INT8 quantization using
    onnxruntime's CalibrationDataReader. Must support custom calibration datasets
    per task and allow skipping sensitive ops (softmax, LayerNorm).
  acceptance_criteria:
    - Static INT8 quantization produces valid ONNX model
    - Calibration dataset loading works for OCR, legal, and audio tasks
    - Accuracy drop on IIIT-5K ≤ 1% vs FP32 baseline
    - Size reduction ≥ 3.5× vs FP32 input
    - Unit test passes
  dependencies: [SHRINK-001]
  labels: [compression, quantization, priority-high, phase-2]
  milestone: "Phase 2: Quantization + Pruning"
  estimate: "4d"
```

---

## 10. TECHNOLOGY STACK

| Layer | Technology |
|---|---|
| Model framework | PyTorch 2.x + HuggingFace Transformers |
| ONNX export | `torch.onnx`, `optimum` |
| Quantization | `onnxruntime`, `auto-gptq`, `bitsandbytes` |
| Pruning | `nn_pruning`, custom |
| Distillation | HuggingFace `Trainer` + custom loss |
| Android runtime | TFLite, ONNX Runtime Android, llama.cpp |
| iOS runtime | CoreML, `coremltools`, ONNX Runtime iOS |
| Benchmarking | Custom + `onnxruntime` profiling |
| Testing | pytest |
| CI | GitHub Actions |
| Python version | 3.10+ |
