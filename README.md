# ShrinkLLM

**Compress capable AI models to run efficiently on smartphones.**

ShrinkLLM is an open-source, end-to-end pipeline for taking production-grade AI models (OCR, legal document reasoning, audio classification, and more) and compressing + adapting them for on-device inference on Android and iOS.

---

## Why ShrinkLLM?

Modern AI models are too large for smartphones. ShrinkLLM bridges the gap using:

- **Knowledge Distillation** — transfer intelligence from a large teacher to a small student
- **Quantization** — reduce weights from FP32 → INT8 / INT4
- **Structured Pruning** — remove redundant attention heads and MLP blocks
- **ONNX / TFLite / CoreML** — multi-runtime mobile deployment

---

## Target Use Cases

| Task | Teacher | Student | Target Size |
|---|---|---|---|
| OCR | TrOCR-Large | TrOCR-Small / MobileViT | < 30 MB |
| Legal Doc Reasoning | Mistral-7B / Phi-3 | Phi-3-mini / Gemma-2B | < 200 MB (4-bit) |
| Audio Classification | Wav2Vec2-Large | MobileNet Audio | < 5 MB |

---

## Quick Start

```bash
git clone https://github.com/nikolareljin/shrink-llm.git
cd shrink-llm
git submodule update --init --recursive
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Export a model to ONNX
python scripts/export_to_onnx.py --model microsoft/trocr-base-printed --task ocr --output models/student/trocr_base.onnx

# Quantize to INT8
python scripts/quantize.py --input models/student/trocr_base.onnx --precision int8 --mode dynamic --output models/student/trocr_int8.onnx

# Benchmark
python scripts/benchmark.py --model models/student/trocr_int8.onnx --task ocr --dataset datasets/ocr/
```

## Development Setup

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Baseline requirements:

- Python 3.10 or newer for local development.
- Initialize `scripts/script-helpers` after cloning so shared shell helpers are available.
- Keep model weights, raw datasets, and mobile build outputs outside git-tracked source paths.

Optional extras:

```bash
pip install -e ".[dev,tflite]"
pip install -e ".[dev,coreml]"
pip install -e ".[dev,gptq,audio]"
```

Common checks:

```bash
pytest -q
ruff check scripts/ compression/ benchmarks/ tests/
black --check scripts/ compression/ benchmarks/ tests/
python scripts/run_pipeline.py --config configs/ocr_pipeline.yaml --dry-run
```

---

## Pipeline Overview

```
Teacher Model
     │
     ▼
ONNX Export ──► Quantization (INT8/INT4) ──► Pruning ──► Distillation ──► Fine-tune
                                                                              │
                                                              ┌───────────────┤
                                                              ▼               ▼
                                                         TFLite           CoreML
                                                        (Android)          (iOS)
```

---

## Documentation

- [Architecture Blueprint](docs/architecture.md)
- [Compression Pipeline](docs/compression_pipeline.md)
- [Mobile Deployment Guide](docs/mobile_deployment.md)
- [Benchmarking Guide](docs/benchmarking.md)
- [Roadmap](docs/roadmap.md)
- [Contributing](docs/contributing.md)

---

## Repository Structure

```
shrink-llm/
├── models/
│   ├── teacher/          # Full-size pretrained models
│   └── student/          # Compressed/distilled models
├── compression/
│   ├── quantization/     # INT8, INT4, mixed precision
│   ├── pruning/          # Structured pruning logic
│   └── distillation/     # Knowledge distillation training
├── datasets/
│   ├── ocr/
│   ├── legal/
│   └── audio/
├── mobile_deployment/
│   ├── android/          # TFLite, ONNX Runtime Mobile
│   ├── ios/              # CoreML, Metal
│   └── onnx_mobile/      # ONNX mobile packaging
├── scripts/              # CLI pipeline scripts
├── benchmarks/           # Benchmark runner + results
├── configs/              # YAML configs per task/pipeline
├── notebooks/            # Exploratory Jupyter notebooks
├── tests/                # Unit + integration tests
└── docs/                 # Full documentation
```

Generated model weights, raw datasets, mobile conversion artifacts, TensorFlow
SavedModels, runtime scratch files, and local `TODO.txt` planning files are
intentionally ignored.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
