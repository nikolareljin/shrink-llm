# AGENTS.md — ShrinkLLM

## Project Summary

ShrinkLLM is an ML model compression pipeline targeting smartphone deployment.
It compresses teacher models (OCR, legal reasoning, audio classification) into
mobile-ready student models via quantization, pruning, and knowledge distillation.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

## Key Commands

```bash
# Run all tests
pytest -q

# Run a single test file
pytest -q tests/test_quantization.py

# Export a model to ONNX
python scripts/export_to_onnx.py --model microsoft/trocr-base-printed --task ocr --output models/student/trocr_base.onnx

# Quantize to INT8
python scripts/quantize.py --input models/student/trocr_base.onnx --precision int8 --mode dynamic --output models/student/trocr_int8.onnx

# Run full pipeline
python scripts/run_pipeline.py --config configs/ocr_pipeline.yaml

# Benchmark
python scripts/benchmark.py --model models/student/trocr_int8.onnx --task ocr --dataset datasets/ocr/

# Convert to TFLite
python scripts/convert_to_tflite.py --input models/student/trocr_int8.onnx --output models/student/trocr.tflite

# Convert to CoreML
python scripts/convert_to_coreml.py --input models/student/trocr_int8.onnx --output models/student/trocr.mlpackage
```

## Project Structure

- `compression/` — quantization, pruning, distillation modules
- `scripts/` — CLI pipeline scripts (see docs/architecture.md §5 for full spec)
- `configs/` — YAML pipeline configs per task
- `benchmarks/` — benchmark runner + results (JSON + Markdown)
- `mobile_deployment/` — Android (TFLite/ORT) and iOS (CoreML) packaging
- `datasets/` — download + preprocess scripts per task
- `tests/` — pytest unit tests

## Conventions

- Python 3.10+, PEP 8, 4-space indent, type hints required
- All scripts have `--help` with complete argument documentation
- Benchmark results saved to `benchmarks/results/` as JSON + Markdown
- Never commit model weights or large dataset files (use `.gitignore`)
- Config files are YAML in `configs/`

## Compression Pipeline Stages

1. `export_to_onnx.py` — ONNX export
2. `quantize.py` — INT8/INT4 quantization
3. `prune.py` — structured pruning
4. `distill.py` — knowledge distillation
5. `convert_to_tflite.py` / `convert_to_coreml.py` / `convert_to_onnx_mobile.py` — mobile conversion
6. `benchmark.py` — full evaluation

## CI

GitHub Actions workflows run `pytest -q` on push to `main` and PRs.
