# Compression Pipeline

ShrinkLLM runs models through these stages:

1. Export the selected teacher or student baseline to ONNX.
2. Apply quantization and, where appropriate, pruning.
3. Run distillation or recovery fine-tuning.
4. Convert validated artifacts to mobile runtimes.
5. Benchmark the resulting model against task targets.

This document is a placeholder for the detailed stage-by-stage guide referenced by the README.
