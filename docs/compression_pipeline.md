# Compression Pipeline

ShrinkLLM runs models through these stages, in the order `VALID_STAGES` in
`scripts/run_pipeline.py` defines. Pruning and distillation come **first**, because both operate
on the PyTorch model — a graph that has already been exported and quantized can no longer be
trained or structurally modified:

1. `prune` — structured or magnitude pruning of the PyTorch student.
2. `distill` — knowledge distillation from the teacher, and recovery fine-tuning.
3. `export` — export the resulting student to ONNX.
4. `quantize` — INT8/INT4 on the ONNX graph. GPTQ is the exception: it reads the PyTorch model
   directly and emits a directory rather than an `.onnx`.
5. `convert_tflite` / `convert_coreml` / `convert_onnx_mobile` — mobile runtime artifacts.
6. `benchmark` — measure the result against the config's `success_criteria`.

Every stage is optional. Select a subset with `--stages`, or give the config a `stages:` list,
which becomes the default when `--stages` is not passed. `run_pipeline.py` validates the
selection: `quantize`, the converters and `benchmark` each require `export`, except under GPTQ,
whose directory artifact the ONNX-consuming stages cannot read at all.

This document is otherwise a placeholder for the detailed stage-by-stage guide referenced by the
README.
