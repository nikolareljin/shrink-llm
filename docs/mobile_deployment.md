# Mobile Deployment Guide

ShrinkLLM targets three mobile artifact families.

| Target | Stage | Status |
|---|---|---|
| TFLite (Android) | `convert_tflite` | Working |
| ONNX Runtime Mobile (either platform) | `convert_onnx_mobile` | Working |
| CoreML (iOS) | `convert_coreml` | **Unavailable — see below** |

## CoreML is currently unavailable

`scripts/convert_to_coreml.py` raises `NotImplementedError` on every entry point. coremltools
removed ONNX as an input format in 6.0, and `pyproject.toml` requires `>=7.2`, so no supported
version can read what the rest of the pipeline produces — `ct.convert()` accepts only TorchScript,
TensorFlow and MIL sources.

The shipped configs omit `convert_coreml` from their `stages` lists for this reason. Restoring it
needs an ONNX→TorchScript or ONNX→MIL front end, tracked as `SHRINK-020`; the post-conversion path
(compute precision, weight quantization, deployment target, compute units) is kept intact in
`_convert_via_coremltools` for that work.

## TFLite

`convert_to_tflite.py` goes ONNX → TensorFlow SavedModel → TFLite using **onnx2tf**. Install it
with `pip install -e ".[tflite]"`. `onnx-tf` is retained only as a fallback for pre-existing
environments: its last release requires `tensorflow-addons`, archived in May 2024 and capped at
TensorFlow 2.14, so it cannot be installed alongside a current TensorFlow.

`--quantization int8` requires `--representative-dataset`: a directory of `.npz` files, each
mapping every model input name to an array. This is the same format `quantize.py` reads for static
quantization, so one directory serves both. Integer weights with **float** input and output
tensors is the default, because that is what on-device runtimes feeding token ids and reading
probabilities expect; `--integer-io` opts into fully-integer tensors for accelerators that require
them.

## ONNX Runtime Mobile

`convert_to_onnx_mobile.py` applies ORT graph optimizations and can emit the `.ort` flatbuffers
format with `--generate-ort`. `--enable-nhwc` depends on `onnxruntime.tools.transpose_optimizer`,
which is absent from current onnxruntime; when it is missing the model is copied through unchanged
and the log says so explicitly rather than implying the layout was converted.

## Artifacts beside the model

A compressed classifier is not usable on a device by itself — it needs its tokenizer vocabulary
and label set, in a layout the consuming application can rely on. That bundle, and the versioned
manifest describing it, is `SHRINK-017`; see
`docs/superpowers/specs/2026-07-30-text-classification-design.md` §4 for the contract.
