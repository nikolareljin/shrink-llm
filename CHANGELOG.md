# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- `docs/text_classification.md` — gap analysis for producing text classifiers, driven by the
  first external consumer (a downstream on-device classifier shipping TFLite). Records that
  `export_to_onnx.py`'s `classification` task is *image* classification, that `distill.py` is
  gated to `SUPPORTED_TASKS = ("legal",)` while `run_pipeline.py` silently skips the
  distillation stage for anything else, that nothing emits the tokenizer a compressed
  classifier needs to be usable on-device, and that no config carries a precision gate.
- Roadmap Phase 7 and issues `SHRINK-015`–`SHRINK-019` covering that work: text-classification
  export with a single-input, softmax-in-graph wrapper; sequence-classification distillation;
  an app-artifact exporter with digests; `configs/scam_pipeline.yaml` introducing
  `min_precision` to `success_criteria`; and the matching benchmark-stage task, without which a
  text-classification run fails at the last stage.
- `docs/reviews/pr-33-findings.md` and `docs/reviews/codebase-audit.md` — the reviews that
  produced the fixes below.
- Configs may declare a `stages:` list, used as the default when `--stages` is not passed. All
  three shipped configs now carry one and run end to end; `legal_pipeline.yaml` previously
  aborted on a config error before doing any work.
- `SHRINK-020` — restore CoreML conversion behind a supported converter front end.
- `quantize.py --group-size`, wired from `quantization.group_size`, which was previously read by
  nothing.
- `convert_to_tflite.py --integer-io` for accelerators that require fully-integer tensors.
- `tests/test_packaging.py` — locks the packaging invariants below.
- `benchmark.py` accepts `--task classification`, matching `export_to_onnx.py`. The task lists
  are now `SUPPORTED_TASKS` constants with a test asserting the benchmark's stays a superset of
  the exporter's, so a task can no longer be exportable but not benchmarkable.

### Fixed

- **The test suite could not run on Python 3.10**, which `requires-python = ">=3.10"` allows and
  which CI now exercises. Two test modules imported `tomllib`, stdlib only from 3.11, so the 3.10
  leg would have failed at collection time rather than on one test. Parsing moved to a
  `tests/conftest.py` fixture that falls back to `tomli`, now declared in the `dev` extra under a
  `python_version < "3.11"` marker. The fixture also resolves `pyproject.toml` from the repo root
  instead of the working directory.
- `convert_to_coreml.py` imported `coremltools` before raising its "ONNX input is not supported"
  error, purely to name the version. Without the optional `coreml` extra installed — the common
  case — that surfaced as "install coremltools", sending the user to install a package that
  cannot do the job. The explanation is now unconditional.
- `convert_to_tflite.py` read only the keyword half of a SavedModel's
  `structured_input_signature`. A signature exposing its inputs positionally yielded an empty
  input list, and the representative-dataset generator then fed TFLite nothing while reporting
  success. It now falls back to the positional structure and raises when neither is present.
- `convert_to_tflite.py`'s `ImportError` guard wrapped the `onnx2tf.convert()` call as well as
  the import, so an `ImportError` raised *inside* onnx2tf — a missing TensorFlow, say — was
  reported as "onnx2tf not installed" and the real cause discarded.
- `benchmark.py` reported `VmRSS` as peak memory. Peak RSS is `VmHWM`; `VmRSS` is current
  residency. Both memory samples were also taken *after* the model was loaded and the full
  benchmark loop had run, so `rss_after_load` measured neither, and `rss_delta` measured one
  extra inference rather than the model load. The reported fields are now `rss_baseline`,
  `rss_after_load`, `model_load_delta` and `peak_rss`, each sampled where its name implies.
- `benchmark.py` built `run_id` from naive local time while `timestamp` used UTC — two clocks in
  one record.
- **`quantize.py` crashed on every invocation.** Both `quantize_dynamic` and `quantize_static`
  were called with `optimize_model=True`, which is not a parameter of either on any onnxruntime
  in range of the declared `>=1.18.0` floor — every dynamic and static quantization run raised
  `TypeError: got an unexpected keyword argument 'optimize_model'`. This broke the README Quick
  Start, `SHRINK-002`, `SHRINK-003` and the `quantize` stage of every config. No test invoked
  either function; `tests/test_quantization.py` now does.
- **`--skip-ops` protected nothing.** onnxruntime's `nodes_to_exclude` takes node *names*, and
  the code passed op *types* (`Softmax`, `LayerNormalization`, `Gelu`), which match no node.
  Op types are now resolved against the graph, so the sensitive layers that
  `DEFAULT_SKIP_OPS` names are actually kept in FP32.
- **Magnitude pruning crashed on every model in the README.** `torch.quantile` raises above 2²⁴
  elements, and BERT-base's embedding alone is 23.4M — `magnitude` is `run_pipeline.py`'s default
  method, and the `attention_heads` path falls back to it. Replaced with `torch.kthvalue`, which
  has no size limit and returns the identical threshold.
- Calibration data is read from `.npz` with `allow_pickle=False`. It was `np.load(..., 
  allow_pickle=True)` on a directory named by a config value, so loading a calibration set could
  execute arbitrary code. `.npz` also gives `quantize.py` and `convert_to_tflite.py` one shared
  format — they previously disagreed (pickled dict vs bare array) while `run_pipeline.py` handed
  both the same directory.
- Synthetic calibration data is derived from the model's own input signature instead of being
  guessed as `float32 [1, 128]`, which was the wrong rank for `pixel_values` and the wrong dtype
  for `input_ids`.
- `convert_to_tflite.py --quantization int8` no longer sets int8 I/O without a representative
  dataset, a combination TFLite rejects. It now requires the dataset, feeds inputs in the model's
  own order — the old single-element list miscalibrated every multi-input model — and defaults to
  float I/O with integer weights, which is what the on-device contract in
  `docs/text_classification.md` specifies.
- `convert_to_coreml.py` failed with coremltools' generic "unable to determine the type of the
  model". coremltools dropped ONNX input in 6.0 and this project requires `>=7.2`, so the stage
  has never worked; it now says exactly that and points at `SHRINK-020`. Its `--quantization
  fp16` path also passed `dtype="float16"` to a linear quantizer that accepts only integer
  types, and its int8 handler logged "Falling back to FP16" while silently keeping the
  unquantized model.
- `convert_to_onnx_mobile.py --generate-ort` converted every `.onnx` in the output directory as
  a side effect of asking for one; it now stages the requested model alone. `--target` was
  accepted and never read, and has been removed.
- The local `datasets/` directory shadowed the HuggingFace `datasets` dependency: it carried an
  `__init__.py` and was declared as a package, so `import datasets` from the repo root — where
  `run_pipeline.py` runs every stage — returned an empty stub with no `load_dataset`. It is now
  a plain data directory.
- The `tflite` extra installed `onnx-tf`, whose last release requires `tensorflow-addons`
  (archived May 2024, capped at TF 2.14) against a declared `tensorflow>=2.16.0`; and it omitted
  `onnx2tf`, which is the converter the code actually prefers. `onnxconverter-common`, needed by
  the fp16 path, was undeclared entirely.
- `MLPPruner.collect_activations` divided by the requested batch count rather than the number of
  batches delivered, biasing the activation frequencies that the pruning threshold is computed
  from.
- `prune.py --task` rejected `classification`, which `export_to_onnx.py` accepts. Both task lists
  are now `SUPPORTED_TASKS` constants with parity tests.
- CI tested Python 3.11 only, against `requires-python = ">=3.10"`; it now runs 3.10, 3.11 and
  3.12, and lints `mobile_deployment/`, which was packaged but never checked.
- `release.yml` swallowed every tag and push failure with `|| echo`, so a release that tagged
  nothing still reported success. Only an already-existing tag is treated as success now.
- The README pipeline diagram and `docs/compression_pipeline.md` both put quantization before
  pruning and distillation; the code prunes and distils the PyTorch model *before* export.
- `distill.py` computed causal-LM cross-entropy against **unshifted** labels. A causal LM's
  logits at position `t` predict token `t+1`; HuggingFace shifts internally when you pass
  `labels=`, but `DistillationTrainer.compute_loss` pops them out and does the CE by hand. The
  student was being trained to predict the token it had just been given. Shifting is now an
  explicit `DistillationConfig.shift_labels` flag, set for the causal path and left off for the
  sequence-classification path `SHRINK-016` adds.
- `DistillationLoss` scaled its KL term by the sequence length. `kl_div(reduction="batchmean")`
  divides by `input.size(0)`, which on `[B, T, V]` is `B` — a per-sequence mean set against a
  per-token CE. At `T = 512` the effective KL weight was ~4600× the CE weight instead of 9×, and
  it moved whenever `max_length` did. Both terms are now per-token means.
- `distill.py` raised `TypeError` on transformers ≥ 4.46, which removed
  `TrainingArguments(evaluation_strategy=...)`. Uses `eval_strategy`; the dependency floor moves
  to `transformers>=4.41.0`, the first release accepting it.
- `run_pipeline.py` never passed `success_criteria.min_accuracy` to `benchmark.py`, which has
  implemented `--min-accuracy` all along. `configs/audio_pipeline.yaml` has carried
  `min_accuracy: 0.88` against a gate that was never invoked.
- `run_pipeline.py` reported configured-but-unenforced criteria (`min_f1`, `max_cer`,
  `max_accuracy_drop_pct`) at `debug`, which never prints at the default level — so a threshold
  that was silently ignored produced a green run. Now a warning naming the key and the value.
- `run_pipeline.py` dropped skipped stages entirely: the `continue` ran before both
  `results[stage]` and `_update_manifest`, so a stage that was requested but could not be built
  left no trace in the summary or in `manifest.json`. Such stages are now recorded as `SKIPPED`
  with a `skipped` manifest entry.
- `export_to_onnx.py --no-save-config` described a `model_config.json` that is never written;
  the file is `<output stem>.json`.

## [0.3.0] - 2026-04-25

### Added

- Stable artifact manifest schema v1: pipeline-level metadata (`version`, `pipeline_run_id`, `model_id`, `task`, `config_path`, `config_hash`, `total_stages`) is now written to `manifest.json` at pipeline start before any stage runs, addressing the schema and per-stage tracking aspects of issue #21.
- Per-stage artifact tracking: files created during each stage are discovered via directory snapshot diff and recorded in the manifest with relative paths and sizes in MB.
- Per-stage `exit_code` and structured `error` message in manifest stage records; failure records now include the non-zero exit code and a human-readable reason.
- Benchmark acceptance gates: `--max-size-mb`, `--max-latency-ms-p95`, and `--min-accuracy` CLI args; `benchmark.py` exits with code 1 when any threshold is not met, resolving issue #24.
- `gate_results` (per-criterion pass/fail map) and `passed` (overall boolean) fields added to `BenchmarkResult` and included in JSON output.
- Pipeline runner forwards the currently wired YAML `success_criteria` thresholds to the benchmark stage: `max_size_mb` and `max_latency_ms`.

## [0.2.0] - 2026-04-21

### Changed

- Renamed the `baby_cry` task identifier to `audio` throughout all scripts, configs, datasets, tests, and documentation. The audio classification pipeline is unchanged; only the internal task name is updated for the public release.

### Added

- Wired all pipeline stages in `run_pipeline.py`: `prune`, `distill`, `convert_tflite`, `convert_coreml`, and `convert_onnx_mobile` now generate correct CLI argument lists from the YAML config. Previously these stages returned empty args and were silently skipped.
- Added artifact manifest output to `run_pipeline.py`: each completed stage appends a record to `manifest.json` in the output directory with stage name, args, status, and timestamp.
- Implemented attention head pruning in `prune.py`: the `--method attention_heads` option now computes head importance scores via forward-pass hooks and zeros out the lowest-scoring head weight slices in Q/K/V projections. Falls back to magnitude pruning when the model architecture does not expose attention weights.
- Added synthetic dataloader helpers to `prune.py` so attention head and MLP pruning can run without a real dataset.

### Fixed

- Resolved Python CI lint failures by migrating deprecated Ruff lint settings into `tool.ruff.lint`, removing unused imports and locals, and updating optional annotations to modern `X | None` syntax.
- Reformatted benchmark, conversion, pruning, quantization, pipeline, and test modules so `black --check scripts/ compression/ benchmarks/ tests/` passes consistently in CI.
- Replaced shared reusable GitHub workflows with local workflow definitions that use Node 24-ready `actions/checkout@v5` and `actions/setup-python@v6`, removing the Node 20 Actions deprecation warning from CI.

## [0.1.1] - 2026-03-13

### Fixed

- Corrected the public clone URL and expanded contributor setup steps to include submodule bootstrap and expected local tooling.
- Added ignore rules for additional generated mobile/runtime artifacts such as `.mlmodel`, `.mlmodelc`, TensorFlow protobufs, and local output directories.

## [0.1.0] - 2026-03-13

### Added

- Added the documented repository skeleton for benchmarks, datasets, mobile deployment, models, notebooks, and compression subpackages.
- Added placeholder project docs for compression pipeline, mobile deployment, benchmarking, and contributing workflows.

### Changed

- Updated the README quick-start and development setup commands to match the current CLI interfaces and local workflow.
- Expanded `.gitignore` coverage for generated mobile artifacts, checkpoints, logs, and local scratch files.
