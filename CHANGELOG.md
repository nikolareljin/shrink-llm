# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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
