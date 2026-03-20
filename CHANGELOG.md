# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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
