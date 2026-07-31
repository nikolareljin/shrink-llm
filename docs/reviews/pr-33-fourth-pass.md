# PR #33 fourth-pass review — findings

Reviewed: 2026-07-30
Scope: areas the first three passes never opened — the long-form documentation, the repository
structure it describes, the planning documents this PR committed, and the PR's review state.

The previous passes found 28, then 7, then 5 findings, and the last round was mostly test
quality. Rather than re-tread audited code, this pass went where nothing had looked yet. The
result is one theme, not scattered defects: **the documentation describes a repository that does
not exist**, and two of the earlier fixes updated one document while leaving its counterpart
contradicting it.

No new defects were found in `scripts/` behaviour. That is worth stating plainly rather than
manufacturing findings to justify the pass.

---

## A. The documented architecture is not the built one

### A1 — `docs/architecture.md` documents 17 files that do not exist (high)

The 768-line blueprint's "REPOSITORY STRUCTURE" section describes 32 Python modules. Seventeen
have never existed:

| Documented location | Purpose given |
|---|---|
| `compression/quantization/onnx_quantizer.py` | ONNX quantization |
| `compression/quantization/torch_quantizer.py` | PyTorch static/dynamic quant |
| `compression/quantization/gptq_quantizer.py` | GPTQ 4-bit |
| `compression/pruning/attention_pruner.py` | Head importance + removal |
| `compression/pruning/mlp_pruner.py` | FFN block pruning |
| `compression/pruning/magnitude_pruner.py` | Weight magnitude pruning |
| `compression/distillation/trainer.py` | KD training loop |
| `compression/distillation/losses.py` | KL, MSE, cosine losses |
| `compression/distillation/callbacks.py` | Checkpointing, early stopping |
| `datasets/{ocr,legal,audio}/download.py` | Dataset downloaders |
| `datasets/{ocr,legal,audio}/preprocess.py` | Preprocessing |
| `benchmarks/runner.py` | Device + accuracy benchmarks |
| `benchmarks/metrics.py` | CER, F1, accuracy, latency |
| `mobile_deployment/android/tflite_packager.py` | TFLite packaging |
| `mobile_deployment/android/onnx_mobile_packager.py` | ORT Mobile packaging |
| `mobile_deployment/ios/coreml_packager.py` | CoreML builder |
| `mobile_deployment/onnx_mobile/optimizer.py` | ONNX graph optimizations |

Every one of those responsibilities lives in `scripts/` instead. `compression/`,
`benchmarks/` and `mobile_deployment/` contain nothing but one-line `__init__.py` files —
packages `pyproject.toml` ships and CI now lints, holding no code at all.

This is not a cosmetic gap. A contributor reading the blueprint would look for
`compression/pruning/magnitude_pruner.py` to fix a pruning bug and find an empty package, while
the actual defect sat in `scripts/prune.py`.

**Fix**: rewrite the structure section to describe what exists, and mark the unbuilt parts as
aspirational rather than present.

### A2 — the blueprint's `datasets/` design is the defect this PR removed (high)

```
├── datasets/
│   ├── __init__.py
│   ├── ocr/
│   │   ├── download.py
```

That `datasets/__init__.py` is exactly what made the data directory shadow the HuggingFace
`datasets` dependency, fixed in `6632941`. The blueprint still prescribes it, so anyone
implementing the documented design would reintroduce the bug — and the test added to prevent it
(`test_datasets_dir_has_no_init`) would then read as an obstacle rather than a guard.

**Fix**: replace that subtree with the data-only directory it now is, and say why it must not be
a package.

### A3 — the blueprint conflicts with this PR's approved Phase 7 plan (medium)

Two placements disagree with `docs/superpowers/specs/2026-07-30-text-classification-design.md`:

| Concern | Blueprint | Approved spec |
|---|---|---|
| Metrics | `benchmarks/metrics.py` | `compression/metrics.py` |
| Evaluation loop | `benchmarks/runner.py` | `compression/evaluation.py` |
| Dataset loading | `datasets/<task>/download.py` | `compression/data/` |

The third is not a matter of taste — the blueprint's location cannot be used at all (A2). For the
first two the spec is the live decision, approved after the blueprint was written, and it keeps
the new modules together with the compression code that consumes them.

**Fix**: reconcile the blueprint to the spec, noting the change rather than silently rewriting
intent. Flagged in the summary so the choice is visible.

### A4 — the OCR accuracy target is stated backwards (medium)

```
| OCR | ... | ≥ 95% CER | < 200 ms/page |
```

CER is character **error** rate — lower is better. The same document's §12 correctly annotates
"CER (↓)", and `configs/ocr_pipeline.yaml` sets `max_cer: 0.05`. The target should read
**≤ 5% CER**; as written it demands a model that is wrong 95% of the time.

### A5 — three notebooks are documented; `notebooks/` is empty (low)

`01_model_exploration.ipynb`, `02_compression_analysis.ipynb` and
`03_benchmark_visualization.ipynb` do not exist.

---

## B. Documentation contradicting changes made in this PR

These are self-inflicted: earlier passes updated one document and left its counterpart stating
the opposite.

### B1 — `docs/contributing.md` still tells contributors CI runs 3.11 only (medium)

> Python 3.10+ locally. CI runs on Python 3.11, so use 3.11 when you want the closest match.

CI now runs 3.10, 3.11 and 3.12 (`53eef13`). The advice is stale, and it points a contributor at
the one version where the `tomllib` breakage would not have shown up.

### B2 — `docs/contributing.md`'s lint command differs from CI's (medium)

```bash
ruff check scripts/ compression/ benchmarks/ tests/
black --check scripts/ compression/ benchmarks/ tests/
```

CI now also lints `mobile_deployment/`. A contributor following this document lints a narrower
tree than CI does, passes locally, and fails the gate — the precise failure mode the CI change
was meant to close.

### B3 — `docs/contributing.md` still requires the unused submodule (low)

It instructs `git submodule update --init --recursive` twice and lists "a checked-out
`scripts/script-helpers` submodule" as an expected part of the environment. `README.md` was
corrected in `ded0ad1` to call it optional and unused — the repository contains no shell scripts
— but `contributing.md` was missed.

### B4 — the committed implementation plan hardcodes a local virtualenv path (medium)

`docs/superpowers/plans/2026-07-30-text-classification.md` contains **36** occurrences of
`.venv/bin/python`, `.venv/bin/ruff` and `.venv/bin/black`. That is this machine's layout written
into a document whose stated audience is "an engineer with zero context for our codebase". On any
environment that activates its virtualenv normally, or uses a different directory name, every
verification step in the plan fails.

**Fix**: use bare `python`, `pytest`, `ruff` and `black`, and state the activation assumption once
in the plan's Global Constraints.

### B5 — the plan ships code it tells the implementer not to use (low)

Task 1 Step 3 contains a deliberately clumsy expression with a following note to replace it. The
plan's own rules forbid placeholders; an implementer working through tasks mechanically may paste
it. The correct form is short and should simply be written.

---

## C. Other documentation defects

### C1 — `docs/benchmarking.md` names the wrong output location (medium)

> Benchmark outputs belong in `benchmarks/results/` as JSON and Markdown

`run_pipeline.py` writes to `<output-dir>/benchmarks/` — `models/student/benchmarks/` by default.
`benchmarks/results/` exists and is always empty.

### C2 — `docs/mobile_deployment.md` lists CoreML as a supported target (medium)

It names "CoreML for iOS deployment" alongside TFLite and ORT Mobile with no indication that the
stage raises `NotImplementedError` (SHRINK-020), which this PR made explicit everywhere else.

---

## D. PR state

### D1 — the only review on the PR is the Actions budget error (informational)

`copilot-pull-request-reviewer` submitted a review whose entire body is:

> The job was not started because a GitHub Actions budget is preventing further Actions use.

There are zero inline comments and zero review threads, resolved or otherwise. So there is no
outstanding review feedback — but there is also no actual review.

`AGENTS.md` requires re-requesting Copilot review after addressing findings when the PR was
already reviewed by Copilot. That is pointless until the budget is restored, since the re-review
would produce the same message. Recorded so the step is not skipped by accident once Actions runs
again.

---

## Checked and clean

- No new defects in `scripts/` behaviour. The third pass's fixes hold up.
- `generate_markdown` iterates `result.memory_mb.items()`, so the memory-key rename in `1bb8a5d`
  broke no consumer.
- All 159 tests pass; `ruff` and `black` are clean; all three shipped configs dry-run.
- `docs/roadmap.md`, `docs/todos.yaml` and `docs/text_classification.md` are consistent with each
  other and with `SHRINK-015` … `SHRINK-020`.
- `.gitignore` covers `.venv/`, `venv/` and model output paths.

---

## Not addressed here

- **The unbuilt architecture.** Correcting the blueprint documents the gap; it does not close it.
  Whether `compression/`, `benchmarks/` and `mobile_deployment/` should ever hold the code the
  blueprint describes, or whether `scripts/` is the right home and the packages should be dropped,
  is a design decision beyond this PR. The Phase 7 plan begins filling `compression/` either way.
- **`SHRINK-020`**, and accuracy evaluation, as before.
