# PR #33 follow-up review — findings

Reviewed: 2026-07-30
Scope: the changes made in PR #33 itself, plus a further pass over the codebase.

The previous two reviews looked at someone else's work. This one looks at the fixes those
reviews produced. Four findings are **self-inflicted** — defects introduced, or made worse, by
the PR's own changes — and one of those is a regression in the common case.

Severity is about consequence: **high** = a supported configuration fails or a claim is false,
**medium** = a flag or metric does not do what it says, **low** = cosmetic.

---

## 0. The finding that invalidates the others' evidence

### I1 — CI has never run any of this code (high)

`gh run list` reports every run on the branch as `failure`, including the run for commit
`511bb1b`, which predates all of these changes. The annotation is not a test failure:

```
failure: The job was not started because an Actions budget is preventing further use.
```

So the `PR Gate` job never started. Nothing in this PR has been verified by CI — including the
**Python 3.10 matrix entry the PR itself added**, which is exactly where S1 below would have been
caught.

Local verification used Python 3.12 only, because no other interpreter is installed here. Every
"142 passed" claim in the PR body is true *on 3.12* and unverified elsewhere. The PR body should
say so, and does not.

**Action**: the Actions budget needs raising or the run needs re-triggering before this merges.
That is an account-level setting, not something the repository can fix.

---

## 1. Self-inflicted findings

### S1 — `tomllib` breaks the Python 3.10 floor the PR added (high)

```python
import tomllib   # tests/test_packaging.py:8, tests/test_quantization.py:185
```

`tomllib` entered the standard library in **3.11**. `pyproject.toml` declares
`requires-python = ">=3.10"`, and this PR added `"3.10"` to the CI matrix — so the PR
simultaneously introduced a test that cannot run on 3.10 and the CI job that would have proven
it. Both would have failed together on the first green run.

Two files, four call sites. On 3.10 the failure is collection-time
`ModuleNotFoundError: No module named 'tomllib'`, so the whole suite fails, not one test.

**Fix**: import `tomli` on 3.10 and `tomllib` above it, with `tomli` added to the `dev` extra
under a `python_version < "3.11"` marker.

### S2 — the Core ML error regressed the common case (medium)

`convert_onnx_to_coreml` now raises a clear `NotImplementedError` explaining that coremltools
dropped ONNX input. But the import sits **above** the raise, purely so the message can
interpolate `ct.__version__`:

```python
try:
    import coremltools as ct
except ImportError:
    raise ImportError("Install coremltools: pip install coremltools")

raise NotImplementedError(f"... coremltools {ct.__version__} does not accept ONNX input ...")
```

The `coreml` extra is optional and not installed by default, so the **common** path is now:
tell the user to install a package that cannot do the job, and say nothing about why. The
actionable message only appears for users who already installed the extra.

Before this PR the same user got a confusing coremltools error. Now they get a confidently wrong
instruction, which is worse.

**Fix**: raise `NotImplementedError` unconditionally, and mention the installed version only when
coremltools happens to be importable.

### S3 — the representative dataset can be built from an empty input list (medium)

```python
loaded = tf.saved_model.load(str(tf_saved_model_dir))
signature = loaded.signatures["serving_default"]
_args, kwargs = signature.structured_input_signature
return list(kwargs)
```

`structured_input_signature` returns `(args, kwargs)`. This reads only `kwargs`. A SavedModel
whose serving signature exposes its inputs **positionally** yields `{}`, so `input_names` is
`[]`, and `_representative_dataset_gen` then yields empty lists for every sample. TFLite
receives a representative dataset that feeds nothing.

The `missing` check inside the generator does not catch this: iterating an empty `input_names`
finds nothing missing.

**Fix**: fall back to the positional `args` structure when `kwargs` is empty, and raise if both
are empty rather than returning a list that silently miscalibrates.

### S4 — `convert_to_coreml.py`'s module docstring describes behaviour it no longer has (low)

It still opens "Convert ONNX model to CoreML for iOS/macOS deployment" with an install hint,
while every entry point raises. `COMPUTE_UNITS_MAP` and `DEPLOYMENT_TARGET_MAP` also remain as
identity dicts used only to populate argparse `choices` — flagged as cosmetic in the audit and
left in place.

---

## 2. Pre-existing findings not caught by the earlier passes

### P1 — reported memory is not peak, and is measured at the wrong time (medium)

```python
    @staticmethod
    def peak_rss_mb() -> float:
        """Read peak RSS from /proc/self/status on Linux."""
        ...
                    if line.startswith("VmRSS:"):
```

`VmRSS` is **current** resident set size. Peak RSS is `VmHWM` ("high water mark"). The method
name, the docstring and the field it reads disagree.

Worse, `main()` samples it at the wrong point:

```python
    runner = get_runner(model_path, args.runtime)      # model loaded here
    latency_raw = profiler.profile(runner, dummy_inputs)   # hundreds of inferences here
    mem_before = MemoryProfiler.peak_rss_mb()          # ... and only now is "before" taken
    runner(dummy_inputs)
    mem_after = MemoryProfiler.peak_rss_mb()
```

So `rss_after_load` is measured long after load and after the full benchmark loop, and
`rss_delta` measures a single extra inference rather than the cost of loading the model. Both
names promise load-time memory; neither measures it. These values are written to the JSON and
Markdown reports as if they were meaningful.

**Fix**: read `VmHWM` for peak, sample a baseline **before** constructing the runner, and rename
the reported fields to what they actually measure.

### P2 — `convert_onnx_to_tf`'s ImportError handler covers the conversion call (medium)

```python
    try:
        import onnx2tf
        log.info(...)
        onnx2tf.convert(...)          # <- inside the try
    except ImportError:
        log.warning("onnx2tf not installed. Trying onnx-tf...")
```

The handler is meant to catch a missing `onnx2tf`. It also catches any `ImportError` raised
*inside* `onnx2tf.convert` — a missing TensorFlow, a missing optional backend — and then falls
through to `onnx-tf`, which this PR just removed from the extras. The user is told
"Install onnx2tf: pip install onnx2tf" when `onnx2tf` is installed and working, and the real
cause is discarded.

**Fix**: narrow the `try` to the import statement alone.

### P3 — `run_id` uses local naive time while `timestamp` uses UTC (low)

```python
    run_id = f"{args.task}_{Path(model_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ...
    timestamp=datetime.now(timezone.utc).isoformat(),
```

Two clocks in one record. `run_pipeline._init_manifest` already uses UTC for its run id; this is
the odd one out, and comparing a benchmark's `run_id` against its `timestamp` across a timezone
boundary is misleading.

---

## 3. Checked and correct

Recorded so these are not re-examined:

- `convert_onnx_models_to_ort(model_path_or_dir: pathlib.Path)` — the temp-directory staging
  added in this PR passes a `Path`, matching the declared signature.
- `magnitude_threshold` at `sparsity=0.0` returns the minimum absolute value and the strict
  `<` comparison then zeroes nothing, which is correct.
- `model_input_specs` correctly excludes initializers that appear as graph inputs under older
  opsets, which the previous `[inp.name for inp in model.graph.input]` did not.
- `SimpleCalibrationDataReader`'s new `input_specs` keyword has exactly one caller
  (`static_quantize`), so the signature change breaks nothing.
- `resolve_requested_stages` composes correctly with `validate_stage_selection`: a config whose
  `stages:` omits a prerequisite still fails through the existing guard.
- No new construct in the changed Python files requires 3.11+ **except** `tomllib` (S1); a sweep
  for `ExceptionGroup`, `typing.Self`, `StrEnum`, `TaskGroup`, `datetime.UTC`,
  `itertools.batched`, `Path.walk` and `type` statements found nothing.

---

## 4. Claims in the PR body needing correction

- "CI: test on 3.10-3.12" is accurate as configuration and false in effect until S1 is fixed.
- The verification section reports `142 passed` without stating the interpreter. It was Python
  3.12 only, and CI has not corroborated it (I1).

---

## Not addressed here

- **Accuracy evaluation** remains unimplemented, so `benchmark.py --dataset` is accepted and
  ignored, and `accuracy={}` is hardcoded with a `# populate from eval_dataset if provided`
  comment. This is `SHRINK-018` and Task 9 of the Phase 7 plan, not a regression.
- **`SHRINK-020`** — restoring Core ML needs a real converter front end.
- **The identity dicts** in `convert_to_coreml.py` are left until that task rewrites the module.
