# Codebase audit — 2026-07-30

Full read of `scripts/`, `configs/`, `.github/workflows/`, packaging and docs, following the
PR #33 review in [`pr-33-findings.md`](pr-33-findings.md).

Every finding below was reproduced locally against the pinned dependency floors
(`onnxruntime 1.28`, `transformers 5.14`, `torch 2.13`, `coremltools 9.0`) unless the entry says
otherwise. Severity is about consequence:

- **critical** — a documented, advertised command cannot run at all
- **high** — silently wrong output, or a whole subsystem is dead
- **medium** — a flag, key or guarantee does not do what it says
- **low** — cosmetic or dead code

The headline: **the two core compression stages, quantization and magnitude pruning, both crash
on every invocation**, and the CoreML target is dead. The test suite does not exercise any of
them, which is why all three survived.

---

## 1. Quantization — `scripts/quantize.py`

### Q1 — every quantize run raises `TypeError` (critical)

```python
quantize_dynamic(..., nodes_to_exclude=list(skip_ops), optimize_model=True)
quantize_static(...,  nodes_to_exclude=list(skip_ops), optimize_model=True)
```

`optimize_model` is not a parameter of either function. It was removed from the onnxruntime
quantization API; `pyproject.toml` requires `onnxruntime>=1.18.0`, and no version in that range
accepts it. Reproduced:

```
TypeError: quantize_dynamic() got an unexpected keyword argument 'optimize_model'
```

Dropping the argument makes the same call succeed. This breaks the README Quick Start's second
step, `SHRINK-002`, `SHRINK-003`, and the `quantize` stage of every pipeline config.

`tests/test_quantization.py` has four tests and none of them calls `dynamic_quantize` or
`static_quantize` — it covers the calibration reader, the `DEFAULT_SKIP_OPS` constant and the
size logger only. That is the gap that let this ship.

### Q2 — `--skip-ops` protects nothing (high)

`nodes_to_exclude` is documented by onnxruntime as *"List of nodes names to exclude"*. The code
passes **op types** (`Softmax`, `LayerNormalization`, `Gelu`). No node is named `Softmax`, so the
list matches nothing and every sensitive op is quantized anyway.

`SHRINK-002`'s acceptance criterion "Must support skip_ops to protect sensitive layers" is
therefore unmet, and the accuracy loss it exists to prevent is happening silently.

**Fix**: resolve op types to the node names carrying them, from the loaded graph.

### Q3 — `--precision fp16` needs an undeclared dependency (medium)

`fp16_quantize` imports `onnxconverter_common`, which is not in `dependencies` or any extra. The
flag is reachable from the CLI and from `quantization.precision: fp16` in a config.

### Q4 — synthetic calibration data ignores the model's actual inputs (medium)

```python
for _ in range(min(64, self.max_samples)):
    yield {name: np.random.randn(1, 128).astype(np.float32) for name in self.input_names}
```

Every input gets `float32 [1, 128]` regardless of what it is. For `pixel_values` the model wants
`[1, 3, 384, 384]`; for `input_ids` it wants `int64`. Static quantization with no calibration
files logs one warning and then hands onnxruntime data of the wrong rank and dtype.

**Fix**: derive shape and dtype from the ONNX graph's input `type` instead of guessing.

### Q5 — `_log_size_comparison` divides by the input size (low)

`ZeroDivisionError` on a zero-byte input.

### Q6 — calibration data is unpickled from a user-supplied directory (high)

```python
data = np.load(f, allow_pickle=True).item()
```

`allow_pickle=True` executes arbitrary code contained in the `.npy` file. The directory comes
from `--calibration-data`, or from `quantization.calibration_data` / `benchmark.dataset` in a
pipeline config — all values a shared or downloaded config can set, pointing at files a user
never inspected. Loading a calibration set should not be able to run code.

**Fix**: store calibration samples as `.npz`, which maps names to arrays natively and loads with
`allow_pickle=False`. This also resolves T2, since a single format then serves both consumers.

---

## 2. Pruning — `scripts/prune.py`

### P1 — magnitude pruning crashes on every model in the README (critical)

```python
threshold = param.data.abs().quantile(args.sparsity)
```

`torch.quantile` raises above 2²⁴ elements. Reproduced:

```
16,777,216 elements  ok
16,777,217 elements  RuntimeError: quantile() input tensor is too large
```

Embedding matrices exceed that comfortably:

| Model | Embedding | Elements | `quantile` |
|---|---|---|---|
| BERT-base | 30522 × 768 | 23,440,896 | crash |
| Phi-3-mini | 32064 × 3072 | 98,500,608 | crash |
| Mistral-7B | 32000 × 4096 | 131,072,000 | crash |

`magnitude` is `run_pipeline.py`'s **default** method (`p.get("method", "magnitude")`), and the
`attention_heads` path falls back to the same code when no attention layers are found. So the
default pruning configuration fails on every target the README lists — including the smallest.

**Fix**: `torch.kthvalue`, which has no such limit and returns the identical threshold (verified
to 4 decimal places on a 20M-element tensor).

### P2 — activation averages divide by the wrong count (medium)

`collect_activations` accumulates over however many batches the loader yields, then divides by
`num_batches` unconditionally. A loader shorter than `num_batches` biases every neuron's
activation frequency toward zero, which is the quantity the pruning threshold is computed from.

### P3 — `--task` rejects `classification` (medium)

Same parity defect as `benchmark.py` (fixed in the PR-33 pass): `export_to_onnx.py` accepts
`classification`, `prune.py` does not, and `run_pipeline.py` passes the config's task to both.

---

## 3. Converters

### CM1 — the CoreML path is dead (high)

`convert_onnx_to_coreml` calls `ct.convert(onnx_model, ...)`. coremltools removed ONNX as an
input format in 6.0; `pyproject.toml` requires `>=7.2`. Reproduced on coremltools 9.0:

```
ValueError: Unable to determine the type of the model, i.e. the source framework.
Please provide the value of argument "source", from one of
["tensorflow", "pytorch", "milinternal"].
```

There is no argument that makes an ONNX `ModelProto` acceptable. `convert_coreml` is a stage in
`VALID_STAGES`, wired into `ocr_pipeline.yaml` and `audio_pipeline.yaml`, and it cannot work.

**Fix**: fail immediately with an actionable message naming the supported route, rather than
surfacing coremltools' generic "unable to determine the type" error five frames down. Restoring
the capability needs a real ONNX→TorchScript/MIL front end — filed as `SHRINK-020`, not
attempted here.

### CM2 — `--quantization fp16` raises (medium)

```python
op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="float16")
```

Reproduced: `ValueError: Invalid dtype float16. Only support int8/uint8/int4/uint4.`

FP16 weights are a `ct.convert(compute_precision=...)` setting, not a linear-quantizer dtype.
Unlike the int8 branch this one has no error handling, so it is a hard crash — and
`quantization: fp16` under `mobile.ios` is exactly what `ocr_pipeline.yaml` and
`audio_pipeline.yaml` configure.

### CM3 — the int8 fallback message is false (medium)

```python
except Exception as e:
    log.warning("INT8 quantization failed: %s. Falling back to FP16.", e)
```

Nothing falls back. The handler logs and proceeds with the **unquantized** model, then saves it
and reports success. A user reading the log believes they have an FP16 model.

### CM4 — an unknown deployment target is silently dropped (medium)

`getattr(ct.target, minimum_deployment_target, None)` passes `None` on a miss, so coremltools
picks its own default and the user's requested minimum is lost without a word.

### CM5 — `compute_units.replace("_", "_")` (low)

No-op.

### T1 — int8 TFLite conversion is misconfigured (high)

`convert_tf_to_tflite` sets `inference_input_type = inference_output_type = tf.int8`
unconditionally in the int8 branch, *then* attaches the representative dataset only if one was
supplied. Without a representative dataset TFLite cannot produce a full-integer model, and int8
I/O is rejected at convert time.

Separately, int8 I/O contradicts the on-device contract PR #33 documents — `int32 [1, 256]` in,
`float32 [1, 2]` out. Quantized weights with float I/O is the normal shape for that consumer.

### T2 — the two scripts disagree about the `.npy` format (high)

| Script | Reads | Yields |
|---|---|---|
| `quantize.py` | `np.load(f, allow_pickle=True).item()` — a pickled **dict** | dict per input name |
| `convert_to_tflite.py` | `np.load(f)` — a bare **array** | `[data]`, a one-element list |

`run_pipeline.py` sources the TFLite representative dataset from
`quantization.calibration_data` when `mobile.android.representative_dataset` is unset — so both
scripts are handed the *same* directory, and at most one of them can read it. The single-element
list also breaks any two-input model, which is every `legal`-task export.

### T3 — the `tflite` extra cannot install what the code uses (medium)

```toml
tflite = ["tensorflow>=2.16.0", "onnx-tf>=1.10.0"]
```

`convert_onnx_to_tf` prefers **`onnx2tf`**, which is in neither the extra nor the dependencies;
`onnx-tf` is only the fallback. And `onnx-tf 1.10.0` declares `tensorflow_addons`, which was
archived in May 2024 and supports TF ≤ 2.14 — it cannot coexist with `tensorflow>=2.16.0`.

So the documented install produces, at best, the fallback path, and more likely a resolution
failure.

### OM1 — `--target` does nothing (medium)

`optimize_graph(input, output, optimization_level, target, enable_nhwc)` uses neither `target`
nor `enable_nhwc` in its body. `--target android|ios` is an inert flag.

### OM2 — `--generate-ort` converts the whole directory (medium)

```python
convert_onnx_models_to_ort.convert_onnx_models_to_ort(str(input_path.parent))
```

Passing the *parent directory* converts every `.onnx` under it. In a pipeline output directory
that is the base export, the quantized model and the mobile-optimized model — silently, as a
side effect of asking for one.

### OM3 — `--enable-nhwc` silently degrades to a file copy (low)

`onnxruntime.tools.transpose_optimizer` does not exist in current onnxruntime (verified). The
`ImportError` handler copies the input to the output unchanged and warns, so the flag reports
success while doing nothing. The degradation is deliberate; the message should say the layout
was **not** converted.

---

## 4. Packaging

### PK1 — the local `datasets/` package shadows the HuggingFace `datasets` dependency (high)

`datasets/__init__.py` exists and `pyproject.toml` packages it via
`include = [..., "datasets*"]`. `run_pipeline.py` invokes stages as `scripts/<name>.py`, so it
must run from the repo root — which puts the root on `sys.path`. Reproduced:

```
>>> import datasets
resolved to: /home/nikos/Projects/shrink-llm/datasets/__init__.py
has load_dataset: False
```

`datasets>=2.19.0` is a declared dependency that is unreachable from the place the pipeline runs.
Any dataset-loading code — including the adapter `SHRINK-016` needs — gets an empty stub instead.

**Fix**: delete `datasets/__init__.py` and drop `datasets*` from the packages list. The directory
stays as a data folder, so config paths like `datasets/ocr/eval/` are unaffected, and PEP 420
gives a real installed package precedence over a namespace portion.

### PK2 — the `scripts/script-helpers` submodule is unused (medium)

`.gitmodules` declares it and the README tells users to `git submodule update --init
--recursive` twice, calling it required. The repository contains **no shell scripts at all**, and
nothing references the submodule.

---

## 5. Configs

### CF1 — `configs/legal_pipeline.yaml` cannot be run (high)

```
$ python scripts/run_pipeline.py --config configs/legal_pipeline.yaml --dry-run
run_pipeline.py: error: quantization.mode='gptq' produces a directory artifact incompatible
with ONNX-dependent stages: benchmark, convert_coreml, convert_onnx_mobile, convert_tflite.
```

`--stages` defaults to all stages, and the config selects GPTQ, so the shipped config errors out
before doing anything. The guard is correct; nothing tells the config which stages it supports.

**Fix**: let a config declare `stages:` and have `run_pipeline` use it as the default when
`--stages` is not passed.

### CF2 — the legal config's Android block is invalid (medium)

```yaml
mobile:
  android:
    format: gguf
    quantization: Q4_K_M
```

`build_stage_args` passes `mobile.android.quantization` straight to `convert_to_tflite.py
--quantization`, whose choices are `none|fp16|int8` — `Q4_K_M` is an argparse error. `format` is
read by nothing, and GGUF is a roadmap v1.2 item.

### CF3 — `quantization.bits` and `group_size` are never read (low)

`build_stage_args` forwards only `precision`, `mode`, `calibration_data`, `calibration_samples`
and `skip_ops`. `gptq_quantize` hardcodes `group_size=128` and derives bits from precision, so
both keys in `legal_pipeline.yaml` are decoration.

---

## 6. CI

### CI1 — only Python 3.11 is tested (medium)

`requires-python = ">=3.10"` and black targets `py310`/`py311`/`py312`, but `ci.yml` and
`pr-gate.yml` both pin `3.11`. Two of the three supported versions are never exercised.

### CI2 — two packaged directories are never linted (medium)

`ruff check scripts/ compression/ benchmarks/ tests/` omits `datasets/` and `mobile_deployment/`,
both of which `pyproject.toml` ships.

### CI3 — `pr-gate.yml` forwards an attacker-controlled branch name (low, not fixed here)

```yaml
release_branch: ${{ github.head_ref }}
```

`github.head_ref` is the PR's source branch name, which anyone opening a fork PR chooses. It is
passed as an input to `nikolareljin/ci-helpers`, and whether that is exploitable depends on
whether the reusable workflow interpolates it into a `run:` block — which is outside this
repository.

Blast radius is limited: the trigger is `pull_request`, not `pull_request_target`, so a fork PR
runs with a read-only token and no secrets. Recorded so the shared workflow gets checked; not
changed here, since the fix belongs in `ci-helpers`.

### CI4 — a failed release exits zero (medium)

```bash
git tag "v${VERSION}"        || echo "Tag already exists"
git push origin "v${VERSION}" || echo "Tag already pushed"
```

`|| echo` swallows *every* failure, not just the already-exists case — auth failures, protected
refs, network errors. The job goes green having released nothing.

---

## 7. Docs

### D1 / D2 — the documented stage order is backwards (medium)

`README.md`:

```
ONNX Export ──► Quantization ──► Pruning ──► Distillation ──► Fine-tune
```

`docs/compression_pipeline.md`: "2. Apply quantization and, where appropriate, pruning. 3. Run
distillation".

`VALID_STAGES` is `prune → distill → export → quantize → convert_* → benchmark`. Both documents
put quantization before pruning and distillation; the code prunes and distils on the PyTorch
model *before* exporting. A reader following either document builds the wrong mental model of
where their config's knobs apply.

### D3 — the README Quick Start's quantize step is broken

A consequence of Q1, fixed with it.

---

## Not fixed here

- **`SHRINK-020`** (new): restore CoreML support behind a real converter front end. CM1 is
  reduced to a clear failure, not repaired — repairing it means adding an ONNX→TorchScript or
  ONNX→MIL path, which is a feature, not a fix.
- **Accuracy evaluation.** `benchmark.py` still computes no accuracy, so `--min-accuracy` skips
  with a warning and `min_f1` / `max_cer` / `max_accuracy_drop_pct` remain unenforced. Tracked by
  `SHRINK-018`; the PR-33 pass made the silence audible, which is as far as it goes without an
  evaluation implementation.
- **The `distill.py` / `prune.py` training loops** are still commented-out scaffolding
  (`"inject dataset to enable"`). That is the known state of the project, not a regression.
