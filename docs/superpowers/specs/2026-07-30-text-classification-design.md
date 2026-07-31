# Text classification, and shrink-llm as a reusable component — design

Date: 2026-07-30
Issues: `SHRINK-015` … `SHRINK-019`
Status: proposed

---

## 1. Goal

Two goals, and the second constrains the first.

**Produce text classifiers.** ShrinkLLM compresses OCR, causal-LM and audio models. It cannot
produce a text classifier: `export_to_onnx.py`'s `classification` task is *image* classification,
`distill.py` is gated to causal-LM, nothing emits a tokenizer, and no accuracy-family gate is
enforced for any task. The gap analysis is in [`../../text_classification.md`](../../text_classification.md).

**Be consumable by several applications.** ShrinkLLM is a component in other projects' build
pipelines, not a standalone tool for one model. Multiple applications, on different runtimes,
with different label sets and different definitions of an expensive error, must be able to
consume its output without any of them re-deriving how a compressed model is laid out.

The second goal is why the artifact bundle is specified as a versioned contract rather than as
four files a script happens to write, and why nothing in the new code is specific to any one
application's domain.

**The framing this sets.** ShrinkLLM's job is to *prepare* models — classifiers today, generative
models alongside them — for use by applications it does not own. Text classification is the first
task to go through that path end to end, but it is an instance of the path, not the point of it.
Every design choice below that could have been made specific to classification is instead made at
the level of "a prepared model and its bundle", with the classification-specific parts confined to
`task_config`, `compression/metrics.py` and `compression/data/text_classification.py`. §9 lists the
extension points this buys and what they cost, which is close to nothing when taken up front and a
breaking schema change when taken later.

---

## 2. Decisions

| Question | Decision |
|---|---|
| Scope | All five issues, full depth. No scaffolding left behind. |
| Dataset input | JSONL (`{"text", "label"}`) **and** HuggingFace hub dataset ids. |
| Metrics | Shared `compression/metrics.py`, not classification-only code inside `benchmark.py`. |
| Distillation | Enable real training for **both** the classification and causal-LM paths. |
| Runtimes in the contract | TFLite and ORT Mobile. Core ML when `SHRINK-020` lands. |
| Delivery | A validated directory plus manifest. No archiving, no OTA transport. |
| Interface | CLI primary, plus a thin Python surface for callers driving stages from their own code. |

---

## 3. Architecture

New library code lives under `compression/`; scripts stay orchestration-only. `benchmark.py` is
already 369 lines covering latency, memory, size, reporting and gating, and evaluation does not
belong in it.

```
compression/
  data/
    __init__.py                 # load_split() dispatch
    sources.py                  # JSONL path or hub id -> records
    text_classification.py      # records -> tokenized Dataset + collator
    causal_lm.py                # records -> packed blocks; asserts shared vocab
  metrics.py                    # accuracy, precision, recall, F1 (binary/macro/micro/per-class)
  evaluation.py                 # runtime-agnostic predict loop -> predictions + metrics
  artifacts.py                  # bundle contract v1: schema, writer, reader, validator

scripts/
  export_to_onnx.py             # + text-classification task, single-input wrapper
  export_app_artifacts.py       # NEW: write a bundle
  validate_artifacts.py         # NEW: verify a bundle against the contract
  distill.py                    # + sequence classification, real training
  benchmark.py                  # + text-classification task, evaluation, accuracy gates
  run_pipeline.py               # + location-independent stage dispatch, accuracy gates

configs/text_classification_pipeline.yaml   # NEW, generic template
docs/app_integration.md                     # NEW, consumer-facing contract documentation
```

### Naming constraint

Nothing new may be importable as top-level `datasets`. The audit removed `datasets/__init__.py`
precisely because it shadowed the HuggingFace dependency for anything run from the repo root.
`compression.data` is safe; `compression/datasets.py` would technically work but reads as the
thing that was just deleted, and is therefore excluded.

---

## 4. The artifact contract

The bundle is the interface between ShrinkLLM and every consuming application. It is versioned
so that its format can change without silently breaking an application that has not updated.

### 4.1 Layout

```
artifacts/
  model.tflite                 # or model.ort / model.onnx
  tokenizer_vocab.txt          # WordPiece, line number == token id
  tokenizer_meta.json          # max_length, unk/pad/cls/sep tokens
  labels.txt                   # classification only; one per line, index-aligned
  manifest.json                # the contract
```

### 4.2 `manifest.json`, contract version 1

```jsonc
{
  "contract_version": 1,
  "task": "text-classification",
  "created_utc": "2026-07-30T18:00:00Z",
  "source_model": "org/student-model",

  "model": { "file": "model.tflite", "runtime": "tflite" },

  "input_signature": [
    { "name": "input_ids", "dtype": "int32", "shape": [1, 256] }
  ],
  "output_signature": [
    { "name": "probabilities", "dtype": "float32", "shape": [1, 2], "normalized": true }
  ],

  "tokenizer": {
    "kind": "wordpiece",
    "vocab_file": "tokenizer_vocab.txt",
    "meta_file": "tokenizer_meta.json"
  },

  "task_config": {
    "labels": ["negative", "positive"],
    "positive_label": "positive"
  },

  "files": {
    "model.tflite":        "sha256:…",
    "tokenizer_vocab.txt": "sha256:…",
    "tokenizer_meta.json": "sha256:…",
    "labels.txt":          "sha256:…"
  }
}
```

**Required keys**: `contract_version`, `task`, `model`, `input_signature`, `output_signature`,
`files`.

`labels` sits under `task_config`, **not** at the top level. A generative or causal-LM bundle has
no label array and no fixed-size output vector; making `labels` a required top-level key would
force a `contract_version` bump — a breaking change for every shipped consumer — the first time a
non-classification task produced a bundle. `task_config` is task-specific by construction and
carries that difference without touching the core schema.

`normalized: true` records that softmax is applied in-graph, so a consumer that normalises a
multi-class output by its sum and one that reads it as a probability distribution agree.

### 4.3 Validation

`shrink-validate-artifacts <dir>` and `compression.artifacts.validate_bundle(path)` check:

- `contract_version` is known
- every file in `files` exists and its sha256 matches
- the vocab is non-empty and line-indexed, with no duplicate tokens
- for classification, `len(labels)` equals the output signature's last dimension
- the declared `input_signature` matches the model file's actual signature, where the runtime can
  be loaded

A consuming application runs this in its own CI against a bundle it downloaded. A mismatched
vocabulary otherwise produces plausible-looking scores rather than an error, which is the failure
mode the whole bundle exists to prevent.

---

## 5. Components

### 5.1 `compression/data`

`sources.py` resolves one spec to records of `{"text": str, "label": str | int}`:

| Spec | Resolution |
|---|---|
| `data/train.jsonl` | read the JSONL file |
| `data/` | read `<split>.jsonl` from the directory |
| `sst2`, `org/dataset` | `datasets.load_dataset` from the hub |

An **existing path always wins** over hub resolution, so a local file is never silently fetched
from the network. Column names are configurable (`text_column`, `label_column`) because hub
datasets disagree — `sentence` versus `text` is common.

Labels resolve through the model config's `id2label`, so the on-device label file and the training
labels cannot drift. An unrecognised label string is an error, never a silent drop.

`text_classification.py` produces a tokenized `torch.utils.data.Dataset` plus a collator. When
teacher and student use **different tokenizers**, the collator emits both sets, the student's under
the usual keys and the teacher's under a `teacher_` prefix.

`causal_lm.py` packs text into fixed-length blocks and **asserts a shared vocabulary** between
teacher and student, failing with a diff when they differ.

### 5.2 Why the two paths differ

For **sequence classification**, the KL is over a fixed-size class vector, so teacher and student
may tokenize the same raw text differently — each with its own tokenizer. That is what makes a
DeBERTa-teacher / MobileBERT-student pair legal at all.

For **causal LM**, the KL is per-token over the vocabulary. Mismatched vocabularies make that
arithmetic meaningless: it would train happily and produce garbage. Hence the assertion.

Hidden-state alignment follows the same split. Classification aligns the **mean-pooled**
last hidden state over non-pad tokens, which is defined regardless of tokenization; causal LM
aligns full per-token hidden states, which is safe because the vocabulary — and therefore the
sequence length — is shared. The existing `HiddenStateProjector` handles differing hidden sizes in
both cases.

### 5.3 `compression/metrics.py`

Pure functions over `(y_true, y_pred)`: `accuracy`, and `precision`/`recall`/`f1` in `binary`,
`macro` and `micro` flavours plus per-class, returned as a plain dict. No model, runtime or I/O
dependency, so it is trivially testable against a hand-checked confusion matrix.

Binary averaging needs a positive class, hence `positive_label` in configs. For a consumer whose
expensive error is a false positive, "precision" means precision **on that class**; macro
averaging would dilute it with the easy negative class and report a comfortable number.

`max_cer` stays unenforced. It is OCR-specific — an edit distance, not a confusion matrix — and out
of scope here; `run_pipeline.py` continues to warn that it is not evaluated.

### 5.4 `compression/evaluation.py`

A runtime-agnostic loop: given an `inference_fn` (which `benchmark.py` already builds for ONNX
Runtime, TFLite and Core ML), a tokenizer, and evaluation records, produce predictions and hand
them to `metrics`. Batched, with a progress log. Knows nothing about argparse or report formats.

### 5.5 `scripts/export_to_onnx.py` — SHRINK-015

`SingleInputClassifier` derives `attention_mask` in-graph and applies softmax. Two shapes:

- default — one input `input_ids int32 [batch, max_length]`, one output `float32 [batch, num_labels]` whose rows sum to 1
- `--two-input-export` — `input_ids` and `attention_mask`, raw logits

`TASK_CONFIGS` is a static dict and cannot express both, so a `resolve_task_config(task, two_input,
max_length)` function replaces the direct lookup for this task; the other tasks keep their existing
entries.

The traced dummy input is `dtype=torch.int32` explicitly. `torch.randint` defaults to int64 and
`torch.onnx.export` records the dtype it traces, so without this the requirement is silently unmet.
Sequence length is fixed because consumers need it; batch stays dynamic.

The metadata JSON gains `num_labels`, `id2label` and `max_length`, which the artifact exporter reads
rather than re-deriving.

New flags: `--max-length` (default 256), `--two-input-export`.

### 5.6 `scripts/export_app_artifacts.py` — SHRINK-017

Writes a bundle via `compression.artifacts`. WordPiece only: a BPE or SentencePiece tokenizer has
no line-indexed `vocab.txt`, and the script fails saying so rather than writing a file a consumer
would misread.

### 5.7 `scripts/validate_artifacts.py`

CLI over `validate_bundle`. Non-zero exit on any violation, with every violation listed rather than
just the first.

### 5.8 `scripts/distill.py` — SHRINK-016

`SUPPORTED_TASKS = ("legal", "text-classification")`. Task-appropriate model loaders. Real training
enabled for both paths — the commented-out `DistillationTrainer` block becomes live code.

`compute_loss` grows a teacher-inputs path: keys prefixed `teacher_` are routed to the teacher, and
when absent the teacher receives the student's inputs, which is the shared-vocabulary case. This
keeps the existing causal-LM behaviour intact.

New flags: `--text-column`, `--label-column`, `--eval-dataset`, `--max-length`.

### 5.9 `scripts/benchmark.py` — SHRINK-019 and evaluation

Adds `text-classification` to `SUPPORTED_TASKS` with an `int32` `input_ids` dummy input; adds
`--tokenizer`, `--min-precision`, `--min-recall`, `--min-f1`; and populates
`BenchmarkResult.accuracy`, a declared-but-always-empty dict since it was written.

Gate semantics match the existing ones: a gate whose data cannot be computed **fails** rather than
passing silently.

### 5.10 `scripts/run_pipeline.py`

- Stage scripts resolve against the **installed package location**, not a relative `scripts/` path.
  The current `[sys.executable, f"scripts/{script}.py"]` only works when the process CWD is the
  ShrinkLLM checkout, which prevents any other project from driving the pipeline.
- Config-relative path resolution: `dataset:` and similar resolve relative to the **config file**,
  so an application can keep its config and data together in its own repository.
- The `task != "legal"` distill gate is removed, and a configured stage that cannot run the
  configured task now **exits non-zero** instead of being skipped.
- `min_precision`, `min_recall` and `min_f1` are passed through to `benchmark.py`.

### 5.11 `configs/text_classification_pipeline.yaml` — SHRINK-018

A documented generic template, not an application-specific config. Teacher, student, label set,
`positive_label` and every threshold are configuration. The "expensive error" that `min_precision`
guards is whatever the consuming application says it is; the library does not know which.

The teacher must be a **fine-tuned sequence classifier**. There is no credible off-the-shelf one to
name for an arbitrary task, so the template carries a placeholder and a comment saying it must be
fine-tuned on the consumer's labelled data first. Naming a real hub model that classifies something
else would be worse than a placeholder.

### 5.12 Packaging and the Python surface

New console entry points: `shrink-app-artifacts`, `shrink-validate-artifacts`,
`shrink-convert-tflite`, `shrink-convert-coreml`, `shrink-convert-onnx-mobile` — the converters have
none today.

The thin Python surface exports what is worth calling directly, with no new abstraction layer over
the existing functions:

```python
from compression.artifacts import write_bundle, read_bundle, validate_bundle
from compression.metrics import classification_metrics
from compression.data import load_split
```

### 5.13 `docs/app_integration.md`

Consumer-facing: the contract, a worked walkthrough of loading a bundle, the validation step, and
what a `contract_version` bump obliges a consumer to do.

---

## 6. Behaviour changes

Two are deliberate and visible:

1. **`run_pipeline.py` exits non-zero** when a configured stage cannot run the configured task,
   where it previously logged and skipped. This is `SHRINK-016`'s acceptance criterion. The
   `SKIPPED` manifest record added in the audit pass remains for stages skipped for other reasons,
   such as a missing `teacher`.
2. **Stage dispatch no longer depends on the working directory.** Any workflow that relied on
   running from the checkout continues to work; workflows that could not run from elsewhere now can.

---

## 7. Testing

Test-driven, following the existing suite's style.

- **Metrics** — hand-computed fixtures over a confusion matrix small enough to verify by eye,
  including the degenerate cases: a class with no predictions, a class with no true instances,
  and single-class input.
- **Data** — JSONL round-trips, hub-versus-path precedence, unknown-label rejection, column
  renaming, and the dual-tokenizer collator's `teacher_` keys.
- **Export** — a 2-layer `BertConfig` built in-process rather than a real hub download. Asserts one
  input, `int32`, rows summing to 1 within 1e-5, the two-input variant's shape, and numerical
  agreement with the PyTorch model within 1e-4.
- **Artifacts** — bundle round-trip, digest mismatch detection, label/output-shape mismatch
  detection, unknown `contract_version` rejection, and non-WordPiece rejection.
- **Distillation** — the classification path with deliberately mismatched tokenizers, and the
  causal-LM path's shared-vocabulary assertion.
- **Pipeline** — the new config resolves and every stage builds arguments; the accuracy gates reach
  `benchmark.py`; a stage that cannot run the configured task exits non-zero.
- One opt-in test marked `slow` exercises a real hub model end to end. It is excluded from the
  default run, since CI must not depend on downloading MobileBERT.

---

## 8. Out of scope

- **Core ML bundles** — blocked on `SHRINK-020`. The manifest's `runtime` field and signature blocks
  are already generic, so Core ML slots in without a `contract_version` bump.
- **Archiving and OTA transport** — how a bundle reaches a device is the consuming application's
  business.
- **`max_cer`** — OCR-specific, and it stays warning loudly that it is not evaluated.
- **Multi-label classification** — single-label multi-class only. Multi-label needs per-class
  thresholds and a different metric aggregation; it would go in `task_config` when needed.
- **Finishing any consuming application's integration.** The contract and validator make integration
  possible; performing one is that application's work.

---

## 9. Extension points

Deliberate accommodations for uses beyond this pass, none of which are built here:

- **Externally fine-tuned models.** Every loader goes through `from_pretrained`, which accepts a
  local directory as readily as a hub id, so a model produced by a separate fine-tuning tool is a
  valid `model:` or `teacher:` value with no changes. A test covers a local-directory model source.
- **Generative and other tasks.** `task_config` isolates task-specific manifest content, so a
  causal-LM bundle with no label array does not force a schema bump.
- **Other runtimes.** `model.runtime` plus the signature blocks describe a model without assuming
  its format.
- **Programmatic drivers.** Location-independent stage dispatch and the Python surface let another
  tool drive ShrinkLLM stages from its own build, rather than shelling out from a specific
  directory.

---

## 10. Issue mapping

| Issue | Delivered by |
|---|---|
| `SHRINK-015` | §5.5 export, §5.1 tokenization |
| `SHRINK-016` | §5.1 data, §5.2 dual-tokenizer design, §5.8 distillation |
| `SHRINK-017` | §4 contract, §5.6 exporter, §5.7 validator |
| `SHRINK-018` | §5.3 metrics, §5.4 evaluation, §5.9 gates, §5.11 config |
| `SHRINK-019` | §5.9 benchmark task |
