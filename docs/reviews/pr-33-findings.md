# PR #33 review — findings

PR: `docs: text-classification gap analysis and Phase 7 plan`
Branch: `feat/text-classification-task`
Reviewed: 2026-07-30

PR #33 is documentation and planning only. Its risk is therefore not runtime behaviour but
**accuracy**: a gap analysis is copied into the implementation that follows it, so a claim about
the code that is not true becomes a bug in Phase 7 rather than a typo in a document.

Every assertion in `docs/text_classification.md`, `docs/roadmap.md`, `docs/todos.yaml` and the
`README.md` diff was checked against the line of source it describes. Three did not survive.
Reviewing the surrounding pipeline for the same class of problem surfaced eight more.

Severity is about consequence, not size: **high** = the plan or a run gives a wrong answer,
**medium** = work is missed or a claim misleads, **low** = cosmetic.

---

## A. Findings in the PR

### A1 — `SHRINK-018` cites a gate that does not exist (high)

> "run_pipeline must enforce `min_precision` the way it enforces `min_f1`."
> — `docs/todos.yaml`, `SHRINK-018`

`run_pipeline.py` lists `min_f1` in `_known_unimplemented`, next to `max_cer`,
`max_accuracy_drop_pct` and `min_accuracy`. It is parsed, logged at `debug`, and then dropped —
no `--min-f1` argument is ever passed, and `benchmark.py` has no F1 gate to receive one.

`configs/legal_pipeline.yaml` sets `min_f1: 0.80` today and that threshold has never been
checked. So the acceptance criterion points the implementer at a reference implementation that
is itself a no-op — the likely outcome is `min_precision` getting wired the same way and
inheriting the same silence.

`docs/text_classification.md` §1.4 carries the softer version of the same error: "`min_f1` hides
that asymmetry" reads as though an F1 gate is running and merely measuring the wrong thing.

**Fix**: state that no accuracy-family gate is currently enforced, and make `SHRINK-018` require
the wiring rather than assume it. Tracked additionally as C4/C5 below, which fix the silence
itself.

### A2 — Phase 7 has no work item for the benchmark stage (high)

`benchmark.py` declares `--task` with `choices=["ocr", "legal", "audio"]`, and its
`build_dummy_inputs` has branches for exactly those three. `run_pipeline.py` passes the config's
`task` straight through to that flag.

So the end-to-end path in the "Definition of done" — export → quantize → TFLite → artifacts —
stops one stage short of where a pipeline run actually ends. A `task: text-classification` config
reaches the benchmark stage and dies on an argparse error, and no acceptance criterion in
`SHRINK-015` … `SHRINK-018` mentions `benchmark.py`.

**Fix**: add `SHRINK-019` for the benchmark stage, and make it a dependency of the Phase 7 exit
criteria.

### A3 — the README teacher/student pairing contradicts `SHRINK-016` (high)

```
| Text Classification *(planned)* | Phi-3 / Mistral | MobileBERT | < 30 MB |
```

`SHRINK-016` requires "KL is computed over class logits at the configured temperature". A causal
LM has no class logits — Phi-3 emits a distribution over ~32k vocabulary items, and MobileBERT
with a classification head emits two. There is no KL between them; the shapes do not even align.

Distilling a causal-LM teacher into a sequence classifier is a real technique, but it needs a
different construction (verbalised label tokens, or teacher logits restricted to label-word ids)
and none of that is in the plan. As written, an implementer following the README picks a teacher
that cannot satisfy the acceptance criteria of the issue.

**Fix**: name a sequence-classification teacher (DeBERTa-v3-base / BERT-large) in the table.

### A4 — the generated `tokenizer_config.json` collides with HuggingFace's file of that name (medium)

`docs/text_classification.md` §1.3 and `SHRINK-017` specify an artifact directory holding
`tokenizer_vocab.txt` ("HuggingFace `vocab.txt` verbatim") beside a generated
`tokenizer_config.json` carrying `max_length`, `unk_token`, `pad_token`, `cls_token`, `sep_token`.

`tokenizer_config.json` is already a HuggingFace filename with a different, much larger schema.
The result is a directory that looks like a tokenizer directory and is not one: pointing
`AutoTokenizer.from_pretrained` at it either fails confusingly or silently constructs a tokenizer
from a partial config. The exporter's whole purpose is to stop the model and its vocabulary
drifting apart; naming the artifact after an upstream file with different contents works against
that.

**Fix**: rename to `tokenizer_meta.json`.

### A5 — the `int32 [1, 256]` requirement has no mechanism behind it (medium)

`SHRINK-015` requires "exactly one input, `int32 [1, 256]`", and the exit criteria in
`roadmap.md` repeat it. But `torch.onnx.export` records the dtype of the dummy input it traces,
and the existing `build_dummy_inputs` produces `input_ids` with `torch.randint(...)`, whose
default dtype is `int64`. The wrapper sketch in §2 does not mention dtype either.

Nothing in the plan causes the requirement to be met, and an int64 input tensor is exactly the
kind of mismatch that shows up only once the `.tflite` is on a device.

**Fix**: state the `dtype=torch.int32` requirement where the wrapper is described.

### A6 — Phase 7 duplicates an existing roadmap item (low)

`roadmap.md` already lists, under "Future Phases": "**v1.1**: Add support for more tasks
(sentiment analysis, image segmentation)". Sentiment analysis is text classification. The new
Phase 7 supersedes half of that line without saying so, leaving two roadmap entries for the same
work.

**Fix**: drop sentiment analysis from the v1.1 line and point it at Phase 7.

### A7 — the described skip is quieter than the PR says (low)

`docs/text_classification.md` §1.2 says a text-classification run "silently runs *without* its
distillation stage". It is worse than that. `build_stage_args` returns `[]`, and
`run_pipeline.main` then does:

```python
if not stage_args and stage not in ("export",):
    log.warning("Stage '%s' produced no args, skipping", stage)
    continue
```

The `continue` fires before `results[stage]` is assigned and before `_update_manifest`. The
skipped stage is absent from the end-of-run summary **and** from `manifest.json` — so the
artifact record of the run does not merely mislabel the skip, it contains no evidence a
distillation stage was ever requested. See C6.

---

## B. Confirmed accurate

Recorded so the PR is not re-litigated on these points:

- `export_to_onnx.py`'s `classification` task really is image classification —
  `AutoModelForImageClassification`, `pixel_values` `[1, 3, 224, 224]`, no `input_ids` path.
- `distill.py` really does open `SUPPORTED_TASKS = ("legal",)` and load both models with
  `AutoModelForCausalLM`.
- `run_pipeline.py` really does emit `Distill stage supports only task='legal', skipping task='%s'`.
- No stage emits a tokenizer.
- `configs/` really has only `ocr`, `legal` and `audio`, and none carries `min_precision`.
- The `python scripts/...` invocations in §3 use flags that exist: `quantize.py` accepts
  `--input/--precision/--mode/--output`; `convert_to_tflite.py` accepts `--input/--output/--validate`.

---

## C. Findings in the codebase

Independent of the PR, found while verifying it. C4 and C5 are the mechanism behind A1.

### C1 — `distill.py` breaks on transformers ≥ 4.46 (high)

`TrainingArguments(evaluation_strategy=...)` was deprecated in 4.41 in favour of `eval_strategy`
and removed in 4.46. `pyproject.toml` pins `transformers>=4.40.0` with no upper bound, so a fresh
install of the project gets a version where `scripts/distill.py` raises `TypeError` before doing
any work.

**Fix**: use `eval_strategy`; raise the floor to `>=4.41.0`, the first release that accepts it.

### C2 — the KL term is scaled by sequence length (high)

```python
loss_kl = functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)
```

`reduction="batchmean"` divides the summed divergence by `input.size(0)`. For causal-LM logits of
shape `[B, T, V]` that is `B` — so the result is the mean **per sequence**, i.e. `T` times the
per-token value, while `loss_ce` beside it is a per-token mean.

`alpha=0.1` / `beta=0.9` therefore do not describe the ratio they appear to. With `T = 512` the
effective KL weight is roughly 4600× the CE weight, and it changes whenever `max_length` changes
— tuning done at one sequence length silently stops holding at another.

`tests/test_distillation.py` misses this: `test_alpha_beta_weights` sets `beta=0.0`, which is the
one setting under which the KL term cannot be observed.

**Fix**: flatten to `[B*T, V]` before `kl_div` so both terms are per-token means. Add a test that
asserts the two normalisations agree.

### C3 — causal-LM cross-entropy is computed against unshifted labels (high)

```python
loss_ce = functional.cross_entropy(
    student_logits.view(-1, student_logits.size(-1)),
    labels.view(-1),
)
```

A causal LM's logits at position `t` predict token `t+1`. HuggingFace models shift internally when
you pass `labels=`, but `DistillationTrainer.compute_loss` pops `labels` out of `inputs` and does
the cross-entropy by hand — without the shift. The student is trained to predict the token it was
just given, which is the identity function and learnable to near-zero loss while teaching nothing.

The task gate makes this the *only* CE path that runs today (`SUPPORTED_TASKS = ("legal",)`), so
the one supported task is the one that is wrong.

**Fix**: shift when the logits are 3-D (sequence models) and leave 2-D (classification) logits
alone — which is also the shape `SHRINK-016` needs.

### C4 — recognised-but-unwired success criteria are hidden at `debug` (high)

```python
for key in sorted(set(criteria) & _known_unimplemented):
    log.debug("success_criteria key %r is recognized but not yet wired ...")
```

The default handler is `INFO`, so this never prints. A config author writes `min_f1: 0.80`, the
run reports `✓ benchmark: OK`, and nothing anywhere says the threshold was not checked. Compare
the *unknown*-key branch immediately above, which does warn — an unrecognised key is loud and an
unenforced one is silent, which is backwards.

**Fix**: `log.warning`, naming the key and saying the gate was not evaluated.

### C5 — `min_accuracy` is dropped even though the gate exists (high)

`benchmark.py` implements `--min-accuracy` (`evaluate_gates`, line 205). `run_pipeline.py`
classifies `min_accuracy` as unimplemented and never passes it. `configs/audio_pipeline.yaml`
sets `min_accuracy: 0.88`.

So the audio pipeline has had a working gate and a configured threshold and has never connected
them. (`benchmark.py` will still skip the gate for want of accuracy data, but it says so at
`warning` — which is the point: the failure becomes visible.)

**Fix**: wire `--min-accuracy` through; move the key out of `_known_unimplemented`.

### C6 — skipped stages leave no record (medium)

As described in A7: the `continue` at `run_pipeline.py:749` precedes both `results[stage] = ...`
and `_update_manifest(...)`. A skipped stage vanishes from the summary and the manifest.
`manifest.json` is the artifact record of a run — a stage that was requested and did not run
belongs in it.

**Fix**: record `SKIPPED` in the summary and append a `skipped` stage entry to the manifest.

### C7 — `benchmark.py` rejects a task `export_to_onnx.py` accepts (medium)

`export_to_onnx.py` exports `classification`; `benchmark.py` accepts only `ocr`, `legal`, `audio`.
A `task: classification` pipeline runs four stages and then fails at the fifth on
`argument --task: invalid choice`. This is the same shape as A2 and worth fixing now so Phase 7
does not add a second instance of it.

**Fix**: add `classification` to the choices and a matching `pixel_values [1, 3, 224, 224]` branch
in `build_dummy_inputs`, mirroring the exporter.

### C8 — `--no-save-config` help names a file that is never written (low)

`export_to_onnx.py`'s help says "Skip saving model_config.json"; `save_model_config` writes
`output_path.with_suffix(".json")`, i.e. `model_int8.json` for `model_int8.onnx`.

**Fix**: describe the actual path.

---

## Not changed

- `tests/test_distillation.py::TestSupportedTasks` asserts `SUPPORTED_TASKS == ("legal",)`. It is
  a change-detector that `SHRINK-016` must update, which is the correct time to do so.
- Making `run_pipeline` **fail** rather than skip when a stage cannot run the configured task is
  `SHRINK-016`'s job and is a breaking change to existing configs. C6 only makes the skip
  visible and recorded; the exit-code change stays with the issue that owns it.
