# Text classification — the gap, and what closes it

Status: proposed
Last updated: 2026-07-30

ShrinkLLM compresses OCR, causal-LM ("legal") and audio models today. It cannot currently
produce a **text classifier**, which is the artifact a downstream consumer needs
([the downstream consumer](https://example.invalid/downstream-consumer): an on-device scam detector that runs a
TFLite classifier through `tflite_flutter`).

This document records what is missing, why each piece is missing, and what "done" looks like.
The corresponding issues are `SHRINK-015` … `SHRINK-018` in [`todos.yaml`](todos.yaml).

---

## 1. What is missing

### 1.1 `classification` means *image* classification

`scripts/export_to_onnx.py` advertises a `classification` task, and a reader reasonably assumes
it covers text. It does not:

```python
elif task == "classification":
    from transformers import AutoFeatureExtractor, AutoModelForImageClassification

    processor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
```

and the dummy input is `pixel_values` of shape `[1, 3, 224, 224]`. There is no path that loads
`AutoModelForSequenceClassification` or traces `input_ids`.

**Consequence**: exporting MobileBERT, DistilBERT or any sequence classifier fails at load with
a config/architecture mismatch, not with a useful message.

### 1.2 Distillation is causal-LM only

`scripts/distill.py` opens with:

```python
SUPPORTED_TASKS = ("legal",)
```

and `run_pipeline.py` logs `Distill stage supports only task='legal', skipping task='%s'` for
anything else — so a text-classification pipeline silently runs *without* its distillation
stage, producing an undistilled student and no warning that the headline feature was skipped.

The loss machinery itself is task-agnostic: `DistillationLoss` is CE + KL + optional hidden
alignment, which is exactly what sequence-classification distillation needs. Only the task gate,
the model loaders and the dataset adapter are LLM-shaped.

### 1.3 Nothing emits the tokenizer

A compressed classifier is not usable on a phone by itself. The consumer needs, beside the
`.tflite`:

| File | Format |
|---|---|
| `tokenizer_vocab.txt` | one WordPiece token per line, line number is the id |
| `tokenizer_config.json` | `max_length`, `unk_token`, `pad_token`, and (see §2) `cls_token`, `sep_token` |
| `<task>_labels.txt` | one label per line, index-aligned with the model's output |

HuggingFace's `vocab.txt` is already the first format verbatim, so this is a copy plus two small
generated files — but no stage in the pipeline emits them, and each consumer inventing its own
layout is how the model and its tokenizer drift apart.

### 1.4 No pipeline config for the task

`configs/` has `ocr_pipeline.yaml`, `legal_pipeline.yaml`, `audio_pipeline.yaml`. A text
classification config needs one addition to the `success_criteria` shape: **`min_precision`**.
For a reporting product a false positive is far more expensive than a false negative, and an F1
gate hides that asymmetry.

---

## 2. Two constraints the consumer's runtime imposes

These are worth recording here because they change what the *export* stage must produce. They
were found by reading the downstream consumer's runtime against MobileBERT's input signature; the full analysis
is in that repo's `docs/ON_DEVICE_MODEL_PIPELINE.md`.

**One input tensor.** The consumer calls `Interpreter.run(input, output)` — a single input. A
`transformers` sequence classifier takes `input_ids` *and* `attention_mask`. The export must
therefore wrap the model so the mask is derived inside the graph:

```python
class SingleInputClassifier(torch.nn.Module):
    """Derive the attention mask in-graph so the exported model takes one input."""

    def __init__(self, model, pad_id: int):
        super().__init__()
        self.model, self.pad_id = model, pad_id

    def forward(self, input_ids):
        attention_mask = (input_ids != self.pad_id).long()
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return torch.softmax(logits, dim=-1)
```

Two-input export should stay available behind a flag, since not every consumer has that limit.

**Softmax in the graph.** The consumer normalises a multi-class output by its **sum**, not with
a softmax. Exporting raw logits therefore produces a number that looks like a probability and is
not one. The wrapper above applies softmax, which makes both readings agree.

---

## 3. Definition of done

An off-the-shelf `google/mobilebert-uncased` fine-tuned on any two-class text dataset can be
taken through:

```bash
python scripts/export_to_onnx.py --model <student> --task text-classification --output model.onnx
python scripts/quantize.py --input model.onnx --precision int8 --mode dynamic --output model_int8.onnx
python scripts/convert_to_tflite.py --input model_int8.onnx --output model.tflite --validate
python scripts/export_app_artifacts.py --model <student> --output artifacts/
```

and the result loads and scores on-device without the consumer writing conversion code. Size
≤ 30 MB, single input `int32 [1, 256]`, output `float32 [1, 2]` summing to 1.

Distillation (`SHRINK-016`) is the accuracy work and comes after that path exists end to end —
the cheapest proof that the plumbing is right needs no teacher, no corpus and no GPU.
