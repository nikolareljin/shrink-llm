# Text Classification & Component Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ShrinkLLM produce text classifiers end to end, and make its output consumable by applications it does not own, via a versioned artifact contract.

**Architecture:** New library modules under `compression/` (data loading, metrics, evaluation, artifact contract) keep the `scripts/` CLIs as thin orchestration. Classification-specific behaviour is confined to `compression/metrics.py`, `compression/data/text_classification.py` and the manifest's `task_config` block, so generative and other tasks slot in without a schema break.

**Tech Stack:** Python 3.10–3.12, PyTorch, HuggingFace `transformers` + `datasets`, ONNX / onnxruntime, TensorFlow Lite, pytest.

**Spec:** `docs/superpowers/specs/2026-07-30-text-classification-design.md`

## Global Constraints

- **Never create a top-level module or package named `datasets`.** It shadows the HuggingFace dependency for anything run from the repo root, which is where `run_pipeline.py` runs every stage. New data code goes in `compression/data/`.
- Python floor is 3.10; `pyproject.toml` declares `requires-python = ">=3.10"`. CI runs 3.10, 3.11 and 3.12.
- Line length 100 (`ruff` and `black` are both configured to it). `ruff` lint selects `["E", "F", "W", "I", "N", "UP"]`.
- Lint and format command: `ruff check scripts/ compression/ benchmarks/ mobile_deployment/ tests/` and `black --check` over the same paths.
- Full test command: `pytest -q`. All tests must pass before any commit.
- `transformers>=4.41.0` — use `TrainingArguments(eval_strategy=...)`, never `evaluation_strategy`.
- Calibration and sample data on disk is `.npz` loaded with `allow_pickle=False`. Never `allow_pickle=True`.
- `torch.quantile` raises above 2**24 elements — use `scripts.prune.magnitude_threshold` for any thresholding over model parameters.
- Contract version for this work is `1`. Do not bump it.
- No commits are pushed. Commit locally only; the user pushes.
- Do not name any specific downstream application in code, config, or documentation.

---

### Task 1: Classification metrics

**Files:**
- Create: `compression/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], num_labels: int, positive_index: int | None = None) -> dict`
    returning keys `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `precision_micro`, `recall_micro`, `f1_micro`, `per_class` (list of dicts with `label_index`, `precision`, `recall`, `f1`, `support`), and — only when `positive_index` is not None — `precision_binary`, `recall_binary`, `f1_binary`.
  - `confusion_counts(y_true, y_pred, num_labels) -> list[dict]` with `tp`, `fp`, `fn`, `support` per class index.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for classification metrics."""

from __future__ import annotations

import pytest


class TestConfusionCounts:
    def test_counts_are_per_class(self):
        from compression.metrics import confusion_counts

        # class 0: predicted 0 twice, once correctly. class 1: predicted 1 twice, both correct.
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]

        counts = confusion_counts(y_true, y_pred, num_labels=2)

        assert counts[0] == {"tp": 1, "fp": 0, "fn": 1, "support": 2}
        assert counts[1] == {"tp": 2, "fp": 1, "fn": 0, "support": 2}

    def test_rejects_mismatched_lengths(self):
        from compression.metrics import confusion_counts

        with pytest.raises(ValueError, match="same length"):
            confusion_counts([0, 1], [0], num_labels=2)

    def test_rejects_out_of_range_labels(self):
        from compression.metrics import confusion_counts

        with pytest.raises(ValueError, match="out of range"):
            confusion_counts([0, 2], [0, 0], num_labels=2)


class TestClassificationMetrics:
    def test_hand_checked_confusion_matrix(self):
        from compression.metrics import classification_metrics

        # 6 samples. class 1 ("positive") is the class we care about.
        #   true: 0 0 0 1 1 1
        #   pred: 0 0 1 1 1 0
        # class 1: tp=2, fp=1, fn=1 -> precision 2/3, recall 2/3, f1 2/3
        # class 0: tp=2, fp=1, fn=1 -> precision 2/3, recall 2/3, f1 2/3
        y_true = [0, 0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 1, 0]

        m = classification_metrics(y_true, y_pred, num_labels=2, positive_index=1)

        assert m["accuracy"] == pytest.approx(4 / 6)
        assert m["precision_binary"] == pytest.approx(2 / 3)
        assert m["recall_binary"] == pytest.approx(2 / 3)
        assert m["f1_binary"] == pytest.approx(2 / 3)
        assert m["precision_macro"] == pytest.approx(2 / 3)

    def test_binary_keys_absent_without_positive_index(self):
        from compression.metrics import classification_metrics

        m = classification_metrics([0, 1], [0, 1], num_labels=2)

        assert "precision_binary" not in m
        assert m["accuracy"] == pytest.approx(1.0)

    def test_class_with_no_predictions_scores_zero_not_nan(self):
        """A class the model never predicts has undefined precision; report 0.0."""
        from compression.metrics import classification_metrics

        m = classification_metrics([0, 0, 1], [0, 0, 0], num_labels=2, positive_index=1)

        assert m["precision_binary"] == 0.0
        assert m["recall_binary"] == 0.0
        assert m["f1_binary"] == 0.0

    def test_class_with_no_true_instances_scores_zero_not_nan(self):
        from compression.metrics import classification_metrics

        m = classification_metrics([0, 0], [0, 1], num_labels=2, positive_index=1)

        assert m["recall_binary"] == 0.0
        assert m["per_class"][1]["support"] == 0

    def test_micro_average_equals_accuracy_for_single_label(self):
        """With one prediction per sample, micro-F1 collapses to accuracy."""
        from compression.metrics import classification_metrics

        y_true = [0, 1, 2, 1, 0]
        y_pred = [0, 1, 1, 1, 2]

        m = classification_metrics(y_true, y_pred, num_labels=3)

        assert m["f1_micro"] == pytest.approx(m["accuracy"])

    def test_perfect_predictions(self):
        from compression.metrics import classification_metrics

        m = classification_metrics([0, 1, 2], [0, 1, 2], num_labels=3, positive_index=2)

        assert m["accuracy"] == 1.0
        assert m["f1_macro"] == 1.0
        assert m["precision_binary"] == 1.0

    def test_rejects_positive_index_out_of_range(self):
        from compression.metrics import classification_metrics

        with pytest.raises(ValueError, match="positive_index"):
            classification_metrics([0], [0], num_labels=2, positive_index=5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'compression.metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
"""
metrics.py — Classification metrics for compression quality gates.

Pure functions over label sequences. No model, runtime or I/O dependency, so a gate can be
checked without loading anything.
"""

from __future__ import annotations

from collections.abc import Sequence


def _safe_ratio(numerator: int, denominator: int) -> float:
    """Ratio, or 0.0 when undefined.

    A class the model never predicts has undefined precision, and a class with no true
    instances has undefined recall. Reporting 0.0 keeps gates comparable; NaN would make
    every threshold comparison silently false.
    """
    return numerator / denominator if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_ratio(int(2 * precision * recall * 1_000_000), int((precision + recall) * 1_000_000))


def confusion_counts(
    y_true: Sequence[int], y_pred: Sequence[int], num_labels: int
) -> list[dict[str, int]]:
    """Per-class true positives, false positives, false negatives and support."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must be the same length, got {len(y_true)} and {len(y_pred)}"
        )
    counts = [{"tp": 0, "fp": 0, "fn": 0, "support": 0} for _ in range(num_labels)]
    for true, pred in zip(y_true, y_pred):
        if not (0 <= true < num_labels) or not (0 <= pred < num_labels):
            raise ValueError(
                f"label out of range for num_labels={num_labels}: true={true}, pred={pred}"
            )
        counts[true]["support"] += 1
        if true == pred:
            counts[true]["tp"] += 1
        else:
            counts[pred]["fp"] += 1
            counts[true]["fn"] += 1
    return counts


def _prf(count: dict[str, int]) -> tuple[float, float, float]:
    precision = _safe_ratio(count["tp"], count["tp"] + count["fp"])
    recall = _safe_ratio(count["tp"], count["tp"] + count["fn"])
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_labels: int,
    positive_index: int | None = None,
) -> dict:
    """Accuracy plus precision/recall/F1 in macro, micro and (optionally) binary flavours.

    positive_index selects the class for the binary metrics. Consumers whose expensive error
    is a false positive on one specific class need precision on *that* class; macro averaging
    dilutes it with the easy classes and reports a comfortable number.
    """
    if positive_index is not None and not (0 <= positive_index < num_labels):
        raise ValueError(
            f"positive_index {positive_index} out of range for num_labels={num_labels}"
        )

    counts = confusion_counts(y_true, y_pred, num_labels)
    per_class = []
    for index, count in enumerate(counts):
        precision, recall, f1 = _prf(count)
        per_class.append(
            {
                "label_index": index,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": count["support"],
            }
        )

    total = len(y_true)
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)

    tp = sum(c["tp"] for c in counts)
    fp = sum(c["fp"] for c in counts)
    fn = sum(c["fn"] for c in counts)
    micro_precision = _safe_ratio(tp, tp + fp)
    micro_recall = _safe_ratio(tp, tp + fn)
    micro_f1 = (
        0.0
        if (micro_precision + micro_recall) == 0
        else 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    )

    result = {
        "accuracy": _safe_ratio(correct, total),
        "precision_macro": _safe_ratio(int(sum(c["precision"] for c in per_class) * 1e9), num_labels * int(1e9)),
        "recall_macro": _safe_ratio(int(sum(c["recall"] for c in per_class) * 1e9), num_labels * int(1e9)),
        "f1_macro": _safe_ratio(int(sum(c["f1"] for c in per_class) * 1e9), num_labels * int(1e9)),
        "precision_micro": micro_precision,
        "recall_micro": micro_recall,
        "f1_micro": micro_f1,
        "per_class": per_class,
    }

    if positive_index is not None:
        entry = per_class[positive_index]
        result["precision_binary"] = entry["precision"]
        result["recall_binary"] = entry["recall"]
        result["f1_binary"] = entry["f1"]

    return result
```

**Note for the implementer:** the `_safe_ratio(int(... * 1e9), ...)` pattern above is
deliberately clumsy — replace the macro averages with plain
`sum(...) / num_labels if num_labels else 0.0` and delete the unused `_f1` helper. The tests
define the contract; write the clean version that passes them.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_metrics.py -q`
Expected: PASS, 9 tests

- [ ] **Step 5: Lint, format, full suite**

Run: `.venv/bin/ruff check compression/ tests/ && .venv/bin/black --check compression/ tests/ && .venv/bin/python -m pytest -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add compression/metrics.py tests/test_metrics.py
git commit -m "feat: add classification metrics module"
```

---

### Task 2: Dataset sources

**Files:**
- Create: `compression/data/__init__.py`, `compression/data/sources.py`
- Test: `tests/test_data_sources.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `load_records(spec: str | Path, split: str = "train", text_column: str = "text", label_column: str = "label") -> list[dict]` returning `[{"text": str, "label": str | int}, ...]`.
  - `resolve_source(spec) -> tuple[str, Path | str]` returning `("path", Path)` or `("hub", str)`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for dataset source resolution."""

from __future__ import annotations

import json

import pytest


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


class TestResolveSource:
    def test_existing_file_resolves_to_path(self, tmp_path):
        from compression.data.sources import resolve_source

        f = tmp_path / "train.jsonl"
        f.write_text("")
        kind, value = resolve_source(f)
        assert kind == "path"
        assert value == f

    def test_existing_directory_resolves_to_path(self, tmp_path):
        from compression.data.sources import resolve_source

        kind, value = resolve_source(tmp_path)
        assert kind == "path"

    def test_nonexistent_spec_resolves_to_hub(self):
        from compression.data.sources import resolve_source

        kind, value = resolve_source("glue/sst2")
        assert kind == "hub"
        assert value == "glue/sst2"

    def test_existing_path_wins_over_hub_lookup(self, tmp_path, monkeypatch):
        """A local file must never be silently fetched from the network."""
        from compression.data.sources import resolve_source

        monkeypatch.chdir(tmp_path)
        (tmp_path / "sst2").mkdir()
        kind, _ = resolve_source("sst2")
        assert kind == "path"


class TestLoadRecords:
    def test_reads_a_jsonl_file(self, tmp_path):
        from compression.data.sources import load_records

        f = tmp_path / "train.jsonl"
        _write_jsonl(f, [{"text": "hello", "label": "a"}, {"text": "bye", "label": "b"}])

        records = load_records(f)

        assert records == [
            {"text": "hello", "label": "a"},
            {"text": "bye", "label": "b"},
        ]

    def test_reads_split_from_a_directory(self, tmp_path):
        from compression.data.sources import load_records

        _write_jsonl(tmp_path / "eval.jsonl", [{"text": "x", "label": "a"}])

        records = load_records(tmp_path, split="eval")

        assert records == [{"text": "x", "label": "a"}]

    def test_renames_columns(self, tmp_path):
        from compression.data.sources import load_records

        f = tmp_path / "train.jsonl"
        _write_jsonl(f, [{"sentence": "hi", "target": 1}])

        records = load_records(f, text_column="sentence", label_column="target")

        assert records == [{"text": "hi", "label": 1}]

    def test_missing_split_file_names_the_directory(self, tmp_path):
        from compression.data.sources import load_records

        with pytest.raises(FileNotFoundError, match="test.jsonl"):
            load_records(tmp_path, split="test")

    def test_missing_column_is_an_error(self, tmp_path):
        from compression.data.sources import load_records

        f = tmp_path / "train.jsonl"
        _write_jsonl(f, [{"text": "hi"}])

        with pytest.raises(ValueError, match="label"):
            load_records(f)

    def test_blank_lines_are_skipped(self, tmp_path):
        from compression.data.sources import load_records

        f = tmp_path / "train.jsonl"
        f.write_text('{"text": "a", "label": "x"}\n\n{"text": "b", "label": "y"}\n')

        assert len(load_records(f)) == 2

    def test_malformed_json_reports_the_line_number(self, tmp_path):
        from compression.data.sources import load_records

        f = tmp_path / "train.jsonl"
        f.write_text('{"text": "a", "label": "x"}\nnot json\n')

        with pytest.raises(ValueError, match="line 2"):
            load_records(f)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_data_sources.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'compression.data'`

- [ ] **Step 3: Write minimal implementation**

`compression/data/__init__.py`:

```python
"""Dataset loading for compression stages.

Deliberately NOT named `datasets`: a top-level package of that name shadows the HuggingFace
dependency for anything run from the repo root, which is where run_pipeline.py runs stages.
"""

from compression.data.sources import load_records, resolve_source

__all__ = ["load_records", "resolve_source"]
```

`compression/data/sources.py`:

```python
"""
sources.py — Resolve a dataset spec to records of {"text", "label"}.

A spec is either a filesystem path (a .jsonl file, or a directory holding <split>.jsonl) or a
HuggingFace hub dataset id. An existing path always wins, so a local file is never silently
fetched from the network.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def resolve_source(spec: str | Path) -> tuple[str, Path | str]:
    """Return ("path", Path) for anything that exists on disk, else ("hub", str)."""
    path = Path(spec)
    if path.exists():
        return "path", path
    return "hub", str(spec)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: malformed JSON on line {line_number}: {exc}") from exc
    return rows


def _project(rows, text_column: str, label_column: str, origin: str) -> list[dict]:
    records = []
    for index, row in enumerate(rows):
        if text_column not in row:
            raise ValueError(
                f"{origin}: record {index} has no {text_column!r} column; "
                f"available: {sorted(row)}. Pass text_column= to rename."
            )
        if label_column not in row:
            raise ValueError(
                f"{origin}: record {index} has no {label_column!r} column; "
                f"available: {sorted(row)}. Pass label_column= to rename."
            )
        records.append({"text": row[text_column], "label": row[label_column]})
    return records


def load_records(
    spec: str | Path,
    split: str = "train",
    text_column: str = "text",
    label_column: str = "label",
) -> list[dict]:
    """Load one split as a list of {"text", "label"} records."""
    kind, value = resolve_source(spec)

    if kind == "path":
        path = Path(value)
        if path.is_dir():
            path = path / f"{split}.jsonl"
            if not path.exists():
                raise FileNotFoundError(
                    f"No {split}.jsonl in {value}. Expected <split>.jsonl per split."
                )
        rows = _read_jsonl(path)
        return _project(rows, text_column, label_column, str(path))

    log.info("Loading hub dataset %r split %r", value, split)
    from datasets import load_dataset

    dataset = load_dataset(str(value), split=split)
    return _project(list(dataset), text_column, label_column, f"hub:{value}[{split}]")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_data_sources.py -q`
Expected: PASS, 11 tests

- [ ] **Step 5: Lint, format, full suite**

Run: `.venv/bin/ruff check compression/ tests/ && .venv/bin/black --check compression/ tests/ && .venv/bin/python -m pytest -q`

- [ ] **Step 6: Commit**

```bash
git add compression/data tests/test_data_sources.py
git commit -m "feat: add dataset source resolution for jsonl and hub specs"
```

---

### Task 3: Text-classification dataset and dual-tokenizer collator

**Files:**
- Create: `compression/data/text_classification.py`
- Modify: `compression/data/__init__.py`
- Test: `tests/test_data_text_classification.py`

**Interfaces:**
- Consumes: `compression.data.sources.load_records`.
- Produces:
  - `build_label_map(id2label: dict[int, str]) -> dict[str, int]`
  - `encode_labels(records: list[dict], label_map: dict[str, int]) -> list[int]`
  - `TextClassificationDataset(records: list[dict], label_map: dict[str, int])` — a `torch.utils.data.Dataset` yielding `{"text": str, "label": int}`
  - `DualTokenizerCollator(student_tokenizer, max_length: int, teacher_tokenizer=None)` — callable returning a dict with `input_ids`, `attention_mask`, `labels`, and when a distinct teacher tokenizer is given, `teacher_input_ids` and `teacher_attention_mask`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the text-classification dataset and collator."""

from __future__ import annotations

import pytest
import torch


class _FakeTokenizer:
    """Minimal stand-in: encodes each character as its ordinal, offset by a vocab marker."""

    def __init__(self, marker: int = 0, vocab: set[str] | None = None):
        self.marker = marker
        self.vocab = vocab or set()

    def __call__(self, texts, truncation=True, padding="max_length", max_length=8,
                 return_tensors="pt"):
        ids = []
        mask = []
        for text in texts:
            row = [self.marker + ord(c) for c in text][:max_length]
            attention = [1] * len(row)
            pad = max_length - len(row)
            ids.append(row + [0] * pad)
            mask.append(attention + [0] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

    def get_vocab(self):
        return {token: index for index, token in enumerate(sorted(self.vocab))}


class TestBuildLabelMap:
    def test_inverts_id2label(self):
        from compression.data.text_classification import build_label_map

        assert build_label_map({0: "negative", 1: "positive"}) == {"negative": 0, "positive": 1}

    def test_accepts_string_keys(self):
        """transformers configs round-trip id2label through JSON, making keys strings."""
        from compression.data.text_classification import build_label_map

        assert build_label_map({"0": "a", "1": "b"}) == {"a": 0, "b": 1}

    def test_rejects_duplicate_label_names(self):
        from compression.data.text_classification import build_label_map

        with pytest.raises(ValueError, match="duplicate"):
            build_label_map({0: "same", 1: "same"})


class TestEncodeLabels:
    def test_maps_string_labels(self):
        from compression.data.text_classification import encode_labels

        records = [{"text": "a", "label": "positive"}, {"text": "b", "label": "negative"}]
        assert encode_labels(records, {"negative": 0, "positive": 1}) == [1, 0]

    def test_passes_through_integer_labels_in_range(self):
        from compression.data.text_classification import encode_labels

        records = [{"text": "a", "label": 1}]
        assert encode_labels(records, {"negative": 0, "positive": 1}) == [1]

    def test_unknown_label_is_an_error_not_a_silent_drop(self):
        from compression.data.text_classification import encode_labels

        records = [{"text": "a", "label": "spam"}]
        with pytest.raises(ValueError, match="spam"):
            encode_labels(records, {"negative": 0, "positive": 1})

    def test_integer_label_out_of_range_is_an_error(self):
        from compression.data.text_classification import encode_labels

        with pytest.raises(ValueError, match="out of range"):
            encode_labels([{"text": "a", "label": 7}], {"negative": 0, "positive": 1})


class TestDataset:
    def test_length_and_items(self):
        from compression.data.text_classification import TextClassificationDataset

        records = [{"text": "a", "label": "positive"}, {"text": "b", "label": "negative"}]
        ds = TextClassificationDataset(records, {"negative": 0, "positive": 1})

        assert len(ds) == 2
        assert ds[0] == {"text": "a", "label": 1}


class TestDualTokenizerCollator:
    def test_single_tokenizer_emits_student_keys_only(self):
        from compression.data.text_classification import DualTokenizerCollator

        collate = DualTokenizerCollator(_FakeTokenizer(), max_length=8)
        batch = collate([{"text": "ab", "label": 1}, {"text": "cd", "label": 0}])

        assert set(batch) == {"input_ids", "attention_mask", "labels"}
        assert batch["input_ids"].shape == (2, 8)
        assert batch["labels"].tolist() == [1, 0]

    def test_distinct_teacher_tokenizer_adds_prefixed_keys(self):
        """Classification KL is over class logits, so the two models may tokenize differently."""
        from compression.data.text_classification import DualTokenizerCollator

        student = _FakeTokenizer(marker=0)
        teacher = _FakeTokenizer(marker=1000)
        collate = DualTokenizerCollator(student, max_length=8, teacher_tokenizer=teacher)

        batch = collate([{"text": "ab", "label": 1}])

        assert "teacher_input_ids" in batch
        assert "teacher_attention_mask" in batch
        assert batch["teacher_input_ids"][0][0].item() != batch["input_ids"][0][0].item()

    def test_same_tokenizer_object_does_not_duplicate_inputs(self):
        from compression.data.text_classification import DualTokenizerCollator

        shared = _FakeTokenizer()
        collate = DualTokenizerCollator(shared, max_length=8, teacher_tokenizer=shared)

        batch = collate([{"text": "ab", "label": 1}])

        assert "teacher_input_ids" not in batch

    def test_labels_are_long_for_cross_entropy(self):
        from compression.data.text_classification import DualTokenizerCollator

        collate = DualTokenizerCollator(_FakeTokenizer(), max_length=8)
        batch = collate([{"text": "a", "label": 1}])

        assert batch["labels"].dtype == torch.long
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_data_text_classification.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
"""
text_classification.py — Labelled-text datasets for distillation and evaluation.

Teacher and student may use different tokenizers. That is legal here and nowhere else in the
pipeline: sequence-classification KL is computed over a fixed-size class vector, so each model
can tokenize the same raw text its own way. Causal-LM distillation cannot do this, because its
KL is per-token over the vocabulary — see compression/data/causal_lm.py.
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


def build_label_map(id2label: dict) -> dict[str, int]:
    """Invert a transformers config's id2label into name -> index.

    Keys arrive as ints in memory and as strings after a JSON round-trip; both are accepted.
    """
    label_map: dict[str, int] = {}
    for raw_index, name in id2label.items():
        index = int(raw_index)
        if name in label_map:
            raise ValueError(
                f"id2label has duplicate label name {name!r} (indices "
                f"{label_map[name]} and {index}); label names must be unique"
            )
        label_map[name] = index
    return label_map


def encode_labels(records: list[dict], label_map: dict[str, int]) -> list[int]:
    """Map each record's label to its index.

    An unrecognised label is an error rather than a silent drop: quietly discarding records
    changes the class balance the model trains on without saying so.
    """
    num_labels = len(label_map)
    encoded = []
    for index, record in enumerate(records):
        label = record["label"]
        if isinstance(label, bool):
            raise ValueError(f"record {index}: boolean label {label!r} is ambiguous")
        if isinstance(label, int):
            if not (0 <= label < num_labels):
                raise ValueError(
                    f"record {index}: integer label {label} out of range for "
                    f"{num_labels} labels"
                )
            encoded.append(label)
            continue
        if label not in label_map:
            raise ValueError(
                f"record {index}: unknown label {label!r}; "
                f"the model's id2label declares {sorted(label_map)}"
            )
        encoded.append(label_map[label])
    return encoded


class TextClassificationDataset(Dataset):
    """Raw text plus an integer label. Tokenization happens in the collator."""

    def __init__(self, records: list[dict], label_map: dict[str, int]):
        self.texts = [record["text"] for record in records]
        self.labels = encode_labels(records, label_map)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict:
        return {"text": self.texts[index], "label": self.labels[index]}


class DualTokenizerCollator:
    """Tokenize a batch for the student, and for a distinct teacher when there is one.

    Teacher tensors are emitted under a `teacher_` prefix. When the teacher shares the
    student's tokenizer, no prefixed keys are emitted at all and the trainer reuses the
    student's inputs — which keeps the causal-LM path's behaviour unchanged.
    """

    def __init__(self, student_tokenizer, max_length: int, teacher_tokenizer=None):
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.max_length = max_length
        self.tokenizers_differ = (
            teacher_tokenizer is not None
            and teacher_tokenizer is not student_tokenizer
            and self._vocab_of(teacher_tokenizer) != self._vocab_of(student_tokenizer)
        )

    @staticmethod
    def _vocab_of(tokenizer):
        getter = getattr(tokenizer, "get_vocab", None)
        return getter() if callable(getter) else None

    def _encode(self, tokenizer, texts):
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __call__(self, features: list[dict]) -> dict:
        texts = [feature["text"] for feature in features]
        encoded = self._encode(self.student_tokenizer, texts)
        batch = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor([f["label"] for f in features], dtype=torch.long),
        }
        if self.tokenizers_differ:
            teacher_encoded = self._encode(self.teacher_tokenizer, texts)
            batch["teacher_input_ids"] = teacher_encoded["input_ids"]
            batch["teacher_attention_mask"] = teacher_encoded["attention_mask"]
        return batch
```

Then extend `compression/data/__init__.py`:

```python
from compression.data.sources import load_records, resolve_source
from compression.data.text_classification import (
    DualTokenizerCollator,
    TextClassificationDataset,
    build_label_map,
    encode_labels,
)

__all__ = [
    "DualTokenizerCollator",
    "TextClassificationDataset",
    "build_label_map",
    "encode_labels",
    "load_records",
    "resolve_source",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_data_text_classification.py -q`
Expected: PASS, 12 tests

- [ ] **Step 5: Lint, format, full suite**

Run: `.venv/bin/ruff check compression/ tests/ && .venv/bin/black --check compression/ tests/ && .venv/bin/python -m pytest -q`

- [ ] **Step 6: Commit**

```bash
git add compression/data tests/test_data_text_classification.py
git commit -m "feat: add text-classification dataset and dual-tokenizer collator"
```

---

### Task 4: Causal-LM dataset with shared-vocabulary assertion

**Files:**
- Create: `compression/data/causal_lm.py`
- Modify: `compression/data/__init__.py`
- Test: `tests/test_data_causal_lm.py`

**Interfaces:**
- Consumes: `compression.data.sources.load_records`.
- Produces:
  - `assert_shared_vocabulary(teacher_tokenizer, student_tokenizer) -> None`, raising `ValueError` when vocabularies differ.
  - `CausalLMBlockDataset(records: list[dict], tokenizer, block_size: int)` — a `Dataset` yielding `{"input_ids": Tensor, "attention_mask": Tensor, "labels": Tensor}`.
  - `causal_lm_collator(features) -> dict` stacking those tensors.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for causal-LM packing and the shared-vocabulary requirement."""

from __future__ import annotations

import pytest
import torch


class _VocabTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab
        self.eos_token_id = 0

    def get_vocab(self):
        return dict(self._vocab)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self._vocab.get(ch, 1) for ch in text]}


class TestAssertSharedVocabulary:
    def test_identical_vocabularies_pass(self):
        from compression.data.causal_lm import assert_shared_vocabulary

        vocab = {"a": 0, "b": 1}
        assert_shared_vocabulary(_VocabTokenizer(vocab), _VocabTokenizer(dict(vocab))) is None

    def test_differing_vocabularies_raise(self):
        """Per-token KL across mismatched vocabularies trains happily and produces garbage."""
        from compression.data.causal_lm import assert_shared_vocabulary

        with pytest.raises(ValueError, match="vocabular"):
            assert_shared_vocabulary(
                _VocabTokenizer({"a": 0, "b": 1}),
                _VocabTokenizer({"a": 0, "c": 1}),
            )

    def test_error_names_the_differing_tokens(self):
        from compression.data.causal_lm import assert_shared_vocabulary

        with pytest.raises(ValueError) as excinfo:
            assert_shared_vocabulary(
                _VocabTokenizer({"a": 0, "b": 1}),
                _VocabTokenizer({"a": 0, "c": 1}),
            )
        assert "size" in str(excinfo.value) or "b" in str(excinfo.value) or "c" in str(excinfo.value)

    def test_size_mismatch_is_reported_before_content(self):
        from compression.data.causal_lm import assert_shared_vocabulary

        with pytest.raises(ValueError, match="2 vs 3"):
            assert_shared_vocabulary(
                _VocabTokenizer({"a": 0, "b": 1}),
                _VocabTokenizer({"a": 0, "b": 1, "c": 2}),
            )


class TestCausalLMBlockDataset:
    def test_packs_into_fixed_blocks(self):
        from compression.data.causal_lm import CausalLMBlockDataset

        tokenizer = _VocabTokenizer({ch: i for i, ch in enumerate("abcdefghij")})
        records = [{"text": "abcde", "label": None}, {"text": "fghij", "label": None}]

        ds = CausalLMBlockDataset(records, tokenizer, block_size=4)

        # 10 tokens packed into blocks of 4 -> 2 whole blocks, remainder dropped
        assert len(ds) == 2
        assert ds[0]["input_ids"].shape == (4,)

    def test_labels_mirror_input_ids(self):
        """The trainer's CE shifts internally; the dataset supplies unshifted labels."""
        from compression.data.causal_lm import CausalLMBlockDataset

        tokenizer = _VocabTokenizer({ch: i for i, ch in enumerate("abcdefgh")})
        ds = CausalLMBlockDataset([{"text": "abcdefgh"}], tokenizer, block_size=4)

        item = ds[0]
        assert torch.equal(item["labels"], item["input_ids"])

    def test_attention_mask_is_all_ones_for_packed_blocks(self):
        from compression.data.causal_lm import CausalLMBlockDataset

        tokenizer = _VocabTokenizer({ch: i for i, ch in enumerate("abcdefgh")})
        ds = CausalLMBlockDataset([{"text": "abcdefgh"}], tokenizer, block_size=4)

        assert ds[0]["attention_mask"].tolist() == [1, 1, 1, 1]

    def test_too_little_text_is_an_error(self):
        from compression.data.causal_lm import CausalLMBlockDataset

        tokenizer = _VocabTokenizer({ch: i for i, ch in enumerate("abc")})
        with pytest.raises(ValueError, match="block_size"):
            CausalLMBlockDataset([{"text": "ab"}], tokenizer, block_size=8)


class TestCausalLMCollator:
    def test_stacks_a_batch(self):
        from compression.data.causal_lm import CausalLMBlockDataset, causal_lm_collator

        tokenizer = _VocabTokenizer({ch: i for i, ch in enumerate("abcdefgh")})
        ds = CausalLMBlockDataset([{"text": "abcdefgh"}], tokenizer, block_size=4)

        batch = causal_lm_collator([ds[0], ds[1]])

        assert batch["input_ids"].shape == (2, 4)
        assert batch["labels"].shape == (2, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_data_causal_lm.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
"""
causal_lm.py — Packed-block datasets for causal-LM distillation.

Unlike sequence classification, causal-LM distillation requires teacher and student to share a
vocabulary: its KL is computed per token over the vocabulary, so mismatched vocabularies make
the comparison meaningless arithmetic that trains happily and produces garbage. The assertion
here is what stops that from being a silent failure.
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


def assert_shared_vocabulary(teacher_tokenizer, student_tokenizer) -> None:
    """Raise unless the two tokenizers agree on every token and id."""
    teacher_vocab = teacher_tokenizer.get_vocab()
    student_vocab = student_tokenizer.get_vocab()

    if len(teacher_vocab) != len(student_vocab):
        raise ValueError(
            "Causal-LM distillation requires a shared vocabulary, but the tokenizers differ "
            f"in size: {len(teacher_vocab)} vs {len(student_vocab)}. Its KL is per-token over "
            "the vocabulary, so mismatched vocabularies compare unrelated distributions."
        )

    if teacher_vocab != student_vocab:
        only_teacher = sorted(set(teacher_vocab) - set(student_vocab))[:5]
        only_student = sorted(set(student_vocab) - set(teacher_vocab))[:5]
        raise ValueError(
            "Causal-LM distillation requires a shared vocabulary, but the tokenizers differ. "
            f"Teacher-only tokens (first 5): {only_teacher}. "
            f"Student-only tokens (first 5): {only_student}."
        )


class CausalLMBlockDataset(Dataset):
    """Concatenate every record's text and cut it into fixed-length blocks.

    Labels mirror input_ids unshifted; DistillationLoss shifts them when
    DistillationConfig.shift_labels is set, which is the causal-LM configuration.
    """

    def __init__(self, records: list[dict], tokenizer, block_size: int):
        token_ids: list[int] = []
        for record in records:
            encoded = tokenizer(record["text"], add_special_tokens=False)
            token_ids.extend(encoded["input_ids"])
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None:
                token_ids.append(eos)

        num_blocks = len(token_ids) // block_size
        if num_blocks == 0:
            raise ValueError(
                f"Corpus has {len(token_ids)} tokens, fewer than one block_size of "
                f"{block_size}. Supply more text or lower block_size."
            )
        usable = num_blocks * block_size
        if usable < len(token_ids):
            log.info("Dropping %d trailing tokens to fill whole blocks", len(token_ids) - usable)

        self.blocks = torch.tensor(token_ids[:usable], dtype=torch.long).view(
            num_blocks, block_size
        )

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, index: int) -> dict:
        block = self.blocks[index]
        return {
            "input_ids": block,
            "attention_mask": torch.ones_like(block),
            "labels": block.clone(),
        }


def causal_lm_collator(features: list[dict]) -> dict:
    """Stack pre-shaped blocks; no padding is needed since every block is block_size long."""
    return {
        key: torch.stack([feature[key] for feature in features])
        for key in ("input_ids", "attention_mask", "labels")
    }
```

Add to `compression/data/__init__.py` exports: `CausalLMBlockDataset`, `assert_shared_vocabulary`, `causal_lm_collator`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_data_causal_lm.py -q`
Expected: PASS, 10 tests

- [ ] **Step 5: Lint, format, full suite**

- [ ] **Step 6: Commit**

```bash
git add compression/data tests/test_data_causal_lm.py
git commit -m "feat: add causal-LM block dataset with shared-vocabulary assertion"
```

---

### Task 5: Artifact bundle contract

**Files:**
- Create: `compression/artifacts.py`
- Test: `tests/test_artifacts.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `CONTRACT_VERSION: int = 1`
  - `sha256_file(path: Path) -> str` returning `"sha256:<hex>"`
  - `TensorSpec` — `TypedDict`-shaped dict with `name`, `dtype`, `shape`
  - `write_bundle(output_dir, *, task, model_path, runtime, input_signature, output_signature, vocab_path=None, tokenizer_meta=None, labels=None, positive_label=None, source_model=None) -> Path` returning the manifest path
  - `read_bundle(bundle_dir) -> dict`
  - `validate_bundle(bundle_dir) -> list[str]` returning violation strings, empty when valid

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the app artifact bundle contract."""

from __future__ import annotations

import json

import pytest


def _make_bundle(tmp_path, **overrides):
    from compression.artifacts import write_bundle

    model = tmp_path / "model.tflite"
    model.write_bytes(b"fake tflite bytes")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\nworld\n")

    kwargs = dict(
        task="text-classification",
        model_path=model,
        runtime="tflite",
        input_signature=[{"name": "input_ids", "dtype": "int32", "shape": [1, 256]}],
        output_signature=[
            {"name": "probabilities", "dtype": "float32", "shape": [1, 2], "normalized": True}
        ],
        vocab_path=vocab,
        tokenizer_meta={"max_length": 256, "unk_token": "[UNK]", "pad_token": "[PAD]",
                        "cls_token": "[CLS]", "sep_token": "[SEP]"},
        labels=["negative", "positive"],
        positive_label="positive",
        source_model="org/student",
    )
    kwargs.update(overrides)
    out = tmp_path / "artifacts"
    return write_bundle(out, **kwargs), out


class TestWriteBundle:
    def test_writes_every_declared_file(self, tmp_path):
        manifest_path, out = _make_bundle(tmp_path)

        assert manifest_path.name == "manifest.json"
        for name in ("model.tflite", "tokenizer_vocab.txt", "tokenizer_meta.json", "labels.txt"):
            assert (out / name).exists(), name

    def test_vocab_is_byte_identical_to_the_source(self, tmp_path):
        _, out = _make_bundle(tmp_path)

        assert (out / "tokenizer_vocab.txt").read_bytes() == (tmp_path / "vocab.txt").read_bytes()

    def test_labels_are_one_per_line_in_index_order(self, tmp_path):
        _, out = _make_bundle(tmp_path)

        assert (out / "labels.txt").read_text().splitlines() == ["negative", "positive"]

    def test_manifest_carries_the_contract_version(self, tmp_path):
        manifest_path, _ = _make_bundle(tmp_path)

        assert json.loads(manifest_path.read_text())["contract_version"] == 1

    def test_labels_live_under_task_config_not_top_level(self, tmp_path):
        """A generative bundle has no labels; keeping them nested avoids a version bump."""
        manifest_path, _ = _make_bundle(tmp_path)
        manifest = json.loads(manifest_path.read_text())

        assert "labels" not in manifest
        assert manifest["task_config"]["labels"] == ["negative", "positive"]
        assert manifest["task_config"]["positive_label"] == "positive"

    def test_required_core_keys_are_present(self, tmp_path):
        manifest_path, _ = _make_bundle(tmp_path)
        manifest = json.loads(manifest_path.read_text())

        for key in ("contract_version", "task", "model", "input_signature",
                    "output_signature", "files"):
            assert key in manifest, key

    def test_every_file_has_a_digest(self, tmp_path):
        manifest_path, out = _make_bundle(tmp_path)
        files = json.loads(manifest_path.read_text())["files"]

        assert set(files) == {"model.tflite", "tokenizer_vocab.txt", "tokenizer_meta.json",
                              "labels.txt"}
        assert all(digest.startswith("sha256:") for digest in files.values())

    def test_bundle_without_labels_omits_labels_file(self, tmp_path):
        """Task-agnostic: a bundle for a task with no label set is still valid."""
        _, out = _make_bundle(tmp_path, task="causal-lm", labels=None, positive_label=None)

        assert not (out / "labels.txt").exists()

    def test_rejects_positive_label_not_in_labels(self, tmp_path):
        with pytest.raises(ValueError, match="positive_label"):
            _make_bundle(tmp_path, positive_label="nonexistent")


class TestValidateBundle:
    def test_a_freshly_written_bundle_is_valid(self, tmp_path):
        from compression.artifacts import validate_bundle

        _, out = _make_bundle(tmp_path)

        assert validate_bundle(out) == []

    def test_detects_a_tampered_file(self, tmp_path):
        from compression.artifacts import validate_bundle

        _, out = _make_bundle(tmp_path)
        (out / "tokenizer_vocab.txt").write_text("different content\n")

        violations = validate_bundle(out)

        assert any("tokenizer_vocab.txt" in v and "digest" in v for v in violations)

    def test_detects_a_missing_file(self, tmp_path):
        from compression.artifacts import validate_bundle

        _, out = _make_bundle(tmp_path)
        (out / "labels.txt").unlink()

        assert any("labels.txt" in v and "missing" in v for v in validate_bundle(out))

    def test_detects_label_count_mismatch_with_output_shape(self, tmp_path):
        """A mismatched vocab or label set produces plausible scores, never an error."""
        from compression.artifacts import validate_bundle

        manifest_path, out = _make_bundle(tmp_path)
        manifest = json.loads(manifest_path.read_text())
        manifest["output_signature"][0]["shape"] = [1, 5]
        manifest_path.write_text(json.dumps(manifest))

        assert any("label" in v for v in validate_bundle(out))

    def test_rejects_an_unknown_contract_version(self, tmp_path):
        from compression.artifacts import validate_bundle

        manifest_path, out = _make_bundle(tmp_path)
        manifest = json.loads(manifest_path.read_text())
        manifest["contract_version"] = 99
        manifest_path.write_text(json.dumps(manifest))

        assert any("contract_version" in v for v in validate_bundle(out))

    def test_detects_a_duplicate_vocab_token(self, tmp_path):
        from compression.artifacts import validate_bundle, write_bundle

        vocab = tmp_path / "dup.txt"
        vocab.write_text("[PAD]\nhello\nhello\n")
        _, out = _make_bundle(tmp_path, vocab_path=vocab)

        assert any("duplicate" in v for v in validate_bundle(out))

    def test_missing_manifest_is_a_violation_not_a_crash(self, tmp_path):
        from compression.artifacts import validate_bundle

        empty = tmp_path / "empty"
        empty.mkdir()

        assert any("manifest.json" in v for v in validate_bundle(empty))


class TestReadBundle:
    def test_round_trips(self, tmp_path):
        from compression.artifacts import read_bundle

        _, out = _make_bundle(tmp_path)

        manifest = read_bundle(out)

        assert manifest["task"] == "text-classification"
        assert manifest["model"]["runtime"] == "tflite"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_artifacts.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'compression.artifacts'`

- [ ] **Step 3: Write minimal implementation**

Key points the implementer must honour:

- `CONTRACT_VERSION = 1`, `SUPPORTED_CONTRACT_VERSIONS = frozenset({1})`
- `write_bundle` copies the model and vocab with `shutil.copyfile` (byte-identical), writes `tokenizer_meta.json` and `labels.txt`, computes digests **after** all files are written, and writes `manifest.json` last so a partial bundle has no manifest to mislead a consumer
- `manifest["files"]` must not include `manifest.json` itself — its digest cannot cover itself
- `created_utc` uses `datetime.now(timezone.utc).isoformat()`
- `validate_bundle` returns a **list of every** violation, never raising on the first, so a consumer's CI reports everything at once
- `validate_bundle` reads `labels.txt` and compares its length with `output_signature[-1]["shape"][-1]`

```python
"""
artifacts.py — The on-device artifact bundle contract.

A bundle is the interface between ShrinkLLM and applications that consume its models. It is
versioned so the format can change without silently breaking a consumer that has not updated,
and validatable so a consumer can check a bundle in its own CI rather than discovering a
mismatched vocabulary as plausible-looking scores.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

CONTRACT_VERSION = 1
SUPPORTED_CONTRACT_VERSIONS = frozenset({1})

MANIFEST_NAME = "manifest.json"
MODEL_NAMES = {"tflite": "model.tflite", "onnxruntime_mobile": "model.ort", "onnx": "model.onnx"}
VOCAB_NAME = "tokenizer_vocab.txt"
TOKENIZER_META_NAME = "tokenizer_meta.json"
LABELS_NAME = "labels.txt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def write_bundle(
    output_dir: Path,
    *,
    task: str,
    model_path: Path,
    runtime: str,
    input_signature: list[dict],
    output_signature: list[dict],
    vocab_path: Path | None = None,
    tokenizer_meta: dict | None = None,
    labels: list[str] | None = None,
    positive_label: str | None = None,
    source_model: str | None = None,
) -> Path:
    """Write a contract-conformant bundle and return the manifest path."""
    if labels is not None and positive_label is not None and positive_label not in labels:
        raise ValueError(
            f"positive_label {positive_label!r} is not among labels {labels}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = MODEL_NAMES.get(runtime, Path(model_path).name)
    shutil.copyfile(model_path, output_dir / model_name)

    written = [model_name]

    if vocab_path is not None:
        shutil.copyfile(vocab_path, output_dir / VOCAB_NAME)
        written.append(VOCAB_NAME)
    if tokenizer_meta is not None:
        (output_dir / TOKENIZER_META_NAME).write_text(json.dumps(tokenizer_meta, indent=2))
        written.append(TOKENIZER_META_NAME)
    if labels is not None:
        (output_dir / LABELS_NAME).write_text("\n".join(labels) + "\n")
        written.append(LABELS_NAME)

    manifest: dict = {
        "contract_version": CONTRACT_VERSION,
        "task": task,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_model": source_model,
        "model": {"file": model_name, "runtime": runtime},
        "input_signature": input_signature,
        "output_signature": output_signature,
    }
    if vocab_path is not None:
        manifest["tokenizer"] = {
            "kind": "wordpiece",
            "vocab_file": VOCAB_NAME,
            "meta_file": TOKENIZER_META_NAME if tokenizer_meta is not None else None,
        }
    task_config: dict = {}
    if labels is not None:
        task_config["labels"] = labels
    if positive_label is not None:
        task_config["positive_label"] = positive_label
    if task_config:
        manifest["task_config"] = task_config

    # Digests last, and manifest.json last of all: a partial bundle then has no manifest to
    # mislead a consumer, and no digest can cover the file that contains it.
    manifest["files"] = {name: sha256_file(output_dir / name) for name in written}

    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Wrote bundle to %s (%d files)", output_dir, len(written))
    return manifest_path


def read_bundle(bundle_dir: Path) -> dict:
    return json.loads((Path(bundle_dir) / MANIFEST_NAME).read_text())


def validate_bundle(bundle_dir: Path) -> list[str]:
    """Return every contract violation. Empty list means the bundle is valid."""
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.exists():
        return [f"{MANIFEST_NAME} is missing from {bundle_dir}"]

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{MANIFEST_NAME} is not valid JSON: {exc}"]

    violations: list[str] = []

    version = manifest.get("contract_version")
    if version not in SUPPORTED_CONTRACT_VERSIONS:
        violations.append(
            f"unknown contract_version {version!r}; this build understands "
            f"{sorted(SUPPORTED_CONTRACT_VERSIONS)}"
        )

    for key in ("task", "model", "input_signature", "output_signature", "files"):
        if key not in manifest:
            violations.append(f"manifest is missing required key {key!r}")
    if violations and "files" not in manifest:
        return violations

    for name, expected in manifest.get("files", {}).items():
        path = bundle_dir / name
        if not path.exists():
            violations.append(f"{name} is missing but declared in the manifest")
            continue
        actual = sha256_file(path)
        if actual != expected:
            violations.append(f"{name} digest mismatch: expected {expected}, got {actual}")

    labels_path = bundle_dir / LABELS_NAME
    if labels_path.exists():
        labels = [line for line in labels_path.read_text().splitlines() if line]
        output_signature = manifest.get("output_signature") or [{}]
        shape = output_signature[-1].get("shape") or []
        if shape and shape[-1] != len(labels):
            violations.append(
                f"label count {len(labels)} does not match output shape {shape} "
                f"(last dimension {shape[-1]})"
            )

    vocab_path = bundle_dir / VOCAB_NAME
    if vocab_path.exists():
        tokens = vocab_path.read_text().splitlines()
        if not tokens:
            violations.append(f"{VOCAB_NAME} is empty")
        seen: set[str] = set()
        duplicates = {token for token in tokens if token in seen or seen.add(token)}
        if duplicates:
            violations.append(
                f"{VOCAB_NAME} has duplicate token(s): {sorted(duplicates)[:5]}; "
                "line number must be the token id"
            )

    return violations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_artifacts.py -q`
Expected: PASS, 18 tests

- [ ] **Step 5: Lint, format, full suite**

- [ ] **Step 6: Commit**

```bash
git add compression/artifacts.py tests/test_artifacts.py
git commit -m "feat: add versioned app artifact bundle contract"
```

---

### Task 6: Text-classification ONNX export (SHRINK-015)

**Files:**
- Modify: `scripts/export_to_onnx.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `SingleInputClassifier(model, pad_id: int)` — `nn.Module` taking `input_ids` only, returning softmaxed probabilities
  - `resolve_task_config(task: str, two_input: bool = False) -> dict` with keys `input_names`, `output_names`, `dynamic_axes`
  - `build_dummy_inputs(task, device, max_length: int = 256, two_input: bool = False)` — extended signature
  - `TASK_CONFIGS` gains a `"text-classification"` key

- [ ] **Step 1: Write the failing test** (append to `tests/test_export.py`)

```python
class TestSingleInputClassifier:
    def _tiny_classifier(self, num_labels=2):
        from transformers import BertConfig, BertForSequenceClassification

        config = BertConfig(
            vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=16, max_position_embeddings=32, num_labels=num_labels,
        )
        return BertForSequenceClassification(config).eval()

    def test_takes_one_input_and_returns_probabilities(self):
        import torch

        from scripts.export_to_onnx import SingleInputClassifier

        wrapped = SingleInputClassifier(self._tiny_classifier(), pad_id=0)
        input_ids = torch.randint(1, 64, (2, 8), dtype=torch.int32)

        out = wrapped(input_ids)

        assert out.shape == (2, 2)
        assert torch.allclose(out.sum(-1), torch.ones(2), atol=1e-5)

    def test_derives_the_attention_mask_from_pad_ids(self):
        import torch

        from scripts.export_to_onnx import SingleInputClassifier

        model = self._tiny_classifier()
        wrapped = SingleInputClassifier(model, pad_id=0)

        padded = torch.tensor([[5, 6, 0, 0]], dtype=torch.int32)
        unpadded = torch.tensor([[5, 6]], dtype=torch.int32)

        # Padding must not change the result, which it only can if the mask is derived.
        assert torch.allclose(wrapped(padded), wrapped(unpadded), atol=1e-4)


class TestTextClassificationTaskConfig:
    def test_task_is_registered(self):
        from scripts.export_to_onnx import TASK_CONFIGS

        assert "text-classification" in TASK_CONFIGS

    def test_single_input_config_has_one_input_and_probabilities(self):
        from scripts.export_to_onnx import resolve_task_config

        config = resolve_task_config("text-classification", two_input=False)

        assert config["input_names"] == ["input_ids"]
        assert config["output_names"] == ["probabilities"]

    def test_two_input_config_adds_attention_mask_and_returns_logits(self):
        from scripts.export_to_onnx import resolve_task_config

        config = resolve_task_config("text-classification", two_input=True)

        assert config["input_names"] == ["input_ids", "attention_mask"]
        assert config["output_names"] == ["logits"]

    def test_other_tasks_are_unchanged(self):
        from scripts.export_to_onnx import TASK_CONFIGS, resolve_task_config

        assert resolve_task_config("ocr") == TASK_CONFIGS["ocr"]

    def test_batch_axis_is_dynamic_and_sequence_is_fixed(self):
        """Consumers need a fixed sequence length; batch may vary."""
        from scripts.export_to_onnx import resolve_task_config

        axes = resolve_task_config("text-classification")["dynamic_axes"]

        assert axes["input_ids"] == {0: "batch_size"}


class TestTextClassificationDummyInputs:
    def test_dummy_input_ids_are_int32(self):
        """torch.onnx.export records the dtype it traces; randint defaults to int64."""
        import torch

        from scripts.export_to_onnx import build_dummy_inputs

        inputs = build_dummy_inputs("text-classification", "cpu", max_length=256)

        assert inputs["input_ids"].dtype == torch.int32
        assert inputs["input_ids"].shape == (1, 256)

    def test_two_input_variant_adds_the_mask(self):
        from scripts.export_to_onnx import build_dummy_inputs

        inputs = build_dummy_inputs("text-classification", "cpu", max_length=64, two_input=True)

        assert set(inputs) == {"input_ids", "attention_mask"}


class TestTextClassificationExportEndToEnd:
    def test_exports_and_matches_pytorch(self, tmp_path):
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch

        from scripts.export_to_onnx import (
            SingleInputClassifier,
            build_dummy_inputs,
            export,
        )
        from transformers import BertConfig, BertForSequenceClassification

        config = BertConfig(
            vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=16, max_position_embeddings=64, num_labels=2,
        )
        model = SingleInputClassifier(BertForSequenceClassification(config).eval(), pad_id=0)

        dummy = build_dummy_inputs("text-classification", "cpu", max_length=32)
        out_path = tmp_path / "clf.onnx"
        export(model, dummy, out_path, opset=17, task="text-classification")

        onnx.checker.check_model(onnx.load(str(out_path)))

        session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        onnx_out = session.run(None, {"input_ids": dummy["input_ids"].numpy()})[0]

        with torch.no_grad():
            torch_out = model(dummy["input_ids"]).numpy()

        assert onnx_out.shape == (1, 2)
        assert np.allclose(onnx_out.sum(-1), 1.0, atol=1e-5)
        assert np.abs(onnx_out - torch_out).max() < 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_export.py -q`
Expected: FAIL with `ImportError: cannot import name 'SingleInputClassifier'`

- [ ] **Step 3: Write minimal implementation**

Add to `scripts/export_to_onnx.py`:

```python
DEFAULT_MAX_LENGTH = 256


class SingleInputClassifier(torch.nn.Module):
    """Derive the attention mask in-graph so the exported model takes one input.

    Consumers that call a runtime's single-tensor run() cannot supply a separate
    attention_mask, and normalise multi-class output by its sum rather than applying a
    softmax — so the softmax belongs in the graph too, otherwise raw logits read as
    probabilities without being any.
    """

    def __init__(self, model: torch.nn.Module, pad_id: int):
        super().__init__()
        self.model = model
        self.pad_id = pad_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        attention_mask = (input_ids != self.pad_id).long()
        logits = self.model(input_ids=input_ids.long(), attention_mask=attention_mask).logits
        return torch.softmax(logits, dim=-1)
```

Register the task in `TASK_CONFIGS`:

```python
    "text-classification": {
        "model_class": "AutoModelForSequenceClassification",
        "processor_class": "AutoTokenizer",
        "input_names": ["input_ids"],
        "output_names": ["probabilities"],
        "dynamic_axes": {"input_ids": {0: "batch_size"}, "probabilities": {0: "batch_size"}},
    },
```

Add the resolver:

```python
def resolve_task_config(task: str, two_input: bool = False) -> dict:
    """Return the export config for a task.

    text-classification has two shapes and a static dict cannot express both, so its config
    is computed. Every other task returns its TASK_CONFIGS entry unchanged.
    """
    if task != "text-classification" or not two_input:
        return TASK_CONFIGS[task]
    return {
        **TASK_CONFIGS[task],
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    }
```

Extend `build_dummy_inputs`:

```python
def build_dummy_inputs(
    task: str, device: str, max_length: int = DEFAULT_MAX_LENGTH, two_input: bool = False
) -> dict[str, torch.Tensor]:
    ...
    elif task == "text-classification":
        # int32 explicitly: torch.randint defaults to int64 and torch.onnx.export records
        # the dtype it traces, so the declared int32 input would otherwise be int64.
        inputs = {
            "input_ids": torch.randint(
                1, 100, (1, max_length), dtype=torch.int32, device=device
            )
        }
        if two_input:
            inputs["attention_mask"] = torch.ones(
                1, max_length, dtype=torch.int32, device=device
            )
        return inputs
```

Extend `load_model` with the `text-classification` branch loading
`AutoModelForSequenceClassification` and `AutoTokenizer`, wrapping in `SingleInputClassifier`
with `tokenizer.pad_token_id` unless `--two-input-export`. Change `export()` and
`validate_onnx()` to use `resolve_task_config(task, two_input)`. Add `--max-length` (default
`DEFAULT_MAX_LENGTH`) and `--two-input-export` arguments, and extend `save_model_config` to
record `num_labels`, `id2label` and `max_length`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_export.py -q`
Expected: PASS

- [ ] **Step 5: Verify the parity tests still hold**

Run: `.venv/bin/python -m pytest tests/test_benchmarks.py tests/test_pruning.py -q`
Expected: FAIL — `benchmark.py` and `prune.py` do not yet accept `text-classification`. This is the parity guard doing its job; Tasks 9 and 10 fix it. Add `text-classification` to `SUPPORTED_TASKS` in **both** `scripts/benchmark.py` and `scripts/prune.py` now, with a `build_dummy_inputs` branch in each, to keep the suite green.

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/export_to_onnx.py scripts/benchmark.py scripts/prune.py tests/test_export.py
git commit -m "feat: add text-classification ONNX export with single-input wrapper"
```

---

### Task 7: App artifact exporter and validator CLIs (SHRINK-017)

**Files:**
- Create: `scripts/export_app_artifacts.py`, `scripts/validate_artifacts.py`
- Modify: `pyproject.toml` (entry points)
- Test: `tests/test_export_app_artifacts.py`

**Interfaces:**
- Consumes: `compression.artifacts.write_bundle`, `validate_bundle`.
- Produces:
  - `load_tokenizer_assets(tokenizer_dir: Path) -> tuple[Path, dict]` returning the vocab path and tokenizer meta, raising `ValueError` for non-WordPiece tokenizers
  - `labels_from_config(model_config: dict) -> list[str]` ordered by index
  - `main()` for each script

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the app artifact exporter."""

from __future__ import annotations

import json

import pytest


def _tokenizer_dir(tmp_path, with_vocab=True):
    d = tmp_path / "tok"
    d.mkdir()
    if with_vocab:
        (d / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\nhi\n")
    (d / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 128, "unk_token": "[UNK]", "pad_token": "[PAD]",
                    "cls_token": "[CLS]", "sep_token": "[SEP]"})
    )
    return d


class TestLoadTokenizerAssets:
    def test_returns_vocab_path_and_meta(self, tmp_path):
        from scripts.export_app_artifacts import load_tokenizer_assets

        vocab, meta = load_tokenizer_assets(_tokenizer_dir(tmp_path))

        assert vocab.name == "vocab.txt"
        assert meta["unk_token"] == "[UNK]"
        assert meta["max_length"] == 128

    def test_non_wordpiece_tokenizer_is_rejected(self, tmp_path):
        """A BPE/SentencePiece tokenizer has no line-indexed vocab.txt."""
        from scripts.export_app_artifacts import load_tokenizer_assets

        with pytest.raises(ValueError, match="WordPiece"):
            load_tokenizer_assets(_tokenizer_dir(tmp_path, with_vocab=False))

    def test_max_length_override_wins(self, tmp_path):
        from scripts.export_app_artifacts import load_tokenizer_assets

        _, meta = load_tokenizer_assets(_tokenizer_dir(tmp_path), max_length=256)

        assert meta["max_length"] == 256


class TestLabelsFromConfig:
    def test_orders_by_index(self):
        from scripts.export_app_artifacts import labels_from_config

        labels = labels_from_config({"id2label": {"1": "positive", "0": "negative"}})

        assert labels == ["negative", "positive"]

    def test_missing_id2label_is_an_error(self):
        from scripts.export_app_artifacts import labels_from_config

        with pytest.raises(ValueError, match="id2label"):
            labels_from_config({})

    def test_non_contiguous_indices_are_an_error(self):
        from scripts.export_app_artifacts import labels_from_config

        with pytest.raises(ValueError, match="contiguous"):
            labels_from_config({"id2label": {"0": "a", "2": "b"}})


class TestValidatorCLI:
    def test_reports_every_violation_not_just_the_first(self, tmp_path, capsys):
        from compression.artifacts import write_bundle
        from scripts.validate_artifacts import run

        model = tmp_path / "m.tflite"
        model.write_bytes(b"x")
        vocab = tmp_path / "v.txt"
        vocab.write_text("[PAD]\na\n")
        out = tmp_path / "bundle"
        write_bundle(
            out, task="text-classification", model_path=model, runtime="tflite",
            input_signature=[{"name": "input_ids", "dtype": "int32", "shape": [1, 8]}],
            output_signature=[{"name": "probabilities", "dtype": "float32", "shape": [1, 2]}],
            vocab_path=vocab, tokenizer_meta={"max_length": 8}, labels=["a", "b"],
        )
        (out / "labels.txt").unlink()
        (out / "tokenizer_vocab.txt").write_text("tampered\n")

        exit_code = run(out)

        assert exit_code != 0
        captured = capsys.readouterr().out
        assert "labels.txt" in captured
        assert "tokenizer_vocab.txt" in captured

    def test_valid_bundle_exits_zero(self, tmp_path):
        from compression.artifacts import write_bundle
        from scripts.validate_artifacts import run

        model = tmp_path / "m.tflite"
        model.write_bytes(b"x")
        out = tmp_path / "bundle"
        write_bundle(
            out, task="causal-lm", model_path=model, runtime="tflite",
            input_signature=[{"name": "input_ids", "dtype": "int32", "shape": [1, 8]}],
            output_signature=[{"name": "logits", "dtype": "float32", "shape": [1, 8, 64]}],
        )

        assert run(out) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_export_app_artifacts.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

`scripts/export_app_artifacts.py` — argparse over `--model`, `--onnx-metadata`, `--model-file`,
`--runtime`, `--output`, `--max-length`, `--positive-label`, `--task`. It reads `id2label` from
the model directory's `config.json` or from the exporter's metadata JSON, loads the tokenizer
assets, and calls `write_bundle`.

`load_tokenizer_assets` raises when `vocab.txt` is absent:

```python
    raise ValueError(
        f"{tokenizer_dir} has no vocab.txt. Only WordPiece tokenizers can be exported as a "
        "line-indexed vocabulary; a BPE or SentencePiece tokenizer needs its own artifact "
        "format, which the contract does not yet define."
    )
```

`labels_from_config` sorts by integer index and rejects non-contiguous index sets.

`scripts/validate_artifacts.py`:

```python
def run(bundle_dir: Path) -> int:
    violations = validate_bundle(bundle_dir)
    if not violations:
        print(f"OK: {bundle_dir} conforms to contract version {CONTRACT_VERSION}")
        return 0
    print(f"INVALID: {bundle_dir} has {len(violations)} violation(s):")
    for violation in violations:
        print(f"  - {violation}")
    return 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_export_app_artifacts.py -q`
Expected: PASS, 9 tests

- [ ] **Step 5: Add entry points to `pyproject.toml`**

```toml
shrink-app-artifacts = "scripts.export_app_artifacts:main"
shrink-validate-artifacts = "scripts.validate_artifacts:main"
shrink-convert-tflite = "scripts.convert_to_tflite:main"
shrink-convert-coreml = "scripts.convert_to_coreml:main"
shrink-convert-onnx-mobile = "scripts.convert_to_onnx_mobile:main"
```

- [ ] **Step 6: Lint, format, full suite, commit**

```bash
git add scripts/export_app_artifacts.py scripts/validate_artifacts.py pyproject.toml \
        tests/test_export_app_artifacts.py
git commit -m "feat: add app artifact exporter and bundle validator"
```

---

### Task 8: Evaluation loop

**Files:**
- Create: `compression/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `compression.metrics.classification_metrics`.
- Produces:
  - `predict_probabilities(inference_fn, texts, tokenizer, max_length, input_names, batch_size=16) -> np.ndarray` of shape `(n, num_labels)`
  - `evaluate_classification(inference_fn, records, tokenizer, label_map, *, max_length, input_names, positive_label=None, batch_size=16) -> dict` returning the metrics dict plus `num_samples`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the runtime-agnostic evaluation loop."""

from __future__ import annotations

import numpy as np
import pytest


class _EchoTokenizer:
    def __call__(self, texts, truncation=True, padding="max_length", max_length=8,
                 return_tensors="np"):
        ids = []
        for text in texts:
            row = [ord(c) for c in text][:max_length]
            ids.append(row + [0] * (max_length - len(row)))
        arr = np.array(ids, dtype=np.int32)
        return {"input_ids": arr, "attention_mask": (arr != 0).astype(np.int32)}


def _oracle_fn(inputs):
    """Predict class 1 when the first token is odd, else class 0."""
    first = inputs["input_ids"][:, 0]
    probs = np.zeros((len(first), 2), dtype=np.float32)
    probs[first % 2 == 1, 1] = 1.0
    probs[first % 2 == 0, 0] = 1.0
    return [probs]


class TestPredictProbabilities:
    def test_returns_one_row_per_input(self):
        from compression.evaluation import predict_probabilities

        probs = predict_probabilities(
            _oracle_fn, ["a", "b", "c"], _EchoTokenizer(), max_length=8,
            input_names=["input_ids"], batch_size=2,
        )

        assert probs.shape == (3, 2)

    def test_batches_do_not_change_results(self):
        from compression.evaluation import predict_probabilities

        texts = ["a", "b", "c", "d", "e"]
        one = predict_probabilities(_oracle_fn, texts, _EchoTokenizer(), max_length=8,
                                    input_names=["input_ids"], batch_size=1)
        many = predict_probabilities(_oracle_fn, texts, _EchoTokenizer(), max_length=8,
                                     input_names=["input_ids"], batch_size=4)

        assert np.array_equal(one, many)

    def test_only_declared_inputs_are_passed(self):
        """A single-input model must not receive attention_mask."""
        from compression.evaluation import predict_probabilities

        seen = {}

        def spy(inputs):
            seen["keys"] = set(inputs)
            return _oracle_fn(inputs)

        predict_probabilities(spy, ["a"], _EchoTokenizer(), max_length=8,
                              input_names=["input_ids"], batch_size=1)

        assert seen["keys"] == {"input_ids"}


class TestEvaluateClassification:
    def test_computes_metrics_against_labels(self):
        from compression.evaluation import evaluate_classification

        # 'a'=97 odd -> class 1. 'b'=98 even -> class 0.
        records = [{"text": "a", "label": "positive"}, {"text": "b", "label": "negative"}]

        result = evaluate_classification(
            _oracle_fn, records, _EchoTokenizer(),
            {"negative": 0, "positive": 1},
            max_length=8, input_names=["input_ids"], positive_label="positive",
        )

        assert result["accuracy"] == pytest.approx(1.0)
        assert result["precision_binary"] == pytest.approx(1.0)
        assert result["num_samples"] == 2

    def test_unknown_positive_label_is_an_error(self):
        from compression.evaluation import evaluate_classification

        with pytest.raises(ValueError, match="positive_label"):
            evaluate_classification(
                _oracle_fn, [{"text": "a", "label": "positive"}], _EchoTokenizer(),
                {"negative": 0, "positive": 1},
                max_length=8, input_names=["input_ids"], positive_label="missing",
            )

    def test_empty_dataset_is_an_error(self):
        from compression.evaluation import evaluate_classification

        with pytest.raises(ValueError, match="empty"):
            evaluate_classification(
                _oracle_fn, [], _EchoTokenizer(), {"negative": 0, "positive": 1},
                max_length=8, input_names=["input_ids"],
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_evaluation.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
"""
evaluation.py — Run a compressed model over a labelled dataset and score it.

Runtime-agnostic: it takes the same `inference_fn` callable benchmark.py already builds for
ONNX Runtime and TFLite, so accuracy is measured on the artifact that will actually ship
rather than on the PyTorch model it came from.
"""

from __future__ import annotations

import logging

import numpy as np

from compression.data.text_classification import encode_labels
from compression.metrics import classification_metrics

log = logging.getLogger(__name__)


def predict_probabilities(
    inference_fn,
    texts: list[str],
    tokenizer,
    max_length: int,
    input_names: list[str],
    batch_size: int = 16,
) -> np.ndarray:
    """Tokenize, run in batches, and stack the first output of each batch."""
    outputs = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            chunk,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="np",
        )
        inputs = {name: np.asarray(encoded[name]) for name in input_names if name in encoded}
        outputs.append(np.asarray(inference_fn(inputs)[0]))
        if start and start % (batch_size * 20) == 0:
            log.info("Evaluated %d/%d samples", start, len(texts))
    return np.concatenate(outputs, axis=0)


def evaluate_classification(
    inference_fn,
    records: list[dict],
    tokenizer,
    label_map: dict[str, int],
    *,
    max_length: int,
    input_names: list[str],
    positive_label: str | None = None,
    batch_size: int = 16,
) -> dict:
    """Score a classifier over labelled records."""
    if not records:
        raise ValueError("Cannot evaluate an empty dataset")

    positive_index = None
    if positive_label is not None:
        if positive_label not in label_map:
            raise ValueError(
                f"positive_label {positive_label!r} is not a known label; "
                f"the model declares {sorted(label_map)}"
            )
        positive_index = label_map[positive_label]

    texts = [record["text"] for record in records]
    y_true = encode_labels(records, label_map)

    probabilities = predict_probabilities(
        inference_fn, texts, tokenizer, max_length, input_names, batch_size
    )
    y_pred = probabilities.argmax(axis=-1).tolist()

    result = classification_metrics(
        y_true, y_pred, num_labels=len(label_map), positive_index=positive_index
    )
    result["num_samples"] = len(records)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_evaluation.py -q`
Expected: PASS, 6 tests

- [ ] **Step 5: Lint, format, full suite, commit**

```bash
git add compression/evaluation.py tests/test_evaluation.py
git commit -m "feat: add runtime-agnostic classification evaluation"
```

---

### Task 9: Benchmark task, evaluation and accuracy gates (SHRINK-019, part of SHRINK-018)

**Files:**
- Modify: `scripts/benchmark.py`
- Test: `tests/test_benchmarks.py`

**Interfaces:**
- Consumes: `compression.evaluation.evaluate_classification`, `compression.data.load_records`, `compression.data.text_classification.build_label_map`.
- Produces: `--tokenizer`, `--min-precision`, `--min-recall`, `--min-f1`, `--positive-label`, `--max-length` CLI arguments; `evaluate_gates` handles the new gates.

- [ ] **Step 1: Write the failing test** (append to `tests/test_benchmarks.py`)

```python
class TestTFLiteRunnerDtypes:
    def test_int_inputs_are_not_cast_to_float(self):
        """Token ids must reach the interpreter as ints; casting them to float32 is silent
        corruption of a text classifier's only input."""
        import inspect

        import scripts.benchmark as benchmark

        source = inspect.getsource(benchmark.TFLiteRunner.__call__)
        assert "astype(np.float32)" not in source, (
            "TFLiteRunner must cast each input to that tensor's declared dtype, "
            "not unconditionally to float32."
        )
        assert 'detail["dtype"]' in source


class TestTextClassificationBenchmarkTask:
    def test_task_is_supported(self):
        from scripts.benchmark import SUPPORTED_TASKS

        assert "text-classification" in SUPPORTED_TASKS

    def test_dummy_inputs_are_int32(self):
        import numpy as np

        from scripts.benchmark import build_dummy_inputs

        inputs = build_dummy_inputs("text-classification")

        assert set(inputs) == {"input_ids"}
        assert inputs["input_ids"].dtype == np.int32


class TestAccuracyGates:
    def _args(self, **overrides):
        import argparse

        defaults = dict(
            max_size_mb=None, max_latency_ms_p95=None, min_accuracy=None,
            min_precision=None, min_recall=None, min_f1=None, dataset=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def _result(self, accuracy):
        from scripts.benchmark import BenchmarkResult

        return BenchmarkResult(
            run_id="r", timestamp="t", model_name="m", model_path="p", model_size_mb=1.0,
            task="text-classification", runtime="onnxruntime", accuracy=accuracy,
        )

    def test_precision_gate_passes_above_threshold(self):
        from scripts.benchmark import evaluate_gates

        result = self._result({"precision_binary": 0.98})
        evaluate_gates(result, self._args(min_precision=0.97))

        assert result.gate_results["min_precision"] is True
        assert result.passed is True

    def test_precision_gate_fails_below_threshold(self):
        from scripts.benchmark import evaluate_gates

        result = self._result({"precision_binary": 0.90})
        evaluate_gates(result, self._args(min_precision=0.97))

        assert result.gate_results["min_precision"] is False
        assert result.passed is False

    def test_gate_fails_when_the_metric_was_not_computed(self):
        """A gate whose data is unavailable must fail, never silently pass."""
        from scripts.benchmark import evaluate_gates

        result = self._result({})
        evaluate_gates(result, self._args(min_precision=0.97))

        assert result.gate_results["min_precision"] is False

    def test_f1_gate_uses_binary_f1_when_available(self):
        from scripts.benchmark import evaluate_gates

        result = self._result({"f1_binary": 0.8, "f1_macro": 0.2})
        evaluate_gates(result, self._args(min_f1=0.7))

        assert result.gate_results["min_f1"] is True

    def test_f1_gate_falls_back_to_macro(self):
        from scripts.benchmark import evaluate_gates

        result = self._result({"f1_macro": 0.75})
        evaluate_gates(result, self._args(min_f1=0.7))

        assert result.gate_results["min_f1"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_benchmarks.py -q`
Expected: FAIL — `astype(np.float32)` present; `min_precision` not handled

- [ ] **Step 3: Write minimal implementation**

Fix `TFLiteRunner.__call__`:

```python
    def __call__(self, inputs: dict) -> list:
        for detail in self.input_details:
            key = detail["name"].split(":")[0]
            if key in inputs:
                # Cast to the tensor's declared dtype. Casting everything to float32 corrupts
                # integer inputs — a text classifier's only input is int32 token ids.
                self.interpreter.set_tensor(
                    detail["index"], inputs[key].astype(detail["dtype"])
                )
        self.interpreter.invoke()
        return [self.interpreter.get_tensor(d["index"]) for d in self.output_details]
```

Add `"text-classification"` to `SUPPORTED_TASKS` and a `build_dummy_inputs` branch returning
`{"input_ids": np.random.randint(1, 1000, (1, 256)).astype(np.int32)}`.

Extend `evaluate_gates` with a shared helper:

```python
def _accuracy_gate(result, gates, name, threshold, keys):
    """Fail a configured gate whose metric was not computed, rather than skipping it."""
    if threshold is None:
        return
    for key in keys:
        if key in result.accuracy:
            gates[name] = float(result.accuracy[key]) >= threshold
            return
    log.warning("%s specified but %s was not computed; gate fails", name, " or ".join(keys))
    gates[name] = False
```

called as:

```python
    _accuracy_gate(result, gates, "min_precision", args.min_precision,
                   ("precision_binary", "precision_macro"))
    _accuracy_gate(result, gates, "min_recall", args.min_recall,
                   ("recall_binary", "recall_macro"))
    _accuracy_gate(result, gates, "min_f1", args.min_f1, ("f1_binary", "f1_macro"))
```

Add CLI arguments `--tokenizer`, `--positive-label`, `--max-length` (default 256),
`--min-precision`, `--min-recall`, `--min-f1`. In `main`, when `--dataset` and `--tokenizer`
are both given and the task is `text-classification`, load records with
`load_records(args.dataset, split="eval")`, build the label map from the tokenizer directory's
`config.json` (or `--labels`), and set `result.accuracy = evaluate_classification(...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_benchmarks.py -q`
Expected: PASS

- [ ] **Step 5: Full suite, lint, commit**

```bash
git add scripts/benchmark.py tests/test_benchmarks.py
git commit -m "feat: add text-classification benchmarking with accuracy gates"
```

---

### Task 10: Sequence-classification distillation with real training (SHRINK-016)

**Files:**
- Modify: `scripts/distill.py`
- Test: `tests/test_distillation.py`

**Interfaces:**
- Consumes: `compression.data` (`TextClassificationDataset`, `DualTokenizerCollator`, `CausalLMBlockDataset`, `causal_lm_collator`, `assert_shared_vocabulary`, `load_records`, `build_label_map`).
- Produces:
  - `SUPPORTED_TASKS = ("legal", "text-classification")`
  - `load_models(task, teacher_id, student_id, device) -> tuple[teacher, student, teacher_tok, student_tok]`
  - `build_datasets(task, args, teacher_tok, student_tok, label_map) -> tuple[train_ds, eval_ds, collator]`
  - `DistillationTrainer.compute_loss` handling `teacher_`-prefixed inputs
  - `pool_hidden(hidden, attention_mask) -> Tensor` — mean over non-pad positions

- [ ] **Step 1: Write the failing test** (append to `tests/test_distillation.py`)

```python
class TestSupportedTasksIncludesTextClassification:
    def test_text_classification_is_supported(self):
        from scripts.distill import SUPPORTED_TASKS

        assert "text-classification" in SUPPORTED_TASKS
        assert "legal" in SUPPORTED_TASKS


class TestPoolHidden:
    def test_averages_only_unpadded_positions(self):
        from scripts.distill import pool_hidden

        hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
        mask = torch.tensor([[1, 1, 0]])

        pooled = pool_hidden(hidden, mask)

        assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))

    def test_shape_is_batch_by_hidden(self):
        from scripts.distill import pool_hidden

        pooled = pool_hidden(torch.randn(4, 7, 16), torch.ones(4, 7, dtype=torch.long))

        assert pooled.shape == (4, 16)

    def test_all_padding_does_not_divide_by_zero(self):
        from scripts.distill import pool_hidden

        pooled = pool_hidden(torch.randn(1, 3, 4), torch.zeros(1, 3, dtype=torch.long))

        assert torch.isfinite(pooled).all()


class TestComputeLossTeacherInputs:
    def _trainer(self, align_hidden=False):
        from scripts.distill import DistillationConfig, DistillationTrainer

        class FakeModel(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.bias = bias
                self.linear = torch.nn.Linear(4, 2)

            def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False):
                batch = input_ids.shape[0]
                logits = self.linear(input_ids.float()) + self.bias
                out = type("Out", (), {})()
                out.logits = logits
                if output_hidden_states:
                    out.hidden_states = [torch.randn(batch, input_ids.shape[1], 4)]
                return out

        trainer = DistillationTrainer.__new__(DistillationTrainer)
        trainer.teacher_model = FakeModel(1.0)
        trainer.distill_config = DistillationConfig(alpha=1.0, beta=0.0,
                                                    align_hidden=align_hidden)
        from scripts.distill import DistillationLoss

        trainer.distill_loss = DistillationLoss(trainer.distill_config)
        trainer.projector = None
        trainer.state = type("S", (), {"global_step": 1})()
        return trainer, FakeModel(0.0)

    def test_teacher_prefixed_inputs_are_routed_to_the_teacher(self):
        """Classification allows different tokenizers, so the teacher gets its own tensors."""
        trainer, student = self._trainer()

        seen = {}
        original_forward = trainer.teacher_model.forward

        def spy(input_ids=None, attention_mask=None, output_hidden_states=False):
            seen["input_ids"] = input_ids.clone()
            return original_forward(input_ids, attention_mask, output_hidden_states)

        trainer.teacher_model.forward = spy

        teacher_ids = torch.ones(2, 4) * 7
        inputs = {
            "input_ids": torch.ones(2, 4),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "labels": torch.tensor([0, 1]),
            "teacher_input_ids": teacher_ids,
            "teacher_attention_mask": torch.ones(2, 4, dtype=torch.long),
        }

        trainer.compute_loss(student, inputs)

        assert torch.equal(seen["input_ids"], teacher_ids)

    def test_absent_teacher_keys_reuse_the_student_inputs(self):
        """Shared-vocabulary case; keeps the causal-LM path's behaviour unchanged."""
        trainer, student = self._trainer()

        seen = {}
        original_forward = trainer.teacher_model.forward

        def spy(input_ids=None, attention_mask=None, output_hidden_states=False):
            seen["input_ids"] = input_ids.clone()
            return original_forward(input_ids, attention_mask, output_hidden_states)

        trainer.teacher_model.forward = spy

        student_ids = torch.ones(2, 4) * 3
        trainer.compute_loss(
            student,
            {
                "input_ids": student_ids,
                "attention_mask": torch.ones(2, 4, dtype=torch.long),
                "labels": torch.tensor([0, 1]),
            },
        )

        assert torch.equal(seen["input_ids"], student_ids)

    def test_returns_a_scalar_loss(self):
        trainer, student = self._trainer()

        loss = trainer.compute_loss(
            student,
            {
                "input_ids": torch.ones(2, 4),
                "attention_mask": torch.ones(2, 4, dtype=torch.long),
                "labels": torch.tensor([0, 1]),
            },
        )

        assert loss.ndim == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_distillation.py -q`
Expected: FAIL — `SUPPORTED_TASKS` lacks the task, `pool_hidden` undefined

- [ ] **Step 3: Write minimal implementation**

Replace the `TestSupportedTasks::test_distillation_supports_only_legal_task` assertion in the
existing test file — it is the change-detector the audit flagged, and this task is when it
legitimately changes.

Add `pool_hidden`:

```python
def pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool hidden states over non-padding positions.

    Classification aligns pooled representations rather than per-token ones: teacher and
    student may tokenize differently, so their sequence lengths need not match and a
    per-position MSE would be comparing unrelated positions.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts
```

Rewrite `compute_loss` to split teacher inputs:

```python
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = dict(inputs)
        labels = inputs.pop("labels", None)

        teacher_inputs = {
            key[len("teacher_") :]: value
            for key, value in list(inputs.items())
            if key.startswith("teacher_")
        }
        for key in list(inputs):
            if key.startswith("teacher_"):
                inputs.pop(key)
        # No teacher_ keys means a shared vocabulary, so the teacher reads the student's
        # tensors — which is the causal-LM path, unchanged.
        if not teacher_inputs:
            teacher_inputs = inputs

        student_outputs = model(**inputs, output_hidden_states=self.distill_config.align_hidden)
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                **teacher_inputs, output_hidden_states=self.distill_config.align_hidden
            )
            teacher_logits = teacher_outputs.logits

        student_hidden = teacher_hidden = None
        if self.distill_config.align_hidden:
            student_hidden = student_outputs.hidden_states[-1]
            teacher_hidden = teacher_outputs.hidden_states[-1]
            if self.distill_config.pool_hidden_states:
                student_hidden = pool_hidden(student_hidden, inputs["attention_mask"])
                teacher_hidden = pool_hidden(teacher_hidden, teacher_inputs["attention_mask"])
            if self.projector is not None:
                student_hidden = self.projector(student_hidden)

        losses = self.distill_loss(
            student_logits, teacher_logits, labels, student_hidden, teacher_hidden
        )
        loss = losses["loss_total"]

        if self.state.global_step % 50 == 0:
            log_msg = " | ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
            log.info("Step %d | %s", self.state.global_step, log_msg)

        return (loss, student_outputs) if return_outputs else loss
```

Add `pool_hidden_states: bool = False` to `DistillationConfig`, set `True` for
text-classification and `False` for legal.

Add `load_models` dispatching on task, `build_datasets` using the Task 2–4 modules, and replace
the commented-out trainer block with live construction and `trainer.train()`, followed by
`student.save_pretrained` and `tokenizer.save_pretrained`. For `legal`, call
`assert_shared_vocabulary(teacher_tok, student_tok)` before training.

New CLI arguments: `--eval-dataset`, `--text-column`, `--label-column`, `--max-length`,
`--block-size`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_distillation.py -q`
Expected: PASS

- [ ] **Step 5: Full suite, lint, commit**

```bash
git add scripts/distill.py tests/test_distillation.py
git commit -m "feat: support sequence-classification distillation and enable training"
```

---

### Task 11: Pipeline dispatch, config-relative paths, and gate wiring

**Files:**
- Modify: `scripts/run_pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `stage_command(script: str) -> list[str]` returning an absolute, CWD-independent command
  - `resolve_config_path(value: str, config_path: Path) -> str`
  - `TASK_CAPABLE_STAGES: dict[str, set[str]]` mapping a stage to the tasks it supports
  - `main` exits non-zero when a configured stage cannot run the configured task

- [ ] **Step 1: Write the failing test** (append to `tests/test_pipeline.py`)

```python
class TestStageCommand:
    def test_uses_an_absolute_path_to_the_installed_script(self):
        """run_pipeline must work from any CWD, not only the shrink-llm checkout."""
        from pathlib import Path

        from scripts.run_pipeline import stage_command

        command = stage_command("export_to_onnx")
        script = command[1]

        assert Path(script).is_absolute(), command
        assert Path(script).exists(), script

    def test_command_starts_with_the_running_interpreter(self):
        import sys

        from scripts.run_pipeline import stage_command

        assert stage_command("quantize")[0] == sys.executable

    def test_unknown_script_is_an_error(self):
        import pytest

        from scripts.run_pipeline import stage_command

        with pytest.raises(FileNotFoundError):
            stage_command("no_such_stage")


class TestResolveConfigPath:
    def test_relative_paths_resolve_against_the_config_file(self, tmp_path):
        """A consuming project keeps its config and data together in its own repo."""
        from scripts.run_pipeline import resolve_config_path

        config_path = tmp_path / "nested" / "pipeline.yaml"
        config_path.parent.mkdir()
        config_path.write_text("")
        (config_path.parent / "data").mkdir()

        resolved = resolve_config_path("data", config_path)

        assert resolved == str(config_path.parent / "data")

    def test_absolute_paths_are_returned_unchanged(self, tmp_path):
        from scripts.run_pipeline import resolve_config_path

        assert resolve_config_path(str(tmp_path), tmp_path / "c.yaml") == str(tmp_path)

    def test_hub_ids_are_left_alone(self, tmp_path):
        """A hub dataset id is not a path and must not be rewritten."""
        from scripts.run_pipeline import resolve_config_path

        assert resolve_config_path("org/dataset", tmp_path / "c.yaml") == "org/dataset"


class TestTaskCapableStages:
    def test_distill_supports_text_classification(self):
        from scripts.run_pipeline import TASK_CAPABLE_STAGES

        assert "text-classification" in TASK_CAPABLE_STAGES["distill"]

    def test_unsupported_task_for_a_stage_is_detected(self):
        from scripts.run_pipeline import unsupported_stages

        assert unsupported_stages(["distill"], "audio") == ["distill"]

    def test_supported_combination_is_empty(self):
        from scripts.run_pipeline import unsupported_stages

        assert unsupported_stages(["export", "quantize"], "text-classification") == []


class TestAccuracyGateWiring:
    def test_min_precision_reaches_benchmark(self, tmp_path):
        from scripts.run_pipeline import build_stage_args, init_pipeline_state

        config = _base_config()
        config["task"] = "text-classification"
        config["success_criteria"] = {"min_precision": 0.97}
        config["positive_label"] = "positive"
        state = init_pipeline_state(config, tmp_path)

        args = build_stage_args("benchmark", config, tmp_path, state)

        assert "--min-precision" in args
        assert args[args.index("--min-precision") + 1] == "0.97"
        assert "--positive-label" in args

    def test_min_f1_is_no_longer_reported_as_unenforced(self, tmp_path, caplog):
        import logging

        from scripts.run_pipeline import build_stage_args, init_pipeline_state

        config = _base_config()
        config["success_criteria"] = {"min_f1": 0.8}
        state = init_pipeline_state(config, tmp_path)

        with caplog.at_level(logging.WARNING, logger="scripts.run_pipeline"):
            args = build_stage_args("benchmark", config, tmp_path, state)

        assert "--min-f1" in args
        assert not any("min_f1" in r.message and "NOT be enforced" in r.message
                       for r in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: FAIL — `stage_command` undefined

- [ ] **Step 3: Write minimal implementation**

```python
_SCRIPTS_DIR = Path(__file__).resolve().parent


def stage_command(script: str) -> list[str]:
    """Absolute command for a stage script.

    Resolved against this file's directory rather than a relative "scripts/" path, which only
    worked when the process CWD was the ShrinkLLM checkout and therefore prevented any other
    project from driving the pipeline.
    """
    path = _SCRIPTS_DIR / f"{script}.py"
    if not path.exists():
        raise FileNotFoundError(f"No stage script at {path}")
    return [sys.executable, str(path)]


def resolve_config_path(value: str, config_path: Path) -> str:
    """Resolve a relative filesystem path against the config file's directory.

    Values that do not name an existing path — hub dataset ids, model ids — are returned
    unchanged.
    """
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    relative = (Path(config_path).resolve().parent / candidate)
    if relative.exists():
        return str(relative)
    return value


TASK_CAPABLE_STAGES: dict[str, set[str]] = {
    "prune": {"ocr", "legal", "audio", "classification", "text-classification"},
    "distill": {"legal", "text-classification"},
    "export": {"ocr", "legal", "audio", "classification", "text-classification"},
    "quantize": {"ocr", "legal", "audio", "classification", "text-classification"},
    "convert_tflite": {"ocr", "legal", "audio", "classification", "text-classification"},
    "convert_coreml": {"ocr", "legal", "audio", "classification", "text-classification"},
    "convert_onnx_mobile": {"ocr", "legal", "audio", "classification", "text-classification"},
    "benchmark": {"ocr", "legal", "audio", "classification", "text-classification"},
}


def unsupported_stages(stages: list[str], task: str) -> list[str]:
    return [s for s in stages if task not in TASK_CAPABLE_STAGES.get(s, set())]
```

In `run_stage`, replace `cmd = [sys.executable, f"scripts/{script}.py"] + args_list` with
`cmd = stage_command(script) + args_list`.

In `main`, after resolving stages, exit non-zero on `unsupported_stages`:

```python
    blocked = unsupported_stages(stages, str(config.get("task", "")))
    if blocked:
        parser.error(
            f"Stage(s) {', '.join(blocked)} cannot run task "
            f"{config.get('task')!r}. Remove them from --stages or the config's stages list."
        )
```

Remove the `if task != "legal": ... return []` branch from the distill stage builder.

In the benchmark stage builder, move `min_f1` from `_known_unimplemented` to
`_implemented_criteria`, add `min_precision` and `min_recall`, and emit the flags plus
`--positive-label`, `--tokenizer` and `--dataset` when configured.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: PASS

- [ ] **Step 5: Verify the pipeline runs from a different working directory**

```bash
cd /tmp && /home/nikos/Projects/shrink-llm/.venv/bin/python \
  /home/nikos/Projects/shrink-llm/scripts/run_pipeline.py \
  --config /home/nikos/Projects/shrink-llm/configs/ocr_pipeline.yaml --dry-run
```
Expected: completes, with absolute stage script paths in the logged commands

- [ ] **Step 6: Full suite, lint, commit**

```bash
git add scripts/run_pipeline.py tests/test_pipeline.py
git commit -m "feat: make pipeline dispatch location-independent and wire accuracy gates"
```

---

### Task 12: Pipeline config template (SHRINK-018)

**Files:**
- Create: `configs/text_classification_pipeline.yaml`
- Test: `tests/test_pipeline.py` (existing shipped-config tests cover it automatically)

**Interfaces:**
- Consumes: `stages:` support and the accuracy gates from Task 11.
- Produces: a config the existing `test_shipped_configs_resolve_to_a_valid_stage_selection` and `test_shipped_configs_build_args_for_every_selected_stage` tests exercise.

- [ ] **Step 1: Write the config**

```yaml
# Text Classification Pipeline — ShrinkLLM
#
# A template. Every value below is meant to be changed by the consuming project: the label
# set, the thresholds, and which class counts as expensive to get wrong.
#
# The teacher MUST be a sequence classifier fine-tuned on your labelled data. There is no
# off-the-shelf model that classifies an arbitrary task, so this is a placeholder — replace it
# with your own fine-tuned checkpoint (a local directory works as well as a hub id).

task: text-classification
model: google/mobilebert-uncased
teacher: ./models/teacher-classifier   # replace: a fine-tuned sequence classifier

onnx_opset: 17
max_length: 256

# convert_coreml is omitted: coremltools >= 6 does not accept ONNX input (SHRINK-020).
stages:
  - distill
  - export
  - quantize
  - convert_tflite
  - benchmark

quantization:
  precision: int8
  mode: dynamic
  skip_ops:
    - Softmax
    - LayerNormalization

distillation:
  dataset: datasets/text_classification/train.jsonl
  eval_dataset: datasets/text_classification/eval.jsonl
  temperature: 4.0
  alpha: 0.3
  beta: 0.7
  gamma: 0.1
  align_hidden: true
  epochs: 5
  batch_size: 16
  lr: 5e-5

mobile:
  android:
    runtime: tflite
    quantization: fp16

benchmark:
  runtime: tflite
  warmup_runs: 10
  benchmark_runs: 100
  dataset: datasets/text_classification/eval.jsonl

# The class whose false positives are expensive. min_precision is measured on THIS class;
# macro averaging would dilute it with the easy class and report a comfortable number.
positive_label: positive

success_criteria:
  min_precision: 0.97
  max_size_mb: 30
  max_latency_ms: 120
```

- [ ] **Step 2: Run the shipped-config tests**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -k shipped -q`
Expected: PASS

- [ ] **Step 3: Dry-run the config**

Run: `.venv/bin/python scripts/run_pipeline.py --config configs/text_classification_pipeline.yaml --dry-run`
Expected: every stage builds a command; no errors

- [ ] **Step 4: Update `SHRINK-018` in `docs/todos.yaml`**

Retitle to "Add text-classification pipeline config with a precision gate" and replace the
application-specific wording with the generic template description. Mark `SHRINK-015` through
`SHRINK-019` as done by adding `status: done` to each.

- [ ] **Step 5: Full suite, lint, commit**

```bash
git add configs/text_classification_pipeline.yaml docs/todos.yaml
git commit -m "feat: add text-classification pipeline config with a precision gate"
```

---

### Task 13: Consumer documentation

**Files:**
- Create: `docs/app_integration.md`
- Modify: `README.md`, `CHANGELOG.md`, `docs/text_classification.md`, `docs/roadmap.md`
- Test: `tests/test_packaging.py`

**Interfaces:**
- Consumes: everything above.
- Produces: documentation only, plus one test asserting every declared entry point resolves.

- [ ] **Step 1: Write the failing test** (append to `tests/test_packaging.py`)

```python
class TestConsoleEntryPoints:
    def test_every_entry_point_target_is_importable(self):
        """A declared entry point that cannot be imported fails only at install time."""
        import importlib

        scripts = _pyproject()["project"]["scripts"]

        for name, target in scripts.items():
            module_name, _, function_name = target.partition(":")
            module = importlib.import_module(module_name)
            assert hasattr(module, function_name), f"{name}: {target} has no {function_name}"

    def test_converters_and_artifact_tools_have_entry_points(self):
        scripts = _pyproject()["project"]["scripts"]

        for expected in (
            "shrink-app-artifacts",
            "shrink-validate-artifacts",
            "shrink-convert-tflite",
        ):
            assert expected in scripts, expected


class TestConsumerDocumentation:
    def test_app_integration_doc_exists_and_states_the_contract_version(self):
        from pathlib import Path

        from compression.artifacts import CONTRACT_VERSION

        text = Path("docs/app_integration.md").read_text()

        assert f"contract_version" in text
        assert str(CONTRACT_VERSION) in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_packaging.py -q`
Expected: FAIL — `docs/app_integration.md` does not exist

- [ ] **Step 3: Write `docs/app_integration.md`**

Sections, each with real content and no placeholders:

1. **What a bundle is** — the directory layout and the one-sentence purpose of each file.
2. **The manifest, field by field** — the full annotated example from the spec, with the
   required core keys called out and the reason `labels` sits under `task_config`.
3. **Loading a bundle** — a worked walkthrough: read `manifest.json`, refuse an unknown
   `contract_version`, verify digests, load the vocab into a line-indexed array, tokenize,
   run the model, read the probabilities, map `argmax` through `labels.txt`.
4. **Validating a bundle** — `shrink-validate-artifacts <dir>` in the consumer's CI, and the
   Python equivalent `validate_bundle(path) -> list[str]`.
5. **What a `contract_version` bump obliges you to do** — the current version is 1; a consumer
   must refuse versions it does not recognise rather than best-effort parsing.
6. **Producing a bundle** — the CLI sequence from export through `shrink-app-artifacts`.

- [ ] **Step 4: Update the other documents**

- `README.md` — move Text Classification from *(planned)* to supported in the task table; add
  a short "Using ShrinkLLM from another project" section pointing at `docs/app_integration.md`.
- `docs/text_classification.md` — change Status from `proposed` to `implemented` and replace
  the "what is missing" framing with pointers to the shipped modules.
- `docs/roadmap.md` — tick the Phase 7 checkboxes; leave `SHRINK-020` open.
- `CHANGELOG.md` — under Unreleased, an `### Added` entry per issue.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_packaging.py -q`
Expected: PASS

- [ ] **Step 6: Full verification**

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check scripts/ compression/ benchmarks/ mobile_deployment/ tests/
.venv/bin/black --check scripts/ compression/ benchmarks/ mobile_deployment/ tests/
for c in configs/*.yaml; do .venv/bin/python scripts/run_pipeline.py --config "$c" --dry-run >/dev/null || echo "FAILED $c"; done
```
Expected: all pass, no config failures

- [ ] **Step 7: Commit**

```bash
git add docs/ README.md CHANGELOG.md tests/test_packaging.py
git commit -m "docs: document the app artifact contract and Phase 7 completion"
```

---

## Self-review

**Spec coverage.** Every spec section maps to a task: §4 contract → Task 5; §5.1 data → Tasks 2–4;
§5.2 dual-tokenizer design → Tasks 3, 4, 10; §5.3 metrics → Task 1; §5.4 evaluation → Task 8;
§5.5 export → Task 6; §5.6/§5.7 artifacts CLIs → Task 7; §5.8 distillation → Task 10; §5.9
benchmark → Task 9; §5.10 pipeline → Task 11; §5.11 config → Task 12; §5.12 packaging → Tasks 7
and 13; §5.13 docs → Task 13. §6's two behaviour changes are Task 11. §7's testing strategy is
distributed across every task's Step 1.

**Placeholder scan.** One deliberate exception: Task 1 Step 3 contains a clumsy expression with an
explicit instruction to replace it, because the tests define the contract and the clean form is
obvious. No "TBD", no "add error handling", no "similar to Task N".

**Type consistency.** `SUPPORTED_TASKS` is a tuple in `benchmark.py`, `prune.py` and `distill.py`.
`validate_bundle` returns `list[str]` in Tasks 5, 7 and 13. `classification_metrics` keys
(`precision_binary`, `precision_macro`, …) are consistent across Tasks 1, 8 and 9.
`build_dummy_inputs` has different signatures in `export_to_onnx.py` (with `max_length`,
`two_input`) and `benchmark.py` (task only) — that difference is pre-existing and intentional, and
both are exercised by their own tests.

**Known ordering note.** Task 6 makes `tests/test_benchmarks.py` and `tests/test_pruning.py` fail
via the task-parity guards added in the audit pass. Task 6 Step 5 resolves this within the task
rather than leaving the suite red between commits.
