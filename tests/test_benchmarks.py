"""Tests for benchmarking pipeline."""

from __future__ import annotations

import json

import numpy as np
import pytest


class TestBuildDummyInputs:
    def test_ocr(self):
        from scripts.benchmark import build_dummy_inputs

        inputs = build_dummy_inputs("ocr")
        assert "pixel_values" in inputs
        assert inputs["pixel_values"].shape == (1, 3, 384, 384)

    def test_legal(self):
        from scripts.benchmark import build_dummy_inputs

        inputs = build_dummy_inputs("legal")
        assert "input_ids" in inputs
        assert "attention_mask" in inputs

    def test_audio(self):
        from scripts.benchmark import build_dummy_inputs

        inputs = build_dummy_inputs("audio")
        assert "input_values" in inputs
        assert inputs["input_values"].dtype == np.float32

    def test_unknown_task_raises(self):
        from scripts.benchmark import build_dummy_inputs

        with pytest.raises(ValueError):
            build_dummy_inputs("invalid_task")


class TestBenchmarkResult:
    def test_result_is_serializable(self):
        from dataclasses import asdict

        from scripts.benchmark import BenchmarkResult

        result = BenchmarkResult(
            run_id="test_001",
            timestamp="2024-01-01T00:00:00Z",
            model_name="test_model",
            model_path="/tmp/test.onnx",
            model_size_mb=42.5,
            task="ocr",
            runtime="onnxruntime",
            accuracy={"cer": 0.043},
            latency_ms={"mean": 87.3, "p95": 103.2},
            memory_mb={"rss_after_load": 284.6},
        )
        d = asdict(result)
        serialized = json.dumps(d)
        assert "test_model" in serialized


class TestMarkdownGeneration:
    def test_generates_markdown_file(self, tmp_path):
        from scripts.benchmark import BenchmarkResult, generate_markdown

        result = BenchmarkResult(
            run_id="test_001",
            timestamp="2024-01-01T00:00:00Z",
            model_name="trocr_int8",
            model_path="/tmp/test.onnx",
            model_size_mb=84.3,
            task="ocr",
            runtime="onnxruntime",
            accuracy={"cer": 0.043, "wer": 0.089},
            latency_ms={"mean": 87.3, "p95": 103.2},
            memory_mb={"rss_after_load": 284.6},
            size_reduction_pct=93.7,
        )

        out = tmp_path / "result.md"
        generate_markdown(result, out)

        content = out.read_text()
        assert "trocr_int8" in content
        assert "Latency" in content
        assert "93.7%" in content


class TestLatencyProfiler:
    def test_profiles_correctly(self):
        from scripts.benchmark import LatencyProfiler

        call_count = 0

        def fake_inference(inputs):
            nonlocal call_count
            call_count += 1

        profiler = LatencyProfiler(warmup_runs=3, benchmark_runs=10)
        result = profiler.profile(fake_inference, {})

        assert call_count == 13  # 3 warmup + 10 benchmark
        assert "mean" in result
        assert "p95" in result
        assert result["mean"] >= 0

    def test_zero_benchmark_runs_raises(self, monkeypatch, tmp_path):
        import pytest

        from scripts.benchmark import main

        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"\x00" * 100)
        monkeypatch.setattr(
            "sys.argv",
            ["benchmark.py", "--model", str(model_file), "--task", "ocr", "--benchmark-runs", "0"],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2

    def test_negative_warmup_runs_raises(self, monkeypatch, tmp_path):
        import pytest

        from scripts.benchmark import main

        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"\x00" * 100)
        monkeypatch.setattr(
            "sys.argv",
            ["benchmark.py", "--model", str(model_file), "--task", "ocr", "--warmup-runs", "-1"],
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2


class TestMainExitCode:
    def test_exits_1_when_gate_fails(self, monkeypatch, tmp_path):
        from unittest.mock import MagicMock, patch

        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"\x00" * 1_000_000)  # 1 MB

        monkeypatch.setattr(
            "sys.argv",
            [
                "benchmark",
                "--model",
                str(model_file),
                "--task",
                "legal",
                "--max-size-mb",
                "0.001",  # 1 MB > 0.001 MB → gate fails
                "--benchmark-runs",
                "2",
                "--warmup-runs",
                "0",
            ],
        )

        with patch("scripts.benchmark.get_runner", return_value=MagicMock(return_value=[])):
            with pytest.raises(SystemExit) as exc_info:
                from scripts.benchmark import main

                main()

        assert exc_info.value.code == 1


def _make_result(**overrides):
    from scripts.benchmark import BenchmarkResult

    defaults = dict(
        run_id="test_001",
        timestamp="2026-04-25T00:00:00Z",
        model_name="test_model",
        model_path="/tmp/test.onnx",
        model_size_mb=10.0,
        task="legal",
        runtime="onnxruntime",
        latency_ms={"mean": 50.0, "p95": 80.0, "p99": 90.0, "p50": 45.0, "min": 40.0, "max": 100.0},
    )
    defaults.update(overrides)
    return BenchmarkResult(**defaults)


def _make_args(**kwargs):
    import argparse

    defaults = dict(max_size_mb=None, max_latency_ms_p95=None, min_accuracy=None)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestEvaluateGates:
    def test_no_thresholds_passes(self):
        from scripts.benchmark import evaluate_gates

        result = _make_result()
        evaluate_gates(result, _make_args())
        assert result.passed is True
        assert result.gate_results == {}

    def test_size_within_limit_passes(self):
        from scripts.benchmark import evaluate_gates

        result = _make_result(model_size_mb=10.0)
        evaluate_gates(result, _make_args(max_size_mb=20.0))
        assert result.gate_results["max_size_mb"] is True
        assert result.passed is True

    def test_size_exceeds_limit_fails(self):
        from scripts.benchmark import evaluate_gates

        result = _make_result(model_size_mb=50.0)
        evaluate_gates(result, _make_args(max_size_mb=20.0))
        assert result.gate_results["max_size_mb"] is False
        assert result.passed is False

    def test_latency_within_limit_passes(self):
        from scripts.benchmark import evaluate_gates

        result = _make_result()
        evaluate_gates(result, _make_args(max_latency_ms_p95=100.0))
        assert result.gate_results["max_latency_ms_p95"] is True
        assert result.passed is True

    def test_latency_exceeds_limit_fails(self):
        from scripts.benchmark import evaluate_gates

        result = _make_result()
        evaluate_gates(result, _make_args(max_latency_ms_p95=50.0))
        assert result.gate_results["max_latency_ms_p95"] is False
        assert result.passed is False

    def test_latency_gate_fails_when_p95_missing(self):
        from scripts.benchmark import evaluate_gates

        result = _make_result(latency_ms={"mean": 50.0})  # p95 absent
        evaluate_gates(result, _make_args(max_latency_ms_p95=100.0))
        assert result.gate_results["max_latency_ms_p95"] is False
        assert result.passed is False

    def test_multiple_gates_all_must_pass(self):
        from scripts.benchmark import evaluate_gates

        result = _make_result(model_size_mb=10.0)
        # size passes, latency fails
        evaluate_gates(result, _make_args(max_size_mb=20.0, max_latency_ms_p95=50.0))
        assert result.gate_results["max_size_mb"] is True
        assert result.gate_results["max_latency_ms_p95"] is False
        assert result.passed is False

    def test_gate_results_included_in_json(self):
        from dataclasses import asdict

        from scripts.benchmark import evaluate_gates

        result = _make_result(model_size_mb=10.0)
        evaluate_gates(result, _make_args(max_size_mb=20.0))
        d = asdict(result)
        assert "gate_results" in d
        assert "passed" in d
        assert d["gate_results"]["max_size_mb"] is True
        assert d["passed"] is True

    def test_min_accuracy_skipped_when_accuracy_empty_and_no_dataset(self):
        from scripts.benchmark import evaluate_gates

        # No dataset provided: gate cannot be evaluated — skip rather than fail
        result = _make_result(accuracy={})
        evaluate_gates(result, _make_args(min_accuracy=0.9))
        assert "min_accuracy" not in result.gate_results
        assert result.passed is True

    def test_min_accuracy_skips_when_dataset_provided_but_accuracy_empty(self):
        import argparse

        from scripts.benchmark import evaluate_gates

        # Dataset was provided but accuracy evaluation is not yet implemented — gate skipped
        result = _make_result(accuracy={})
        args = argparse.Namespace(
            max_size_mb=None,
            max_latency_ms_p95=None,
            min_accuracy=0.9,
            dataset="/some/dataset",
        )
        evaluate_gates(result, args)
        assert "min_accuracy" not in result.gate_results
        assert result.passed is True

    def test_min_accuracy_passes_when_data_available(self):
        from scripts.benchmark import evaluate_gates

        result = _make_result(accuracy={"accuracy": 0.95})
        evaluate_gates(result, _make_args(min_accuracy=0.9))
        assert result.gate_results["min_accuracy"] is True
        assert result.passed is True

    def test_min_accuracy_uses_zero_accuracy_not_fallback(self):
        from scripts.benchmark import evaluate_gates

        # accuracy=0.0 is a valid (failing) result; must NOT fall through to f1
        result = _make_result(accuracy={"accuracy": 0.0, "f1": 0.99})
        evaluate_gates(result, _make_args(min_accuracy=0.5))
        assert result.gate_results["min_accuracy"] is False
        assert result.passed is False

    def test_min_accuracy_falls_back_to_f1_when_accuracy_key_absent(self):
        from scripts.benchmark import evaluate_gates

        result = _make_result(accuracy={"f1": 0.95})
        evaluate_gates(result, _make_args(min_accuracy=0.9))
        assert result.gate_results["min_accuracy"] is True
        assert result.passed is True
