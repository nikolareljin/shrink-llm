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
