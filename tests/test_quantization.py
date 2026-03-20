"""Tests for quantization pipeline."""

from __future__ import annotations


class TestCalibrationDataReader:
    def test_synthetic_fallback_when_no_files(self, tmp_path):
        from scripts.quantize import SimpleCalibrationDataReader

        reader = SimpleCalibrationDataReader(
            tmp_path, ["input_ids", "attention_mask"], max_samples=10
        )
        batch = reader.get_next()
        assert batch is not None
        assert "input_ids" in batch or "attention_mask" in batch

    def test_exhausts_correctly(self, tmp_path):
        from scripts.quantize import SimpleCalibrationDataReader

        reader = SimpleCalibrationDataReader(tmp_path, ["input_values"], max_samples=5)
        count = 0
        while reader.get_next() is not None:
            count += 1
        assert count == 5  # synthetic fallback produces min(64, max_samples) batches


class TestDefaultSkipOps:
    def test_default_skip_ops_are_sensible(self):
        from scripts.quantize import DEFAULT_SKIP_OPS

        assert "Softmax" in DEFAULT_SKIP_OPS
        assert "LayerNormalization" in DEFAULT_SKIP_OPS


class TestSizeComparison:
    def test_size_comparison_logs(self, tmp_path, caplog):
        from scripts.quantize import _log_size_comparison

        before = tmp_path / "before.onnx"
        after = tmp_path / "after.onnx"
        before.write_bytes(b"0" * 1000000)
        after.write_bytes(b"0" * 250000)

        import logging

        with caplog.at_level(logging.INFO):
            _log_size_comparison(before, after)

        assert "75.0%" in caplog.text or "reduction" in caplog.text
