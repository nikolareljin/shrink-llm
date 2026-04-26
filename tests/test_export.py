"""Tests for ONNX export pipeline."""

from __future__ import annotations

import pytest
import torch


class TestBuildDummyInputs:
    def test_ocr_inputs(self):
        from scripts.export_to_onnx import build_dummy_inputs

        inputs = build_dummy_inputs("ocr", "cpu")
        assert "pixel_values" in inputs
        assert inputs["pixel_values"].shape == (1, 3, 384, 384)
        assert inputs["pixel_values"].dtype == torch.float32

    def test_legal_inputs(self):
        from scripts.export_to_onnx import build_dummy_inputs

        inputs = build_dummy_inputs("legal", "cpu")
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert inputs["input_ids"].shape == (1, 128)

    def test_audio_inputs(self):
        from scripts.export_to_onnx import build_dummy_inputs

        inputs = build_dummy_inputs("audio", "cpu")
        assert "input_values" in inputs
        assert inputs["input_values"].shape == (1, 16000)

    def test_unknown_task_raises(self):
        from scripts.export_to_onnx import build_dummy_inputs

        with pytest.raises(ValueError, match="Unknown task"):
            build_dummy_inputs("unknown_task", "cpu")


class TestTaskConfigs:
    def test_all_tasks_have_required_keys(self):
        from scripts.export_to_onnx import TASK_CONFIGS

        required = {"model_class", "processor_class", "input_names", "output_names", "dynamic_axes"}
        for task, cfg in TASK_CONFIGS.items():
            assert required.issubset(cfg.keys()), f"Task '{task}' missing keys"

    def test_dynamic_axes_keys_match_input_names(self):
        from scripts.export_to_onnx import TASK_CONFIGS

        for task, cfg in TASK_CONFIGS.items():
            for name in cfg["input_names"]:
                assert name in cfg["dynamic_axes"], f"Task '{task}': '{name}' not in dynamic_axes"
