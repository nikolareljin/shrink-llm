"""Tests for pipeline orchestration helpers."""

from __future__ import annotations

import pytest


def _base_config() -> dict:
    return {
        "model": "org/demo-model",
        "task": "legal",
        "teacher": "org/teacher-model",
        "quantization": {
            "precision": "int8",
            "mode": "static",
            "calibration_samples": 32,
            "skip_ops": ["Softmax", "LayerNormalization"],
        },
        "distillation": {
            "dataset": "datasets/legal/train",
            "temperature": 4.0,
            "alpha": 0.2,
            "beta": 0.8,
            "gamma": 0.3,
            "align_hidden": True,
        },
        "mobile": {
            "ios": {
                "deployment_target": "iOS17",
                "compute_units": "CPU_AND_NE",
                "quantization": "fp16",
            }
        },
        "benchmark": {"dataset": "datasets/legal/eval"},
    }


class TestOrderStages:
    def test_orders_requested_stages_canonically(self):
        from scripts.run_pipeline import order_stages

        stages = order_stages(["benchmark", "quantize", "prune", "export"])
        assert stages == ["prune", "export", "quantize", "benchmark"]


class TestPipelineState:
    def test_prune_updates_export_model_path(self, tmp_path):
        from scripts.run_pipeline import (
            build_stage_args,
            init_pipeline_state,
            update_pipeline_state,
        )

        config = _base_config()
        state = init_pipeline_state(config, tmp_path)
        update_pipeline_state("prune", state, config, tmp_path)

        args = build_stage_args("export", config, tmp_path, state)
        assert args[1] == str(tmp_path / "pruned")
        assert args[5] == str(tmp_path / "demo-model_pruned_base.onnx")


class TestBuildStageArgs:
    def test_quantize_static_uses_dataset_fallback_and_skip_ops(self, tmp_path):
        from scripts.run_pipeline import build_stage_args, init_pipeline_state

        config = _base_config()
        state = init_pipeline_state(config, tmp_path)

        args = build_stage_args("quantize", config, tmp_path, state)
        assert "--calibration-data" in args
        assert args[args.index("--calibration-data") + 1] == "datasets/legal/eval"
        assert args[args.index("--skip-ops") + 1] == "Softmax,LayerNormalization"

    def test_init_pipeline_state_rejects_gptq(self, tmp_path):
        from scripts.run_pipeline import init_pipeline_state

        config = _base_config()
        config["quantization"] = {"precision": "int4", "mode": "gptq"}

        with pytest.raises(ValueError, match="gptq"):
            init_pipeline_state(config, tmp_path)

    def test_distill_forwards_gamma_and_align_hidden(self, tmp_path):
        from scripts.run_pipeline import build_stage_args, init_pipeline_state

        config = _base_config()
        state = init_pipeline_state(config, tmp_path)

        args = build_stage_args("distill", config, tmp_path, state)
        assert args[args.index("--gamma") + 1] == "0.3"
        assert "--align-hidden" in args

    def test_distill_skips_unsupported_tasks(self, tmp_path):
        from scripts.run_pipeline import build_stage_args, init_pipeline_state

        config = _base_config()
        config["task"] = "audio"
        state = init_pipeline_state(config, tmp_path)

        assert build_stage_args("distill", config, tmp_path, state) == []

    def test_convert_coreml_forwards_quantization(self, tmp_path):
        from scripts.run_pipeline import build_stage_args, init_pipeline_state

        config = _base_config()
        state = init_pipeline_state(config, tmp_path)

        args = build_stage_args("convert_coreml", config, tmp_path, state)
        assert args[args.index("--quantization") + 1] == "fp16"

    def test_convert_coreml_falls_back_to_exported_onnx_without_quantize(self, tmp_path):
        from scripts.run_pipeline import build_stage_args, init_pipeline_state

        config = _base_config()
        config.pop("quantization")
        state = init_pipeline_state(config, tmp_path)

        args = build_stage_args("convert_coreml", config, tmp_path, state)
        assert args[args.index("--input") + 1] == str(tmp_path / "demo-model_base.onnx")

    def test_benchmark_falls_back_to_exported_onnx_without_quantize(self, tmp_path):
        from scripts.run_pipeline import build_stage_args, init_pipeline_state

        config = _base_config()
        config.pop("quantization")
        state = init_pipeline_state(config, tmp_path)

        args = build_stage_args("benchmark", config, tmp_path, state)
        assert args[args.index("--model") + 1] == str(tmp_path / "demo-model_base.onnx")
