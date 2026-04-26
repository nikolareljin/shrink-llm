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

    def test_validate_stage_selection_requires_export(self):
        from scripts.run_pipeline import validate_stage_selection

        with pytest.raises(ValueError, match="requires stage\\(s\\): export"):
            validate_stage_selection(["prune", "convert_coreml"])


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

    def test_distill_without_saved_artifacts_keeps_current_state(self, tmp_path):
        from scripts.run_pipeline import init_pipeline_state, update_pipeline_state

        config = _base_config()
        state = init_pipeline_state(config, tmp_path)
        original_state = dict(state)

        update_pipeline_state("distill", state, config, tmp_path)

        assert state == original_state


class TestBuildStageArgs:
    def test_quantize_static_uses_dataset_fallback_and_skip_ops(self, tmp_path):
        from scripts.run_pipeline import build_stage_args, init_pipeline_state

        config = _base_config()
        state = init_pipeline_state(config, tmp_path)

        args = build_stage_args("quantize", config, tmp_path, state)
        assert "--calibration-data" in args
        assert args[args.index("--calibration-data") + 1] == "datasets/legal/eval"
        assert args[args.index("--skip-ops") + 1] == "Softmax,LayerNormalization"

    def test_init_pipeline_state_gptq_returns_directory_artifact(self, tmp_path):
        from scripts.run_pipeline import init_pipeline_state

        config = _base_config()
        config["quantization"] = {"precision": "int4", "mode": "gptq"}

        state = init_pipeline_state(config, tmp_path)
        assert "_gptq" in str(state["quant_path"])
        assert not str(state["quant_path"]).endswith(".onnx")

    def test_validate_gptq_stage_compat_raises_for_onnx_stages(self):
        from scripts.run_pipeline import _validate_gptq_stage_compat

        config = _base_config()
        config["quantization"] = {"precision": "int4", "mode": "gptq"}

        with pytest.raises(ValueError, match="gptq"):
            _validate_gptq_stage_compat(config, ["prune", "quantize", "convert_coreml"])

    def test_validate_gptq_stage_compat_allows_non_onnx_stages(self):
        from scripts.run_pipeline import _validate_gptq_stage_compat

        config = _base_config()
        config["quantization"] = {"precision": "int4", "mode": "gptq"}

        _validate_gptq_stage_compat(config, ["prune", "quantize", "distill"])

    def test_init_pipeline_state_rejects_unsupported_precision_for_dynamic_mode(self, tmp_path):
        from scripts.run_pipeline import init_pipeline_state

        config = _base_config()
        config["quantization"] = {"precision": "int4", "mode": "dynamic"}

        with pytest.raises(ValueError, match="Unsupported precision 'int4'"):
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
        assert args[args.index("--output-json") + 1].endswith("demo-model_base.json")

    def test_benchmark_uses_quantized_label_after_quantize(self, tmp_path):
        from scripts.run_pipeline import (
            build_stage_args,
            init_pipeline_state,
            update_pipeline_state,
        )

        config = _base_config()
        state = init_pipeline_state(config, tmp_path)

        update_pipeline_state("quantize", state, config, tmp_path)

        args = build_stage_args("benchmark", config, tmp_path, state)
        assert args[args.index("--output-json") + 1].endswith("demo-model_int8.json")


class TestUpdateManifest:
    def test_creates_manifest_when_missing(self, tmp_path):
        import json

        from scripts.run_pipeline import _update_manifest

        path = tmp_path / "manifest.json"
        _update_manifest(path, "export", ["--model", "m"], True)

        data = json.loads(path.read_text())
        assert len(data["stages"]) == 1
        rec = data["stages"][0]
        assert rec["stage"] == "export"
        assert rec["args"] == ["--model", "m"]
        assert rec["status"] == "ok"
        assert rec["exit_code"] == 0
        assert rec["artifacts"] == []
        assert rec["error"] is None

    def test_appends_to_existing_manifest(self, tmp_path):
        import json

        from scripts.run_pipeline import _update_manifest

        path = tmp_path / "manifest.json"
        _update_manifest(path, "export", [], True)
        _update_manifest(path, "quantize", [], False)

        data = json.loads(path.read_text())
        assert len(data["stages"]) == 2
        assert data["stages"][0]["stage"] == "export"
        assert data["stages"][1]["stage"] == "quantize"
        assert data["stages"][1]["status"] == "failed"

    def test_resets_on_malformed_json(self, tmp_path):
        import json

        from scripts.run_pipeline import _update_manifest

        path = tmp_path / "manifest.json"
        path.write_text("not valid json {{")
        _update_manifest(path, "prune", [], True)

        data = json.loads(path.read_text())
        assert len(data["stages"]) == 1
        assert data["stages"][0]["stage"] == "prune"

    def test_resets_on_non_dict_json(self, tmp_path):
        import json

        from scripts.run_pipeline import _update_manifest

        path = tmp_path / "manifest.json"
        path.write_text(json.dumps([1, 2, 3]))
        _update_manifest(path, "distill", [], True)

        data = json.loads(path.read_text())
        assert len(data["stages"]) == 1
        assert data["stages"][0]["stage"] == "distill"

    def test_resets_stages_when_not_list(self, tmp_path):
        import json

        from scripts.run_pipeline import _update_manifest

        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"stages": "corrupted", "extra": "kept"}))
        _update_manifest(path, "benchmark", [], True)

        data = json.loads(path.read_text())
        assert data["extra"] == "kept"
        assert len(data["stages"]) == 1
        assert data["stages"][0]["stage"] == "benchmark"

    def test_records_exit_code_and_error_on_failure(self, tmp_path):
        import json

        from scripts.run_pipeline import _update_manifest

        path = tmp_path / "manifest.json"
        error = {"message": "Stage 'export' exited with code 1", "exit_code": 1}
        _update_manifest(path, "export", [], False, exit_code=1, error=error)

        data = json.loads(path.read_text())
        rec = data["stages"][0]
        assert rec["status"] == "failed"
        assert rec["exit_code"] == 1
        assert "exited with code 1" in rec["error"]["message"]
        assert rec["error"]["exit_code"] == 1

    def test_records_artifacts_for_successful_stage(self, tmp_path):
        import json

        from scripts.run_pipeline import _update_manifest

        path = tmp_path / "manifest.json"
        artifacts = [{"path": "model_base.onnx", "size_mb": 42.5}]
        _update_manifest(path, "export", [], True, exit_code=0, artifacts=artifacts)

        data = json.loads(path.read_text())
        rec = data["stages"][0]
        assert rec["artifacts"] == [{"path": "model_base.onnx", "size_mb": 42.5}]


class TestManifestHelpers:
    def test_init_manifest_writes_metadata(self, tmp_path):
        import json

        from scripts.run_pipeline import _init_manifest

        config = {"model": "org/demo-model", "task": "legal"}
        config_path = tmp_path / "pipeline.yaml"
        config_path.write_text("model: org/demo-model\ntask: legal\n")
        manifest_path = tmp_path / "manifest.json"

        _init_manifest(manifest_path, config, config_path, ["export", "quantize"])

        data = json.loads(manifest_path.read_text())
        assert data["version"] == "1"
        assert data["model_id"] == "org/demo-model"
        assert data["task"] == "legal"
        assert data["total_stages"] == 2
        assert data["config_hash"].startswith("sha256:")
        assert "pipeline_run_id" in data
        assert data["stages"] == []

    def test_init_manifest_preserves_unknown_keys(self, tmp_path):
        import json

        from scripts.run_pipeline import _init_manifest

        config = {"model": "org/model", "task": "ocr"}
        config_path = tmp_path / "pipeline.yaml"
        config_path.write_text("model: org/model\ntask: ocr\n")
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"custom_key": "preserved", "stages": []}))

        _init_manifest(manifest_path, config, config_path, ["export"])

        data = json.loads(manifest_path.read_text())
        assert data["custom_key"] == "preserved"
        assert data["version"] == "1"
        assert data["stages"] == []

    def test_collect_new_files_excludes_manifest(self, tmp_path):
        from scripts.run_pipeline import _collect_new_files, _snapshot_dir

        before = _snapshot_dir(tmp_path)
        (tmp_path / "model.onnx").write_bytes(b"\x00" * 1024)
        (tmp_path / "manifest.json").write_text("{}")

        result = _collect_new_files(tmp_path, before)

        paths = [r["path"] for r in result]
        assert "model.onnx" in paths
        assert "manifest.json" not in paths

    def test_collect_new_files_reports_size(self, tmp_path):
        from scripts.run_pipeline import _collect_new_files, _snapshot_dir

        before = _snapshot_dir(tmp_path)
        (tmp_path / "artifact.onnx").write_bytes(b"\x00" * 2_000_000)  # exactly 2 MB (1e6)

        result = _collect_new_files(tmp_path, before)

        assert len(result) == 1
        assert result[0]["size_mb"] == pytest.approx(2.0, abs=0.001)

    def test_collect_new_files_detects_overwritten_file(self, tmp_path):
        import os

        from scripts.run_pipeline import _collect_new_files, _snapshot_dir

        artifact = tmp_path / "model.onnx"
        artifact.write_bytes(b"\x00" * 512)
        before = _snapshot_dir(tmp_path)
        artifact.write_bytes(b"\x00" * 1024)  # overwrite same path with different size
        # Force mtime_ns forward to guarantee detection on coarse-resolution filesystems
        prev_mtime_ns = before[artifact][0]
        os.utime(artifact, ns=(prev_mtime_ns + 1_000_000_000, prev_mtime_ns + 1_000_000_000))

        result = _collect_new_files(tmp_path, before)

        assert len(result) == 1
        assert result[0]["path"] == "model.onnx"

    def test_collect_new_files_detects_directory_artifact(self, tmp_path):
        from scripts.run_pipeline import _collect_new_files, _snapshot_dir

        before = _snapshot_dir(tmp_path)
        pkg = tmp_path / "model.mlpackage"
        pkg.mkdir()
        (pkg / "weights.bin").write_bytes(b"\x00" * 1_000_000)
        (pkg / "metadata.json").write_text("{}")

        result = _collect_new_files(tmp_path, before)

        assert len(result) == 1
        assert result[0]["path"] == "model.mlpackage"
        assert result[0]["size_mb"] == pytest.approx(1.0, abs=0.001)

    def test_collect_new_files_lists_files_inside_preexisting_dir(self, tmp_path):
        from scripts.run_pipeline import _collect_new_files, _snapshot_dir

        # Pre-existing subdir (e.g. benchmarks/) was there before the stage
        existing_dir = tmp_path / "benchmarks"
        existing_dir.mkdir()
        before = _snapshot_dir(tmp_path)

        # Stage adds a file inside it
        (existing_dir / "result.json").write_bytes(b"{}" * 10)

        result = _collect_new_files(tmp_path, before)

        assert len(result) == 1
        assert result[0]["path"] == "benchmarks/result.json"

    def test_benchmark_stage_args_include_success_criteria(self, tmp_path):
        from scripts.run_pipeline import build_stage_args, init_pipeline_state

        config = _base_config()
        config["success_criteria"] = {"max_size_mb": 20, "max_latency_ms": 100}
        state = init_pipeline_state(config, tmp_path)

        args = build_stage_args("benchmark", config, tmp_path, state)
        assert "--max-size-mb" in args
        assert args[args.index("--max-size-mb") + 1] == "20"
        assert "--max-latency-ms-p95" in args
        assert args[args.index("--max-latency-ms-p95") + 1] == "100"

    def test_benchmark_stage_args_maps_min_f1_to_min_accuracy(self, tmp_path):
        from scripts.run_pipeline import build_stage_args, init_pipeline_state

        config = _base_config()
        config["success_criteria"] = {"min_f1": 0.80}
        state = init_pipeline_state(config, tmp_path)

        args = build_stage_args("benchmark", config, tmp_path, state)
        assert "--min-accuracy" in args
        assert args[args.index("--min-accuracy") + 1] == "0.8"

    def test_benchmark_stage_args_logs_unknown_criteria(self, tmp_path, caplog):
        import logging

        from scripts.run_pipeline import build_stage_args, init_pipeline_state

        config = _base_config()
        config["success_criteria"] = {"max_size_mb": 20, "totally_unknown_key": 99}
        state = init_pipeline_state(config, tmp_path)

        with caplog.at_level(logging.DEBUG, logger="scripts.run_pipeline"):
            build_stage_args("benchmark", config, tmp_path, state)

        assert any("totally_unknown_key" in r.message for r in caplog.records)

    def test_init_manifest_run_id_has_millisecond_precision(self, tmp_path):
        import json
        import re

        from scripts.run_pipeline import _init_manifest

        config = {"model": "org/model", "task": "legal"}
        config_path = tmp_path / "pipeline.yaml"
        config_path.write_text("model: org/model\ntask: legal\n")
        manifest_path = tmp_path / "manifest.json"

        _init_manifest(manifest_path, config, config_path, ["export"])

        data = json.loads(manifest_path.read_text())
        assert re.match(r"legal_\d{8}_\d{6}_\d{3}$", data["pipeline_run_id"])
