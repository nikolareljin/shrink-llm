"""
run_pipeline.py — Orchestrate the full ShrinkLLM compression pipeline from a YAML config.

Runs stages: prune → distill → export → quantize → convert → benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

VALID_STAGES = [
    "prune",
    "distill",
    "export",
    "quantize",
    "convert_tflite",
    "convert_coreml",
    "convert_onnx_mobile",
    "benchmark",
]

STAGE_PREREQUISITES = {
    "quantize": {"export"},
    "convert_tflite": {"export"},
    "convert_coreml": {"export"},
    "convert_onnx_mobile": {"export"},
    "benchmark": {"export"},
}


def run_stage(script: str, args_list: list[str], dry_run: bool = False) -> bool:
    cmd = [sys.executable, f"scripts/{script}.py"] + args_list
    log.info("Running: %s", " ".join(cmd))
    if dry_run:
        log.info("[DRY RUN] Skipping execution.")
        return True
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.error("Stage '%s' failed with return code %d", script, result.returncode)
        return False
    return True


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def order_stages(requested_stages: Iterable[str]) -> list[str]:
    requested = {stage.strip() for stage in requested_stages if stage.strip()}
    return [stage for stage in VALID_STAGES if stage in requested]


def validate_stage_selection(stages: Iterable[str]) -> None:
    stages = list(stages)
    selected = set(stages)
    for stage in stages:
        missing = sorted(STAGE_PREREQUISITES.get(stage, set()) - selected)
        if missing:
            missing_list = ", ".join(missing)
            raise ValueError(f"Stage '{stage}' requires stage(s): {missing_list}")


def _validated_quantization_settings(config: dict) -> tuple[str, str]:
    q = config.get("quantization", {})
    precision = str(q.get("precision", "int8")).lower()
    mode = str(q.get("mode", "dynamic")).lower()
    supported_modes = {"dynamic", "static", "gptq"}
    supported_precisions = {"int8", "fp16"}
    gptq_precisions = {"int4", "mixed", "int8", "fp16"}

    if mode not in supported_modes:
        supported_modes_list = ", ".join(sorted(supported_modes))
        raise ValueError(
            f"Unsupported quantization mode '{mode}'. Supported modes: {supported_modes_list}."
        )

    if mode == "gptq":
        if precision not in gptq_precisions:
            supported_precisions_list = ", ".join(sorted(gptq_precisions))
            raise ValueError(
                f"Unsupported precision '{precision}' for mode '{mode}'. "
                f"Supported precisions for gptq: {supported_precisions_list}."
            )
    elif precision not in supported_precisions:
        supported_precisions_list = ", ".join(sorted(supported_precisions))
        raise ValueError(
            f"Unsupported precision '{precision}' for mode '{mode}'. "
            f"Use one of: {supported_precisions_list}."
        )

    return mode, precision


def _quantized_artifact_info(config: dict, output_dir: Path, model_label: str) -> tuple[Path, str]:
    mode, precision = _validated_quantization_settings(config)
    if mode == "gptq":
        raise ValueError(
            "quantization.mode='gptq' is not supported by run_pipeline.py: "
            "GPTQ produces a directory artifact, but downstream convert_* and benchmark "
            "stages expect an ONNX file path. Use a non-GPTQ quantization mode for this pipeline."
        )
    return output_dir / f"{model_label}_{precision}.onnx", f"{model_label}_{precision}"


def init_pipeline_state(config: dict, output_dir: Path) -> dict[str, Path | str]:
    model_id = str(config.get("model", ""))
    model_label = Path(model_id).name if model_id else "model"
    onnx_path = output_dir / f"{model_label}_base.onnx"
    quant_path, quant_label = _quantized_artifact_info(config, output_dir, model_label)
    return {
        "model_id": model_id,
        "model_label": model_label,
        "onnx_path": onnx_path,
        "quant_path": quant_path,
        "quant_label": quant_label,
        "current_onnx_path": onnx_path,
        "current_onnx_label": onnx_path.stem,
    }


def _has_model_artifacts(model_dir: Path) -> bool:
    return any((model_dir / name).exists() for name in ("config.json", "tokenizer_config.json"))


def update_pipeline_state(
    stage: str, state: dict[str, Path | str], config: dict, output_dir: Path
) -> None:
    if stage == "quantize":
        state["current_onnx_path"] = state["quant_path"]
        state["current_onnx_label"] = state["quant_label"]
        return

    if stage not in {"prune", "distill"}:
        return

    suffix = "pruned" if stage == "prune" else "distilled"
    stage_output = output_dir / suffix
    if stage == "distill" and not _has_model_artifacts(stage_output):
        log.warning(
            "Stage '%s' did not produce reusable model artifacts in %s; keeping current model state.",
            stage,
            stage_output,
        )
        return

    model_label = f"{state['model_label']}_{suffix}"
    onnx_path = output_dir / f"{model_label}_base.onnx"
    quant_path, quant_label = _quantized_artifact_info(config, output_dir, model_label)
    state.update(
        {
            "model_id": str(stage_output),
            "model_label": model_label,
            "onnx_path": onnx_path,
            "quant_path": quant_path,
            "quant_label": quant_label,
            "current_onnx_path": onnx_path,
            "current_onnx_label": onnx_path.stem,
        }
    )


def build_stage_args(
    stage: str,
    config: dict,
    output_dir: Path,
    state: dict[str, Path | str],
) -> list[str]:
    """Build CLI args for each stage from config."""
    model_id = str(state["model_id"])
    task = config.get("task", "")
    model_label = str(state["model_label"])
    onnx_path = str(state["onnx_path"])
    quant_path = str(state["quant_path"])
    current_onnx_input = str(state.get("current_onnx_path", state["onnx_path"]))

    if stage == "export":
        return [
            "--model",
            model_id,
            "--task",
            task,
            "--output",
            onnx_path,
            "--opset",
            str(config.get("onnx_opset", 17)),
            "--validate",
        ]

    elif stage == "quantize":
        q = config.get("quantization", {})
        mode, precision = _validated_quantization_settings(config)
        args = [
            "--input",
            onnx_path,
            "--output",
            quant_path,
            "--precision",
            precision,
            "--mode",
            mode,
        ]
        if mode == "static":
            calibration_data = (
                q.get("calibration_data")
                or config.get("benchmark", {}).get("dataset")
                or f"datasets/{task}"
            )
            args += ["--calibration-data", str(calibration_data)]
        if q.get("calibration_samples"):
            args += ["--calibration-samples", str(q["calibration_samples"])]
        skip_ops = q.get("skip_ops")
        if skip_ops:
            if isinstance(skip_ops, list):
                skip_ops = ",".join(str(op) for op in skip_ops)
            else:
                skip_ops = str(skip_ops)
            if skip_ops:
                args += ["--skip-ops", skip_ops]
        return args

    elif stage == "prune":
        p = config.get("pruning", {})
        return [
            "--model",
            model_id,
            "--task",
            task,
            "--method",
            p.get("method", "magnitude"),
            "--sparsity",
            str(p.get("sparsity", 0.3)),
            "--output-dir",
            str(output_dir / "pruned"),
            "--finetune-epochs",
            str(p.get("finetune_epochs", 0)),
        ]

    elif stage == "distill":
        d = config.get("distillation", {})
        teacher = config.get("teacher", "")
        if not teacher:
            log.warning("Distill stage requested but 'teacher' not set in config, skipping")
            return []
        if task != "legal":
            log.warning("Distill stage supports only task='legal', skipping task='%s'", task)
            return []
        args = [
            "--teacher",
            teacher,
            "--student",
            model_id,
            "--task",
            task,
            "--dataset",
            str(d.get("dataset", "datasets/train")),
            "--output-dir",
            str(output_dir / "distilled"),
            "--temperature",
            str(d.get("temperature", 6.0)),
            "--alpha",
            str(d.get("alpha", 0.1)),
            "--beta",
            str(d.get("beta", 0.9)),
            "--gamma",
            str(d.get("gamma", 0.1)),
            "--epochs",
            str(d.get("epochs", 10)),
            "--batch-size",
            str(d.get("batch_size", 16)),
            "--lr",
            str(d.get("lr", 5e-5)),
        ]
        if d.get("align_hidden"):
            args.append("--align-hidden")
        if d.get("fp16"):
            args.append("--fp16")
        return args

    elif stage == "convert_tflite":
        m = config.get("mobile", {}).get("android", {})
        quantization = m.get("quantization", "int8")
        representative_dataset = None

        if quantization == "int8":
            representative_dataset = (
                m.get("representative_dataset")
                or config.get("quantization", {}).get("calibration_data")
                or config.get("benchmark", {}).get("dataset")
            )
            if not representative_dataset:
                log.warning(
                    "convert_tflite requested quantization='int8' but no representative dataset "
                    "was configured (checked mobile.android.representative_dataset, "
                    "quantization.calibration_data, benchmark.dataset). Downgrading to 'fp16'."
                )
                quantization = "fp16"

        args = [
            "--input",
            current_onnx_input,
            "--output",
            str(output_dir / f"{model_label}.tflite"),
            "--quantization",
            quantization,
        ]
        if representative_dataset and quantization == "int8":
            args.extend(["--representative-dataset", str(representative_dataset)])
        return args

    elif stage == "convert_coreml":
        m = config.get("mobile", {}).get("ios", {})
        return [
            "--input",
            current_onnx_input,
            "--output",
            str(output_dir / f"{model_label}.mlpackage"),
            "--minimum-deployment-target",
            m.get("deployment_target", "iOS16"),
            "--compute-units",
            m.get("compute_units", "ALL"),
            "--quantization",
            m.get("quantization", "none"),
        ]

    elif stage == "convert_onnx_mobile":
        return [
            "--input",
            current_onnx_input,
            "--output",
            str(output_dir / f"{model_label}_mobile.onnx"),
        ]

    elif stage == "benchmark":
        b = config.get("benchmark", {})
        result_name = str(state.get("current_onnx_label", Path(current_onnx_input).stem))
        args = [
            "--model",
            current_onnx_input,
            "--task",
            task,
            "--runtime",
            b.get("runtime", "onnxruntime"),
            "--warmup-runs",
            str(b.get("warmup_runs", 10)),
            "--benchmark-runs",
            str(b.get("benchmark_runs", 100)),
            "--output-json",
            str(output_dir / "benchmarks" / f"{result_name}.json"),
            "--output-md",
            str(output_dir / "benchmarks" / f"{result_name}.md"),
        ]
        if b.get("dataset"):
            args += ["--dataset", b["dataset"]]
        return args

    return []


def _update_manifest(manifest_path: Path, stage: str, args_list: list[str], success: bool) -> None:
    """Append a stage record to the pipeline manifest JSON."""
    record: dict = {
        "stage": stage,
        "args": args_list,
        "status": "ok" if success else "failed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    existing: dict = {"stages": []}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text())
            if isinstance(loaded, dict):
                existing = loaded
            else:
                log.warning(
                    "Manifest %s contained %s JSON, expected object — starting fresh",
                    manifest_path,
                    type(loaded).__name__,
                )
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read manifest %s: %s — starting fresh", manifest_path, exc)
    stages = existing.get("stages")
    if not isinstance(stages, list):
        if "stages" in existing:
            log.warning(
                "Manifest %s has non-list 'stages' field (%s) — resetting",
                manifest_path,
                type(stages).__name__,
            )
        existing["stages"] = []
    existing["stages"].append(record)
    manifest_path.write_text(json.dumps(existing, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full ShrinkLLM compression pipeline")
    parser.add_argument("--config", required=True, type=Path, help="Pipeline YAML config file")
    parser.add_argument(
        "--stages",
        default=",".join(VALID_STAGES),
        help=f"Comma-separated stages to run (default: all). Options: {', '.join(VALID_STAGES)}",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/student"), help="Output directory"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    config = load_config(args.config)
    if not isinstance(config, dict):
        parser.error(f"Config file must be a YAML mapping, got {type(config).__name__}")

    if not config.get("model"):
        parser.error("Config must specify a non-empty 'model' field")
    if not config.get("task"):
        parser.error("Config must specify a non-empty 'task' field")

    requested_stages = [s.strip() for s in args.stages.split(",")]
    stages = order_stages(requested_stages)
    try:
        validate_stage_selection(stages)
    except ValueError as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Pipeline config: %s", args.config)
    log.info("Stages: %s", stages)
    log.info("Output dir: %s", args.output_dir)

    script_map = {
        "export": "export_to_onnx",
        "quantize": "quantize",
        "prune": "prune",
        "distill": "distill",
        "convert_tflite": "convert_to_tflite",
        "convert_coreml": "convert_to_coreml",
        "convert_onnx_mobile": "convert_to_onnx_mobile",
        "benchmark": "benchmark",
    }

    manifest_path = args.output_dir / "manifest.json"
    state = init_pipeline_state(config, args.output_dir)
    results = {}
    unknown_stages = [stage for stage in requested_stages if stage and stage not in VALID_STAGES]
    for stage in unknown_stages:
        log.warning("Unknown stage '%s' — skipping", stage)

    for stage in stages:
        stage_args = build_stage_args(stage, config, args.output_dir, state)
        if not stage_args and stage not in ("export",):
            log.warning("Stage '%s' produced no args, skipping", stage)
            continue
        success = run_stage(script_map[stage], stage_args, dry_run=args.dry_run)
        results[stage] = "OK" if success else "FAILED"
        if not args.dry_run:
            _update_manifest(manifest_path, stage, stage_args, success)
        if success:
            update_pipeline_state(stage, state, config, args.output_dir)
        if not success:
            log.error("Pipeline aborted at stage '%s'", stage)
            break

    log.info("\nPipeline summary:")
    for stage, status in results.items():
        symbol = "✓" if status == "OK" else "✗"
        log.info("  %s %s: %s", symbol, stage, status)

    if not args.dry_run and manifest_path.exists():
        log.info("Manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
