"""
run_pipeline.py — Orchestrate the full ShrinkLLM compression pipeline from a YAML config.

Runs stages: prune → distill → export → quantize → convert → benchmark
"""

from __future__ import annotations

import argparse
import hashlib
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


def run_stage(script: str, args_list: list[str], dry_run: bool = False) -> tuple[int, bool]:
    cmd = [sys.executable, f"scripts/{script}.py"] + args_list
    log.info("Running: %s", " ".join(cmd))
    if dry_run:
        log.info("[DRY RUN] Skipping execution.")
        return 0, True
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.error("Stage '%s' failed with return code %d", script, result.returncode)
        return result.returncode, False
    return result.returncode, True


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def order_stages(requested_stages: Iterable[str]) -> list[str]:
    requested = {stage.strip() for stage in requested_stages if stage.strip()}
    return [stage for stage in VALID_STAGES if stage in requested]


_RUNTIME_CONVERT_STAGE = {
    "tflite": "convert_tflite",
    "coreml": "convert_coreml",
    "onnxruntime_mobile": "convert_onnx_mobile",
}


def validate_stage_selection(stages: Iterable[str], config: dict | None = None) -> None:
    stages = list(stages)
    selected = set(stages)
    prerequisites = dict(STAGE_PREREQUISITES)
    if config is not None:
        mode, _ = _validated_quantization_settings(config)
        if mode == "gptq":
            prerequisites["quantize"] = prerequisites.get("quantize", set()) - {"export"}
        runtime = (config.get("benchmark") or {}).get("runtime", "onnxruntime")
        convert_stage = _RUNTIME_CONVERT_STAGE.get(runtime)
        if convert_stage:
            prerequisites["benchmark"] = prerequisites.get("benchmark", set()) | {convert_stage}
    for stage in stages:
        missing = sorted(prerequisites.get(stage, set()) - selected)
        if missing:
            missing_list = ", ".join(missing)
            raise ValueError(f"Stage '{stage}' requires stage(s): {missing_list}")


def _validated_quantization_settings(config: dict) -> tuple[str, str]:
    q = config.get("quantization")
    if q is None:
        q = {}
    elif not isinstance(q, dict):
        raise ValueError(
            "Invalid 'quantization' configuration: expected a mapping/object, "
            f"got {type(q).__name__}."
        )
    precision = str(q.get("precision", "int8")).lower()
    mode = str(q.get("mode", "dynamic")).lower()
    supported_modes = {"dynamic", "static", "gptq"}
    supported_precisions = {"int8", "fp16"}
    gptq_precisions = {"int4", "int8"}

    if mode not in supported_modes:
        supported_modes_list = ", ".join(sorted(supported_modes))
        raise ValueError(
            f"Unsupported quantization mode '{mode}'. Supported modes: {supported_modes_list}."
        )

    if mode == "static" and precision == "fp16":
        raise ValueError(
            "quantization.mode='static' requires calibration data and only supports int8 precision. "
            "Use precision='int8' for static quantization, or switch to mode='dynamic' for fp16."
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


_ONNX_REQUIRED_STAGES = {
    "convert_tflite",
    "convert_coreml",
    "convert_onnx_mobile",
    "benchmark",
}


def _quantized_artifact_info(config: dict, output_dir: Path, model_label: str) -> tuple[Path, str]:
    mode, precision = _validated_quantization_settings(config)
    if mode == "gptq":
        artifact_label = f"{model_label}_{precision}_gptq"
        return output_dir / artifact_label, artifact_label
    return output_dir / f"{model_label}_{precision}.onnx", f"{model_label}_{precision}"


def _validate_gptq_stage_compat(config: dict, stages: list[str]) -> None:
    mode, _ = _validated_quantization_settings(config)
    if mode != "gptq":
        return
    conflicting = sorted(_ONNX_REQUIRED_STAGES & set(stages))
    if conflicting:
        raise ValueError(
            f"quantization.mode='gptq' produces a directory artifact incompatible with "
            f"ONNX-dependent stages: {', '.join(conflicting)}. "
            "Remove those stages or switch to a non-GPTQ quantization mode."
        )


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
    stage: str,
    state: dict[str, Path | str],
    config: dict,
    output_dir: Path,
    dry_run: bool = False,
) -> None:
    if stage == "quantize":
        state["current_onnx_path"] = state["quant_path"]
        state["current_onnx_label"] = state["quant_label"]
        return

    if stage not in {"prune", "distill"}:
        return

    suffix = "pruned" if stage == "prune" else "distilled"
    stage_output = output_dir / suffix
    if stage == "distill" and not dry_run and not _has_model_artifacts(stage_output):
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
    dry_run: bool = False,
) -> list[str]:
    """Build CLI args for each stage from config."""
    model_id = str(state["model_id"])
    task = config.get("task", "")
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
        q = config.get("quantization") or {}
        mode, precision = _validated_quantization_settings(config)
        args = [
            "--output",
            quant_path,
            "--precision",
            precision,
            "--mode",
            mode,
        ]
        if mode == "gptq":
            args += ["--model-id", model_id]
        else:
            args = ["--input", onnx_path] + args
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
        p = config.get("pruning") or {}
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
        d = config.get("distillation") or {}
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
            elif not dry_run:
                dataset_path = Path(representative_dataset)
                if not dataset_path.exists() or not dataset_path.is_dir():
                    log.warning(
                        "convert_tflite requested quantization='int8' but representative dataset "
                        "path '%s' does not exist or is not a directory. Downgrading to 'fp16'.",
                        dataset_path,
                    )
                    representative_dataset = None
                    quantization = "fp16"
                elif not any(dataset_path.glob("*.npy")):
                    log.warning(
                        "convert_tflite requested quantization='int8' but representative dataset "
                        "directory '%s' contains no '*.npy' files. Downgrading to 'fp16'.",
                        dataset_path,
                    )
                    representative_dataset = None
                    quantization = "fp16"

        args = [
            "--input",
            current_onnx_input,
            "--output",
            str(output_dir / f"{Path(current_onnx_input).stem}.tflite"),
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
            str(output_dir / f"{Path(current_onnx_input).stem}.mlpackage"),
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
            str(output_dir / f"{Path(current_onnx_input).stem}_mobile.onnx"),
        ]

    elif stage == "benchmark":
        b = config.get("benchmark", {})
        runtime = b.get("runtime", "onnxruntime")
        input_stem = Path(current_onnx_input).stem
        _runtime_model = {
            "tflite": str(output_dir / f"{input_stem}.tflite"),
            "coreml": str(output_dir / f"{input_stem}.mlpackage"),
            "onnxruntime_mobile": str(output_dir / f"{input_stem}_mobile.onnx"),
        }
        benchmark_model = _runtime_model.get(runtime, current_onnx_input)
        result_name = Path(benchmark_model).stem
        args = [
            "--model",
            benchmark_model,
            "--task",
            task,
            "--runtime",
            runtime,
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
        criteria = config.get("success_criteria") or {}
        _known_criteria = {
            "max_size_mb",
            "max_latency_ms",
            "min_accuracy",
            "min_f1",
        }
        unsupported = sorted(set(criteria) - _known_criteria)
        if unsupported:
            log.warning("Ignoring unrecognized success_criteria keys: %s", ", ".join(unsupported))
        if criteria.get("max_size_mb") is not None:
            args += ["--max-size-mb", str(criteria["max_size_mb"])]
        if criteria.get("max_latency_ms") is not None:
            args += ["--max-latency-ms-p95", str(criteria["max_latency_ms"])]
        min_accuracy = criteria.get("min_accuracy")
        min_f1 = criteria.get("min_f1")
        if min_accuracy is not None and min_f1 is not None:
            log.warning(
                "success_criteria defines both min_accuracy and min_f1; using min_accuracy, ignoring min_f1"
            )
        if min_accuracy is None and min_f1 is not None:
            min_accuracy = min_f1
        if min_accuracy is not None:
            args += ["--min-accuracy", str(min_accuracy)]
        return args

    return []


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except OSError as exc:
        log.warning("Could not hash %s: %s — config_hash will be empty", path, exc)
        return ""
    return f"sha256:{h.hexdigest()}"


def _snapshot_dir(d: Path) -> dict[Path, tuple[int, int]]:
    """Return {path: (mtime_ns, size)} for files directly under d or one subdirectory deep.

    Pipeline stages write artifacts at depth 0 (output_dir/*.onnx) or depth 1
    (output_dir/benchmarks/, output_dir/pruned/). Limiting to two levels keeps
    overhead O(artifacts) rather than O(entire tree).
    Using nanosecond mtime and size together catches overwrites on filesystems
    with coarse timestamp resolution (e.g. FAT32, some network mounts).
    """
    if not d.exists():
        return {}
    result: dict[Path, tuple[int, int]] = {}
    for p in d.iterdir():
        if p.is_file():
            s = p.stat()
            result[p] = (s.st_mtime_ns, s.st_size)
        elif p.is_dir():
            for child in p.iterdir():
                if child.is_file():
                    s = child.stat()
                    result[child] = (s.st_mtime_ns, s.st_size)
    return result


def _collect_new_files(output_dir: Path, before: dict[Path, tuple[int, int]]) -> list[dict]:
    """Return files created or overwritten since the before snapshot, excluding manifest.json.

    Scans the same two-level depth as _snapshot_dir so comparisons are consistent.
    """
    candidates: list[Path] = []
    for p in output_dir.iterdir():
        if p.is_file():
            candidates.append(p)
        elif p.is_dir():
            candidates.extend(child for child in p.iterdir() if child.is_file())
    result = []
    for p in sorted(candidates):
        if p.name == "manifest.json":
            continue
        stat = p.stat()
        if p not in before or (stat.st_mtime_ns, stat.st_size) != before[p]:
            size_mb = round(stat.st_size / 1_000_000, 3)
            result.append({"path": str(p.relative_to(output_dir)), "size_mb": size_mb})
    return result


def _init_manifest(manifest_path: Path, config: dict, config_path: Path, stages: list[str]) -> None:
    """Write pipeline-level metadata to the manifest file before stages run.

    If a valid manifest already exists (e.g. from a prior run), unknown top-level
    keys are preserved; only the per-run fields are reset.
    """
    _now = datetime.now(timezone.utc)
    run_id = (
        f"{config.get('task', 'unknown')}_"
        f"{_now.strftime('%Y%m%d_%H%M%S')}_{_now.microsecond // 1000:03d}"
    )
    manifest: dict = {}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text())
            if isinstance(loaded, dict):
                manifest = loaded
            else:
                log.warning(
                    "Manifest %s contained %s JSON, expected object — replacing",
                    manifest_path,
                    type(loaded).__name__,
                )
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read manifest %s: %s — replacing", manifest_path, exc)
    manifest.update(
        {
            "version": "1",
            "pipeline_run_id": run_id,
            "model_id": config.get("model", ""),
            "task": config.get("task", ""),
            "config_path": str(config_path),
            "config_hash": _sha256_file(config_path) if config_path.exists() else "",
            "total_stages": len(stages),
            "stages": [],
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _update_manifest(
    manifest_path: Path,
    stage: str,
    args_list: list[str],
    success: bool,
    exit_code: int = 0,
    artifacts: list[dict] | None = None,
    error: dict | None = None,
) -> None:
    """Append a stage record to the pipeline manifest JSON."""
    record: dict = {
        "stage": stage,
        "args": args_list,
        "status": "ok" if success else "failed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exit_code": exit_code,
        "artifacts": artifacts or [],
        "error": error,
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
        validate_stage_selection(stages, config)
        _validate_gptq_stage_compat(config, stages)
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

    if not args.dry_run:
        _init_manifest(manifest_path, config, args.config, stages)

    for stage in stages:
        stage_args = build_stage_args(stage, config, args.output_dir, state, dry_run=args.dry_run)
        if not stage_args and stage not in ("export",):
            log.warning("Stage '%s' produced no args, skipping", stage)
            continue
        before = _snapshot_dir(args.output_dir) if not args.dry_run else {}
        exit_code, success = run_stage(script_map[stage], stage_args, dry_run=args.dry_run)
        results[stage] = "OK" if success else "FAILED"
        if not args.dry_run:
            artifacts = _collect_new_files(args.output_dir, before)
            error_obj = (
                None
                if success
                else {
                    "message": f"Stage '{stage}' exited with code {exit_code}",
                    "exit_code": exit_code,
                }
            )
            _update_manifest(
                manifest_path,
                stage,
                stage_args,
                success,
                exit_code=exit_code,
                artifacts=artifacts,
                error=error_obj,
            )
        if success:
            update_pipeline_state(stage, state, config, args.output_dir, dry_run=args.dry_run)
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
