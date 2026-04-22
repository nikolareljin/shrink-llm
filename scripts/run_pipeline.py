"""
run_pipeline.py — Orchestrate the full ShrinkLLM compression pipeline from a YAML config.

Runs stages: export → quantize → prune → distill → convert → benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

VALID_STAGES = [
    "export",
    "quantize",
    "prune",
    "distill",
    "convert_tflite",
    "convert_coreml",
    "convert_onnx_mobile",
    "benchmark",
]


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


def build_stage_args(stage: str, config: dict, output_dir: Path) -> list[str]:
    """Build CLI args for each stage from config."""
    model_id = config.get("model", "")
    task = config.get("task", "")
    model_stem = Path(model_id).name if model_id else "model"
    onnx_path = str(output_dir / f"{model_stem}_base.onnx")
    quant_path = str(output_dir / f"{model_stem}_int8.onnx")

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
        args = [
            "--input",
            onnx_path,
            "--output",
            quant_path,
            "--precision",
            q.get("precision", "int8"),
            "--mode",
            q.get("mode", "dynamic"),
        ]
        if q.get("calibration_data"):
            args += ["--calibration-data", q["calibration_data"]]
        if q.get("calibration_samples"):
            args += ["--calibration-samples", str(q["calibration_samples"])]
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
            log.warning("Distill stage requested but 'teacher' not set in config — skipping")
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
            "--epochs",
            str(d.get("epochs", 10)),
            "--batch-size",
            str(d.get("batch_size", 16)),
            "--lr",
            str(d.get("lr", 5e-5)),
        ]
        if d.get("fp16"):
            args.append("--fp16")
        return args

    elif stage == "convert_tflite":
        m = config.get("mobile", {}).get("android", {})
        return [
            "--input",
            quant_path,
            "--output",
            str(output_dir / f"{model_stem}.tflite"),
            "--quantization",
            m.get("quantization", "int8"),
        ]

    elif stage == "convert_coreml":
        m = config.get("mobile", {}).get("ios", {})
        return [
            "--input",
            quant_path,
            "--output",
            str(output_dir / f"{model_stem}.mlpackage"),
            "--minimum-deployment-target",
            m.get("deployment_target", "iOS16"),
            "--compute-units",
            m.get("compute_units", "ALL"),
        ]

    elif stage == "convert_onnx_mobile":
        return [
            "--input",
            quant_path,
            "--output",
            str(output_dir / f"{model_stem}_mobile.onnx"),
        ]

    elif stage == "benchmark":
        b = config.get("benchmark", {})
        result_name = f"{model_stem}_int8"
        args = [
            "--model",
            quant_path,
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
            existing = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not read manifest %s: %s — starting fresh", manifest_path, exc)
    existing.setdefault("stages", []).append(record)
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

    if not config.get("model"):
        parser.error("Config must specify a non-empty 'model' field")
    if not config.get("task"):
        parser.error("Config must specify a non-empty 'task' field")

    stages = [s.strip() for s in args.stages.split(",")]
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
    results = {}
    for stage in stages:
        if stage not in VALID_STAGES:
            log.warning("Unknown stage '%s' — skipping", stage)
            continue
        stage_args = build_stage_args(stage, config, args.output_dir)
        if not stage_args and stage not in ("export",):
            log.warning("Stage '%s' produced no args — skipping", stage)
            continue
        success = run_stage(script_map[stage], stage_args, dry_run=args.dry_run)
        results[stage] = "OK" if success else "FAILED"
        if not args.dry_run:
            _update_manifest(manifest_path, stage, stage_args, success)
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
