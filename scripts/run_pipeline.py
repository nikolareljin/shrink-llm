"""
run_pipeline.py — Orchestrate the full ShrinkLLM compression pipeline from a YAML config.

Runs stages: export → quantize → prune → distill → convert → benchmark
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

VALID_STAGES = ["export", "quantize", "prune", "distill", "convert_tflite", "convert_coreml", "convert_onnx_mobile", "benchmark"]


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
    onnx_path = str(output_dir / f"{Path(model_id).name}_base.onnx")
    quant_path = str(output_dir / f"{Path(model_id).name}_int8.onnx")

    if stage == "export":
        return [
            "--model", model_id,
            "--task", task,
            "--output", onnx_path,
            "--opset", str(config.get("onnx_opset", 17)),
            "--validate",
        ]
    elif stage == "quantize":
        q = config.get("quantization", {})
        args = [
            "--input", onnx_path,
            "--output", quant_path,
            "--precision", q.get("precision", "int8"),
            "--mode", q.get("mode", "dynamic"),
        ]
        if q.get("calibration_data"):
            args += ["--calibration-data", q["calibration_data"]]
        return args
    elif stage == "benchmark":
        b = config.get("benchmark", {})
        result_name = f"{Path(model_id).name}_int8"
        return [
            "--model", quant_path,
            "--task", task,
            "--runtime", b.get("runtime", "onnxruntime"),
            "--warmup-runs", str(b.get("warmup_runs", 10)),
            "--benchmark-runs", str(b.get("benchmark_runs", 100)),
            "--output-json", str(output_dir / "benchmarks" / f"{result_name}.json"),
            "--output-md", str(output_dir / "benchmarks" / f"{result_name}.md"),
        ]
    else:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full ShrinkLLM compression pipeline")
    parser.add_argument("--config", required=True, type=Path, help="Pipeline YAML config file")
    parser.add_argument(
        "--stages",
        default=",".join(VALID_STAGES),
        help=f"Comma-separated stages to run (default: all). Options: {', '.join(VALID_STAGES)}",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/student"), help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    config = load_config(args.config)
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

    results = {}
    for stage in stages:
        if stage not in VALID_STAGES:
            log.warning("Unknown stage '%s' — skipping", stage)
            continue
        stage_args = build_stage_args(stage, config, args.output_dir)
        success = run_stage(script_map[stage], stage_args, dry_run=args.dry_run)
        results[stage] = "OK" if success else "FAILED"
        if not success:
            log.error("Pipeline aborted at stage '%s'", stage)
            break

    log.info("\nPipeline summary:")
    for stage, status in results.items():
        symbol = "✓" if status == "OK" else "✗"
        log.info("  %s %s: %s", symbol, stage, status)


if __name__ == "__main__":
    main()
