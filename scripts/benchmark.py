"""
benchmark.py — Measure accuracy, latency, memory, and model size.

Outputs results as JSON and Markdown. Supports ONNX Runtime, TFLite, and CoreML runtimes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, quantiles

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    run_id: str
    timestamp: str
    model_name: str
    model_path: str
    model_size_mb: float
    task: str
    runtime: str
    accuracy: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)
    memory_mb: dict = field(default_factory=dict)
    size_reduction_pct: float | None = None
    teacher_name: str | None = None
    teacher_size_mb: float | None = None
    gate_results: dict = field(default_factory=dict)
    passed: bool = True


class LatencyProfiler:
    def __init__(self, warmup_runs: int = 10, benchmark_runs: int = 100):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

    def profile(self, inference_fn, inputs: dict) -> dict:
        log.info("Warming up (%d runs)...", self.warmup_runs)
        for _ in range(self.warmup_runs):
            inference_fn(inputs)

        log.info("Benchmarking (%d runs)...", self.benchmark_runs)
        times = []
        for _ in range(self.benchmark_runs):
            t0 = time.perf_counter()
            inference_fn(inputs)
            times.append((time.perf_counter() - t0) * 1000)

        qs = quantiles(times, n=100)
        return {
            "mean": round(mean(times), 2),
            "p50": round(qs[49], 2),
            "p95": round(qs[94], 2),
            "p99": round(qs[98], 2),
            "min": round(min(times), 2),
            "max": round(max(times), 2),
        }


class MemoryProfiler:
    @staticmethod
    def peak_rss_mb() -> float:
        """Read peak RSS from /proc/self/status on Linux."""
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except Exception:
            pass
        return 0.0


class ONNXRuntimeRunner:
    def __init__(self, model_path: str):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self.sess.get_inputs()]

    def __call__(self, inputs: dict) -> list:
        ort_inputs = {k: v for k, v in inputs.items() if k in self.input_names}
        return self.sess.run(None, ort_inputs)


class TFLiteRunner:
    def __init__(self, model_path: str):
        import tensorflow as tf

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, inputs: dict) -> list:
        for detail in self.input_details:
            key = detail["name"].split(":")[0]
            if key in inputs:
                self.interpreter.set_tensor(detail["index"], inputs[key].astype(np.float32))
        self.interpreter.invoke()
        return [self.interpreter.get_tensor(d["index"]) for d in self.output_details]


def get_runner(model_path: str, runtime: str):
    if runtime in ("onnxruntime", "onnxruntime_mobile"):
        return ONNXRuntimeRunner(model_path)
    elif runtime == "tflite":
        return TFLiteRunner(model_path)
    elif runtime == "coreml":
        raise NotImplementedError("CoreML benchmarking must run on macOS/iOS")
    else:
        raise ValueError(f"Unknown runtime: {runtime}")


def build_dummy_inputs(task: str) -> dict:
    if task == "ocr":
        return {"pixel_values": np.random.randn(1, 3, 384, 384).astype(np.float32)}
    elif task == "legal":
        return {
            "input_ids": np.random.randint(0, 1000, (1, 128)).astype(np.int64),
            "attention_mask": np.ones((1, 128), dtype=np.int64),
        }
    elif task == "audio":
        return {"input_values": np.random.randn(1, 16000).astype(np.float32)}
    else:
        raise ValueError(f"Unknown task: {task}")


def generate_markdown(result: BenchmarkResult, output_path: Path) -> None:
    lines = [
        f"# Benchmark: {result.model_name}",
        f"**Run ID**: `{result.run_id}`  ",
        f"**Timestamp**: {result.timestamp}  ",
        f"**Task**: {result.task}  ",
        f"**Runtime**: {result.runtime}",
        "",
        "## Model",
        "| Property | Value |",
        "|---|---|",
        f"| Name | `{result.model_name}` |",
        f"| Size | {result.model_size_mb:.1f} MB |",
    ]
    if result.teacher_size_mb:
        lines.append(f"| Teacher Size | {result.teacher_size_mb:.1f} MB |")
    if result.size_reduction_pct:
        lines.append(f"| Size Reduction | {result.size_reduction_pct:.1f}% |")

    if result.accuracy:
        lines += ["", "## Accuracy"]
        lines += ["| Metric | Value |", "|---|---|"]
        for k, v in result.accuracy.items():
            lines.append(f"| {k} | {v} |")

    if result.latency_ms:
        lines += ["", "## Latency (ms)"]
        lines += ["| Metric | Value |", "|---|---|"]
        for k, v in result.latency_ms.items():
            lines.append(f"| {k} | {v} |")

    if result.memory_mb:
        lines += ["", "## Memory (MB)"]
        lines += ["| Metric | Value |", "|---|---|"]
        for k, v in result.memory_mb.items():
            lines.append(f"| {k} | {v} |")

    output_path.write_text("\n".join(lines))
    log.info("Markdown report → %s", output_path)


def evaluate_gates(result: BenchmarkResult, args: argparse.Namespace) -> None:
    """Check benchmark results against acceptance thresholds; update result in place."""
    gates: dict[str, bool] = {}
    if args.max_size_mb is not None:
        gates["max_size_mb"] = result.model_size_mb <= args.max_size_mb
    if args.max_latency_ms_p95 is not None:
        gates["max_latency_ms_p95"] = result.latency_ms.get("p95", 0.0) <= args.max_latency_ms_p95
    if args.min_accuracy is not None and result.accuracy:
        acc = result.accuracy.get("accuracy") or result.accuracy.get("f1") or 0.0
        gates["min_accuracy"] = float(acc) >= args.min_accuracy
    result.gate_results = gates
    result.passed = all(gates.values()) if gates else True

    if gates:
        print(f"\n{'='*50}")
        print("  BENCHMARK GATES")
        print(f"{'='*50}")
        for criterion, ok in gates.items():
            symbol = "PASS" if ok else "FAIL"
            print(f"  [{symbol}] {criterion}")
        overall = "ALL PASSED" if result.passed else "FAILED"
        print(f"  Overall: {overall}")
        print(f"{'='*50}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a compressed model")
    parser.add_argument(
        "--model", required=True, help="Model file path (.onnx, .tflite, .mlpackage)"
    )
    parser.add_argument("--task", required=True, choices=["ocr", "legal", "audio"])
    parser.add_argument(
        "--runtime",
        default="onnxruntime",
        choices=["onnxruntime", "onnxruntime_mobile", "tflite", "coreml"],
    )
    parser.add_argument("--dataset", type=Path, help="Evaluation dataset directory")
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--benchmark-runs", type=int, default=100)
    parser.add_argument("--output-json", type=Path, help="JSON output path")
    parser.add_argument("--output-md", type=Path, help="Markdown output path")
    parser.add_argument("--teacher-size-mb", type=float, help="Teacher model size for comparison")
    parser.add_argument("--teacher-name", help="Teacher model name for report")
    parser.add_argument(
        "--max-size-mb", type=float, default=None, help="Gate: fail if model exceeds this size"
    )
    parser.add_argument(
        "--max-latency-ms-p95",
        type=float,
        default=None,
        help="Gate: fail if p95 latency exceeds this",
    )
    parser.add_argument(
        "--min-accuracy", type=float, default=None, help="Gate: fail if accuracy falls below this"
    )
    args = parser.parse_args()

    model_path = args.model
    model_size_mb = os.path.getsize(model_path) / 1e6
    run_id = f"{args.task}_{Path(model_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    log.info("Model: %s (%.1f MB)", model_path, model_size_mb)

    runner = get_runner(model_path, args.runtime)
    dummy_inputs = build_dummy_inputs(args.task)

    profiler = LatencyProfiler(warmup_runs=args.warmup_runs, benchmark_runs=args.benchmark_runs)
    latency = profiler.profile(runner, dummy_inputs)
    log.info("Latency: mean=%.1f ms, p95=%.1f ms", latency["mean"], latency["p95"])

    mem_before = MemoryProfiler.peak_rss_mb()
    runner(dummy_inputs)
    mem_after = MemoryProfiler.peak_rss_mb()

    size_reduction = None
    if args.teacher_size_mb:
        size_reduction = round((1 - model_size_mb / args.teacher_size_mb) * 100, 1)

    result = BenchmarkResult(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_name=Path(model_path).stem,
        model_path=model_path,
        model_size_mb=round(model_size_mb, 2),
        task=args.task,
        runtime=args.runtime,
        accuracy={},  # populate from eval_dataset if provided
        latency_ms=latency,
        memory_mb={
            "rss_after_load": round(mem_after, 1),
            "rss_delta": round(mem_after - mem_before, 1),
        },
        size_reduction_pct=size_reduction,
        teacher_name=args.teacher_name,
        teacher_size_mb=args.teacher_size_mb,
    )

    evaluate_gates(result, args)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(asdict(result), indent=2))
        log.info("JSON report → %s", args.output_json)

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        generate_markdown(result, args.output_md)

    # Print summary
    print(f"\n{'='*50}")
    print(f"  BENCHMARK SUMMARY: {result.model_name}")
    print(f"{'='*50}")
    print(f"  Size:       {result.model_size_mb:.1f} MB", end="")
    if size_reduction:
        print(f" ({size_reduction:.1f}% reduction)", end="")
    print()
    print(f"  Latency:    mean={latency['mean']:.1f} ms | p95={latency['p95']:.1f} ms")
    print(f"{'='*50}\n")

    if not result.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
