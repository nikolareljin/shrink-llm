# Benchmarking Guide

`scripts/benchmark.py` measures latency, memory and model size, and evaluates the thresholds a
pipeline config declares under `success_criteria`.

## Where results are written

`run_pipeline.py` writes a benchmark's JSON and Markdown reports to a `benchmarks/` directory
**inside the run's output directory** — `models/student/benchmarks/` by default, or
`<--output-dir>/benchmarks/`. They sit beside `manifest.json`, which records every stage of the
run that produced them.

The top-level `benchmarks/results/` directory is for results curated and checked in for
comparison across runs; nothing writes there automatically.

Running `scripts/benchmark.py` directly writes wherever `--output-json` and `--output-md` point.

## Reported metrics

| Group | Fields | Notes |
|---|---|---|
| Size | `model_size_mb` | Directory artifacts (GPTQ, `.mlpackage`) are summed recursively |
| Latency | `mean`, `p50`, `p95`, `p99`, `min`, `max` | Milliseconds, over `--benchmark-runs` after `--warmup-runs` |
| Memory | `rss_baseline`, `rss_after_load`, `model_load_delta`, `peak_rss` | Linux only; 0.0 elsewhere |
| Accuracy | task-dependent | **Not yet computed** — see below |

`peak_rss` reads `VmHWM`, the process high-water mark. The other three read `VmRSS`, sampled once
before the model is loaded and again immediately after, so `model_load_delta` isolates the cost of
loading the model rather than of running inference.

## Gates

`success_criteria` in a pipeline config translate into gate flags on the result. A gate whose data
cannot be computed **fails** rather than passing silently, and a criterion with no implementation
behind it logs a warning naming the threshold that will not be enforced.

| Criterion | Status |
|---|---|
| `max_size_mb` | Enforced |
| `max_latency_ms` | Enforced, against p95 |
| `min_accuracy` | Passed through to `benchmark.py`, but skipped until accuracy evaluation exists |
| `min_f1`, `max_cer`, `max_accuracy_drop_pct` | Recognised, not enforced; warns |

Accuracy evaluation is `SHRINK-018`. Until it lands, `BenchmarkResult.accuracy` is empty and any
accuracy-family gate reports that it was not evaluated rather than reporting a pass.
