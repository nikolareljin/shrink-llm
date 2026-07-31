"""
convert_to_onnx_mobile.py — Optimize ONNX graph for mobile ONNX Runtime.

Applies graph-level optimizations and layout conversions for Android/iOS deployment.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import onnx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def optimize_graph(
    input_path: Path,
    output_path: Path,
    optimization_level: str,
) -> None:
    """Apply ONNX Runtime graph optimizations for mobile.

    Deliberately platform-agnostic: ORT's graph optimizations are not target-specific, so the
    previous `target` and `enable_nhwc` parameters were accepted and never read. NHWC layout is
    handled by convert_layout_to_nhwc before this runs.
    """
    import onnxruntime as ort

    log.info("Running ONNX graph optimization (level=%s)...", optimization_level)

    level_map = {
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = level_map[optimization_level]
    sess_options.optimized_model_filepath = str(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load model to trigger optimization + save
    _ = ort.InferenceSession(
        str(input_path),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    log.info("Optimized model saved: %s", output_path)
    _log_size_comparison(input_path, output_path)


def convert_layout_to_nhwc(input_path: Path, output_path: Path) -> None:
    """Convert NCHW convolution layout to NHWC for mobile NPU efficiency."""
    try:
        from onnxruntime.tools import transpose_optimizer

        log.info("Applying NHWC layout conversion...")
        onnx_model = onnx.load(str(input_path))
        optimized = transpose_optimizer.optimize_model(onnx_model)
        onnx.save(optimized, str(output_path))
        log.info("NHWC layout applied → %s", output_path)
    except ImportError:
        log.warning(
            "onnxruntime.tools.transpose_optimizer is not available in onnxruntime — the model "
            "was copied through UNCHANGED and is still NCHW. --enable-nhwc had no effect."
        )
        import shutil

        shutil.copy(str(input_path), str(output_path))


def downgrade_opset(input_path: Path, output_path: Path, target_opset: int = 13) -> None:
    """Downgrade ONNX opset for compatibility with older mobile ORT versions."""
    log.info("Downgrading opset to %d...", target_opset)
    model = onnx.load(str(input_path))
    downgraded = onnx.version_converter.convert_version(model, target_opset)
    onnx.save(downgraded, str(output_path))
    log.info("Opset downgraded model saved → %s", output_path)


def generate_ort_model(input_path: Path, output_path: Path) -> None:
    """Generate .ort format (ORT flatbuffers) for faster mobile loading."""
    import shutil
    import tempfile

    from onnxruntime.tools import convert_onnx_models_to_ort

    log.info("Generating .ort flatbuffers format...")
    # Stage the model alone in a temp directory first. convert_onnx_models_to_ort converts
    # every .onnx under the path it is given, so passing input_path.parent silently converted
    # the base export, the quantized model and the mobile model as a side effect of asking
    # for one.
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / input_path.name
        shutil.copy(str(input_path), str(staged))
        convert_onnx_models_to_ort.convert_onnx_models_to_ort(staged)

        produced = sorted(Path(tmp).glob("**/*.ort"))
        if not produced:
            raise RuntimeError(f"convert_onnx_models_to_ort produced no .ort file for {input_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(produced[0]), str(output_path))
    log.info(".ort model → %s", output_path)


def _log_size_comparison(before: Path, after: Path) -> None:
    if not after.exists():
        return
    before_mb = before.stat().st_size / 1e6
    after_mb = after.stat().st_size / 1e6
    delta = after_mb - before_mb
    log.info("Size: %.1f MB → %.1f MB (%+.1f MB)", before_mb, after_mb, delta)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize ONNX model for mobile ONNX Runtime")
    parser.add_argument("--input", required=True, type=Path, help="Input ONNX model path")
    parser.add_argument(
        "--output", required=True, type=Path, help="Output optimized ONNX model path"
    )
    parser.add_argument(
        "--optimization-level",
        default="extended",
        choices=["basic", "extended", "all"],
        help="ORT graph optimization level (default: extended)",
    )
    parser.add_argument(
        "--enable-nhwc",
        action="store_true",
        help="Convert NCHW → NHWC for mobile NPU efficiency",
    )
    parser.add_argument(
        "--downgrade-opset",
        type=int,
        help="Downgrade ONNX opset to this version (e.g. 13 for older ORT builds)",
    )
    parser.add_argument(
        "--generate-ort",
        action="store_true",
        help="Also generate .ort flatbuffers format for faster loading",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    current_input = args.input

    if args.enable_nhwc:
        nhwc_path = args.output.with_stem(args.output.stem + "_nhwc")
        convert_layout_to_nhwc(current_input, nhwc_path)
        current_input = nhwc_path

    if args.downgrade_opset:
        opset_path = args.output.with_stem(args.output.stem + f"_opset{args.downgrade_opset}")
        downgrade_opset(current_input, opset_path, args.downgrade_opset)
        current_input = opset_path

    optimize_graph(current_input, args.output, args.optimization_level)

    if args.generate_ort:
        ort_output = args.output.with_suffix(".ort")
        generate_ort_model(args.output, ort_output)

    log.info("Done → %s", args.output)


if __name__ == "__main__":
    main()
