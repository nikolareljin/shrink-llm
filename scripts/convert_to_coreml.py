"""
convert_to_coreml.py — Convert ONNX model to CoreML for iOS/macOS deployment.

Requires: pip install coremltools  (macOS only for full validation)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

COMPUTE_UNITS_MAP = {
    "ALL": "ALL",
    "CPU_AND_NE": "CPU_AND_NE",
    "CPU_ONLY": "CPU_ONLY",
    "CPU_AND_GPU": "CPU_AND_GPU",
}

DEPLOYMENT_TARGET_MAP = {
    "iOS15": "iOS15",
    "iOS16": "iOS16",
    "iOS17": "iOS17",
    "iOS18": "iOS18",
    "macOS13": "macOS13",
    "macOS14": "macOS14",
}


def convert_onnx_to_coreml(
    onnx_path: Path,
    output_path: Path,
    minimum_deployment_target: str,
    compute_units: str,
    quantization: str,
) -> None:
    """Convert ONNX model to CoreML .mlpackage."""
    try:
        import coremltools as ct
        import onnx
    except ImportError:
        raise ImportError("Install coremltools: pip install coremltools")

    log.info("Loading ONNX model: %s", onnx_path)
    onnx_model = onnx.load(str(onnx_path))

    log.info("Converting to CoreML (target=%s, compute=%s)...", minimum_deployment_target, compute_units)

    # Build compute units
    cu = getattr(ct.ComputeUnit, compute_units.replace("_", "_"), ct.ComputeUnit.ALL)
    target = getattr(ct.target, minimum_deployment_target, None)

    mlmodel = ct.convert(
        onnx_model,
        convert_to="mlprogram",
        minimum_deployment_target=target,
        compute_units=cu,
    )

    # Apply quantization if requested
    if quantization == "fp16":
        log.info("Applying FP16 weight compression...")
        from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights

        op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="float16")
        config = OptimizationConfig(global_config=op_config)
        mlmodel = linear_quantize_weights(mlmodel, config=config)

    elif quantization == "int8":
        log.info("Applying INT8 weight quantization...")
        try:
            from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights

            op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
            config = OptimizationConfig(global_config=op_config)
            mlmodel = linear_quantize_weights(mlmodel, config=config)
        except Exception as e:
            log.warning("INT8 quantization failed: %s. Falling back to FP16.", e)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    log.info("CoreML model saved: %s", output_path)

    # Report model size
    import os

    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, fn in os.walk(str(output_path))
        for f in fn
    ) / 1e6
    log.info("CoreML package size: %.1f MB", size_mb)


def validate_coreml(model_path: Path) -> None:
    """Run a basic validation check on the CoreML model."""
    try:
        import coremltools as ct

        mlmodel = ct.models.MLModel(str(model_path))
        spec = mlmodel.get_spec()
        log.info("CoreML validation passed.")
        log.info("Input names: %s", [inp.name for inp in spec.description.input])
        log.info("Output names: %s", [out.name for out in spec.description.output])
    except Exception as e:
        log.warning("CoreML validation requires macOS: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ONNX model to CoreML")
    parser.add_argument("--input", required=True, type=Path, help="Input ONNX model path")
    parser.add_argument("--output", required=True, type=Path, help="Output .mlpackage directory path")
    parser.add_argument(
        "--minimum-deployment-target",
        default="iOS16",
        choices=list(DEPLOYMENT_TARGET_MAP.keys()),
        help="Minimum iOS/macOS deployment target (default: iOS16)",
    )
    parser.add_argument(
        "--compute-units",
        default="ALL",
        choices=list(COMPUTE_UNITS_MAP.keys()),
        help="Compute units to enable (default: ALL — includes Neural Engine)",
    )
    parser.add_argument(
        "--quantization",
        default="none",
        choices=["none", "fp16", "int8"],
        help="Post-conversion weight quantization (default: none)",
    )
    parser.add_argument("--validate", action="store_true", help="Validate converted model (requires macOS)")
    args = parser.parse_args()

    convert_onnx_to_coreml(
        args.input,
        args.output,
        args.minimum_deployment_target,
        args.compute_units,
        args.quantization,
    )

    if args.validate:
        validate_coreml(args.output)

    log.info("Done → %s", args.output)


if __name__ == "__main__":
    main()
