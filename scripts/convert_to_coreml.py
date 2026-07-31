"""
convert_to_coreml.py — CoreML conversion for iOS/macOS deployment. CURRENTLY UNAVAILABLE.

Every entry point raises: coremltools removed ONNX as an input format in 6.0, and this project
requires >=7.2, so there is no supported version that can read what the rest of the pipeline
produces. SHRINK-020 tracks restoring it behind a TorchScript or MIL front end;
_convert_via_coremltools below holds the post-conversion path intact for that work.

Requires (once restored): pip install -e ".[coreml]"  (macOS only for full validation)
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
    """Raise, explaining why ONNX cannot be converted to CoreML on a supported coremltools.

    coremltools removed ONNX as an input format in 6.0 and this project requires >=7.2, so
    ct.convert() on a ModelProto fails with a generic "unable to determine the type of the
    model" several frames down. Restoring the capability needs an ONNX->TorchScript or
    ONNX->MIL front end -- tracked as SHRINK-020.

    The version is reported only when coremltools happens to be importable. Importing it first
    would mean anyone without the optional `coreml` extra got "install coremltools" instead of
    this explanation -- sending them to install a package that cannot do the job.
    """
    try:
        import coremltools as ct

        installed = f"coremltools {ct.__version__}"
    except ImportError:
        installed = "coremltools"

    raise NotImplementedError(
        f"Cannot convert '{onnx_path}' to CoreML: {installed} does not accept ONNX input. "
        "ONNX support was removed in coremltools 6.0 and this project requires >=7.2; "
        "ct.convert() accepts only TorchScript, TensorFlow or MIL sources.\n"
        "Until SHRINK-020 lands, drop 'convert_coreml' from --stages or from the config's "
        "stages list. See docs/reviews/codebase-audit.md (CM1)."
    )


def _convert_via_coremltools(  # pragma: no cover - unreachable until SHRINK-020
    source_model,
    output_path: Path,
    minimum_deployment_target: str,
    compute_units: str,
    quantization: str,
) -> None:
    """Convert a coremltools-supported source model and apply weight compression.

    Kept intact so SHRINK-020 only has to supply `source_model`; not reachable today.
    """
    import coremltools as ct

    log.info(
        "Converting to CoreML (target=%s, compute=%s)...", minimum_deployment_target, compute_units
    )

    try:
        cu = getattr(ct.ComputeUnit, compute_units)
    except AttributeError:
        raise ValueError(
            f"Unknown compute units '{compute_units}'. coremltools {ct.__version__} offers: "
            f"{', '.join(u.name for u in ct.ComputeUnit)}"
        ) from None
    # Previously getattr(..., None), which silently discarded the caller's requested minimum
    # and let coremltools pick its own default.
    try:
        target = getattr(ct.target, minimum_deployment_target)
    except AttributeError:
        raise ValueError(
            f"Unknown deployment target '{minimum_deployment_target}' for coremltools "
            f"{ct.__version__}."
        ) from None

    convert_kwargs: dict = {
        "convert_to": "mlprogram",
        "minimum_deployment_target": target,
        "compute_units": cu,
    }
    if quantization == "fp16":
        # FP16 is a convert-time precision, not a linear-quantizer dtype:
        # OpLinearQuantizerConfig accepts only int8/uint8/int4/uint4.
        log.info("Requesting FP16 compute precision...")
        convert_kwargs["compute_precision"] = ct.precision.FLOAT16

    mlmodel = ct.convert(source_model, **convert_kwargs)

    if quantization == "int8":
        log.info("Applying INT8 weight quantization...")
        from coremltools.optimize.coreml import (
            OpLinearQuantizerConfig,
            OptimizationConfig,
            linear_quantize_weights,
        )

        op_config = OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8")
        # Let this raise. The previous handler logged "Falling back to FP16" and then saved the
        # *unquantized* model, reporting success for a model that met none of the size targets.
        mlmodel = linear_quantize_weights(
            mlmodel, config=OptimizationConfig(global_config=op_config)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    log.info("CoreML model saved: %s", output_path)

    # Report model size
    import os

    size_mb = (
        sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, dn, fn in os.walk(str(output_path))
            for f in fn
        )
        / 1e6
    )
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
    parser.add_argument(
        "--output", required=True, type=Path, help="Output .mlpackage directory path"
    )
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
    parser.add_argument(
        "--validate", action="store_true", help="Validate converted model (requires macOS)"
    )
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
