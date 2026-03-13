"""
convert_to_tflite.py — Convert ONNX model to TFLite for Android deployment.

Requires: pip install tensorflow onnx-tf  (or onnx2tf for newer models)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def convert_onnx_to_tf(onnx_path: Path, tf_saved_model_dir: Path) -> None:
    """Convert ONNX to TensorFlow SavedModel using onnx2tf."""
    try:
        import onnx2tf

        log.info("Converting ONNX → TF SavedModel via onnx2tf...")
        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(tf_saved_model_dir),
            non_verbose=False,
        )
    except ImportError:
        log.warning("onnx2tf not installed. Trying onnx-tf...")
        try:
            import onnx
            from onnx_tf.backend import prepare

            model = onnx.load(str(onnx_path))
            tf_rep = prepare(model)
            tf_rep.export_graph(str(tf_saved_model_dir))
        except ImportError:
            raise ImportError("Install onnx2tf: pip install onnx2tf")

    log.info("TF SavedModel saved to %s", tf_saved_model_dir)


def convert_tf_to_tflite(
    tf_saved_model_dir: Path,
    output_path: Path,
    quantization: str,
    representative_dataset_dir: Path | None,
    optimize_for: str,
) -> None:
    """Convert TF SavedModel to TFLite."""
    import tensorflow as tf

    log.info("Converting TF SavedModel → TFLite (quantization=%s)...", quantization)
    converter = tf.lite.TFLiteConverter.from_saved_model(str(tf_saved_model_dir))

    if optimize_for == "latency":
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    elif optimize_for == "size":
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    else:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantization == "fp16":
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "int8":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        if representative_dataset_dir:
            import numpy as np

            def representative_dataset_gen():
                files = list(representative_dataset_dir.glob("*.npy"))[:200]
                for f in files:
                    data = np.load(f)
                    yield [data.astype(np.float32)]

            converter.representative_dataset = representative_dataset_gen

    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)

    size_mb = len(tflite_model) / 1e6
    log.info("TFLite model saved: %s (%.1f MB)", output_path, size_mb)


def validate_tflite(tflite_path: Path) -> None:
    """Run a sample inference to validate the TFLite model."""
    import numpy as np
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for detail in input_details:
        shape = detail["shape"]
        dtype = detail["dtype"]
        interpreter.set_tensor(detail["index"], np.zeros(shape, dtype=dtype))

    interpreter.invoke()
    outputs = [interpreter.get_tensor(d["index"]) for d in output_details]
    log.info("TFLite validation passed. Output shapes: %s", [o.shape for o in outputs])


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ONNX model to TFLite")
    parser.add_argument("--input", required=True, type=Path, help="Input ONNX model path")
    parser.add_argument("--output", required=True, type=Path, help="Output .tflite file path")
    parser.add_argument(
        "--quantization",
        default="none",
        choices=["none", "fp16", "int8"],
        help="TFLite quantization mode (default: none)",
    )
    parser.add_argument("--representative-dataset", type=Path, help="Directory with .npy files for full-int8 calibration")
    parser.add_argument(
        "--optimize-for",
        default="latency",
        choices=["latency", "size", "default"],
        help="Optimization target (default: latency)",
    )
    parser.add_argument("--skip-tf-conversion", action="store_true", help="Skip ONNX→TF step (use existing SavedModel)")
    parser.add_argument("--tf-saved-model-dir", type=Path, help="Path for intermediate TF SavedModel")
    parser.add_argument("--validate", action="store_true", help="Run validation after conversion")
    args = parser.parse_args()

    tf_dir = args.tf_saved_model_dir or args.input.parent / f"{args.input.stem}_tf_saved_model"

    if not args.skip_tf_conversion:
        convert_onnx_to_tf(args.input, tf_dir)

    convert_tf_to_tflite(tf_dir, args.output, args.quantization, args.representative_dataset, args.optimize_for)

    if args.validate:
        validate_tflite(args.output)

    log.info("Done → %s", args.output)


if __name__ == "__main__":
    main()
