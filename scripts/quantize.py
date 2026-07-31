"""
quantize.py — Quantize ONNX models to INT8 or INT4.

Supports dynamic, static (with calibration), and GPTQ quantization modes.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Generator
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Ops that are sensitive to quantization and should stay in FP32
DEFAULT_SKIP_OPS = {"Softmax", "LayerNormalization", "Gelu"}


_ONNX_TO_NUMPY_DTYPE = {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.FLOAT16: np.float16,
    onnx.TensorProto.DOUBLE: np.float64,
    onnx.TensorProto.INT8: np.int8,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT64: np.int64,
    onnx.TensorProto.UINT8: np.uint8,
    onnx.TensorProto.BOOL: np.bool_,
}


def model_input_specs(model: onnx.ModelProto) -> dict[str, tuple[tuple[int, ...], type]]:
    """Map each graph input to its (shape, numpy dtype).

    Dynamic axes carry no dimension value in the graph; they are resolved to 1 for batch-like
    leading axes and to a small concrete size elsewhere, which is all calibration needs.
    """
    specs: dict[str, tuple[tuple[int, ...], type]] = {}
    initializers = {init.name for init in model.graph.initializer}
    for inp in model.graph.input:
        if inp.name in initializers:  # initializers appear as graph inputs in older opsets
            continue
        tensor_type = inp.type.tensor_type
        dtype = _ONNX_TO_NUMPY_DTYPE.get(tensor_type.elem_type, np.float32)
        shape: list[int] = []
        for axis, dim in enumerate(tensor_type.shape.dim):
            if dim.HasField("dim_value") and dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(1 if axis == 0 else 128)
        specs[inp.name] = (tuple(shape), dtype)
    return specs


class SimpleCalibrationDataReader(CalibrationDataReader):
    """Loads calibration samples from a directory of .npz files.

    Each .npz maps graph input names to arrays, so one file is one calibration sample. .npz is
    used rather than .npy because a dict of named arrays is exactly what a multi-input model
    needs, and because it loads with allow_pickle=False — calibration directories come from
    config values and are not necessarily authored by whoever runs the pipeline, so loading one
    must not be able to execute code.
    """

    def __init__(
        self,
        data_dir: Path,
        input_names: list[str],
        max_samples: int = 512,
        input_specs: dict[str, tuple[tuple[int, ...], type]] | None = None,
    ):
        self.data_dir = data_dir
        self.input_names = input_names
        self.max_samples = max_samples
        self.input_specs = input_specs or {}
        self._iter = self._load()

    def _synthetic_sample(self) -> dict[str, np.ndarray]:
        """Build one sample matching the model's declared input shapes and dtypes.

        Guessing float32 [1, 128] for every input hands onnxruntime the wrong rank for
        pixel_values and the wrong dtype for input_ids.
        """
        sample: dict[str, np.ndarray] = {}
        for name in self.input_names:
            shape, dtype = self.input_specs.get(name, ((1, 128), np.float32))
            if np.issubdtype(dtype, np.floating):
                sample[name] = np.random.randn(*shape).astype(dtype)
            elif dtype is np.bool_:
                sample[name] = np.ones(shape, dtype=dtype)
            else:
                sample[name] = np.ones(shape, dtype=dtype)
        return sample

    def _load(self) -> Generator[dict[str, np.ndarray], None, None]:
        files = sorted(self.data_dir.glob("*.npz"))[: self.max_samples]
        if not files:
            legacy = sorted(self.data_dir.glob("*.npy"))
            if legacy:
                log.warning(
                    "%s contains %d .npy file(s) but no .npz — calibration data must be .npz "
                    "(a mapping of input name to array). Using synthetic data instead.",
                    self.data_dir,
                    len(legacy),
                )
            else:
                log.warning(
                    "No .npz files found in %s — using synthetic calibration data derived from "
                    "the model's input signature. Static quantization accuracy will suffer.",
                    self.data_dir,
                )
            for _ in range(min(64, self.max_samples)):
                yield self._synthetic_sample()
            return
        for f in files:
            with np.load(f, allow_pickle=False) as data:
                yield {k: data[k] for k in data.files if k in self.input_names}

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            return next(self._iter)
        except StopIteration:
            return None


def nodes_with_op_types(model: onnx.ModelProto, op_types: set[str]) -> list[str]:
    """Resolve op types to the names of nodes carrying them.

    onnxruntime's nodes_to_exclude takes node *names*; passing op types (Softmax,
    LayerNormalization, ...) matches nothing and silently quantizes every sensitive op.
    """
    if not op_types:
        return []
    names = [node.name for node in model.graph.node if node.op_type in op_types and node.name]
    unmatched = op_types - {node.op_type for node in model.graph.node}
    if unmatched:
        log.info("No nodes of type(s) %s in this graph", ", ".join(sorted(unmatched)))
    log.info("Excluding %d node(s) of type(s) %s", len(names), ", ".join(sorted(op_types)))
    return names


def dynamic_quantize(input_path: Path, output_path: Path, skip_ops: set[str]) -> None:
    """Dynamic INT8 quantization — no calibration required."""
    log.info("Applying dynamic INT8 quantization...")
    model = onnx.load(str(input_path))
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        nodes_to_exclude=nodes_with_op_types(model, skip_ops),
    )
    _log_size_comparison(input_path, output_path)


def static_quantize(
    input_path: Path,
    output_path: Path,
    calibration_data_dir: Path,
    calibration_samples: int,
    skip_ops: set[str],
) -> None:
    """Static INT8 quantization with calibration dataset."""
    log.info("Applying static INT8 quantization with calibration from %s...", calibration_data_dir)

    # Determine input names from model
    model = onnx.load(str(input_path))
    input_specs = model_input_specs(model)
    input_names = list(input_specs)

    reader = SimpleCalibrationDataReader(
        calibration_data_dir, input_names, calibration_samples, input_specs=input_specs
    )

    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=nodes_with_op_types(model, skip_ops),
    )
    _log_size_comparison(input_path, output_path)


def fp16_quantize(input_path: Path, output_path: Path) -> None:
    """Convert model weights to FP16."""
    try:
        from onnxconverter_common import float16
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "FP16 conversion needs onnxconverter-common: pip install onnxconverter-common"
        ) from exc

    log.info("Converting to FP16...")
    model = onnx.load(str(input_path))
    fp16_model = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(fp16_model, str(output_path))
    _log_size_comparison(input_path, output_path)


def gptq_quantize(model_id: str, output_dir: Path, bits: int = 4, group_size: int = 128) -> None:
    """GPTQ 4-bit quantization for LLMs (requires auto-gptq)."""
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Install auto-gptq: pip install auto-gptq")

    log.info("Applying GPTQ %d-bit quantization to %s...", bits, model_id)
    quantize_config = BaseQuantizeConfig(bits=bits, group_size=group_size, desc_act=False)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config)

    # Minimal calibration dataset
    calibration_dataset = [
        tokenizer(
            "Legal document analysis requires careful reading of clauses.", return_tensors="pt"
        )
        for _ in range(128)
    ]
    model.quantize(calibration_dataset)
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    log.info("GPTQ model saved to %s", output_dir)


def _log_size_comparison(before: Path, after: Path) -> None:
    before_mb = before.stat().st_size / 1e6
    after_mb = after.stat().st_size / 1e6
    if before_mb == 0:
        log.info("Size: 0.0 MB → %.1f MB (input was empty; no ratio to report)", after_mb)
        return
    reduction = (1 - after_mb / before_mb) * 100
    log.info(
        "Size: %.1f MB → %.1f MB (%.1f%% reduction)",
        before_mb,
        after_mb,
        reduction,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize an ONNX model")
    parser.add_argument(
        "--input", type=Path, help="Input ONNX model path (required for non-GPTQ modes)"
    )
    parser.add_argument("--output", required=True, type=Path, help="Output quantized model path")
    parser.add_argument(
        "--precision",
        default="int8",
        choices=["int8", "int4", "fp16"],
        help="Target precision (default: int8)",
    )
    parser.add_argument(
        "--mode",
        default="dynamic",
        choices=["dynamic", "static", "gptq"],
        help="Quantization mode (default: dynamic)",
    )
    parser.add_argument(
        "--calibration-data",
        type=Path,
        help="Directory of .npz calibration samples, each mapping input name to array "
        "(required for static)",
    )
    parser.add_argument(
        "--calibration-samples", type=int, default=512, help="Number of calibration samples"
    )
    parser.add_argument(
        "--skip-ops",
        default=",".join(DEFAULT_SKIP_OPS),
        help=f"Comma-separated op types to keep in FP32 (default: {','.join(DEFAULT_SKIP_OPS)})",
    )
    parser.add_argument("--model-id", help="HuggingFace model ID for GPTQ mode")
    parser.add_argument(
        "--group-size", type=int, default=128, help="GPTQ group size (default: 128)"
    )
    args = parser.parse_args()

    _valid_precisions = {"dynamic": {"int8", "fp16"}, "static": {"int8"}, "gptq": {"int4", "int8"}}
    allowed = _valid_precisions.get(args.mode, {"int8", "fp16"})
    if args.precision not in allowed:
        parser.error(
            f"--precision '{args.precision}' is not supported for mode '{args.mode}'. "
            f"Allowed: {sorted(allowed)}"
        )

    if args.mode == "gptq":
        args.output.mkdir(parents=True, exist_ok=True)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    skip_ops = set(args.skip_ops.split(",")) if args.skip_ops else set()

    if args.mode == "gptq":
        if not args.model_id:
            parser.error("--model-id is required for GPTQ mode")
        gptq_bits = {"int4": 4, "int8": 8}
        gptq_quantize(
            args.model_id,
            args.output,
            bits=gptq_bits[args.precision],
            group_size=args.group_size,
        )
    elif not args.input:
        parser.error("--input is required for non-GPTQ modes")
    elif args.precision == "fp16":
        fp16_quantize(args.input, args.output)
    elif args.mode == "static":
        if not args.calibration_data:
            parser.error("--calibration-data is required for static quantization")
        static_quantize(
            args.input, args.output, args.calibration_data, args.calibration_samples, skip_ops
        )
    else:
        dynamic_quantize(args.input, args.output, skip_ops)

    log.info("Quantization complete → %s", args.output)


if __name__ == "__main__":
    main()
