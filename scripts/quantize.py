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


class SimpleCalibrationDataReader(CalibrationDataReader):
    """Loads numpy arrays from a directory as calibration data."""

    def __init__(self, data_dir: Path, input_names: list[str], max_samples: int = 512):
        self.data_dir = data_dir
        self.input_names = input_names
        self.max_samples = max_samples
        self._iter = self._load()

    def _load(self) -> Generator[dict[str, np.ndarray], None, None]:
        files = sorted(self.data_dir.glob("*.npy"))[: self.max_samples]
        if not files:
            # Fall back to synthetic calibration data
            log.warning(
                "No .npy files found in %s — using synthetic calibration data", self.data_dir
            )
            for _ in range(min(64, self.max_samples)):
                yield {
                    name: np.random.randn(1, 128).astype(np.float32) for name in self.input_names
                }
            return
        for f in files:
            data = np.load(f, allow_pickle=True).item()
            yield {k: v for k, v in data.items() if k in self.input_names}

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            return next(self._iter)
        except StopIteration:
            return None


def dynamic_quantize(input_path: Path, output_path: Path, skip_ops: set[str]) -> None:
    """Dynamic INT8 quantization — no calibration required."""
    log.info("Applying dynamic INT8 quantization...")
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        nodes_to_exclude=list(skip_ops),
        optimize_model=True,
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
    input_names = [inp.name for inp in model.graph.input]

    reader = SimpleCalibrationDataReader(calibration_data_dir, input_names, calibration_samples)

    quantize_static(
        model_input=str(input_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=list(skip_ops),
        optimize_model=True,
    )
    _log_size_comparison(input_path, output_path)


def fp16_quantize(input_path: Path, output_path: Path) -> None:
    """Convert model weights to FP16."""
    from onnxconverter_common import float16

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
        choices=["int8", "int4", "fp16", "mixed"],
        help="Target precision (default: int8)",
    )
    parser.add_argument(
        "--mode",
        default="dynamic",
        choices=["dynamic", "static", "gptq"],
        help="Quantization mode (default: dynamic)",
    )
    parser.add_argument(
        "--calibration-data", type=Path, help="Calibration dataset directory (required for static)"
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
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    skip_ops = set(args.skip_ops.split(",")) if args.skip_ops else set()

    if args.mode == "gptq":
        if not args.model_id:
            parser.error("--model-id is required for GPTQ mode")
        gptq_bits_by_precision = {"int4": 4, "int8": 8}
        if args.precision not in gptq_bits_by_precision:
            parser.error(f"--precision must be one of {list(gptq_bits_by_precision)} for GPTQ mode")
        gptq_quantize(args.model_id, args.output, bits=gptq_bits_by_precision[args.precision])
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
