"""
export_to_onnx.py — Export HuggingFace models to ONNX format.

Supports encoder-only, encoder-decoder (seq2seq), and audio classification models.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from transformers import AutoConfig, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TASK_CONFIGS: dict[str, dict] = {
    "ocr": {
        "model_class": "VisionEncoderDecoderModel",
        "processor_class": "TrOCRProcessor",
        "input_names": ["pixel_values"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "pixel_values": {0: "batch_size"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
    },
    "legal": {
        "model_class": "AutoModelForCausalLM",
        "processor_class": "AutoTokenizer",
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
    },
    "baby_cry": {
        "model_class": "AutoModelForAudioClassification",
        "processor_class": "AutoFeatureExtractor",
        "input_names": ["input_values"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "input_values": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
    },
    "classification": {
        "model_class": "AutoModelForImageClassification",
        "processor_class": "AutoFeatureExtractor",
        "input_names": ["pixel_values"],
        "output_names": ["logits"],
        "dynamic_axes": {"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
    },
}


def load_model(model_id: str, task: str, device: str) -> tuple:
    """Load model and processor/tokenizer for the given task."""
    log.info("Loading model '%s' for task '%s'", model_id, task)
    config = AutoConfig.from_pretrained(model_id)

    if task == "ocr":
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        processor = TrOCRProcessor.from_pretrained(model_id)
        model = VisionEncoderDecoderModel.from_pretrained(model_id)
    elif task == "legal":
        from transformers import AutoModelForCausalLM

        processor = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    elif task == "baby_cry":
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        processor = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModelForAudioClassification.from_pretrained(model_id)
    elif task == "classification":
        from transformers import AutoFeatureExtractor, AutoModelForImageClassification

        processor = AutoFeatureExtractor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id)
    else:
        raise ValueError(f"Unknown task: {task}")

    model = model.to(device).eval()
    log.info("Model loaded. Parameters: %s", f"{sum(p.numel() for p in model.parameters()):,}")
    return model, processor, config


def build_dummy_inputs(task: str, device: str) -> dict[str, torch.Tensor]:
    """Create dummy input tensors for ONNX tracing."""
    if task == "ocr":
        return {"pixel_values": torch.randn(1, 3, 384, 384, device=device)}
    elif task == "legal":
        return {
            "input_ids": torch.randint(0, 1000, (1, 128), device=device),
            "attention_mask": torch.ones(1, 128, dtype=torch.long, device=device),
        }
    elif task in ("baby_cry",):
        return {"input_values": torch.randn(1, 16000, device=device)}
    elif task == "classification":
        return {"pixel_values": torch.randn(1, 3, 224, 224, device=device)}
    else:
        raise ValueError(f"Unknown task: {task}")


def export(
    model: torch.nn.Module,
    dummy_inputs: dict[str, torch.Tensor],
    output_path: Path,
    opset: int,
    task: str,
) -> None:
    """Export model to ONNX."""
    task_cfg = TASK_CONFIGS[task]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Exporting to ONNX (opset=%d) → %s", opset, output_path)
    with torch.no_grad():
        torch.onnx.export(
            model,
            args=tuple(dummy_inputs.values()),
            f=str(output_path),
            opset_version=opset,
            input_names=task_cfg["input_names"],
            output_names=task_cfg["output_names"],
            dynamic_axes=task_cfg["dynamic_axes"],
            do_constant_folding=True,
        )
    log.info("ONNX export complete: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)


def validate_onnx(onnx_path: Path, dummy_inputs: dict[str, torch.Tensor]) -> None:
    """Validate the exported ONNX model with ORT."""
    log.info("Validating ONNX model...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {k: v.cpu().numpy() for k, v in dummy_inputs.items()}
    outputs = sess.run(None, ort_inputs)
    log.info("ORT validation passed. Output shapes: %s", [o.shape for o in outputs])


def save_model_config(output_path: Path, model_id: str, task: str, config) -> None:
    """Save model metadata alongside the ONNX file."""
    meta_path = output_path.with_suffix(".json")
    meta = {
        "model_id": model_id,
        "task": task,
        "onnx_path": str(output_path),
        "model_config": config.to_dict() if hasattr(config, "to_dict") else {},
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Model config saved to %s", meta_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HuggingFace model to ONNX")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument(
        "--task",
        required=True,
        choices=list(TASK_CONFIGS.keys()),
        help="Task type for input/output configuration",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output .onnx file path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Device for model loading"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Run ORT inference validation after export"
    )
    parser.add_argument(
        "--no-save-config", action="store_true", help="Skip saving model_config.json"
    )
    args = parser.parse_args()

    model, processor, config = load_model(args.model, args.task, args.device)
    dummy_inputs = build_dummy_inputs(args.task, args.device)
    export(model, dummy_inputs, args.output, args.opset, args.task)

    if args.validate:
        validate_onnx(args.output, dummy_inputs)

    if not args.no_save_config:
        save_model_config(args.output, args.model, args.task, config)

    log.info("Done.")


if __name__ == "__main__":
    main()
