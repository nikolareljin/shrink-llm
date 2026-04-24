"""
prune.py — Structured pruning for HuggingFace transformer models.

Supports attention head pruning, MLP neuron pruning, and full layer dropping.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, TrainingArguments

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class HeadImportanceScorer:
    """Compute attention head importance from average attention weights."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.head_importance: dict[str, torch.Tensor] = {}
        self._hooks: list = []

    def register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            name_lower = name.lower()
            if any(token in name_lower for token in ("attention", "attn")) and hasattr(
                module, "num_heads"
            ):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            attn_weights = None
            if isinstance(output, torch.Tensor):
                attn_weights = output
            elif isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
            if not isinstance(attn_weights, torch.Tensor):
                return
            if attn_weights.ndim == 4:
                self.head_importance[name] = attn_weights.detach().abs().mean(dim=(0, 2, 3))
            elif attn_weights.ndim == 3:
                # 3D means (batch, seq, seq) — already averaged over heads.  Only usable
                # when the module truly has a single head; skip otherwise to avoid producing
                # a length-1 score vector for a multi-head layer.
                if module.num_heads > 1:
                    log.warning(
                        "Layer %s returned 3D attention weights (averaged over heads) "
                        "but module.num_heads=%s; skipping per-head scoring.",
                        name,
                        module.num_heads,
                    )
                    return
                self.head_importance[name] = (
                    attn_weights.detach().abs().mean(dim=(0, 1, 2)).unsqueeze(0)
                )
            else:
                log.warning(
                    "Layer %s returned attention weights with unexpected shape %s; skipping.",
                    name,
                    list(attn_weights.shape),
                )

        return hook

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def score_heads(self, dataloader, num_batches: int = 50) -> dict[str, torch.Tensor]:
        """Run forward passes to compute head importance scores."""
        self.register_hooks()
        self.model.eval()
        scores: dict[str, list] = {}

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                self.head_importance.clear()
                model_inputs = {k: v for k, v in batch.items() if k != "labels"}
                try:
                    self.model(**model_inputs, output_attentions=True)
                except TypeError:
                    self.model(**model_inputs)
                for name, importance in self.head_importance.items():
                    scores.setdefault(name, []).append(importance.cpu())

        self.remove_hooks()
        return {name: torch.stack(v).mean(0) for name, v in scores.items()}


class MLPPruner:
    """Prune low-activation neurons from FFN/MLP layers."""

    def __init__(self, model: nn.Module, sparsity: float = 0.3):
        self.model = model
        self.sparsity = sparsity
        self.activation_stats: dict[str, torch.Tensor] = {}

    def collect_activations(self, dataloader, num_batches: int = 50) -> None:
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and "intermediate" in name:

                def make_hook(n):
                    def hook(m, inp, out):
                        freq = (out.detach().abs() > 0.01).float().mean(dim=(0, 1))
                        self.activation_stats[n] = self.activation_stats.get(n, freq * 0) + freq

                    return hook

                hooks.append(module.register_forward_hook(make_hook(name)))

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                self.model(**{k: v for k, v in batch.items() if k != "labels"})

        for h in hooks:
            h.remove()
        for name in self.activation_stats:
            self.activation_stats[name] /= num_batches

    def prune(self) -> int:
        """Zero out low-activation neurons. Returns number of pruned neurons."""
        total_pruned = 0
        for name, module in self.model.named_modules():
            if name in self.activation_stats and isinstance(module, nn.Linear):
                freq = self.activation_stats[name]
                threshold = freq.quantile(self.sparsity)
                mask = (freq >= threshold).float()
                module.weight.data *= mask.unsqueeze(1)
                if module.bias is not None:
                    module.bias.data *= mask
                total_pruned += int((mask == 0).sum())
        log.info("Pruned %d neurons from MLP layers", total_pruned)
        return total_pruned


class LayerDropper:
    """Remove entire transformer layers from encoder/decoder models."""

    @staticmethod
    def drop_layers(model: nn.Module, keep_layer_indices: list[int]) -> nn.Module:
        """Keep only the specified layer indices."""
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            original = model.encoder.layer
            model.encoder.layer = nn.ModuleList([original[i] for i in keep_layer_indices])
            log.info("Encoder: %d → %d layers", len(original), len(keep_layer_indices))
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            original = model.transformer.h
            model.transformer.h = nn.ModuleList([original[i] for i in keep_layer_indices])
            log.info("Decoder: %d → %d layers", len(original), len(keep_layer_indices))
        return model

    @staticmethod
    def even_layer_selection(num_layers: int) -> list[int]:
        """Select every other layer (DistilBERT approach)."""
        return list(range(0, num_layers, 2))


_SYNTHETIC_BATCHES = 64


def _make_synthetic_dataloader(task: str, device: str, num_batches: int = _SYNTHETIC_BATCHES):
    """Yield synthetic input batches lazily to avoid pre-allocating all tensors on device."""
    for _ in range(num_batches):
        if task == "ocr":
            batch: dict[str, Any] = {"pixel_values": torch.randn(1, 3, 384, 384)}
        elif task == "legal":
            batch = {
                "input_ids": torch.randint(0, 1000, (1, 128)),
                "attention_mask": torch.ones(1, 128, dtype=torch.long),
            }
        elif task == "audio":
            batch = {"input_values": torch.randn(1, 16000)}
        else:
            batch = {
                "input_ids": torch.randint(0, 1000, (1, 64)),
                "attention_mask": torch.ones(1, 64, dtype=torch.long),
            }
        yield {k: v.to(device) for k, v in batch.items()}


def _zero_attention_heads(model: nn.Module, layer_name: str, prune_indices: list[int]) -> None:
    """Zero out Q/K/V weight slices corresponding to pruned attention heads."""
    for name, module in model.named_modules():
        if name != layer_name:
            continue
        if not hasattr(module, "num_heads"):
            continue
        num_heads: int = module.num_heads
        head_dim: int | None = getattr(module, "head_dim", None)
        if num_heads <= 0:
            log.warning(
                "Layer %s: invalid num_heads=%s; skipping head pruning.", layer_name, num_heads
            )
            break

        zeroed_any = False
        for proj_attr in ("q_proj", "k_proj", "v_proj", "query", "key", "value"):
            proj: nn.Linear | None = getattr(module, proj_attr, None)
            if proj is None or not isinstance(proj, nn.Linear):
                continue
            out_features = proj.weight.shape[0]
            if head_dim is not None:
                if head_dim <= 0:
                    log.warning(
                        "Layer %s projection %s: invalid head_dim=%s; skipping projection.",
                        layer_name,
                        proj_attr,
                        head_dim,
                    )
                    continue
                expected_out_features = num_heads * head_dim
                if out_features != expected_out_features:
                    log.warning(
                        "Layer %s projection %s: out_features=%s is inconsistent with "
                        "num_heads=%s and head_dim=%s; skipping projection.",
                        layer_name,
                        proj_attr,
                        out_features,
                        num_heads,
                        head_dim,
                    )
                    continue
                proj_head_dim = head_dim
            else:
                if out_features % num_heads != 0:
                    log.warning(
                        "Layer %s projection %s: out_features=%s is not divisible by "
                        "num_heads=%s; skipping projection.",
                        layer_name,
                        proj_attr,
                        out_features,
                        num_heads,
                    )
                    continue
                proj_head_dim = out_features // num_heads

            invalid_indices = [idx for idx in prune_indices if idx < 0 or idx >= num_heads]
            if invalid_indices:
                log.warning(
                    "Layer %s projection %s: prune indices %s are out of range for num_heads=%s; "
                    "skipping projection.",
                    layer_name,
                    proj_attr,
                    invalid_indices,
                    num_heads,
                )
                continue

            with torch.no_grad():
                for idx in prune_indices:
                    start = idx * proj_head_dim
                    end = start + proj_head_dim
                    proj.weight.data[start:end, :] = 0.0
                    if proj.bias is not None:
                        proj.bias.data[start:end] = 0.0
            zeroed_any = True

        if not zeroed_any:
            log.warning(
                "Layer %s: no standard projection attributes (q_proj/k_proj/v_proj/query/key/value) "
                "found as nn.Linear — head pruning had no effect. "
                "Fused QKV projections or non-Linear layers are not supported.",
                layer_name,
            )
        break


def _load_model_for_task(model_id: str, task: str, device: str) -> tuple[nn.Module, Any]:
    """Load the appropriate model class and tokenizer/processor for each task."""
    if task == "ocr":
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        model = VisionEncoderDecoderModel.from_pretrained(model_id, torch_dtype=torch.float32)
        processor = TrOCRProcessor.from_pretrained(model_id)
    elif task == "audio":
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        model = AutoModelForAudioClassification.from_pretrained(model_id, torch_dtype=torch.float32)
        processor = AutoFeatureExtractor.from_pretrained(model_id)
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        processor = AutoTokenizer.from_pretrained(model_id)
    return model.to(device), processor


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured pruning for transformer models")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument(
        "--task", required=True, choices=["ocr", "legal", "audio"], help="Task type"
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=["attention_heads", "mlp", "layers", "magnitude"],
        help="Pruning method",
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.3, help="Fraction of neurons/heads to prune (0–1)"
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory to save pruned model"
    )
    parser.add_argument(
        "--finetune-epochs", type=int, default=0, help="Epochs to fine-tune after pruning"
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading model %s for task '%s'...", args.model, args.task)
    model, processor = _load_model_for_task(args.model, args.task, args.device)

    before_params = count_parameters(model)
    log.info("Parameters before pruning: %s", f"{before_params:,}")

    if args.method == "attention_heads":
        scorer = HeadImportanceScorer(model)
        synthetic_inputs = _make_synthetic_dataloader(args.task, args.device)
        scores = scorer.score_heads(synthetic_inputs, num_batches=50)
        if scores:
            log.info("Head importance scores computed for %d attention layers", len(scores))
            for layer_name, layer_scores in scores.items():
                n_heads = len(layer_scores)
                n_prune = min(n_heads, max(0, int(n_heads * args.sparsity)))
                log.info("Layer %s: pruning %d/%d heads", layer_name, n_prune, n_heads)
                if n_prune > 0:
                    _, prune_indices = layer_scores.topk(n_prune, largest=False)
                    _zero_attention_heads(model, layer_name, prune_indices.tolist())
            log.info("Attention head pruning complete.")
        else:
            log.warning(
                "No attention layers found via hooks — model may not expose attention weights. "
                "Falling back to magnitude pruning."
            )
            for name, param in model.named_parameters():
                if param.requires_grad and param.dim() >= 2:
                    threshold = param.data.abs().quantile(args.sparsity)
                    param.data[param.data.abs() < threshold] = 0.0

    elif args.method == "layers":
        config = AutoConfig.from_pretrained(args.model)
        num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 12))
        keep = LayerDropper.even_layer_selection(num_layers)
        model = LayerDropper.drop_layers(model, keep)

    elif args.method == "mlp":
        pruner = MLPPruner(model, sparsity=args.sparsity)
        log.info("Collecting MLP activation statistics (synthetic data)...")
        synthetic_inputs = _make_synthetic_dataloader(args.task, args.device)
        pruner.collect_activations(synthetic_inputs)
        pruner.prune()

    elif args.method == "magnitude":
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                threshold = param.data.abs().quantile(args.sparsity)
                param.data[param.data.abs() < threshold] = 0.0
        log.info("Applied magnitude pruning with sparsity=%.2f", args.sparsity)

    after_params = count_parameters(model)
    reduction = (1 - after_params / before_params) * 100
    log.info("Parameters after pruning: %s (%.1f%% reduction)", f"{after_params:,}", reduction)

    if args.finetune_epochs > 0:
        log.info("Fine-tuning pruned model for %d epochs...", args.finetune_epochs)
        _training_args = TrainingArguments(
            output_dir=str(args.output_dir / "finetune_checkpoints"),
            num_train_epochs=args.finetune_epochs,
            per_device_train_batch_size=4,
            learning_rate=1e-5,
            logging_steps=50,
            save_strategy="epoch",
        )
        # trainer = Trainer(model=model, args=training_args, ...)  # inject dataset
        log.info("(Fine-tune trainer configured — inject dataset to enable)")

    model.save_pretrained(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    log.info("Pruned model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
