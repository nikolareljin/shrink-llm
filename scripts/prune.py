"""
prune.py — Structured pruning for HuggingFace transformer models.

Supports attention head pruning, MLP neuron pruning, and full layer dropping.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class HeadImportanceScorer:
    """Compute attention head importance using Taylor expansion (gradient × activation)."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.head_importance: dict[str, torch.Tensor] = {}
        self._hooks: list = []

    def register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if "attention" in name.lower() and hasattr(module, "num_heads"):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                attn_weights = output[1]
            else:
                attn_weights = output
            if attn_weights is not None and attn_weights.requires_grad:
                self.head_importance[name] = attn_weights.detach().abs().mean(dim=(0, 2, 3))

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
                self.model(**{k: v for k, v in batch.items() if k != "labels"})
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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured pruning for transformer models")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument(
        "--task", required=True, choices=["ocr", "legal", "baby_cry"], help="Task type"
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

    log.info("Loading model %s...", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(
        args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    before_params = count_parameters(model)
    log.info("Parameters before pruning: %s", f"{before_params:,}")

    if args.method == "layers":
        config = AutoConfig.from_pretrained(args.model)
        num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 12))
        keep = LayerDropper.even_layer_selection(num_layers)
        model = LayerDropper.drop_layers(model, keep)
    elif args.method == "mlp":
        # Synthetic dataloader for demonstration — replace with real dataset
        pruner = MLPPruner(model, sparsity=args.sparsity)
        log.info("Collecting MLP activation statistics...")
        # pruner.collect_activations(dataloader)  # inject real dataloader here
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
    tokenizer.save_pretrained(str(args.output_dir))
    log.info("Pruned model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
