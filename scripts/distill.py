"""
distill.py — Knowledge distillation training (teacher → student).

Supports soft-label KL divergence, hard-label cross-entropy, and
intermediate hidden state alignment losses.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SUPPORTED_TASKS = ("legal",)


@dataclass
class DistillationConfig:
    temperature: float = 6.0  # Softening temperature for KL loss
    alpha: float = 0.1  # Weight for hard label (CE) loss
    beta: float = 0.9  # Weight for soft label (KL) loss
    gamma: float = 0.1  # Weight for hidden state alignment loss
    align_hidden: bool = False  # Whether to align intermediate representations


class DistillationLoss(nn.Module):
    """Combined knowledge distillation loss."""

    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_hidden: torch.Tensor | None = None,
        teacher_hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        temperature = self.config.temperature

        # Hard label loss (cross-entropy against ground truth)
        loss_ce = functional.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
        )

        # Soft label loss (KL divergence against teacher soft targets)
        student_log_probs = functional.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = functional.softmax(teacher_logits / temperature, dim=-1)
        loss_kl = functional.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
            temperature**2
        )

        total = self.config.alpha * loss_ce + self.config.beta * loss_kl

        losses = {"loss_ce": loss_ce, "loss_kl": loss_kl, "loss_total": total}

        # Hidden state alignment loss (MSE between projected hidden states)
        if self.config.align_hidden and student_hidden is not None and teacher_hidden is not None:
            loss_hidden = functional.mse_loss(student_hidden, teacher_hidden.detach())
            total = total + self.config.gamma * loss_hidden
            losses["loss_hidden"] = loss_hidden
            losses["loss_total"] = total

        return losses


class HiddenStateProjector(nn.Module):
    """Project student hidden dim to teacher hidden dim for alignment."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DistillationTrainer(Trainer):
    """HuggingFace Trainer subclass with distillation loss."""

    def __init__(
        self,
        teacher_model: nn.Module,
        distill_config: DistillationConfig,
        projector: HiddenStateProjector | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.distill_config = distill_config
        self.distill_loss = DistillationLoss(distill_config)
        self.projector = projector

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)

        # Student forward
        student_outputs = model(**inputs, output_hidden_states=self.distill_config.align_hidden)
        student_logits = student_outputs.logits

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                **inputs, output_hidden_states=self.distill_config.align_hidden
            )
            teacher_logits = teacher_outputs.logits

        # Optional hidden state alignment
        student_hidden = teacher_hidden = None
        if self.distill_config.align_hidden:
            student_hidden = student_outputs.hidden_states[-1]
            teacher_hidden = teacher_outputs.hidden_states[-1]
            if self.projector is not None:
                student_hidden = self.projector(student_hidden)

        losses = self.distill_loss(
            student_logits, teacher_logits, labels, student_hidden, teacher_hidden
        )
        loss = losses["loss_total"]

        # Log individual loss components
        if self.state.global_step % 50 == 0:
            log_msg = " | ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
            log.info("Step %d | %s", self.state.global_step, log_msg)

        return (loss, student_outputs) if return_outputs else loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge distillation training")
    parser.add_argument("--teacher", required=True, help="Teacher model HuggingFace ID or path")
    parser.add_argument("--student", required=True, help="Student model HuggingFace ID or path")
    parser.add_argument(
        "--task",
        required=True,
        choices=list(SUPPORTED_TASKS),
        help="Task type. Distillation currently supports causal-LM legal models only.",
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Training dataset directory")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Output directory for distilled model"
    )
    parser.add_argument("--temperature", type=float, default=6.0)
    parser.add_argument("--alpha", type=float, default=0.1, help="CE loss weight")
    parser.add_argument("--beta", type=float, default=0.9, help="KL loss weight")
    parser.add_argument("--gamma", type=float, default=0.1, help="Hidden state alignment weight")
    parser.add_argument("--align-hidden", action="store_true", help="Enable hidden state alignment")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading teacher: %s", args.teacher)
    teacher = (
        AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.float32)
        .to(args.device)
        .eval()
    )

    log.info("Loading student: %s", args.student)
    student = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.float32).to(
        args.device
    )

    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optional projector for hidden state alignment
    _projector = None
    if args.align_hidden:
        t_dim = teacher.config.hidden_size
        s_dim = student.config.hidden_size
        if t_dim != s_dim:
            _projector = HiddenStateProjector(s_dim, t_dim).to(args.device)
            log.info("Hidden state projector: %d → %d", s_dim, t_dim)

    _distill_config = DistillationConfig(
        temperature=args.temperature,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        align_hidden=args.align_hidden,
    )

    _training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=args.fp16,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )

    log.info(
        "Distillation config: T=%.1f α=%.2f β=%.2f γ=%.2f",
        args.temperature,
        args.alpha,
        args.beta,
        args.gamma,
    )
    log.info("NOTE: Inject your dataset into DistillationTrainer to enable training.")
    log.info(
        "Teacher params: %s | Student params: %s",
        f"{sum(p.numel() for p in teacher.parameters()):,}",
        f"{sum(p.numel() for p in student.parameters()):,}",
    )

    # trainer = DistillationTrainer(
    #     teacher_model=teacher,
    #     distill_config=distill_config,
    #     projector=_projector,
    #     model=student,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=tokenizer,
    # )
    # trainer.train()
    # student.save_pretrained(str(args.output_dir))
    # tokenizer.save_pretrained(str(args.output_dir))
    log.info(
        "Distillation trainer configured. Uncomment training code and inject datasets to begin."
    )


if __name__ == "__main__":
    main()
