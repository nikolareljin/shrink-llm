"""Tests for knowledge distillation loss and components."""

from __future__ import annotations

import torch


class TestDistillationLoss:
    def setup_method(self):
        from scripts.distill import DistillationConfig, DistillationLoss

        self.config = DistillationConfig(temperature=4.0, alpha=0.1, beta=0.9, gamma=0.1)
        self.loss_fn = DistillationLoss(self.config)

    def test_loss_is_scalar(self):
        batch, seq, vocab = 2, 10, 100
        student_logits = torch.randn(batch, seq, vocab)
        teacher_logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))

        losses = self.loss_fn(student_logits, teacher_logits, labels)
        assert losses["loss_total"].ndim == 0
        assert losses["loss_ce"].ndim == 0
        assert losses["loss_kl"].ndim == 0

    def test_loss_is_positive(self):
        batch, seq, vocab = 2, 10, 100
        student_logits = torch.randn(batch, seq, vocab)
        teacher_logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))

        losses = self.loss_fn(student_logits, teacher_logits, labels)
        assert losses["loss_total"].item() > 0

    def test_alpha_beta_weights(self):
        """When alpha=1 and beta=0, total ≈ CE loss."""
        from scripts.distill import DistillationConfig, DistillationLoss

        config = DistillationConfig(temperature=1.0, alpha=1.0, beta=0.0, gamma=0.0)
        loss_fn = DistillationLoss(config)

        batch, seq, vocab = 2, 10, 100
        student_logits = torch.randn(batch, seq, vocab)
        teacher_logits = torch.zeros(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))

        losses = loss_fn(student_logits, teacher_logits, labels)
        assert abs(losses["loss_total"].item() - losses["loss_ce"].item()) < 1e-4

    def test_hidden_alignment_loss(self):
        from scripts.distill import DistillationConfig, DistillationLoss

        config = DistillationConfig(
            temperature=4.0, alpha=0.1, beta=0.9, gamma=0.5, align_hidden=True
        )
        loss_fn = DistillationLoss(config)

        batch, seq, vocab = 2, 10, 100
        hidden_dim = 64
        student_logits = torch.randn(batch, seq, vocab)
        teacher_logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))
        student_hidden = torch.randn(batch, seq, hidden_dim)
        teacher_hidden = torch.randn(batch, seq, hidden_dim)

        losses = loss_fn(student_logits, teacher_logits, labels, student_hidden, teacher_hidden)
        assert "loss_hidden" in losses
        assert losses["loss_hidden"].item() >= 0


class TestHiddenStateProjector:
    def test_projects_correctly(self):
        from scripts.distill import HiddenStateProjector

        proj = HiddenStateProjector(student_dim=512, teacher_dim=1024)
        x = torch.randn(2, 10, 512)
        out = proj(x)
        assert out.shape == (2, 10, 1024)


class TestSupportedTasks:
    def test_distillation_supports_only_legal_task(self):
        from scripts.distill import SUPPORTED_TASKS

        assert SUPPORTED_TASKS == ("legal",)
