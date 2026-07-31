"""Tests for knowledge distillation loss and components."""

from __future__ import annotations

import pytest
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


class TestKLNormalisation:
    """kl_div's 'batchmean' divides by input.size(0).

    On [B, T, V] that is B, which would make the KL term a per-sequence mean while the CE
    term beside it is a per-token mean — scaling the two against each other by the sequence
    length and making alpha/beta depend on max_length.
    """

    def _kl(self, batch, seq, vocab, seed):
        import torch.nn.functional as functional

        from scripts.distill import DistillationConfig, DistillationLoss

        torch.manual_seed(seed)
        loss_fn = DistillationLoss(DistillationConfig(temperature=2.0, alpha=0.0, beta=1.0))
        student = torch.randn(batch, seq, vocab)
        teacher = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))
        losses = loss_fn(student, teacher, labels)

        # Reference: mean per-token KL, computed independently of the implementation.
        log_p = functional.log_softmax(student.reshape(-1, vocab) / 2.0, dim=-1)
        q = functional.softmax(teacher.reshape(-1, vocab) / 2.0, dim=-1)
        reference = (q * (q.clamp_min(1e-12).log() - log_p)).sum(-1).mean() * 4.0
        return losses["loss_kl"], reference

    def test_kl_is_a_per_token_mean(self):
        got, reference = self._kl(2, 10, 100, seed=0)
        assert torch.allclose(
            got, reference, atol=1e-5
        ), f"loss_kl={got.item():.6f} vs per-token reference {reference.item():.6f}"

    def test_kl_does_not_scale_with_sequence_length(self):
        short, _ = self._kl(4, 8, 50, seed=1)
        long, _ = self._kl(4, 64, 50, seed=1)
        # Different draws, so not equal — but they must be the same order of magnitude,
        # not 8x apart, which is what dividing by B instead of B*T produced.
        assert (
            0.5 < (long / short).item() < 2.0
        ), f"KL scaled with sequence length: {short.item():.4f} → {long.item():.4f}"


class TestLabelShift:
    """Causal LMs predict token t+1 from position t, so the hand-rolled CE must shift."""

    def test_shift_makes_a_perfect_next_token_predictor_lossless(self):
        from scripts.distill import DistillationConfig, DistillationLoss

        vocab = 7
        labels = torch.tensor([[1, 2, 3, 4, 5]])
        # Logits at position t are one-hot on label t+1 — a perfect causal predictor.
        logits = torch.full((1, 5, vocab), -20.0)
        for t in range(4):
            logits[0, t, labels[0, t + 1]] = 20.0

        shifted = DistillationLoss(DistillationConfig(alpha=1.0, beta=0.0, shift_labels=True))
        unshifted = DistillationLoss(DistillationConfig(alpha=1.0, beta=0.0, shift_labels=False))

        assert shifted(logits, logits, labels)["loss_ce"].item() < 1e-4
        # Without the shift the same model is scored against the token it was just given.
        assert unshifted(logits, logits, labels)["loss_ce"].item() > 1.0

    def test_shift_is_off_by_default(self):
        from scripts.distill import DistillationConfig

        assert DistillationConfig().shift_labels is False

    def test_shift_ignores_two_dimensional_classification_logits(self):
        from scripts.distill import DistillationConfig, DistillationLoss

        loss_fn = DistillationLoss(DistillationConfig(alpha=1.0, beta=0.0, shift_labels=True))
        student = torch.randn(4, 2)
        labels = torch.randint(0, 2, (4,))
        # Must not raise, and must not drop a row.
        losses = loss_fn(student, torch.randn(4, 2), labels)
        assert losses["loss_ce"].ndim == 0


class TestTrainingArgumentsCompat:
    def test_eval_strategy_is_the_accepted_spelling(self):
        """evaluation_strategy was removed in transformers 4.46; eval_strategy replaced it."""
        import inspect

        transformers = pytest.importorskip("transformers")

        params = inspect.signature(transformers.TrainingArguments.__init__).parameters
        assert "eval_strategy" in params

    def test_distill_does_not_use_the_removed_spelling(self):
        import inspect

        import scripts.distill as distill

        source = inspect.getsource(distill.main)
        assert "evaluation_strategy=" not in source
        assert "eval_strategy=" in source


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
