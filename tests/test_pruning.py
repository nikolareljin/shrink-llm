"""Tests for structured pruning modules."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class TestMagnitudeThreshold:
    """torch.quantile raises above 2**24 elements; every transformer embedding exceeds that."""

    def test_matches_quantile_below_the_limit(self):
        from scripts.prune import magnitude_threshold

        torch.manual_seed(0)
        t = torch.randn(1000, 100)  # 100k elements — quantile still works here
        got = magnitude_threshold(t, 0.3)
        reference = t.abs().quantile(0.3)
        assert torch.allclose(got, reference, atol=1e-3), f"{got.item()} vs {reference.item()}"

    def test_handles_tensors_larger_than_the_quantile_limit(self):
        from scripts.prune import magnitude_threshold

        # 2**24 is torch.quantile's hard cap. BERT-base's embedding (30522x768 = 23.4M)
        # is over it, so the old param.data.abs().quantile(...) crashed on every real model.
        big = torch.randn(2**24 + 1024)
        threshold = magnitude_threshold(big, 0.3)
        assert threshold.ndim == 0
        assert threshold > 0
        with pytest.raises(RuntimeError, match="too large"):
            torch.quantile(big.abs(), 0.3)

    def test_empty_tensor_yields_zero(self):
        from scripts.prune import magnitude_threshold

        assert magnitude_threshold(torch.empty(0), 0.5).item() == 0.0


class TestApplyMagnitudePruning:
    def test_zeroes_the_requested_fraction(self):
        from scripts.prune import _apply_magnitude_pruning

        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(64, 64, bias=False))
        _apply_magnitude_pruning(model, 0.5)

        weight = model[0].weight.data
        zero_fraction = (weight == 0).float().mean().item()
        assert 0.45 < zero_fraction < 0.55, zero_fraction

    def test_full_sparsity_zeroes_everything(self):
        from scripts.prune import _apply_magnitude_pruning

        model = nn.Sequential(nn.Linear(8, 8, bias=False))
        _apply_magnitude_pruning(model, 1.0)
        assert model[0].weight.data.count_nonzero().item() == 0

    def test_leaves_one_dimensional_parameters_alone(self):
        from scripts.prune import _apply_magnitude_pruning

        model = nn.Sequential(nn.Linear(8, 8))
        before = model[0].bias.data.clone()
        _apply_magnitude_pruning(model, 0.9)
        assert torch.equal(model[0].bias.data, before)


class TestTaskParityWithExporter:
    def test_prune_accepts_every_exporter_task(self):
        from scripts.export_to_onnx import TASK_CONFIGS
        from scripts.prune import SUPPORTED_TASKS

        missing = sorted(set(TASK_CONFIGS) - set(SUPPORTED_TASKS))
        assert not missing, (
            f"prune.py rejects task(s) export_to_onnx.py accepts: {missing}. "
            "run_pipeline.py passes the config's task to both."
        )

    def test_every_supported_task_has_synthetic_inputs(self):
        from scripts.prune import SUPPORTED_TASKS, _make_synthetic_dataloader

        for task in SUPPORTED_TASKS:
            batch = next(_make_synthetic_dataloader(task, "cpu", num_batches=1))
            assert batch, f"no synthetic inputs for supported task {task!r}"


class TestMLPPrunerAveraging:
    def test_divides_by_batches_actually_seen(self):
        """A short loader must not bias activation frequencies toward zero.

        The frequencies are what the pruning threshold is computed from, so dividing by the
        requested batch count rather than the delivered one silently changes which neurons
        get pruned.
        """
        from scripts.prune import MLPPruner

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.intermediate = nn.Linear(4, 6)

            def forward(self, x):
                return self.intermediate(x)

        model = Net()
        with torch.no_grad():  # every unit always active, so frequency must average to 1.0
            model.intermediate.weight.fill_(0.0)
            model.intermediate.bias.fill_(1.0)

        pruner = MLPPruner(model, sparsity=0.3)
        two_batches = [{"x": torch.randn(1, 3, 4)} for _ in range(2)]
        pruner.collect_activations(iter(two_batches), num_batches=50)

        freq = pruner.activation_stats["intermediate"]
        assert torch.allclose(freq, torch.ones_like(freq)), freq


class TestLayerDropper:
    def test_even_layer_selection(self):
        from scripts.prune import LayerDropper

        result = LayerDropper.even_layer_selection(12)
        assert result == [0, 2, 4, 6, 8, 10]

    def test_even_layer_selection_odd_count(self):
        from scripts.prune import LayerDropper

        result = LayerDropper.even_layer_selection(7)
        assert result == [0, 2, 4, 6]

    def test_drop_layers_reduces_count(self):
        from scripts.prune import LayerDropper

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.layer = nn.ModuleList([nn.Linear(64, 64) for _ in range(12)])

        model = FakeModel()
        keep = [0, 2, 4, 6, 8, 10]
        model = LayerDropper.drop_layers(model, keep)
        assert len(model.encoder.layer) == 6


class TestMLPPruner:
    def test_prune_zeros_low_activation_neurons(self):
        from scripts.prune import MLPPruner

        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        pruner = MLPPruner(model, sparsity=0.5)

        # Manually set activation stats
        pruner.activation_stats["0"] = torch.rand(128)

        # Should not crash
        pruned = pruner.prune()
        assert isinstance(pruned, int)


class TestHeadImportanceScorer:
    def test_score_heads_requests_attentions_and_collects_scores(self):
        from scripts.prune import HeadImportanceScorer

        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 2

            def forward(self, input_ids=None, output_attentions=False, **kwargs):
                assert output_attentions is True
                weights = torch.ones(1, self.num_heads, 4, 4)
                return torch.zeros(1, 4, 8), weights

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = FakeAttention()

            def forward(self, **kwargs):
                return self.attention(**kwargs)

        scorer = HeadImportanceScorer(FakeModel())
        scores = scorer.score_heads(
            [{"input_ids": torch.ones(1, 4, dtype=torch.long)}], num_batches=1
        )

        assert "attention" in scores
        assert scores["attention"].shape == (2,)
        assert torch.allclose(scores["attention"], torch.ones(2))

    def test_register_hooks_matches_attn_module_names(self):
        from scripts.prune import HeadImportanceScorer

        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 2

            def forward(self, input_ids=None, output_attentions=False, **kwargs):
                weights = torch.ones(1, self.num_heads, 4, 4)
                return torch.zeros(1, 4, 8), weights

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = FakeAttention()

            def forward(self, **kwargs):
                return self.self_attn(**kwargs)

        scorer = HeadImportanceScorer(FakeModel())
        scores = scorer.score_heads(
            [{"input_ids": torch.ones(1, 4, dtype=torch.long)}], num_batches=1
        )

        assert "self_attn" in scores

    def test_score_heads_falls_back_when_model_rejects_output_attentions(self):
        from scripts.prune import HeadImportanceScorer

        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 2

            def forward(self, input_ids=None, **kwargs):
                weights = torch.ones(1, self.num_heads, 4, 4)
                return torch.zeros(1, 4, 8), weights

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = FakeAttention()

            def forward(self, input_ids=None):
                return self.attention(input_ids=input_ids)

        scorer = HeadImportanceScorer(FakeModel())
        scores = scorer.score_heads(
            [{"input_ids": torch.ones(1, 4, dtype=torch.long)}], num_batches=1
        )

        assert "attention" in scores
        assert torch.allclose(scores["attention"], torch.ones(2))

    def test_score_heads_accepts_tensor_attention_outputs(self):
        from scripts.prune import HeadImportanceScorer

        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 2

            def forward(self, input_ids=None, output_attentions=False, **kwargs):
                return torch.ones(1, self.num_heads, 4, 4)

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = FakeAttention()

            def forward(self, **kwargs):
                return self.attention(**kwargs)

        scorer = HeadImportanceScorer(FakeModel())
        scores = scorer.score_heads(
            [{"input_ids": torch.ones(1, 4, dtype=torch.long)}], num_batches=1
        )

        assert "attention" in scores
        assert torch.allclose(scores["attention"], torch.ones(2))


class TestZeroAttentionHeads:
    def test_skips_invalid_projection_shapes(self):
        from scripts.prune import _zero_attention_heads

        class FakeAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_heads = 3
                self.q_proj = nn.Linear(8, 8)

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = FakeAttention()

        model = FakeModel()
        before = model.attention.q_proj.weight.detach().clone()
        _zero_attention_heads(model, "attention", [0])
        assert torch.equal(model.attention.q_proj.weight, before)


class TestCountParameters:
    def test_counts_correctly(self):
        from scripts.prune import count_parameters

        model = nn.Linear(10, 10, bias=True)
        # 10*10 weights + 10 bias = 110
        assert count_parameters(model) == 110
