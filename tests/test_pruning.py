"""Tests for structured pruning modules."""

from __future__ import annotations

import torch
import torch.nn as nn


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
