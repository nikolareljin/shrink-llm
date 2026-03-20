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


class TestCountParameters:
    def test_counts_correctly(self):
        from scripts.prune import count_parameters

        model = nn.Linear(10, 10, bias=True)
        # 10*10 weights + 10 bias = 110
        assert count_parameters(model) == 110
