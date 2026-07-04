"""
test_classifier.py — Unit tests for GenreClassifier (PyTorch model).

Tests model architecture, forward pass, and utility functions.
Does NOT require real data — uses synthetic tensors.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musiclab.classifier import GenreClassifier


@pytest.fixture
def model():
    return GenreClassifier(input_dim=8, hidden=64, num_genres=8)


@pytest.fixture
def batch():
    return torch.randn(16, 8)


class TestGenreClassifier:
    def test_init(self, model):
        assert isinstance(model, GenreClassifier)
        params = sum(p.numel() for p in model.parameters())
        assert params > 0
        # 8*64 + 64 + 64*32 + 32 + 32*8 + 8 ≈ 2888
        assert 2000 < params < 5000

    def test_forward_shape(self, model, batch):
        out = model(batch)
        assert out.shape == (16, 8)  # batch_size × num_genres

    def test_forward_single(self, model):
        x = torch.randn(1, 8)
        out = model(x)
        assert out.shape == (1, 8)

    def test_training_mode(self, model):
        """Dropout should be active in training mode."""
        model.train()
        x = torch.randn(100, 8)
        out1 = model(x)
        out2 = model(x)
        # With dropout active, two forward passes should differ
        assert not torch.allclose(out1, out2)

    def test_eval_mode(self, model):
        """Dropout should be inactive in eval mode."""
        model.eval()
        x = torch.randn(100, 8)
        out1 = model(x)
        out2 = model(x)
        # In eval mode, results should be identical
        assert torch.allclose(out1, out2)

    def test_custom_num_genres(self):
        model = GenreClassifier(input_dim=8, hidden=32, num_genres=5)
        out = model(torch.randn(4, 8))
        assert out.shape == (4, 5)

    def test_device_move(self, model):
        """Model should be movable to CPU (default)."""
        out = model(torch.randn(2, 8))
        assert out.device.type == "cpu"

    def test_gradient_flow(self, model):
        """Verify gradients flow through the network."""
        x = torch.randn(4, 8)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that first layer has gradients
        first_param = list(model.parameters())[0]
        assert first_param.grad is not None
        assert first_param.grad.abs().sum() > 0


class TestPredictUtility:
    def test_predict_returns_sorted(self):
        from musiclab.classifier import predict

        model = GenreClassifier(num_genres=3)
        model.eval()
        genres = ["techno", "house", "ambient"]
        result = predict(model, [0.5, 0.0, 1.0, 0.7, 0.65, 0.4, 0.5, 0.6], genres)

        assert len(result) == 3
        # Should be sorted by probability descending
        assert result[0][1] >= result[1][1] >= result[2][1]
        # All probs should sum to ~1.0
        total_prob = sum(p for _, p in result)
        assert total_prob == pytest.approx(1.0, abs=0.01)
