import pytest
import torch

from module import grad_shares


def test_grad_shares_split_the_gradient_norm_at_the_embedding():
    emb = torch.randn(3, 4, requires_grad=True)
    shares = grad_shares({"a": (2 * emb).sum(), "b": (6 * emb).sum()}, emb)
    assert float(shares["a"]) == pytest.approx(0.25) and float(shares["b"]) == pytest.approx(0.75)
    assert emb.grad is None
