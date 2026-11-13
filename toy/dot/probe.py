"""Linear position probe on frozen states (Sobal et al. 2022, App. A.4)."""

import torch


def fit_probe(features, batches):
    """Least-squares linear map with bias from features to clean positions.

    ``features(batch)`` returns states (B, T, D). The least-squares solution is the
    optimum of the single linear layer the paper trains by gradient descent.
    """
    xtx = xty = None
    for batch in batches:
        x = features(batch).flatten(0, 1).double()
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        y = batch["pos"].flatten(0, 1).double()
        xtx = x.T @ x if xtx is None else xtx + x.T @ x
        xty = x.T @ y if xty is None else xty + x.T @ y
    ridge = 1e-6 * torch.eye(xtx.shape[0], dtype=xtx.dtype, device=xtx.device)  # states can be rank deficient
    return torch.linalg.solve(xtx + ridge, xty)


def rmse(features, weights, batches):
    """Square root of the mean squared position error at each time step, (T,)."""
    sq, count = 0.0, 0
    for batch in batches:
        x = features(batch).double()
        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        sq = sq + (x @ weights - batch["pos"].double()).square().sum(-1).sum(0)
        count += x.shape[0]
    return (sq / count).sqrt()
