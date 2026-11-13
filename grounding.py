"""Pixel-grounding helpers shared by the pixel twin and grounded LeWM objectives."""

import torch
import torch.nn.functional as F


def next_frame_targets(pixels, n_preds, size):
    """Frames ``n_preds`` steps ahead of each context frame, area-resized to ``size`` (h, w).

    pixels: (B, T, C, H, W) preprocessed frames. Returns (B, T - n_preds, C, h, w).
    """
    target = pixels[:, n_preds:]
    b = target.shape[0]
    target = F.interpolate(target.flatten(0, 1).float(), size=size, mode="area")
    return target.unflatten(0, (b, -1))


def copy_last_loss(emb, ctx_len, n_preds):
    """Prediction error of the baseline that repeats each context embedding."""
    return (emb[:, :ctx_len] - emb[:, n_preds:]).pow(2).mean()


def grad_shares(terms, emb):
    """Share of the gradient norm at ``emb`` held by each loss term.

    terms: name -> scalar loss, each already scaled by its weight in the total loss.
    The shares sum to one over the given terms.
    """
    norms = {}
    for name, term in terms.items():
        (grad,) = torch.autograd.grad(term, emb, retain_graph=True)
        norms[name] = grad.detach().float().norm()
    total = sum(norms.values()).clamp_min(torch.finfo(torch.float32).tiny)
    return {name: norm / total for name, norm in norms.items()}
