"""MotionJEPA (Karmann et al., 2026, https://arxiv.org/abs/2609.23881) on top of LeWM's JEPA.

Ported from the training code of vivoTechResearch/motionjepa@d732c4f (train.py
``lejepa_forward``, jepa.py ``encode_diff``, module.py ``DiffImageLatentPredictor``),
MIT licence, see LICENSE.motionjepa.

A second ViT encodes the difference of consecutive frames, and an MLP predicts that
embedding from the two image latents. Encoding and planning are LeWM's, so the main latent
is the planning latent.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from jepa import JEPA


class DiffPredictor(nn.Module):
    """Difference embedding from the latents of two consecutive frames."""

    def __init__(self, z_dim, diff_dim, hidden_dim=512, depth=3):
        super().__init__()
        layers = []
        dim = 2 * z_dim
        for _ in range(depth - 1):
            layers.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()])
            dim = hidden_dim
        layers.append(nn.Linear(dim, diff_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, z_next):
        return self.net(torch.cat([z_t, z_next], dim=-1))


class MotionJEPA(JEPA):
    def __init__(self, *args, diff_encoder, diff_projector, diff_predictor, diff_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.diff_encoder = diff_encoder
        self.diff_projector = diff_projector
        self.diff_predictor = diff_predictor
        self.diff_size = diff_size

    def diff_losses(self, info, sigreg, weights):
        """Loss terms added to LeWM's, from an ``info`` dict that ``encode`` has filled."""
        emb = info["emb"]
        pixels = info["pixels"].float()
        diff = pixels[:, 1:] - pixels[:, :-1]
        b = diff.size(0)
        diff = rearrange(diff, "b t ... -> (b t) ...")
        if diff.size(-1) != self.diff_size:
            diff = F.interpolate(diff, size=(self.diff_size, self.diff_size), mode="bilinear", align_corners=False)
        cls = self.diff_encoder(diff, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        diff_emb = rearrange(self.diff_projector(cls), "(b t) d -> b t d", b=b)

        pred_loss = F.mse_loss(self.diff_predictor(emb[:, :-1], emb[:, 1:]), diff_emb)
        sigreg_loss = sigreg(diff_emb.transpose(0, 1))
        return {
            "diff_pred_loss": pred_loss,
            "diff_sigreg_loss": sigreg_loss,
            "diff_loss": weights.diff_pred_weight * pred_loss + weights.diff_sigreg_weight * sigreg_loss,
        }
