"""H-JEPA's Push-T model: a two-level temporal hierarchy on top of the LeWM encoder.

Level 1 is the LeWM ``JEPA`` itself (``encoder``, ``projector``, ``predictor``,
``pred_proj``, ``action_encoder``), so code that reads the bottom level keeps
working. Level 2 (``upper``) encodes every ``stride``-th level-1 latent and
predicts at that stride, conditioned on a latent action pooled from the
level-1 action embeddings. ``levels`` returns the latents of both levels.

Ported from Zhang et al. 2026, "H-JEPA: End-to-End Learning of Hierarchical World
Models for Visual Planning" (arXiv:2610.06805), kevinghst/H-JEPA@97054ec (MIT),
config ``pusht_hjepa_l2``.

MIT License, Copyright (c) 2026 Wancong Zhang, Basile Terver, Mike Rabbat,
Yann LeCun, Randall Balestriero. See LICENSE.hjepa.
"""

import torch
from einops import rearrange
from torch import nn

from jepa import JEPA


class ResidualLatentMLP(nn.Module):
    """Level-2 encoder: layer norm, one hidden layer."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.input_norm(x))))


class PerStepMLP(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, d_model)

    def forward(self, x):
        return self.fc1(x)


class SequenceEncoder(nn.Module):
    """Pools a chunk of level-1 action embeddings into one latent action (CLS token)."""

    def __init__(
        self,
        output_dim=4,
        input_dim=10,
        d_model=64,
        nhead=2,
        num_layers=1,
        ff_mult=2,
        dropout=0.1,
        max_chunk=15,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.step_mlp = PerStepMLP(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(max_chunk + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model), nn.Linear(d_model, output_dim)
        )
        self.final_ln = nn.LayerNorm(output_dim, elementwise_affine=True, eps=1e-6)

    def forward(self, x, padding_mask):
        """x: (B, T, L, input_dim), padding_mask: (B, T, L) with True on padding."""
        B, T, L, C = x.shape
        x = self.step_mlp(x.reshape(B * T, L, C)) + self.pos[1 : L + 1]
        x = torch.cat([self.pos[0].expand(B * T, 1, -1), x], dim=1)
        pad = torch.cat([padding_mask.new_zeros(B * T, 1), padding_mask.reshape(B * T, L)], dim=1)
        out = self.head(self.encoder(x, src_key_padding_mask=pad)[:, 0])
        return self.final_ln(out).view(B, T, -1)


class LatentActionEmbedder(nn.Module):
    """Embeds a latent action for the level-2 predictor."""

    def __init__(self, input_dim, emb_dim, mlp_scale=4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, mlp_scale * emb_dim), nn.SiLU(), nn.Linear(mlp_scale * emb_dim, emb_dim)
        )

    def forward(self, x):
        return self.embed(x.float())


class UpperLevel(nn.Module):
    """Level 2. Frames are subsampled with ``stride`` and chunks of ``stride`` actions are pooled."""

    def __init__(self, encoder, predictor, pred_proj, action_encoder, action_pool, stride, queue_size):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.pred_proj = pred_proj
        self.action_encoder = action_encoder
        self.action_pool = action_pool
        self.stride = stride
        self.queue_size = queue_size
        # Latent actions seen in training. The planner takes its action prior and clip range from them.
        self.register_buffer("latent_action_queue", torch.empty(0, 0), persistent=True)

    predict = JEPA.predict

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # The queue grows during training, so size it to the saved one before the copy.
        key = prefix + "latent_action_queue"
        if key in state_dict:
            self.latent_action_queue = state_dict[key].new_empty(state_dict[key].shape)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def embed(self, emb):
        """(B, T, D) level-1 latents to (B, ceil(T / stride), D) level-2 latents."""
        starts = torch.arange(0, emb.size(1), self.stride, device=emb.device)
        x = emb[:, starts]
        return rearrange(self.encoder(rearrange(x, "b t d -> (b t) d")), "(b t) d -> b t d", b=emb.size(0))

    def pool(self, act_emb):
        """(B, T, D) level-1 action embeddings to (B, ceil(T / stride), A) latent actions.

        Chunk t holds actions t * stride .. (t + 1) * stride - 1, zero padded past the end.
        """
        B, T, D = act_emb.shape
        starts = torch.arange(0, T, self.stride, device=act_emb.device)
        idx = starts[:, None] + torch.arange(self.stride, device=act_emb.device)
        valid = idx < T
        chunks = act_emb[:, idx.clamp(max=T - 1).reshape(-1)].reshape(B, len(starts), self.stride, D)
        chunks = chunks * valid[None, :, :, None].to(chunks.dtype)
        latent = self.action_pool(chunks, (~valid)[None].expand(B, -1, -1))
        if self.training and self.queue_size > 0:
            samples = latent[:, 0].detach()
            queue = self.latent_action_queue
            if queue.numel():
                samples = torch.cat([queue.to(device=samples.device, dtype=samples.dtype), samples])
            self.latent_action_queue = samples[-self.queue_size :]
        return latent


class HJEPA(JEPA):
    """Two-level Push-T H-JEPA on top of the ordinary LeWM JEPA.

    ``encode`` is inherited from :class:`JEPA` and therefore always has ordinary,
    full-frame level-1 semantics.  The sparse crop used by the hierarchy objective
    is available only through :meth:`encode_hierarchy`.
    """

    def __init__(
        self, encoder, predictor, action_encoder, upper, projector=None, pred_proj=None,
        history_size=3, num_preds=1, upper_lr=1e-5, checkpoint_vit=False,
    ):
        if num_preds != 1:
            raise ValueError("HJEPA supports one-step prediction (num_preds=1)")
        super().__init__(encoder, predictor, action_encoder, projector, pred_proj)
        self.upper = upper
        self.history_size = history_size
        self.num_preds = num_preds
        self.keep = history_size + num_preds
        self.window = (self.keep - 1) * upper.stride + 1
        self.upper_lr = upper_lr
        if checkpoint_vit:
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def encode_hierarchy(self, info):
        """Sparse-encode one canonical training window and build both levels.

        The level-1 loss receives a randomly located contiguous crop of ``keep``
        frames. The upper level reads every strided frame in the complete window.
        The union is encoded once before projection, preserving the official sparse
        level-1 encoding and allowing gradients through both levels.
        """
        pixels = info["pixels"]
        if pixels.size(1) != self.window:
            raise ValueError(f"HJEPA hierarchy expects {self.window} frames, got {pixels.size(1)}")

        pixels = pixels.float()
        batch_size, steps = pixels.shape[:2]
        start = torch.randint(
            0, steps - self.keep + 1, (), device=pixels.device
        ).item()
        frames = sorted(
            set(range(start, start + self.keep))
            | set(range(0, steps, self.upper.stride))
        )
        idx = torch.as_tensor(frames, device=pixels.device)
        cls = self.encoder(
            rearrange(pixels[:, idx], "b t ... -> (b t) ..."),
            interpolate_pos_encoding=True,
        )
        selected = rearrange(
            self.projector(cls.last_hidden_state[:, 0]),
            "(b t) d -> b t d",
            b=batch_size,
        )
        full = selected.new_zeros(batch_size, steps, selected.size(-1))
        full[:, idx] = selected

        act_emb = self.action_encoder(info["action"])
        info["emb"] = full[:, start : start + self.keep]
        info["act_emb"] = act_emb[:, start : start + self.keep]
        info["upper_emb"] = self.upper.embed(full)
        info["upper_act_emb"] = self.upper.action_encoder(self.upper.pool(act_emb))
        return info

    def training_losses(self, batch, sigreg, loss_cfg):
        """Compute the explicit two-level H-JEPA training objective."""
        info = self.encode_hierarchy(batch)
        emb, act_emb = info["emb"], info["act_emb"]
        pred = self.predict(emb[:, : self.history_size], act_emb[:, : self.history_size])
        info["pred_loss"] = (pred - emb[:, 1:]).pow(2).mean()
        info["sigreg_loss"] = sigreg(emb.transpose(0, 1))

        upper_emb, upper_act = info["upper_emb"], info["upper_act_emb"]
        upper_pred = self.upper.predict(
            upper_emb[:, : self.history_size], upper_act[:, : self.history_size]
        )
        info["pred_loss_upper"] = (upper_pred - upper_emb[:, 1:]).pow(2).mean()
        info["sigreg_loss_upper"] = sigreg(upper_emb.transpose(0, 1))

        bottom_weight = loss_cfg.sigreg.weight
        upper_weight = loss_cfg.hjepa.weight
        upper_sigreg_weight = loss_cfg.hjepa.sigreg_weight
        info["loss"] = (
            info["pred_loss"] + bottom_weight * info["sigreg_loss"]
            + upper_weight
            * (info["pred_loss_upper"] + upper_sigreg_weight * info["sigreg_loss_upper"])
        )
        return info

    @torch.no_grad()
    def levels(self, pixels):
        """Return full temporal level-1 and strided level-2 inference latents."""
        info = JEPA.encode(self, {"pixels": pixels})
        return [info["emb"], self.upper.embed(info["emb"])]

    def param_groups(self, lr):
        upper = list(self.upper.parameters())
        ids = {id(p) for p in upper}
        return [
            {"params": [p for p in self.parameters() if id(p) not in ids], "lr": lr},
            {"params": upper, "lr": self.upper_lr},
        ]
