"""Frozen copy of the H-JEPA code a Push-T ``pusht_hjepa_l2`` forward and plan runs through.

Source: https://github.com/kevinghst/H-JEPA at 97054ecaff83e13839195ec4b92d74c08a0127c5
(Zhang, Terver, Rabbat, LeCun, Balestriero, "H-JEPA: End-to-End Learning of Hierarchical
World Models for Visual Planning", arXiv:2610.06805), MIT licensed, see LICENSE next to this file.
The classes and functions are verbatim from ``h_jepa/`` (``loss.py``, ``models/``,
``hierarchical_solver.py``) and ``stable_worldmodel/solver/solver.py``;
only ``build_pusht_hjepa_l2`` is new. Do not edit them: the tests compare the port against them.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import ViTConfig, ViTModel
import numpy as np
from typing import Any


def init_module_weights(m, std=0.02):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (ECF means averaged across DDP ranks)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        cos, sin = x_t.cos().mean(-3), x_t.sin().mean(-3)
        n = proj.size(-2)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # untracked by autograd: each rank backprops its local ECF, DDP's grad average
            # then yields the gradient of the global statistic
            world_size = torch.distributed.get_world_size()
            for ecf in (cos, sin):
                torch.distributed.all_reduce(ecf)
                ecf.data.div_(world_size)
            n = n * world_size
        err = (cos - self.phi).square() + sin.square()
        statistic = (err @ self.weights) * n
        return statistic.mean()


def build_projector(projector_cfg, *, input_dim: int, output_dim: int) -> nn.Module:
    projector_type = str(projector_cfg.get("type", "mlp")).lower()

    if projector_type == "identity":
        if int(input_dim) != int(output_dim):
            raise ValueError(
                "Identity projector cannot change feature size: "
                f"got input_dim={input_dim}, output_dim={output_dim}."
            )
        return nn.Identity()

    if projector_type == "layernorm":
        return nn.LayerNorm(output_dim)

    if projector_type == "mlp":
        return MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=2048,
            norm_fn=nn.BatchNorm1d,
        )

    raise ValueError(f"Unsupported projector type '{projector_cfg.get('type')}'")


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):
        x = self.input_proj(x)

        if c is not None:
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)
        x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
        patch_embed=True,
    ):
        super().__init__()
        self.patch_embed = (
            nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
            if patch_embed
            else None
        )
        feature_dim = smoothed_dim if patch_embed else input_dim
        self.embed = nn.Sequential(
            nn.Linear(feature_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D) or (B, T, C, D)
        """
        x = x.float()
        if self.patch_embed is not None:
            if x.dim() == 3:
                x = x.permute(0, 2, 1)
                x = self.patch_embed(x)
                x = x.permute(0, 2, 1)
            elif x.dim() == 4:
                b, t, c, d = x.shape
                x = x.reshape(b, t * c, d)
                x = x.permute(0, 2, 1)
                x = self.patch_embed(x)
                x = x.permute(0, 2, 1)
                x = x.reshape(b, t, c, -1)
            else:
                raise ValueError(f"Expected 3D or 4D input, got shape {tuple(x.shape)}")
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
        final_ln=False,
    ):
        super().__init__()
        out_dim = output_dim or input_dim
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.final_norm = nn.LayerNorm(out_dim) if final_ln else nn.Identity()
    
    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.final_norm(self.net(x))


class ResidualLatentMLP(nn.Module):
    """MLP for refining latent vectors."""

    def __init__(
        self,
        input_dim,
        hidden_dim=384,
        output_dim=None,
        act_fn=nn.GELU,
        final_ln=True,
        trunc_normal_init=False,
    ):
        super().__init__()
        out_dim = output_dim or input_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = act_fn()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.final_norm = nn.LayerNorm(out_dim) if final_ln else nn.Identity()
        if trunc_normal_init:
            self.apply(init_module_weights)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.final_norm(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class PerStepMLP(nn.Module):
    """
    Effectively similar to a per-step Conv1d projection used in DINO-WM.
    Maps each token vector independently from input_dim -> d_model.
    """

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, L, input_dim)

        Returns:
            Tensor of shape (B, L, d_model)
        """
        return self.fc1(x)


class SequenceEncoder(nn.Module):
    """
    Encode a fixed or variable-length sequence chunk into one vector (CLS token).

    Inputs:
        inputs:       (B, T, L, input_dim)
        padding_mask: (B, T, L), where True = padding (ignore)

    Outputs:
        z: (B, T, output_dim)
    """

    def __init__(
        self,
        output_dim: int = 4,
        input_dim: int = 10,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 1,
        ff_mult: int = 2,
        dropout: float = 0.1,
        max_chunk: int = 15,
        norm_first: bool = True,
    ):
        super().__init__()

        self.max_chunk = max_chunk
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.step_mlp = PerStepMLP(input_dim=input_dim, d_model=d_model)

        self.pos = nn.Parameter(torch.zeros(max_chunk + 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
        )

        self.final_ln = nn.LayerNorm(output_dim, elementwise_affine=True, eps=1e-6)

    def forward(
        self,
        inputs: torch.Tensor,
        padding_mask: torch.Tensor = None,
    ):
        """
        Args:
            inputs: Tensor of shape (B, T, L, input_dim)
            padding_mask: Bool tensor of shape (B, T, L), where True = padding;
                None means every position is valid.

        Returns:
            z: (B, T, output_dim)
        """
        B, T, L, C = inputs.shape
        assert C == self.input_dim, (
            f"Last dim of inputs must be {self.input_dim}, got {C}"
        )

        if padding_mask is None:
            padding_mask = torch.zeros(B, T, L, dtype=torch.bool, device=inputs.device)

        x = inputs.reshape(B * T, L, C)
        m = padding_mask.reshape(B * T, L)

        x = self.step_mlp(x)

        pos = self.pos[1 : L + 1].unsqueeze(0)
        x = x + pos

        cls_tok = self.pos[0].unsqueeze(0).unsqueeze(0).expand(B * T, 1, -1)
        x = torch.cat([cls_tok, x], dim=1)

        pad_cls = torch.zeros(B * T, 1, dtype=torch.bool, device=x.device)
        src_key_padding_mask = torch.cat([pad_cls, m], dim=1)

        y = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        out = self.head(y[:, 0, :])

        z = self.final_ln(out)
        z = z.view(B, T, -1)
        return z


VIT_SIZE_CONFIGS = {
    "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
    "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
    "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
    "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
    "huge": {"hidden_size": 1280, "num_hidden_layers": 32, "num_attention_heads": 16},
}


def create_hf_vit(
    size: str = "tiny",
    patch_size: int = 16,
    image_size: int = 224,
    use_mask_token: bool = True,
    **kwargs,
) -> nn.Module:
    if size not in VIT_SIZE_CONFIGS:
        raise ValueError(
            f"Invalid size '{size}'. Choose from {list(VIT_SIZE_CONFIGS.keys())}"
        )

    config_params = dict(VIT_SIZE_CONFIGS[size])
    config_params["intermediate_size"] = config_params["hidden_size"] * 4
    config_params["image_size"] = image_size
    config_params["patch_size"] = patch_size
    config_params.update(kwargs)

    config = ViTConfig(**config_params)
    model = ViTModel(
        config,
        add_pooling_layer=False,
        use_mask_token=use_mask_token,
    )

    model.config.interpolate_pos_encoding = True
    return model


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


def _run_encoder(encoder, x):
    if x.ndim >= 4:
        return encoder(x, interpolate_pos_encoding=True)
    return encoder(x)


def _split_encoder_output(
    encoder_output,
    allow_cls_token: bool = True,
) -> torch.Tensor:
    hidden = (
        encoder_output.last_hidden_state
        if hasattr(encoder_output, "last_hidden_state")
        else encoder_output
    )

    if hidden.ndim == 2:
        return hidden
    if hidden.ndim == 3 and allow_cls_token:
        return hidden[:, 0]

    if hidden.ndim == 3:
        raise ValueError(
            "Chunked temporal encoder inputs must be collapsed by the encoder "
            "to rank-2 shape (B*T, D); got rank-3 output "
            f"with shape {tuple(hidden.shape)}"
        )

    raise ValueError(
        "Encoder output must be rank-2 or rank-3, "
        f"got shape {tuple(hidden.shape)}"
    )


class ProjectedEncoder(nn.Module):
    """Encoder followed by its projection head."""

    def __init__(self, encoder, projector=None):
        super().__init__()
        self.encoder = encoder
        self.projector = projector or nn.Identity()

    def forward(self, x, *, allow_cls_token: bool = True):
        output = _run_encoder(self.encoder, x)
        hidden = _split_encoder_output(output, allow_cls_token=allow_cls_token)
        return self.projector(hidden)

    def encode_info(self, info, *, key: str, allow_cls_token: bool = True):
        x = rearrange(info[key].float(), "b t ... -> (b t) ...")
        return {"embed": self(x, allow_cls_token=allow_cls_token)}


class ProjectedPredictor(nn.Module):
    """Predictor followed by its projection head."""

    def __init__(self, predictor, projector=None):
        super().__init__()
        self.predictor = predictor
        self.projector = projector or nn.Identity()

    def forward(self, emb, act_emb):
        preds = self.predictor(emb, act_emb)
        preds = self.projector(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) ... -> b t ...", b=emb.size(0))


class JEPA(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_embed,
        action_encoder=None,
        level:int=1,
        action_queue_size: int = 0,
        temporal_stride: int = 1,
        temporal_window_size: int = 1,
        target_length: int | None = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_embed = action_embed
        self.action_encoder = action_encoder or nn.Identity()
        self.level = level
        self.temporal_stride = int(temporal_stride)
        self.temporal_window_size = int(temporal_window_size)
        self.target_length = None if target_length is None else int(target_length)
        self.configure_action_queue(action_queue_size)

    def _state_windows(
        self,
        x: torch.Tensor,
        starts: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        """Subsample or chunk a time sequence using full state windows only."""
        if window_size == 1:
            return x.index_select(1, starts)

        offsets = torch.arange(window_size, device=x.device)
        indices = starts[:, None] + offsets[None, :]
        flat = indices.reshape(-1)
        chunks = x.index_select(1, flat)
        return chunks.reshape(x.size(0), starts.numel(), window_size, *x.shape[2:])

    def _action_windows(
        self,
        x: torch.Tensor,
        starts: torch.Tensor,
        state_window_size: int,
        stride: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunk actions with zero padding.

        The first action in chunk T aligns with the last state in chunk T.
        The returned mask is True for padded positions.
        """
        seq_len = x.size(1)
        action_starts = starts + state_window_size - 1
        offsets = torch.arange(stride, device=x.device)
        indices = action_starts[:, None] + offsets[None, :]
        valid = indices < seq_len
        padding_mask = ~valid
        indices = indices.clamp(max=seq_len - 1)

        flat = indices.reshape(-1)
        chunks = x.index_select(1, flat)
        chunks = chunks.reshape(x.size(0), starts.numel(), stride, *x.shape[2:])

        valid = valid.unsqueeze(0)
        while valid.ndim < chunks.ndim:
            valid = valid.unsqueeze(-1)
        mask = padding_mask.unsqueeze(0).expand(x.size(0), -1, -1)
        return chunks * valid.to(dtype=chunks.dtype), mask

    def _chunk_temporal_info(
        self,
        info: dict,
        key: str,
        stride: int,
        window_size: int,
    ) -> dict:
        stride = int(stride)
        window_size = int(window_size)
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        if key not in info:
            raise KeyError(f"Input key '{key}' not found in info")

        ref = info[key]
        if not torch.is_tensor(ref) or ref.ndim < 3:
            raise ValueError(
                f"Expected '{key}' to be a tensor with shape (B, T, ...), "
                f"got {type(ref)} with shape {getattr(ref, 'shape', None)}"
            )

        seq_len = ref.size(1)
        if seq_len < window_size:
            raise ValueError(
                f"Sequence length {seq_len} is shorter than window_size={window_size}"
            )
        starts = torch.arange(0, seq_len - window_size + 1, stride, device=ref.device)
        chunked = dict(info)

        for info_key, value in info.items():
            if not torch.is_tensor(value) or value.ndim < 3:
                continue
            if value.size(0) != ref.size(0) or value.size(1) != seq_len:
                continue

            if info_key == "action":
                action_chunks, action_mask = self._action_windows(
                    value,
                    starts=starts,
                    state_window_size=window_size,
                    stride=stride,
                )
                chunked[info_key] = action_chunks
                chunked["action_mask"] = action_mask
            elif not info_key.startswith("action"):
                chunked[info_key] = self._state_windows(
                    value,
                    starts=starts,
                    window_size=window_size,
                )

        return chunked

    def configure_action_queue(self, queue_size: int = 0) -> None:
        """Configure latent action queue used to initialize planner priors."""
        queue_size = int(queue_size)

        self.action_queue_size = queue_size
        if queue_size == 0:
            self.latent_action_queue = None
            return

        if "latent_action_queue" not in self._buffers:
            self.register_buffer(
                "latent_action_queue", torch.empty(0, 0), persistent=True
            )
        elif self.latent_action_queue is None:
            self._buffers["latent_action_queue"] = torch.empty(0, 0)

        queue = self.latent_action_queue
        if torch.is_tensor(queue) and queue.ndim == 2 and queue.numel() > 0:
            self._buffers["latent_action_queue"] = queue[-queue_size:]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # latent_action_queue grows during training, so a fresh model's empty (0,0)
        # buffer can't receive a checkpoint's (queue_size, A_emb) tensor under strict
        # load. Resize it to the saved shape before the parent copies into it.
        key = prefix + "latent_action_queue"
        if key in state_dict and "latent_action_queue" in self._buffers:
            saved = state_dict[key]
            self._buffers["latent_action_queue"] = saved.new_empty(saved.shape)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _update_latent_action_queue(self, latent_actions: torch.Tensor) -> None:
        """latent_actions: (B, T, A_emb)"""

        if not self.training or self.action_queue_size <= 0:
            return

        assert latent_actions.ndim == 3, "latent_actions must be (B, T, A_emb)"

        samples = latent_actions[:, 0].detach()

        queue = self.latent_action_queue
        queue_size = int(self.action_queue_size)

        if queue.numel() == 0:
            updated = samples
        else:
            updated = torch.cat(
                [queue.to(device=samples.device, dtype=samples.dtype), samples],
                dim=0,
            )

        self._buffers["latent_action_queue"] = updated[-queue_size:]

    def get_latent_action_queue(self) -> torch.Tensor | None:
        """Return latent action queue, or None if disabled/empty."""
        queue = getattr(self, "latent_action_queue", None)
        if not torch.is_tensor(queue) or queue.numel() == 0:
            return None

        return queue

    def encode(
        self,
        info,
        key="pixels",
        temporal_stride: int | None = None,
        chunk_temporal_inputs: bool = False,
    ):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """
        chunked_state_input = False
        if chunk_temporal_inputs:
            if temporal_stride is None:
                temporal_stride = getattr(self, "temporal_stride", 1)
            temporal_window_size = getattr(self, "temporal_window_size", 1)
            chunked_state_input = int(temporal_window_size) > 1

            info = self._chunk_temporal_info(
                info,
                key=key,
                stride=temporal_stride,
                window_size=temporal_window_size,
            )

        B = info[key].size(0)
        encoder_outputs = self.encoder.encode_info(
            info,
            key=key,
            allow_cls_token=not chunked_state_input,
        )
        for output_key, value in encoder_outputs.items():
            info[f"{output_key}_0"] = rearrange(
                value,
                "(b t) ... -> b t ...",
                b=B,
            )

        if "action" in info:
            action_mask = info.get("action_mask")
            if action_mask is None:
                latent_actions = self.action_encoder(info["action"])
            else:
                latent_actions = self.action_encoder(info["action"], action_mask)
            self._update_latent_action_queue(latent_actions)
            info["action_0"] = self.action_embed(latent_actions)

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        return self.predictor(emb, act_emb)

    def parallel_unroll(self, emb, act_emb, nsteps):
        """nsteps passes over the band: each pass predicts frames 1..T-1 from the previous pass (the
        encoder band on pass 0, detached after), with ground-truth frame 0 re-injected on the left.
        Returns every pass stacked: (nsteps, B, T, D), aligned with emb.
        """
        T = emb.size(1) - 1
        pred_input = emb[:, :T]
        passes = []
        for _ in range(nsteps):
            predicted = torch.cat([emb[:, :1], self.predict(pred_input, act_emb[:, :T])], dim=1)
            passes.append(predicted)
            pred_input = torch.cat([emb[:, :1], predicted[:, 1:T].detach()], dim=1)
        return torch.stack(passes)

    def rollout(self, info, action_sequence, history_size: int = 1):
        """Rollout the model given an initial info dict and action sequence."""

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        if "embed_0" not in _init:
            with torch.no_grad():
                _init = self.encode(_init)
        emb = info["embed_0"] = _init["embed_0"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_embed(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_embed(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_embed_0"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Planning cost: MSE between the last cost_last_n predicted embeddings and the goal."""
        pred_emb = info_dict["predicted_embed_0"][:, :, 1:, :]
        cost_last_n = max(1, int(info_dict.get("cost_last_n", 1)))
        cost_window = min(cost_last_n, pred_emb.shape[-2])

        pred_emb = pred_emb[:, :, -cost_window:, :]
        goal_emb = info_dict["goal_embed_0"].expand_as(pred_emb).detach()
        cost = F.mse_loss(pred_emb, goal_emb, reduction="none")

        intermediate_weight = float(info_dict.get("intermediate_cost_weight", 1.0))
        if intermediate_weight != 1.0 and cost.shape[2] > 1:
            weights = cost.new_ones(cost.shape[2])
            weights[:-1] = intermediate_weight
            cost = cost * weights.view(1, 1, -1, 1)

        return cost.sum(dim=tuple(range(2, cost.ndim)))

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        if "goal_embed_0" not in info_dict:
            assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        if "goal_embed_0" not in info_dict:
            _goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
            _goal["pixels"] = _goal["goal"]

            for k in info_dict:
                if k.startswith("goal_"):
                    _goal[k[len("goal_") :]] = _goal.pop(k)

            _goal.pop("action")
            with torch.no_grad():
                _goal = self.encode(_goal)

            # keep the sample axis, (B, 1, T, D): expand_as against (B, S, W, D) is wrong for B > 1
            info_dict["goal_embed_0"] = _goal["embed_0"].unsqueeze(1)

        info_dict = self.rollout(info_dict, action_candidates)

        return self.criterion(info_dict)


class HJEPA(nn.Module):
	"""Hierarchical container for multiple JEPA models.

	Args:
		jepas: Ordered JEPA modules where jepas[0] is level 1,
			jepas[1] is level 2, etc.
	"""

	def __init__(self, jepas: Sequence[nn.Module]):
		super().__init__()

		if not isinstance(jepas, Sequence):
			raise TypeError("jepas must be a sequence of nn.Module instances")
		if len(jepas) == 0:
			raise ValueError("jepas must contain at least one JEPA module")

		non_modules = [i for i, module in enumerate(jepas) if not isinstance(module, nn.Module)]
		if non_modules:
			raise TypeError(
				"All entries in jepas must be nn.Module instances; "
				f"invalid indices: {non_modules}"
			)

		# ModuleList ensures every JEPA is registered as a submodule.
		self.jepas = nn.ModuleList(jepas)

	@property
	def num_levels(self) -> int:
		"""Number of levels in the hierarchy."""
		return len(self.jepas)

	def get_level(self, level: int) -> nn.Module:
		"""Get a JEPA by 1-based level index (level=1 -> jepas[0])."""
		if level < 1 or level > self.num_levels:
			raise IndexError(
				f"level must be in [1, {self.num_levels}], got {level}"
			)
		return self.jepas[level - 1]

	def __getitem__(self, idx: int) -> nn.Module:
		"""Get a JEPA by 0-based index (idx=0 -> level 1)."""
		return self.jepas[idx]

	def __len__(self) -> int:
		"""Number of levels in the hierarchy."""
		return self.num_levels

	def _level_uses_proprio(self, level: int) -> bool:
		"""Whether this level encodes pixel and proprio streams with a fusion encoder."""
		encoder = getattr(self.jepas[level - 1], "encoder", None)
		return hasattr(encoder, "pixel_encoder") and hasattr(encoder, "proprio_encoder")

	def _get_level_input_key(self, level: int) -> str:
		"""Key of the previous level's output this level encodes: pixel_embed_{level-1}
		for fusion (proprio) levels, embed_{level-1} otherwise."""
		if self._level_uses_proprio(level):
			return f"pixel_embed_{level - 1}"
		return f"embed_{level - 1}"

	def _collect_level_info(self, info: dict, level: int) -> dict:
		"""Return the entries of info with a "_level{level}" suffix, with the suffix
		stripped (e.g. pixels_level1 -> pixels)."""
		suffix = f"_level{level}"
		level_info = {}
		for info_key, value in info.items():
			if info_key.endswith(suffix):
				level_info[info_key[: -len(suffix)]] = value
		return level_info

	def _add_level_target_aliases(self, info: dict, level: int) -> None:
		"""Keep suffixed target streams alongside unsuffixed encoder inputs."""
		for info_key, value in list(info.items()):
			if info_key.startswith("action"):
				continue
			if not torch.is_tensor(value) or value.ndim < 3:
				continue
			info[f"{info_key}_level{level}"] = value

	def _random_crop_level_outputs(
		self,
		output: dict,
		level: int,
		target_len: int,
		extra_keys: set[str] | None = None,
		start: int | None = None,
	) -> None:
		"""Crop this level's outputs in place to target_len time steps: embed/
		pixel_embed/proprio_embed/action_{level}, every "*_level{level}" key and
		extra_keys, all at the same offset (start, or a random one if None).
		Only tensors whose time length matches the level's are cropped."""
		target_len = int(target_len)
		if target_len <= 0:
			return

		level_keys = {
			f"embed_{level}",
			f"pixel_embed_{level}",
			f"proprio_embed_{level}",
			f"action_{level}",
		}
		ref_keys = [f"embed_{level}", f"action_{level}"]
		if extra_keys:
			level_keys.update(extra_keys)
			ref_keys.extend(sorted(extra_keys))

		ref = None
		for ref_key in ref_keys:
			value = output.get(ref_key)
			if torch.is_tensor(value) and value.ndim >= 3:
				ref = value
				break

		if ref is None or ref.size(1) <= target_len:
			return

		seq_len = ref.size(1)
		if start is None:
			start = self._draw_crop_start(seq_len, target_len, ref.device)
		indices = torch.arange(start, start + target_len, device=ref.device)
		suffix = f"_level{level}"

		for output_key, value in list(output.items()):
			if not torch.is_tensor(value) or value.ndim < 3:
				continue
			if value.size(1) != seq_len:
				continue
			if output_key in level_keys or output_key.endswith(suffix):
				output[output_key] = value.index_select(1, indices)

	def _draw_crop_start(self, seq_len: int, target_len: int, device) -> int:
		"""Uniformly draw the start of a target_len crop of a seq_len sequence."""
		return torch.randint(0, seq_len - target_len + 1, (), device=device).item()

	def _level_geometry(self, level: int) -> tuple[int, int]:
		"""(stride, window_size) with which this level chunks the level below."""
		jepa = self.jepas[level - 1]
		return int(getattr(jepa, "temporal_stride", 1)), int(getattr(jepa, "temporal_window_size", 1))

	def _draw_crops_and_level1_frames(self, seq_len: int, max_level: int, device):
		"""Draw every level's crop start up front and return the level-1 frames
		that any kept step at any level reads (directly or through the windows
		of the levels in between). Frames outside this set would be cropped away
		unseen, so the level-1 encoder can skip them."""
		lengths = {1: int(seq_len)}
		for level in range(2, max_level + 1):
			stride, window = self._level_geometry(level)
			lengths[level] = (lengths[level - 1] - window) // stride + 1

		crop_starts = {}
		kept = {}
		for level in range(1, max_level + 1):
			target_len = getattr(self.jepas[level - 1], "target_length", None)
			if target_len is None or int(target_len) <= 0 or lengths[level] <= int(target_len):
				crop_starts[level] = None
				kept[level] = set(range(lengths[level]))
				continue
			start = self._draw_crop_start(lengths[level], int(target_len), device)
			crop_starts[level] = start
			kept[level] = set(range(start, start + int(target_len)))

		needed = set(kept[max_level])
		for level in range(max_level, 1, -1):
			stride, window = self._level_geometry(level)
			needed = kept[level - 1] | {t * stride + k for t in needed for k in range(window)}
		return crop_starts, sorted(needed)

	def _encode_level1(self, info: dict, key: str, frame_indices) -> None:
		"""Run the level-1 encoder on the given frames (all of them when dense) and
		scatter the results into zero-filled full-length embed_1 / pixel_embed_1 /
		proprio_embed_1.
		Actions are encoded on the full stream (cheap, and higher-level pooling
		needs every chunk)."""
		ref = info[key]
		idx = torch.as_tensor(frame_indices, device=ref.device, dtype=torch.long)
		sparse_info = dict(info)
		sparse_info[key] = ref.index_select(1, idx)
		if "proprio" in info:
			sparse_info["proprio"] = info["proprio"].index_select(1, idx)

		level_out = self.jepas[0].encode(sparse_info, key=key)

		seq_len = ref.size(1)
		for output_name in ("embed", "pixel_embed", "proprio_embed"):
			sparse = level_out.get(f"{output_name}_0")
			if sparse is None:
				continue
			full = sparse.new_zeros(sparse.size(0), seq_len, *sparse.shape[2:])
			full[:, idx] = sparse
			info[f"{output_name}_1"] = full
		if "action_0" in level_out:
			info["action_1"] = level_out["action_0"]

	def _store_chunked_state_outputs(
		self,
		info: dict,
		level_out: dict,
		level: int,
	) -> None:
		"""Copy the target streams that JEPA.encode chunked to this level's time
		steps (level_out's "*_level{level-1}" keys) into info as "*_level{level}"."""
		ref = level_out.get("embed_0")
		if not torch.is_tensor(ref) or ref.ndim < 3:
			return

		current_len = ref.size(1)
		previous_level_suffix = f"_level{level - 1}"
		for info_key, value in level_out.items():
			if (
				info_key.startswith("action")
				or info_key.startswith("embed_")
				or info_key.startswith("pixel_embed_")
				or info_key.startswith("proprio_embed_")
				or info_key.startswith("proprio_input")
			):
				continue
			if not torch.is_tensor(value) or value.ndim < 3:
				continue

			if not info_key.endswith(previous_level_suffix):
				continue
			base_key = info_key[: -len(previous_level_suffix)]
			output_key = f"{base_key}_level{level}"

			if value.size(1) != current_len:
				raise ValueError(
					f"Cannot store '{output_key}': source '{info_key}' has "
					f"time length {value.size(1)}, but level {level} embeddings "
					f"have time length {current_len}."
				)
			info[output_key] = value

	def encode_hierarchical(self, info: dict, sparse_level1_encode: bool = True) -> dict:
		"""Encode the level-1 sequence ("pixels_level1", "action_level1") through
		all levels and randomly crop each level's outputs to its target length."""
		level_info = self._collect_level_info(info, 1)
		self._add_level_target_aliases(level_info, 1)
		level1_working_keys = set(level_info)

		seq_len = level_info["pixels"].size(1)
		if sparse_level1_encode:
			crop_starts, frame_indices = self._draw_crops_and_level1_frames(
				seq_len, self.num_levels, level_info["pixels"].device
			)
		else:
			crop_starts, frame_indices = {}, range(seq_len)
		self._encode_level1(level_info, "pixels", frame_indices)

		for level in range(2, self.num_levels + 1):
			jepa = self.jepas[level - 1]
			level_info["action"] = level_info[f"action_{level - 1}"]
			if self._level_uses_proprio(level):
				level_info[jepa.encoder.proprio_key] = level_info[f"proprio_embed_{level - 1}"]

			level_out = jepa.encode(
				level_info,
				key=self._get_level_input_key(level),
				chunk_temporal_inputs=True,
			)
			self._store_chunked_state_outputs(level_info, level_out, level)
			for output_name in ("embed", "pixel_embed", "proprio_embed"):
				if f"{output_name}_0" in level_out:
					level_info[f"{output_name}_{level}"] = level_out[f"{output_name}_0"]
			level_info[f"action_{level}"] = level_out["action_0"]

		output = dict(info)
		output.update(
			{
				output_key: value
				for output_key, value in level_info.items()
				if output_key not in level1_working_keys and output_key != "proprio_input"
			}
		)

		for level in range(1, self.num_levels + 1):
			target_len = getattr(self.jepas[level - 1], "target_length", None)
			if target_len is None:
				continue
			extra_keys = level1_working_keys if level == 1 else None
			self._random_crop_level_outputs(
				output,
				level=level,
				target_len=target_len,
				extra_keys=extra_keys,
				start=crop_starts.get(level),
			)

		return output


class LevelCostModel(nn.Module):
    """Cost model for one level: rolls out in the level's own latent space and scores the
    predictions against the goal in the goal's latent space.

    ``get_cost`` runs three stages:
      1. roll out this level's predictor from ``embed_0`` with the candidate actions;
      2. if the goal lives at a higher level, encode the predicted latents (and, for the
         action cost, the candidate actions) up to that level with the upper JEPA encoders.
         The goal is usually the next waypoint of the level above, compared at that level's
         stride. When every level in between was skipped at horizon 1 with
         ``horizon_one_goal_cost_space='upper'``, the goal is the real goal at the highest
         skipped level, and the predictions are encoded up the chain at stride 1 ("dense"), so
         every predicted step is scored;
      3. compare with ``goal_embed_0`` (``JEPA.criterion``), plus the weighted action cost.
    """

    def __init__(
        self,
        model: Any,
        upper_models: list[Any],
    ) -> None:
        super().__init__()
        self.model = model
        self.upper_models = upper_models
        self.level = self.model.level

    def rollout(self, info_dict: dict, action_candidates: torch.Tensor) -> dict:
        return self.model.rollout(info_dict, action_candidates)

    def get_latent_action_queue(self) -> torch.Tensor | None:
        """Forward latent-action queue access to the wrapped level model."""
        return self.model.get_latent_action_queue()

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        info_dict = self.rollout(info_dict, action_candidates)

        pred_emb = info_dict["predicted_embed_0"]
        use_upper_space = bool(info_dict.get('goal_embed_0_in_upper_space', True))
        goal_level = int(info_dict.get('goal_embed_0_level', self.level + 1))
        action_weight = float(info_dict.get("action_cost_weight", 0.0))
        pred_for_cost, action_for_cost = self._project_to_goal_space(
            pred_emb,
            action_candidates if action_weight != 0.0 else None,
            use_upper_space=use_upper_space,
            dense=bool(info_dict.get("dense_upper_goal_projection", False)),
            goal_level=goal_level,
        )

        cost_info = dict(info_dict)
        cost_info["predicted_embed_0"] = pred_for_cost
        cost = self.model.criterion(cost_info)
        if action_weight != 0.0:
            action_cost = self._action_cost(action_for_cost, cost_info)
            if action_cost is not None:
                cost = cost + action_weight * action_cost
        return cost

    def _project_to_goal_space(
        self,
        pred_emb: torch.Tensor,
        action_candidates: torch.Tensor | None,
        *,
        use_upper_space: bool,
        dense: bool,
        goal_level: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode predicted latents (and candidate actions) up to the goal's level."""
        if not self.upper_models or not use_upper_space:
            return pred_emb, None

        # Front-pad the predicted stream so that every window of the upper encoder is full.
        # Real-stride projection uses causal windows: pad window_size - 1. Dense (stride-1)
        # projection slides full windows over the stream: pad only when the stream is
        # shorter than one window. Actions get the same front pad, with zeros. With
        # window_size 1 (all paper models) the pad is 0.
        window_size = self.upper_models[0].temporal_window_size
        pad = max(0, window_size - pred_emb.shape[-2]) if dense else window_size - 1
        states = self._prefix_repeat_first(pred_emb, pad)

        pred_for_cost = self._encode_states_up(states, dense=dense, goal_level=goal_level)
        action_for_cost = None
        if action_candidates is not None:
            action_for_cost = self._encode_actions_up(
                states, action_candidates, pad=pad, dense=dense
            )
        return pred_for_cost, action_for_cost

    def _encode_states_up(
        self,
        states: torch.Tensor,
        *,
        dense: bool,
        goal_level: int,
    ) -> torch.Tensor:
        """Encode the padded predicted stream with the level above, then chain up to goal_level."""
        # None: each upper encoder uses its own temporal_stride.
        temporal_stride = 1 if dense else None
        b, s = states.shape[:2]
        input_key = f'embed_{self.level}'
        upper_out = self.upper_models[0].encode(
            {input_key: states.flatten(0, 1)},
            key=input_key,
            chunk_temporal_inputs=True,
            temporal_stride=temporal_stride,
        )
        for upper_level in range(self.level + 2, goal_level + 1):
            input_key = f'embed_{upper_level - 1}'
            upper_out = self.upper_models[upper_level - self.level - 1].encode(
                {input_key: upper_out['embed_0']},
                key=input_key,
                chunk_temporal_inputs=True,
                temporal_stride=temporal_stride,
            )
        return upper_out["embed_0"].unflatten(0, (b, s))

    def _encode_actions_up(
        self,
        states: torch.Tensor,
        action_candidates: torch.Tensor,
        *,
        pad: int,
        dense: bool,
    ) -> torch.Tensor:
        """Pool the candidate actions into the upper level's macro-actions."""
        upper_model = self.upper_models[0]
        b, s = states.shape[:2]
        # Embed, front-pad like the states, and pad the end to the state length so the
        # upper model chunks states and actions with the same windows.
        action_emb = self.model.action_embed(action_candidates)
        action_emb = self._prefix_zeros(action_emb, pad)
        action_emb = self._pad_zeros_to_length(action_emb, states.shape[-2])
        input_key = f'embed_{self.level}'
        chunked = upper_model._chunk_temporal_info(
            {input_key: states.flatten(0, 1), "action": action_emb.flatten(0, 1)},
            key=input_key,
            stride=1 if dense else upper_model.temporal_stride,
            window_size=upper_model.temporal_window_size,
        )
        pooled = upper_model.action_encoder(chunked["action"], chunked["action_mask"])

        # The final pooled action points beyond the final encoded upper state.
        return pooled.unflatten(0, (b, s))[..., :-1, :]

    def _action_cost(
        self,
        pred_action: torch.Tensor | None,
        info_dict: dict,
    ) -> torch.Tensor | None:
        target_action = info_dict.get("goal_action")
        if pred_action is None or target_action is None:
            return None

        cost_last_n = max(1, int(info_dict.get("cost_last_n", 1)))
        cost_window = min(cost_last_n, pred_action.shape[-2], target_action.shape[-2])
        pred_action = pred_action[..., -cost_window:, :]
        target_action = target_action[..., -cost_window:, :].expand_as(pred_action)

        cost = F.mse_loss(
            pred_action,
            target_action.detach(),
            reduction="none",
        )
        return cost.sum(dim=tuple(range(2, cost.ndim)))

    @staticmethod
    def _prefix_repeat_first(x: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0:
            return x

        prefix_shape = (*x.shape[:-2], count, x.shape[-1])
        prefix = x[..., :1, :].expand(prefix_shape)
        return torch.cat([prefix, x], dim=-2)

    @staticmethod
    def _prefix_zeros(x: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0:
            return x

        prefix = x.new_zeros(*x.shape[:-2], count, x.shape[-1])
        return torch.cat([prefix, x], dim=-2)

    @staticmethod
    def _pad_zeros_to_length(x: torch.Tensor, target_len: int) -> torch.Tensor:
        pad_len = target_len - x.shape[-2]
        if pad_len == 0:
            return x

        suffix = x.new_zeros(*x.shape[:-2], pad_len, x.shape[-1])
        return torch.cat([x, suffix], dim=-2)


def decreasing_horizon_schedule(
    *,
    num_calls: int,
    configured_horizon: int,
) -> list[int]:
    """Build a monotone decreasing per-call horizon schedule.

    When there are enough calls to represent every horizon value, each value
    greater than 1 receives the same number of calls and horizon 1 receives any
    remainder. If there are fewer calls than horizon values, keep the configured
    horizon as the first call and horizon 1 as the final call.
    """
    num_calls = int(num_calls)
    configured_horizon = int(configured_horizon)
    if num_calls <= 0:
        raise ValueError(
            f'Number of planning calls must be positive, got {num_calls}.'
        )
    if configured_horizon <= 0:
        raise ValueError(
            f'Planning horizon must be positive, got {configured_horizon}.'
        )
    if configured_horizon == 1:
        return [1] * num_calls
    if num_calls == 1:
        return [configured_horizon]
    if num_calls < configured_horizon:
        span = configured_horizon - 1
        denom = num_calls - 1
        return [
            max(1, math.ceil(1 + span * (num_calls - 1 - idx) / denom))
            for idx in range(num_calls)
        ]

    horizon_values = list(range(configured_horizon, 0, -1))
    calls_per_horizon = num_calls // len(horizon_values)
    extra_calls = num_calls % len(horizon_values)
    counts = [calls_per_horizon] * len(horizon_values)
    for idx in range(len(horizon_values) - extra_calls, len(horizon_values)):
        counts[idx] += 1

    schedule = []
    for horizon, count in zip(horizon_values, counts, strict=True):
        schedule.extend([horizon] * count)
    return schedule


def build_pusht_hjepa_l2():
    """The ``HJEPA`` of config ``pusht_hjepa_l2``, condensed from ``hjepa_utils.create_world_model``."""

    def mlp_projector():
        return build_projector({"type": "mlp"}, input_dim=192, output_dim=192)

    level1 = JEPA(
        encoder=ProjectedEncoder(create_hf_vit(size="tiny", patch_size=14, image_size=224), mlp_projector()),
        predictor=ProjectedPredictor(
            ARPredictor(
                num_frames=3, input_dim=192, output_dim=192, hidden_dim=192, depth=4, heads=8, mlp_dim=1024,
                dim_head=64, dropout=0.1, emb_dropout=0.0,
            ),
            mlp_projector(),
        ),
        action_embed=Embedder(input_dim=10, emb_dim=192),
        action_encoder=None,
        level=1,
        action_queue_size=0,
        temporal_stride=1,
        temporal_window_size=1,
        target_length=4,
    )
    level2 = JEPA(
        encoder=ProjectedEncoder(
            ResidualLatentMLP(input_dim=192, hidden_dim=384, output_dim=192, final_ln=False),
            build_projector({"type": "identity"}, input_dim=192, output_dim=192),
        ),
        predictor=ProjectedPredictor(
            ARPredictor(
                num_frames=3, input_dim=192, output_dim=192, hidden_dim=192, depth=6, heads=16, mlp_dim=2048,
                dim_head=64, dropout=0.1, emb_dropout=0.0,
            ),
            mlp_projector(),
        ),
        action_embed=Embedder(input_dim=4, emb_dim=192, patch_embed=False),
        action_encoder=SequenceEncoder(output_dim=4, input_dim=192, max_chunk=3),
        level=2,
        action_queue_size=4000,
        temporal_stride=2,
        temporal_window_size=1,
        target_length=4,
    )
    return HJEPA(jepas=[level1, level2])
