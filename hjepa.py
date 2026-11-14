"""H-JEPA's Push-T model: a two-level temporal hierarchy on top of the LeWM encoder.

Level 1 is the LeWM ``JEPA`` itself (``encoder``, ``projector``, ``predictor``,
``pred_proj``, ``action_encoder``), so code that reads the bottom level keeps
working. Level 2 (``upper``) encodes every ``stride``-th level-1 latent and
predicts at that stride, conditioned on a latent action pooled from the
level-1 action embeddings. ``levels`` returns the latents of both levels.

Ported from Zhang et al. 2026, "H-JEPA: End-to-End Learning of Hierarchical World
Models for Visual Planning" (arXiv:2610.06805), kevinghst/H-JEPA@97054ec (MIT),
config ``pusht_hjepa_l2``. Run ``python hjepa.py CKPT RUN`` to convert a
released ``*_object.ckpt`` into a run loadable by ``load_pretrained(RUN/weights.pt)``.

MIT License, Copyright (c) 2026 Wancong Zhang, Basile Terver, Mike Rabbat, Yann LeCun, Randall Balestriero
(tests/reference/LICENSE).
"""

import pickle
import re
import sys
import types
from pathlib import Path

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
    """LeWM ``JEPA`` as level 1 plus one ``UpperLevel``.

    Training clips span ``window`` level-1 frames, enough for ``history_size + num_preds``
    level-2 steps. ``encode`` then keeps a random ``history_size + num_preds`` frames of
    level 1 for its loss and leaves the level-2 latents in ``info``; ``upper_loss`` scores them.
    Other inputs, as in planning, go through the LeWM ``encode``.
    """

    def __init__(
        self, encoder, predictor, action_encoder, upper, projector=None, pred_proj=None,
        history_size=3, num_preds=1, upper_lr=1e-5,
    ):
        super().__init__(encoder, predictor, action_encoder, projector, pred_proj)
        assert num_preds == 1, "the level-2 loss predicts one step"
        self.upper = upper
        self.history_size = history_size
        self.keep = history_size + num_preds
        self.window = (self.keep - 1) * upper.stride + 1
        self.upper_lr = upper_lr

    def encode(self, info):
        if info["pixels"].size(1) != self.window:
            return super().encode(info)

        pixels = info["pixels"].float()
        B, T = pixels.shape[:2]
        start = torch.randint(0, T - self.keep + 1, (), device=pixels.device).item()
        # Only the frames level 1 keeps and level 2 reads are encoded; the rest are never used.
        frames = sorted(set(range(start, start + self.keep)) | set(range(0, T, self.upper.stride)))
        idx = torch.as_tensor(frames, device=pixels.device)
        cls = self.encoder(rearrange(pixels[:, idx], "b t ... -> (b t) ..."), interpolate_pos_encoding=True)
        emb = rearrange(self.projector(cls.last_hidden_state[:, 0]), "(b t) d -> b t d", b=B)
        full = emb.new_zeros(B, T, emb.size(-1))
        full[:, idx] = emb

        act_emb = self.action_encoder(info["action"])
        info["emb"] = full[:, start : start + self.keep]
        info["act_emb"] = act_emb[:, start : start + self.keep]
        info["upper_emb"] = self.upper.embed(full)
        info["upper_act_emb"] = self.upper.action_encoder(self.upper.pool(act_emb))
        return info

    def upper_loss(self, info, sigreg, weight):
        """Level-2 prediction plus SIGReg loss on the latents ``encode`` left in ``info``."""
        emb, act_emb = info["upper_emb"], info["upper_act_emb"]
        pred = self.upper.predict(emb[:, : self.history_size], act_emb[:, : self.history_size])
        info["pred_loss_upper"] = (pred - emb[:, 1:]).pow(2).mean()
        info["sigreg_loss_upper"] = sigreg(emb.transpose(0, 1))
        return info["pred_loss_upper"] + weight * info["sigreg_loss_upper"]

    @torch.no_grad()
    def levels(self, pixels):
        """Latents of every level for ``pixels`` (B, T, C, H, W): ``[level 1 (B, T, D), level 2 (B, ceil(T / stride), D)]``."""
        info = JEPA.encode(self, {"pixels": pixels})
        return [info["emb"], self.upper.embed(info["emb"])]

    def param_groups(self, lr):
        upper = list(self.upper.parameters())
        ids = {id(p) for p in upper}
        return [
            {"params": [p for p in self.parameters() if id(p) not in ids], "lr": lr},
            {"params": upper, "lr": self.upper_lr},
        ]


def optimizer_factory(model, cfg):
    """Optimizer for ``spt.Module`` with level 1 at ``cfg['lr']`` and level 2 at ``model.upper_lr``."""
    import stable_pretraining as spt

    return lambda _params: spt.optim.create_optimizer(model.param_groups(cfg["lr"]), cfg)


# --- released checkpoints -------------------------------------------------------------------


class _Stub(nn.Module):
    pass


class _StubUnpickler(pickle.Unpickler):
    """Rebuilds H-JEPA's pickled model as plain modules, so their code is not needed to read the weights."""

    def find_class(self, module, name):
        module = "builtins" if module == "__builtin__" else module
        if module.split(".")[0] in ("torch", "builtins", "collections", "_codecs", "numpy", "copyreg"):
            return super().find_class(module, name)
        cls = type(name, (_Stub,), {})
        cls.__module__ = module
        return cls


_pickle_shim = types.ModuleType("stub_pickle")
vars(_pickle_shim).update({k: getattr(pickle, k) for k in dir(pickle)})
_pickle_shim.Unpickler = _StubUnpickler

_QKV = {"q": "query", "k": "key", "v": "value"}
_VIT_RENAMES = [
    (r"\.attention\.([qkv])_proj\.", lambda m: f".attention.attention.{_QKV[m[1]]}."),
    (r"\.attention\.o_proj\.", lambda m: ".attention.output.dense."),
    (r"\.mlp\.fc1\.", lambda m: ".intermediate.dense."),
    (r"\.mlp\.fc2\.", lambda m: ".output.dense."),
]
_PREFIXES = [
    ("jepas.0.encoder.encoder.", "encoder."),
    ("jepas.0.encoder.projector.", "projector."),
    ("jepas.0.predictor.predictor.", "predictor."),
    ("jepas.0.predictor.projector.", "pred_proj."),
    ("jepas.0.action_embed.", "action_encoder."),
    ("jepas.1.latent_action_queue", "upper.latent_action_queue"),
    ("jepas.1.encoder.encoder.", "upper.encoder."),
    ("jepas.1.predictor.predictor.", "upper.predictor."),
    ("jepas.1.predictor.projector.", "upper.pred_proj."),
    ("jepas.1.action_embed.", "upper.action_encoder."),
    ("jepas.1.action_encoder.", "upper.action_pool."),
]


def from_released(state):
    """Rename a released H-JEPA state dict (``HJEPA.jepas.N...``, transformers 5 ViT keys) to ``HJEPA``'s."""
    out = {}
    for key, value in state.items():
        for old, new in _PREFIXES:
            if key.startswith(old):
                key = new + key[len(old) :]
                break
        else:
            raise KeyError(key)
        if key.startswith("encoder."):
            key = key.replace("encoder.layers.", "encoder.encoder.layer.")
            for pattern, repl in _VIT_RENAMES:
                key = re.sub(pattern, repl, key)
        if key == "encoder.embeddings.mask_token":  # unused, our encoder is built without it
            continue
        out[key] = value
    return out


def load_released(path):
    """State dict of a released ``*_object.ckpt`` under ``HJEPA``'s names."""
    model = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_pickle_shim)
    return from_released(model.state_dict())


def convert(path, run_name):
    """Save a released checkpoint as ``checkpoints/<run_name>/weights.pt`` with its ``config.json``."""
    import hydra
    import stable_worldmodel as swm
    from hydra.utils import instantiate
    from omegaconf import open_dict

    with hydra.initialize_config_dir(str(Path(__file__).parent / "config" / "train"), version_base=None):
        cfg = hydra.compose("lewm", overrides=["model=hjepa"])
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * 2
    model = instantiate(cfg.model)
    model.load_state_dict(load_released(path))
    swm.wm.utils.save_pretrained(model, run_name=run_name, config=cfg.model, filename="weights.pt")


if __name__ == "__main__":
    convert(*sys.argv[1:3])
