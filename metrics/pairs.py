"""What a representation keys on: paired-input cosine and tag-colour probe.

A sample of frames is encoded twice with a corner tag (see ``tag.py``): once
with a colour of its own (``own``) and once with the colour of a partner frame
(``swap``). Three mean-centred cosines compare the two passes:

  same_tag      frame A against frame B stamped with A's colour (content differs)
  same_content  frame A against itself stamped with B's colour (tag differs)
  baseline      frame A against frame B, each with its own colour

An encoder that follows the scene reads high on same_content and near 0 on
same_tag; one that follows the tag reads the other way round. The colour probe
decodes the stamped colour from the latent and reports error relative to
chance, 100 being chance and 0 exact.

Pairing and cosines follow Enigma-Incorporated/The-Obsessed-Encoder@37819425
(``common/pair_metrics.py``, ``leworldmodel/additional_files/callbacks/
pair_analysis.py``). Their frames are freshly rendered; ours come from the
dataset, a sample the encoder may have seen in training, with a colour drawn
from the tag's hash.
"""

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F

from metrics import probes
from utils import get_img_preprocessor

REPS = ("backbone", "projection")
PAIRINGS = ("same_tag", "same_content", "baseline")


def cyclic_derangement(n, rng):
    """Partner of every position: one cycle over a random order, no fixed points."""
    order = rng.permutation(n)
    partner = np.empty(n, dtype=np.int64)
    partner[order] = np.roll(order, -1)
    return partner


def sample_frames(dataset, n, seed):
    """First frame of ``n`` random clips as (n, 3, H, W) uint8, without tag."""
    idx = np.random.default_rng(seed).choice(len(dataset), n, replace=False)
    return torch.stack([dataset[int(i)]["pixels"][0] for i in idx])


def stamp(frames, colors, size):
    """Copy of ``frames`` with square ``size`` in the corner set to ``colors`` (n, 3)."""
    out = frames.clone()
    out[:, :, :size, :size] = colors[:, :, None, None]
    return out


@torch.inference_mode()
def encode(model, frames, img_size, batch_size=128):
    """Backbone (CLS token) and projector output of uint8 ``frames``, on CPU."""
    device = next(model.parameters()).device
    preprocess = get_img_preprocessor("pixels", "pixels", img_size)
    out = {rep: [] for rep in REPS}
    for i in range(0, len(frames), batch_size):
        pixels = preprocess({"pixels": frames[i : i + batch_size]})["pixels"].to(device).float()
        cls = model.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        out["backbone"].append(cls.float().cpu())
        out["projection"].append(model.projector(cls).float().cpu())
    return {rep: torch.cat(v) for rep, v in out.items()}


def pair_cosines(own, swap, partner):
    """Mean cosine of each pairing, centred on the mean of the ``own`` pass."""
    n = len(partner)
    identity = np.arange(n)
    inverse = np.empty(n, dtype=np.int64)
    inverse[partner] = identity
    # swap row k carries the colour of partner(k), so colour k sits on swap row inverse[k]
    pairs = {
        "same_tag": (own, identity, swap, inverse),
        "same_content": (own, identity, swap, identity),
        "baseline": (own, identity, own, partner),
    }
    mu = own.mean(0)
    return {
        name: float(F.cosine_similarity(a[ai] - mu, b[bi] - mu, dim=1).mean())
        for name, (a, ai, b, bi) in pairs.items()
    }


def tag_probe(z, colors, kinds=("linear", "mlp"), train_fraction=0.8, seed=0):
    """Decode the stamped colour from latents ``z`` (n, d); mean over r, g, b."""
    colors = colors.numpy().astype(np.float64)
    order = np.random.default_rng(seed).permutation(len(z))
    cut = round(train_fraction * len(z))
    tr, te = order[:cut], order[cut:]
    z = z.numpy()
    controls = [
        probes.control_mae(colors[tr], colors[tr, c], colors[te], colors[te, c], False, seed)
        for c in range(3)
    ]
    out = {}
    for kind in kinds:
        res = [
            probes.probe_errors(kind, z[tr], colors[tr, c], z[te], colors[te, c], False, controls[c], seed)
            for c in range(3)
        ]
        out[kind] = {k: float(np.mean([r[k] for r in res])) for k in ("mae", "eps", "r2")}
    return out


def measure(model, frames, tag, img_size, kinds=("linear", "mlp"), train_fraction=0.8, seed=0):
    """Pair cosines and colour probe of ``model`` on ``frames`` stamped by ``tag``."""
    n = len(frames)
    colors = torch.stack([tag.color(j) for j in range(n)])
    partner = cyclic_derangement(n, np.random.default_rng(seed))
    own = encode(model, stamp(frames, colors, tag.size), img_size)
    swap = encode(model, stamp(frames, colors[partner], tag.size), img_size)
    return {
        "pair": {rep: pair_cosines(own[rep], swap[rep], partner) for rep in REPS},
        "tag_probe": {
            rep: tag_probe(own[rep], colors, kinds, train_fraction, seed) for rep in REPS
        },
    }


class PairCallback(pl.Callback):
    """Logs ``pair/{rep}/{pairing}`` every ``every_n_steps`` optimizer steps."""

    def __init__(self, frames, tag, img_size, every_n_steps, seed=0):
        self.frames = frames
        self.tag = tag
        self.img_size = img_size
        self.every = every_n_steps
        self.seed = seed

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == 0 or step % self.every or not trainer.is_global_zero:
            return
        n = len(self.frames)
        colors = torch.stack([self.tag.color(j) for j in range(n)])
        partner = cyclic_derangement(n, np.random.default_rng(self.seed))
        was_training = pl_module.training
        pl_module.eval()
        try:
            own = encode(pl_module.model, stamp(self.frames, colors, self.tag.size), self.img_size)
            swap = encode(pl_module.model, stamp(self.frames, colors[partner], self.tag.size), self.img_size)
        finally:
            pl_module.train(was_training)
        metrics = {
            f"pair/{rep}/{name}": value
            for rep in REPS
            for name, value in pair_cosines(own[rep], swap[rep], partner).items()
        }
        for logger in trainer.loggers:
            logger.log_metrics(metrics, step=step)
