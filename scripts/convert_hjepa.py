"""Convert a released H-JEPA Push-T ``*_object.ckpt`` into a run ``eval.py`` can load.

Usage, from the repository root:

    python -m scripts.convert_hjepa CKPT RUN

writes ``checkpoints/RUN/weights.pt`` and ``config.json``; plan with ``policy=RUN/weights.pt``.
The released model is a pickle of H-JEPA's classes. They are read as plain modules, so
their code is not needed, and renamed to ``hjepa.HJEPA``'s keys. The checkpoint was saved
with transformers 5, whose ViT names differ from the 4.57 ones used here.

Weights from kevinghst/H-JEPA@97054ec (MIT, Copyright (c) 2026 Wancong Zhang, Basile Terver,
Mike Rabbat, Yann LeCun, Randall Balestriero).
"""

import pickle
import re
import sys
import types
from pathlib import Path

import torch
from torch import nn


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
    import stable_pretraining  # noqa: F401, registers the eval resolver the configs use
    import stable_worldmodel as swm
    from hydra.utils import instantiate
    from omegaconf import open_dict

    with hydra.initialize_config_dir(str(Path(__file__).parent.parent / "config" / "train"), version_base=None):
        cfg = hydra.compose("lewm", overrides=["model=hjepa"])
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * 2
    model = instantiate(cfg.model)
    model.load_state_dict(load_released(path))
    swm.wm.utils.save_pretrained(model, run_name=run_name, config=cfg.model, filename="weights.pt")


if __name__ == "__main__":
    convert(*sys.argv[1:3])
