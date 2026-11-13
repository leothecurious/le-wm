"""Latents of evaluation windows, the only place the suite runs the encoder."""

from dataclasses import dataclass

import torch

PRECISION = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass
class Latents:
    """Per-window latents and labels.

    z        (N, T, D)  encoded frames, the projector output the planner uses
    action   (N, T, A)  normalized actions, action t follows frame t
    valid    (N, T)     real (unpadded) frames, always a prefix
    target   (N, T, M)  probe targets in dataset units
    episode  (N,)       dataset episode of each window
    train    (N,)       probe-train side of the episode split
    """

    z: torch.Tensor
    action: torch.Tensor
    valid: torch.Tensor
    target: torch.Tensor
    episode: torch.Tensor
    train: torch.Tensor
    target_names: list
    target_angle: list
    target_group: list

    def save(self, path):
        torch.save(vars(self), path)

    @classmethod
    def load(cls, path):
        return cls(**torch.load(path, weights_only=False))


@torch.inference_mode()
def extract(model, loader, precision="fp32"):
    """Encode every frame of each window. Returns z, action and valid."""
    device = next(model.parameters()).device
    dtype = PRECISION[precision]
    out = {"z": [], "action": [], "valid": []}
    for batch in loader:
        pixels = batch["pixels"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            z = model.encode({"pixels": pixels})["emb"]
        out["z"].append(z.float().cpu())
        out["action"].append(batch["action"])
        out["valid"].append(batch["valid"])
    return {k: torch.cat(v) for k, v in out.items()}
