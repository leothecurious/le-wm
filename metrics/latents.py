"""Latents of evaluation windows, the only place the suite runs the encoder
and the rollout."""

from dataclasses import asdict, dataclass

import torch

PRECISION = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass
class Latents:
    """Per-window latents and labels.

    z        (N, T, D)  encoded frames, the projector output the planner uses
    z_roll   (N, T, D)  first ``context`` entries encoded, the rest open-loop
                        predictions from recorded actions; horizon k sits at
                        index ``context - 1 + k``
    action   (N, T, A)  normalized actions, action t follows frame t
    valid    (N, T)     real (unpadded) frames, always a prefix
    target   (N, T, M)  probe targets in dataset units
    episode  (N,)       dataset episode of each window
    train    (N,)       probe-train side of the episode split
    """

    z: torch.Tensor
    z_roll: torch.Tensor
    action: torch.Tensor
    valid: torch.Tensor
    target: torch.Tensor
    episode: torch.Tensor
    train: torch.Tensor
    target_names: list
    target_angle: list
    target_group: list
    context: int
    history_size: int

    def save(self, path):
        torch.save(asdict(self), path)

    @classmethod
    def load(cls, path):
        return cls(**torch.load(path, weights_only=False))


@torch.inference_mode()
def extract(model, loader, context, history_size, precision="fp32"):
    """Encode every frame of each window and roll out from its first
    ``context`` frames with the recorded actions.

    Returns z, z_roll, action and valid; rollout steps past a window's last
    valid frame are computed on padding and must stay masked.
    """
    device = next(model.parameters()).device
    dtype = PRECISION[precision]
    out = {"z": [], "z_roll": [], "action": [], "valid": []}
    for batch in loader:
        pixels = batch["pixels"].to(device, non_blocking=True)
        action = batch["action"].to(device, non_blocking=True)
        num_frames = pixels.size(1)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            z = model.encode({"pixels": pixels})["emb"]
            info = {"pixels": pixels[:, None, :context]}
            roll = model.rollout(info, action[:, None, : num_frames - 1], history_size)
        out["z"].append(z.float().cpu())
        out["z_roll"].append(roll["predicted_emb"][:, 0].float().cpu())
        out["action"].append(batch["action"])
        out["valid"].append(batch["valid"])
    return {k: torch.cat(v) for k, v in out.items()}
