"""Training objectives on the moving-dot model.

All errors are means over elements, as in LeWM's loss, so beta and lam weigh
terms of the same scale as the Push-T objectives.
"""

import torch
import torch.nn.functional as F

from module import grad_shares

OBJECTIVES = ("sigreg", "pixel", "grounded", "latent_twin", "supervised")
SIGREG_WEIGHT = 0.09  # LeWM's


def compute(model, batch, objective, sigreg=None, beta=0.0, lam=0.0, shares=False):
    """``loss`` and its terms for ``objective``; ``pred_loss`` and ``copy_loss`` always.
    With ``shares``, objectives with both a latent and a pixel term add each one's
    share of the gradient norm at the encoder output."""
    frames, actions = batch["frames"], batch["actions"]
    s = model.encode(frames)  # (B, T, D)
    pred = model.predict(s[:, 0], actions)  # s~_2..s~_T
    target = s[:, 1:]
    pred_loss = F.mse_loss(pred, target.detach() if objective == "latent_twin" else target)
    with torch.no_grad():
        out = {"pred_loss": pred_loss.detach(), "copy_loss": F.mse_loss(s[:, :-1], target)}

    if model.decoder is not None:
        out["pixel_loss"] = F.mse_loss(model.decode(pred), frames[:, 1:])
    if objective in ("sigreg", "grounded"):
        out["sigreg_loss"] = sigreg(s.transpose(0, 1))

    terms = {}
    if objective == "sigreg":
        out["loss"] = pred_loss + SIGREG_WEIGHT * out["sigreg_loss"]
    elif objective == "grounded":
        terms = {"latent": pred_loss, "pixel": beta * out["pixel_loss"]}
        out["loss"] = terms["latent"] + terms["pixel"] + SIGREG_WEIGHT * out["sigreg_loss"]
    elif objective == "pixel":
        out["loss"] = out["pixel_loss"]
    elif objective == "latent_twin":
        terms = {"latent": lam * pred_loss, "pixel": out["pixel_loss"]}
        out["loss"] = terms["latent"] + terms["pixel"]
    elif objective == "supervised":
        states = torch.cat([s[:, :1], pred], dim=1)
        out["position_loss"] = (model.head(states) - batch["pos"]).square().sum(-1).mean()
        out["loss"] = out["position_loss"]
    else:
        raise ValueError(objective)

    if shares and terms:
        out.update({f"{k}_grad_share": v for k, v in grad_shares(terms, s).items()})
    return out
