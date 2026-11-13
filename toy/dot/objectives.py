"""Training objectives on the moving-dot model.

All errors are means over elements, as in LeWM's loss, so beta and lam weigh
terms of the same scale as the Push-T objectives.
"""

import torch
import torch.nn.functional as F

from grounding import grad_shares

OBJECTIVES = ("sigreg", "pixel", "grounded", "latent_twin", "supervised")
SIGREG_WEIGHT = 0.09  # LeWM's


def compute(model, batch, objective, sigreg=None, beta=0.0, lam=0.0, shares=False):
    """Loss terms for ``objective`` on ``batch``.

    Returns a dict with ``loss`` and its terms (``pred_loss`` and ``copy_loss``
    always; ``pixel_loss``, ``sigreg_loss`` and ``position_loss`` where the objective has
    them). With ``shares`` the gradient-norm share of the latent and pixel terms at the
    encoder output is added for the objectives that have both.
    """
    frames, actions = batch["frames"], batch["actions"]
    s = model.encode(frames)  # (B, T, D)
    pred = model.predict(s[:, 0], actions)  # s~_2..s~_T
    target = s[:, 1:]
    pred_loss = F.mse_loss(pred, target.detach() if objective == "latent_twin" else target)
    with torch.no_grad():
        out = {"pred_loss": pred_loss.detach(), "copy_loss": F.mse_loss(s[:, :-1], target)}

    terms = {}
    if objective == "sigreg":
        out["sigreg_loss"] = sigreg(s.transpose(0, 1))
        out["loss"] = pred_loss + SIGREG_WEIGHT * out["sigreg_loss"]
    elif objective == "grounded":
        out["sigreg_loss"] = sigreg(s.transpose(0, 1))
        out["pixel_loss"] = F.mse_loss(model.decode(pred), frames[:, 1:])
        out["loss"] = pred_loss + SIGREG_WEIGHT * out["sigreg_loss"] + beta * out["pixel_loss"]
        terms = {"latent": pred_loss, "pixel": beta * out["pixel_loss"]}
    elif objective == "pixel":
        out["pixel_loss"] = F.mse_loss(model.decode(pred), frames[:, 1:])
        out["loss"] = out["pixel_loss"]
    elif objective == "latent_twin":
        out["pixel_loss"] = F.mse_loss(model.decode(pred), frames[:, 1:])
        out["loss"] = out["pixel_loss"] + lam * pred_loss
        terms = {"latent": lam * pred_loss, "pixel": out["pixel_loss"]}
    elif objective == "supervised":
        states = torch.cat([s[:, :1], pred], dim=1)
        out["position_loss"] = (model.head(states) - batch["pos"]).square().sum(-1).mean()
        out["loss"] = out["position_loss"]
    else:
        raise ValueError(objective)

    if shares and terms:
        out.update({f"{k}_grad_share": v for k, v in grad_shares(terms, s).items()})
    return out
