"""Train and evaluate one moving-dot run.

    python -m toy.dot.train objective=grounded noise=fixed alpha=1.5 beta=1 lr=1e-3 seed=1 subdir=NAME

Arguments are ``key=value`` overrides of DEFAULTS. The run directory
$STABLEWM_HOME/checkpoints/<subdir> receives config.yaml, metrics.csv (one row
per epoch), weights.pt and result.json with the probe RMSE and the weights' sha256.
"""

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from module import SIGReg
from toy.dot import data, probe
from toy.dot.model import DotModel
from toy.dot.objectives import OBJECTIVES, compute

DEFAULTS = {
    "objective": "sigreg",  # one of OBJECTIVES
    "noise": "none",  # none | fixed | changing
    "alpha": 0.0,
    "seed": 1,
    "lr": 1e-3,
    "beta": 1.0,  # weight of the pixel term in the grounded objective
    "lam": 1.0,  # weight of the latent term in the latent-augmented twin
    "epochs": 10,
    "batch_size": 500,
    "train_size": data.SPLIT_SIZES["train"],
    "probe_size": data.SPLIT_SIZES["probe"],
    "val_size": data.SPLIT_SIZES["val"],
    "eval_size": data.SPLIT_SIZES["eval"],
    "grad_share_every": 50,
    "subdir": None,
    "output_model_name": None,
}


def run_dir(cfg):
    home = Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))
    return home / "checkpoints" / (cfg.subdir or "dot")


def loader(cfg, split, device, size, **kw):
    return data.batches(split, cfg.noise, cfg.alpha, batch_size=cfg.batch_size, size=size, device=device, **kw)


def validate(model, cfg, objective, sigreg, device):
    """Mean of each loss term over the validation split."""
    sums, n = {}, 0
    with torch.no_grad():
        for batch in loader(cfg, "val", device, cfg.val_size):
            for k, v in compute(model, batch, objective, sigreg, cfg.beta, cfg.lam).items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            n += 1
    return {f"val/{k}": v / n for k, v in sums.items()}


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def run(cfg):
    assert cfg.objective in OBJECTIVES and cfg.noise in data.NOISES
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.merge(cfg, {"loss": {"beta": cfg.beta, "lam": cfg.lam}}), out / "config.yaml")

    model = DotModel(
        decoder=cfg.objective in ("pixel", "grounded", "latent_twin"), position_head=cfg.objective == "supervised"
    ).to(device)
    sigreg = SIGReg(knots=17, num_proj=1024).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    rows, step = [], 0
    for epoch in range(cfg.epochs):
        model.train()
        sums, shares, n = {}, {"latent": [], "pixel": []}, 0
        for batch in loader(cfg, "train", device, cfg.train_size, seed=cfg.seed, epoch=epoch, shuffle=True):
            measure = step % cfg.grad_share_every == 0
            terms = compute(model, batch, cfg.objective, sigreg, cfg.beta, cfg.lam, shares=measure)
            opt.zero_grad(set_to_none=True)
            terms["loss"].backward()
            opt.step()
            for k in shares:
                if f"{k}_grad_share" in terms:
                    shares[k].append(float(terms[f"{k}_grad_share"]))
            for k, v in terms.items():
                if "loss" in k:
                    sums[k] = sums.get(k, 0.0) + float(v.detach())
            n += 1
            step += 1
        row = {"epoch": epoch, "step": step, **{f"train/{k}": v / n for k, v in sums.items()}}
        for k, v in shares.items():
            if v:
                row[f"train/{k}_grad_share"] = sum(v) / len(v)
                row[f"train/{k}_grad_share_min"] = min(v)
        model.eval()
        rows.append({**row, **validate(model, cfg, cfg.objective, sigreg, device)})
        write_csv(out / "metrics.csv", rows)
        if not all(torch.isfinite(torch.tensor(v)) for v in rows[-1].values()):
            raise RuntimeError(f"non-finite loss in epoch {epoch}")

    model.eval()
    with torch.no_grad():
        features = lambda batch: model.rollout(batch["frames"], batch["actions"])
        weights = probe.fit_probe(features, loader(cfg, "probe", device, cfg.probe_size))
        val = probe.rmse(features, weights, loader(cfg, "val", device, cfg.val_size))
        test = probe.rmse(features, weights, loader(cfg, "eval", device, cfg.eval_size))
    torch.save(model.state_dict(), out / "weights.pt")
    digest = hashlib.sha256((out / "weights.pt").read_bytes()).hexdigest()
    result = {"val_rmse": float(val.mean()), "rmse": float(test.mean()), "rmse_by_step": test.tolist(), "weights_sha256": digest}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "rmse_by_step"}))
    return result


def main(argv):
    cfg = OmegaConf.merge(OmegaConf.create(DEFAULTS), OmegaConf.from_dotlist(argv))
    run(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
