"""Job files for the moving-dot runs, one ``<name> <command>`` line per run for scripts/gpu_lane.sh.

    python -m toy.dot.jobs lr > lr.jobs        # learning rates, clean frames, seed 1
    python -m toy.dot.jobs weights --lr sigreg=A,pixel=B > weights.jobs
    python -m toy.dot.jobs grid --lr sigreg=A,pixel=B,grounded=C,latent_twin=D,supervised=E \\
        --beta X --lam Y > grid.jobs

``lr`` runs all five objectives at each learning rate, with beta and lam at 1. Choose each
objective's learning rate by validation probe RMSE (``val_rmse`` in result.json). ``weights``
sweeps beta (grounded) and lam (latent_twin) at the learning rates of sigreg and pixel; pick
them with scripts/pick_beta.py. ``grid`` is every objective on clean frames and on both noise
schedules at each alpha, seeds 1 and 2, seed 1 first.
"""

import argparse

from toy.dot.objectives import OBJECTIVES

LRS = (1e-4, 3e-4, 1e-3, 3e-3)
WEIGHTS = (0.1, 1, 10)
ALPHAS = (0.5, 1.5, 3)
SCHEDULES = ("fixed", "changing")
SEEDS = (1, 2)


def line(name, **kw):
    args = " ".join(f"{k}={v}" for k, v in kw.items())
    return f"{name} scripts/run.sh {name} '-m toy.dot.train' {args}"


def settings():
    yield "none", 0
    for schedule in SCHEDULES:
        for alpha in ALPHAS:
            yield schedule, alpha


def lr_jobs():
    for objective in OBJECTIVES:
        for lr in LRS:
            yield line(f"dot_lr_{objective}_{lr:g}", objective=objective, lr=lr, seed=1)


def weight_jobs(lr):
    for w in WEIGHTS:
        yield line(f"dot_beta_{w:g}", objective="grounded", beta=w, lr=lr["sigreg"], seed=1)
    for w in WEIGHTS:
        yield line(f"dot_lam_{w:g}", objective="latent_twin", lam=w, lr=lr["pixel"], seed=1)


def grid_jobs(lr, beta, lam):
    for seed in SEEDS:
        for objective in OBJECTIVES:
            for noise, alpha in settings():
                name = f"dot_{objective}_{noise}{alpha:g}_seed{seed}"
                yield line(name, objective=objective, noise=noise, alpha=alpha, lr=lr[objective], beta=beta, lam=lam, seed=seed)


def parse_lr(text):
    return {k: float(v) for k, v in (item.split("=") for item in text.split(","))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("lr", "weights", "grid"))
    parser.add_argument("--lr", type=parse_lr)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--lam", type=float)
    args = parser.parse_args()
    if args.stage == "lr":
        jobs = lr_jobs()
    elif args.stage == "weights":
        jobs = weight_jobs(args.lr)
    else:
        jobs = grid_jobs(args.lr, args.beta, args.lam)
    print("\n".join(jobs))


if __name__ == "__main__":
    main()
