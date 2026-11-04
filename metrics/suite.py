"""Metric functions over ``Latents``.

Each metric is ``fn(lat, **params) -> dict``; the ones that also need the
model take a ``model`` argument. ``run`` evaluates a configured list of them.

Sources:
  rollout_consistency, state_probe   Gonzalez Laiz & Schneider, arXiv
                                     2610.37588 (Eq. 6, App. B and C.2)
  rank_spectrum, goal_rank,          EpiWM worldmodel/analysis/probes.py
  rollout_error                      (the-puzzler/epijepa@0d43616)
  surprise_ratios                    AC-MTM surprise_diagnostics.py
                                     (jackboyla/action-contrastive-jepa@43c96f6)
"""

import inspect

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from metrics import probes


def _frames(lat, min_t=0):
    """Window and position of every valid frame at or after ``min_t``, and
    whether its window is on the train side."""
    valid = lat.valid.numpy().copy()
    valid[:, :min_t] = False
    n, t = np.nonzero(valid)
    return n, t, lat.train.numpy()[n]


def _split(train, max_train, max_test, seed):
    """Index arrays of train and test frames, subsampled to the given caps."""
    rng = np.random.default_rng(seed)
    out = []
    for side, cap in ((train, max_train), (~train, max_test)):
        idx = np.nonzero(side)[0]
        if cap is not None and len(idx) > cap:
            idx = np.sort(rng.choice(idx, cap, replace=False))
        out.append(idx)
    return out


def _selected(lat, groups):
    return [i for i, g in enumerate(lat.target_group) if groups is None or g in groups]


def _state_features(lat, n, t, columns=None):
    """True state at frames (n, t): every target, angles as (cos, sin)."""
    y = lat.target.numpy()[n, t]
    columns = range(y.shape[1]) if columns is None else columns
    return np.concatenate(
        [probes.encode_target(y[:, j], lat.target_angle[j]) for j in columns], 1
    )


def rollout_consistency(lat, **_):
    """Latent rollout consistency LRC(k) for every horizon k.

    LRC(k) is the variance-weighted R^2 of predicted latents against encoded
    latents k predictor steps after the last context frame, pooled over
    windows. Persistence predicts the last context latent at every horizon.
    """
    z, zr, valid = lat.z.numpy(), lat.z_roll.numpy(), lat.valid.numpy()
    c = lat.context
    out = {"k": [], "lrc": [], "persistence": [], "n": []}
    for k in range(1, z.shape[1] - c + 1):
        m = valid[:, c - 1 + k]
        if m.sum() < 2:
            break
        target = z[m, c - 1 + k]
        out["k"].append(k)
        out["lrc"].append(float(r2_score(target, zr[m, c - 1 + k], multioutput="variance_weighted")))
        out["persistence"].append(float(r2_score(target, z[m, c - 1], multioutput="variance_weighted")))
        out["n"].append(int(m.sum()))
    return out


def rollout_error(lat, horizons=(1, 5), max_train=20000, seed=0, **_):
    """Open-loop rollout error at each horizon, in latent and probed-state space.

    A linear probe per target is fit on encoded latents of train frames and
    applied to predicted latents of test windows; ``floor`` applies it to the
    encoded latents at the same frames, the decoding error alone. ``norm``
    averages MAE / target std over the non-angle targets.
    """
    z, zr, valid = lat.z.numpy(), lat.z_roll.numpy(), lat.valid.numpy()
    target, train, c = lat.target.numpy(), lat.train.numpy(), lat.context
    n, t, side = _frames(lat)
    tr, _ = _split(side, max_train, 0, seed)
    horizons = [
        k for k in horizons
        if c - 1 + k < z.shape[1] and ((~train) & valid[:, c - 1 + k]).any()
    ]
    out = {"k": horizons, "latent_mse": [], "n": [], "norm": [], "norm_floor": [], "targets": {}}
    test = [(~train) & valid[:, c - 1 + k] for k in horizons]
    for k in horizons:
        m = valid[:, c - 1 + k]
        out["latent_mse"].append(float(((zr[m, c - 1 + k] - z[m, c - 1 + k]) ** 2).mean()))
    for j, name in enumerate(lat.target_names):
        angle = lat.target_angle[j]
        y = target[n, t, j]
        probe = probes.fit_probe("linear", z[n[tr], t[tr]], probes.encode_target(y[tr], angle))
        res = {"mae": [], "floor": []}
        for k, m in zip(horizons, test):
            yk = target[m, c - 1 + k, j]
            for key, src in (("mae", zr), ("floor", z)):
                p = probes.decode_target(probe.predict(src[m, c - 1 + k]), angle)
                res[key].append(probes.mae(p, yk, angle))
        out["targets"][name] = res
    scalar = [j for j, a in enumerate(lat.target_angle) if not a]
    sd = {j: target[n, t, j].std() for j in scalar}
    for i, m in enumerate(test):
        out["n"].append(int(m.sum()))
        for key, col in (("norm", "mae"), ("norm_floor", "floor")):
            out[key].append(float(np.mean([
                out["targets"][lat.target_names[j]][col][i] / sd[j] for j in scalar
            ])))
    return out


def state_probe(
    lat, groups=None, kinds=("linear", "mlp"), windows=(1,),
    max_train=20000, max_test=5000, seed=0, **_,
):
    """Decode each target from the latent with each probe kind and window.

    Window w reads [z_{t-w+1}; ...; z_t]. All windows use the same frames
    (t >= max(windows) - 1), so they differ only in the input.
    """
    z = lat.z.numpy()
    n, t, side = _frames(lat, max(windows) - 1)
    tr, te = _split(side, max_train, max_test, seed)
    state = _state_features(lat, n, t)
    out = {}
    for j in _selected(lat, groups):
        angle = lat.target_angle[j]
        y = lat.target.numpy()[n, t, j]
        control = probes.control_mae(state[tr], y[tr], state[te], y[te], angle, seed)
        res = {"control_mae": control, "n_train": len(tr), "n_test": len(te)}
        for w in windows:
            X = probes.windowed(z, n, t, w)
            for kind in kinds:
                res[f"{kind}_w{w}"] = probes.probe_errors(
                    kind, X[tr], y[tr], X[te], y[te], angle, control, seed
                )
        out[lat.target_names[j]] = res
    return out


def reverse_probe(
    lat, groups=("state",), kinds=("linear",), max_train=20000, max_test=5000, seed=0, **_,
):
    """Share of latent variance the true state explains: variance-weighted
    R^2 of the latent regressed on the targets in ``groups``."""
    z = lat.z.numpy()
    n, t, side = _frames(lat)
    tr, te = _split(side, max_train, max_test, seed)
    X = _state_features(lat, n, t, _selected(lat, groups))
    y = z[n, t]
    out = {}
    for kind in kinds:
        m = probes.fit_probe(kind, X[tr], y[tr], seed)
        out[kind] = float(r2_score(y[te], m.predict(X[te]), multioutput="variance_weighted"))
    return out


def effective_rank(Z):
    """exp of the entropy of the normalized singular values of centered Z."""
    Z = torch.as_tensor(Z, dtype=torch.float64)
    sv = torch.linalg.svdvals(Z - Z.mean(0))
    p = sv / sv.sum()
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum())), sv


def rank_spectrum(lat, frame=None, top=32, **_):
    """Effective rank, top-2 variance share and PCA spectrum of the latents
    of one window frame, or of every valid frame when ``frame`` is None."""
    if frame is None:
        m = lat.valid
        Z = lat.z[m]
    else:
        m = lat.valid[:, frame]
        Z = lat.z[m, frame]
    erank, sv = effective_rank(Z)
    ev = sv**2 / (sv**2).sum()
    return {
        "erank": erank,
        "top2_var": float(ev[:2].sum()),
        "spectrum": ev[:top].tolist(),
        "dim": int(lat.z.shape[-1]),
        "n": int(m.sum()),
    }


def goal_rank(lat, length=6, **_):
    """Spearman correlation between ||z_t - z_goal|| and steps to the goal,
    per non-overlapping chunk of ``length`` valid frames whose last frame is
    the goal, averaged over chunks."""
    z, n_valid = lat.z.numpy(), lat.valid.numpy().sum(1)
    steps = np.arange(length - 1, 0, -1)
    rho = []
    for i, nv in enumerate(n_valid):
        for s in range(0, nv - length + 1, length):
            d = np.linalg.norm(z[i, s : s + length - 1] - z[i, s + length - 1], axis=-1)
            rho.append(spearmanr(d, steps).correlation)
    return {"rho": float(np.nanmean(rho)), "n": len(rho)}


def shifted_permutation(size, rng):
    """A random permutation with no fixed points (a roll if none is found)."""
    if size <= 1:
        return torch.arange(size)
    index = np.arange(size)
    for _ in range(128):
        perm = rng.permutation(size)
        if not np.any(perm == index):
            break
    else:
        perm = np.roll(index, 1)
    return torch.as_tensor(perm, dtype=torch.long)


def _ratio_stats(num, den):
    r = num / (den + 1e-12)
    return {
        "mean": float(r.mean()),
        "median": float(np.median(r)),
        "p10": float(np.percentile(r, 10)),
        "p90": float(np.percentile(r, 90)),
        "frac_gt_normal": float((num > den).mean()),
    }


@torch.inference_mode()
def surprise_ratios(lat, model, n=4096, seed=42, batch_size=64, **_):
    """Teacher-forced one-step error on counterfactual vs recorded transitions.

    Clips of history + 1 valid frames are cut without overlap from each
    window. Within each batch a derangement swaps in another clip's actions
    (``counterfactual_action``) or target latents (``state_discontinuity``);
    each is reported as a per-clip ratio to the recorded transition's error.
    """
    h = lat.history_size
    device = next(model.parameters()).device
    n_valid = lat.valid.sum(1).tolist()
    clips = [(i, s) for i, nv in enumerate(n_valid) for s in range(0, nv - h, h + 1)]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(clips), min(n, len(clips)), replace=False)
    errs = {"normal": [], "counterfactual_action": [], "state_discontinuity": []}
    for b in range(0, len(pick), batch_size):
        idx = [clips[i] for i in pick[b : b + batch_size]]
        emb = torch.stack([lat.z[i, s : s + h + 1] for i, s in idx]).to(device)
        action = torch.stack([lat.action[i, s : s + h + 1] for i, s in idx]).to(device)
        ctx, target = emb[:, :h], emb[:, 1:]
        pred = model.predict(ctx, model.action_encoder(action)[:, :h])
        perm = shifted_permutation(len(idx), rng).to(device)
        wrong = model.predict(ctx, model.action_encoder(action[perm, :h]))
        errs["normal"].append((pred - target).pow(2).mean((1, 2)))
        errs["counterfactual_action"].append((wrong - target).pow(2).mean((1, 2)))
        errs["state_discontinuity"].append((pred - target[perm]).pow(2).mean((1, 2)))
    errs = {k: torch.cat(v).double().cpu().numpy() for k, v in errs.items()}
    out = {"n": len(pick), "normal_mse": float(errs["normal"].mean())}
    for k in ("counterfactual_action", "state_discontinuity"):
        out[k] = _ratio_stats(errs[k], errs["normal"])
    return out


METRICS = {
    f.__name__: f
    for f in (
        rollout_consistency, rollout_error, state_probe, reverse_probe,
        rank_spectrum, goal_rank, surprise_ratios,
    )
}


def run(lat, metrics, model=None):
    """Evaluate each configured metric.

    ``metrics`` is a list of ``{name, fn?, **params}``; ``fn`` picks the
    function (default: ``name``), so one function can run under several
    names with different params.
    """
    results = {}
    for entry in metrics:
        params = dict(entry)
        name = params.pop("name")
        fn = METRICS[params.pop("fn", name)]
        if "model" in inspect.signature(fn).parameters:
            params["model"] = model
        print(f"metric {name}", flush=True)
        results[name] = fn(lat, **params)
    return results
