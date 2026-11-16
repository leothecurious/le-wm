"""Metric functions over ``Latents``.

Each metric is ``fn(lat, **params) -> dict``. ``run`` evaluates a configured
list of them.

Sources:
  state_probe   Gonzalez Laiz & Schneider, arXiv 2610.37588 (App. B and C.2)
  rank_spectrum EpiWM worldmodel/analysis/probes.py (the-puzzler/epijepa@0d43616)
"""

import numpy as np
import torch
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


def rollout_probe(lat, max_train=None, max_test=None, seed=0, **_):
    """Joint linear position readout shared across all open-loop timesteps.

    Dot coordinates are fractions of the image side length. The paper-style
    summary averages timestep Euclidean RMSEs, rather than pooling frames.
    """
    if lat.predicted_z is None:
        raise ValueError("rollout_probe requires predicted_z from rollout extraction")
    if any(lat.target_names.count(name) != 1 for name in ("dot_x", "dot_y")):
        raise ValueError("rollout_probe requires unique dot_x and dot_y targets")
    columns = [lat.target_names.index(name) for name in ("dot_x", "dot_y")]
    if any(lat.target_angle[j] for j in columns):
        raise ValueError("dot_x and dot_y must be non-angular coordinates")
    if lat.predicted_z.shape[:2] != lat.valid.shape:
        raise ValueError("predicted_z must align with the valid frame mask")
    n, t, side = _frames(lat)
    tr, te = _split(side, max_train, max_test, seed)
    if not len(tr) or not len(te):
        raise ValueError("rollout_probe requires valid train and test frames")
    X = lat.predicted_z.numpy()[n, t]
    y = lat.target.numpy()[n, t][:, columns]
    probe = probes.fit_probe("linear", X[tr], y[tr], seed)
    squared_error = ((probe.predict(X[te]) - y[te]) ** 2).sum(axis=1)
    center_squared_error = ((0.5 - y[te]) ** 2).sum(axis=1)
    test_t = t[te]
    counts = [int((test_t == step).sum()) for step in range(lat.valid.shape[1])]

    def summarize(errors):
        per_step = [
            float(np.sqrt(errors[test_t == step].mean())) if count else None
            for step, count in enumerate(counts)
        ]
        return {
            "rmse_per_step": per_step,
            "mean_rmse": float(np.mean([value for value in per_step if value is not None])),
            "pooled_rmse": float(np.sqrt(errors.mean())),
        }

    return {
        "units": "coordinate side length",
        "n_train": len(tr),
        "n_test": len(te),
        "n_test_per_step": counts,
        **summarize(squared_error),
        "center_control": summarize(center_squared_error),
    }


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


METRICS = {
    f.__name__: f
    for f in (state_probe, rollout_probe, reverse_probe, rank_spectrum)
}


def run(lat, metrics):
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
        print(f"metric {name}", flush=True)
        results[name] = fn(lat, **params)
    return results
