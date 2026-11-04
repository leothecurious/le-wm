"""The probe protocol of Gonzalez Laiz & Schneider (arXiv 2610.37588,
App. C.2), shared by every metric that decodes a target.

Inputs are standardized and targets stay in dataset units. Angles are fit as
(cos, sin), decoded with atan2 and scored by circular MAE. The error ratio
eps = 100 * MAE / MAE_control compares against an MLP from the true state to
shuffled targets, so 0 is exact decoding and 100 is chance.
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MLP = {
    "hidden_layer_sizes": (128, 64),
    "activation": "relu",
    "alpha": 1e-4,
    "learning_rate_init": 1e-3,
    "early_stopping": True,
    "n_iter_no_change": 20,
    "validation_fraction": 0.1,
    "max_iter": 300,
}


def fit_probe(kind, X, y, seed=0):
    """OLS (``linear``) or the two-layer MLP (``mlp``) on standardized X."""
    if kind == "linear":
        head = LinearRegression()
    elif kind == "mlp":
        head = MLPRegressor(**MLP, random_state=seed)
    else:
        raise ValueError(kind)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    return make_pipeline(StandardScaler(), head).fit(X, y)


def windowed(z, n, t, width):
    """Concatenated latents [z_{t-width+1}; ...; z_t] of frames (n, t).

    z is (N, T, D); n and t index windows and positions, with every
    t >= width - 1. Returns (len(n), width * D), oldest frame first.
    """
    return np.concatenate([z[n, t - lag] for lag in range(width - 1, -1, -1)], -1)


def encode_target(y, angle):
    """(n,) targets as probe outputs: (n, 2) cos/sin for angles, (n, 1) else."""
    return np.stack([np.cos(y), np.sin(y)], -1) if angle else y[:, None]


def decode_target(p, angle):
    p = p.reshape(len(p), -1)
    return np.arctan2(p[:, 1], p[:, 0]) if angle else p[:, 0]


def circular_mae(pred, true):
    """Mean absolute angle difference wrapped to [0, pi]."""
    d = np.mod(pred - true + np.pi, 2 * np.pi) - np.pi
    return float(np.abs(d).mean())


def mae(pred, true, angle):
    return circular_mae(pred, true) if angle else float(np.abs(pred - true).mean())


def control_mae(state_tr, y_tr, state_te, y_te, angle, seed=0):
    """MAE of the MLP probe from true state to train targets shuffled across
    samples, the reference error of an input that carries no information."""
    perm = np.random.default_rng(seed).permutation(len(y_tr))
    m = fit_probe("mlp", state_tr, encode_target(y_tr[perm], angle), seed)
    return mae(decode_target(m.predict(state_te), angle), y_te, angle)


def probe_errors(kind, X_tr, y_tr, X_te, y_te, angle, control, seed=0):
    """Fit one probe for one target; MAE, eps, R^2 and Pearson r on test.

    For angles R^2 and r are taken over the (cos, sin) outputs.
    """
    m = fit_probe(kind, X_tr, encode_target(y_tr, angle), seed)
    p = m.predict(X_te).reshape(len(X_te), -1)
    y = encode_target(y_te, angle)
    err = mae(decode_target(p, angle), y_te, angle)
    r = [pearsonr(p[:, i], y[:, i])[0] if np.ptp(p[:, i]) > 0 else np.nan for i in range(y.shape[1])]
    return {
        "mae": err,
        "eps": 100 * err / control,
        "r2": float(r2_score(y, p, multioutput="variance_weighted")),
        "r": float(np.nanmean(r)) if not np.all(np.isnan(r)) else float("nan"),
    }
