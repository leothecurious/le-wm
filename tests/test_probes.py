import numpy as np

from metrics import probes


def test_circular_mae_wraps():
    err = probes.circular_mae(np.deg2rad([359.0]), np.deg2rad([1.0]))
    np.testing.assert_allclose(np.rad2deg(err), 2.0)


def test_eps_spans_exact_to_chance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 8))
    tr, te = slice(0, 1600), slice(1600, None)
    for y, lo, hi in ((X @ rng.normal(size=8), 0, 1), (rng.normal(size=2000), 85, 115)):
        state = y[:, None]
        control = probes.control_mae(state[tr], y[tr], state[te], y[te], angle=False)
        res = probes.probe_errors("linear", X[tr], y[tr], X[te], y[te], False, control)
        assert lo <= res["eps"] <= hi, res
