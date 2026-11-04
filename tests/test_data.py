import h5py
import numpy as np
import stable_worldmodel as swm
from omegaconf import OmegaConf

from metrics.data import column_targets


def test_derivatives_stay_inside_episodes(tmp_path):
    x = np.array([0, 1, 2, 3, 100, 110, 120], dtype=np.float32)
    angle = np.array([3.0, 3.1, -3.1, -3.0, 0.0, 0.1, 0.2], dtype=np.float32)
    path = tmp_path / "toy.h5"
    with h5py.File(path, "w", libver="latest") as f:
        f["ep_len"] = np.array([4, 3])
        f["ep_offset"] = np.array([0, 4])
        f["state"] = np.stack([x, angle], 1)
        f["action"] = np.zeros((7, 2), dtype=np.float32)
    ds = swm.data.HDF5Dataset(path=path, keys_to_load=["state", "action"])

    specs = OmegaConf.create([
        {"name": "vx", "column": "state", "index": 0, "derivative": True},
        {"name": "omega", "column": "state", "index": 1, "derivative": True, "angle": True},
    ])
    d = column_targets(ds, specs)

    np.testing.assert_allclose(d[:, 0], [1, 1, 1, 1, 10, 10, 10])
    across_wrap = (2 * np.pi - 6.1) / 2
    np.testing.assert_allclose(d[:4, 1], [0.1, across_wrap, across_wrap, 0.1], atol=1e-5)
    np.testing.assert_allclose(d[4:, 1], 0.1, atol=1e-5)
