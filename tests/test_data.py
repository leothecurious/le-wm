import h5py
import numpy as np
import pytest
import stable_worldmodel as swm
from omegaconf import OmegaConf

from metrics.data import column_targets


def toy_dataset(tmp_path, state):
    path = tmp_path / "toy.h5"
    with h5py.File(path, "w", libver="latest") as f:
        f["ep_len"] = np.array([4, 3])
        f["ep_offset"] = np.array([0, 4])
        f["state"] = state
        f["action"] = np.zeros((7, 2), dtype=np.float32)
    return swm.data.HDF5Dataset(path=path, keys_to_load=["state", "action"])


def test_targets_read_the_indexed_column(tmp_path):
    state = np.arange(14, dtype=np.float32).reshape(7, 2)
    ds = toy_dataset(tmp_path, state)
    specs = OmegaConf.create([{"name": "b", "column": "state", "index": 1}])

    np.testing.assert_allclose(column_targets(ds, specs)[:, 0], state[:, 1])


def test_constant_target_is_rejected(tmp_path):
    ds = toy_dataset(tmp_path, np.ones((7, 2), dtype=np.float32))
    specs = OmegaConf.create([{"name": "a", "column": "state", "index": 0}])

    with pytest.raises(AssertionError, match="constant"):
        column_targets(ds, specs)
