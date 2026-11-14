from types import SimpleNamespace

import pytest

from utils import SaveCkptCallback


def saved_steps(step_interval, steps, is_global_zero=True):
    callback = SaveCkptCallback("run", cfg=None, step_interval=step_interval)
    saved = []
    callback._save = lambda model, name: saved.append(name)
    module = SimpleNamespace(model=None)
    for step in steps:
        trainer = SimpleNamespace(global_step=step, is_global_zero=is_global_zero)
        callback.on_train_batch_end(trainer, module, None, None, 0)
    return saved


def test_step_checkpoints_land_on_multiples_of_the_interval_and_skip_step_zero():
    assert saved_steps(5, range(0, 16)) == ["step_5", "step_10", "step_15"]


@pytest.mark.parametrize("interval", [None, 0])
def test_no_interval_saves_no_step_checkpoints(interval):
    assert saved_steps(interval, range(0, 20)) == []


def test_only_the_main_process_saves():
    assert saved_steps(5, range(0, 16), is_global_zero=False) == []
