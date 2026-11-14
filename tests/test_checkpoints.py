from types import SimpleNamespace

import pytest

from utils import SaveCkptCallback


def saved(step_interval=None, steps=(), epoch_interval=1, max_epochs=0, is_global_zero=True):
    callback = SaveCkptCallback("run", cfg=None, epoch_interval=epoch_interval, step_interval=step_interval)
    names = []
    callback._save = lambda model, name: names.append(name)
    module = SimpleNamespace(model=None)
    for step in steps:
        trainer = SimpleNamespace(global_step=step, is_global_zero=is_global_zero)
        callback.on_train_batch_end(trainer, module, None, None, 0)
    for epoch in range(max_epochs):
        trainer = SimpleNamespace(current_epoch=epoch, max_epochs=max_epochs, is_global_zero=is_global_zero)
        callback.on_train_epoch_end(trainer, module)
    return names


def test_step_checkpoints_land_on_multiples_of_the_interval_and_skip_step_zero():
    assert saved(5, range(0, 16)) == ["step_5", "step_10", "step_15"]


@pytest.mark.parametrize("interval", [None, 0])
def test_no_interval_saves_no_step_checkpoints(interval):
    assert saved(interval, range(0, 20)) == []


@pytest.mark.parametrize(
    "interval, max_epochs, names",
    [
        (1, 3, ["epoch_1", "epoch_2", "epoch_3"]),
        (2, 4, ["epoch_2", "epoch_4"]),
        (2, 3, ["epoch_2", "epoch_3"]),
    ],
)
def test_epoch_checkpoints_save_each_interval_and_the_last_epoch_once(interval, max_epochs, names):
    assert saved(epoch_interval=interval, max_epochs=max_epochs) == names


def test_only_the_main_process_saves():
    assert saved(5, range(0, 16), max_epochs=3, is_global_zero=False) == []
