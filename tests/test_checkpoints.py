from types import SimpleNamespace

from utils import SaveCkptCallback


def saved(step_interval, steps, epoch_interval, max_epochs):
    callback = SaveCkptCallback("run", cfg=None, epoch_interval=epoch_interval, step_interval=step_interval)
    names = []
    callback._save = lambda model, name: names.append(name)
    module = SimpleNamespace(model=None)
    for step in steps:
        callback.on_train_batch_end(SimpleNamespace(global_step=step, is_global_zero=True), module, None, None, 0)
    for epoch in range(max_epochs):
        trainer = SimpleNamespace(current_epoch=epoch, max_epochs=max_epochs, is_global_zero=True)
        callback.on_train_epoch_end(trainer, module)
    return names


def test_checkpoint_names_follow_the_step_and_epoch_intervals():
    # step 0 is skipped, and the last epoch is saved once even when it is off the interval
    assert saved(5, range(16), epoch_interval=2, max_epochs=3) == ["step_5", "step_10", "step_15", "epoch_2", "epoch_3"]
