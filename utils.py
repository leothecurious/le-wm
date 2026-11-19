import os
from pathlib import Path

import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def split_episodes(dataset, train_fraction, generator):
    """Split clips by original episode ID, excluding episodes without clips."""
    episodes = np.array([ep for ep, _ in dataset.clip_indices])
    eligible = np.unique(episodes)
    order = torch.randperm(len(eligible), generator=generator).numpy()
    n_train = int(train_fraction * len(eligible))
    if not 0 < n_train < len(eligible):
        raise ValueError("episode split needs at least one episode on each side")
    train_mask = np.isin(episodes, eligible[order[:n_train]])
    return (
        torch.utils.data.Subset(dataset, np.flatnonzero(train_mask).tolist()),
        torch.utils.data.Subset(dataset, np.flatnonzero(~train_mask).tolist()),
    )


def get_column_stats(dataset, source, episodes=None):
    """Row-weighted column moments, optionally restricted to whole episodes."""
    col_data = np.array(dataset.get_col_data(source))
    if episodes is not None:
        col_data = np.concatenate([
            col_data[dataset.offsets[ep]:dataset.offsets[ep] + dataset.lengths[ep]]
            for ep in sorted(episodes)
        ])
    data = torch.from_numpy(col_data)
    data = data[~torch.isnan(data).any(dim=1)]
    return {"mean": data.mean(0, keepdim=True), "std": data.std(0, keepdim=True)}


def get_column_normalizer(dataset, source: str, target: str, stats=None):
    """Normalize a column using supplied moments, or fit them on the dataset."""
    stats = get_column_stats(dataset, source) if stats is None else stats
    mean = torch.as_tensor(stats["mean"]).reshape(1, -1)
    std = torch.as_tensor(stats["std"]).reshape(1, -1)
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)


class ResumeCkptCallback(Callback):
    """Replaces the full Lightning checkpoint at ``path`` after every epoch.

    The Manager's own ``last.ckpt`` lives in a timestamped run directory, which a
    rerun of the same command never finds. This one has a fixed path to resume from.
    """

    def __init__(self, path):
        super().__init__()
        self.path = Path(path)

    def on_train_epoch_end(self, trainer, pl_module):
        tmp = self.path.with_name(self.path.name + ".tmp")
        trainer.save_checkpoint(tmp)
        if trainer.is_global_zero:
            os.replace(tmp, self.path)


class RNGStateCallback(Callback):
    """Checkpoint the shuffle generator (or, for a generated dataset, its streams) and the global
    torch RNGs, which Lightning checkpoints omit. States are saved at epoch end.

    Each new DataLoader iterator draws its workers' base seed from the shuffle
    generator. With persistent workers a resumed run builds one more iterator
    than an uninterrupted run, so on resume that draw goes to a throwaway
    generator. Data loading has no randomness, so the worker seed is unused.
    """

    def __init__(self, train_loader: torch.utils.data.DataLoader):
        super().__init__()
        self.train_loader = train_loader
        self.generated = hasattr(train_loader.dataset, "streams")
        self.data_generator = None if self.generated else train_loader.sampler.generator

    def state_dict(self):
        state = {
            "data_generator": self.train_loader.dataset.state_dict() if self.generated else self.data_generator.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def load_state_dict(self, state_dict):
        if self.generated:
            self.train_loader.dataset.load_state_dict(state_dict["data_generator"])
        else:
            self.data_generator.set_state(state_dict["data_generator"])
        if self.train_loader.persistent_workers and self.train_loader.num_workers > 0:
            self.train_loader.generator = torch.Generator()
        torch.set_rng_state(state_dict["torch"])
        if "cuda" in state_dict and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state_dict["cuda"])


class SaveCkptCallback(Callback):
    """Saves ``weights_epoch_N.pt`` every ``epoch_interval`` epochs (only at epochs in ``epochs``, when
    given) and at the last one, and ``weights_step_N.pt`` every ``step_interval`` optimizer steps (None =
    never), through save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1, step_interval=None, cache_dir=None, epochs=None):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epochs = epochs
        self.epoch_interval = epoch_interval
        self.step_interval = step_interval
        self.cache_dir = cache_dir

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if self.step_interval and step and step % self.step_interval == 0 and trainer.is_global_zero:
            self._save(pl_module.model, f"step_{step}")

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            epoch = trainer.current_epoch + 1
            wanted = epoch % self.epoch_interval == 0 if self.epochs is None else epoch in self.epochs
            if wanted or epoch == trainer.max_epochs:
                self._save(pl_module.model, f"epoch_{epoch}")

    def _save(self, model, name):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_{name}.pt',
            cache_dir=self.cache_dir,
        )

class GapCallback(Callback):
    """Logs each epoch's mean SIGReg and prediction loss on training batches and on validation, and
    their validation-to-training ratios. A ratio well above 1 means the encoder fits what it has seen
    rather than what the data generates. Training batches are in train mode (dropout on), validation in eval mode."""

    KEYS = ("sigreg_loss", "pred_loss")

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.sums = {s: {k: 0.0 for k in self.KEYS} for s in ("train", "val")}
        self.counts = {"train": 0, "val": 0}

    def _add(self, stage, outputs):
        for k in self.KEYS:
            self.sums[stage][k] = self.sums[stage][k] + outputs[k].detach().float()
        self.counts[stage] += 1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._add("train", outputs)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.sanity_checking:
            self._add("val", outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        mean = {s: {k: v / max(self.counts[s], 1) for k, v in self.sums[s].items()} for s in self.sums}
        for k in self.KEYS:
            name = k.removesuffix("_loss")
            pl_module.log(f"gap/{name}_train", mean["train"][k], sync_dist=True)
            if self.counts["val"]:
                pl_module.log(f"gap/{name}_val", mean["val"][k], sync_dist=True)
                pl_module.log(f"gap/{name}_val_over_train", mean["val"][k] / mean["train"][k], sync_dist=True)
        self.reset()
