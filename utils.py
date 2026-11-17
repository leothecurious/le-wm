import os
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def rechunk_h5(path, rows=65536, chunk_bytes=1 << 20, block_bytes=1 << 26):
    """Rewrite an HDF5Writer file in place with large chunks.

    The writer gives every per-row column one-row chunks so that appending an episode is cheap. HDF5 then
    keeps one chunk-index entry per row, which costs several KB of RAM per row once a column is read and
    makes per-frame pixel reads slow. Columns with ndim > 2 (frames) get one episode per chunk when all
    episodes share a length, else ``chunk_bytes`` worth of rows; every other column gets ``rows`` rows.
    The copy goes to a sibling file, is compared with the original block by block, and then replaces it.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".rechunk.tmp")
    try:
        with h5py.File(path, "r") as src, h5py.File(tmp, "w", libver="latest") as dst:
            lengths = np.unique(src["ep_len"][:])
            dst.attrs.update(src.attrs)
            for name, d in src.items():
                row_bytes = max(1, int(np.prod(d.shape[1:], dtype=np.int64)) * d.dtype.itemsize)
                if d.ndim <= 2:
                    n = rows
                elif len(lengths) == 1:
                    n = int(lengths[0])
                else:
                    n = chunk_bytes // row_bytes
                n = max(1, min(n, d.shape[0]))
                out = dst.create_dataset(
                    name, shape=d.shape, maxshape=d.maxshape, dtype=d.dtype, chunks=(n, *d.shape[1:])
                )
                out.attrs.update(d.attrs)
                block = n * max(1, block_bytes // (n * row_bytes))
                for i in range(0, d.shape[0], block):
                    out[i:i + block] = d[i:i + block]
                for i in range(0, d.shape[0], block):
                    if d[i:i + block].tobytes() != out[i:i + block].tobytes():
                        raise RuntimeError(f"rechunk changed column {name!r} in rows {i}:{i + block}")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


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
            for ep in sorted(set(episodes))
        ])
    data = torch.from_numpy(col_data)
    data = data[~torch.isnan(data).any(dim=1)]
    return {"mean": data.mean(0, keepdim=True).clone(),
            "std": data.std(0, keepdim=True).clone()}


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
    """Checkpoint the shuffle generator and the global torch RNGs, which
    Lightning checkpoints omit. States are saved at epoch end.

    Each new DataLoader iterator draws its workers' base seed from the shuffle
    generator. With persistent workers a resumed run builds one more iterator
    than an uninterrupted run, so on resume that draw goes to a throwaway
    generator. Data loading has no randomness, so the worker seed is unused.
    """

    def __init__(self, train_loader: torch.utils.data.DataLoader):
        super().__init__()
        self.train_loader = train_loader
        self.data_generator = train_loader.sampler.generator

    def state_dict(self):
        state = {
            "data_generator": self.data_generator.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def load_state_dict(self, state_dict):
        self.data_generator.set_state(state_dict["data_generator"])
        if self.train_loader.persistent_workers and self.train_loader.num_workers > 0:
            self.train_loader.generator = torch.Generator()
        torch.set_rng_state(state_dict["torch"])
        if "cuda" in state_dict and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state_dict["cuda"])


class SaveCkptCallback(Callback):
    """Saves ``weights_epoch_N.pt`` every ``epoch_interval`` epochs and at the last one, and
    ``weights_step_N.pt`` every ``step_interval`` optimizer steps (None = never), through save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1, step_interval=None, cache_dir=None):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
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
            if epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs:
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