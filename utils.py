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


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

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
    """Saves ``weights_epoch_N.pt`` every ``epoch_interval`` epochs and ``weights_step_N.pt``
    every ``step_interval`` optimizer steps (None = never), through save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1, step_interval=None):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        self.step_interval = step_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if self.step_interval and step and step % self.step_interval == 0 and trainer.is_global_zero:
            self._save(pl_module.model, f"step_{step}")

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, f"epoch_{trainer.current_epoch + 1}")

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, f"epoch_{trainer.current_epoch + 1}")

    def _save(self, model, name):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_{name}.pt',
        )