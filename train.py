import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

from grounding import copy_last_loss, grad_shares, next_frame_targets
from metrics.pairs import PairCallback, load_frames
from module import SIGReg
from tag import PixelTag
from utils import get_column_normalizer, get_img_preprocessor, RNGStateCallback, SaveCkptCallback


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

def pixel_term(self, batch, pred_emb, cfg):
    """Mean squared error between the decoded predicted embeddings and the next
    frames, which are the preprocessed frames area-resized to the decoder's
    output size."""
    pred_pixels = self.model.decode(pred_emb)  # (B, T, C, h, w)
    target = next_frame_targets(batch["pixels"], cfg.num_preds, pred_pixels.shape[-2:])
    return (pred_pixels.float() - target).pow(2).mean()


def log_latent_terms(self, output, emb, pred_emb, stage, cfg):
    """Latent prediction error, SIGReg and, on validation, the error of copying
    the last context embedding forward."""
    with torch.no_grad():
        output["pred_loss"] = (pred_emb - emb[:, cfg.num_preds:]).pow(2).mean()
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
        if stage == "validate":
            output["copy_loss"] = copy_last_loss(emb, cfg.history_size, cfg.num_preds)


def log_losses(self, output, stage):
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)


def pixel_forward(self, batch, stage, cfg):
    """Encode observations, predict next states and decode them to the next
    frames; the loss is the pixel error alone.

    The latent prediction error and SIGReg are logged for comparison with the
    latent objective but take no part in the loss.
    """

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]  # (B, T, D)
    pred_emb = self.model.predict(emb[:, : cfg.history_size], output["act_emb"][:, : cfg.history_size])

    output["pixel_loss"] = pixel_term(self, batch, pred_emb, cfg)
    output["loss"] = output["pixel_loss"]
    log_latent_terms(self, output, emb, pred_emb, stage, cfg)
    log_losses(self, output, stage)
    return output


def grounded_forward(self, batch, stage, cfg):
    """LeWM's loss plus ``loss.beta`` times the pixel error of the decoded predicted embeddings.

    Every ``loss.grad_share_every`` training steps the share of the gradient
    norm at the embeddings held by each of the two prediction terms is logged,
    as the epoch mean and the epoch minimum. SIGReg's gradient is not part of the shares.
    """

    beta = cfg.loss.beta
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]  # (B, T, D)
    pred_emb = self.model.predict(emb[:, : cfg.history_size], output["act_emb"][:, : cfg.history_size])

    pred_loss = (pred_emb - emb[:, cfg.num_preds:]).pow(2).mean()
    sigreg_loss = self.sigreg(emb.transpose(0, 1))
    pixel_loss = pixel_term(self, batch, pred_emb, cfg)
    output["pred_loss"], output["sigreg_loss"], output["pixel_loss"] = pred_loss, sigreg_loss, pixel_loss
    output["loss"] = pred_loss + cfg.loss.sigreg.weight * sigreg_loss + beta * pixel_loss

    if stage == "validate":
        output["copy_loss"] = copy_last_loss(emb.detach(), cfg.history_size, cfg.num_preds)
    elif self.global_step % cfg.loss.grad_share_every == 0:
        shares = grad_shares({"latent": pred_loss, "pixel": beta * pixel_loss}, emb)
        self.log_dict({f"{stage}/{k}_grad_share": v for k, v in shares.items()},
                      on_step=False, on_epoch=True, sync_dist=True)
        for k, v in shares.items():
            self.log(f"{stage}/{k}_grad_share_min", v, on_step=False, on_epoch=True, reduce_fx="min", sync_dist=True)
    log_losses(self, output, stage)
    return output


FORWARD = {"latent": lejepa_forward, "pixel": pixel_forward, "grounded": grounded_forward}


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    # Seed before building the model so weight init is reproducible too.
    pl.seed_everything(cfg.seed, workers=True)
    # Deterministic mode NaN-fills every new allocation to expose reads of uninitialized
    # memory. Nothing here reads uninitialized memory, so the fill only costs GPU time.
    torch.utils.deterministic.fill_uninitialized_memory = False

    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    # Pixels leave the workers as uint8, a quarter of the float32 bytes, and are
    # preprocessed on the GPU by the data module's gpu_transform.
    transforms = []

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform
    tag = PixelTag(**cfg.tag)
    pair_callbacks = []
    if cfg.pair.every_n_steps:
        frames = load_frames(dataset_name, cfg.pair.n, seed=0, cache_dir=cache_dir)
        pair_callbacks.append(PairCallback(frames, tag, cfg.img_size, cfg.pair.every_n_steps))
    dataset = tag.wrap(dataset)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(
        train=train,
        val=val,
        gpu_transform=get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size),
    )
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(FORWARD[cfg.loss.kind], cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))
    else:
        # metrics.csv next to the checkpoints, so every run keeps its training curve
        logger = CSVLogger(save_dir=run_dir, name="", version="")

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback, RNGStateCallback(train), *pair_callbacks],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=cfg.get("resume_from") or (ckpt_path if ckpt_path.exists() else None),
        seed=cfg.seed,
    )

    manager()
    return


if __name__ == "__main__":
    run()
