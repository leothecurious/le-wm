import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf, open_dict

from metrics.pairs import PairCallback, load_frames
from module import SIGReg, grad_shares
from noise import PixelNoise
from tag import PixelTag
from utils import get_column_normalizer, get_img_preprocessor, ResumeCkptCallback, RNGStateCallback, SaveCkptCallback


def lewm_forward(self, batch, stage, cfg):
    """Encode, predict the next embeddings and compute the ``loss.kind`` objective.

    latent: LeWM's prediction error plus SIGReg. pixel: the error of the predicted
    embeddings decoded to the next frames, area-resized to the decoder's output.
    grounded: latent plus ``loss.beta`` times pixel. latent_twin: pixel plus
    ``loss.lam`` times the prediction error against a stop-gradient target. Every
    kind logs the latent terms and, on validation, the error of copying the last
    context embedding forward.
    """
    kind, ctx_len, n_preds = cfg.loss.kind, cfg.history_size, cfg.num_preds

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]  # (B, T, D)
    pred_emb = self.model.predict(emb[:, :ctx_len], output["act_emb"][:, :ctx_len])
    target_emb = emb[:, n_preds:].detach() if kind == "latent_twin" else emb[:, n_preds:]

    grad = torch.is_grad_enabled()
    with torch.set_grad_enabled(grad and kind != "pixel"):
        output["pred_loss"] = (pred_emb - target_emb).pow(2).mean()
    with torch.set_grad_enabled(grad and kind in ("latent", "grounded")):
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

    terms = {}  # the two prediction terms of hybrid objectives, as weighted in the loss
    if kind == "latent":
        output["loss"] = output["pred_loss"] + cfg.loss.sigreg.weight * output["sigreg_loss"]
        if hasattr(self.model, "diff_losses"):
            output.update(self.model.diff_losses(output, self.sigreg, cfg.loss.motion))
            output["loss"] = output["loss"] + output["diff_loss"]
    else:
        pred_pixels = self.model.decode(pred_emb).float()  # (B, T, C, h, w)
        target = batch["pixels"][:, n_preds:]
        target = F.interpolate(target.flatten(0, 1).float(), size=pred_pixels.shape[-2:], mode="area")
        output["pixel_loss"] = (pred_pixels - target.unflatten(0, pred_pixels.shape[:2])).pow(2).mean()
        if kind == "pixel":
            output["loss"] = output["pixel_loss"]
        elif kind == "grounded":
            terms = {"latent": output["pred_loss"], "pixel": cfg.loss.beta * output["pixel_loss"]}
            output["loss"] = sum(terms.values()) + cfg.loss.sigreg.weight * output["sigreg_loss"]
        elif kind == "latent_twin":
            terms = {"latent": cfg.loss.lam * output["pred_loss"], "pixel": output["pixel_loss"]}
            output["loss"] = sum(terms.values())
        else:
            raise ValueError(kind)

    if stage == "validate":
        output["copy_loss"] = (emb[:, :ctx_len] - emb[:, n_preds:]).detach().pow(2).mean()
    elif terms and self.global_step % cfg.loss.grad_share_every == 0:
        # Share of the gradient norm at the embeddings held by each prediction term; SIGReg's is left out.
        for k, v in grad_shares(terms, emb).items():
            self.log(f"{stage}/{k}_grad_share", v, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{stage}/{k}_grad_share_min", v, on_step=False, on_epoch=True, reduce_fx="min", sync_dist=True)

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


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
    dataset = PixelNoise(**cfg.noise).wrap(tag.wrap(dataset))

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
        forward=partial(lewm_forward, cfg=cfg),
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
        step_interval=cfg.checkpoint.every_n_steps,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            object_dump_callback,
            RNGStateCallback(train),
            ResumeCkptCallback(ckpt_path),
            *pair_callbacks,
        ],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=cfg.get("resume_from") or (ckpt_path if ckpt_path.exists() else None),
        weights_only=False,
        seed=cfg.seed,
    )

    manager()
    return


if __name__ == "__main__":
    run()
