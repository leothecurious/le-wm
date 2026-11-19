import hashlib
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

import dot
from metrics.pairs import PairCallback, load_frames
from module import SIGReg, grad_shares
from noise import PixelNoise
from tag import PixelTag
from utils import (get_column_normalizer, get_column_stats, get_img_preprocessor,
                   split_episodes, GapCallback, ResumeCkptCallback, RNGStateCallback, SaveCkptCallback)


def lewm_forward(self, batch, stage, cfg):
    """Encode, predict the next embeddings and compute the ``loss.kind`` objective.

    latent: LeWM's prediction error plus SIGReg. pixel: the error of the predicted
    embeddings decoded to the next frames, area-resized to the decoder's output.
    grounded: latent plus ``loss.beta`` times pixel. latent_twin: pixel plus
    ``loss.lam`` times the prediction error against a stop-gradient target. Every
    kind logs the latent terms and, on validation, the error of copying each
    context embedding one step forward.
    """
    kind, ctx_len, n_preds = cfg.loss.kind, cfg.history_size, cfg.num_preds
    prefix = {"fit": "train", "validate": "val"}.get(stage, stage)
    if kind not in ("latent", "pixel", "grounded", "latent_twin"):
        raise ValueError(kind)

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

    if stage == "validate":
        output["copy_loss"] = (emb[:, :ctx_len] - emb[:, n_preds:]).detach().pow(2).mean()
    elif terms and self.global_step % cfg.loss.grad_share_every == 0:
        # Share of the gradient norm at the embeddings held by each prediction term; SIGReg's is left out.
        for k, v in grad_shares(terms, emb).items():
            self.log(f"{prefix}/{k}_grad_share", v, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{prefix}/{k}_grad_share_min", v, on_step=False, on_epoch=True, reduce_fx="min", sync_dist=True)

    losses_dict = {f"{prefix}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def generated_data(cfg):
    """Loaders over moving-dot clips generated on the fly (dot.py). The epoch is ``steps_per_epoch`` fresh
    batches of the training stream; validation is a fixed held-out set drawn from its own stream."""
    gen, length, batch_size = cfg.data.generator, cfg.history_size + cfg.num_preds, cfg.loader.batch_size
    assert cfg.tag.mode == "off" and not cfg.pair.every_n_steps, "generated dots have no tag or pair metric"
    device = torch.device("cpu" if cfg.trainer.accelerator == "cpu" or not torch.cuda.is_available() else "cuda")
    stats = dot.action_stats(gen, cfg.seed, device, gen.stats_sequences)
    with open_dict(cfg):
        cfg.data.action_normalizer = {key: value.flatten().tolist() for key, value in stats.items()}
        cfg.model.action_encoder.input_dim = stats["mean"].numel()
    train = dot.ClipStream(
        gen, dot.Streams(cfg.seed, "train", device), stats, cfg.noise.mode, cfg.noise.alpha,
        length, batch_size, gen.steps_per_epoch, gen.chunk,
    )
    val = dot.HeldOutClips(
        gen, cfg.seed, device, stats, cfg.noise.mode, cfg.noise.alpha, length, batch_size, gen.val_sequences
    )
    return *(torch.utils.data.DataLoader(d, batch_size=None) for d in (train, val)), []


def recorded_data(cfg):
    """Loaders over a recorded HDF5 dataset, split by clip or by episode, and the pair callbacks."""
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cfg.paths.data, **dataset_cfg
    )
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    split_unit = cfg.split_unit
    if split_unit == "episodes":
        train_set, val_set = split_episodes(dataset, cfg.train_split, rnd_gen)
        train_episodes = {dataset.clip_indices[i][0] for i in train_set.indices}
    elif split_unit == "clips":
        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
        )
        train_episodes = None
    else:
        raise ValueError(f"unknown split_unit: {split_unit}")

    # Pixels leave the workers as uint8, a quarter of the float32 bytes, and are
    # preprocessed on the GPU by the data module's gpu_transform.
    transforms = []

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            stats = get_column_stats(dataset, col, episodes=train_episodes)
            normalizer = get_column_normalizer(dataset, col, col, stats=stats)
            transforms.append(normalizer)
            if col == "action":
                cfg.data.action_normalizer = {
                    key: value.flatten().tolist() for key, value in stats.items()
                }

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform
    tag = PixelTag(**cfg.tag)
    pair_callbacks = []
    if cfg.pair.every_n_steps:
        frames = load_frames(dataset_name, cfg.pair.n, seed=0, cache_dir=cfg.paths.data)
        pair_callbacks.append(PairCallback(frames, tag, cfg.img_size, cfg.pair.every_n_steps))
    # in place, so train_set and val_set stamp too
    PixelNoise(**cfg.noise).wrap(tag.wrap(dataset))

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    return train, val, pair_callbacks


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(cfg.paths.runs, sub_folder='checkpoints'), run_id)
    # metrics.csv next to the checkpoints, so every run keeps its training curve
    loggers = [CSVLogger(save_dir=run_dir, name="", version="")]
    if cfg.wandb.enabled:
        # Opened before any work, so a bad key or an offline or disabled mode fails at once. W&B takes the
        # commit from the repo holding save_dir unless given git_root, and that may be another checkout or none.
        wandb_logger = WandbLogger(
            **OmegaConf.to_container(cfg.wandb.config, resolve=True),
            save_dir=str(run_dir), settings={"git_root": str(Path(__file__).resolve().parent)},
        )
        if wandb_logger.experiment.offline or wandb_logger.experiment.disabled:
            raise RuntimeError("wandb.enabled needs an online W&B run")
        loggers.append(wandb_logger)

    # Seed before building the model so weight init is reproducible too.
    pl.seed_everything(cfg.seed, workers=True)
    # Deterministic mode NaN-fills every new allocation to expose reads of uninitialized
    # memory. Nothing here reads uninitialized memory, so the fill only costs GPU time.
    torch.utils.deterministic.fill_uninitialized_memory = False

    #########################
    ##       dataset       ##
    #########################

    train, val, pair_callbacks = (generated_data if cfg.data.get("generator") else recorded_data)(cfg)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "step",
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

    # stable-pretraining writes its run registry and metrics.csv under <cache_dir>/runs; one cache per run keeps each findable by name
    spt.set(cache_dir=str(swm.data.utils.get_cache_dir(cfg.paths.runs, sub_folder='spt') / run_id))
    if not cfg.checkpoint.lightning:
        spt.set(requeue_checkpoint=False)
    # its wandb_resume.json lands in the working directory; the run id is subdir already
    spt.set(default_callbacks={"wandb_checkpoint": False})

    # after the W&B run opens, so a run that fails to start leaves no config.yaml that looks like a real run
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    if cfg.wandb.enabled:
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, cache_dir=cfg.paths.runs, epoch_interval=1,
        step_interval=cfg.checkpoint.every_n_steps, epochs=cfg.checkpoint.epochs,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[
            object_dump_callback,
            RNGStateCallback(train),
            GapCallback(),
            *([ResumeCkptCallback(ckpt_path)] if cfg.checkpoint.resume else []),
            *pair_callbacks,
        ],
        num_sanity_val_steps=1,
        logger=loggers,
        enable_checkpointing=cfg.checkpoint.lightning,
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
    if cfg.wandb.enabled:
        weights = swm.data.utils.get_cache_dir(cfg.paths.runs, sub_folder='checkpoints') / cfg.output_model_name
        weights = weights / f"weights_epoch_{cfg.trainer.max_epochs}.pt"
        wandb_logger.experiment.summary["checkpoint/sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    if not cfg.checkpoint.keep_resume:
        ckpt_path.unlink(missing_ok=True)


if __name__ == "__main__":
    run()
