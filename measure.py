import json
import time
from pathlib import Path

import hydra
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf

from metrics import data, pairs, suite
from metrics.latents import Latents, extract
from tag import PixelTag, checkpoint_dir, resolve_seed


def measure_pairs(cfg, model):
    """Pair cosines on dataset frames stamped with the tag's colours."""
    frames = pairs.load_frames(cfg.data.name, cfg.pairs.n, cfg.data.seed, cfg.get("cache_dir"))
    tag = PixelTag("episode", cfg.data.tag.size, cfg.data.tag.seed)
    return pairs.measure(model, frames, tag, cfg.data.img_size, cfg.data.seed)


def get_latents(cfg, model, latents_path):
    if cfg.get("latents_path"):
        return Latents.load(cfg.latents_path)

    w = cfg.windows
    loader, target, episode, train = data.build(
        cfg.data, cfg.targets, w.frames, w.min_frames, cfg.get("cache_dir")
    )
    names, angle, group = data.target_meta(cfg.targets)
    out = extract(model, loader, w.precision)
    lat = Latents(
        **out,
        target=torch.from_numpy(target),
        episode=torch.from_numpy(episode),
        train=torch.from_numpy(train),
        target_names=names,
        target_angle=angle,
        target_group=group,
    )
    lat.save(latents_path)
    return lat


@hydra.main(version_base=None, config_path="./config/measure", config_name="pusht")
def run(cfg: DictConfig):
    """Compute the metric suite for one checkpoint."""
    cfg = OmegaConf.merge({"data": {"tag": {"mode": "off", "size": 5, "seed": None}}}, cfg)
    model = swm.wm.utils.load_pretrained(cfg.policy)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval()
    model.requires_grad_(False)

    cache_dir = swm.data.utils.get_cache_dir()
    results_path = Path(cache_dir, cfg.policy).parent
    resolve_seed(cfg.data.tag, checkpoint_dir(cache_dir, cfg.policy))
    results_path.mkdir(parents=True, exist_ok=True)
    stem = Path(cfg.output.filename).stem
    latents_path = results_path / f"{Path(cfg.policy).stem}_{stem}_latents.pt"

    start_time = time.time()
    lat = get_latents(cfg, model, latents_path)
    metrics = suite.run(lat, OmegaConf.to_container(cfg.metrics, resolve=True))
    if cfg.get("pairs"):
        metrics["pairs"] = measure_pairs(cfg, model)
    end_time = time.time()

    print(json.dumps(metrics, indent=1))

    with (results_path / cfg.output.filename).open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {json.dumps(metrics)}\n")
        f.write(f"latents: {cfg.get('latents_path') or latents_path}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")

    record = {
        "policy": cfg.policy,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics,
    }
    with (results_path / f"{stem}.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    run()
