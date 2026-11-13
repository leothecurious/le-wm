import json
import time
from pathlib import Path

import hydra
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf

import tag
from metrics import data, suite
from metrics.latents import Latents, extract


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
    model = swm.wm.utils.load_pretrained(cfg.policy)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval()
    model.requires_grad_(False)

    results_path = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
    tag.resolve_seed(cfg.data.tag, results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    stem = Path(cfg.output.filename).stem
    latents_path = results_path / f"{Path(cfg.policy).stem}_{stem}_latents.pt"

    start_time = time.time()
    lat = get_latents(cfg, model, latents_path)
    metrics = suite.run(lat, OmegaConf.to_container(cfg.metrics, resolve=True))
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
