
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Checkpoints &amp; Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**
```bash
uv sync
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `<data>/datasets/`. The roots come from `config/paths/local.yaml`, which is untracked: copy `config/paths/local.yaml.template` and set `data` (datasets) and `runs` (checkpoints, training logs and results). Without the file both default to `$STABLEWM_HOME` (`~/.stable_worldmodel/`), and `paths.data=...` on the command line overrides either.

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `<data>/datasets/pusht_expert_train.h5`.

The moving-dot benchmark adapts [Sobal et al. (2022)](https://arxiv.org/abs/2211.10831),
Section 3 and Appendices A.2/A.4, to LeWM. Sequences are generated on the fly (`dot.py`), as in the paper's
code, so nothing is recorded and no sequence or noise pattern repeats:

```bash
bash scripts/run.sh dot_grounded_clean --config-name=lewm_grounded data=dot_online
uv run measure.py --config-name=dot policy=dot_grounded_clean/weights_epoch_10.pt
```

### Moving-dot generation and evaluation protocol

Every batch of the training stream is drawn fresh on the training device from generators that are
independent of the model's RNG. Each stage (train, validation, probe, test) has its own geometry and noise
generators, seeded from the run seed and the stage name, so objectives at one seed see identical
sequences in the same order, and no two stages share a sequence or a noise pattern. Streams reproduce on
the same device type. An epoch is `data.generator.steps_per_epoch` batches (11,250, the epoch of a recorded
90,000-sequence training split). Validation is a fixed set of `data.generator.val_sequences` held-out
sequences, regenerated from its seed at the start of each run, whose batches hold clips of distinct
sequences like training batches. Each epoch logs the mean SIGReg and prediction loss on training batches
and on validation, and their validation-to-training ratios (`gap/*` in `metrics.csv`); a SIGReg ratio well
above 1 means the encoder fits the sequences it has seen. Action normalization is fitted on
`data.generator.stats_sequences` sequences of its own stream and saved in the checkpoint's `config.yaml`.

Uniform noise is fixed within a sequence (`noise.mode=episode`) or redrawn per frame (`noise.mode=frame`).
Measurement regenerates the probe and test sequences from the checkpoint's geometry, run seed, noise mode and
strength, and action normalizer, which are saved in its `config.yaml`; `data.noise.mode` and `data.noise.alpha`
override the noise.

The primary `rollout_probe` freezes the encoder and predictor, encodes only the
first frame, and rolls forward with the true actions for all 16 transitions.
One linear readout is fitted jointly to x and y on `data.probe.n` probe sequences.
It scores all 17 time steps on `data.test.n` test sequences, reporting Euclidean
position RMSE per time step and their arithmetic mean, in units of the square's
side. Pooled RMSE and a fixed center predictor are also reported.

Intentional adaptations: LeWM's ViT and action-conditioned transformer replace
the paper's CNN and recurrent predictor; frames are 32 by 32 rather than 28 by
28, replicated to RGB and quantized to uint8. Noise is blended by dividing by
`1 + alpha` to retain that storage range. We keep LeWM image preprocessing,
one-step training, and the roadmap's latent, pixel, grounded and latent-twin
objectives. The paper trains on full sequences and uses a linear probe trained
by gradient descent; our linear readout uses least squares. Dot geometry follows
the reference code where it differs from the paper's text (see
`config/train/data/dot_online.yaml`).

- The encoder and predictor have 17.9M parameters (26.7M with the pixel decoder), against about 1.18M in the paper.
- Training runs 10 epochs of 11,250 steps, each batch fresh, whereas the paper streams about 1M fresh sequences once.
- RMSE is Euclidean over (x, y); compare r, the rollout RMSE over the centre-predictor RMSE.

`trainer.max_epochs` sets the learning-rate schedule's full budget. A two-epoch
run has its own warmup and decay, rather than taking the first two epochs of a
longer schedule. Hyperparameters selected
on clean frames are a transfer experiment; the paper's main results tune each
noise setting separately.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

To launch training:
```bash
uv run train.py data=pusht
```

Checkpoints are saved to `<runs>/checkpoints/<subdir>`, and `metrics.csv` to `<runs>/spt/<subdir>/runs`.

### Weights & Biases

Off by default. `wandb.enabled=true` logs to W&B in addition to the local files, which stay the durable record
(`metrics.csv` under `<runs>/spt/<subdir>/runs`, `dot_metrics.jsonl` and the like next to the results). Entity and
project come from `WANDB_ENTITY` and `WANDB_PROJECT`. The run is named after `output_model_name`, its id is `subdir`,
`wandb.group` is the study (for example `dot2`) and `wandb.tags` are free tags. A run that cannot be opened, a
missing entity or project, `WANDB_MODE=disabled` and a mode other than the one in the environment all abort before
training starts. Offline logging needs `WANDB_MODE=offline` set explicitly; its files are written under
`<runs>/checkpoints/<subdir>/wandb` and uploaded with `wandb sync`.

```bash
WANDB_ENTITY=... WANDB_PROJECT=... bash scripts/run.sh dot2_latent_off_seed0 data=dot_online wandb.enabled=true wandb.group=dot2
```

`measure.py` resumes the training run by its id (`wandb.resume=must`), so the curve and the measured numbers share
one run, and fails if that run does not exist. Measuring one run twice (a different noise, an earlier epoch) under
one set of keys overwrites the first; set `wandb.label=<name>` to keep both. The key namespaces are listed in the
docstring of `tracking.py`.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
uv run eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
uv run eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Metrics

`measure.py` computes state and reverse probes and effective rank for one checkpoint. Configs live under `config/measure/`; `policy` resolves as in `eval.py`. Latents are saved next to the results, and `latents_path=<file>` reruns the metrics without re-encoding.

```bash
uv run measure.py --config-name=pusht policy=pusht_seed1/weights_epoch_10.pt
uv run pytest
```

## Pretrained Checkpoints

Pretrained LeWM checkpoints for each environment are mirrored on the Hugging Face
Hub (model repos), alongside the datasets (dataset repos) in the same collection:

- [`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht)
- [`quentinll/lewm-cube`](https://huggingface.co/quentinll/lewm-cube)
- [`quentinll/lewm-tworooms`](https://huggingface.co/quentinll/lewm-tworooms)
- [`quentinll/lewm-reacher`](https://huggingface.co/quentinll/lewm-reacher)

The full baseline checkpoint suite (PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM, DINO-WM-noprop)
is available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e):

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

### From the Drive archive

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

Place the extracted files under `$STABLEWM_HOME/` and load via:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

`AutoCostModel` accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

### From the Hugging Face mirror

The HF model repos ship the LeWM checkpoint as a `weights.pt` (state dict) plus a
`config.json` describing the model. Convert once to produce the `_object.ckpt`
that `eval.py` expects:

```bash
# download weights.pt + config.json
uv run hf download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht

# convert to object checkpoint under $STABLEWM_HOME/pusht/lewm_object.ckpt
uv run python - <<'PY'
import json, torch, stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

src = Path(swm.data.utils.get_cache_dir(), "hf_pusht")
out = Path(swm.data.utils.get_cache_dir(), "pusht", "lewm_object.ckpt")

cfg = json.loads((src / "config.json").read_text())
encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False, use_mask_token=False,
)
mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                    hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)
model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**cfg["predictor"]),
    action_encoder=Embedder(**cfg["action_encoder"]),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)
sd = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
PY
```

After conversion, load via `swm.policy.AutoCostModel('pusht/lewm')` as usual.

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
