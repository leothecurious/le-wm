
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

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

The moving-dot benchmark adapts [Sobal et al. (2022)](https://arxiv.org/abs/2211.10831),
Section 3 and Appendices A.2/A.4, to LeWM. Record three independent datasets:

```bash
uv run collect_dot.py name=/absolute/path/dot_train.h5
uv run collect_dot.py --config-name=dot_probe name=/absolute/path/dot_probe.h5
uv run collect_dot.py --config-name=dot_test name=/absolute/path/dot_test.h5
PATH="$HOME/.local/bin:$PATH" bash scripts/run.sh dot_grounded_clean train.py --config-name=lewm_grounded data=dot data.dataset.name=/absolute/path/dot_train.h5
uv run measure.py --config-name=dot policy=dot_grounded_clean/weights_epoch_10.pt data.name=/absolute/path/dot_test.h5 data.probe.name=/absolute/path/dot_probe.h5
```

### Moving-dot evaluation protocol

The dot training config uses one observed frame per context and holds out whole
episodes for validation. No validation frame or episode-fixed noise pattern is
shared with an encoder-training episode. Action normalization is fitted only on
training episodes, saved in the checkpoint's `config.yaml`, and reused for both
probe fitting and test. Other datasets retain their existing clip split.

Uniform noise is fixed within an episode (`noise.mode=episode`) or redrawn per
frame (`noise.mode=frame`). Set `noise.alpha` during training and the matching
`data.noise.mode` and `data.noise.alpha` during measurement. Noise depends on
local episode/frame IDs and a noise seed, not on dataset identity. By default,
probe fitting uses the training noise seed plus 1 and test uses it plus 2.
Measurement rejects shared dataset paths and reused stage noise seeds. Explicit
measurement seeds may override those defaults, but must remain distinct. New
trajectories and new noise are both required for this evaluation protocol.

The primary `rollout_probe` freezes the encoder and predictor, encodes only the
first frame, and rolls forward with the recorded actions for all 16 transitions.
One linear readout is fitted jointly to x and y on the separate probe dataset.
It scores all 17 time steps on the separate test dataset, reporting Euclidean
position RMSE per time step and their arithmetic mean, in units of the square's
side. Pooled RMSE and a fixed center predictor are also reported. The existing
linear/MLP encoder probes and effective rank remain separate diagnostics; they
are not the paper's rollout score.

The default recording presets contain 10,000 episodes per stage for validation.
`dot_paper_train`, `dot_paper_probe`, and `dot_paper_test` collection presets
request the paper's 1,000,000 / 300,000 / 10,000 sequences; `measure.py
--config-name=dot_paper` requests all probe and test sequences. These large
presets require a separately budgeted run: recorded RGB data and saved latents
consume substantial disk/RAM. Counts do not establish convergence, and the
encoder's validation fraction reduces the number of training episodes actually
used. Dataset-sidecar YAML files retain collection settings and seeds.

Intentional adaptations: LeWM's ViT and action-conditioned transformer replace
the paper's CNN and recurrent predictor; frames are 32 by 32 rather than 28 by
28, replicated to RGB and quantized to uint8. Noise is blended by dividing by
`1 + alpha` to retain that storage range. We keep LeWM image preprocessing,
one-step training, and the roadmap's latent, pixel, grounded and latent-twin
objectives. The paper trains on full sequences and uses a linear probe trained
by gradient descent; our linear readout uses least squares. These are an adapted
benchmark, not an exact numerical reproduction. Dataset/noise seeds are explicit
project choices, not seed values reported by the paper.

- Action directions use Von Mises concentration 0.2, the reference code's value (vladisai/JEPA_SSL_NeurIPS_2022, `data/single.py:22`); the paper's App. A.2 says 1.3.
- The encoder and predictor have 17.9M parameters (26.7M with the pixel decoder), against about 1.18M in the paper.
- Training runs 10 epochs over a fixed 100k-sequence set (about 11k steps per epoch), whereas the paper streams about 1M fresh sequences once.
- RMSE sums the x and y errors, so absolute values differ from the paper's; compare r, the rollout RMSE over the centre-predictor RMSE.

`trainer.max_epochs` sets the learning-rate schedule's full budget. A two-epoch
run has its own warmup and decay, rather than taking the first two epochs of a
longer schedule. Keep collection configs, checkpoint hashes, noise settings,
validation split and training budget with each result. Hyperparameters selected
on clean frames are a transfer experiment; the paper's main results tune each
noise setting separately.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
uv run train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

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
