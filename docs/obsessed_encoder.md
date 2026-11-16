# Obsessed encoder on LeWM

Reproduction of the LeWM case in Enigma's
[The-Obsessed-Encoder](https://github.com/Enigma-Incorporated/The-Obsessed-Encoder)
(`37819425`): a 5×5 colour square stamped in the top-left corner of every
Push-T frame. When the colour repeats within an episode, the encoder can use it
to predict the next latent and stops encoding the scene. Their LeWM is upstream
`lucas-maes/le-wm@8edfeb3`, trained for 10 epochs with seeds 0–2.

## Arms

| Arm | Frames | Role |
|---|---|---|
| `baseline` | clean | reference |
| `tag_episode` | one colour per episode | test |
| `tag_frame` | a new colour every frame | control |

`tag_episode` and `tag_frame` differ only in `tag.mode`. The arms are config
groups under `config/train/arm/`. Each logs the pair cosines every 100 steps
and saves `weights_step_N.pt` every 5000 steps (72 MB each), next to the per-epoch
weights. A run is about 139k steps.
The baseline's pair cosines are measured by stamping a tag the encoder never saw.

## Commands

Training, with `S` the seed and `ARM` one of `baseline`, `tag_episode`, `tag_frame`:

```bash
python train.py data=pusht +arm=$ARM seed=$S
```

The run is saved as `${ARM}_seed${S}`. Measurement and planning read the tag
seed from the run's `config.yaml` and need the tag mode set explicitly. Use the
arm's own mode to score on the frames it trained on, or `off` for clean frames.

```bash
RUN=${ARM}_seed${S}/weights_epoch_10.pt
python measure.py --config-name=pusht policy=$RUN data.tag.mode=$MODE
python eval.py --config-name=pusht policy=$RUN tag.mode=$MODE
```

`measure.py` reports the pair cosines (`pairs`), the state probes and the
effective rank. The pair cosines always stamp a tag, whatever `data.tag.mode` is.
Planning over training time uses the step checkpoints:

```bash
for N in 5000 10000 20000 50000; do
  python eval.py --config-name=pusht policy=${ARM}_seed${S}/weights_step_$N.pt tag.mode=$MODE
done
```

## H-JEPA

H-JEPA ([Zhang et al. 2026](https://arxiv.org/abs/2610.06805),
[code](https://github.com/kevinghst/H-JEPA) at `97054ec`) adds a second level on top
of LeWM that predicts every second step from a latent action. `model=hjepa` is its Push-T
configuration (`pusht_hjepa_l2`) and works with the arms unchanged:

```bash
python train.py data=pusht +arm=$ARM seed=$S model=hjepa
```

The run is saved as `hjepa_${ARM}_seed${S}` (`hjepa` without an arm), with the same weights and pair logging as LeWM.
Level 1 sees 6 of the 7 clip frames, about 1.5 times LeWM's encoder activations, and batch 128 does not fit in
16 GB. `model.checkpoint_vit` is therefore on by default: it recomputes the ViT in the backward pass,
which leaves the maths unchanged. Set `model.checkpoint_vit=false` on a larger GPU for speed.
The schedule is LeWM's (10 epochs, our data loader); H-JEPA trained 9. Level 1 is the LeWM
`JEPA` with H-JEPA's smaller predictor (4 layers, 8 heads, 1024 feed-forward instead of
6, 16, 2048), so `measure.py` reads it unchanged. `model.levels(pixels)` returns the latents
of both levels.

`eval.py` plans with H-JEPA's top-down planner, level 2 first and level 1 toward its first subgoal:

```bash
python eval.py --config-name=pusht_hjepa policy=hjepa_${ARM}_seed${S}/weights_epoch_10.pt tag.mode=$MODE
```

H-JEPA's own Push-T protocol (50 tasks, goal 75 steps ahead, 90-step budget) is a config
override away. The tasks are sampled from `pusht_expert_train` here, not H-JEPA's fixed set from its validation split:

```bash
python eval.py --config-name=pusht_hjepa policy=... eval.goal_offset_steps=75 eval.eval_budget=90
```

The released checkpoint (`jepa-world-models/h-jepa`, `pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt`, 93 MB)
converts into a run `eval.py` can load:

```bash
hf download jepa-world-models/h-jepa pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt --local-dir $DIR
python -m scripts.convert_hjepa $DIR/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt hjepa_released
python eval.py --config-name=pusht_hjepa policy=hjepa_released/weights.pt eval.goal_offset_steps=75 eval.eval_budget=90
```

The tests compare the port with a frozen copy of H-JEPA's code in `tests/reference`;
`HJEPA_CKPT=$DIR/.../pusht_hjepa_l2_object.ckpt` adds a check on the released weights.

## MotionJEPA

`config/train/lewm_motion.yaml` adds the MotionJEPA branch
([Karmann et al. 2026](https://arxiv.org/abs/2609.23881)) to LeWM, ported from the
MIT-licensed [vivoTechResearch/motionjepa](https://github.com/vivoTechResearch/motionjepa)
at `d732c4f`. A second ViT encodes the difference of consecutive frames, and an MLP
predicts that embedding from the two image latents. The loss weights are theirs, 0.25 on
SIGReg of the latent, 2 on SIGReg of the difference embedding and 0.5 on the difference
prediction. The image encoder, planner and training loop are LeWM's, and planning costs
the main latent. The difference branch is 8-dimensional on 112-pixel difference images,
as in their benchmark config.

Their benchmark trains at batch size 64. Here the batch size stays at LeWM's 128, so the
two models are compared under one training setup. Add `loader.batch_size=64` to match theirs. Their learning-rate warmup is 0.5% of the training steps and ours is 1%, so batch 64 alone gives twice their warmup steps; at batch 128 the two warmups have the same number of steps.

Their "ours" runs also train a pixel decoder and pose probes on detached latents. These add
no encoder gradient and are left out, but they count toward the global gradient-norm clip
of 1.0 in their runs.

The arms work as they are, and the run is saved as `motion_${ARM}_seed${S}`.

```bash
python train.py --config-name=lewm_motion data=pusht +arm=$ARM seed=$S
```

Measurement and planning take `policy=motion_${ARM}_seed${S}/weights_epoch_10.pt` as above.

## What reproduces

- `tag_episode`: backbone same-tag above 0.5 and same-content within ±0.2 early in
  training, a state-probe error well above the baseline's, and planning success
  well below it.
- `tag_frame`: same-content above same-tag, with the state probe and planning
  close to the baseline.
- Both across three seeds.

## Results

Final weights (`weights_epoch_10.pt`), trained at `28a4554`. The baselines are
`pusht_seed1` and `pusht_seed2`, trained earlier with the same hyperparameters.
Planning is 50 episodes on the frames the arm trained on, clean for the
baselines. Pair cosines are backbone reads. Signal error is the state-probe
error ratio ε (0 exact, 100 chance), the mean over the five state targets of the
better of the linear and MLP probes, on the arm's own frames and on clean frames.
Effective rank is on clean frames.

| Run | Seed | Planning | same-tag | same-content | Signal error, own / clean | Effective rank |
|---|---|---|---|---|---|---|
| `pusht_seed1` (baseline) | 1 | 98% | 0.02 | 0.99 | 4.4 | 121 |
| `pusht_seed2` (baseline) | 2 | 92% | 0.01 | 0.99 | 4.4 | 122 |
| `tag_episode` | 0 | 4% | 0.95 | 0.02 | 88.8 / 46.8 | 4.1 |
| `tag_episode` | 1 | 2% | 0.81 | 0.20 | 67.2 / 52.2 | 5.9 |
| `tag_episode` | 2 | 4% | 0.93 | 0.07 | 88.7 / 62.8 | 6.2 |
| `tag_frame` | 0 | 96% | 0.01 | 1.00 | 4.3 / 4.3 | 119 |
| `tag_frame` | 1 | 94% | 0.01 | 1.00 | 4.3 / 4.3 | 121 |
| `tag_frame` | 2 | 92% | 0.00 | 1.00 | 4.5 / 4.5 | 122 |

The per-episode square costs LeWM 63 to 85 points of signal error over the
per-frame square in every seed, against a gap of 0.003 between the two clean
seeds, and its planning stays at 1 or 2 of 50 starts. The per-frame square
matches the clean baseline. Same-tag exceeds same-content for the per-episode
square in every seed, but seed 1 ends at same-content 0.2005, just outside the
±0.2 band above. In `tag_episode` the pair cosines cross between steps 600 and
900 (seed 0: same-tag 0.25 and same-content 0.71 at step 600, same-tag first
above 0.5 at step 900, 0.78 and 0.19 by step 1400; seed 1: same-tag first above
0.5 at step 700), later than the first-measurement swap Enigma reports. The step
checkpoints are every 5000 steps, where Enigma saves every 500 and plans every
2000, so planning before step 5000 is not scored here. Seed 0 plans from
2%, 6% and 10% of starts at steps 5k, 15k and 25k. On clean frames the per-episode
models collapse to four to six dimensions with 94% to 99% of the variance in the
top two, so without its colour the embedding keeps little of the scene. Unlike
Enigma's report, the per-episode models' validation prediction loss is above the
per-frame models' (0.006 to 0.026 against 0.003 to 0.004).

Seed 0 and the baselines were scored at `28a4554`, seeds 1 and 2 at `73d9c1e`,
which reads each run's own tag seed.

## Where the pair frames come from

Enigma renders its pair frames fresh from the simulator. Here they are drawn from
the dataset, one frame from each of 256 episodes, so the encoder may have seen them
in training.
