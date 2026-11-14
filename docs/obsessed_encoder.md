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
two models are compared under one training setup. Add `loader.batch_size=64` to match theirs.

The arms are unchanged. The run needs its own name, or it would write over the LeWM run.

```bash
python train.py --config-name=lewm_motion data=pusht +arm=$ARM seed=$S output_model_name=motion_${ARM}_seed${S}
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
baselines. The state probe is the mean over the five state targets, on clean
frames unless marked. Pair cosines are backbone reads.

| Run | Planning success | same-tag | same-content | State probe R² (linear / MLP) | Effective rank |
|---|---|---|---|---|---|
| `pusht_seed1` (baseline) | 98% | 0.02 | 0.99 | 0.95 / 0.99 | 121 |
| `pusht_seed2` (baseline) | 92% | 0.01 | 0.99 | 0.94 / 0.99 | 122 |
| `tag_episode` seed 0 | 4% | 0.95 | 0.02 | 0.62 / 0.71 | 4.0 |
| `tag_episode` seed 0, tagged frames | | | | 0.18 / −1.34 | 115 |
| `tag_frame` seed 0 | 96% | 0.01 | 1.00 | 0.96 / 0.99 | 119 |

Seed 0 reproduces both arms. In `tag_episode` the pair cosines swap at about step 600
(same-tag 0.25, same-content 0.71 at step 600) and by step 1400 they are
0.78 and 0.19, against 0.84 and 0.13 at step 25k. Planning is 2%, 6% and 10%
at steps 5k, 15k and 25k. On clean frames its latent collapses to about four
dimensions with 99% of the variance in the top two. The MLP probe on tagged
frames is below the control and is the least stable number here, so read the
linear one.

Seeds 1 and 2 of both arms were still training on 14 Nov and will be added.

## Where the pair frames come from

Enigma renders its pair frames fresh from the simulator. Here they are drawn from
the dataset, one frame from each of 256 episodes, so the encoder may have seen them
in training.
