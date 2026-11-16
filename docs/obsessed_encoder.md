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
uv run train.py data=pusht +arm=$ARM seed=$S
```

The run is saved as `${ARM}_seed${S}`. Measurement and planning read the tag
seed from the run's `config.yaml` and need the tag mode set explicitly. Use the
arm's own mode to score on the frames it trained on, or `off` for clean frames.

```bash
RUN=${ARM}_seed${S}/weights_epoch_10.pt
uv run measure.py --config-name=pusht policy=$RUN data.tag.mode=$MODE
uv run eval.py --config-name=pusht policy=$RUN tag.mode=$MODE
```

`measure.py` reports the pair cosines (`pairs`), the state probes and the
effective rank. The pair cosines always stamp a tag, whatever `data.tag.mode` is.
Planning over training time uses the step checkpoints:

```bash
for N in 5000 10000 25000 50000 120000; do
  uv run eval.py --config-name=pusht policy=${ARM}_seed${S}/weights_step_$N.pt tag.mode=$MODE
done
```

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

The per-episode square costs LeWM 62.9 to 84.5 points of signal error over the
per-frame square in every seed, against a gap of 0.003 between the two clean
seeds, and its planning stays at 1 or 2 of 50 starts. The per-frame square
matches the clean baseline. Same-tag exceeds same-content for the per-episode
square in every seed, but seed 1 ends at same-content 0.2005, just outside the
±0.2 band above. On clean frames the per-episode models collapse to four to six
dimensions.

Seed 0 and the baselines were scored at `28a4554`, seeds 1 and 2 at `73d9c1e`,
which reads each run's own tag seed.

## Chance floor

A random-action policy succeeds on 0 to 2% of the 50 starts, so the per-episode
models' 1 or 2 successes of 50 at epoch 10 sit at this floor.

## Planning over training

Success in percent on the 50 starts, on the arm's own frames, for seeds 0, 1, 2.
One start is 2 points. The checkpoints are every 5000 steps where Enigma saves every
500 and plans every 2000, so planning before step 5000 is not scored here.

| Step | `tag_episode` | `tag_frame` |
|---|---|---|
| 5k | 0, 0, 2 | 10, 16, 2 |
| 10k | 4, 6, 8 | 18, 28, 36 |
| 25k | 10, 6, 4 | 84, 78, 88 |
| 50k | 6, 4, 2 | 94, 94, 88 |
| 120k | 2, 4, 4 | 94, 94, 98 |

The per-frame arm is at 78 to 88% by step 25k. Over all saved steps the per-episode
arm peaks at 14% or less, between steps 10k and 60k, and from step 80k it is back
at 0 to 6%, close to the floor.

## Epoch 5 readout

Weights at epoch 5 (step 69,664), scored at `73d9c1e` on the arm's own frames,
effective rank included.

| Run | Seed | Planning | same-tag | same-content | Signal error | Effective rank |
|---|---|---|---|---|---|---|
| `tag_episode` | 0 | 4% | 0.88 | 0.10 | 63.5 | 102 |
| `tag_episode` | 1 | 4% | 0.79 | 0.20 | 64.7 | 42 |
| `tag_episode` | 2 | 4% | 0.92 | 0.07 | 83.7 | 67 |
| `tag_frame` | 0 | 88% | 0.01 | 1.00 | 4.6 | 113 |
| `tag_frame` | 1 | 92% | 0.01 | 1.00 | 4.4 | 115 |
| `tag_frame` | 2 | 92% | 0.00 | 1.00 | 4.7 | 116 |

The per-episode models already plan at the floor, with signal error 63.5 to 83.7
against 67.2 to 88.8 at epoch 10. The per-frame models plan 88 to 92% against 92 to 96%.

## Swap timing

In `tag_episode` same-tag first passes 0.5 at step 900 in seed 0 and step 700 in
seed 1, where Enigma reports the swap at its first measurement. Enigma's pair code
on a fresh seed-0 run puts it between steps 600 and 700, so the measurement does
not explain the later swap. With the learning-rate schedule shortened to 1000
steps (`+trainer.max_steps=1000`) it moves to between steps 200 and 300, so the
swap follows the schedule. Whether Enigma's step axis matches ours is open.

## Prediction loss

On the smoothed training `fit/pred_loss` of Enigma's figure 8, the per-episode
loss is below the per-frame loss early in training in every seed (mean 0.0087
against 0.019 at step 25k). From step 61k seeds 1 and 2 stay above it and end at
0.019 and 0.030 against 0.007, while seed 0 ends below at 0.0066. The epoch-end
validation loss ends above the per-frame loss in every seed (0.006 to 0.026
against 0.003 to 0.004).

## Where the pair frames come from

Enigma renders its pair frames fresh from the simulator. Here they are drawn from
the dataset, one frame from each of 256 episodes, so the encoder may have seen them
in training.

Enigma's pair code (`PairAnalysisCallback`) was run on the epoch-10 weights of the
six arm runs and both baselines. Its same-tag and same-content cosines, backbone
and projection, differ from the dataset-frame values by at most 0.024 (0.021 on
the backbone).
