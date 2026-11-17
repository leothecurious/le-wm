# H-JEPA integration

## Scope

The two-level Push-T H-JEPA from [the official implementation](https://github.com/kevinghst/H-JEPA/tree/97054ecaff83e13839195ec4b92d74c08a0127c5),
config `pusht_hjepa_l2`. The source commit is
`97054ecaff83e13839195ec4b92d74c08a0127c5`; attribution is in `LICENSE.hjepa`.
This port does not implement its deeper hierarchies, temporal state windows, proprioceptive fusion, or multi-step training variants.

## Interfaces

- Level 1 inherits `JEPA`: `encode`, `predict`, `rollout`, and `get_cost` retain their ordinary meanings. Encoding returns **every input frame**, even for a seven-frame window. Existing level-1 metrics and the common LeWM planner need no model-specific branch.
- `encode_hierarchy` is an explicit training operation. With the defaults, a clip has seven frames at frameskip 5. Level 1 keeps a randomly located contiguous four-frame crop; level 2 reads frames 0, 2, 4, 6. Only their union is image-encoded, **before BatchNorm projection**, as in the official sparse path.
- The upper encoder maps level-1 latents through LayerNorm → Linear → GELU → Linear. Its transformer pools each pair of level-1 action embeddings into a four-dimensional latent action. The final partial chunk is zero-padded and masked.
- Each level predicts the next latent at three context positions. Neither target latents nor the connection between levels are detached. Each level adds SIGReg; the two level losses are summed. The persistent latent-action queue records the first pooled action of each training clip and supplies the hierarchical planner's prior.
- `levels(pixels)` returns full level-1 latents and strided level-2 latents. The existing measurement suite scores level 1; it does not silently switch to the upper representation.

`hjepa.py` owns the hierarchy and its objective; `train.py` selects it through `loss.kind: hjepa`. Model YAML contains only architecture. The top-level training recipe owns clip geometry, loss settings, and run naming. The optimizer boundary uses `param_groups` to keep learning rates at 5e-5 and 1e-5. Baseline LeWM and MotionJEPA keep their existing forward paths.

## Training and ablations

```bash
uv run train.py --config-name=hjepa +arm=baseline seed=0
uv run train.py --config-name=hjepa +arm=tag_episode seed=0
uv run train.py --config-name=hjepa +arm=tag_frame seed=0
uv run train.py --config-name=hjepa +arm=randgoal seed=0
```

Runs are named `hjepa_${ARM}_seed${seed}`, including `randgoal`. All use the same dataset wrappers, preprocessing, checkpointing, seeding, and logging as the fork. No separate H-JEPA training pipeline is introduced.

Controlled objective ablations change one setting from the chosen H-JEPA run:

- `loss.hjepa.weight=0`: remove the upper objective; retain the sparse-frame geometry and architecture. This is **not** the flat LeWM baseline.
- `loss.hjepa.sigreg_weight=0`: remove upper SIGReg only.
- `model.upper_lr=5e-5`: match the upper learning rate to the lower one.

`model.checkpoint_vit=true` enables activation recomputation when memory is limiting. The default is off. This is a runtime option, not a different objective.

The training defaults intentionally match the fork's study setup, not the paper's full experiment: 10 epochs and seed 3072 before overrides, a 90/10 clip split, and the fork's preprocessing/runtime. The official Push-T recipe uses 9 epochs, seed 42 before overrides, and separate `pusht_expert_train` / `pusht_expert_val` datasets. The port retains the official smaller level-1 predictor (depth 4, 8 heads, MLP 1024), rather than LeWM's depth 6 / 16 heads / MLP 2048. A comparison against default LeWM therefore changes architecture as well as hierarchy; it is not a one-variable hierarchy ablation. Supported validation and training here are single-device; the fork's SIGReg is single-GPU.

## Measurement and planning

```bash
uv run measure.py --config-name=pusht policy=hjepa_baseline_seed0/weights_epoch_10.pt
uv run eval.py --config-name=pusht policy=hjepa_baseline_seed0/weights_epoch_10.pt
```

Use the common planner above for model comparisons. Match `tag.mode`, evaluation data, seeds, and budgets across models; tag seeds are read from each saved run as usual.

The optional official two-level planner is explicit:

```bash
uv run eval.py --config-name=pusht_hjepa policy=hjepa_baseline_seed0/weights_epoch_10.pt
```

It optimizes upper latent actions and uses the first predicted upper state/action as the lower subgoal. Its horizon decays with the evaluation budget. Its optimizer, candidate counts, action prior/clipping, and action-matching cost differ from the common LeWM planner, so changing to it is a separate planner comparison. The evaluation task defaults remain the fork's; `eval.goal_offset_steps=75 eval.eval_budget=90` selects the official task lengths, but does not recreate its fixed validation tasks, normalizers, or simulator version.

Released model-object checkpoints are used only for one-off port validation. There is no maintained pickle-conversion path: evaluating such weights against the fork's recomputed action normalizers would not establish official planning reproducibility.

## Noise studies (#8)

Noise belongs to the dataset wrapper, not the hierarchy. Keep #8's tag → noise wrapper order, seed handling, and measurement preprocessing. Its `lewm_forward` must dispatch `loss.kind: hjepa` to `training_losses` before ordinary encoding, with the same action NaN handling and logging as this branch; retain the optimizer parameter-group hook. Pixel/grounded objectives are not implemented by H-JEPA and must not be selected implicitly. The unmerged #8 branch is not included here.

## Validation

Direct parity was run on CPU in float32 with PyTorch `2.11.0+cu130` and Transformers `4.57.6`. The pinned official classes and `hjepa_forward` ran in the fork's environment, not in a separate official environment. Inputs were synthetic normalized pixels/actions shared by both implementations.

- **Training:** a one-block image encoder and one-block predictors, with the canonical latent dimensions and action-pooling module. Nonzero AdaLN weights exercised action gradients. Across all four possible crop starts, both levels' embeddings, action embeddings, prediction/SIGReg losses, and all 111 participating parameter gradients matched exactly. After global norm clipping at 1.0 and one AdamW update at the two configured learning rates, the largest state difference was `5.960464477539063e-08`. Queue and BatchNorm buffers were included in that comparison.
- **Released weights:** the complete seed-42 `pusht_hjepa_l2_object.ckpt` from Hugging Face `jepa-world-models/h-jepa`, revision `08fc8a21208671a87741f2960aeba9b5db4420e0`, loaded strictly after name mapping. Its SHA256 is `3fea307b2f86ab5f4097fdd06ecf3c52bbb41355b689cf792606ab70e711bc36`. At the full 224-pixel, 12-block image-encoder architecture, both levels' embeddings/action embeddings and all loss terms matched exactly in evaluation mode. The latent-action queue had shape `(4000, 4)`.
- **Native training/checkpoints:** `train.run` completed two CPU training steps and validation on a recorded synthetic HDF5 dataset with an episode tag. Both optimizer groups had the intended initial learning rates. The fork's `save_pretrained` / `load_pretrained` path loaded the model with its populated queue and preserved seven-frame encoding. A second invocation resumed the full Lightning checkpoint to global step 4; the queue grew from `(4, 4)` to `(8, 4)`.
- **Planner:** direct comparisons to the official `LevelCostModel` covered native level-1 and level-2 costs and projected level-1 costs with action matching, at horizons 2 and 4, with batch 2 and three candidates. Maximum cost difference was `6.103515625e-05`; maximum action-gradient difference was `1.0151416063308716e-07`. A tiny CPU hierarchical solve completed the full 90-step horizon schedule, including its one-step tail. These checks establish cost/gradient and orchestration behavior, not complete stochastic solver or environment-result equivalence.
- **Regression suite:** `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run pytest -q` passed: **29 tests**. Plugin autoload was disabled to exclude the workstation's unrelated ROS pytest plugin. `git diff --check` passed.
- **Maintained coverage:** three focused behavioral tests protect upper-to-lower/target/action gradients, weighted objective composition, queue checkpoint loading, full-frame metric extraction at the hierarchy's training-window length, and action-cost alignment when the lower plan outlasts its subgoal-action sequence. The official reference is not vendored into the test suite.

The one-off comparison was executed with `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run /tmp/hjepa-integration-review/actual_parity.py`; the native smoke used `STABLEWM_HOME=/tmp/hjepa-integration-review/cache uv run /tmp/hjepa-integration-review/validate_native.py` (its subsequent planner check was rerun separately); machine-local source, scripts, JSON measurements, and logs are under `/tmp/hjepa-integration-review/`. The direct reference check is validation evidence, not a second maintained implementation.

Numerical port parity is distinct from reproducing training curves or planning success. This does not validate the official dependency stack, multi-GPU SIGReg, mixed-precision training, full-budget optimization, dataset/task equivalence, or the paper's success rates. No GPU ablation was launched.
