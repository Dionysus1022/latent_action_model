# Phase3 Learned Corrector Design

## Goal

Phase3 adds a learned action-chunk corrector for the closed-loop diffusion planner. It should repair only the remaining actions of envs whose LeWM prediction error crosses the corrective threshold, instead of running a full diffusion replan.

## Approach

The first version uses the existing diffusion planner dataset (`z_cur`, `z_goal`, `teacher_plan`, `meta`) and synthetic action drift. For each sample, it splits the expert chunk at `correction_interval`, rolls out a clean prefix and a noisy prefix with frozen LeWM dynamics, and trains a small MLP to map:

```text
[z_real_like, z_goal, z_real_like - z_pred, flatten(noisy_remainder)] -> clean_remainder
```

The noisy prefix provides the drift signal. The noisy remainder prevents the action loss from becoming an identity objective.

## Components

- `diffusion/corrector.py`
  - `ActionChunkCorrectorConfig`
  - `ActionChunkCorrector`
  - save/load helpers for corrector bundles
- `diffusion/corrector_training.py`
  - `CorrectorTrainingDataset`
  - train/eval loop
  - Hydra entrypoint
- `train_corrective_diffusion.py`
  - top-level training command
- `diffusion/policy.py`
  - load a corrector when `corrective.mode=learned`
  - at corrective checkpoints, apply corrected remaining actions only to triggered envs
  - keep `none` and `replan` behavior unchanged
- `eval.py`
  - parse `corrective.corrector_path`
  - print/write correction count, correction norm, action delta norm, and correction time

## Runtime Behavior

At a checkpoint:

```text
1. Compute z_real, z_pred, error_latent, prediction_error.
2. Select triggered envs with the same per-env threshold logic as Phase2.
3. Slice current plan remainder from the action buffer.
4. Run the corrector on triggered envs.
5. Replace only those env rows in future buffered actions and current plan.
```

The corrector does not call the diffusion planner at runtime, so `global_planning_calls` should not increase at corrective checkpoints.

## Training Loss

The first version uses:

```text
L = lambda_action * smooth_l1(u_corr, clean_remainder)
  + lambda_goal * mse(F(z_real_like, u_corr), z_goal)
  + lambda_smooth * mean(||u_corr[t+1] - u_corr[t]||^2)
```

`lambda_goal` can be set to `0` for a fast smoke test. LeWM stays frozen.

## Validation

- Model shape tests for batch size 1 and >1.
- Dataset sample tests verify noisy remainder differs from target.
- Eval config tests verify learned mode resolves `corrector_path`.
- Policy tests verify learned correction only changes triggered env rows.
- Training smoke test verifies one optimization step returns finite losses.
