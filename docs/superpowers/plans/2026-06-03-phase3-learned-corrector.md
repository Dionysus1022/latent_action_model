# Phase3 Learned Corrector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a learned corrector mode that repairs drifted remaining action chunks without full diffusion replanning.

**Architecture:** Add a small corrector model and bundle format under `diffusion/`, train it from existing planner datasets with synthetic action drift, then integrate it into `DiffusionPlannerPolicy` for `corrective.mode=learned`. Existing `none` and `replan` modes remain unchanged.

**Tech Stack:** Python, PyTorch, Hydra/OmegaConf, pytest/unittest, existing LeWM latent rollout helper.

---

### Task 1: Corrector Model And Bundle

**Files:**
- Create: `diffusion/corrector.py`
- Create: `tests/test_action_chunk_corrector.py`

- [ ] Write tests for output shape, residual behavior, and bundle round-trip.
- [ ] Run `./.venv/bin/python -m pytest tests/test_action_chunk_corrector.py -q` and verify the tests fail because `diffusion.corrector` is missing.
- [ ] Implement `ActionChunkCorrectorConfig`, `ActionChunkCorrector`, `CorrectorBundle`, `save_corrector_bundle`, and `load_corrector_bundle`.
- [ ] Re-run the test file and verify it passes.

### Task 2: Corrector Training Dataset And Loss

**Files:**
- Modify: `diffusion/corrector_training.py`
- Create: `tests/test_corrector_training.py`

- [ ] Write tests that build a tiny dataset bundle and verify the noisy prefix/remainder differ from the clean target.
- [ ] Write tests for finite one-batch loss with a fake additive world model.
- [ ] Run `./.venv/bin/python -m pytest tests/test_corrector_training.py -q` and verify failing tests.
- [ ] Implement dataset sample construction, rollout-based goal loss, smoothness loss, train/eval epoch helpers, and Hydra-compatible `hydra_main`.
- [ ] Re-run the test file and verify it passes.

### Task 3: Learned Policy Runtime

**Files:**
- Modify: `diffusion/policy.py`
- Modify: `tests/test_diffusion_policy_prediction_error.py`

- [ ] Write tests that inject a fake corrector and verify only triggered env rows in the future action buffer are changed.
- [ ] Write tests that learned mode raises if no corrector is loaded.
- [ ] Run the policy tests and verify failing tests.
- [ ] Add corrector constructor/from-bundle arguments and learned checkpoint handling.
- [ ] Add correction stats: count, norm, action delta norm, and total correction time.
- [ ] Re-run policy tests and verify they pass.

### Task 4: Config, Eval, And CLI

**Files:**
- Create: `train_corrective_diffusion.py`
- Create: `config/corrector/train.yaml`
- Create: `config/corrector/task/pusht.yaml`
- Create: `config/corrector/task/tworoom.yaml`
- Modify: `config/eval/planner/diffusion.yaml`
- Modify: `eval.py`
- Modify: `tests/test_eval_corrective_config.py`

- [ ] Write config tests for `corrective.mode=learned` and `corrective.corrector_path`.
- [ ] Run config tests and verify failing tests.
- [ ] Add config parsing and policy wiring for `corrector_path`.
- [ ] Add training CLI/config defaults.
- [ ] Add eval result fields for correction stats.
- [ ] Re-run config tests and verify they pass.

### Task 5: Documentation And Verification

**Files:**
- Modify: `README.md`

- [ ] Add Phase3 training and eval commands.
- [ ] Run focused tests:

```bash
./.venv/bin/python -m pytest \
  tests/test_action_chunk_corrector.py \
  tests/test_corrector_training.py \
  tests/test_diffusion_policy_prediction_error.py \
  tests/test_eval_corrective_config.py -q
```

- [ ] Run compile check:

```bash
./.venv/bin/python -m py_compile \
  diffusion/corrector.py \
  diffusion/corrector_training.py \
  diffusion/policy.py \
  eval.py \
  train_corrective_diffusion.py
```

- [ ] Run Hydra config dry checks for PushT learned mode.
