# Trajectory Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add trajectory-quality metrics to `eval.py` and validate them on cube, pusht, tworoom, and reacher.

**Architecture:** Keep metric math in a small standalone module, then add an optional trajectory-recording evaluation path in `eval.py`. Reuse the existing dataset-conditioned reset/callable logic and preserve the existing success-rate semantics.

**Tech Stack:** Python, NumPy, Hydra/OmegaConf, stable_worldmodel, pytest.

---

### Task 1: Pure Trajectory Metrics

**Files:**
- Create: `trajectory_quality.py`
- Create: `tests/test_trajectory_quality.py`

- [ ] Write failing tests for straight paths, action smoothness, and task-specific goal distances.
- [ ] Implement `compute_trajectory_quality(...)` and task-specific distance helpers.
- [ ] Run `./.venv/bin/python -m pytest tests/test_trajectory_quality.py -q`.

### Task 2: Eval Integration

**Files:**
- Modify: `eval.py`

- [ ] Add `trajectory_quality` defaults and config resolution.
- [ ] Add a local evaluation loop that records per-step actions, task states, goal states, success, and video frames.
- [ ] Print and append `[trajectory-quality]` metrics when enabled.
- [ ] Save raw trajectory arrays to `trajectory_quality_<task>.npz` when requested.

### Task 3: Smoke Tests

**Files:**
- Modify: `tests/test_trajectory_quality.py` or add eval-specific tests if needed.

- [ ] Run pure unit tests.
- [ ] Run smoke eval on each task with `eval.num_eval=5 trajectory_quality.enabled=true`.
- [ ] If full smoke is too slow for all tasks, report exactly which commands completed and which were not run.
