#!/usr/bin/env python3
"""Evaluate GC-IDM vs CEM planning on LeWM.

Usage:
    python eval_idm.py --dataset tworoom --idm ./checkpoints/tworoom_gcidm.pt
    python eval_idm.py --dataset tworoom --cem-only
    python eval_idm.py --dataset tworoom --idm ./checkpoints/tworoom_gcidm.pt --compare
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if (WORKSPACE_ROOT / "eval.py").exists() and str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import torch
import stable_worldmodel as swm
import stable_pretraining as spt
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
from torchvision import tv_tensors

from idm.dataset import load_lewm_model
from idm.model import IDMConfig
from train_idm import load_idm
from local_data_paths import hdf5_name_and_cache_dir, resolve_dataset_h5


class SolverTimingWrapper:
    """Record actual solver-call time, not buffered policy action time."""

    def __init__(self, solver, device: torch.device) -> None:
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "_num_replans", 0)
        object.__setattr__(self, "_planning_time_total_sec", 0.0)

    def __getattr__(self, name):
        return getattr(self.solver, name)

    def __setattr__(self, name, value):
        if name in {"solver", "device", "_num_replans", "_planning_time_total_sec"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.solver, name, value)

    def __call__(self, *args, **kwargs):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            return self.solver(*args, **kwargs)
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self._planning_time_total_sec += time.perf_counter() - start
            self._num_replans += 1


def _episode_successes_text(metrics: dict) -> str | None:
    if "episode_successes" not in metrics:
        return None
    successes = np.asarray(metrics["episode_successes"], dtype=bool).reshape(-1)
    return ",".join("1" if bool(value) else "0" for value in successes)


def _print_trajectory_quality(quality: dict | None) -> None:
    if quality is None:
        return
    summary = quality["summary"]
    for key in [
        "final_goal_distance_mean",
        "min_goal_distance_mean",
        "path_length_mean",
        "straight_line_ratio_mean",
        "action_l2_mean_mean",
        "action_delta_l2_mean_mean",
        "action_jerk_l2_mean_mean",
        "steps_to_success_mean",
        "latent_monotonicity_mean",
        "latent_monotonic_step_fraction_mean",
        "final_latent_goal_distance_mean",
        "min_latent_goal_distance_mean",
    ]:
        if key in summary:
            print(f"[trajectory-quality] {key}={float(summary[key]):.6f}")


def make_lgbs_latent_encoder(model, device: torch.device, batch_size: int = 256):
    jepa = model.model if hasattr(model, "model") else model
    encoder = jepa.encoder
    projector = jepa.projector

    @torch.no_grad()
    def encode_pair(pixels, goals):
        def _encode(array):
            np_array = np.asarray(array)
            if np_array.ndim != 5:
                raise ValueError(f"Expected pixels with shape [episodes, steps, H, W, C], got {np_array.shape}.")
            episode_count, step_count = np_array.shape[:2]
            flat = np_array.reshape(episode_count * step_count, *np_array.shape[2:])
            if flat.shape[-1] == 3:
                flat = np.transpose(flat, (0, 3, 1, 2))
            tensor = torch.from_numpy(flat).float().to(device) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            tensor = (tensor - mean) / std
            chunks = []
            for start in range(0, tensor.shape[0], int(batch_size)):
                end = min(tensor.shape[0], start + int(batch_size))
                enc_out = encoder(tensor[start:end], interpolate_pos_encoding=True)
                chunks.append(projector(enc_out.last_hidden_state[:, 0]).detach().cpu().numpy())
            encoded = np.concatenate(chunks, axis=0)
            return encoded.reshape(episode_count, step_count, encoded.shape[-1])

        return _encode(pixels), _encode(goals)

    return encode_pair


def evaluate_lgbs_policy(
    *,
    world,
    dataset,
    start_steps,
    goal_offset,
    eval_budget,
    episodes_idx,
    callables,
    task: str,
    trajectory_quality: bool = False,
    save_video: bool = False,
    video_path: str | Path | None = None,
    latent_encoder=None,
):
    if not trajectory_quality:
        metrics = evaluate_from_dataset_compat(
            world,
            dataset=dataset,
            start_steps=start_steps,
            goal_offset=goal_offset,
            eval_budget=eval_budget,
            episodes_idx=episodes_idx,
            callables=callables,
        )
        return metrics, None

    from eval import run_evaluation_with_trajectory_quality

    quality_cfg = {
        "enabled": True,
        "save_npz": False,
        "truncate_after_success": True,
        "save_video": bool(save_video),
    }
    metrics, quality = run_evaluation_with_trajectory_quality(
        world=world,
        dataset=dataset,
        episodes_idx=episodes_idx,
        start_steps=start_steps,
        goal_offset_steps=goal_offset,
        eval_budget=eval_budget,
        callables=callables,
        video_path=Path(video_path or "/tmp/lgbs_trajectory_quality"),
        task=task,
        quality_cfg=quality_cfg,
        latent_encoder=latent_encoder,
    )
    return metrics, quality


def print_unified_eval_summary(
    *,
    task: str,
    planner_type: str,
    solver: str,
    checkpoint: str,
    metrics: dict,
    evaluation_time_sec: float,
    planning_time_total_sec: float,
    global_planning_calls: int,
    effective_replans_per_episode: int,
    quality: dict | None = None,
    idm_path: str | None = None,
) -> None:
    planner_line = (
        f"[planner] type={planner_type} task={task} solver={solver} "
        f"policy={checkpoint}"
    )
    if idm_path is not None:
        planner_line += f" idm={idm_path}"
    print(planner_line)

    success_rate = metrics.get("success_rate", None)
    if success_rate is not None:
        print(f"[summary] success_rate={float(success_rate):.4f}")
    episode_successes = _episode_successes_text(metrics)
    if episode_successes is not None:
        print(f"[summary] episode_successes={episode_successes}")
    print(f"[summary] evaluation_time={float(evaluation_time_sec):.4f}s")
    _print_trajectory_quality(quality)

    avg_planning_time_sec = (
        float(planning_time_total_sec) / float(global_planning_calls)
        if int(global_planning_calls) > 0
        else 0.0
    )
    print(f"[planner-stats] global_planning_calls={int(global_planning_calls)}")
    print(f"[planner-stats] effective_replans_per_episode={int(effective_replans_per_episode)}")
    print(f"[planner-stats] planning_time_total_sec={float(planning_time_total_sec):.6f}")
    print(f"[planner-stats] avg_planning_time_sec={avg_planning_time_sec:.6f}")

    ms_per_episode = 1000.0 * float(evaluation_time_sec) / max(1, len(np.asarray(metrics.get("episode_successes", []))))
    if not math.isfinite(ms_per_episode) or ms_per_episode == 0.0:
        ms_per_episode = 0.0
    print(
        f"[lgbs-stats] solver={solver} "
        f"ms_per_episode={ms_per_episode:.6f} "
        f"ms_per_plan={1000.0 * avg_planning_time_sec:.6f} "
        f"total_time={float(evaluation_time_sec):.6f}"
    )


# ---------------------------------------------------------------------------
# Config per dataset
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    env: str
    dataset_name: str
    cache_keys: list[str]
    img_size: int
    embed_dim: int
    action_block: int
    raw_action_dim: int
    callables: list[dict]
    world_kwargs: dict | None = None


DATASET_CONFIGS = {
    "pusht": EvalConfig(
        env="swm/PushT-v1",
        dataset_name="pusht_expert_train",
        cache_keys=["action", "proprio", "state"],
        img_size=224, embed_dim=192, action_block=1, raw_action_dim=2,
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
        ],
    ),
    "tworoom": EvalConfig(
        env="swm/TwoRoom-v1",
        dataset_name="tworoom",
        cache_keys=["action", "proprio"],
        img_size=224, embed_dim=192, action_block=1, raw_action_dim=2,
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "proprio"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_proprio"}}},
        ],
    ),
    "cube": EvalConfig(
        env="swm/OGBCube-v0",
        dataset_name="ogbench/cube_single_expert",
        cache_keys=["action"],
        img_size=224, embed_dim=192, action_block=1, raw_action_dim=5,
        callables=[
            {"method": "set_state", "args": {
                "qpos": {"value": "qpos"}, "qvel": {"value": "qvel"},
            }},
            {"method": "set_target_pos", "args": {
                "cube_id": {"value": 0, "in_dataset": False},
                "target_pos": {"value": "goal_privileged_block_0_pos"},
                "target_quat": {"value": "goal_privileged_block_0_quat"},
            }},
        ],
        world_kwargs={"env_type": "single", "ob_type": "states", "multiview": False,
                       "width": 224, "height": 224, "visualize_info": False,
                       "terminate_at_goal": True},
    ),
    "reacher": EvalConfig(
        env="swm/ReacherDMControl-v0",
        dataset_name="dmc/reacher_random",
        cache_keys=["action"],
        img_size=224, embed_dim=192, action_block=1, raw_action_dim=2,
        callables=[
            {"method": "set_state", "args": {
                "qpos": {"value": "qpos"}, "qvel": {"value": "qvel"},
            }},
            {"method": "set_target_qpos", "args": {
                "target_qpos": {"value": "goal_qpos"},
            }},
        ],
        world_kwargs={"task": "qpos_match"},
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_img_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


def build_normalizers(dataset, keys):
    process = {}
    for col in keys:
        if col == "pixels":
            continue
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        scaler = preprocessing.StandardScaler().fit(col_data)
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler
    return process


def load_eval_dataset(task: str, dcfg: EvalConfig, dataset_h5: str | None = None):
    dataset_path = resolve_dataset_h5(task, dataset_h5)
    dataset_name, cache_dir = hdf5_name_and_cache_dir(dataset_path)
    print(f"[dataset] task={task} h5={dataset_path}")
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=dcfg.cache_keys,
        cache_dir=cache_dir,
    )


def resolve_checkpoint_path_for_autocost(checkpoint_path: str) -> str:
    path = Path(checkpoint_path).expanduser()
    suffix = "_object.ckpt"
    if path.is_file() and path.name.endswith(suffix):
        return str(path.with_name(path.name[: -len(suffix)]))
    return str(path)


def evaluate_from_dataset_compat(
    world,
    *,
    dataset,
    start_steps,
    goal_offset,
    eval_budget,
    episodes_idx,
    callables,
):
    try:
        return world.evaluate(
            dataset=dataset,
            start_steps=start_steps,
            goal_offset=goal_offset,
            eval_budget=eval_budget,
            episodes_idx=episodes_idx,
            callables=callables,
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return world.evaluate_from_dataset(
            dataset=dataset,
            episodes_idx=episodes_idx,
            start_steps=start_steps,
            goal_offset_steps=goal_offset,
            eval_budget=eval_budget,
            callables=callables,
            save_video=False,
        )


def sample_eval_episodes(dataset, num_eval, goal_offset, seed=42,
                         train_split=1.0, split_seed=42):
    """Sample eval episodes. If train_split < 1.0, restrict to held-out episodes only."""
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    ep_indices = np.unique(episode_idx)

    episode_len = np.array([
        np.max(step_idx[episode_idx == ep]) + 1 for ep in ep_indices
    ])
    max_start = episode_len - goal_offset - 1
    max_start_dict = {ep: mx for ep, mx in zip(ep_indices, max_start)}
    max_start_per_row = np.array([max_start_dict[ep] for ep in episode_idx])
    valid_mask = step_idx <= max_start_per_row

    # If train_split < 1.0, restrict to held-out episodes
    if train_split < 1.0:
        n_holdout = max(1, round(len(ep_indices) * (1.0 - train_split)))
        rng_split = np.random.default_rng(split_seed)
        holdout_eps = rng_split.choice(ep_indices, size=n_holdout, replace=False)
        valid_mask = valid_mask & np.isin(episode_idx, holdout_eps)
        print(f"  Eval restricted to {len(holdout_eps)} held-out episodes "
              f"(split_seed={split_seed}, train_split={train_split})")

    valid_indices = np.nonzero(valid_mask)[0]

    if len(valid_indices) == 0:
        raise RuntimeError(
            f"No valid eval starts (goal_offset={goal_offset}, "
            f"train_split={train_split})")

    rng = np.random.default_rng(seed)
    n = min(num_eval, len(valid_indices) - 1)
    if n < num_eval:
        print(f"  WARNING: reduced num_eval {num_eval} -> {n} "
              f"(only {len(valid_indices)} valid starts)")
    sampled = rng.choice(len(valid_indices) - 1, size=n, replace=False)
    sampled = np.sort(valid_indices[sampled])

    eval_episodes = dataset.get_row_data(sampled)[col_name]
    eval_start_idx = dataset.get_row_data(sampled)["step_idx"]
    return eval_episodes, eval_start_idx


# ---------------------------------------------------------------------------
# GoalConditionedPolicy
# ---------------------------------------------------------------------------

class GoalConditionedPolicy:
    """One forward pass per step: encode obs + goal → IDM → action."""

    def __init__(self, jepa_model, idm, eval_budget=50,
                 process=None, transform=None,
                 device=torch.device("cpu"), cache_goal_encoding=True):
        self.jepa = jepa_model
        self.idm = idm
        self.eval_budget = eval_budget
        self.process = process or {}
        self.transform = transform or {}
        self.device = device
        self._step_count = 0
        self._plan_times: list[float] = []
        self.cache_goal_encoding = cache_goal_encoding
        self._cached_goal_input = None
        self._cached_z_goal = None

    def set_env(self, env):
        self.env = env

    def _prepare_info(self, info_dict):
        out = {}
        for k, v in list(info_dict.items()):
            is_numpy = isinstance(v, (np.ndarray, np.generic))

            if k in self.process and is_numpy:
                shape = v.shape
                if len(shape) > 2:
                    v = v.reshape(-1, *shape[2:])
                v = self.process[k].transform(v)
                v = v.reshape(shape)

            if k in self.transform:
                shape = None
                if is_numpy or torch.is_tensor(v):
                    if v.ndim > 2:
                        shape = v.shape
                        v = v.reshape(-1, *shape[2:])
                if k.startswith("pixels") or k.startswith("goal"):
                    if is_numpy:
                        v = np.transpose(v, (0, 3, 1, 2))
                    else:
                        v = v.permute(0, 3, 1, 2)
                v = torch.stack([self.transform[k](tv_tensors.Image(x)) for x in v])
                is_numpy = isinstance(v, (np.ndarray, np.generic))
                if shape is not None:
                    v = v.reshape(*shape[:2], *v.shape[1:])

            if is_numpy and v.dtype.kind not in "USO":
                v = torch.from_numpy(v)
            out[k] = v
        return out

    @torch.no_grad()
    def get_action(self, info_dict, **kwargs):
        t0 = time.perf_counter()
        info_dict = self._prepare_info(info_dict)

        pixels = info_dict["pixels"][:, -1].to(self.device)
        goal = info_dict["goal"][:, -1].to(self.device)

        enc_out = self.jepa.encoder(pixels, interpolate_pos_encoding=True)
        z_current = self.jepa.projector(enc_out.last_hidden_state[:, 0])

        if (self.cache_goal_encoding
                and self._cached_goal_input is not None
                and self._cached_goal_input.shape == goal.shape
                and torch.equal(self._cached_goal_input, goal)):
            z_goal = self._cached_z_goal
        else:
            enc_out_g = self.jepa.encoder(goal, interpolate_pos_encoding=True)
            z_goal = self.jepa.projector(enc_out_g.last_hidden_state[:, 0])
            if self.cache_goal_encoding:
                self._cached_goal_input = goal
                self._cached_z_goal = z_goal

        remaining = min(self.eval_budget - self._step_count, self.idm.max_horizon)
        steps_remaining = torch.full(
            (z_current.shape[0],), remaining, device=self.device, dtype=torch.long,
        )

        action = self.idm(z_current, z_goal, steps_remaining)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self._plan_times.append(time.perf_counter() - t0)
        self._step_count += 1

        return action.cpu().reshape(*self.env.action_space.shape).numpy()

    @property
    def avg_plan_time_ms(self):
        if not self._plan_times:
            return 0.0
        return 1000 * sum(self._plan_times) / len(self._plan_times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate GC-IDM vs CEM")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("--idm", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--cem-only", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--no-goal-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-split", type=float, default=1.0,
                        help="If < 1.0, eval only on held-out episodes (match training split)")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dataset-h5",
        default=None,
        help="Explicit H5 path. Defaults to this workspace's /data/ykz task mapping.",
    )
    parser.add_argument("--dataset-name-override", default=None)
    parser.add_argument("--trajectory-quality", action="store_true")
    parser.add_argument("--trajectory-quality-save-video", action="store_true")
    args = parser.parse_args()

    dcfg = DATASET_CONFIGS[args.dataset]
    if args.dataset_name_override is not None:
        import dataclasses
        dcfg = dataclasses.replace(dcfg, dataset_name=args.dataset_name_override)
    device = torch.device(args.device)
    data_dir = os.environ.get("STABLEWM_HOME", "./stable-wm-data")
    ckpt_dir = os.path.join(data_dir, "checkpoints", args.dataset, "lewm")
    ckpt_path = args.checkpoint or ckpt_dir

    # Load dataset + transforms
    if args.dataset_name_override is not None and args.dataset_h5 is None:
        dataset = swm.data.HDF5Dataset(
            dcfg.dataset_name,
            keys_to_cache=dcfg.cache_keys,
        )
        print(f"[dataset] task={args.dataset} stable_worldmodel_name={dcfg.dataset_name}")
    else:
        dataset = load_eval_dataset(args.dataset, dcfg, args.dataset_h5)
    img_tf = get_img_transform(dcfg.img_size)
    transform = {"pixels": img_tf, "goal": img_tf}
    process = build_normalizers(dataset, dcfg.cache_keys)

    eval_episodes, eval_start_idx = sample_eval_episodes(
        dataset, args.num_eval, args.goal_offset, args.seed,
        train_split=args.train_split, split_seed=args.split_seed,
    )

    results = {}

    # --- CEM ---
    if args.cem_only or args.compare:
        print("\n" + "=" * 60)
        print("  CEM Planning Evaluation")
        print("=" * 60)

        ckpt_obj = os.path.join(ckpt_path, "lewm_object.ckpt")
        if not os.path.exists(ckpt_obj) and os.path.exists(os.path.join(ckpt_path, "config.json")):
            _model = load_lewm_model(ckpt_path, "cpu")
            torch.save(_model, ckpt_obj)
            del _model

        cost_model = swm.policy.AutoCostModel(resolve_checkpoint_path_for_autocost(ckpt_path))
        cost_model = cost_model.to(device).eval()
        cost_model.requires_grad_(False)
        cost_model.interpolate_pos_encoding = True

        config = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
        solver = swm.solver.CEMSolver(model=cost_model, device=device, seed=args.seed)
        cem_policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform,
        )

        world = swm.World(
            env_name=dcfg.env, num_envs=args.num_eval,
            image_shape=(dcfg.img_size, dcfg.img_size),
            max_episode_steps=2 * max(args.eval_budget, args.goal_offset) + 5,
            **(dcfg.world_kwargs or {}),
        )
        world.set_policy(cem_policy)

        # Warmup
        _dummy = torch.randn(min(args.num_eval, 8), 3, dcfg.img_size, dcfg.img_size, device=device)
        with torch.no_grad():
            for _ in range(3):
                cost_model.encoder(_dummy, interpolate_pos_encoding=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        del _dummy

        _ep_arr = np.array(eval_episodes.tolist())
        _start_arr = np.array(eval_start_idx.tolist())
        _ = dataset.load_chunk(_ep_arr, _start_arr, _start_arr + args.goal_offset + 1)
        del _

        cem_plan_times: list[float] = []
        _orig = cem_policy.get_action
        def _timed(info_dict, **kw):
            if device.type == "cuda": torch.cuda.synchronize()
            _t = time.perf_counter()
            _r = _orig(info_dict, **kw)
            if device.type == "cuda": torch.cuda.synchronize()
            cem_plan_times.append(time.perf_counter() - _t)
            return _r
        cem_policy.get_action = _timed

        t0 = time.time()
        cem_metrics = evaluate_from_dataset_compat(
            world,
            dataset=dataset, start_steps=eval_start_idx.tolist(),
            goal_offset=args.goal_offset, eval_budget=args.eval_budget,
            episodes_idx=eval_episodes.tolist(), callables=dcfg.callables,
        )
        cem_time = time.time() - t0
        cem_plan_ms = 1000 * sum(cem_plan_times) / len(cem_plan_times) if cem_plan_times else 0.0

        results["cem"] = cem_metrics
        results["cem_ms"] = 1000 * cem_time / args.num_eval
        results["cem_plan_ms"] = cem_plan_ms
        print(f"CEM metrics: {cem_metrics}")
        print(f"CEM wall-clock: {cem_time:.1f}s total, {results['cem_ms']:.0f} ms/episode")
        print(f"CEM pure-compute: {cem_plan_ms:.1f} ms/plan-call ({len(cem_plan_times)} calls)")

    # --- GC-IDM ---
    if not args.cem_only:
        assert args.idm, "--idm required"
        print("\n" + "=" * 60)
        print("  GC-IDM Evaluation")
        print("=" * 60)

        model = load_lewm_model(ckpt_path, str(device))
        jepa = model.model if hasattr(model, "model") else model
        jepa.eval()
        jepa.requires_grad_(False)

        idm = load_idm(args.idm, str(device))

        # Speed benchmark
        z_test = torch.randn(1, dcfg.embed_dim, device=device)
        for _ in range(10):
            idm(z_test, z_test, torch.tensor([25], device=device))
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(500):
            idm(z_test, z_test, torch.tensor([25], device=device))
        if device.type == "cuda": torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - t0) / 500
        print(f"GC-IDM speed: {ms:.2f} ms/forward ({1000/ms:.0f} Hz)")

        idm_policy = GoalConditionedPolicy(
            jepa_model=jepa, idm=idm, eval_budget=args.eval_budget,
            process=process, transform=transform, device=device,
            cache_goal_encoding=not args.no_goal_cache,
        )

        world = swm.World(
            env_name=dcfg.env, num_envs=args.num_eval,
            image_shape=(dcfg.img_size, dcfg.img_size),
            max_episode_steps=2 * max(args.eval_budget, args.goal_offset) + 5,
            **(dcfg.world_kwargs or {}),
        )
        world.set_policy(idm_policy)

        # Warmup
        _dummy = torch.randn(min(args.num_eval, 8), 3, dcfg.img_size, dcfg.img_size, device=device)
        with torch.no_grad():
            for _ in range(3):
                jepa.encoder(_dummy, interpolate_pos_encoding=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        del _dummy

        _ep_arr = np.array(eval_episodes.tolist())
        _start_arr = np.array(eval_start_idx.tolist())
        _ = dataset.load_chunk(_ep_arr, _start_arr, _start_arr + args.goal_offset + 1)
        del _

        t0 = time.time()
        idm_metrics, idm_quality = evaluate_lgbs_policy(
            world=world,
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=args.goal_offset,
            eval_budget=args.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=dcfg.callables,
            task=args.dataset,
            trajectory_quality=bool(args.trajectory_quality),
            save_video=bool(args.trajectory_quality_save_video),
            latent_encoder=make_lgbs_latent_encoder(jepa, device) if bool(args.trajectory_quality) else None,
        )
        idm_time = time.time() - t0

        results["idm"] = idm_metrics
        results["idm_ms"] = 1000 * idm_time / args.num_eval
        results["idm_plan_ms"] = idm_policy.avg_plan_time_ms
        print(f"IDM metrics: {idm_metrics}")
        print(f"IDM time: {idm_time:.1f}s total, {results['idm_ms']:.0f} ms/episode")
        print(f"IDM avg plan time: {idm_policy.avg_plan_time_ms:.2f} ms")
        print_unified_eval_summary(
            task=args.dataset,
            planner_type="lgbs",
            solver="gcidm",
            checkpoint=str(ckpt_path),
            idm_path=str(args.idm),
            metrics=idm_metrics,
            evaluation_time_sec=idm_time,
            planning_time_total_sec=float(sum(idm_policy._plan_times)),
            global_planning_calls=len(idm_policy._plan_times),
            effective_replans_per_episode=int(args.eval_budget),
            quality=idm_quality,
        )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"{'Method':<20} {'Success Rate':>15} {'ms/episode':>12} {'ms/plan':>10}")
    print("-" * 60)

    if "cem" in results:
        sr = results["cem"].get("success_rate", "?")
        print(f"{'CEM':<20} {str(sr)+'%':>15} {results['cem_ms']:>12.0f} {results['cem_plan_ms']:>10.0f}")

    if "idm" in results:
        sr = results["idm"].get("success_rate", "?")
        print(f"{'GC-IDM':<20} {str(sr)+'%':>15} {results['idm_ms']:>12.0f} {results['idm_plan_ms']:>10.1f}")

    if "cem" in results and "idm" in results:
        speedup = results["cem_plan_ms"] / max(results["idm_plan_ms"], 0.1)
        delta = results["idm"].get("success_rate", 0) - results["cem"].get("success_rate", 0)
        print(f"\nSpeedup: {speedup:.0f}x")
        print(f"Success rate delta: {delta:+.1f}%")


if __name__ == "__main__":
    main()
