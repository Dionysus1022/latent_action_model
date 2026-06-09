import os

os.environ["MUJOCO_GL"] = "egl"

import math
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

from diffusion.policy import DiffusionPlannerPolicy
from planners.multi_candidate_policy import MultiCandidatePolicy
from planners.single_peak_policy import SinglePeakPolicy
from trajectory_quality import (
    compute_latent_monotonicity,
    compute_task_goal_distances,
    compute_trajectory_quality,
)


TASK_ALIASES = {
    "pusht": "pusht",
    "tworoom": "tworoom",
    "two-room": "tworoom",
    "two_room": "tworoom",
    "reacher": "reacher",
    "researcher": "reacher",
}

LEGACY_EVAL_CONFIG_ALIASES = {
    **{f"{task}_mpc": (task, "mpc") for task in ["cube", "pusht", "reacher", "tworoom"]},
    **{f"{task}_diffusion": (task, "diffusion") for task in ["cube", "pusht", "reacher", "tworoom"]},
    **{f"{task}_consistency": (task, "consistency") for task in ["cube", "pusht", "reacher", "tworoom"]},
    "pusht_diffusion_corrective": ("pusht", "corrective_learned"),
}

PROFILE_SECTION_DEFAULTS = {
    "diffusion_refinement": {
        "enabled": None,
        "steps": None,
        "step_size": None,
        "topk": None,
        "goal_weight": None,
        "prior_weight": None,
        "smoothness_weight": None,
        "grad_clip_norm": None,
    },
    "corrective": {
        "enabled": None,
        "mode": None,
        "corrector_path": None,
        "correction_interval": None,
        "execute_horizon": None,
        "error_threshold": None,
        "trigger_stat": None,
        "trigger_quantile": None,
        "trigger_scope": None,
        "error_metric": None,
        "logging": {
            "log_prediction_error": None,
            "log_correction_norm": None,
            "log_replan_count": None,
        },
    },
}

PROFILE_SCALAR_DEFAULTS = {
    "diffusion_selection_mode": None,
    "diffusion_num_candidates": None,
    "diffusion_truncation_steps": None,
    "diffusion_start_timestep": None,
    "diffusion_eta": None,
    "diffusion_noise_scale": None,
    "diffusion_sampling_temperature": None,
    "diffusion_runtime_execute_steps": None,
}

TRAJECTORY_QUALITY_DEFAULTS = {
    "enabled": False,
    "save_npz": True,
    "truncate_after_success": True,
    "save_video": True,
}


_NO_PROFILE_OVERRIDE = object()


class PlanningStatsSolver:
    """Wrap a planning solver and record actual solver-call planning time."""

    def __init__(self, solver: Any) -> None:
        self.solver = solver
        self._num_replans = 0
        self._planning_time_total_sec = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.solver, name)

    def configure(self, *args: Any, **kwargs: Any) -> Any:
        return self.solver.configure(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def solve(self, *args: Any, **kwargs: Any) -> dict:
        self._sync_cuda_if_available()
        start_time = time.perf_counter()
        try:
            return self.solver.solve(*args, **kwargs)
        finally:
            self._sync_cuda_if_available()
            self._planning_time_total_sec += time.perf_counter() - start_time
            self._num_replans += 1

    @staticmethod
    def _sync_cuda_if_available() -> None:
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass


def read_planning_stat(policy: Any, stat_name: str) -> Any:
    value = getattr(policy, stat_name, None)
    if value is not None:
        return value
    solver = getattr(policy, "solver", None)
    if solver is None:
        return None
    return getattr(solver, stat_name, None)


def default_planner_config_name(task_name: str) -> str:
    return task_name


def normalize_task_name(task_name: str | None) -> str | None:
    if task_name in [None, "", "null"]:
        return None
    normalized = str(task_name).strip().lower()
    return TASK_ALIASES.get(normalized, normalized)


def normalize_config_name(config_name: str | None) -> tuple[str | None, str | None]:
    normalized_name = normalize_task_name(config_name)
    if normalized_name is None:
        return None, None
    legacy = LEGACY_EVAL_CONFIG_ALIASES.get(str(normalized_name))
    if legacy is None:
        return normalized_name, None
    task_name, profile_name = legacy
    return task_name, profile_name


def has_eval_profile_override(argv: list[str]) -> bool:
    return any(
        arg.startswith("eval_profile=")
        or arg.startswith("+eval_profile=")
        or arg.startswith("++eval_profile=")
        for arg in argv
    )


def normalize_eval_cli_args(argv: list[str]) -> list[str]:
    normalized: list[str] = [argv[0]]
    pending_config_name = False
    pending_task_flag = False
    requested_task: str | None = None
    requested_profile: str | None = None
    saw_config_name = False

    for arg in argv[1:]:
        if pending_config_name:
            config_name, profile_name = normalize_config_name(arg)
            normalized.append(config_name or arg)
            requested_profile = requested_profile or profile_name
            saw_config_name = True
            pending_config_name = False
            continue
        if pending_task_flag:
            requested_task = normalize_task_name(arg)
            pending_task_flag = False
            continue

        if arg == "--config-name":
            normalized.append(arg)
            pending_config_name = True
            continue
        if arg.startswith("--config-name="):
            config_name, profile_name = normalize_config_name(arg.split("=", 1)[1])
            config_name = config_name or arg.split("=", 1)[1]
            normalized.append(f"--config-name={config_name}")
            requested_profile = requested_profile or profile_name
            saw_config_name = True
            continue
        if arg == "--task":
            pending_task_flag = True
            continue
        if arg.startswith("--task="):
            requested_task = normalize_task_name(arg.split("=", 1)[1])
            continue
        if arg.startswith("task="):
            requested_task = normalize_task_name(arg.split("=", 1)[1])
            continue

        normalized.append(arg)

    if pending_config_name:
        raise ValueError("--config-name requires a value.")
    if pending_task_flag:
        raise ValueError("--task requires a value.")
    if requested_task is not None and not saw_config_name:
        normalized[1:1] = ["--config-name", default_planner_config_name(requested_task)]
        if not has_eval_profile_override(normalized):
            normalized.append("eval_profile=mpc")
    elif requested_profile is not None and not has_eval_profile_override(normalized):
        normalized.append(f"eval_profile={requested_profile}")
    return normalized


def _profile_override_diff(value, default):
    if isinstance(value, dict) and isinstance(default, dict):
        diff = {}
        for key, item in value.items():
            if key not in default:
                diff[key] = item
                continue
            item_diff = _profile_override_diff(item, default[key])
            if item_diff is not _NO_PROFILE_OVERRIDE:
                diff[key] = item_diff
        if diff:
            return diff
        return _NO_PROFILE_OVERRIDE
    if value == default:
        return _NO_PROFILE_OVERRIDE
    return value


def value_or_default(value, default):
    if value in [None, "", "null"]:
        return default
    return value


def resolve_eval_profile_config(cfg: DictConfig) -> DictConfig:
    """Merge cfg.profiles[cfg.eval_profile] into the top-level eval config.

    Task configs can keep all planner variants in one file:
        eval_profile: diffusion
        profiles:
          diffusion:
            planner_type: diffusion
            ...

    The returned config is what the rest of eval.py consumes.
    """
    profile_name = cfg.get("eval_profile", None)
    if profile_name in [None, "", "null"]:
        return cfg

    profiles = cfg.get("profiles", None)
    if profiles is None:
        raise KeyError(
            f"eval_profile='{profile_name}' was set, but cfg.profiles is missing."
        )
    profile_name = str(profile_name)
    if profile_name not in profiles:
        available = list(profiles.keys()) if hasattr(profiles, "keys") else []
        raise KeyError(
            f"Unknown eval_profile '{profile_name}'. Available profiles: {available}."
        )

    selected = OmegaConf.create(OmegaConf.to_container(profiles[profile_name], resolve=False))
    base = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    section_overrides = {}
    for section_name, default_value in PROFILE_SECTION_DEFAULTS.items():
        if section_name not in base:
            continue
        section_value = OmegaConf.to_container(base[section_name], resolve=False)
        section_diff = _profile_override_diff(section_value, default_value)
        del base[section_name]
        if section_diff is not _NO_PROFILE_OVERRIDE:
            section_overrides[section_name] = section_diff

    scalar_overrides = {}
    for scalar_name, default_value in PROFILE_SCALAR_DEFAULTS.items():
        if scalar_name not in base:
            continue
        scalar_value = base[scalar_name]
        del base[scalar_name]
        if scalar_value != default_value:
            scalar_overrides[scalar_name] = scalar_value

    resolved = OmegaConf.merge(base, selected, scalar_overrides, section_overrides)
    resolved.eval_profile = profile_name
    if "profiles" in resolved:
        del resolved["profiles"]
    if "solver_defaults" in resolved:
        del resolved["solver_defaults"]
    return resolved


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def infer_dataset_column(
    *,
    available_columns: list[str],
    field_name: str,
    exact_candidates: list[str],
    prefix_candidates: list[str] | None = None,
) -> str:
    for candidate in exact_candidates:
        if candidate in available_columns:
            return candidate
    if prefix_candidates:
        prefix_matches = [
            column
            for column in available_columns
            if any(column.startswith(prefix) for prefix in prefix_candidates)
        ]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        if len(prefix_matches) > 1:
            raise KeyError(
                f"Could not infer '{field_name}': multiple prefix matches {prefix_matches} "
                f"for prefixes {prefix_candidates}."
            )
    raise KeyError(
        f"Could not infer '{field_name}'. Tried exact candidates {exact_candidates} "
        f"and prefixes {prefix_candidates or []}. Available columns: {available_columns}."
    )


def resolve_dataset_cache_dir(cfg: DictConfig, dataset_name: str) -> Path:
    dataset_root = Path(cfg.get("cache_dir") or swm.data.utils.get_cache_dir()).expanduser()
    if dataset_root.suffix == ".h5":
        expected_rel_path = Path(f"{dataset_name}.h5")
        if dataset_root.name != expected_rel_path.name:
            raise FileNotFoundError(
                f"Dataset file {dataset_root} does not match dataset_name={dataset_name}."
            )
        parent_depth = max(0, len(expected_rel_path.parts) - 1)
        return dataset_root.parents[parent_depth]

    nested_root = dataset_root / dataset_name
    if (dataset_root / f"{dataset_name}.h5").exists():
        return dataset_root
    if (nested_root / f"{dataset_name}.h5").exists():
        return nested_root
    return dataset_root


def resolve_explicit_dataset_source(
    dataset_h5: str,
    dataset_name: str | None = None,
) -> tuple[str, Path]:
    h5_path = Path(dataset_h5).expanduser()
    if h5_path.suffix != ".h5":
        raise ValueError(f"dataset_h5 must point to a .h5 file, got {h5_path}.")
    if dataset_name not in [None, "", "null"]:
        requested_name = str(dataset_name).strip()
        expected_rel_path = Path(f"{requested_name}.h5")
        parent_depth = max(0, len(expected_rel_path.parts) - 1)
        if parent_depth < len(h5_path.parents):
            candidate_cache_dir = h5_path.parents[parent_depth]
            if candidate_cache_dir / expected_rel_path == h5_path:
                return requested_name, candidate_cache_dir
    return h5_path.stem, h5_path.parent


def resolve_eval_callables(cfg: DictConfig) -> list[dict]:
    callables = OmegaConf.to_container(cfg.eval.get("callables"), resolve=True)
    if callables is None:
        return []
    if not isinstance(callables, list):
        raise TypeError(
            "cfg.eval.callables must be a list of callable specs, "
            f"but got {type(callables).__name__}."
        )
    return callables


def resolve_trajectory_quality_config(cfg: DictConfig) -> dict:
    quality_cfg = cfg.get("trajectory_quality", None)
    if quality_cfg is None:
        return dict(TRAJECTORY_QUALITY_DEFAULTS)
    container = OmegaConf.to_container(quality_cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(
            "cfg.trajectory_quality must resolve to a dictionary, "
            f"got {type(container).__name__}."
        )
    resolved = dict(TRAJECTORY_QUALITY_DEFAULTS)
    for key in resolved:
        if key in container:
            resolved[key] = container[key]
    return {
        "enabled": bool(resolved["enabled"]),
        "save_npz": bool(resolved["save_npz"]),
        "truncate_after_success": bool(resolved["truncate_after_success"]),
        "save_video": bool(resolved["save_video"]),
    }


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _last_history_frame(value):
    array = _to_numpy(value)
    if array.ndim >= 3 and int(array.shape[1]) == 1:
        return array[:, -1, ...]
    return array


def _record_info_keys(info: dict, keys: list[str]) -> dict[str, np.ndarray]:
    record = {}
    for key in keys:
        if key in info:
            record[key] = np.asarray(_last_history_frame(info[key])).copy()
    return record


def _stack_time_records(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not records:
        return {}
    keys = sorted({key for record in records for key in record})
    stacked = {}
    for key in keys:
        values = [record[key] for record in records if key in record]
        if len(values) != len(records):
            continue
        stacked[key] = np.stack(values, axis=1)
    return stacked


def _safe_dataset_rows(dataset, indices: np.ndarray) -> dict:
    rows = dataset.get_row_data(indices)
    return {
        key: _to_numpy(value)
        for key, value in rows.items()
        if isinstance(value, (np.ndarray, torch.Tensor))
    }


def _build_goal_trace_aliases(
    *,
    requested_task: str | None,
    dataset_goal_rows: dict[str, np.ndarray],
    runtime_trace: dict[str, np.ndarray],
    eval_budget: int,
) -> dict[str, np.ndarray]:
    task_name = normalize_task_name(requested_task)
    aliases: dict[str, np.ndarray] = {}

    def add_constant(alias_key: str, source_key: str):
        if alias_key in runtime_trace:
            return
        if source_key not in dataset_goal_rows:
            return
        value = np.asarray(dataset_goal_rows[source_key])
        repeated = np.broadcast_to(
            value[:, None, ...],
            (value.shape[0], int(eval_budget), *value.shape[1:]),
        ).copy()
        aliases[alias_key] = repeated

    if task_name == "reacher":
        add_constant("goal_qpos", "qpos")
    elif task_name == "tworoom":
        add_constant("goal_proprio", "proprio")
    elif task_name == "pusht":
        add_constant("goal_state", "state")
    elif task_name == "cube":
        add_constant("goal_privileged_block_0_pos", "privileged_block_0_pos")
        add_constant("goal_privileged/block_0_pos", "privileged_block_0_pos")
        add_constant("goal_privileged_block_0_yaw", "privileged_block_0_yaw")
        add_constant("goal_privileged/block_0_yaw", "privileged_block_0_yaw")

    return aliases


def _trajectory_state_keys_for_task(task: str | None) -> list[str]:
    task_name = normalize_task_name(task)
    common = ["action", "pixels", "goal"]
    if task_name == "reacher":
        return [*common, "qpos", "qvel"]
    if task_name == "tworoom":
        return [*common, "proprio", "state", "distance_to_target"]
    if task_name == "pusht":
        return [*common, "proprio", "state"]
    if task_name == "cube":
        return [
            *common,
            "qpos",
            "qvel",
            "privileged/block_0_pos",
            "privileged/block_0_quat",
            "privileged/block_0_yaw",
            "privileged_block_0_pos",
            "privileged_block_0_quat",
            "privileged_block_0_yaw",
        ]
    return common


def run_evaluation_with_trajectory_quality(
    *,
    world,
    dataset,
    episodes_idx: list[int],
    start_steps: list[int],
    goal_offset_steps: int,
    eval_budget: int,
    callables: list[dict],
    video_path: Path,
    task: str | None,
    quality_cfg: dict,
    latent_encoder: Any | None = None,
) -> tuple[dict, dict]:
    ep_idx_arr = np.asarray(episodes_idx)
    start_steps_arr = np.asarray(start_steps)
    end_steps = start_steps_arr + int(goal_offset_steps)
    if len(ep_idx_arr) != len(start_steps_arr):
        raise ValueError("episodes_idx and start_steps must have the same length.")
    if len(ep_idx_arr) != int(world.num_envs):
        raise ValueError("Number of episodes to evaluate must match world.num_envs.")

    data = dataset.load_chunk(ep_idx_arr, start_steps_arr, end_steps)
    columns = dataset.column_names

    init_step_per_env: dict[str, list] = {}
    goal_step_per_env: dict[str, list] = {}
    for ep in data:
        for col in columns:
            if col.startswith("goal"):
                continue
            if col not in ep or not isinstance(ep[col], (torch.Tensor, np.ndarray)):
                continue
            value = ep[col]
            if col.startswith("pixels"):
                value = value.permute(0, 2, 3, 1) if torch.is_tensor(value) else np.moveaxis(value, 1, -1)
            init_data = _to_numpy(value[0])
            goal_data = _to_numpy(value[-1])
            init_step_per_env.setdefault(col, []).append(init_data)
            goal_step_per_env.setdefault(col, []).append(goal_data)

    init_step = {key: np.stack(value) for key, value in init_step_per_env.items()}
    goal_step = {}
    for key, value in goal_step_per_env.items():
        goal_key = "goal" if key == "pixels" else f"goal_{key}"
        goal_step[goal_key] = np.stack(value)

    seeds = init_step.get("seed")
    variations_dict = {
        key.removeprefix("variation."): value
        for key, value in init_step.items()
        if key.startswith("variation.")
    }
    options = [{} for _ in range(world.num_envs)]
    if variations_dict:
        for env_index in range(world.num_envs):
            options[env_index]["variation"] = list(variations_dict.keys())
            options[env_index]["variation_values"] = {
                key: value[env_index] for key, value in variations_dict.items()
            }

    init_step.update(dict(goal_step))
    world.reset(seed=seeds, options=options)

    for env_index, env in enumerate(world.envs.unwrapped.envs):
        env_unwrapped = env.unwrapped
        for spec in callables:
            method_name = spec["method"]
            if not hasattr(env_unwrapped, method_name):
                continue
            method = getattr(env_unwrapped, method_name)
            args = spec.get("args", spec)
            prepared_args = {}
            for args_name, args_data in args.items():
                value_key = args_data.get("value", None)
                is_in_dataset = args_data.get("in_dataset", True)
                if is_in_dataset:
                    if value_key not in init_step:
                        continue
                    prepared_args[args_name] = np.copy(init_step[value_key][env_index])
                else:
                    prepared_args[args_name] = args_data.get("value")
            method(**prepared_args)

    shape_prefix = world.infos["pixels"].shape[:2]
    init_step = {
        key: np.broadcast_to(value[:, None, ...], shape_prefix + value.shape[1:])
        for key, value in init_step.items()
    }
    goal_step = {
        key: np.broadcast_to(value[:, None, ...], shape_prefix + value.shape[1:])
        for key, value in goal_step.items()
    }
    world.infos.update({key: np.copy(value) for key, value in init_step.items()})
    world.infos.update({key: np.copy(value) for key, value in goal_step.items()})

    results = {
        "success_rate": 0.0,
        "episode_successes": np.zeros(len(episodes_idx), dtype=bool),
        "seeds": seeds,
    }

    target_frames = torch.stack([ep["pixels"] for ep in data]).numpy()
    video_frames = np.empty(
        (world.num_envs, int(eval_budget), *world.infos["pixels"].shape[-3:]),
        dtype=np.uint8,
    )
    info_keys = _trajectory_state_keys_for_task(task)
    trace_records: list[dict[str, np.ndarray]] = []
    success_records = []

    for step_index in range(int(eval_budget)):
        video_frames[:, step_index] = world.infos["pixels"][:, -1]
        world.infos.update({key: np.copy(value) for key, value in goal_step.items()})
        world.step()
        results["episode_successes"] = np.logical_or(
            results["episode_successes"], world.terminateds
        )
        trace_records.append(_record_info_keys(world.infos, info_keys))
        success_records.append(np.asarray(results["episode_successes"], dtype=bool).copy())
        world.envs.unwrapped._autoreset_envs = np.zeros((world.num_envs,))

    video_frames[:, -1] = world.infos["pixels"][:, -1]
    results["success_rate"] = float(np.sum(results["episode_successes"])) / len(episodes_idx) * 100.0

    if bool(quality_cfg["save_video"]):
        import imageio

        target_len = target_frames.shape[1]
        video_path.mkdir(parents=True, exist_ok=True)
        for env_index in range(world.num_envs):
            out = imageio.get_writer(
                video_path / f"rollout_{env_index}.mp4",
                fps=15,
                codec="libx264",
            )
            goals = np.vstack([target_frames[env_index, -1], target_frames[env_index, -1]])
            for t in range(int(eval_budget)):
                stacked_frame = np.vstack(
                    [video_frames[env_index, t], target_frames[env_index, t % target_len]]
                )
                out.append_data(np.hstack([stacked_frame, goals]))
            out.close()
        print(f"Video saved to {video_path}")

    runtime_trace = _stack_time_records(trace_records)
    dataset_goal_rows = {
        key.removeprefix("goal_"): value
        for key, value in goal_step.items()
        if key.startswith("goal_")
    }
    runtime_trace.update(
        _build_goal_trace_aliases(
            requested_task=task,
            dataset_goal_rows=dataset_goal_rows,
            runtime_trace=runtime_trace,
            eval_budget=int(eval_budget),
        )
    )

    distance_outputs = compute_task_goal_distances(task or "", runtime_trace)
    actions = runtime_trace.get("action")
    if actions is None:
        raise KeyError("Trajectory trace did not contain runtime action records.")
    state_for_path = distance_outputs.pop("state_for_path")
    successes_by_step = np.stack(success_records, axis=1)
    summary, per_episode = compute_trajectory_quality(
        states=state_for_path,
        actions=actions,
        goal_distances=distance_outputs["goal_distance"],
        successes_by_step=successes_by_step,
        truncate_after_success=bool(quality_cfg["truncate_after_success"]),
    )
    latent_summary = None
    latent_per_episode = None
    if latent_encoder is not None and "pixels" in runtime_trace and "goal" in runtime_trace:
        latent_array, goal_latent_array = latent_encoder(runtime_trace["pixels"], runtime_trace["goal"])
        latent_summary, latent_per_episode = compute_latent_monotonicity(
            latents=latent_array,
            goal_latents=goal_latent_array,
            successes_by_step=successes_by_step,
            truncate_after_success=bool(quality_cfg["truncate_after_success"]),
        )
        summary.update(latent_summary)
    quality = {
        "summary": summary,
        "per_episode": per_episode,
        "distances": distance_outputs,
        "trace": runtime_trace,
        "successes_by_step": successes_by_step,
    }
    if latent_summary is not None:
        quality["latent_summary"] = latent_summary
        quality["latent_per_episode"] = latent_per_episode

    if seeds is not None:
        assert np.unique(seeds).shape[0] == len(episodes_idx), "Some episode seeds are identical!"
    return results, quality


def resolve_corrective_config(cfg: DictConfig) -> dict:
    corrective_cfg = cfg.get("corrective", None)
    if corrective_cfg is None:
        return {
            "enabled": False,
            "mode": "none",
            "corrector_path": None,
            "correction_interval": 2,
            "execute_horizon": None,
            "error_threshold": 0.5,
            "trigger_stat": "max",
            "trigger_quantile": 0.9,
            "trigger_scope": "per_env",
            "error_metric": "l2",
            "logging": {
                "log_prediction_error": False,
            },
        }
    container = OmegaConf.to_container(corrective_cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(
            "cfg.corrective must resolve to a dictionary, "
            f"got {type(container).__name__}."
        )
    logging_cfg = container.get("logging") or {}
    if not isinstance(logging_cfg, dict):
        raise TypeError("cfg.corrective.logging must resolve to a dictionary.")

    enabled = bool(value_or_default(container.get("enabled", None), False))
    mode = str(value_or_default(container.get("mode", None), "none")).lower().strip()
    if mode not in {"none", "replan", "learned"}:
        raise ValueError(
            f"Unsupported corrective.mode '{mode}'. Expected none, replan, or learned."
        )
    trigger_stat = str(value_or_default(container.get("trigger_stat", None), "max")).lower().strip()
    if trigger_stat not in {"max", "mean", "quantile"}:
        raise ValueError(
            "Unsupported corrective.trigger_stat "
            f"'{trigger_stat}'. Expected max, mean, or quantile."
        )
    trigger_quantile = float(value_or_default(container.get("trigger_quantile", None), 0.9))
    if trigger_quantile < 0.0 or trigger_quantile > 1.0:
        raise ValueError(
            "corrective.trigger_quantile must be in [0, 1], "
            f"got {trigger_quantile}."
        )
    trigger_scope = str(value_or_default(container.get("trigger_scope", None), "per_env")).lower().strip()
    if trigger_scope not in {"per_env", "batch"}:
        raise ValueError(
            "Unsupported corrective.trigger_scope "
            f"'{trigger_scope}'. Expected per_env or batch."
        )
    corrector_path = value_or_default(container.get("corrector_path", None), None)
    if enabled and mode == "learned":
        if corrector_path in [None, "", "null"]:
            raise ValueError(
                "corrective.corrector_path must be set when corrective.enabled=true "
                "and corrective.mode=learned."
            )

    return {
        "enabled": enabled,
        "mode": mode,
        "corrector_path": corrector_path,
        "correction_interval": int(value_or_default(container.get("correction_interval", None), 2)),
        "execute_horizon": value_or_default(container.get("execute_horizon", None), None),
        "error_threshold": float(value_or_default(container.get("error_threshold", None), 0.5)),
        "trigger_stat": trigger_stat,
        "trigger_quantile": trigger_quantile,
        "trigger_scope": trigger_scope,
        "error_metric": str(value_or_default(container.get("error_metric", None), "l2")).lower().strip(),
        "logging": {
            "log_prediction_error": bool(
                value_or_default(logging_cfg.get("log_prediction_error", None), False)
            ),
        },
    }


def resolve_diffusion_refinement_config(cfg: DictConfig) -> dict:
    refinement_cfg = cfg.get("diffusion_refinement", None)
    if refinement_cfg is None:
        return {
            "enabled": False,
            "steps": 1,
            "step_size": 0.03,
            "topk": None,
            "goal_weight": 1.0,
            "prior_weight": 0.05,
            "smoothness_weight": 0.005,
            "grad_clip_norm": None,
        }
    container = OmegaConf.to_container(refinement_cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(
            "cfg.diffusion_refinement must resolve to a dictionary, "
            f"got {type(container).__name__}."
        )

    topk = container.get("topk", None)
    if topk in [None, "", "null"]:
        topk = None
    else:
        topk = int(topk)

    grad_clip_norm = container.get("grad_clip_norm", None)
    if grad_clip_norm in [None, "", "null"]:
        grad_clip_norm = None
    else:
        grad_clip_norm = float(grad_clip_norm)

    return {
        "enabled": bool(value_or_default(container.get("enabled", None), False)),
        "steps": int(value_or_default(container.get("steps", None), 1)),
        "step_size": float(value_or_default(container.get("step_size", None), 0.03)),
        "topk": topk,
        "goal_weight": float(value_or_default(container.get("goal_weight", None), 1.0)),
        "prior_weight": float(value_or_default(container.get("prior_weight", None), 0.05)),
        "smoothness_weight": float(
            value_or_default(container.get("smoothness_weight", None), 0.005)
        ),
        "grad_clip_norm": grad_clip_norm,
    }


def resolve_diffusion_runtime_execute_steps(
    diffusion_runtime_execute_steps,
    corrective_cfg: dict,
) -> int | None:
    if diffusion_runtime_execute_steps in [None, "", "null"]:
        resolved = None
    else:
        resolved = int(diffusion_runtime_execute_steps)

    if not bool(corrective_cfg.get("enabled", False)):
        return resolved

    corrective_execute_horizon = corrective_cfg.get("execute_horizon", None)
    if corrective_execute_horizon not in [None, "", "null"]:
        resolved = int(corrective_execute_horizon)
    return resolved


def summarize_solver_config(solver_cfg: DictConfig) -> tuple[str, dict]:
    solver_target = str(solver_cfg.get("_target_", "unknown"))
    solver_params = OmegaConf.to_container(solver_cfg, resolve=True)
    if not isinstance(solver_params, dict):
        raise TypeError(
            "cfg.solver must resolve to a dictionary, "
            f"but got {type(solver_params).__name__}."
        )
    solver_params = dict(solver_params)
    solver_params.pop("model", None)
    return solver_target, solver_params


def get_episodes_length(dataset, episodes, episode_key: str, step_key: str):
    episode_idx = dataset.get_col_data(episode_key)
    step_idx = dataset.get_col_data(step_key)
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def sample_eval_episode_starts(
    *,
    dataset,
    ep_indices: np.ndarray,
    goal_offset_steps: int,
    num_eval: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Sample episode ids and start steps without scanning full step columns."""
    lengths = np.asarray(dataset.lengths)
    offsets = np.asarray(dataset.offsets)
    episode_ids = np.asarray(ep_indices)
    if episode_ids.shape[0] != lengths.shape[0]:
        episode_ids = np.arange(lengths.shape[0], dtype=np.int64)

    max_start = lengths.astype(np.int64) - int(goal_offset_steps) - 1
    valid_episode_positions = np.nonzero(max_start >= 0)[0]
    valid_start_count = int(np.sum(max_start[valid_episode_positions] + 1))
    if valid_start_count < int(num_eval):
        raise ValueError(
            "Not enough valid starting points for evaluation: "
            f"{valid_start_count} < {num_eval}."
        )

    rng = np.random.default_rng(seed)
    flat_choices = np.sort(
        rng.choice(valid_start_count, size=int(num_eval), replace=False)
    )
    cumulative = np.cumsum(max_start[valid_episode_positions] + 1)
    chosen_positions = np.searchsorted(cumulative, flat_choices, side="right")
    previous_cumulative = np.concatenate([[0], cumulative[:-1]])
    start_steps = flat_choices - previous_cumulative[chosen_positions]
    episode_positions = valid_episode_positions[chosen_positions]
    eval_episodes = episode_ids[episode_positions]
    global_rows = offsets[episode_positions] + start_steps
    order = np.argsort(global_rows)
    return eval_episodes[order], start_steps[order], valid_start_count


def get_dataset(cfg, dataset_name):
    dataset_h5 = cfg.get("dataset_h5", None)
    if dataset_h5 not in [None, "", "null"]:
        dataset_name, dataset_path = resolve_explicit_dataset_source(
            str(dataset_h5),
            dataset_name=dataset_name,
        )
    else:
        dataset_path = resolve_dataset_cache_dir(cfg, dataset_name)
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    config_name = HydraConfig.get().job.config_name
    cfg = resolve_eval_profile_config(cfg)
    requested_task = normalize_task_name(cfg.get("task", config_name))
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    image_size = int(cfg.eval.img_size)
    world = swm.World(**cfg.world, image_shape=(image_size, image_size))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    available_columns = list(dataset.column_names)
    episode_key = infer_dataset_column(
        available_columns=available_columns,
        field_name="episode_key",
        exact_candidates=["episode_idx", "ep_idx"],
    )
    step_key = infer_dataset_column(
        available_columns=available_columns,
        field_name="step_key",
        exact_candidates=["step_idx"],
    )
    col_name = episode_key
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    planner_type = str(cfg.get("planner_type", "mpc")).lower()
    policy_name = cfg.get("policy", "random")
    diffusion_selection_mode = str(cfg.get("diffusion_selection_mode", "wm_only")).lower()
    diffusion_num_candidates = cfg.get("diffusion_num_candidates", None)
    diffusion_truncation_steps = cfg.get("diffusion_truncation_steps", None)
    diffusion_start_timestep = cfg.get("diffusion_start_timestep", None)
    diffusion_eta = float(cfg.get("diffusion_eta", 0.0))
    diffusion_noise_scale = float(cfg.get("diffusion_noise_scale", 1.0))
    diffusion_sampling_temperature = float(cfg.get("diffusion_sampling_temperature", 1.0))
    diffusion_runtime_execute_steps = cfg.get("diffusion_runtime_execute_steps", None)
    corrective_cfg = resolve_corrective_config(cfg)
    refinement_cfg = resolve_diffusion_refinement_config(cfg)
    trajectory_quality_cfg = resolve_trajectory_quality_config(cfg)
    eval_callables = resolve_eval_callables(cfg)
    if len(eval_callables) == 0:
        print(
            "[metric-warning] "
            "cfg.eval.callables is empty; evaluation will still run, "
            "but dataset-conditioned state/goal setup may be incomplete."
        )

    if diffusion_num_candidates in [None, "", "null"]:
        diffusion_num_candidates = None
    else:
        diffusion_num_candidates = int(diffusion_num_candidates)

    if diffusion_truncation_steps in [None, "", "null"]:
        diffusion_truncation_steps = None
    else:
        diffusion_truncation_steps = int(diffusion_truncation_steps)

    if diffusion_start_timestep in [None, "", "null"]:
        diffusion_start_timestep = None
    else:
        diffusion_start_timestep = int(diffusion_start_timestep)

    diffusion_runtime_execute_steps = resolve_diffusion_runtime_execute_steps(
        diffusion_runtime_execute_steps,
        corrective_cfg,
    )

    if planner_type == "mpc":
        if policy_name != "random":
            model = swm.policy.AutoCostModel(cfg.policy)
            model = model.to("cuda")
            model = model.eval()
            model.requires_grad_(False)
            model.interpolate_pos_encoding = True
            config = swm.PlanConfig(**cfg.plan_config)
            solver = PlanningStatsSolver(hydra.utils.instantiate(cfg.solver, model=model))
            solver_target, solver_params = summarize_solver_config(cfg.solver)
            policy = swm.policy.WorldModelPolicy(
                solver=solver, config=config, process=process, transform=transform
            )
            print(
                "[planner] "
                f"type=mpc task={requested_task or config_name} config={config_name} "
                f"policy={cfg.policy} goal_offset={int(cfg.eval.goal_offset_steps)} "
                f"eval_budget={int(cfg.eval.eval_budget)} "
                f"horizon={int(cfg.plan_config.horizon)} "
                f"receding_horizon={int(cfg.plan_config.receding_horizon)} "
                f"action_block={int(cfg.plan_config.action_block)} "
                f"solver_target={solver_target} solver={solver_params}"
            )
        else:
            policy = swm.policy.RandomPolicy()
            print("[planner] type=random")

    elif planner_type == "single_peak":
        if policy_name == "random":
            raise ValueError("planner_type=single_peak requires cfg.policy to point to a world-model checkpoint.")
        single_peak_bundle = cfg.get("single_peak_bundle", None)
        if single_peak_bundle in [None, "", "null"]:
            raise ValueError("planner_type=single_peak requires cfg.single_peak_bundle.")

        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        policy = SinglePeakPolicy.from_bundle(
            bundle_path=single_peak_bundle,
            world_model=model,
            config=config,
            process=process,
            transform=transform,
            map_location="cpu",
        )
        print(
            "[planner] "
            f"type=single_peak wm_policy={cfg.policy} bundle={single_peak_bundle} "
            f"plan_horizon={policy.plan_horizon} action_dim={policy.action_dim}"
        )

    elif planner_type == "multi_candidate":
        if policy_name == "random":
            raise ValueError("planner_type=multi_candidate requires cfg.policy to point to a world-model checkpoint.")
        multi_candidate_bundle = cfg.get("multi_candidate_bundle", None)
        if multi_candidate_bundle in [None, "", "null"]:
            raise ValueError("planner_type=multi_candidate requires cfg.multi_candidate_bundle.")

        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        policy = MultiCandidatePolicy.from_bundle(
            bundle_path=multi_candidate_bundle,
            world_model=model,
            config=config,
            process=process,
            transform=transform,
            map_location="cpu",
        )
        print(
            "[planner] "
            f"type=multi_candidate wm_policy={cfg.policy} bundle={multi_candidate_bundle} "
            f"plan_horizon={policy.plan_horizon} action_dim={policy.action_dim} "
            f"num_candidates={policy.num_candidates}"
        )

    elif planner_type == "diffusion":
        if policy_name == "random":
            raise ValueError("planner_type=diffusion requires cfg.policy to point to a world-model checkpoint.")
        diffusion_bundle = cfg.get("diffusion_bundle", None)
        if diffusion_bundle in [None, "", "null"]:
            raise ValueError("planner_type=diffusion requires cfg.diffusion_bundle.")

        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        policy = DiffusionPlannerPolicy.from_bundle(
            bundle_path=diffusion_bundle,
            world_model=model,
            config=config,
            process=process,
            transform=transform,
            map_location="cpu",
            diffusion_eta=diffusion_eta,
            num_candidates=diffusion_num_candidates,
            truncation_steps=diffusion_truncation_steps,
            start_timestep=diffusion_start_timestep,
            noise_scale=diffusion_noise_scale,
            sampling_temperature=diffusion_sampling_temperature,
            selection_mode=diffusion_selection_mode,
            goal_offset_steps=int(cfg.eval.goal_offset_steps),
            eval_budget=int(cfg.eval.eval_budget),
            runtime_execute_steps=diffusion_runtime_execute_steps,
            corrective_enabled=bool(corrective_cfg["enabled"]),
            corrective_mode=str(corrective_cfg["mode"]),
            corrective_correction_interval=int(corrective_cfg["correction_interval"]),
            corrective_error_threshold=float(corrective_cfg["error_threshold"]),
            corrective_trigger_stat=str(corrective_cfg["trigger_stat"]),
            corrective_trigger_quantile=float(corrective_cfg["trigger_quantile"]),
            corrective_trigger_scope=str(corrective_cfg["trigger_scope"]),
            corrective_error_metric=str(corrective_cfg["error_metric"]),
            corrective_log_prediction_error=bool(corrective_cfg["enabled"])
            and bool(corrective_cfg["logging"]["log_prediction_error"]),
            corrector_path=corrective_cfg["corrector_path"],
            refinement_enabled=bool(refinement_cfg["enabled"]),
            refinement_steps=int(refinement_cfg["steps"]),
            refinement_step_size=float(refinement_cfg["step_size"]),
            refinement_topk=refinement_cfg["topk"],
            refinement_goal_weight=float(refinement_cfg["goal_weight"]),
            refinement_prior_weight=float(refinement_cfg["prior_weight"]),
            refinement_smoothness_weight=float(refinement_cfg["smoothness_weight"]),
            refinement_grad_clip_norm=refinement_cfg["grad_clip_norm"],
        )
        print(
            "[planner] "
            f"type=diffusion task={policy.task or requested_task or config_name} "
            f"config={config_name} policy={cfg.policy} bundle={diffusion_bundle} "
            f"goal_offset={policy.goal_offset_steps} eval_budget={policy.eval_budget} "
            f"block_horizon={policy.block_horizon} "
            f"receding_horizon={policy.receding_horizon} "
            f"action_block={policy.action_block} "
            f"action_chunk_horizon={policy.action_chunk_horizon} "
            f"action_chunk_dim={policy.action_chunk_dim} "
            f"runtime_execute_steps={policy.runtime_execute_steps} "
            f"replan_interval={policy.flatten_receding_horizon} "
            f"action_dim={policy.action_dim} "
            f"base_num_candidates={policy.base_num_candidates} "
            f"num_candidates={policy.effective_num_candidates} "
            f"proposal_rounds={policy.proposal_rounds} "
            f"denoise_steps={policy.runtime_truncation_steps} "
            f"start_timestep={policy.runtime_start_timestep} "
            f"eta={policy.diffusion_eta:.4f} "
            f"noise_scale={policy.proposal_noise_scale:.4f} "
            f"temperature={policy.proposal_sampling_temperature:.4f} "
            f"selection_mode={policy.selection_mode} "
            f"refinement_enabled={int(policy.refinement_enabled)} "
            f"refinement_steps={policy.refinement_steps} "
            f"refinement_step_size={policy.refinement_step_size:.6f} "
            f"refinement_topk={policy.refinement_topk}"
        )
        if bool(refinement_cfg["enabled"]):
            print(
                "[refinement] "
                f"enabled=1 steps={int(refinement_cfg['steps'])} "
                f"step_size={float(refinement_cfg['step_size']):.6f} "
                f"topk={refinement_cfg['topk']} "
                f"goal_weight={float(refinement_cfg['goal_weight']):.6f} "
                f"prior_weight={float(refinement_cfg['prior_weight']):.6f} "
                f"smoothness_weight={float(refinement_cfg['smoothness_weight']):.6f} "
                f"grad_clip_norm={refinement_cfg['grad_clip_norm']}"
            )
        if bool(corrective_cfg["enabled"]):
            effective_error_interval = int(
                math.ceil(
                    int(corrective_cfg["correction_interval"]) / int(policy.action_block)
                )
                * int(policy.action_block)
            )
            print(
                "[corrective] "
                f"mode={corrective_cfg['mode']} "
                f"logging_prediction_error={int(corrective_cfg['logging']['log_prediction_error'])} "
                f"correction_interval={int(corrective_cfg['correction_interval'])} "
                f"effective_error_interval={effective_error_interval} "
                f"effective_execute_horizon={policy.runtime_execute_steps} "
                f"action_block={policy.action_block} "
                f"error_threshold={float(corrective_cfg['error_threshold']):.6f} "
                f"trigger_stat={corrective_cfg['trigger_stat']} "
                f"trigger_quantile={float(corrective_cfg['trigger_quantile']):.6f} "
                f"trigger_scope={corrective_cfg['trigger_scope']} "
                f"error_metric={corrective_cfg['error_metric']} "
                f"corrector_path={corrective_cfg['corrector_path']}"
            )
            if effective_error_interval != int(corrective_cfg["correction_interval"]):
                print(
                    "[corrective-warning] "
                    "LeWM latent rollout is block-based, so prediction error "
                    f"will be logged at step {effective_error_interval} instead of "
                    f"requested correction_interval={int(corrective_cfg['correction_interval'])}."
                )
            if int(policy.runtime_execute_steps) < effective_error_interval:
                print(
                    "[corrective-warning] "
                    "runtime_execute_steps is smaller than the effective prediction-error "
                    "checkpoint, so this run may emit no prediction_error records. "
                    "Use corrective.execute_horizon=null or set it >= effective_error_interval."
                )

    else:
        raise ValueError(
            "Unsupported planner_type "
            f"'{planner_type}'. Expected one of: "
            "random/mpc via planner_type=mpc, single_peak, multi_candidate, diffusion."
        )

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    eval_episodes, eval_start_idx, valid_start_count = sample_eval_episode_starts(
        dataset=dataset,
        ep_indices=ep_indices,
        goal_offset_steps=int(cfg.eval.goal_offset_steps),
        num_eval=int(cfg.eval.num_eval),
        seed=int(cfg.seed),
    )
    print(valid_start_count, "valid starting points found for evaluation.")
    print(eval_start_idx)

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)
    print(
        "[eval] "
        f"planner_type={planner_type} task={requested_task or config_name} "
        f"config={config_name} policy={policy_name} "
        f"dataset={cfg.eval.dataset_name} goal_offset={int(cfg.eval.goal_offset_steps)} "
        f"eval_budget={int(cfg.eval.eval_budget)} num_eval={int(cfg.eval.num_eval)}"
    )
    if planner_type == "diffusion":
        action_clip_low, action_clip_high = policy.action_clip_range
        print(
            "[planner-runtime] "
            f"planner_type=diffusion task={policy.task or requested_task or config_name} "
            f"config={config_name} policy={cfg.policy} diffusion_bundle={cfg.diffusion_bundle} "
            f"goal_offset={policy.goal_offset_steps} eval_budget={policy.eval_budget} "
            f"block_horizon={policy.block_horizon} receding_horizon={policy.receding_horizon} "
            f"action_block={policy.action_block} action_chunk_horizon={policy.action_chunk_horizon} "
            f"action_dim={policy.action_dim} action_chunk_dim={policy.action_chunk_dim} "
            f"num_candidates={policy.effective_num_candidates} selection_mode={policy.selection_mode} "
            f"action_clip_low={action_clip_low} action_clip_high={action_clip_high}"
        )

    start_time = time.time()
    trajectory_quality_result = None
    if bool(trajectory_quality_cfg["enabled"]):
        metrics, trajectory_quality_result = run_evaluation_with_trajectory_quality(
            world=world,
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset_steps=int(cfg.eval.goal_offset_steps),
            eval_budget=int(cfg.eval.eval_budget),
            episodes_idx=eval_episodes.tolist(),
            callables=eval_callables,
            video_path=results_path,
            task=requested_task or config_name,
            quality_cfg=trajectory_quality_cfg,
        )
    else:
        metrics = world.evaluate_from_dataset(
            dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset_steps=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=eval_callables,
            video_path=results_path,
        )
    end_time = time.time()
    
    print(metrics)
    success_rate = metrics.get("success_rate", None) if isinstance(metrics, dict) else None
    evaluation_time_sec = float(end_time - start_time)
    if success_rate is None:
        print(
            "[metric-warning] "
            "Evaluation metrics do not include 'success_rate'; current output is partial."
        )
    else:
        print(f"[summary] success_rate={float(success_rate):.4f}")
    if isinstance(metrics, dict) and "episode_successes" in metrics:
        episode_successes = np.asarray(metrics["episode_successes"], dtype=bool).reshape(-1)
        episode_successes_text = ",".join("1" if bool(value) else "0" for value in episode_successes)
        print(f"[summary] episode_successes={episode_successes_text}")
    print(f"[summary] evaluation_time={evaluation_time_sec:.4f}s")
    if trajectory_quality_result is not None:
        quality_summary = trajectory_quality_result["summary"]
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
            if key in quality_summary:
                print(f"[trajectory-quality] {key}={float(quality_summary[key]):.6f}")

    prediction_error_summary = None
    if (
        planner_type == "diffusion"
        and hasattr(policy, "get_prediction_error_summary")
        and isinstance(metrics, dict)
        and "episode_successes" in metrics
    ):
        prediction_error_summary = policy.get_prediction_error_summary(
            metrics["episode_successes"]
        )
        if int(prediction_error_summary["prediction_error_count"]) > 0:
            print(
                "[corrective-summary] "
                f"prediction_error_count={prediction_error_summary['prediction_error_count']} "
                f"episode_mean_count={prediction_error_summary['prediction_error_episode_mean_count']} "
                f"mean={float(prediction_error_summary['prediction_error_mean']):.6f} "
                f"max={float(prediction_error_summary['prediction_error_max']):.6f}"
            )
            print(
                "[corrective-summary] "
                f"success_mean={float(prediction_error_summary['successful_prediction_error_mean']):.6f} "
                f"failure_mean={float(prediction_error_summary['failed_prediction_error_mean']):.6f} "
                f"fail_minus_success={float(prediction_error_summary['prediction_error_fail_minus_success']):.6f} "
                f"fail_success_ratio={float(prediction_error_summary['prediction_error_fail_success_ratio']):.6f} "
                f"cohens_d={float(prediction_error_summary['prediction_error_cohens_d_fail_vs_success']):.6f}"
            )

    global_planning_calls = read_planning_stat(policy, "_num_replans")
    corrective_check_count = (
        getattr(policy, "_corrective_check_count", None) if planner_type == "diffusion" else None
    )
    corrective_replan_count = (
        getattr(policy, "_corrective_replan_count", None) if planner_type == "diffusion" else None
    )
    corrective_replan_error_records = (
        getattr(policy, "_corrective_replan_error_records", [])
        if planner_type == "diffusion"
        else []
    )
    corrective_correction_count = (
        getattr(policy, "_corrective_correction_count", None) if planner_type == "diffusion" else None
    )
    corrective_correction_norms = (
        getattr(policy, "_corrective_correction_norms", []) if planner_type == "diffusion" else []
    )
    corrective_action_delta_norms = (
        getattr(policy, "_corrective_action_delta_norms", []) if planner_type == "diffusion" else []
    )
    corrective_correction_time_total_sec = (
        getattr(policy, "_corrective_correction_time_total_sec", None)
        if planner_type == "diffusion"
        else None
    )
    corrective_replan_rate = None
    corrective_replan_error_mean = None
    corrective_replan_error_max = None
    corrective_correction_norm_mean = None
    corrective_action_delta_norm_mean = None
    avg_correction_time_sec = None
    effective_replans_per_episode = None
    planning_time_total_sec = read_planning_stat(policy, "_planning_time_total_sec")
    generation_time_total_sec = read_planning_stat(policy, "_generation_time_total_sec")
    scoring_time_total_sec = read_planning_stat(policy, "_scoring_time_total_sec")
    selection_time_total_sec = read_planning_stat(policy, "_selection_time_total_sec")
    refinement_time_total_sec = read_planning_stat(policy, "_refinement_time_total_sec")
    avg_planning_time_sec = None
    avg_generation_time_sec = None
    avg_refinement_time_sec = None
    avg_scoring_time_sec = None
    avg_selection_time_sec = None
    if planner_type == "diffusion":
        effective_replans_per_episode = int(
            math.ceil(int(cfg.eval.eval_budget) / int(policy.runtime_execute_steps))
        )
    elif planner_type == "mpc" and policy_name != "random":
        effective_replans_per_episode = int(
            math.ceil(
                int(cfg.eval.eval_budget)
                / max(1, int(cfg.plan_config.receding_horizon) * int(cfg.plan_config.action_block))
            )
        )
    if global_planning_calls is not None and int(global_planning_calls) > 0:
        if planning_time_total_sec is not None:
            avg_planning_time_sec = float(planning_time_total_sec) / float(global_planning_calls)
        if generation_time_total_sec is not None:
            avg_generation_time_sec = float(generation_time_total_sec) / float(global_planning_calls)
        if refinement_time_total_sec is not None:
            avg_refinement_time_sec = float(refinement_time_total_sec) / float(global_planning_calls)
        if scoring_time_total_sec is not None:
            avg_scoring_time_sec = float(scoring_time_total_sec) / float(global_planning_calls)
        if selection_time_total_sec is not None:
            avg_selection_time_sec = float(selection_time_total_sec) / float(global_planning_calls)
    if corrective_check_count is not None and int(corrective_check_count) > 0:
        corrective_replan_rate = float(corrective_replan_count or 0) / float(corrective_check_count)
    if corrective_replan_error_records:
        replan_errors = [
            float(record.get("trigger_error", record["max_error"]))
            for record in corrective_replan_error_records
            if "max_error" in record
        ]
        if len(replan_errors) > 0:
            corrective_replan_error_mean = float(np.mean(replan_errors))
            corrective_replan_error_max = float(np.max(replan_errors))
    if corrective_correction_norms:
        corrective_correction_norm_mean = float(np.mean(corrective_correction_norms))
    if corrective_action_delta_norms:
        corrective_action_delta_norm_mean = float(np.mean(corrective_action_delta_norms))
    if corrective_correction_count is not None and int(corrective_correction_count) > 0:
        if corrective_correction_time_total_sec is not None:
            avg_correction_time_sec = float(corrective_correction_time_total_sec) / float(corrective_correction_count)
    if global_planning_calls is not None:
        print(f"[planner-stats] global_planning_calls={global_planning_calls}")
    if corrective_check_count is not None:
        print(f"[corrective-stats] corrective_check_count={corrective_check_count}")
    if corrective_replan_count is not None:
        print(f"[corrective-stats] corrective_replan_count={corrective_replan_count}")
    if corrective_replan_rate is not None:
        print(f"[corrective-stats] corrective_replan_rate={corrective_replan_rate:.6f}")
    if corrective_correction_count is not None:
        print(f"[corrective-stats] corrective_correction_count={corrective_correction_count}")
    if corrective_replan_error_mean is not None:
        print(
            "[corrective-stats] "
            f"mean_prediction_error_before_replan={corrective_replan_error_mean:.6f} "
            f"max_prediction_error_before_replan={corrective_replan_error_max:.6f}"
        )
    if corrective_correction_norm_mean is not None:
        print(
            "[corrective-stats] "
            f"mean_correction_norm={corrective_correction_norm_mean:.6f} "
            f"mean_action_delta_norm={corrective_action_delta_norm_mean:.6f}"
        )
    if corrective_correction_time_total_sec is not None:
        print(f"[corrective-stats] correction_time_total_sec={float(corrective_correction_time_total_sec):.6f}")
    if avg_correction_time_sec is not None:
        print(f"[corrective-stats] avg_correction_time_sec={avg_correction_time_sec:.6f}")
    if effective_replans_per_episode is not None:
        print(f"[planner-stats] effective_replans_per_episode={effective_replans_per_episode}")
    if planning_time_total_sec is not None:
        print(f"[planner-stats] planning_time_total_sec={float(planning_time_total_sec):.6f}")
    if avg_planning_time_sec is not None:
        print(f"[planner-stats] avg_planning_time_sec={avg_planning_time_sec:.6f}")
    if avg_generation_time_sec is not None:
        print(f"[planner-stats] avg_generation_time_sec={avg_generation_time_sec:.6f}")
    if refinement_time_total_sec is not None:
        print(f"[planner-stats] refinement_time_total_sec={float(refinement_time_total_sec):.6f}")
    if avg_refinement_time_sec is not None:
        print(f"[planner-stats] avg_refinement_time_sec={avg_refinement_time_sec:.6f}")
    if planner_type == "diffusion":
        refinement_before = getattr(policy, "_last_refinement_cost_before", None)
        refinement_after = getattr(policy, "_last_refinement_cost_after", None)
        refinement_goal_before = getattr(policy, "_last_refinement_goal_cost_before", None)
        refinement_goal_after = getattr(policy, "_last_refinement_goal_cost_after", None)
        refinement_delta_norm = getattr(policy, "_last_refinement_delta_norm", None)
        refinement_candidate_count = getattr(policy, "_last_refinement_candidate_count", 0)
        refinement_steps_observed = getattr(policy, "_last_refinement_steps", 0)
        if refinement_before is not None and refinement_after is not None:
            print(
                "[refinement-summary] "
                f"candidate_count={int(refinement_candidate_count)} "
                f"steps={int(refinement_steps_observed)} "
                f"cost_before={float(refinement_before):.6f} "
                f"cost_after={float(refinement_after):.6f} "
                f"goal_before={float(refinement_goal_before):.6f} "
                f"goal_after={float(refinement_goal_after):.6f} "
                f"delta_norm={float(refinement_delta_norm):.6f}"
            )
    if avg_scoring_time_sec is not None:
        print(f"[planner-stats] avg_scoring_time_sec={avg_scoring_time_sec:.6f}")
    if avg_selection_time_sec is not None:
        print(f"[planner-stats] avg_selection_time_sec={avg_selection_time_sec:.6f}")

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"planner_type: {planner_type}\n")
        if planner_type == "mpc":
            solver_target, solver_params = summarize_solver_config(cfg.solver)
            f.write(f"task: {requested_task or config_name}\n")
            f.write(f"config_name: {config_name}\n")
            f.write(f"policy: {cfg.policy}\n")
            f.write(f"goal_offset: {int(cfg.eval.goal_offset_steps)}\n")
            f.write(f"eval_budget: {int(cfg.eval.eval_budget)}\n")
            f.write(f"horizon: {int(cfg.plan_config.horizon)}\n")
            f.write(f"receding_horizon: {int(cfg.plan_config.receding_horizon)}\n")
            f.write(f"action_block: {int(cfg.plan_config.action_block)}\n")
            f.write(f"solver_target: {solver_target}\n")
            f.write(f"solver: {solver_params}\n")
        if planner_type == "single_peak":
            f.write(f"single_peak_bundle: {cfg.single_peak_bundle}\n")
        if planner_type == "multi_candidate":
            f.write(f"multi_candidate_bundle: {cfg.multi_candidate_bundle}\n")
        if planner_type == "diffusion":
            f.write(f"diffusion_bundle: {cfg.diffusion_bundle}\n")
            f.write(f"task: {policy.task or requested_task or config_name}\n")
            f.write(f"config_name: {config_name}\n")
            f.write(f"policy: {cfg.policy}\n")
            f.write(f"diffusion_selection_mode: {diffusion_selection_mode}\n")
            f.write(f"diffusion_num_candidates: {policy.effective_num_candidates}\n")
            f.write(f"diffusion_base_num_candidates: {policy.base_num_candidates}\n")
            f.write(f"diffusion_proposal_rounds: {policy.proposal_rounds}\n")
            f.write(f"diffusion_truncation_steps: {policy.runtime_truncation_steps}\n")
            f.write(f"diffusion_start_timestep: {policy.runtime_start_timestep}\n")
            f.write(f"diffusion_eta: {policy.diffusion_eta}\n")
            f.write(f"diffusion_noise_scale: {policy.proposal_noise_scale}\n")
            f.write(f"diffusion_sampling_temperature: {policy.proposal_sampling_temperature}\n")
            f.write(f"diffusion_refinement_enabled: {bool(refinement_cfg['enabled'])}\n")
            f.write(f"diffusion_refinement_steps: {int(refinement_cfg['steps'])}\n")
            f.write(f"diffusion_refinement_step_size: {float(refinement_cfg['step_size'])}\n")
            f.write(f"diffusion_refinement_topk: {refinement_cfg['topk']}\n")
            f.write(f"diffusion_refinement_goal_weight: {float(refinement_cfg['goal_weight'])}\n")
            f.write(f"diffusion_refinement_prior_weight: {float(refinement_cfg['prior_weight'])}\n")
            f.write(f"diffusion_refinement_smoothness_weight: {float(refinement_cfg['smoothness_weight'])}\n")
            f.write(f"diffusion_refinement_grad_clip_norm: {refinement_cfg['grad_clip_norm']}\n")
            f.write(f"goal_offset: {policy.goal_offset_steps}\n")
            f.write(f"eval_budget: {policy.eval_budget}\n")
            f.write(f"block_horizon: {policy.block_horizon}\n")
            f.write(f"receding_horizon: {policy.receding_horizon}\n")
            f.write(f"action_block: {policy.action_block}\n")
            f.write(f"action_chunk_horizon: {int(policy.action_chunk_horizon)}\n")
            f.write(f"action_dim: {int(policy.action_dim)}\n")
            f.write(f"action_chunk_dim: {int(policy.action_chunk_dim)}\n")
            f.write(f"runtime_execute_steps: {int(policy.runtime_execute_steps)}\n")
            f.write(f"replan_interval: {int(policy.flatten_receding_horizon)}\n")
            f.write(f"corrective_enabled: {bool(corrective_cfg['enabled'])}\n")
            f.write(f"corrective_mode: {corrective_cfg['mode']}\n")
            f.write(f"corrective_correction_interval: {int(corrective_cfg['correction_interval'])}\n")
            f.write(f"corrective_error_threshold: {float(corrective_cfg['error_threshold'])}\n")
            f.write(f"corrective_trigger_stat: {corrective_cfg['trigger_stat']}\n")
            f.write(f"corrective_trigger_quantile: {float(corrective_cfg['trigger_quantile'])}\n")
            f.write(f"corrective_trigger_scope: {corrective_cfg['trigger_scope']}\n")
            f.write(f"corrective_error_metric: {corrective_cfg['error_metric']}\n")
            f.write(f"corrective_corrector_path: {corrective_cfg['corrector_path']}\n")
            f.write(
                "corrective_log_prediction_error: "
                f"{bool(corrective_cfg['logging']['log_prediction_error'])}\n"
            )
            action_clip_low, action_clip_high = policy.action_clip_range
            f.write(f"action_clip_low: {action_clip_low}\n")
            f.write(f"action_clip_high: {action_clip_high}\n")
        if global_planning_calls is not None:
            f.write(f"global_planning_calls: {global_planning_calls}\n")
        if corrective_check_count is not None:
            f.write(f"corrective_check_count: {corrective_check_count}\n")
        if corrective_replan_count is not None:
            f.write(f"corrective_replan_count: {corrective_replan_count}\n")
        if corrective_replan_rate is not None:
            f.write(f"corrective_replan_rate: {corrective_replan_rate}\n")
        if corrective_correction_count is not None:
            f.write(f"corrective_correction_count: {corrective_correction_count}\n")
        if corrective_replan_error_mean is not None:
            f.write(f"mean_prediction_error_before_replan: {corrective_replan_error_mean}\n")
            f.write(f"max_prediction_error_before_replan: {corrective_replan_error_max}\n")
        if corrective_correction_norm_mean is not None:
            f.write(f"mean_correction_norm: {corrective_correction_norm_mean}\n")
            f.write(f"mean_action_delta_norm: {corrective_action_delta_norm_mean}\n")
        if corrective_correction_time_total_sec is not None:
            f.write(f"correction_time_total_sec: {float(corrective_correction_time_total_sec)}\n")
        if avg_correction_time_sec is not None:
            f.write(f"avg_correction_time_sec: {avg_correction_time_sec}\n")
        if effective_replans_per_episode is not None:
            f.write(f"effective_replans_per_episode: {effective_replans_per_episode}\n")
        if planning_time_total_sec is not None:
            f.write(f"planning_time_total_sec: {float(planning_time_total_sec)}\n")
        if avg_planning_time_sec is not None:
            f.write(f"avg_planning_time_sec: {avg_planning_time_sec}\n")
        if avg_generation_time_sec is not None:
            f.write(f"avg_generation_time_sec: {avg_generation_time_sec}\n")
        if refinement_time_total_sec is not None:
            f.write(f"refinement_time_total_sec: {float(refinement_time_total_sec)}\n")
        if avg_refinement_time_sec is not None:
            f.write(f"avg_refinement_time_sec: {avg_refinement_time_sec}\n")
        if planner_type == "diffusion":
            f.write(f"last_refinement_cost_before: {getattr(policy, '_last_refinement_cost_before', None)}\n")
            f.write(f"last_refinement_cost_after: {getattr(policy, '_last_refinement_cost_after', None)}\n")
            f.write(f"last_refinement_goal_cost_before: {getattr(policy, '_last_refinement_goal_cost_before', None)}\n")
            f.write(f"last_refinement_goal_cost_after: {getattr(policy, '_last_refinement_goal_cost_after', None)}\n")
            f.write(f"last_refinement_delta_norm: {getattr(policy, '_last_refinement_delta_norm', None)}\n")
            f.write(f"last_refinement_candidate_count: {getattr(policy, '_last_refinement_candidate_count', 0)}\n")
            f.write(f"last_refinement_steps: {getattr(policy, '_last_refinement_steps', 0)}\n")
        if avg_scoring_time_sec is not None:
            f.write(f"avg_scoring_time_sec: {avg_scoring_time_sec}\n")
        if avg_selection_time_sec is not None:
            f.write(f"avg_selection_time_sec: {avg_selection_time_sec}\n")
        if trajectory_quality_result is not None:
            f.write(f"trajectory_quality_enabled: {bool(trajectory_quality_cfg['enabled'])}\n")
            f.write(f"trajectory_quality_summary: {trajectory_quality_result['summary']}\n")
            if bool(trajectory_quality_cfg["save_npz"]):
                npz_path = results_path.parent / f"trajectory_quality_{requested_task or config_name}.npz"
                npz_payload = {}
                for prefix, values in [
                    ("per_episode", trajectory_quality_result["per_episode"]),
                    ("distance", trajectory_quality_result["distances"]),
                    ("trace", trajectory_quality_result["trace"]),
                ]:
                    for key, value in values.items():
                        safe_key = f"{prefix}_{key}".replace("/", "_")
                        array = np.asarray(value)
                        if array.dtype.kind in {"b", "i", "u", "f", "c"}:
                            npz_payload[safe_key] = array
                npz_payload["successes_by_step"] = np.asarray(
                    trajectory_quality_result["successes_by_step"],
                    dtype=bool,
                )
                np.savez_compressed(npz_path, **npz_payload)
                f.write(f"trajectory_quality_npz: {npz_path}\n")
        f.write(f"metrics: {metrics}\n")
        if prediction_error_summary is not None:
            f.write(f"prediction_error_summary: {prediction_error_summary}\n")
        if success_rate is not None:
            f.write(f"success_rate: {float(success_rate)}\n")
        f.write(f"evaluation_time: {evaluation_time_sec} seconds\n")


if __name__ == "__main__":
    sys.argv = normalize_eval_cli_args(list(sys.argv))
    run()
