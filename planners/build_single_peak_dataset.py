from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planners.single_peak_data import (  # noqa: E402
    clone_info_dict,
    encode_current_goal,
    extract_teacher_plan,
    extract_trajectory_action_chunk,
    validate_single_peak_sample,
)


TASK_ALIASES = {
    "pusht": "pusht",
    "tworoom": "tworoom",
    "two-room": "tworoom",
    "two_room": "tworoom",
    "reacher": "reacher",
    "researcher": "reacher",
}


@dataclass(frozen=True)
class TaskSpec:
    requested_task_name: str
    canonical_task_name: str
    eval_config_name: str
    dataset_name: str
    goal_offset_steps: int
    eval_budget: int
    receding_horizon: int
    action_block: int
    action_chunk_horizon: int
    pixels_key: str
    action_key: str
    episode_key: str
    step_key: str
    action_dim: int
    action_chunk_dim: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single-peak planner teacher dataset from LeWM + MPC-CEM teacher.",
    )
    parser.add_argument(
        "--mode",
        choices=["probe", "build"],
        required=True,
        help="probe: sample a small subset and print sanity info; build: save the full requested dataset.",
    )
    parser.add_argument(
        "--task",
        default="pusht",
        help="Task name mapped to config/eval/<task>.yaml when --config is not provided.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional explicit eval config path. Overrides --task when provided.",
    )
    parser.add_argument(
        "--wm-policy",
        required=True,
        help="World-model checkpoint path understood by stable_worldmodel.policy.AutoCostModel.",
    )
    parser.add_argument(
        "--teacher-policy",
        default=None,
        help="Teacher planner checkpoint path. Used only when --label-source=teacher.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help=(
            "Optional dataset name override used by stable_worldmodel.HDF5Dataset. "
            "Defaults to eval.dataset_name or the stem of --dataset-h5."
        ),
    )
    parser.add_argument(
        "--dataset-h5",
        default=None,
        help=(
            "Optional explicit HDF5 path. When provided, the builder reads this file "
            "directly instead of resolving eval.dataset_name under --cache-dir."
        ),
    )
    parser.add_argument(
        "--label-source",
        choices=["trajectory", "teacher"],
        default="trajectory",
        help="trajectory: labels come from raw action chunks in the HDF5 trajectory; teacher: labels come from MPC-CEM.",
    )
    parser.add_argument(
        "--plan-horizon",
        type=int,
        default=None,
        help=(
            "Optional flattened action horizon override. When set, the builder derives "
            "receding_horizon from --plan-horizon and --action-block instead of the eval config."
        ),
    )
    parser.add_argument(
        "--action-block",
        type=int,
        default=None,
        help=(
            "Action block size used with --plan-horizon. Defaults to the eval config action_block."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Number of teacher samples to construct.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output .pt path. Required in build mode; optional in probe mode.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of planning queries solved together in one batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for row sampling.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for world model and teacher solver.",
    )
    parser.add_argument(
        "--on-error",
        choices=["skip", "raise"],
        default="skip",
        help="Failure handling strategy for bad samples.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional stable-worldmodel cache dir for datasets and checkpoints.",
    )
    parser.add_argument(
        "--model-cache-dir",
        default=None,
        help=(
            "Optional cache dir used only for resolving wm/teacher checkpoints. "
            "When omitted, reuses --cache-dir if it points to a directory; otherwise "
            "falls back to the stable-worldmodel default cache root."
        ),
    )
    return parser.parse_args()


def normalize_task_name(task_name: str) -> str:
    normalized = str(task_name).strip().lower()
    return TASK_ALIASES.get(normalized, normalized)


def infer_requested_task_name(args: argparse.Namespace) -> str:
    if args.config is not None:
        return normalize_task_name(Path(args.config).stem)
    return normalize_task_name(args.task)


def require_cfg_value(cfg: DictConfig, path: str) -> Any:
    current: Any = cfg
    traversed: list[str] = []
    for part in path.split("."):
        traversed.append(part)
        if not isinstance(current, (DictConfig, dict)) or part not in current:
            raise KeyError(f"Could not infer '{path}': missing config field '{'.'.join(traversed)}'.")
        if isinstance(current, DictConfig) and OmegaConf.is_missing(current, part):
            raise KeyError(f"Could not infer '{path}': config field '{'.'.join(traversed)}' is missing.")
        current = current[part]
    if current in [None, "", "null"]:
        raise KeyError(f"Could not infer '{path}': config field is empty.")
    return current


def resolve_model_cache_dir(
    *,
    shared_cache_dir: str | None,
    explicit_model_cache_dir: str | None,
) -> str | None:
    if explicit_model_cache_dir not in [None, "", "null"]:
        return str(Path(explicit_model_cache_dir).expanduser())
    if shared_cache_dir in [None, "", "null"]:
        return None
    shared_path = Path(shared_cache_dir).expanduser()
    if shared_path.suffix == ".h5":
        return None
    return str(shared_path)


def validate_model_reference(model_ref: str, flag_name: str) -> None:
    normalized = str(model_ref).strip()
    if normalized in ["", "null"]:
        raise ValueError(f"{flag_name} must point to a real checkpoint path or run name, got '{model_ref}'.")
    if normalized.startswith("/path/to/"):
        raise ValueError(
            f"{flag_name} is still a placeholder: '{model_ref}'. "
            "Pass a real world-model checkpoint path or stable_worldmodel run name."
        )


def load_eval_cfg(args: argparse.Namespace) -> DictConfig:
    config_dir = REPO_ROOT / "config" / "eval"
    if args.config is not None:
        config_path = Path(args.config).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Eval config not found: {config_path}")
        same_eval_dir = config_path.parent.resolve() == config_dir.resolve()
        if same_eval_dir and config_path.suffix == ".yaml":
            with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
                cfg = hydra.compose(config_name=config_path.stem)
        else:
            cfg = OmegaConf.load(config_path)
    else:
        config_name = normalize_task_name(args.task)
        if not (config_dir / f"{config_name}.yaml").exists():
            raise FileNotFoundError(
                "Eval config not found for "
                f"task='{args.task}' (normalized='{config_name}'): {config_dir / f'{config_name}.yaml'}"
            )
        with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = hydra.compose(config_name=config_name)
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    return cfg


def img_transform(cfg: DictConfig):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def resolve_dataset_cache_dir(
    cfg: DictConfig,
    dataset_name: str,
    cache_dir: str | None = None,
) -> Path:
    dataset_root = Path(cache_dir or cfg.get("cache_dir") or swm.data.utils.get_cache_dir()).expanduser()

    # Support either:
    #   1) <root>/<dataset>.h5
    #   2) <root>/<dataset>/<dataset>.h5
    #   3) a direct <dataset>.h5 path
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
        raise ValueError(f"--dataset-h5 must point to a .h5 file, got {h5_path}.")
    if dataset_name not in [None, "", "null"]:
        requested_name = str(dataset_name).strip()
        expected_rel_path = Path(f"{requested_name}.h5")
        parent_depth = max(0, len(expected_rel_path.parts) - 1)
        if parent_depth < len(h5_path.parents):
            candidate_cache_dir = h5_path.parents[parent_depth]
            if candidate_cache_dir / expected_rel_path == h5_path:
                return requested_name, candidate_cache_dir
    return h5_path.stem, h5_path.parent


def infer_dataset_keys_to_load(cfg: DictConfig) -> list[str]:
    keys_to_load = list(cfg.dataset.get("keys_to_load", []) or [])

    # Keep the loader task-agnostic: always load pixels + action, then add
    # task-specific cached state keys such as proprio/state.
    required_keys = ["pixels", "action", *list(cfg.dataset.keys_to_cache)]
    for key in required_keys:
        if key not in keys_to_load:
            keys_to_load.append(key)
    return keys_to_load


def get_dataset(
    cfg: DictConfig,
    dataset_name: str,
    cache_dir: str | None = None,
    dataset_h5: str | None = None,
):
    if dataset_h5 not in [None, "", "null"]:
        dataset_name, dataset_path = resolve_explicit_dataset_source(
            str(dataset_h5),
            dataset_name=dataset_name,
        )
    else:
        dataset_path = resolve_dataset_cache_dir(cfg, dataset_name, cache_dir=cache_dir)
    h5_path = dataset_path / f"{dataset_name}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            "Dataset HDF5 not found. "
            f"Expected {h5_path} for dataset_name='{dataset_name}' and cache_dir='{dataset_path}'."
        )
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_load=infer_dataset_keys_to_load(cfg),
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


def get_available_dataset_columns(dataset) -> list[str]:
    dataset._open()
    if getattr(dataset, "h5_file", None) is None:
        return list(dataset.column_names)
    return list(dataset.h5_file.keys())


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


def resolve_task_spec(args: argparse.Namespace, cfg: DictConfig, dataset) -> TaskSpec:
    requested_task_name = infer_requested_task_name(args)
    canonical_task_name = normalize_task_name(requested_task_name)
    eval_config_name = (
        normalize_task_name(Path(args.config).stem)
        if args.config is not None
        else canonical_task_name
    )

    dataset_name = str(require_cfg_value(cfg, "eval.dataset_name"))
    goal_offset_steps = int(require_cfg_value(cfg, "eval.goal_offset_steps"))
    eval_budget = int(require_cfg_value(cfg, "eval.eval_budget"))
    config_receding_horizon = int(require_cfg_value(cfg, "plan_config.receding_horizon"))
    config_action_block = int(require_cfg_value(cfg, "plan_config.action_block"))
    action_block = config_action_block if args.action_block is None else int(args.action_block)
    if args.plan_horizon is None:
        receding_horizon = config_receding_horizon
    else:
        plan_horizon = int(args.plan_horizon)
        if plan_horizon <= 0:
            raise ValueError(f"--plan-horizon must be positive, got {plan_horizon}.")
        if action_block <= 0:
            raise ValueError(f"--action-block must be positive, got {action_block}.")
        if plan_horizon % action_block != 0:
            raise ValueError(
                "--plan-horizon must be divisible by --action-block: "
                f"{plan_horizon} % {action_block} != 0."
            )
        receding_horizon = int(plan_horizon // action_block)
    if goal_offset_steps <= 0:
        raise ValueError(f"goal_offset_steps must be positive, got {goal_offset_steps}.")
    if eval_budget <= 0:
        raise ValueError(f"eval_budget must be positive, got {eval_budget}.")
    if receding_horizon <= 0:
        raise ValueError(f"receding_horizon must be positive, got {receding_horizon}.")
    if action_block <= 0:
        raise ValueError(f"action_block must be positive, got {action_block}.")

    available_columns = get_available_dataset_columns(dataset)
    pixels_key = infer_dataset_column(
        available_columns=available_columns,
        field_name="pixels_key",
        exact_candidates=["pixels"],
        prefix_candidates=["pixels"],
    )
    action_key = infer_dataset_column(
        available_columns=available_columns,
        field_name="action_key",
        exact_candidates=["action"],
        prefix_candidates=["action"],
    )
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

    action_dim = int(dataset.get_dim(action_key))
    if action_dim <= 0:
        raise ValueError(f"Could not infer a positive action_dim from dataset key '{action_key}': {action_dim}.")

    action_chunk_horizon = int(receding_horizon * action_block)
    action_chunk_dim = int(action_chunk_horizon * action_dim)

    return TaskSpec(
        requested_task_name=requested_task_name,
        canonical_task_name=canonical_task_name,
        eval_config_name=eval_config_name,
        dataset_name=dataset_name,
        goal_offset_steps=goal_offset_steps,
        eval_budget=eval_budget,
        receding_horizon=receding_horizon,
        action_block=action_block,
        action_chunk_horizon=action_chunk_horizon,
        pixels_key=pixels_key,
        action_key=action_key,
        episode_key=episode_key,
        step_key=step_key,
        action_dim=action_dim,
        action_chunk_dim=action_chunk_dim,
    )


def log_task_spec(task_spec: TaskSpec) -> None:
    print(
        "[task-spec] "
        f"requested={task_spec.requested_task_name} "
        f"canonical={task_spec.canonical_task_name} "
        f"eval_config={task_spec.eval_config_name} "
        f"dataset_name={task_spec.dataset_name} "
        f"goal_offset_steps={task_spec.goal_offset_steps} "
        f"eval_budget={task_spec.eval_budget} "
        f"receding_horizon={task_spec.receding_horizon} "
        f"action_block={task_spec.action_block} "
        f"action_chunk_horizon={task_spec.action_chunk_horizon} "
        f"action_dim={task_spec.action_dim} "
        f"action_chunk_dim={task_spec.action_chunk_dim} "
        f"pixels_key={task_spec.pixels_key} "
        f"action_key={task_spec.action_key} "
        f"episode_key={task_spec.episode_key} "
        f"step_key={task_spec.step_key}"
    )


def log_tensor_shapes(tag: str, info_dict: dict[str, Any]) -> None:
    shape_parts: list[str] = []
    for key in sorted(info_dict.keys()):
        value = info_dict[key]
        if torch.is_tensor(value):
            shape_parts.append(f"{key}={tuple(value.shape)}")
    print(f"[{tag}] {' '.join(shape_parts)}")


def get_episodes_length(dataset, episodes: np.ndarray, episode_key: str, step_key: str) -> np.ndarray:
    episode_idx = dataset.get_col_data(episode_key)
    step_idx = dataset.get_col_data(step_key)
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def build_processors(cfg: DictConfig, dataset) -> dict[str, Any]:
    process: dict[str, Any] = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]
    return process


def infer_plan_horizon(cfg: DictConfig) -> int:
    return int(cfg.plan_config.receding_horizon * cfg.plan_config.action_block)


def sample_valid_rows(
    task_spec: TaskSpec,
    dataset,
    num_samples: int,
    seed: int,
    required_context_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ep_indices, _ = np.unique(dataset.get_col_data(task_spec.episode_key), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices, task_spec.episode_key, task_spec.step_key)
    max_start_idx = episode_len - required_context_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    episode_idx_per_row = dataset.get_col_data(task_spec.episode_key)
    step_idx_per_row = dataset.get_col_data(task_spec.step_key)
    max_start_per_row = np.array([max_start_idx_dict[ep_id] for ep_id in episode_idx_per_row])
    valid_mask = step_idx_per_row <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    if len(valid_indices) < num_samples:
        raise ValueError(
            "Requested "
            f"{num_samples} samples but only found {len(valid_indices)} valid starting points "
            f"for dataset_name='{task_spec.dataset_name}' with goal_offset_steps={task_spec.goal_offset_steps} "
            f"and action_chunk_horizon={task_spec.action_chunk_horizon}."
        )
    rng = np.random.default_rng(seed)
    sampled_rows = np.sort(rng.choice(valid_indices, size=num_samples, replace=False))
    # Only index lightweight metadata columns here. dataset.get_row_data(sampled_rows)
    # would also read pixels: [num_samples, H, W, C], which is enormous for 1000k builds.
    sampled_episodes = episode_idx_per_row[sampled_rows]
    sampled_start_steps = step_idx_per_row[sampled_rows]
    return sampled_rows, sampled_episodes, sampled_start_steps


def load_goal_conditioned_batch(
    dataset,
    task_spec: TaskSpec,
    episodes_idx: np.ndarray,
    start_steps: np.ndarray,
    goal_offset_steps: int,
    context_steps: int,
    keep_action_sequence: bool,
) -> dict[str, torch.Tensor]:
    end_steps = start_steps + context_steps
    data = dataset.load_chunk(episodes_idx, start_steps, end_steps)
    goal_index = int(goal_offset_steps - 1)

    current_per_key: dict[str, list[torch.Tensor]] = {}
    goal_per_key: dict[str, list[torch.Tensor]] = {}
    action_sequence: list[torch.Tensor] = []

    for ep in data:
        for col, value in ep.items():
            if not torch.is_tensor(value):
                continue

            resolved_key = col
            if col == task_spec.pixels_key:
                resolved_key = "pixels"
                value = value.permute(0, 2, 3, 1)  # [T, H, W, C]
            elif col == task_spec.action_key:
                resolved_key = "action"

            if goal_index < 0 or goal_index >= int(value.shape[0]):
                raise ValueError(
                    f"goal_offset_steps={goal_offset_steps} selected goal_index={goal_index}, "
                    f"but loaded chunk for column '{resolved_key}' has only {value.shape[0]} steps."
                )

            current_value = value[0]  # [...]
            goal_value = value[goal_index]  # [...]

            if resolved_key == "action" and keep_action_sequence:
                action_sequence.append(value[:context_steps])  # [context_steps, action_dim]
            else:
                current_per_key.setdefault(resolved_key, []).append(current_value)
            goal_key = "goal" if resolved_key == "pixels" else f"goal_{resolved_key}"
            goal_per_key.setdefault(goal_key, []).append(goal_value)

    info_dict: dict[str, torch.Tensor] = {}
    for key, values in current_per_key.items():
        stacked = torch.stack(values, dim=0)  # [B, ...]
        info_dict[key] = stacked.unsqueeze(1)  # [B, 1, ...]
    if keep_action_sequence:
        info_dict["action"] = torch.stack(action_sequence, dim=0)  # [B, context_steps, action_dim]
    for key, values in goal_per_key.items():
        stacked = torch.stack(values, dim=0)  # [B, ...]
        info_dict[key] = stacked.unsqueeze(1)  # [B, 1, ...]
    return info_dict


def prepare_policy_inputs(
    info_dict: dict[str, torch.Tensor],
    process: dict[str, Any],
    transform: dict[str, Any],
) -> dict[str, torch.Tensor]:
    numpy_info: dict[str, Any] = {}
    for key, value in clone_info_dict(info_dict).items():
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        if key == "action":
            value = np.nan_to_num(value, nan=0.0)
        numpy_info[key] = value

    policy = swm.policy.WorldModelPolicy(
        solver=swm.solver.CEMSolver(model=torch.nn.Identity()),
        config=swm.PlanConfig(horizon=1, receding_horizon=1, action_block=1),
        process=process,
        transform=transform,
    )
    prepared = policy._prepare_info(numpy_info)
    if "action" in prepared and torch.is_tensor(prepared["action"]):
        prepared["action"] = torch.nan_to_num(prepared["action"], nan=0.0)
    return prepared


def build_solver(
    cfg: DictConfig,
    teacher_model: torch.nn.Module,
    action_space: Any,
    n_envs: int,
) -> Any:
    plan_config = swm.PlanConfig(**OmegaConf.to_container(cfg.plan_config, resolve=True))
    solver = swm.solver.CEMSolver(
        model=teacher_model,
        batch_size=cfg.solver.batch_size,
        num_samples=cfg.solver.num_samples,
        var_scale=cfg.solver.var_scale,
        n_steps=cfg.solver.n_steps,
        topk=cfg.solver.topk,
        device=cfg.solver.device,
        seed=cfg.solver.seed,
    )
    solver.configure(action_space=action_space, n_envs=n_envs, config=plan_config)
    return solver, plan_config


def make_action_space(cfg: DictConfig, num_envs: int):
    world_kwargs = OmegaConf.to_container(cfg.world, resolve=True)
    world_kwargs["num_envs"] = int(num_envs)
    image_size = int(require_cfg_value(cfg, "eval.img_size"))
    world = swm.World(**world_kwargs, image_shape=(image_size, image_size), verbose=0)
    try:
        return world.action_space
    finally:
        world.close()


def collate_samples(
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    cfg: DictConfig,
    task_spec: TaskSpec,
) -> dict[str, Any]:
    if len(samples) == 0:
        raise ValueError("No samples were successfully constructed.")
    z_cur = torch.stack([sample["z_cur"] for sample in samples], dim=0)  # [N, embed_dim]
    z_goal = torch.stack([sample["z_goal"] for sample in samples], dim=0)  # [N, embed_dim]
    teacher_plan = torch.stack([sample["teacher_plan"] for sample in samples], dim=0)  # [N, action_chunk_dim]
    meta = [sample["meta"] for sample in samples]
    plan_config = dict(OmegaConf.to_container(cfg.plan_config, resolve=True))
    plan_config["receding_horizon"] = int(task_spec.receding_horizon)
    plan_config["action_block"] = int(task_spec.action_block)
    return {
        "z_cur": z_cur,
        "z_goal": z_goal,
        "teacher_plan": teacher_plan,
        "meta": meta,
        "build_info": {
            "mode": args.mode,
            "task": task_spec.canonical_task_name,
            "requested_task": task_spec.requested_task_name,
            "label_source": args.label_source,
            "config_path": str(args.config) if args.config is not None else None,
            "eval_config_name": task_spec.eval_config_name,
            "dataset_name": task_spec.dataset_name,
            "wm_policy": args.wm_policy,
            "teacher_policy": (args.teacher_policy or args.wm_policy) if args.label_source == "teacher" else None,
            "num_samples": len(samples),
            "seed": args.seed,
            "goal_offset_steps": int(task_spec.goal_offset_steps),
            "eval_budget": int(task_spec.eval_budget),
            "action_dim": int(task_spec.action_dim),
            "action_chunk_horizon": int(task_spec.action_chunk_horizon),
            "action_chunk_dim": int(task_spec.action_chunk_dim),
            "pixels_key": task_spec.pixels_key,
            "action_key": task_spec.action_key,
            "episode_key": task_spec.episode_key,
            "step_key": task_spec.step_key,
            "plan_config": plan_config,
            "task_spec": asdict(task_spec),
        },
    }


def maybe_save_dataset(output: dict[str, Any], output_path: str | None) -> None:
    if output_path is None:
        return
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    print(f"[save] wrote dataset to {path}")


def main() -> None:
    args = parse_args()
    if args.mode == "build" and args.output_path is None:
        raise ValueError("--output-path is required in build mode.")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    start_time = time.time()
    cfg = load_eval_cfg(args)
    if args.dataset_name not in [None, "", "null"]:
        cfg.eval.dataset_name = str(args.dataset_name)
    if args.cache_dir is not None:
        cfg.cache_dir = args.cache_dir

    if args.mode == "probe":
        args.num_samples = min(args.num_samples, 8)

    dataset = get_dataset(
        cfg,
        cfg.eval.dataset_name,
        cache_dir=args.cache_dir,
        dataset_h5=args.dataset_h5,
    )
    print(f"[dataset] h5={dataset.h5_path} keys={dataset.column_names}")
    task_spec = resolve_task_spec(args=args, cfg=cfg, dataset=dataset)
    log_task_spec(task_spec)
    model_cache_dir = resolve_model_cache_dir(
        shared_cache_dir=args.cache_dir,
        explicit_model_cache_dir=args.model_cache_dir,
    )
    validate_model_reference(args.wm_policy, "--wm-policy")
    if args.teacher_policy is not None:
        validate_model_reference(args.teacher_policy, "--teacher-policy")
    print(f"[model] wm_policy={args.wm_policy} model_cache_dir={model_cache_dir}")
    process = build_processors(cfg, dataset)
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    plan_horizon = int(task_spec.action_chunk_horizon)
    if args.label_source == "trajectory":
        required_context_steps = max(int(task_spec.goal_offset_steps), int(plan_horizon))
    else:
        required_context_steps = int(task_spec.goal_offset_steps)

    sampled_rows, sampled_episodes, sampled_start_steps = sample_valid_rows(
        task_spec=task_spec,
        dataset=dataset,
        num_samples=args.num_samples,
        seed=args.seed,
        required_context_steps=required_context_steps,
    )

    wm_model = swm.policy.AutoCostModel(args.wm_policy, cache_dir=model_cache_dir).to(args.device).eval()
    wm_model.requires_grad_(False)
    if args.label_source == "teacher":
        teacher_policy = args.teacher_policy or args.wm_policy
        teacher_model = swm.policy.AutoCostModel(teacher_policy, cache_dir=model_cache_dir).to(args.device).eval()
        teacher_model.requires_grad_(False)
    else:
        teacher_policy = None
        teacher_model = None

    all_samples: list[dict[str, Any]] = []
    failures = 0

    for batch_start in range(0, args.num_samples, args.batch_size):
        batch_end = min(batch_start + args.batch_size, args.num_samples)
        batch_slice = slice(batch_start, batch_end)
        batch_rows = sampled_rows[batch_slice]
        batch_episodes = sampled_episodes[batch_slice]
        batch_start_steps = sampled_start_steps[batch_slice]
        current_bs = len(batch_rows)

        print(
            f"[batch] rows {batch_start}:{batch_end} "
            f"(size={current_bs}, first_row={int(batch_rows[0])}, last_row={int(batch_rows[-1])})"
        )

        try:
            raw_info = load_goal_conditioned_batch(
                dataset=dataset,
                task_spec=task_spec,
                episodes_idx=batch_episodes,
                start_steps=batch_start_steps,
                goal_offset_steps=int(task_spec.goal_offset_steps),
                context_steps=int(required_context_steps),
                keep_action_sequence=args.label_source == "trajectory",
            )
            if batch_start == 0:
                log_tensor_shapes("raw-info", raw_info)
            prepared_info = prepare_policy_inputs(raw_info, process=process, transform=transform)
            if batch_start == 0:
                log_tensor_shapes("prepared-info", prepared_info)
            prepared_action_dim = int(prepared_info["action"].shape[-1])
            if prepared_action_dim != int(task_spec.action_dim):
                raise ValueError(
                    "Prepared action dim does not match inferred task action_dim: "
                    f"{prepared_action_dim} != {task_spec.action_dim}."
                )
            plan_config = None
            solver_outputs = None
            if args.label_source == "teacher":
                action_space = make_action_space(cfg, num_envs=current_bs)
                solver, plan_config = build_solver(
                    cfg=cfg,
                    teacher_model=teacher_model,
                    action_space=action_space,
                    n_envs=current_bs,
                )
                solver_outputs = solver.solve(clone_info_dict(prepared_info))
        except Exception as exc:
            if args.on_error == "raise":
                raise
            failures += current_bs
            print(f"[skip-batch] batch rows {batch_start}:{batch_end} failed: {exc}")
            continue

        for env_index in range(current_bs):
            try:
                z_cur, z_goal = encode_current_goal(
                    model=wm_model,
                    info_dict=prepared_info,
                    env_index=env_index,
                )
                if args.label_source == "teacher":
                    teacher_plan = extract_teacher_plan(
                        solver_outputs=solver_outputs,
                        plan_config=plan_config,
                        info_dict=prepared_info,
                        env_index=env_index,
                    )
                else:
                    teacher_plan = extract_trajectory_action_chunk(
                        info_dict=prepared_info,
                        plan_horizon=plan_horizon,
                        env_index=env_index,
                    )
                if int(teacher_plan.numel()) != int(task_spec.action_chunk_dim):
                    raise ValueError(
                        "teacher_plan width does not match inferred task action_chunk_dim: "
                        f"{teacher_plan.numel()} != {task_spec.action_chunk_dim}."
                    )
                sample = validate_single_peak_sample(
                    {
                        "z_cur": z_cur,
                        "z_goal": z_goal,
                        "teacher_plan": teacher_plan,
                        "meta": {
                            "dataset_row": int(batch_rows[env_index]),
                            "episode_id": int(batch_episodes[env_index]),
                            "step": int(batch_start_steps[env_index]),
                            "goal_step": int(batch_start_steps[env_index] + task_spec.goal_offset_steps - 1),
                            "env_index": int(env_index),
                            "plan_horizon": int(plan_horizon),
                            "action_dim": int(task_spec.action_dim),
                            "action_chunk_dim": int(task_spec.action_chunk_dim),
                            "label_source": args.label_source,
                            "action_end_step": int(batch_start_steps[env_index] + plan_horizon - 1),
                            "task": task_spec.canonical_task_name,
                            "dataset_name": task_spec.dataset_name,
                        },
                    }
                )
                all_samples.append(sample)
            except Exception as exc:
                if args.on_error == "raise":
                    raise
                failures += 1
                print(
                    f"[skip-sample] row={int(batch_rows[env_index])} "
                    f"episode={int(batch_episodes[env_index])} step={int(batch_start_steps[env_index])}: {exc}"
                )

    output = collate_samples(all_samples, args=args, cfg=cfg, task_spec=task_spec)
    maybe_save_dataset(output, args.output_path)

    elapsed = time.time() - start_time
    print(
        f"[done] mode={args.mode} success={output['z_cur'].shape[0]} failures={failures} "
        f"label_source={args.label_source} "
        f"z_dim={output['z_cur'].shape[-1]} teacher_plan_dim={output['teacher_plan'].shape[-1]} "
        f"action_dim={task_spec.action_dim} action_chunk_horizon={task_spec.action_chunk_horizon} "
        f"action_chunk_dim={task_spec.action_chunk_dim} "
        f"time={elapsed:.2f}s"
    )

    if args.mode == "probe":
        print("[probe] first sample shapes:")
        print(f"  z_cur: {tuple(output['z_cur'][0].shape)}")  # [embed_dim]
        print(f"  z_goal: {tuple(output['z_goal'][0].shape)}")  # [embed_dim]
        print(f"  teacher_plan: {tuple(output['teacher_plan'][0].shape)}")  # [action_chunk_dim]
        print(f"  action_dim: {task_spec.action_dim}")
        print(f"  action_chunk_horizon: {task_spec.action_chunk_horizon}")
        print(f"  action_chunk_dim: {task_spec.action_chunk_dim}")
        print(f"  meta: {output['meta'][0]}")


if __name__ == "__main__":
    main()
