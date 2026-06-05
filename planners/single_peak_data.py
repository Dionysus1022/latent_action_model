from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


@dataclass
class SinglePeakTeacherSample:
    """Single sample for the single-peak fast planner baseline."""

    z_cur: torch.Tensor
    z_goal: torch.Tensor
    teacher_plan: torch.Tensor
    meta: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "z_cur": self.z_cur,
            "z_goal": self.z_goal,
            "teacher_plan": self.teacher_plan,
            "meta": self.meta,
        }


class SinglePeakTeacherDataset(Dataset):
    """Lightweight in-memory container with a stable sample schema."""

    def __init__(self, samples: Iterable[SinglePeakTeacherSample | dict[str, Any]]):
        self._samples: list[dict[str, Any]] = []
        for sample in samples:
            if isinstance(sample, SinglePeakTeacherSample):
                sample_dict = sample.as_dict()
            else:
                sample_dict = dict(sample)
            self._samples.append(validate_single_peak_sample(sample_dict))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._samples[index]
        return {
            "z_cur": sample["z_cur"].clone(),
            "z_goal": sample["z_goal"].clone(),
            "teacher_plan": sample["teacher_plan"].clone(),
            "meta": deepcopy(sample["meta"]),
        }


def infer_action_chunk_dim(plan_horizon: int, action_dim: int) -> int:
    """Infer flattened action chunk dim from horizon and single-step action dim."""
    if plan_horizon <= 0:
        raise ValueError(f"plan_horizon must be positive, got {plan_horizon}.")
    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}.")
    return int(plan_horizon * action_dim)


def flatten_action_chunk(action_chunk: torch.Tensor) -> torch.Tensor:
    """Flatten an action chunk.

    action_chunk: [plan_horizon, action_dim]
    returns: [plan_horizon * action_dim]
    """
    if not torch.is_tensor(action_chunk):
        raise TypeError(f"Expected torch.Tensor, got {type(action_chunk)}.")
    if action_chunk.ndim != 2:
        raise ValueError(
            f"action_chunk must have shape [plan_horizon, action_dim], got {tuple(action_chunk.shape)}."
        )
    return action_chunk.reshape(-1)


def unflatten_action_chunk(
    flat_action_chunk: torch.Tensor,
    plan_horizon: int,
    action_dim: int,
) -> torch.Tensor:
    """Unflatten an action chunk.

    flat_action_chunk: [plan_horizon * action_dim]
    returns: [plan_horizon, action_dim]
    """
    if not torch.is_tensor(flat_action_chunk):
        raise TypeError(f"Expected torch.Tensor, got {type(flat_action_chunk)}.")
    if flat_action_chunk.ndim != 1:
        raise ValueError(
            f"flat_action_chunk must have shape [plan_horizon * action_dim], got {tuple(flat_action_chunk.shape)}."
        )
    chunk_dim = infer_action_chunk_dim(plan_horizon, action_dim)
    if flat_action_chunk.numel() != chunk_dim:
        raise ValueError(
            "flat_action_chunk size does not match plan_horizon * action_dim: "
            f"{flat_action_chunk.numel()} != {chunk_dim}."
        )
    return flat_action_chunk.reshape(plan_horizon, action_dim)


def encode_current_goal(
    model: torch.nn.Module,
    info_dict: dict[str, Any],
    env_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode current observation and goal observation with the world model encoder.

    z_cur: [embed_dim]
    z_goal: [embed_dim]
    """
    if "pixels" not in info_dict:
        raise KeyError("'pixels' must be present in info_dict.")
    if "goal" not in info_dict:
        raise KeyError("'goal' must be present in info_dict.")

    encoded_current = _encode_pixels_only(
        model=model,
        pixels=info_dict["pixels"][env_index : env_index + 1],
    )
    encoded_goal = _encode_pixels_only(
        model=model,
        pixels=info_dict["goal"][env_index : env_index + 1],
    )

    z_cur = encoded_current["emb"][0, -1].detach().cpu().float()  # [embed_dim]
    z_goal = encoded_goal["emb"][0, -1].detach().cpu().float()  # [embed_dim]

    if z_cur.shape != z_goal.shape:
        raise ValueError(
            f"z_cur and z_goal must have the same shape, got {tuple(z_cur.shape)} and {tuple(z_goal.shape)}."
        )

    return z_cur, z_goal


def build_single_peak_teacher_sample(
    *,
    model: torch.nn.Module,
    solver: Any,
    plan_config: Any,
    info_dict: dict[str, Any],
    env_index: int = 0,
    init_action: torch.Tensor | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct one teacher sample for the single-peak planner baseline.

    info_dict["pixels"]: [num_envs, history, C, H, W]
    info_dict["goal"]: [num_envs, history, C, H, W]
    info_dict["action"]: [num_envs, history, action_dim]
    solver outputs["actions"]: [num_envs, horizon, action_block * action_dim]
    teacher_plan_2d: [receding_horizon * action_block, action_dim]
    teacher_plan: [receding_horizon * action_block * action_dim]
    """
    prepared_info = clone_info_dict(info_dict)

    z_cur, z_goal = encode_current_goal(model=model, info_dict=prepared_info, env_index=env_index)

    outputs = solver.solve(prepared_info, init_action=init_action)
    teacher_plan = extract_teacher_plan(
        solver_outputs=outputs,
        plan_config=plan_config,
        info_dict=prepared_info,
        env_index=env_index,
    )

    sample = {
        "z_cur": z_cur,
        "z_goal": z_goal,
        "teacher_plan": teacher_plan,
        "meta": {
            "env_index": int(env_index),
            "plan_horizon": int(plan_config.receding_horizon * plan_config.action_block),
            "action_dim": int(_infer_single_action_dim(prepared_info, env_index=env_index)),
            "action_chunk_dim": int(teacher_plan.numel()),
            **(meta or {}),
        },
    }
    return validate_single_peak_sample(sample)


def extract_teacher_plan(
    *,
    solver_outputs: dict[str, Any],
    plan_config: Any,
    info_dict: dict[str, Any],
    env_index: int = 0,
) -> torch.Tensor:
    """Extract the executed receding-horizon action chunk from solver outputs."""
    if "actions" not in solver_outputs:
        raise KeyError("solver_outputs must contain the key 'actions'.")

    actions = solver_outputs["actions"]
    if not torch.is_tensor(actions):
        actions = torch.as_tensor(actions)
    if actions.ndim != 3:
        raise ValueError(
            f"solver_outputs['actions'] must have shape [num_envs, horizon, action_block * action_dim], got {tuple(actions.shape)}."
        )

    if env_index < 0 or env_index >= actions.size(0):
        raise IndexError(f"env_index {env_index} is out of range for {actions.size(0)} environments.")

    receding_horizon = int(plan_config.receding_horizon)
    action_block = int(plan_config.action_block)
    action_dim = _infer_single_action_dim(info_dict, env_index=env_index)
    block_action_dim = infer_action_chunk_dim(action_block, action_dim)

    if actions.size(1) < receding_horizon:
        raise ValueError(
            f"solver horizon {actions.size(1)} is smaller than receding_horizon {receding_horizon}."
        )
    if actions.size(2) != block_action_dim:
        raise ValueError(
            "solver action dim does not match action_block * action_dim: "
            f"{actions.size(2)} != {block_action_dim}."
        )

    selected_blocks = actions[env_index, :receding_horizon].detach().cpu().float()  # [R, action_block * action_dim]
    teacher_plan_2d = selected_blocks.reshape(receding_horizon * action_block, action_dim)  # [plan_horizon, action_dim]
    teacher_plan = flatten_action_chunk(teacher_plan_2d)  # [plan_horizon * action_dim]

    # Sanity check the reversible chunk schema.
    recovered = unflatten_action_chunk(
        teacher_plan,
        plan_horizon=receding_horizon * action_block,
        action_dim=action_dim,
    )
    if not torch.equal(recovered, teacher_plan_2d):
        raise ValueError("teacher_plan flatten/unflatten changed the action semantics.")

    return teacher_plan


def extract_trajectory_action_chunk(
    *,
    info_dict: dict[str, Any],
    plan_horizon: int,
    env_index: int = 0,
) -> torch.Tensor:
    """Extract a flattened action chunk directly from trajectory actions.

    info_dict["action"]: [num_envs, context_horizon, action_dim]
    action_chunk_2d: [plan_horizon, action_dim]
    teacher_plan: [plan_horizon * action_dim]
    """
    if "action" not in info_dict:
        raise KeyError("info_dict must contain the key 'action'.")

    actions = info_dict["action"]
    if not torch.is_tensor(actions):
        actions = torch.as_tensor(actions)
    if actions.ndim != 3:
        raise ValueError(
            f"info_dict['action'] must have shape [num_envs, context_horizon, action_dim], got {tuple(actions.shape)}."
        )
    if env_index < 0 or env_index >= actions.size(0):
        raise IndexError(f"env_index {env_index} is out of range for {actions.size(0)} environments.")
    if actions.size(1) < plan_horizon:
        raise ValueError(
            f"Action context horizon {actions.size(1)} is smaller than requested plan_horizon {plan_horizon}."
        )

    action_chunk_2d = actions[env_index, :plan_horizon].detach().cpu().float()  # [plan_horizon, action_dim]
    teacher_plan = flatten_action_chunk(action_chunk_2d)  # [plan_horizon * action_dim]
    action_dim = int(action_chunk_2d.shape[-1])

    recovered = unflatten_action_chunk(
        teacher_plan,
        plan_horizon=plan_horizon,
        action_dim=action_dim,
    )
    if not torch.equal(recovered, action_chunk_2d):
        raise ValueError("trajectory action chunk flatten/unflatten changed the action semantics.")

    return teacher_plan


def validate_single_peak_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one sample dict."""
    required_keys = {"z_cur", "z_goal", "teacher_plan", "meta"}
    missing = required_keys.difference(sample.keys())
    if missing:
        raise KeyError(f"Sample is missing required keys: {sorted(missing)}.")

    z_cur = sample["z_cur"]
    z_goal = sample["z_goal"]
    teacher_plan = sample["teacher_plan"]
    meta = sample["meta"]

    if not torch.is_tensor(z_cur) or z_cur.ndim != 1:
        raise ValueError(f"z_cur must be a 1D tensor, got {type(z_cur)} with shape {getattr(z_cur, 'shape', None)}.")
    if not torch.is_tensor(z_goal) or z_goal.ndim != 1:
        raise ValueError(f"z_goal must be a 1D tensor, got {type(z_goal)} with shape {getattr(z_goal, 'shape', None)}.")
    if z_cur.shape != z_goal.shape:
        raise ValueError(f"z_cur and z_goal must match, got {tuple(z_cur.shape)} and {tuple(z_goal.shape)}.")
    if not torch.is_tensor(teacher_plan) or teacher_plan.ndim != 1:
        raise ValueError(
            "teacher_plan must be a 1D tensor representing a flattened action chunk, "
            f"got {type(teacher_plan)} with shape {getattr(teacher_plan, 'shape', None)}."
        )
    if not isinstance(meta, dict):
        raise ValueError(f"meta must be a dict, got {type(meta)}.")

    action_dim = meta.get("action_dim")
    plan_horizon = meta.get("plan_horizon")
    if action_dim is not None and plan_horizon is not None:
        expected_dim = infer_action_chunk_dim(int(plan_horizon), int(action_dim))
        if teacher_plan.numel() != expected_dim:
            raise ValueError(
                f"teacher_plan dim {teacher_plan.numel()} does not match plan_horizon * action_dim = {expected_dim}."
            )

    return {
        "z_cur": z_cur.detach().cpu().float(),
        "z_goal": z_goal.detach().cpu().float(),
        "teacher_plan": teacher_plan.detach().cpu().float(),
        "meta": deepcopy(meta),
    }


def clone_info_dict(info_dict: dict[str, Any]) -> dict[str, Any]:
    """Clone an info dict without sharing tensor storage."""
    cloned: dict[str, Any] = {}
    for key, value in info_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().clone()
        else:
            cloned[key] = deepcopy(value)
    return cloned


def _encode_pixels_only(
    model: torch.nn.Module,
    pixels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Call JEPA.encode with the real signature used in this repository.

    pixels: [1, history, C, H, W]
    emb: [1, history, embed_dim]
    """
    if not torch.is_tensor(pixels):
        raise TypeError(f"pixels must be a torch.Tensor, got {type(pixels)}.")
    if pixels.ndim != 5:
        raise ValueError(
            f"pixels must have shape [batch, history, C, H, W], got {tuple(pixels.shape)}."
        )

    device = next(model.parameters()).device
    encode_info = {"pixels": pixels.to(device)}
    with torch.inference_mode():
        encoded = model.encode(encode_info)
    if "emb" not in encoded:
        raise KeyError("model.encode(...) must return a dict containing 'emb'.")
    return encoded


def _infer_single_action_dim(info_dict: dict[str, Any], env_index: int) -> int:
    """Infer single-step action dim from current info dict."""
    if "action" not in info_dict:
        raise KeyError("'action' must be present in info_dict to infer action_dim.")
    action = info_dict["action"]
    if not torch.is_tensor(action):
        action = torch.as_tensor(action)
    if action.ndim != 3:
        raise ValueError(
            f"info_dict['action'] must have shape [num_envs, history, action_dim], got {tuple(action.shape)}."
        )
    if env_index < 0 or env_index >= action.size(0):
        raise IndexError(f"env_index {env_index} is out of range for {action.size(0)} environments.")
    action_dim = int(action[env_index].shape[-1])
    if action_dim <= 0:
        raise ValueError(f"Inferred invalid action_dim={action_dim}.")
    return action_dim
