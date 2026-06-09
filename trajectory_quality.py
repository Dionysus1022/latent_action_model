from __future__ import annotations

import math
from typing import Any

import numpy as np


def _as_float_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 3:
        raise ValueError(f"{name} must have shape [episodes, steps, dim], got {array.shape}.")
    return array


def _last_history_frame(array: Any) -> np.ndarray:
    data = np.asarray(array)
    if data.ndim >= 4 and int(data.shape[2]) == 1:
        return data[:, :, -1, ...]
    return data


def _l2(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=-1)


def _nanmean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    if finite.size == 0 or np.all(np.isnan(finite)):
        return float("nan")
    return float(np.nanmean(finite))


def _nanmax(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    if finite.size == 0 or np.all(np.isnan(finite)):
        return float("nan")
    return float(np.nanmax(finite))


def _angle_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = np.abs(a - b)
    return np.minimum(diff, 2.0 * math.pi - diff)


def _require_state_pair(
    state_trace: dict[str, Any],
    current_key: str,
    goal_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    if current_key not in state_trace:
        raise KeyError(f"Missing trajectory state key '{current_key}'.")
    if goal_key not in state_trace:
        raise KeyError(f"Missing trajectory goal key '{goal_key}'.")
    current = _last_history_frame(state_trace[current_key]).astype(np.float64)
    goal = _last_history_frame(state_trace[goal_key]).astype(np.float64)
    if current.shape[:2] != goal.shape[:2]:
        raise ValueError(
            f"State/goal leading dimensions for {current_key}/{goal_key} do not match: "
            f"{current.shape[:2]} != {goal.shape[:2]}."
        )
    return current, goal


def compute_task_goal_distances(
    task: str,
    state_trace: dict[str, Any],
) -> dict[str, np.ndarray]:
    task_name = str(task).lower().strip()

    if task_name in {"reacher", "researcher"}:
        qpos, goal_qpos = _require_state_pair(state_trace, "qpos", "goal_qpos")
        return {
            "goal_distance": _l2(qpos - goal_qpos),
            "state_for_path": qpos,
        }

    if task_name in {"tworoom", "two_room", "two-room"}:
        proprio, goal_proprio = _require_state_pair(state_trace, "proprio", "goal_proprio")
        return {
            "goal_distance": _l2(proprio[..., :2] - goal_proprio[..., :2]),
            "state_for_path": proprio[..., :2],
        }

    if task_name == "pusht":
        state, goal_state = _require_state_pair(state_trace, "state", "goal_state")
        pos_distance = _l2(state[..., :4] - goal_state[..., :4])
        angle_distance = _angle_distance(state[..., 4], goal_state[..., 4])
        return {
            "goal_distance": _l2(state - goal_state),
            "pusht_pos_distance": pos_distance,
            "pusht_angle_distance": angle_distance,
            "state_for_path": state[..., :5],
        }

    if task_name == "cube":
        block_pos, goal_block_pos = _require_state_pair(
            state_trace,
            "privileged/block_0_pos",
            "goal_privileged_block_0_pos",
        )
        output = {
            "goal_distance": _l2(block_pos - goal_block_pos),
            "cube_block_0_pos_distance": _l2(block_pos - goal_block_pos),
            "state_for_path": block_pos,
        }
        if (
            "privileged/block_0_yaw" in state_trace
            and "goal_privileged_block_0_yaw" in state_trace
        ):
            yaw = _last_history_frame(state_trace["privileged/block_0_yaw"]).astype(np.float64)
            goal_yaw = _last_history_frame(state_trace["goal_privileged_block_0_yaw"]).astype(np.float64)
            output["cube_block_0_yaw_distance"] = _angle_distance(
                np.squeeze(yaw, axis=-1) if yaw.shape[-1:] == (1,) else yaw,
                np.squeeze(goal_yaw, axis=-1) if goal_yaw.shape[-1:] == (1,) else goal_yaw,
            )
        return output

    raise ValueError(f"Unsupported trajectory-quality task '{task}'.")


def compute_trajectory_quality(
    *,
    states: Any,
    actions: Any,
    goal_distances: Any,
    successes_by_step: Any | None = None,
    truncate_after_success: bool = True,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    state_array = _as_float_array(states, name="states")
    action_array = _as_float_array(actions, name="actions")
    distance_array = np.asarray(goal_distances, dtype=np.float64)
    if distance_array.ndim != 2:
        raise ValueError(f"goal_distances must have shape [episodes, steps], got {distance_array.shape}.")
    if state_array.shape[:2] != distance_array.shape:
        raise ValueError(
            f"states and goal_distances leading dimensions do not match: "
            f"{state_array.shape[:2]} != {distance_array.shape}."
        )
    if action_array.shape[:2] != distance_array.shape:
        raise ValueError(
            f"actions and goal_distances leading dimensions do not match: "
            f"{action_array.shape[:2]} != {distance_array.shape}."
        )

    successes = None
    if successes_by_step is not None:
        successes = np.asarray(successes_by_step, dtype=bool)
        if successes.shape != distance_array.shape:
            raise ValueError(
                f"successes_by_step shape {successes.shape} does not match goal_distances {distance_array.shape}."
            )

    episode_count, step_count = distance_array.shape
    per_episode: dict[str, list[float]] = {
        "effective_length": [],
        "path_length": [],
        "straight_line_distance": [],
        "straight_line_ratio": [],
        "final_goal_distance": [],
        "min_goal_distance": [],
        "steps_to_success": [],
        "action_l2_mean": [],
        "action_l2_max": [],
        "action_delta_l2_mean": [],
        "action_delta_l2_max": [],
        "action_jerk_l2_mean": [],
        "action_jerk_l2_max": [],
    }

    for episode_index in range(episode_count):
        end_exclusive = step_count
        first_success = None
        if successes is not None and np.any(successes[episode_index]):
            first_success = int(np.argmax(successes[episode_index]))
            if truncate_after_success:
                end_exclusive = first_success + 1

        state_episode = state_array[episode_index, :end_exclusive]
        action_episode = action_array[episode_index, :end_exclusive]
        distance_episode = distance_array[episode_index, :end_exclusive]
        valid_state_mask = np.isfinite(state_episode).all(axis=-1)
        valid_action_mask = np.isfinite(action_episode).all(axis=-1)
        valid_distance_mask = np.isfinite(distance_episode)
        valid_mask = valid_state_mask & valid_action_mask & valid_distance_mask

        state_episode = state_episode[valid_mask]
        action_episode = action_episode[valid_mask]
        distance_episode = distance_episode[valid_mask]

        effective_length = int(len(distance_episode))
        per_episode["effective_length"].append(float(effective_length))
        if effective_length == 0:
            for key in per_episode:
                if key != "effective_length":
                    per_episode[key].append(float("nan"))
            continue

        if effective_length >= 2:
            deltas = np.diff(state_episode, axis=0)
            path_length = float(np.sum(_l2(deltas)))
        else:
            path_length = 0.0
        straight_line_distance = float(np.linalg.norm(state_episode[-1] - state_episode[0]))
        if straight_line_distance > 1e-12:
            straight_line_ratio = path_length / straight_line_distance
        else:
            straight_line_ratio = float("nan")

        action_l2 = _l2(action_episode)
        action_delta = np.diff(action_episode, axis=0)
        action_delta_l2 = _l2(action_delta) if len(action_delta) > 0 else np.array([], dtype=np.float64)
        action_jerk = np.diff(action_episode, n=2, axis=0)
        action_jerk_l2 = _l2(action_jerk) if len(action_jerk) > 0 else np.array([], dtype=np.float64)

        per_episode["path_length"].append(path_length)
        per_episode["straight_line_distance"].append(straight_line_distance)
        per_episode["straight_line_ratio"].append(straight_line_ratio)
        per_episode["final_goal_distance"].append(float(distance_episode[-1]))
        per_episode["min_goal_distance"].append(float(np.min(distance_episode)))
        per_episode["steps_to_success"].append(float("nan") if first_success is None else float(first_success))
        per_episode["action_l2_mean"].append(_nanmean(action_l2))
        per_episode["action_l2_max"].append(_nanmax(action_l2))
        per_episode["action_delta_l2_mean"].append(_nanmean(action_delta_l2))
        per_episode["action_delta_l2_max"].append(_nanmax(action_delta_l2))
        per_episode["action_jerk_l2_mean"].append(_nanmean(action_jerk_l2))
        per_episode["action_jerk_l2_max"].append(_nanmax(action_jerk_l2))

    per_episode_arrays = {
        key: np.asarray(values, dtype=np.float64)
        for key, values in per_episode.items()
    }
    summary = {
        f"{key}_mean": _nanmean(values)
        for key, values in per_episode_arrays.items()
        if key != "effective_length"
    }
    summary.update(
        {
            f"{key}_max": _nanmax(values)
            for key, values in per_episode_arrays.items()
            if key
            in {
                "path_length",
                "straight_line_ratio",
                "final_goal_distance",
                "min_goal_distance",
                "action_l2_max",
                "action_delta_l2_max",
                "action_jerk_l2_max",
            }
        }
    )
    summary["effective_length_mean"] = _nanmean(per_episode_arrays["effective_length"])
    return summary, per_episode_arrays


def compute_latent_monotonicity(
    *,
    latents: Any,
    goal_latents: Any,
    successes_by_step: Any | None = None,
    truncate_after_success: bool = True,
    tolerance: float = 1e-9,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Compute episode-level latent monotonicity toward the goal.

    The LGBS paper reports monotonicity as the fraction of episodes for which
    ||z_t - z_goal|| decreases at every step. We also return the per-episode
    step fraction as a diagnostic, but ``latent_monotonicity_mean`` follows the
    paper's all-steps episode-level definition.
    """
    latent_array = _as_float_array(latents, name="latents")
    goal_array = np.asarray(goal_latents, dtype=np.float64)
    if goal_array.ndim == 2:
        goal_array = np.broadcast_to(
            goal_array[:, None, :],
            (latent_array.shape[0], latent_array.shape[1], goal_array.shape[-1]),
        )
    elif goal_array.ndim < 3:
        raise ValueError(f"goal_latents must have shape [episodes, dim] or [episodes, steps, dim], got {goal_array.shape}.")

    if latent_array.shape != goal_array.shape:
        raise ValueError(
            f"latents and goal_latents must have matching shape after broadcast: "
            f"{latent_array.shape} != {goal_array.shape}."
        )

    successes = None
    if successes_by_step is not None:
        successes = np.asarray(successes_by_step, dtype=bool)
        if successes.shape != latent_array.shape[:2]:
            raise ValueError(
                f"successes_by_step shape {successes.shape} does not match latents {latent_array.shape[:2]}."
            )

    monotonicity = []
    monotonic_step_fraction = []
    final_latent_goal_distance = []
    min_latent_goal_distance = []
    for episode_index in range(latent_array.shape[0]):
        end_exclusive = latent_array.shape[1]
        if successes is not None and np.any(successes[episode_index]) and truncate_after_success:
            end_exclusive = int(np.argmax(successes[episode_index])) + 1

        z_episode = latent_array[episode_index, :end_exclusive]
        g_episode = goal_array[episode_index, :end_exclusive]
        valid = np.isfinite(z_episode).all(axis=-1) & np.isfinite(g_episode).all(axis=-1)
        z_episode = z_episode[valid]
        g_episode = g_episode[valid]
        if z_episode.shape[0] == 0:
            monotonicity.append(float("nan"))
            final_latent_goal_distance.append(float("nan"))
            min_latent_goal_distance.append(float("nan"))
            continue

        distances = _l2(z_episode - g_episode)
        final_latent_goal_distance.append(float(distances[-1]))
        min_latent_goal_distance.append(float(np.min(distances)))
        if distances.shape[0] < 2:
            monotonicity.append(float("nan"))
            monotonic_step_fraction.append(float("nan"))
        else:
            monotonic_steps = distances[1:] <= distances[:-1] + float(tolerance)
            monotonicity.append(float(np.all(monotonic_steps)))
            monotonic_step_fraction.append(float(np.mean(monotonic_steps)))

    per_episode = {
        "latent_monotonicity": np.asarray(monotonicity, dtype=np.float64),
        "latent_monotonic_step_fraction": np.asarray(monotonic_step_fraction, dtype=np.float64),
        "final_latent_goal_distance": np.asarray(final_latent_goal_distance, dtype=np.float64),
        "min_latent_goal_distance": np.asarray(min_latent_goal_distance, dtype=np.float64),
    }
    summary = {
        "latent_monotonicity_mean": _nanmean(per_episode["latent_monotonicity"]),
        "latent_monotonicity_std": float(np.nanstd(per_episode["latent_monotonicity"])),
        "latent_monotonic_step_fraction_mean": _nanmean(per_episode["latent_monotonic_step_fraction"]),
        "final_latent_goal_distance_mean": _nanmean(per_episode["final_latent_goal_distance"]),
        "min_latent_goal_distance_mean": _nanmean(per_episode["min_latent_goal_distance"]),
    }
    return summary, per_episode


__all__ = [
    "compute_latent_monotonicity",
    "compute_task_goal_distances",
    "compute_trajectory_quality",
]
