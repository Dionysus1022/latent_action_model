from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _pool_latent_tokens(z: torch.Tensor) -> torch.Tensor:
    if z.ndim == 2:
        return z
    if z.ndim == 3:
        return z.mean(dim=1)
    raise ValueError(
        "Latents must have shape [B, D] or [B, N, D], "
        f"got {tuple(z.shape)}."
    )


def compute_prediction_error(
    z_real: torch.Tensor,
    z_pred: torch.Tensor,
    metric: str = "l2",
) -> torch.Tensor:
    """Compute detached per-sample latent prediction error.

    Returns:
        Tensor with shape [B]. Gradients never flow to encoder/predictor outputs.
    """
    if not torch.is_tensor(z_real) or not torch.is_tensor(z_pred):
        raise TypeError("z_real and z_pred must be torch.Tensor values.")

    with torch.no_grad():
        real = _pool_latent_tokens(z_real.detach())
        pred = _pool_latent_tokens(z_pred.detach())
        if real.shape != pred.shape:
            raise ValueError(
                "z_real and z_pred must have matching pooled shapes, "
                f"got {tuple(real.shape)} and {tuple(pred.shape)}."
            )

        metric_name = str(metric).lower().strip()
        if metric_name == "l2":
            return torch.linalg.vector_norm(real - pred, ord=2, dim=-1)
        if metric_name == "mse":
            return (real - pred).square().mean(dim=-1)
        if metric_name == "cosine":
            return 1.0 - F.cosine_similarity(real, pred, dim=-1, eps=1e-8)

    raise ValueError(f"Unsupported prediction error metric '{metric}'.")


def compute_trigger_error(
    errors: torch.Tensor,
    *,
    stat: str = "max",
    quantile: float = 0.9,
) -> float:
    """Reduce per-env prediction errors to one global replan trigger value."""
    if not torch.is_tensor(errors):
        raise TypeError("errors must be a torch.Tensor value.")

    with torch.no_grad():
        values = errors.detach().reshape(-1).float()
        if int(values.numel()) == 0:
            return float("nan")

        stat_name = str(stat).lower().strip()
        if stat_name == "max":
            return float(torch.max(values).item())
        if stat_name == "mean":
            return float(torch.mean(values).item())
        if stat_name == "quantile":
            q = float(quantile)
            if q < 0.0 or q > 1.0:
                raise ValueError(f"trigger_quantile must be in [0, 1], got {q}.")
            return float(torch.quantile(values, q, interpolation="lower").item())

    raise ValueError(
        "Unsupported trigger stat "
        f"'{stat}'. Expected one of: max, mean, quantile."
    )


def _as_float_array(values: Sequence[float]) -> np.ndarray:
    if len(values) == 0:
        return np.asarray([], dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


def _mean_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _max_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.max(values))


def _std_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.std(values))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    if a.size + b.size < 3:
        return float("nan")
    var_a = float(np.var(a, ddof=1)) if a.size > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if b.size > 1 else 0.0
    pooled_denom = max(a.size + b.size - 2, 1)
    pooled_std = math.sqrt(((a.size - 1) * var_a + (b.size - 1) * var_b) / pooled_denom)
    if pooled_std <= 1e-12:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def resolve_prediction_error_check(
    *,
    prefix_steps: int,
    action_block: int,
    correction_interval: int,
) -> int | None:
    """Return rollout block count when a prefix is ready for latent-error logging.

    LeWM latent rollout predicts one latent per action block. If the requested
    correction interval is not aligned to action_block, logging is delayed to
    the next available block boundary.
    """
    prefix_steps = int(prefix_steps)
    action_block = int(action_block)
    correction_interval = int(correction_interval)
    if prefix_steps <= 0:
        return None
    if action_block <= 0:
        raise ValueError(f"action_block must be positive, got {action_block}.")
    if correction_interval <= 0:
        raise ValueError(
            f"correction_interval must be positive, got {correction_interval}."
        )
    if prefix_steps < correction_interval:
        return None
    if prefix_steps % action_block != 0:
        return None
    return prefix_steps // action_block


def summarize_prediction_error_records(
    records: Sequence[dict[str, Any]],
    episode_successes: Sequence[bool] | np.ndarray | torch.Tensor,
) -> dict[str, float | int]:
    """Summarize prediction-error records overall and by episode outcome.

    Success/failure comparisons use each episode's mean prediction error, not
    every raw checkpoint. This avoids overweighting episodes that emitted more
    checkpoint records.
    """
    successes = np.asarray(episode_successes, dtype=bool).reshape(-1)
    all_errors: list[float] = []
    episode_errors: dict[int, list[float]] = {}

    for record in records:
        env_index = int(record["env_index"])
        if env_index < 0 or env_index >= int(successes.shape[0]):
            continue
        value = float(record["error"])
        if not math.isfinite(value):
            continue
        all_errors.append(value)
        episode_errors.setdefault(env_index, []).append(value)

    all_arr = _as_float_array(all_errors)
    success_episode_means: list[float] = []
    failure_episode_means: list[float] = []
    for env_index, values in episode_errors.items():
        episode_mean = float(np.mean(values))
        if bool(successes[env_index]):
            success_episode_means.append(episode_mean)
        else:
            failure_episode_means.append(episode_mean)

    success_arr = _as_float_array(success_episode_means)
    failure_arr = _as_float_array(failure_episode_means)
    success_mean = _mean_or_nan(success_arr)
    failure_mean = _mean_or_nan(failure_arr)

    if math.isfinite(success_mean) and math.isfinite(failure_mean):
        diff = float(failure_mean - success_mean)
        ratio = float(failure_mean / max(success_mean, 1e-12))
        failed_higher = float(failure_mean > success_mean)
    else:
        diff = float("nan")
        ratio = float("nan")
        failed_higher = float("nan")

    return {
        "prediction_error_count": int(all_arr.size),
        "prediction_error_mean": _mean_or_nan(all_arr),
        "prediction_error_max": _max_or_nan(all_arr),
        "prediction_error_std": _std_or_nan(all_arr),
        "prediction_error_episode_mean_count": int(success_arr.size + failure_arr.size),
        "successful_episode_count": int(np.sum(successes)),
        "failed_episode_count": int(successes.size - np.sum(successes)),
        "successful_prediction_error_count": int(success_arr.size),
        "failed_prediction_error_count": int(failure_arr.size),
        "successful_prediction_error_mean": success_mean,
        "failed_prediction_error_mean": failure_mean,
        "successful_prediction_error_max": _max_or_nan(success_arr),
        "failed_prediction_error_max": _max_or_nan(failure_arr),
        "prediction_error_fail_minus_success": diff,
        "prediction_error_fail_success_ratio": ratio,
        "prediction_error_failed_higher": failed_higher,
        "prediction_error_cohens_d_fail_vs_success": _cohens_d(failure_arr, success_arr),
    }


__all__ = [
    "compute_prediction_error",
    "compute_trigger_error",
    "resolve_prediction_error_check",
    "summarize_prediction_error_records",
]
