from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from sklearn.cluster import KMeans

from planners.single_peak_data import (
    flatten_action_chunk as _flatten_action_chunk,
    infer_action_chunk_dim,
    unflatten_action_chunk as _unflatten_action_chunk,
)


@dataclass
class ActionAnchorBundle:
    """Portable bundle for action anchors.

    anchors: [K, action_chunk_dim]
    """

    anchors: torch.Tensor
    plan_horizon: int
    action_dim: int
    action_chunk_dim: int
    num_anchors: int
    action_chunk_horizon: int | None = None
    receding_horizon: int | None = None
    action_block: int | None = None
    task: str | None = None
    dataset_path: str | None = None
    max_samples: int | None = None
    fit_method: str = "kmeans"
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    bundle_version: int = 2

    def __post_init__(self) -> None:
        self.plan_horizon = int(self.plan_horizon)
        self.action_dim = int(self.action_dim)
        self.action_chunk_dim = int(self.action_chunk_dim)
        self.num_anchors = int(self.num_anchors)
        self.action_chunk_horizon = int(self.action_chunk_horizon or self.plan_horizon)
        if self.action_chunk_horizon != self.plan_horizon:
            raise ValueError(
                "ActionAnchorBundle requires action_chunk_horizon == plan_horizon, got "
                f"{self.action_chunk_horizon} and {self.plan_horizon}."
            )
        if self.receding_horizon is not None:
            self.receding_horizon = int(self.receding_horizon)
        if self.action_block is not None:
            self.action_block = int(self.action_block)
        if self.max_samples is not None:
            self.max_samples = int(self.max_samples)
        if self.task is not None:
            self.task = str(self.task)
        if self.dataset_path is not None:
            self.dataset_path = str(self.dataset_path)

    def as_dict(self) -> dict[str, Any]:
        return {
            "bundle_version": int(self.bundle_version),
            "anchors": self.anchors,
            "plan_horizon": int(self.plan_horizon),
            "action_dim": int(self.action_dim),
            "action_chunk_dim": int(self.action_chunk_dim),
            "num_anchors": int(self.num_anchors),
            "action_chunk_horizon": int(self.action_chunk_horizon),
            "receding_horizon": None if self.receding_horizon is None else int(self.receding_horizon),
            "action_block": None if self.action_block is None else int(self.action_block),
            "task": None if self.task is None else str(self.task),
            "dataset_path": None if self.dataset_path is None else str(self.dataset_path),
            "max_samples": None if self.max_samples is None else int(self.max_samples),
            "fit_method": str(self.fit_method),
            "seed": None if self.seed is None else int(self.seed),
            "metadata": deepcopy(self.metadata),
        }


def flatten_action_chunk(action_chunk: torch.Tensor) -> torch.Tensor:
    """Flatten an action chunk.

    action_chunk: [plan_horizon, action_dim]
    returns: [plan_horizon * action_dim]
    """
    return _flatten_action_chunk(action_chunk)


def unflatten_action_chunk(
    flat_action_chunk: torch.Tensor,
    plan_horizon: int,
    action_dim: int,
) -> torch.Tensor:
    """Unflatten an action chunk.

    flat_action_chunk: [plan_horizon * action_dim]
    returns: [plan_horizon, action_dim]
    """
    return _unflatten_action_chunk(flat_action_chunk, plan_horizon=plan_horizon, action_dim=action_dim)


def coerce_action_chunk_batch(
    action_chunks: torch.Tensor | Iterable[torch.Tensor | Any],
    *,
    plan_horizon: int,
    action_dim: int,
    name: str,
) -> torch.Tensor:
    """Normalize action chunk samples into a flat batch tensor.

    Accepted input shapes:
        [action_chunk_dim]
        [N, action_chunk_dim]
        [plan_horizon, action_dim]
        [N, plan_horizon, action_dim]

    Returns:
        tensor: [N, action_chunk_dim]
    """
    action_chunk_dim = infer_action_chunk_dim(plan_horizon, action_dim)

    if torch.is_tensor(action_chunks):
        tensor = action_chunks.detach()
        if tensor.ndim == 1:
            if tensor.numel() != action_chunk_dim:
                raise ValueError(
                    f"{name} must have shape [action_chunk_dim], got {tuple(tensor.shape)} "
                    f"for action_chunk_dim={action_chunk_dim}."
                )
            return tensor.unsqueeze(0).cpu().float()  # [1, action_chunk_dim]
        if tensor.ndim == 2:
            if tuple(tensor.shape) == (plan_horizon, action_dim):
                return flatten_action_chunk(tensor).unsqueeze(0).cpu().float()  # [1, action_chunk_dim]
            if int(tensor.shape[-1]) != action_chunk_dim:
                raise ValueError(
                    f"{name} must have shape [N, action_chunk_dim], got {tuple(tensor.shape)} "
                    f"for action_chunk_dim={action_chunk_dim}."
                )
            return tensor.cpu().float()  # [N, action_chunk_dim]
        if tensor.ndim == 3:
            expected_shape = (plan_horizon, action_dim)
            if tuple(tensor.shape[-2:]) != expected_shape:
                raise ValueError(
                    f"{name} must have trailing shape {expected_shape}, got {tuple(tensor.shape)}."
                )
            return torch.stack(
                [flatten_action_chunk(sample) for sample in tensor],
                dim=0,
            ).cpu().float()  # [N, action_chunk_dim]
        raise ValueError(
            f"{name} must have shape [action_chunk_dim], [N, action_chunk_dim], "
            f"[plan_horizon, action_dim], or [N, plan_horizon, action_dim], got {tuple(tensor.shape)}."
        )

    rows: list[torch.Tensor] = []
    for index, sample in enumerate(action_chunks):
        sample_tensor = torch.as_tensor(sample)
        if sample_tensor.ndim == 1:
            if sample_tensor.numel() != action_chunk_dim:
                raise ValueError(
                    f"{name}[{index}] must have numel={action_chunk_dim}, got {sample_tensor.numel()}."
                )
            rows.append(sample_tensor.cpu().float())  # [action_chunk_dim]
            continue
        if sample_tensor.ndim == 2:
            expected_shape = (plan_horizon, action_dim)
            if tuple(sample_tensor.shape) != expected_shape:
                raise ValueError(
                    f"{name}[{index}] must have shape {expected_shape}, got {tuple(sample_tensor.shape)}."
                )
            rows.append(flatten_action_chunk(sample_tensor).cpu().float())  # [action_chunk_dim]
            continue
        raise ValueError(
            f"{name}[{index}] must have shape [action_chunk_dim] or [plan_horizon, action_dim], "
            f"got {tuple(sample_tensor.shape)}."
        )

    if len(rows) == 0:
        raise ValueError(f"{name} must contain at least one action chunk sample.")
    return torch.stack(rows, dim=0)  # [N, action_chunk_dim]


def validate_anchor_tensor(
    anchors: torch.Tensor | Iterable[torch.Tensor | Any],
    *,
    plan_horizon: int,
    action_dim: int,
    num_anchors: int | None = None,
) -> torch.Tensor:
    """Validate and normalize anchor tensors.

    anchors:
        [K, action_chunk_dim]
        [K, plan_horizon, action_dim]

    returns:
        validated_anchors: [K, action_chunk_dim]
    """
    validated = coerce_action_chunk_batch(
        anchors,
        plan_horizon=plan_horizon,
        action_dim=action_dim,
        name="anchors",
    )  # [K, action_chunk_dim]
    if num_anchors is not None and int(validated.shape[0]) != int(num_anchors):
        raise ValueError(
            f"Expected {num_anchors} anchors, got {validated.shape[0]}."
        )
    expected_dim = infer_action_chunk_dim(plan_horizon, action_dim)
    if int(validated.shape[-1]) != expected_dim:
        raise ValueError(
            f"Anchor width {validated.shape[-1]} does not match plan_horizon * action_dim = {expected_dim}."
        )
    return validated.cpu().float()


def build_action_anchor_bundle(
    anchors: torch.Tensor | Iterable[torch.Tensor | Any],
    *,
    plan_horizon: int,
    action_dim: int,
    receding_horizon: int | None = None,
    action_block: int | None = None,
    task: str | None = None,
    dataset_path: str | None = None,
    max_samples: int | None = None,
    fit_method: str = "manual",
    seed: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> ActionAnchorBundle:
    """Package validated anchors into a portable bundle."""
    validated_anchors = validate_anchor_tensor(
        anchors,
        plan_horizon=plan_horizon,
        action_dim=action_dim,
    )  # [K, action_chunk_dim]
    return ActionAnchorBundle(
        anchors=validated_anchors,
        plan_horizon=int(plan_horizon),
        action_dim=int(action_dim),
        action_chunk_dim=int(validated_anchors.shape[-1]),
        num_anchors=int(validated_anchors.shape[0]),
        action_chunk_horizon=int(plan_horizon),
        receding_horizon=None if receding_horizon is None else int(receding_horizon),
        action_block=None if action_block is None else int(action_block),
        task=None if task is None else str(task),
        dataset_path=None if dataset_path is None else str(dataset_path),
        max_samples=None if max_samples is None else int(max_samples),
        fit_method=str(fit_method),
        seed=None if seed is None else int(seed),
        metadata=deepcopy(metadata or {}),
    )


def fit_action_anchors(
    teacher_plan: torch.Tensor | Iterable[torch.Tensor | Any],
    *,
    num_anchors: int,
    plan_horizon: int,
    action_dim: int,
    receding_horizon: int | None = None,
    action_block: int | None = None,
    task: str | None = None,
    dataset_path: str | None = None,
    max_samples: int | None = None,
    seed: int = 42,
    max_iter: int = 300,
    metadata: dict[str, Any] | None = None,
) -> ActionAnchorBundle:
    """Fit action anchors from teacher action chunks using K-means.

    teacher_plan:
        [N, action_chunk_dim]
        [N, plan_horizon, action_dim]
        iterable of [action_chunk_dim] or [plan_horizon, action_dim]

    returns:
        bundle.anchors: [K, action_chunk_dim]
    """
    if num_anchors <= 0:
        raise ValueError(f"num_anchors must be positive, got {num_anchors}.")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}.")

    teacher_plan_tensor = coerce_action_chunk_batch(
        teacher_plan,
        plan_horizon=plan_horizon,
        action_dim=action_dim,
        name="teacher_plan",
    )  # [N, action_chunk_dim]
    num_samples = int(teacher_plan_tensor.shape[0])
    if num_samples < num_anchors:
        raise ValueError(
            f"Cannot fit {num_anchors} anchors from only {num_samples} teacher_plan samples."
        )

    kmeans = KMeans(
        n_clusters=int(num_anchors),
        random_state=int(seed),
        n_init=10,
        max_iter=int(max_iter),
    )
    kmeans.fit(teacher_plan_tensor.numpy())
    anchors = torch.from_numpy(kmeans.cluster_centers_).float()  # [K, action_chunk_dim]
    anchors = validate_anchor_tensor(
        anchors,
        plan_horizon=plan_horizon,
        action_dim=action_dim,
        num_anchors=num_anchors,
    )  # [K, action_chunk_dim]

    bundle_metadata = deepcopy(metadata or {})
    bundle_metadata.update(
        {
            "num_teacher_plans": num_samples,
            "kmeans_inertia": float(kmeans.inertia_),
            "kmeans_iterations": int(kmeans.n_iter_),
        }
    )
    return build_action_anchor_bundle(
        anchors,
        plan_horizon=plan_horizon,
        action_dim=action_dim,
        receding_horizon=receding_horizon,
        action_block=action_block,
        task=task,
        dataset_path=dataset_path,
        max_samples=max_samples,
        fit_method="kmeans",
        seed=seed,
        metadata=bundle_metadata,
    )


def save_anchor_bundle(
    bundle: ActionAnchorBundle,
    path: str | Path,
) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle.as_dict(), output_path)
    return output_path


def load_anchor_bundle(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> ActionAnchorBundle:
    bundle_dict = torch.load(Path(path).expanduser().resolve(), map_location=map_location)
    bundle_dict = validate_anchor_bundle_dict(bundle_dict)
    bundle = ActionAnchorBundle(**bundle_dict)
    bundle.anchors = validate_anchor_tensor(
        bundle.anchors,
        plan_horizon=bundle.plan_horizon,
        action_dim=bundle.action_dim,
        num_anchors=bundle.num_anchors,
    )  # [K, action_chunk_dim]
    return bundle


def _maybe_get_nested(source: dict[str, Any] | None, path: list[str]) -> Any:
    current: Any = source
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _maybe_positive_int(value: Any, *, field_name: str) -> int | None:
    if value in [None, "", "null"]:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"bundle['{field_name}'] must be positive when present, got {parsed}.")
    return parsed


def _normalize_anchor_bundle_dict(bundle_dict: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(bundle_dict)
    metadata = normalized.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    normalized["metadata"] = metadata

    action_chunk_horizon = normalized.get("action_chunk_horizon", normalized.get("plan_horizon"))
    normalized["action_chunk_horizon"] = _maybe_positive_int(
        action_chunk_horizon,
        field_name="action_chunk_horizon",
    )

    receding_horizon = normalized.get("receding_horizon")
    if receding_horizon is None:
        receding_horizon = _maybe_get_nested(metadata, ["source_build_info", "plan_config", "receding_horizon"])
    if receding_horizon is None:
        receding_horizon = _maybe_get_nested(metadata, ["source_build_info", "task_spec", "receding_horizon"])
    normalized["receding_horizon"] = _maybe_positive_int(
        receding_horizon,
        field_name="receding_horizon",
    )

    action_block = normalized.get("action_block")
    if action_block is None:
        action_block = _maybe_get_nested(metadata, ["source_build_info", "plan_config", "action_block"])
    if action_block is None:
        action_block = _maybe_get_nested(metadata, ["source_build_info", "task_spec", "action_block"])
    normalized["action_block"] = _maybe_positive_int(
        action_block,
        field_name="action_block",
    )

    if normalized["receding_horizon"] is None and normalized["action_block"] is not None:
        horizon = normalized["action_chunk_horizon"]
        if horizon is not None and horizon % normalized["action_block"] == 0:
            normalized["receding_horizon"] = int(horizon // normalized["action_block"])
    if normalized["action_block"] is None and normalized["receding_horizon"] is not None:
        horizon = normalized["action_chunk_horizon"]
        if horizon is not None and horizon % normalized["receding_horizon"] == 0:
            normalized["action_block"] = int(horizon // normalized["receding_horizon"])

    task = normalized.get("task")
    if task in [None, "", "null"]:
        task = metadata.get("task")
    if task in [None, "", "null"]:
        task = _maybe_get_nested(metadata, ["source_build_info", "task"])
    normalized["task"] = None if task in [None, "", "null"] else str(task)

    dataset_path = normalized.get("dataset_path")
    if dataset_path in [None, "", "null"]:
        dataset_path = metadata.get("dataset_path")
    if dataset_path in [None, "", "null"]:
        dataset_path = metadata.get("source_dataset")
    normalized["dataset_path"] = None if dataset_path in [None, "", "null"] else str(dataset_path)

    max_samples = normalized.get("max_samples")
    if max_samples is None:
        max_samples = metadata.get("max_samples")
    normalized["max_samples"] = _maybe_positive_int(max_samples, field_name="max_samples")
    return normalized


def validate_anchor_bundle_dict(bundle_dict: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_anchor_bundle_dict(bundle_dict)
    required_keys = {
        "bundle_version",
        "anchors",
        "plan_horizon",
        "action_dim",
        "action_chunk_dim",
        "num_anchors",
        "fit_method",
        "seed",
        "metadata",
    }
    missing = required_keys.difference(normalized.keys())
    if missing:
        raise KeyError(f"Anchor bundle is missing required keys: {sorted(missing)}.")

    plan_horizon = int(normalized["plan_horizon"])
    action_dim = int(normalized["action_dim"])
    action_chunk_dim = int(normalized["action_chunk_dim"])
    num_anchors = int(normalized["num_anchors"])
    expected_dim = infer_action_chunk_dim(plan_horizon, action_dim)
    if action_chunk_dim != expected_dim:
        raise ValueError(
            f"Anchor bundle action_chunk_dim mismatch: {action_chunk_dim} != {expected_dim}."
        )
    action_chunk_horizon = normalized["action_chunk_horizon"]
    if action_chunk_horizon is not None and int(action_chunk_horizon) != int(plan_horizon):
        raise ValueError(
            "Anchor bundle action_chunk_horizon must match plan_horizon, got "
            f"{action_chunk_horizon} and {plan_horizon}."
        )
    receding_horizon = normalized["receding_horizon"]
    action_block = normalized["action_block"]
    if receding_horizon is not None and action_block is not None:
        if int(receding_horizon) * int(action_block) != int(plan_horizon):
            raise ValueError(
                "Anchor bundle rollout shape mismatch: "
                f"receding_horizon * action_block = {receding_horizon} * {action_block} != {plan_horizon}."
            )

    anchors = validate_anchor_tensor(
        normalized["anchors"],
        plan_horizon=plan_horizon,
        action_dim=action_dim,
        num_anchors=num_anchors,
    )  # [K, action_chunk_dim]
    if int(anchors.shape[-1]) != action_chunk_dim:
        raise ValueError(
            f"Anchor tensor width {anchors.shape[-1]} does not match action_chunk_dim {action_chunk_dim}."
        )
    if not isinstance(normalized["fit_method"], str):
        raise ValueError("bundle['fit_method'] must be a string.")
    if normalized["seed"] is not None and not isinstance(normalized["seed"], int):
        raise ValueError("bundle['seed'] must be an int or None.")
    if not isinstance(normalized["metadata"], dict):
        raise ValueError("bundle['metadata'] must be a dict.")
    if normalized["task"] is not None and not isinstance(normalized["task"], str):
        raise ValueError("bundle['task'] must be a string or None.")
    if normalized["dataset_path"] is not None and not isinstance(normalized["dataset_path"], str):
        raise ValueError("bundle['dataset_path'] must be a string or None.")
    return normalized
