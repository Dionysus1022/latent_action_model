from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planners.action_anchors import (  # noqa: E402
    fit_action_anchors,
    save_anchor_bundle,
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
class DatasetAnchorSpec:
    task: str
    action_dim: int
    action_chunk_horizon: int
    action_chunk_dim: int
    receding_horizon: int
    action_block: int
    field_sources: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reusable action anchors from a single-peak dataset bundle.",
    )
    parser.add_argument(
        "--mode",
        choices=["probe", "build"],
        required=True,
        help="probe: inspect/few-sample fit sanity; build: fit anchors and save a bundle.",
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to a single-peak dataset `.pt` bundle containing teacher_plan.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output `.pt` path for the fitted anchor bundle. Required in build mode.",
    )
    parser.add_argument(
        "--num-anchors",
        type=int,
        required=True,
        help="Number of action anchors to fit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling and K-means initialization.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of teacher_plan rows used for fitting.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Preferred split name to use when the dataset exposes split metadata. Default: train.",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Optional subset name when the dataset exposes subset metadata.",
    )
    parser.add_argument(
        "--on-error",
        choices=["raise", "skip"],
        default="skip",
        help="How to handle invalid teacher_plan rows (e.g. NaN / Inf).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=300,
        help="Maximum K-means iterations.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Optional fallback task name when the dataset bundle metadata does not expose one.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=None,
        help="Optional fallback action dimension when the dataset bundle metadata does not expose one.",
    )
    parser.add_argument(
        "--action-chunk-horizon",
        type=int,
        default=None,
        help="Optional fallback action chunk horizon when the dataset bundle metadata does not expose one.",
    )
    parser.add_argument(
        "--receding-horizon",
        type=int,
        default=None,
        help="Optional fallback receding horizon when the dataset bundle metadata does not expose one.",
    )
    parser.add_argument(
        "--action-block",
        type=int,
        default=None,
        help="Optional fallback action block when the dataset bundle metadata does not expose one.",
    )
    return parser.parse_args()


def load_dataset_bundle(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset bundle not found: {dataset_path}")
    return torch.load(dataset_path, map_location="cpu")


def validate_dataset_bundle(dataset_bundle: dict[str, Any]) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    required_keys = {"teacher_plan", "meta"}
    missing = required_keys.difference(dataset_bundle.keys())
    if missing:
        raise KeyError(f"Dataset bundle is missing required keys: {sorted(missing)}.")

    teacher_plan = dataset_bundle["teacher_plan"]
    meta = dataset_bundle["meta"]
    build_info = dataset_bundle.get("build_info", {})

    if not torch.is_tensor(teacher_plan) or teacher_plan.ndim != 2:
        raise ValueError(
            "dataset_bundle['teacher_plan'] must have shape [N, action_chunk_dim], "
            f"got {type(teacher_plan)} with shape {getattr(teacher_plan, 'shape', None)}."
        )
    if not isinstance(meta, list) or len(meta) == 0:
        raise ValueError("dataset_bundle['meta'] must be a non-empty list of dicts.")
    if len(meta) != int(teacher_plan.shape[0]):
        raise ValueError(
            f"meta length {len(meta)} must match teacher_plan rows {teacher_plan.shape[0]}."
        )
    if not isinstance(build_info, dict):
        raise ValueError("dataset_bundle['build_info'] must be a dict when present.")

    return teacher_plan.cpu().float(), meta, build_info


def normalize_task_name(task_name: str) -> str:
    normalized = str(task_name).strip().lower()
    return TASK_ALIASES.get(normalized, normalized)


def choose_first_value(candidates: list[tuple[str, Any]]) -> tuple[Any, str | None]:
    for source, value in candidates:
        if value not in [None, "", "null"]:
            return value, source
    return None, None


def maybe_get_nested(source: dict[str, Any] | None, path: list[str]) -> Any:
    current: Any = source
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def require_positive_int(
    *,
    field_name: str,
    value: Any,
    source: str,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not infer '{field_name}': value from {source} is not an integer: {value!r}."
        ) from exc
    if parsed <= 0:
        raise ValueError(
            f"Could not infer '{field_name}': value from {source} must be positive, got {parsed}."
        )
    return parsed


def append_fallback_warning(
    warnings: list[str],
    *,
    field_name: str,
    source: str | None,
) -> None:
    if source is None or source.startswith("build_info"):
        return
    warnings.append(
        f"{field_name} inferred from {source} because build_info did not provide it."
    )


def infer_dataset_anchor_spec(
    *,
    teacher_plan: torch.Tensor,
    meta: list[dict[str, Any]],
    build_info: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[DatasetAnchorSpec, list[str]]:
    meta0 = meta[0]
    warnings: list[str] = []
    field_sources: dict[str, str] = {}
    teacher_plan_width = int(teacher_plan.shape[-1])  # [N, action_chunk_dim] -> action_chunk_dim

    task_value, task_source = choose_first_value(
        [
            ("build_info.task", build_info.get("task")),
            ("build_info.requested_task", build_info.get("requested_task")),
            ("meta[0].task", meta0.get("task")),
            ("cli --task", args.task),
        ]
    )
    if task_value is None:
        raise KeyError(
            "Could not infer 'task'. Provide it in dataset build_info/meta or pass --task "
            "(aliases: two-room/tworoom, researcher/reacher)."
        )
    task = normalize_task_name(str(task_value))
    append_fallback_warning(warnings, field_name="task", source=task_source)
    field_sources["task"] = str(task_source)

    action_chunk_dim_value, action_chunk_dim_source = choose_first_value(
        [
            ("build_info.action_chunk_dim", build_info.get("action_chunk_dim")),
            ("build_info.task_spec.action_chunk_dim", maybe_get_nested(build_info, ["task_spec", "action_chunk_dim"])),
            ("meta[0].action_chunk_dim", meta0.get("action_chunk_dim")),
        ]
    )
    if action_chunk_dim_value is not None:
        declared_action_chunk_dim = require_positive_int(
            field_name="action_chunk_dim",
            value=action_chunk_dim_value,
            source=str(action_chunk_dim_source),
        )
        if declared_action_chunk_dim != teacher_plan_width:
            raise ValueError(
                "teacher_plan width does not match dataset metadata action_chunk_dim: "
                f"{teacher_plan_width} != {declared_action_chunk_dim}."
            )
        append_fallback_warning(warnings, field_name="action_chunk_dim", source=action_chunk_dim_source)
    field_sources["action_chunk_dim"] = "teacher_plan.shape[-1]"
    action_chunk_dim = int(teacher_plan_width)

    receding_horizon_value, receding_horizon_source = choose_first_value(
        [
            ("build_info.plan_config.receding_horizon", maybe_get_nested(build_info, ["plan_config", "receding_horizon"])),
            ("build_info.task_spec.receding_horizon", maybe_get_nested(build_info, ["task_spec", "receding_horizon"])),
            ("meta[0].receding_horizon", meta0.get("receding_horizon")),
            ("cli --receding-horizon", args.receding_horizon),
        ]
    )
    receding_horizon = None
    if receding_horizon_value is not None:
        receding_horizon = require_positive_int(
            field_name="receding_horizon",
            value=receding_horizon_value,
            source=str(receding_horizon_source),
        )
        field_sources["receding_horizon"] = str(receding_horizon_source)
        append_fallback_warning(warnings, field_name="receding_horizon", source=receding_horizon_source)

    action_block_value, action_block_source = choose_first_value(
        [
            ("build_info.plan_config.action_block", maybe_get_nested(build_info, ["plan_config", "action_block"])),
            ("build_info.task_spec.action_block", maybe_get_nested(build_info, ["task_spec", "action_block"])),
            ("meta[0].action_block", meta0.get("action_block")),
            ("cli --action-block", args.action_block),
        ]
    )
    action_block = None
    if action_block_value is not None:
        action_block = require_positive_int(
            field_name="action_block",
            value=action_block_value,
            source=str(action_block_source),
        )
        field_sources["action_block"] = str(action_block_source)
        append_fallback_warning(warnings, field_name="action_block", source=action_block_source)

    action_chunk_horizon_value, action_chunk_horizon_source = choose_first_value(
        [
            ("build_info.action_chunk_horizon", build_info.get("action_chunk_horizon")),
            (
                "build_info.task_spec.action_chunk_horizon",
                maybe_get_nested(build_info, ["task_spec", "action_chunk_horizon"]),
            ),
            ("meta[0].plan_horizon", meta0.get("plan_horizon")),
            ("cli --action-chunk-horizon", args.action_chunk_horizon),
        ]
    )
    if action_chunk_horizon_value is None and receding_horizon is not None and action_block is not None:
        action_chunk_horizon_value = int(receding_horizon * action_block)
        action_chunk_horizon_source = "derived:receding_horizon*action_block"
    action_chunk_horizon = None
    if action_chunk_horizon_value is not None:
        action_chunk_horizon = require_positive_int(
            field_name="action_chunk_horizon",
            value=action_chunk_horizon_value,
            source=str(action_chunk_horizon_source),
        )
        field_sources["action_chunk_horizon"] = str(action_chunk_horizon_source)
        append_fallback_warning(warnings, field_name="action_chunk_horizon", source=action_chunk_horizon_source)

    action_dim_value, action_dim_source = choose_first_value(
        [
            ("build_info.action_dim", build_info.get("action_dim")),
            ("build_info.task_spec.action_dim", maybe_get_nested(build_info, ["task_spec", "action_dim"])),
            ("meta[0].action_dim", meta0.get("action_dim")),
            ("cli --action-dim", args.action_dim),
        ]
    )
    if action_dim_value is None and action_chunk_horizon is not None:
        if action_chunk_dim % int(action_chunk_horizon) != 0:
            raise KeyError(
                "Could not infer 'action_dim': build_info/meta/CLI did not provide it, "
                f"and teacher_plan width {action_chunk_dim} is not divisible by action_chunk_horizon={action_chunk_horizon}."
            )
        action_dim_value = int(action_chunk_dim // int(action_chunk_horizon))
        action_dim_source = "derived:teacher_plan.shape[-1]/action_chunk_horizon"
    action_dim = None
    if action_dim_value is not None:
        action_dim = require_positive_int(
            field_name="action_dim",
            value=action_dim_value,
            source=str(action_dim_source),
        )
        field_sources["action_dim"] = str(action_dim_source)
        append_fallback_warning(warnings, field_name="action_dim", source=action_dim_source)

    if action_chunk_horizon is None and action_dim is not None:
        if action_chunk_dim % int(action_dim) != 0:
            raise KeyError(
                "Could not infer 'action_chunk_horizon': build_info/meta/CLI did not provide it, "
                f"and teacher_plan width {action_chunk_dim} is not divisible by action_dim={action_dim}."
            )
        action_chunk_horizon = int(action_chunk_dim // int(action_dim))
        field_sources["action_chunk_horizon"] = "derived:teacher_plan.shape[-1]/action_dim"
        append_fallback_warning(
            warnings,
            field_name="action_chunk_horizon",
            source=field_sources["action_chunk_horizon"],
        )

    if action_dim is None:
        raise KeyError(
            "Could not infer 'action_dim'. Provide it in dataset build_info/meta or pass --action-dim."
        )
    if action_chunk_horizon is None:
        raise KeyError(
            "Could not infer 'action_chunk_horizon'. Provide it in dataset build_info/meta or pass "
            "--action-chunk-horizon."
        )
    expected_chunk_dim = int(action_dim * action_chunk_horizon)
    if expected_chunk_dim != action_chunk_dim:
        raise ValueError(
            "Inferred action chunk shape is inconsistent with teacher_plan width: "
            f"action_dim * action_chunk_horizon = {action_dim} * {action_chunk_horizon} = {expected_chunk_dim}, "
            f"but teacher_plan width is {action_chunk_dim}."
        )

    if receding_horizon is None and action_block is not None:
        if action_chunk_horizon % int(action_block) != 0:
            raise KeyError(
                "Could not infer 'receding_horizon': action_chunk_horizon is not divisible by action_block: "
                f"{action_chunk_horizon} % {action_block} != 0."
            )
        receding_horizon = int(action_chunk_horizon // int(action_block))
        field_sources["receding_horizon"] = "derived:action_chunk_horizon/action_block"
        append_fallback_warning(
            warnings,
            field_name="receding_horizon",
            source=field_sources["receding_horizon"],
        )
    if action_block is None and receding_horizon is not None:
        if action_chunk_horizon % int(receding_horizon) != 0:
            raise KeyError(
                "Could not infer 'action_block': action_chunk_horizon is not divisible by receding_horizon: "
                f"{action_chunk_horizon} % {receding_horizon} != 0."
            )
        action_block = int(action_chunk_horizon // int(receding_horizon))
        field_sources["action_block"] = "derived:action_chunk_horizon/receding_horizon"
        append_fallback_warning(
            warnings,
            field_name="action_block",
            source=field_sources["action_block"],
        )

    if receding_horizon is None or action_block is None:
        raise KeyError(
            "Could not infer 'receding_horizon' and 'action_block'. "
            "Please ensure dataset build_info.plan_config is present, or pass "
            "--receding-horizon and --action-block."
        )
    if int(receding_horizon) * int(action_block) != int(action_chunk_horizon):
        raise ValueError(
            "Inferred rollout shape does not match action_chunk_horizon: "
            f"{receding_horizon} * {action_block} != {action_chunk_horizon}."
        )

    return (
        DatasetAnchorSpec(
            task=task,
            action_dim=int(action_dim),
            action_chunk_horizon=int(action_chunk_horizon),
            action_chunk_dim=int(action_chunk_dim),
            receding_horizon=int(receding_horizon),
            action_block=int(action_block),
            field_sources=field_sources,
        ),
        warnings,
    )


def log_dataset_anchor_spec(
    dataset_path: str | Path,
    spec: DatasetAnchorSpec,
    teacher_plan: torch.Tensor,
) -> None:
    print(
        f"[load] dataset={Path(dataset_path).expanduser().resolve()} "
        f"teacher_plan_shape={tuple(teacher_plan.shape)} "
        f"task={spec.task} action_dim={spec.action_dim} "
        f"action_chunk_horizon={spec.action_chunk_horizon} "
        f"action_chunk_dim={spec.action_chunk_dim} "
        f"receding_horizon={spec.receding_horizon} action_block={spec.action_block}"
    )
    print(f"[spec] field_sources={spec.field_sources}")


def maybe_extract_per_sample_field(meta: list[dict[str, Any]], key: str) -> list[Any] | None:
    if len(meta) == 0:
        return None
    if all(isinstance(sample_meta, dict) and key in sample_meta for sample_meta in meta):
        return [sample_meta[key] for sample_meta in meta]
    return None


def resolve_selection_indices(
    *,
    teacher_plan: torch.Tensor,
    meta: list[dict[str, Any]],
    build_info: dict[str, Any],
    split: str,
    subset: str | None,
) -> tuple[torch.Tensor, list[str]]:
    num_rows = int(teacher_plan.shape[0])
    indices = torch.arange(num_rows, dtype=torch.long)
    notes: list[str] = []

    if split != "all":
        split_labels = maybe_extract_per_sample_field(meta, "split")
        build_split = build_info.get("split", None)
        if split_labels is not None:
            mask = torch.tensor([str(value) == split for value in split_labels], dtype=torch.bool)
            indices = indices[mask]
            notes.append(f"split filter from meta['split']: requested={split} kept={int(mask.sum())}/{num_rows}")
        elif build_split is not None:
            if str(build_split) == split:
                notes.append(f"dataset build_info['split']={build_split}; using all rows for requested split={split}")
            else:
                indices = indices[:0]
                notes.append(
                    f"dataset build_info['split']={build_split} does not match requested split={split}; keeping 0 rows"
                )
        elif split == "train":
            notes.append("dataset has no split metadata; using full dataset as the train subset")
        else:
            raise ValueError(
                f"Requested split='{split}' but dataset exposes no split metadata. "
                "Use --split train or --split all for the current dataset."
            )

    if subset is not None:
        subset_labels = maybe_extract_per_sample_field(meta, "subset")
        build_subset = build_info.get("subset", None)
        if subset_labels is not None:
            current_positions = indices.tolist()
            mask = torch.tensor([str(subset_labels[position]) == subset for position in current_positions], dtype=torch.bool)
            indices = indices[mask]
            notes.append(
                f"subset filter from meta['subset']: requested={subset} kept={int(mask.sum())}/{len(current_positions)}"
            )
        elif build_subset is not None:
            if str(build_subset) == subset:
                notes.append(
                    f"dataset build_info['subset']={build_subset}; using current selection for requested subset={subset}"
                )
            else:
                indices = indices[:0]
                notes.append(
                    f"dataset build_info['subset']={build_subset} does not match requested subset={subset}; keeping 0 rows"
                )
        else:
            raise ValueError(
                f"Requested subset='{subset}' but dataset exposes no subset metadata."
            )

    if int(indices.numel()) == 0:
        raise ValueError("Selection produced 0 teacher_plan rows.")
    return indices, notes


def maybe_subsample_indices(
    indices: torch.Tensor,
    *,
    max_samples: int | None,
    seed: int,
    mode: str,
    num_anchors: int,
) -> tuple[torch.Tensor, str | None]:
    if max_samples is not None and max_samples <= 0:
        raise ValueError(f"--max-samples must be positive when provided, got {max_samples}.")

    current_count = int(indices.numel())
    target_count = current_count
    reason = None

    if max_samples is not None:
        target_count = min(current_count, int(max_samples))
        reason = f"max_samples={max_samples}"
    elif mode == "probe":
        probe_cap = max(int(num_anchors) * 4, 256)
        target_count = min(current_count, probe_cap)
        reason = f"probe_cap={probe_cap}"

    if target_count >= current_count:
        return indices, reason

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(indices.numpy(), size=target_count, replace=False))
    return torch.from_numpy(chosen).long(), reason


def filter_invalid_teacher_plan_rows(
    teacher_plan: torch.Tensor,
    indices: torch.Tensor,
    *,
    on_error: str,
) -> tuple[torch.Tensor, int]:
    selected = teacher_plan.index_select(0, indices)  # [M, action_chunk_dim]
    finite_mask = torch.isfinite(selected).all(dim=1)  # [M]
    num_invalid = int((~finite_mask).sum().item())
    if num_invalid == 0:
        return indices, 0
    if on_error == "raise":
        raise ValueError(f"Found {num_invalid} invalid teacher_plan rows (NaN or Inf).")
    filtered = indices[finite_mask]
    if int(filtered.numel()) == 0:
        raise ValueError("All selected teacher_plan rows were invalid after filtering.")
    return filtered, num_invalid


def main() -> None:
    args = parse_args()
    if args.mode == "build" and args.output_path is None:
        raise ValueError("--output-path is required in build mode.")
    if args.num_anchors <= 0:
        raise ValueError("--num-anchors must be positive.")
    if args.action_dim is not None and args.action_dim <= 0:
        raise ValueError("--action-dim must be positive when provided.")
    if args.action_chunk_horizon is not None and args.action_chunk_horizon <= 0:
        raise ValueError("--action-chunk-horizon must be positive when provided.")
    if args.receding_horizon is not None and args.receding_horizon <= 0:
        raise ValueError("--receding-horizon must be positive when provided.")
    if args.action_block is not None and args.action_block <= 0:
        raise ValueError("--action-block must be positive when provided.")

    start_time = time.time()
    dataset_bundle = load_dataset_bundle(args.dataset_path)
    teacher_plan, meta, build_info = validate_dataset_bundle(dataset_bundle)
    dataset_spec, spec_warnings = infer_dataset_anchor_spec(
        teacher_plan=teacher_plan,
        meta=meta,
        build_info=build_info,
        args=args,
    )

    indices, selection_notes = resolve_selection_indices(
        teacher_plan=teacher_plan,
        meta=meta,
        build_info=build_info,
        split=args.split,
        subset=args.subset,
    )
    log_dataset_anchor_spec(args.dataset_path, dataset_spec, teacher_plan)
    for warning in spec_warnings:
        print(f"[warn] {warning}")
    for note in selection_notes:
        print(f"[select] {note}")

    indices, subsample_reason = maybe_subsample_indices(
        indices,
        max_samples=args.max_samples,
        seed=args.seed,
        mode=args.mode,
        num_anchors=args.num_anchors,
    )
    if subsample_reason is not None:
        print(f"[sample] using {int(indices.numel())} teacher_plan rows after {subsample_reason}")
    else:
        print(f"[sample] using all {int(indices.numel())} selected teacher_plan rows")

    indices, num_invalid = filter_invalid_teacher_plan_rows(
        teacher_plan,
        indices,
        on_error=args.on_error,
    )
    if num_invalid > 0:
        print(f"[filter] skipped {num_invalid} invalid teacher_plan rows due to --on-error={args.on_error}")

    selected_teacher_plan = teacher_plan.index_select(0, indices)  # [M, action_chunk_dim]
    if int(selected_teacher_plan.shape[0]) < args.num_anchors:
        raise ValueError(
            f"Need at least {args.num_anchors} teacher_plan rows after filtering, "
            f"got {selected_teacher_plan.shape[0]}."
        )

    metadata = {
        "task": dataset_spec.task,
        "dataset_path": str(Path(args.dataset_path).expanduser().resolve()),
        "source_dataset": str(Path(args.dataset_path).expanduser().resolve()),
        "requested_split": args.split,
        "requested_subset": args.subset,
        "max_samples": args.max_samples,
        "mode": args.mode,
        "source_build_info": build_info,
        "action_dim": int(dataset_spec.action_dim),
        "action_chunk_horizon": int(dataset_spec.action_chunk_horizon),
        "action_chunk_dim": int(dataset_spec.action_chunk_dim),
        "receding_horizon": int(dataset_spec.receding_horizon),
        "action_block": int(dataset_spec.action_block),
        "field_sources": dict(dataset_spec.field_sources),
        "inference_warnings": list(spec_warnings),
        "selection_notes": list(selection_notes),
        "subsample_reason": subsample_reason,
        "num_invalid_teacher_plans": int(num_invalid),
        "num_selected_teacher_plans": int(selected_teacher_plan.shape[0]),
    }
    anchor_bundle = fit_action_anchors(
        selected_teacher_plan,
        num_anchors=args.num_anchors,
        plan_horizon=dataset_spec.action_chunk_horizon,
        action_dim=dataset_spec.action_dim,
        receding_horizon=dataset_spec.receding_horizon,
        action_block=dataset_spec.action_block,
        task=dataset_spec.task,
        dataset_path=str(Path(args.dataset_path).expanduser().resolve()),
        max_samples=args.max_samples,
        seed=args.seed,
        max_iter=args.max_iter,
        metadata=metadata,
    )

    elapsed = time.time() - start_time
    print(
        f"[done] mode={args.mode} num_anchors={anchor_bundle.num_anchors} "
        f"anchor_shape={tuple(anchor_bundle.anchors.shape)} "
        f"fit_method={anchor_bundle.fit_method} time={elapsed:.2f}s"
    )
    print(
        f"[anchor] task={anchor_bundle.task} plan_horizon={anchor_bundle.plan_horizon} "
        f"action_chunk_horizon={anchor_bundle.action_chunk_horizon} "
        f"action_dim={anchor_bundle.action_dim} action_chunk_dim={anchor_bundle.action_chunk_dim} "
        f"receding_horizon={anchor_bundle.receding_horizon} action_block={anchor_bundle.action_block}"
    )
    print(
        f"[anchor] dataset_path={anchor_bundle.dataset_path} "
        f"source_dataset={anchor_bundle.metadata.get('source_dataset', 'unknown')} "
        f"num_selected_teacher_plans={anchor_bundle.metadata.get('num_selected_teacher_plans', 'unknown')}"
    )
    print(f"[anchor] metadata_keys={sorted(anchor_bundle.metadata.keys())}")
    print(f"[anchor] bundle_schema={sorted(anchor_bundle.as_dict().keys())}")

    if args.mode == "probe":
        print("[probe] first anchor preview:")
        print(f"  anchor[0]_shape: {tuple(anchor_bundle.anchors[0].shape)}")  # [action_chunk_dim]
        print(f"  task: {anchor_bundle.task}")
        print(f"  action_dim: {anchor_bundle.action_dim}")
        print(f"  action_chunk_horizon: {anchor_bundle.action_chunk_horizon}")
        print(f"  action_chunk_dim: {anchor_bundle.action_chunk_dim}")
        print(f"  receding_horizon: {anchor_bundle.receding_horizon}")
        print(f"  action_block: {anchor_bundle.action_block}")
        print(f"  anchor[0][:8]: {anchor_bundle.anchors[0][:8].tolist()}")
        return

    output_path = save_anchor_bundle(anchor_bundle, args.output_path)
    print(f"[save] wrote anchor bundle to {output_path}")


if __name__ == "__main__":
    main()
