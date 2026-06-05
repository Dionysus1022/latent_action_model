#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion.anchors import (  # noqa: E402
    ActionAnchorBundle,
    build_action_anchor_bundle,
    save_anchor_bundle,
)
from planners.build_action_anchors import (  # noqa: E402
    filter_invalid_teacher_plan_rows,
    infer_dataset_anchor_spec,
    load_dataset_bundle,
    log_dataset_anchor_spec,
    maybe_subsample_indices,
    resolve_selection_indices,
    validate_dataset_bundle,
)


@dataclass(frozen=True)
class KMeansNearestResult:
    anchors: torch.Tensor
    centroid_indices: torch.Tensor
    selected_dataset_indices: torch.Tensor
    centroid_norm_mean: float
    real_anchor_norm_mean: float
    centroid_to_real_l2_mean: float
    empty_cluster_count: int
    kmeans_inertia: float
    kmeans_iterations: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Reacher-specific action anchors by replacing each K-means centroid "
            "with the nearest real trajectory action chunk."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        default="/data/ykz/reacher/diffusion_pipeline/reacher_planner_dataset.pt",
        help="Path to the Reacher planner dataset `.pt` containing teacher_plan.",
    )
    parser.add_argument(
        "--output-path",
        default="/data/ykz/reacher/diffusion_pipeline/reacher_action_anchors_k128_kmeans_nearest.pt",
        help="Output `.pt` path for the Reacher kmeans-nearest anchor bundle.",
    )
    parser.add_argument("--num-anchors", type=int, default=128, help="Number of anchors to build.")
    parser.add_argument("--seed", type=int, default=42, help="K-means and subsampling seed.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200000,
        help="Optional cap on teacher_plan rows used for fitting.",
    )
    parser.add_argument("--max-iter", type=int, default=300, help="Maximum K-means iterations.")
    parser.add_argument(
        "--split",
        default="train",
        help="Preferred split when dataset exposes split metadata. Default: train.",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Optional subset name when dataset exposes subset metadata.",
    )
    parser.add_argument(
        "--on-error",
        choices=["raise", "skip"],
        default="skip",
        help="How to handle invalid teacher_plan rows.",
    )
    return parser.parse_args(argv)


def _validate_reacher_dataset(task_name: str) -> None:
    if str(task_name).lower().strip() != "reacher":
        raise ValueError(
            "This script is intentionally Reacher-only. "
            f"Dataset task resolved to {task_name!r}."
        )


def _select_nearest_real_samples(
    *,
    teacher_plan: torch.Tensor,
    centroids: torch.Tensor,
    labels: np.ndarray,
    dataset_indices: torch.Tensor,
) -> KMeansNearestResult:
    anchors: list[torch.Tensor] = []
    selected_dataset_indices: list[int] = []
    centroid_to_real_l2: list[float] = []
    used_local_indices: set[int] = set()
    empty_cluster_count = 0

    for cluster_index in range(int(centroids.shape[0])):
        local_indices_np = np.nonzero(labels == cluster_index)[0]
        if local_indices_np.size == 0:
            empty_cluster_count += 1
            local_indices = torch.arange(int(teacher_plan.shape[0]), dtype=torch.long)
        else:
            local_indices = torch.from_numpy(local_indices_np).long()

        cluster_rows = teacher_plan.index_select(0, local_indices)  # [M, D]
        centroid = centroids[cluster_index].view(1, -1)  # [1, D]
        distances = torch.linalg.vector_norm(cluster_rows - centroid, ord=2, dim=1)  # [M]

        order = torch.argsort(distances)
        chosen_local = int(local_indices[int(order[0].item())].item())
        for order_pos in order.tolist():
            candidate_local = int(local_indices[int(order_pos)].item())
            if candidate_local not in used_local_indices:
                chosen_local = candidate_local
                break
        used_local_indices.add(chosen_local)

        anchor = teacher_plan[chosen_local].detach().cpu().float()
        anchors.append(anchor)
        selected_dataset_indices.append(int(dataset_indices[chosen_local].item()))
        centroid_to_real_l2.append(
            float(torch.linalg.vector_norm(anchor - centroids[cluster_index], ord=2).item())
        )

    anchor_tensor = torch.stack(anchors, dim=0)  # [K, D]
    selected_index_tensor = torch.tensor(selected_dataset_indices, dtype=torch.long)
    centroid_index_tensor = torch.arange(int(centroids.shape[0]), dtype=torch.long)
    centroid_to_real_tensor = torch.tensor(centroid_to_real_l2, dtype=torch.float32)
    return KMeansNearestResult(
        anchors=anchor_tensor,
        centroid_indices=centroid_index_tensor,
        selected_dataset_indices=selected_index_tensor,
        centroid_norm_mean=float(torch.linalg.vector_norm(centroids, ord=2, dim=1).mean().item()),
        real_anchor_norm_mean=float(torch.linalg.vector_norm(anchor_tensor, ord=2, dim=1).mean().item()),
        centroid_to_real_l2_mean=float(centroid_to_real_tensor.mean().item()),
        empty_cluster_count=int(empty_cluster_count),
        kmeans_inertia=float("nan"),
        kmeans_iterations=0,
    )


def fit_reacher_kmeans_nearest_anchors(
    teacher_plan: torch.Tensor,
    *,
    num_anchors: int,
    seed: int = 42,
    max_iter: int = 300,
    dataset_indices: torch.Tensor | None = None,
    metadata: dict[str, Any] | None = None,
    plan_horizon: int | None = None,
    action_dim: int | None = None,
    receding_horizon: int | None = None,
    action_block: int | None = None,
    dataset_path: str | None = None,
    max_samples: int | None = None,
) -> ActionAnchorBundle:
    if num_anchors <= 0:
        raise ValueError(f"num_anchors must be positive, got {num_anchors}.")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}.")
    teacher_plan = torch.as_tensor(teacher_plan).detach().cpu().float()
    if teacher_plan.ndim != 2:
        raise ValueError(f"teacher_plan must have shape [N, D], got {tuple(teacher_plan.shape)}.")
    if int(teacher_plan.shape[0]) < int(num_anchors):
        raise ValueError(
            f"Cannot fit {num_anchors} anchors from only {teacher_plan.shape[0]} teacher_plan rows."
        )
    if dataset_indices is None:
        dataset_indices = torch.arange(int(teacher_plan.shape[0]), dtype=torch.long)
    else:
        dataset_indices = torch.as_tensor(dataset_indices, dtype=torch.long).detach().cpu()
    if int(dataset_indices.numel()) != int(teacher_plan.shape[0]):
        raise ValueError(
            "dataset_indices length must match teacher_plan rows: "
            f"{dataset_indices.numel()} != {teacher_plan.shape[0]}."
        )

    kmeans = KMeans(
        n_clusters=int(num_anchors),
        random_state=int(seed),
        n_init=10,
        max_iter=int(max_iter),
    )
    labels = kmeans.fit_predict(teacher_plan.numpy())
    centroids = torch.from_numpy(kmeans.cluster_centers_).float()
    result = _select_nearest_real_samples(
        teacher_plan=teacher_plan,
        centroids=centroids,
        labels=labels,
        dataset_indices=dataset_indices,
    )

    action_chunk_dim = int(teacher_plan.shape[-1])
    if action_dim is None and plan_horizon is None:
        action_dim = action_chunk_dim
        plan_horizon = 1
    elif action_dim is None:
        if action_chunk_dim % int(plan_horizon) != 0:
            raise ValueError("teacher_plan width is not divisible by plan_horizon.")
        action_dim = int(action_chunk_dim // int(plan_horizon))
    elif plan_horizon is None:
        if action_chunk_dim % int(action_dim) != 0:
            raise ValueError("teacher_plan width is not divisible by action_dim.")
        plan_horizon = int(action_chunk_dim // int(action_dim))

    metadata_dict = dict(metadata or {})
    metadata_dict.update(
        {
            "fit_method": "kmeans_nearest_real_sample",
            "kmeans_inertia": float(kmeans.inertia_),
            "kmeans_iterations": int(kmeans.n_iter_),
            "centroid_norm_mean": float(result.centroid_norm_mean),
            "real_anchor_norm_mean": float(result.real_anchor_norm_mean),
            "centroid_to_real_l2_mean": float(result.centroid_to_real_l2_mean),
            "empty_cluster_count": int(result.empty_cluster_count),
            "selected_dataset_indices": result.selected_dataset_indices.tolist(),
        }
    )
    return build_action_anchor_bundle(
        result.anchors,
        plan_horizon=int(plan_horizon),
        action_dim=int(action_dim),
        receding_horizon=None if receding_horizon is None else int(receding_horizon),
        action_block=None if action_block is None else int(action_block),
        task="reacher",
        dataset_path=dataset_path,
        max_samples=max_samples,
        fit_method="kmeans_nearest_real_sample",
        seed=int(seed),
        metadata=metadata_dict,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.num_anchors <= 0:
        raise ValueError("--num-anchors must be positive.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided.")
    start_time = time.time()

    dataset_bundle = load_dataset_bundle(args.dataset_path)
    teacher_plan, meta, build_info = validate_dataset_bundle(dataset_bundle)
    dataset_spec, spec_warnings = infer_dataset_anchor_spec(
        teacher_plan=teacher_plan,
        meta=meta,
        build_info=build_info,
        args=argparse.Namespace(
            task="reacher",
            action_dim=None,
            action_chunk_horizon=None,
            receding_horizon=None,
            action_block=None,
        ),
    )
    _validate_reacher_dataset(dataset_spec.task)

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
        mode="build",
        num_anchors=args.num_anchors,
    )
    print(
        f"[sample] using {int(indices.numel())} teacher_plan rows"
        + (f" after {subsample_reason}" if subsample_reason else "")
    )
    indices, num_invalid = filter_invalid_teacher_plan_rows(
        teacher_plan,
        indices,
        on_error=args.on_error,
    )
    if num_invalid > 0:
        print(f"[filter] skipped {num_invalid} invalid teacher_plan rows due to --on-error={args.on_error}")

    selected_teacher_plan = teacher_plan.index_select(0, indices)
    if int(selected_teacher_plan.shape[0]) < int(args.num_anchors):
        raise ValueError(
            f"Need at least {args.num_anchors} teacher_plan rows after filtering, "
            f"got {selected_teacher_plan.shape[0]}."
        )

    source_dataset = str(Path(args.dataset_path).expanduser().resolve())
    metadata = {
        "task": "reacher",
        "dataset_path": source_dataset,
        "source_dataset": source_dataset,
        "requested_split": args.split,
        "requested_subset": args.subset,
        "max_samples": args.max_samples,
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
    anchor_bundle = fit_reacher_kmeans_nearest_anchors(
        selected_teacher_plan,
        num_anchors=int(args.num_anchors),
        seed=int(args.seed),
        max_iter=int(args.max_iter),
        dataset_indices=indices,
        metadata=metadata,
        plan_horizon=int(dataset_spec.action_chunk_horizon),
        action_dim=int(dataset_spec.action_dim),
        receding_horizon=int(dataset_spec.receding_horizon),
        action_block=int(dataset_spec.action_block),
        dataset_path=source_dataset,
        max_samples=args.max_samples,
    )
    output_path = save_anchor_bundle(anchor_bundle, args.output_path)
    elapsed = time.time() - start_time
    print(
        f"[done] wrote {anchor_bundle.num_anchors} Reacher kmeans-nearest anchors to {output_path} "
        f"time={elapsed:.2f}s"
    )
    print(
        "[stats] "
        f"centroid_norm_mean={anchor_bundle.metadata['centroid_norm_mean']:.6f} "
        f"real_anchor_norm_mean={anchor_bundle.metadata['real_anchor_norm_mean']:.6f} "
        f"centroid_to_real_l2_mean={anchor_bundle.metadata['centroid_to_real_l2_mean']:.6f} "
        f"empty_cluster_count={anchor_bundle.metadata['empty_cluster_count']}"
    )


if __name__ == "__main__":
    main()
