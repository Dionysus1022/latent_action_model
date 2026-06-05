#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import stable_worldmodel as swm
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from planners.build_single_peak_dataset import (  # noqa: E402
    build_processors,
    clone_info_dict,
    get_dataset,
    img_transform,
    load_eval_cfg,
    prepare_policy_inputs,
    resolve_model_cache_dir,
    resolve_task_spec,
    validate_model_reference,
)
from planners.gc_idm_training import progress_iter  # noqa: E402


def configure_gc_idm_world_model_for_encoding(model: torch.nn.Module) -> torch.nn.Module:
    model.interpolate_pos_encoding = True
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paper-style GC-IDM dataset: (z_t, z_{t+h}, h) -> a_t.",
    )
    parser.add_argument("--task", default="reacher")
    parser.add_argument("--config", default=None)
    parser.add_argument("--wm-policy", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-h5", default=None)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-horizon", type=int, default=50)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--model-cache-dir", default=None)
    return parser.parse_args()


def sample_rows_and_horizons(
    *,
    dataset,
    episode_key: str,
    step_key: str,
    num_samples: int,
    max_horizon: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    episode_idx = dataset.get_col_data(episode_key)
    step_idx = dataset.get_col_data(step_key)
    ep_ids, first_rows = np.unique(episode_idx, return_index=True)
    lengths = []
    for ep_id in ep_ids:
        lengths.append(int(np.max(step_idx[episode_idx == ep_id]) + 1))
    length_by_ep = {int(ep_id): int(length) for ep_id, length in zip(ep_ids, lengths)}
    # Require a full max_horizon window so the builder can read a whole batch
    # with one HDF5 load_chunk call. This avoids hundreds of thousands of tiny
    # random HDF5 reads on large datasets such as Reacher.
    max_start_per_row = np.array(
        [length_by_ep[int(ep_id)] - int(max_horizon) - 1 for ep_id in episode_idx],
        dtype=np.int64,
    )
    valid_rows = np.nonzero(step_idx <= max_start_per_row)[0]
    if len(valid_rows) < num_samples:
        raise ValueError(f"Requested {num_samples} samples but only found {len(valid_rows)} valid rows.")

    rng = np.random.default_rng(int(seed))
    sampled_rows = np.sort(rng.choice(valid_rows, size=int(num_samples), replace=False))
    sampled_episodes = episode_idx[sampled_rows].astype(np.int64)
    sampled_steps = step_idx[sampled_rows].astype(np.int64)
    horizons = []
    for ep_id, step in zip(sampled_episodes, sampled_steps):
        horizons.append(int(rng.integers(1, int(max_horizon) + 1)))
    return sampled_rows, sampled_episodes, sampled_steps, np.asarray(horizons, dtype=np.int64)


def load_gc_idm_batch(
    *,
    dataset,
    task_spec,
    episodes_idx: np.ndarray,
    start_steps: np.ndarray,
    horizons: np.ndarray,
) -> dict[str, torch.Tensor]:
    current_per_key: dict[str, list[torch.Tensor]] = {}
    goal_per_key: dict[str, list[torch.Tensor]] = {}
    action_values: list[torch.Tensor] = []

    context_steps = int(np.max(horizons) + 1)
    data = dataset.load_chunk(episodes_idx, start_steps, start_steps + context_steps)

    for sample_index, ep in enumerate(data):
        horizon = int(horizons[sample_index])
        for col, value in ep.items():
            if not torch.is_tensor(value):
                continue

            resolved_key = col
            if col == task_spec.pixels_key:
                resolved_key = "pixels"
                value = value.permute(0, 2, 3, 1)
            elif col == task_spec.action_key:
                resolved_key = "action"

            if resolved_key == "action":
                action_values.append(value[0])
                continue

            current_per_key.setdefault(resolved_key, []).append(value[0])
            goal_key = "goal" if resolved_key == "pixels" else f"goal_{resolved_key}"
            goal_per_key.setdefault(goal_key, []).append(value[horizon])

    info_dict: dict[str, torch.Tensor] = {}
    for key, values in current_per_key.items():
        info_dict[key] = torch.stack(values, dim=0).unsqueeze(1)
    for key, values in goal_per_key.items():
        info_dict[key] = torch.stack(values, dim=0).unsqueeze(1)
    info_dict["action"] = torch.stack(action_values, dim=0).unsqueeze(1)
    return info_dict


@torch.inference_mode()
def encode_current_goal_batch(
    *,
    model: torch.nn.Module,
    prepared_info: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    current_encoded = model.encode({"pixels": prepared_info["pixels"].to(device)})
    goal_encoded = model.encode({"pixels": prepared_info["goal"].to(device)})
    if "emb" not in current_encoded or "emb" not in goal_encoded:
        raise KeyError("world_model.encode(...) must return a dict containing 'emb'.")
    return current_encoded["emb"][:, -1].detach().cpu().float(), goal_encoded["emb"][:, -1].detach().cpu().float()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.max_horizon <= 0:
        raise ValueError("--max-horizon must be positive.")

    start_time = time.time()
    print(
        f"[setup] task={args.task} dataset_h5={args.dataset_h5} "
        f"num_samples={args.num_samples} batch_size={args.batch_size} "
        f"max_horizon={args.max_horizon} device={args.device}",
        flush=True,
    )
    cfg = load_eval_cfg(args)
    if args.dataset_name not in [None, "", "null"]:
        cfg.eval.dataset_name = str(args.dataset_name)
    if args.cache_dir is not None:
        cfg.cache_dir = args.cache_dir
    dataset = get_dataset(
        cfg,
        cfg.eval.dataset_name,
        cache_dir=args.cache_dir,
        dataset_h5=args.dataset_h5,
    )
    print(f"[dataset] h5={dataset.h5_path} keys={dataset.column_names}", flush=True)
    task_spec = resolve_task_spec(args=args, cfg=cfg, dataset=dataset)
    print(
        f"[task-spec] task={task_spec.canonical_task_name} dataset={task_spec.dataset_name} "
        f"action_dim={task_spec.action_dim} pixels_key={task_spec.pixels_key} "
        f"episode_key={task_spec.episode_key} step_key={task_spec.step_key}",
        flush=True,
    )
    process = build_processors(cfg, dataset)
    print("[setup] processors_ready=1", flush=True)
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }
    model_cache_dir = resolve_model_cache_dir(
        shared_cache_dir=args.cache_dir,
        explicit_model_cache_dir=args.model_cache_dir,
    )
    validate_model_reference(args.wm_policy, "--wm-policy")
    model = swm.policy.AutoCostModel(args.wm_policy, cache_dir=model_cache_dir).to(args.device).eval()
    configure_gc_idm_world_model_for_encoding(model)
    model.requires_grad_(False)
    print(f"[setup] world_model_ready=1 policy={args.wm_policy}", flush=True)

    sampled_rows, sampled_episodes, sampled_steps, horizons = sample_rows_and_horizons(
        dataset=dataset,
        episode_key=task_spec.episode_key,
        step_key=task_spec.step_key,
        num_samples=int(args.num_samples),
        max_horizon=int(args.max_horizon),
        seed=int(args.seed),
    )
    print(
        f"[sample] rows={len(sampled_rows)} horizon_min={int(np.min(horizons))} "
        f"horizon_max={int(np.max(horizons))}",
        flush=True,
    )

    z_cur_parts: list[torch.Tensor] = []
    z_goal_parts: list[torch.Tensor] = []
    horizon_parts: list[torch.Tensor] = []
    action_parts: list[torch.Tensor] = []
    meta: list[dict[str, Any]] = []

    batch_starts = range(0, int(args.num_samples), int(args.batch_size))
    batch_iter = progress_iter(
        batch_starts,
        desc="gc-idm build",
        total=len(batch_starts),
        unit="batch",
        leave=True,
    )
    for batch_start in batch_iter:
        batch_end = min(batch_start + int(args.batch_size), int(args.num_samples))
        batch_slice = slice(batch_start, batch_end)
        raw_info = load_gc_idm_batch(
            dataset=dataset,
            task_spec=task_spec,
            episodes_idx=sampled_episodes[batch_slice],
            start_steps=sampled_steps[batch_slice],
            horizons=horizons[batch_slice],
        )
        prepared_info = prepare_policy_inputs(raw_info, process=process, transform=transform)
        z_cur, z_goal = encode_current_goal_batch(model=model, prepared_info=prepared_info)
        action = torch.nan_to_num(prepared_info["action"][:, 0, :].detach().cpu().float(), nan=0.0)

        z_cur_parts.append(z_cur)
        z_goal_parts.append(z_goal)
        horizon_parts.append(torch.as_tensor(horizons[batch_slice], dtype=torch.long))
        action_parts.append(action)
        for row, ep, step, horizon in zip(
            sampled_rows[batch_slice],
            sampled_episodes[batch_slice],
            sampled_steps[batch_slice],
            horizons[batch_slice],
        ):
            meta.append(
                {
                    "dataset_row": int(row),
                    "episode_id": int(ep),
                    "step": int(step),
                    "goal_step": int(step + horizon),
                    "horizon": int(horizon),
                    "task": task_spec.canonical_task_name,
                    "dataset_name": task_spec.dataset_name,
                }
            )
        print(f"[batch] rows {batch_start}:{batch_end} size={batch_end - batch_start}")

    output = {
        "z_cur": torch.cat(z_cur_parts, dim=0),
        "z_goal": torch.cat(z_goal_parts, dim=0),
        "horizon": torch.cat(horizon_parts, dim=0),
        "action": torch.cat(action_parts, dim=0),
        "meta": meta,
        "build_info": {
            "task": task_spec.canonical_task_name,
            "dataset_name": task_spec.dataset_name,
            "wm_policy": args.wm_policy,
            "num_samples": int(args.num_samples),
            "max_horizon": int(args.max_horizon),
            "seed": int(args.seed),
            "action_dim": int(task_spec.action_dim),
            "pixels_key": task_spec.pixels_key,
            "action_key": task_spec.action_key,
            "episode_key": task_spec.episode_key,
            "step_key": task_spec.step_key,
        },
    }
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(
        f"[done] output={output_path} samples={output['z_cur'].shape[0]} "
        f"z_dim={output['z_cur'].shape[-1]} action_dim={output['action'].shape[-1]} "
        f"max_horizon={int(torch.max(output['horizon']).item())} time={time.time() - start_time:.2f}s"
    )


if __name__ == "__main__":
    main()
