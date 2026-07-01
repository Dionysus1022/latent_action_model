#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from diffusion.policy import DiffusionPlannerPolicy
from eval import (
    get_dataset as get_eval_dataset,
    normalize_eval_cli_args,
    resolve_corrective_config,
    resolve_diffusion_refinement_config,
    resolve_diffusion_rerank_penalty_config,
    resolve_diffusion_runtime_execute_steps,
    resolve_eval_profile_config,
    sample_eval_episode_starts,
    value_or_default,
)
from planners.build_single_peak_dataset import (
    TaskSpec,
    load_goal_conditioned_batch,
    prepare_policy_inputs,
    resolve_task_spec,
)
from planners.single_peak_data import clone_info_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile one MLP diffusion planner call, with detailed timing for "
            "candidate generation and world-model rollout scoring."
        )
    )
    parser.add_argument("--config-name", default="reacher", help="Eval config name under config/eval.")
    parser.add_argument(
        "overrides",
        nargs="*",
        help=(
            "Hydra overrides, e.g. eval_profile=diffusion "
            "+dataset_h5=/data/ykz/reacher/reacher.h5."
        ),
    )
    parser.add_argument("--repeat", type=int, default=10, help="Measured repeats after warmup.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup repeats excluded from summary.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of sampled start states per profile call.")
    parser.add_argument("--sample-seed", type=int, default=None, help="Dataset start sampler seed; defaults to cfg.seed.")
    parser.add_argument(
        "--no-full-plan",
        action="store_true",
        help="Skip timing policy.plan_actions(); still profiles generation and WM scoring stages.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON object after the text summary.",
    )
    return parser.parse_args()


def sync_cuda() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        pass


@contextlib.contextmanager
def timed_ms(bucket: dict[str, float], key: str):
    sync_cuda()
    start = time.perf_counter()
    try:
        yield
    finally:
        sync_cuda()
        bucket[key] = (time.perf_counter() - start) * 1000.0


def load_config(config_name: str, overrides: list[str]) -> DictConfig:
    config_dir = REPO_ROOT / "config" / "eval"
    hydra_args = ["profile_diffusion_mlp_timing.py", "--config-name", config_name, *overrides]
    normalized = normalize_eval_cli_args(hydra_args)
    normalized_overrides: list[str] = []
    idx = 1
    while idx < len(normalized):
        item = normalized[idx]
        if item == "--config-name":
            if idx + 1 >= len(normalized):
                raise ValueError("--config-name requires a value.")
            config_name = normalized[idx + 1]
            idx += 2
            continue
        if item.startswith("--config-name="):
            config_name = item.split("=", 1)[1]
            idx += 1
            continue
        normalized_overrides.append(item)
        idx += 1
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra.compose(config_name=config_name, overrides=normalized_overrides)
    return resolve_eval_profile_config(cfg)


def img_transform(cfg: DictConfig):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def build_process(cfg: DictConfig, dataset) -> dict[str, preprocessing.StandardScaler]:
    process = {}
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


def _optional_int(value: Any) -> int | None:
    if value in [None, "", "null"]:
        return None
    return int(value)


def build_policy(cfg: DictConfig, process: dict[str, Any], transform: dict[str, Any]) -> DiffusionPlannerPolicy:
    if str(cfg.get("planner_type", "diffusion")).lower() != "diffusion":
        raise ValueError("This profiler requires eval_profile/planner_type=diffusion.")
    diffusion_bundle = cfg.get("diffusion_bundle", None)
    if diffusion_bundle in [None, "", "null"]:
        raise ValueError("Diffusion profiling requires cfg.diffusion_bundle.")

    model = swm.policy.AutoCostModel(cfg.policy)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    corrective_cfg = resolve_corrective_config(cfg)
    refinement_cfg = resolve_diffusion_refinement_config(cfg)
    rerank_penalty_cfg = resolve_diffusion_rerank_penalty_config(cfg)
    runtime_execute_steps = resolve_diffusion_runtime_execute_steps(
        cfg.get("diffusion_runtime_execute_steps", None),
        corrective_cfg,
    )

    config = swm.PlanConfig(**OmegaConf.to_container(cfg.plan_config, resolve=True))
    return DiffusionPlannerPolicy.from_bundle(
        bundle_path=diffusion_bundle,
        world_model=model,
        config=config,
        process=process,
        transform=transform,
        map_location="cpu",
        diffusion_eta=float(value_or_default(cfg.get("diffusion_eta", None), 0.0)),
        num_candidates=_optional_int(cfg.get("diffusion_num_candidates", None)),
        truncation_steps=_optional_int(cfg.get("diffusion_truncation_steps", None)),
        start_timestep=_optional_int(cfg.get("diffusion_start_timestep", None)),
        noise_scale=float(value_or_default(cfg.get("diffusion_noise_scale", None), 1.0)),
        sampling_temperature=float(value_or_default(cfg.get("diffusion_sampling_temperature", None), 1.0)),
        selection_mode=str(value_or_default(cfg.get("diffusion_selection_mode", None), "wm_only")),
        score_topk=_optional_int(cfg.get("diffusion_score_topk", None)),
        goal_offset_steps=int(cfg.eval.goal_offset_steps),
        eval_budget=int(cfg.eval.eval_budget),
        runtime_execute_steps=runtime_execute_steps,
        corrective_enabled=False,
        refinement_enabled=bool(refinement_cfg["enabled"]),
        refinement_steps=int(refinement_cfg["steps"]),
        refinement_step_size=float(refinement_cfg["step_size"]),
        refinement_topk=refinement_cfg["topk"],
        refinement_goal_weight=float(refinement_cfg["goal_weight"]),
        refinement_prior_weight=float(refinement_cfg["prior_weight"]),
        refinement_smoothness_weight=float(refinement_cfg["smoothness_weight"]),
        refinement_grad_clip_norm=refinement_cfg["grad_clip_norm"],
        rerank_delta_weight=float(rerank_penalty_cfg["delta_weight"]),
        rerank_jerk_weight=float(rerank_penalty_cfg["jerk_weight"]),
        rerank_action_l2_weight=float(rerank_penalty_cfg["action_l2_weight"]),
        rerank_clip_weight=float(rerank_penalty_cfg["clip_weight"]),
    )


def sample_prepared_info(
    *,
    cfg: DictConfig,
    dataset,
    task_spec: TaskSpec,
    process: dict[str, Any],
    transform: dict[str, Any],
    batch_size: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    episode_ids = np.arange(len(dataset.lengths), dtype=np.int64)
    eval_episodes, start_steps, _ = sample_eval_episode_starts(
        dataset=dataset,
        ep_indices=episode_ids,
        goal_offset_steps=int(cfg.eval.goal_offset_steps),
        num_eval=int(batch_size),
        seed=int(seed),
    )
    raw_info = load_goal_conditioned_batch(
        dataset=dataset,
        task_spec=task_spec,
        episodes_idx=eval_episodes,
        start_steps=start_steps,
        goal_offset_steps=int(cfg.eval.goal_offset_steps),
        context_steps=max(int(cfg.eval.goal_offset_steps), int(task_spec.action_chunk_horizon)),
        keep_action_sequence=False,
    )
    return prepare_policy_inputs(raw_info, process=process, transform=transform)


def timed_world_model_scoring(
    policy: DiffusionPlannerPolicy,
    prepared_info: dict[str, torch.Tensor],
    candidates: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    timings: dict[str, float] = {}
    with torch.inference_mode():
        with timed_ms(timings, "wm_prepare_info_ms"):
            scoring_info = policy.prepare_info_for_scoring(prepared_info)
        with timed_ms(timings, "wm_flatten_action_blocks_ms"):
            candidate_blocks = policy.flatten_candidates_to_action_blocks(candidates)
        with timed_ms(timings, "wm_expand_info_ms"):
            expanded_info = policy.expand_prepared_info_for_candidates(
                scoring_info,
                num_candidates=int(candidates.shape[1]),
            )
        if not hasattr(policy.world_model, "rollout") or not hasattr(policy.world_model, "criterion"):
            with timed_ms(timings, "wm_get_cost_ms"):
                costs = policy.compute_world_model_costs_from_rollout(
                    expanded_info=expanded_info,
                    candidate_blocks=candidate_blocks,
                )
            timings["wm_rollout_total_ms"] = timings["wm_get_cost_ms"]
            timings["wm_goal_encode_ms"] = 0.0
            timings["wm_criterion_ms"] = 0.0
            return costs, timings

        device = next(policy.world_model.parameters()).device
        with timed_ms(timings, "wm_to_device_ms"):
            rollout_info = clone_info_dict(expanded_info)
            for key, value in rollout_info.items():
                if torch.is_tensor(value):
                    rollout_info[key] = value.to(device)
            candidate_blocks = candidate_blocks.to(device)

        with timed_ms(timings, "wm_goal_encode_ms"):
            goal_info = {key: value[:, 0] for key, value in rollout_info.items() if torch.is_tensor(value)}
            if "goal" not in goal_info:
                raise KeyError("expanded_info must contain a 'goal' tensor for world-model scoring.")
            goal_info["pixels"] = goal_info["goal"]
            for key in list(goal_info.keys()):
                if key.startswith("goal_"):
                    goal_info[key[len("goal_") :]] = goal_info.pop(key)
            goal_info.pop("action", None)
            encoded_goal = policy.world_model.encode(goal_info)
            goal_emb = encoded_goal["emb"]
            goal_emb = goal_emb.unsqueeze(1).expand(
                int(candidate_blocks.shape[0]),
                int(candidate_blocks.shape[1]),
                int(goal_emb.shape[1]),
                int(goal_emb.shape[2]),
            )
            rollout_info["goal_emb"] = goal_emb

        with timed_ms(timings, "wm_rollout_total_ms"):
            rollout_outputs = policy.world_model.rollout(rollout_info, candidate_blocks)
        with timed_ms(timings, "wm_criterion_ms"):
            costs = policy.world_model.criterion(rollout_outputs)
    return costs, timings


def profile_once(
    policy: DiffusionPlannerPolicy,
    prepared_info: dict[str, torch.Tensor],
    *,
    include_full_plan: bool,
) -> dict[str, float]:
    timings: dict[str, float] = {}
    with torch.inference_mode():
        with timed_ms(timings, "encode_current_goal_ms"):
            z_cur, z_goal = policy.encode_current_goal(prepared_info)
        with timed_ms(timings, "diffusion_generate_candidates_ms"):
            generation = policy.planner.generate_candidates(
                z_cur,
                z_goal,
                eta=policy.diffusion_eta,
                truncation_steps=policy.proposal_truncation_steps,
                start_timestep=policy.proposal_start_timestep,
                noise_scale=policy.proposal_noise_scale,
                sampling_temperature=policy.proposal_sampling_temperature,
                return_intermediates=False,
            )
        candidates = generation["candidates"]
        model_scores = generation["score_logits"]
        with timed_ms(timings, "candidate_concat_ms"):
            if policy.proposal_rounds > 1:
                round_candidates = [candidates]
                round_scores = [model_scores]
                for _ in range(policy.proposal_rounds - 1):
                    outputs = policy.planner.generate_candidates(
                        z_cur,
                        z_goal,
                        eta=policy.diffusion_eta,
                        truncation_steps=policy.proposal_truncation_steps,
                        start_timestep=policy.proposal_start_timestep,
                        noise_scale=policy.proposal_noise_scale,
                        sampling_temperature=policy.proposal_sampling_temperature,
                        return_intermediates=False,
                    )
                    round_candidates.append(outputs["candidates"])
                    round_scores.append(outputs["score_logits"])
                candidates = torch.cat(round_candidates, dim=1)
                model_scores = torch.cat(round_scores, dim=1)

        score_topk_indices = None
        score_topk_candidates = None
        score_topk_scores = None
        if policy.selection_mode == "score_topk_wm":
            topk = policy._resolve_score_topk(int(candidates.shape[1]))
            with timed_ms(timings, "score_topk_ms"):
                score_topk_indices = torch.topk(model_scores, k=topk, dim=-1, largest=True).indices
            with timed_ms(timings, "topk_gather_ms"):
                gather_index = score_topk_indices.unsqueeze(-1).expand(-1, -1, int(candidates.shape[-1]))
                score_topk_candidates = candidates.gather(1, gather_index)
                score_topk_scores = model_scores.gather(1, score_topk_indices)
            score_candidates_for_wm = score_topk_candidates
            scores_for_selection = score_topk_scores
        else:
            timings["score_topk_ms"] = 0.0
            timings["topk_gather_ms"] = 0.0
            score_candidates_for_wm = candidates
            scores_for_selection = model_scores

        world_model_costs, wm_timings = timed_world_model_scoring(policy, prepared_info, score_candidates_for_wm)
        timings.update(wm_timings)
        with timed_ms(timings, "selection_ms"):
            selected_candidates, selected_indices, _ = policy.select_best_candidates(
                score_candidates_for_wm,
                world_model_costs,
                scores_for_selection,
            )
            if score_topk_indices is not None:
                _ = score_topk_indices.gather(1, selected_indices.view(-1, 1)).squeeze(1)
        with timed_ms(timings, "unflatten_selected_plan_ms"):
            _ = selected_candidates.reshape(
                int(selected_candidates.shape[0]),
                policy.plan_horizon,
                policy.action_dim,
            )
        if include_full_plan:
            with timed_ms(timings, "full_policy_plan_actions_ms"):
                _ = policy.plan_actions(prepared_info)

    batch_size = int(score_candidates_for_wm.shape[0])
    num_candidates = int(score_candidates_for_wm.shape[1])
    denoise_steps = int(policy.runtime_truncation_steps)
    timings["batch_size"] = float(batch_size)
    timings["num_candidates"] = float(num_candidates)
    timings["candidate_rollouts"] = float(batch_size * num_candidates)
    timings["denoise_steps"] = float(denoise_steps)
    timings["diffusion_per_denoise_step_ms"] = timings["diffusion_generate_candidates_ms"] / max(1, denoise_steps)
    timings["wm_rollout_per_candidate_ms"] = timings["wm_rollout_total_ms"] / max(1, batch_size * num_candidates)
    timings["wm_scoring_total_ms"] = (
        timings.get("wm_prepare_info_ms", 0.0)
        + timings.get("wm_flatten_action_blocks_ms", 0.0)
        + timings.get("wm_expand_info_ms", 0.0)
        + timings.get("wm_to_device_ms", 0.0)
        + timings.get("wm_goal_encode_ms", 0.0)
        + timings.get("wm_rollout_total_ms", 0.0)
        + timings.get("wm_criterion_ms", 0.0)
        + timings.get("wm_get_cost_ms", 0.0)
    )
    timings["manual_total_without_full_plan_ms"] = (
        timings["encode_current_goal_ms"]
        + timings["diffusion_generate_candidates_ms"]
        + timings["candidate_concat_ms"]
        + timings["wm_scoring_total_ms"]
        + timings["selection_ms"]
        + timings["unflatten_selected_plan_ms"]
    )
    return timings


def summarize(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for row in rows for key in row.keys()})
    summary = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
        if not values:
            continue
        summary[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def main() -> int:
    args = parse_args()
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    cfg = load_config(args.config_name, args.overrides)
    dataset = get_eval_dataset(cfg, cfg.eval.dataset_name)
    task_args = argparse.Namespace(task=args.config_name, config=None)
    task_spec = resolve_task_spec(task_args, cfg, dataset)
    process = build_process(cfg, dataset)
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}
    policy = build_policy(cfg, process=process, transform=transform)

    seed = int(cfg.seed) if args.sample_seed is None else int(args.sample_seed)
    prepared_info = sample_prepared_info(
        cfg=cfg,
        dataset=dataset,
        task_spec=task_spec,
        process=process,
        transform=transform,
        batch_size=int(args.batch_size),
        seed=seed,
    )

    print(
        "[profile-config] "
        f"task={task_spec.canonical_task_name} policy={cfg.policy} bundle={cfg.diffusion_bundle} "
        f"batch_size={args.batch_size} num_candidates={policy.effective_num_candidates} "
        f"denoiser={policy.planner.denoiser_type} denoise_steps={policy.runtime_truncation_steps} "
        f"start_timestep={policy.runtime_start_timestep} action_chunk_horizon={policy.action_chunk_horizon} "
        f"action_dim={policy.action_dim} full_plan={int(not args.no_full_plan)}"
    )

    for idx in range(args.warmup):
        _ = profile_once(policy, prepared_info, include_full_plan=not args.no_full_plan)
        print(f"[warmup] {idx + 1}/{args.warmup}")

    rows = []
    for idx in range(args.repeat):
        row = profile_once(policy, prepared_info, include_full_plan=not args.no_full_plan)
        rows.append(row)
        print(
            "[profile] "
            f"{idx + 1}/{args.repeat} "
            f"manual_total={row['manual_total_without_full_plan_ms']:.3f}ms "
            f"diffusion={row['diffusion_generate_candidates_ms']:.3f}ms "
            f"wm_scoring={row['wm_scoring_total_ms']:.3f}ms "
            f"wm_rollout={row['wm_rollout_total_ms']:.3f}ms "
            f"wm_rollout_per_candidate={row['wm_rollout_per_candidate_ms']:.6f}ms"
        )

    summary = summarize(rows)
    ordered_keys = [
        "manual_total_without_full_plan_ms",
        "full_policy_plan_actions_ms",
        "encode_current_goal_ms",
        "diffusion_generate_candidates_ms",
        "diffusion_per_denoise_step_ms",
        "candidate_concat_ms",
        "wm_scoring_total_ms",
        "wm_prepare_info_ms",
        "wm_flatten_action_blocks_ms",
        "wm_expand_info_ms",
        "wm_to_device_ms",
        "wm_goal_encode_ms",
        "wm_rollout_total_ms",
        "wm_rollout_per_candidate_ms",
        "wm_criterion_ms",
        "selection_ms",
        "unflatten_selected_plan_ms",
    ]
    print("[summary]")
    for key in ordered_keys:
        if key not in summary:
            continue
        stats = summary[key]
        print(
            f"{key}: mean={stats['mean']:.6f}ms std={stats['std']:.6f}ms "
            f"min={stats['min']:.6f}ms max={stats['max']:.6f}ms"
        )

    if args.json:
        import json

        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
