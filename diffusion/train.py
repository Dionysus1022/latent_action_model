from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from diffusion.anchors import ActionAnchorBundle, load_anchor_bundle
from diffusion.model import (
    DiffusionPlannerModel,
    DiffusionPlannerModelConfig,
    infer_model_config_from_dataset_and_anchor_bundle,
    save_diffusion_planner_bundle,
)
from planners.latent_rollout import latent_rollout


TASK_ALIASES = {
    "pusht": "pusht",
    "tworoom": "tworoom",
    "two-room": "tworoom",
    "two_room": "tworoom",
    "reacher": "reacher",
    "researcher": "reacher",
}


class DiffusionPlannerTensorDataset(Dataset):
    """Dataset wrapper for saved latent/action planner `.pt` bundles.

    Expected top-level schema:
        z_cur: [N, latent_dim]
        z_goal: [N, latent_dim]
        teacher_plan: [N, action_chunk_dim]
        meta: list[dict]
    """

    def __init__(self, dataset_bundle: dict[str, Any]):
        required_keys = {"z_cur", "z_goal", "teacher_plan", "meta"}
        missing = required_keys.difference(dataset_bundle.keys())
        if missing:
            raise KeyError(f"Dataset bundle is missing required keys: {sorted(missing)}.")

        self.z_cur = dataset_bundle["z_cur"].float()  # [N, latent_dim]
        self.z_goal = dataset_bundle["z_goal"].float()  # [N, latent_dim]
        self.teacher_plan = dataset_bundle["teacher_plan"].float()  # [N, action_chunk_dim]
        self.meta = dataset_bundle["meta"]
        self.build_info = dataset_bundle.get("build_info", {})

        if self.z_cur.ndim != 2:
            raise ValueError(f"z_cur must have shape [N, latent_dim], got {tuple(self.z_cur.shape)}.")
        if self.z_goal.ndim != 2:
            raise ValueError(f"z_goal must have shape [N, latent_dim], got {tuple(self.z_goal.shape)}.")
        if self.teacher_plan.ndim != 2:
            raise ValueError(
                f"teacher_plan must have shape [N, action_chunk_dim], got {tuple(self.teacher_plan.shape)}."
            )
        if self.z_cur.shape != self.z_goal.shape:
            raise ValueError(
                f"z_cur and z_goal must have the same shape, got {tuple(self.z_cur.shape)} and {tuple(self.z_goal.shape)}."
            )
        if self.z_cur.shape[0] != self.teacher_plan.shape[0]:
            raise ValueError(
                "Dataset sample count mismatch between z_cur and teacher_plan: "
                f"{self.z_cur.shape[0]} != {self.teacher_plan.shape[0]}."
            )
        if len(self.meta) != int(self.z_cur.shape[0]):
            raise ValueError(
                f"meta length {len(self.meta)} must match dataset size {self.z_cur.shape[0]}."
            )

    def __len__(self) -> int:
        return int(self.z_cur.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "z_cur": self.z_cur[index].clone(),  # [latent_dim]
            "z_goal": self.z_goal[index].clone(),  # [latent_dim]
            "teacher_plan": self.teacher_plan[index].clone(),  # [action_chunk_dim]
            "meta": dict(self.meta[index]),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train_diffusion_planner.py",
        description="Train the anchor-conditioned truncated diffusion planner on latent/action datasets.",
    )
    parser.add_argument("--dataset-path", required=True, help="Path to a built planner dataset `.pt` file.")
    parser.add_argument(
        "--val-dataset-path",
        default=None,
        help="Optional separate validation dataset `.pt`. If omitted, the train dataset is split.",
    )
    parser.add_argument(
        "--anchor-bundle-path",
        required=True,
        help="Path to an action anchor bundle `.pt` file.",
    )
    parser.add_argument(
        "--wm-policy",
        default=None,
        help="Optional LeWM checkpoint path. Required when --goal-loss-weight > 0.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for logs and saved runtime bundles.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Train batch size.")
    parser.add_argument("--val-batch-size", type=int, default=256, help="Validation batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio when no val dataset is provided.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--log-every", type=int, default=50, help="Train-step logging frequency within an epoch.")
    parser.add_argument(
        "--loss-preset",
        choices=["legacy", "simple_bce"],
        default="legacy",
        help="Optional loss preset. `legacy` preserves the current multi-term objective; `simple_bce` simplifies it.",
    )
    parser.add_argument("--hidden-dim", type=int, default=512, help="Planner hidden dimension.")
    parser.add_argument("--num-layers", type=int, default=3, help="Condition trunk depth.")
    parser.add_argument("--fusion-num-layers", type=int, default=2, help="Fusion trunk depth.")
    parser.add_argument("--dropout", type=float, default=0.0, help="MLP dropout.")
    parser.add_argument(
        "--activation",
        choices=["gelu", "relu", "silu"],
        default="gelu",
        help="MLP activation.",
    )
    parser.add_argument(
        "--timestep-embedding-dim",
        type=int,
        default=128,
        help="Sinusoidal timestep embedding width.",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=16,
        help="Total diffusion schedule length.",
    )
    parser.add_argument(
        "--truncation-steps",
        type=int,
        default=4,
        help="Truncated reverse diffusion steps.",
    )
    parser.add_argument(
        "--start-timestep",
        type=int,
        default=None,
        help="Optional diffusion start timestep. Defaults to num_train_steps - 1.",
    )
    parser.add_argument(
        "--beta-schedule",
        choices=["linear"],
        default="linear",
        help="Diffusion beta schedule.",
    )
    parser.add_argument("--beta-start", type=float, default=1e-4, help="Diffusion beta start.")
    parser.add_argument("--beta-end", type=float, default=2e-2, help="Diffusion beta end.")
    parser.add_argument(
        "--timestep-sampling",
        choices=["truncation", "full"],
        default="truncation",
        help="Sample training timesteps only from truncation grid or from the full schedule.",
    )
    parser.add_argument(
        "--rec-loss",
        choices=["smooth_l1", "l1"],
        default="smooth_l1",
        help="Reconstruction loss on the positive anchor candidate.",
    )
    parser.add_argument(
        "--cls-loss-type",
        choices=["ce", "bce", "ce_bce"],
        default="ce",
        help="Classification loss type on score_logits [B, K].",
    )
    parser.add_argument(
        "--cls-loss-weight",
        type=float,
        default=1.0,
        help="Weight for CE classification loss. Used when --cls-loss-type is `ce` or `ce_bce`.",
    )
    parser.add_argument(
        "--bce-weight",
        type=float,
        default=1.0,
        help="Weight for BCE positive/negative anchor classification loss.",
    )
    parser.add_argument(
        "--bce-pos-topk",
        type=int,
        default=1,
        help="Top-k nearest anchors marked positive in BCE targets [B, K].",
    )
    parser.add_argument(
        "--rec-loss-weight",
        type=float,
        default=1.0,
        help="Weight for positive-candidate reconstruction loss.",
    )
    parser.add_argument(
        "--goal-loss-weight",
        type=float,
        default=0.0,
        help="Optional latent-only rollout goal-consistency loss weight.",
    )
    parser.add_argument(
        "--goal-loss-history-size",
        type=int,
        default=3,
        help="History window used by world_model.predict(...) inside latent-only rollout.",
    )
    parser.add_argument(
        "--goal-loss-receding-horizon",
        type=int,
        default=None,
        help="Optional override for rollout block count R. Defaults to dataset/anchor plan_config.receding_horizon.",
    )
    parser.add_argument(
        "--goal-loss-action-block",
        type=int,
        default=None,
        help="Optional override for action_block. Defaults to dataset/anchor plan_config.action_block.",
    )
    parser.add_argument(
        "--enable-goal-pool-loss",
        action="store_true",
        help="Enable softmin rollout goal loss over a small candidate pool.",
    )
    parser.add_argument(
        "--goal-pool-weight",
        type=float,
        default=0.003,
        help="Weight for the softmin goal pool loss.",
    )
    parser.add_argument(
        "--goal-pool-topk",
        type=int,
        default=3,
        help="Total number of candidates M used by the goal pool rollout loss, including the positive candidate.",
    )
    parser.add_argument(
        "--goal-pool-tau",
        type=float,
        default=0.5,
        help="Softmin temperature for the rollout goal pool loss.",
    )
    parser.add_argument(
        "--goal-pool-candidate-source",
        choices=["score", "nearest"],
        default="score",
        help="How to choose the extra non-positive candidates for the goal pool rollout loss.",
    )
    parser.add_argument(
        "--aux-rec-topk",
        type=int,
        default=2,
        help="Nearest-anchor top-k used by the auxiliary multi-candidate reconstruction loss.",
    )
    parser.add_argument(
        "--aux-rec-weight",
        type=float,
        default=0.25,
        help="Weight for the auxiliary top-k candidate reconstruction loss.",
    )
    parser.add_argument(
        "--aux-rec-temperature",
        type=float,
        default=1.0,
        help="Temperature for distance-based weights in the auxiliary top-k reconstruction loss.",
    )
    parser.add_argument(
        "--score-ranking-weight",
        type=float,
        default=0.1,
        help="Weight for the soft anchor-distance ranking consistency loss on score logits.",
    )
    parser.add_argument(
        "--score-ranking-temperature",
        type=float,
        default=1.0,
        help="Temperature for the soft anchor-distance ranking targets.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm.",
    )
    return parser.parse_args(argv)


def apply_loss_preset(args: argparse.Namespace) -> dict[str, bool]:
    """Mutate args in-place to realize high-level loss presets."""
    if args.loss_preset == "simple_bce":
        args.cls_loss_type = "bce"
        args.cls_loss_weight = 0.0
        args.aux_rec_weight = 0.0
        args.score_ranking_weight = 0.0
        args.enable_goal_pool_loss = False
        args.goal_pool_weight = 0.0

    return {
        "aux_rec_enabled": float(args.aux_rec_weight) > 0.0,
        "score_rank_enabled": float(args.score_ranking_weight) > 0.0,
        "goal_pool_enabled": bool(args.enable_goal_pool_loss) and float(args.goal_pool_weight) > 0.0,
        "wm_rank_enabled": False,
        "diversity_enabled": False,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset_bundle(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset bundle not found: {dataset_path}")
    return torch.load(dataset_path, map_location="cpu")


def normalize_task_name(task_name: str | None) -> str | None:
    if task_name in [None, "", "null"]:
        return None
    normalized = str(task_name).strip().lower()
    return TASK_ALIASES.get(normalized, normalized)


def maybe_get_nested(source: dict[str, Any] | None, path: list[str]) -> Any:
    current: Any = source
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def maybe_positive_int(value: Any) -> int | None:
    if value in [None, "", "null"]:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"Expected a positive integer, got {parsed}.")
    return parsed


def infer_dataset_runtime_spec(dataset_bundle: dict[str, Any]) -> dict[str, Any]:
    build_info = dataset_bundle.get("build_info", {})
    meta = dataset_bundle.get("meta", [])
    meta0 = meta[0] if isinstance(meta, list) and len(meta) > 0 and isinstance(meta[0], dict) else {}
    teacher_plan = dataset_bundle.get("teacher_plan")
    action_chunk_dim = None
    if torch.is_tensor(teacher_plan):
        if teacher_plan.ndim != 2:
            raise ValueError(
                "dataset_bundle['teacher_plan'] must have shape [N, action_chunk_dim], "
                f"got {tuple(teacher_plan.shape)}."
            )
        action_chunk_dim = int(teacher_plan.shape[-1])  # [N, action_chunk_dim]

    task = normalize_task_name(
        build_info.get("task")
        or build_info.get("requested_task")
        or meta0.get("task")
    )
    action_dim = maybe_positive_int(
        build_info.get("action_dim")
        or maybe_get_nested(build_info, ["task_spec", "action_dim"])
        or meta0.get("action_dim")
    )
    action_chunk_horizon = maybe_positive_int(
        build_info.get("action_chunk_horizon")
        or maybe_get_nested(build_info, ["task_spec", "action_chunk_horizon"])
        or meta0.get("plan_horizon")
    )
    receding_horizon = maybe_positive_int(
        maybe_get_nested(build_info, ["plan_config", "receding_horizon"])
        or maybe_get_nested(build_info, ["task_spec", "receding_horizon"])
        or meta0.get("receding_horizon")
    )
    action_block = maybe_positive_int(
        maybe_get_nested(build_info, ["plan_config", "action_block"])
        or maybe_get_nested(build_info, ["task_spec", "action_block"])
        or meta0.get("action_block")
    )

    if action_chunk_horizon is None and action_dim is not None and action_chunk_dim is not None:
        if action_chunk_dim % int(action_dim) == 0:
            action_chunk_horizon = int(action_chunk_dim // int(action_dim))
    if action_dim is None and action_chunk_horizon is not None and action_chunk_dim is not None:
        if action_chunk_dim % int(action_chunk_horizon) == 0:
            action_dim = int(action_chunk_dim // int(action_chunk_horizon))

    return {
        "task": task,
        "action_dim": action_dim,
        "action_chunk_horizon": action_chunk_horizon,
        "action_chunk_dim": action_chunk_dim,
        "receding_horizon": receding_horizon,
        "action_block": action_block,
    }


def validate_anchor_dataset_compatibility(
    *,
    dataset_bundle: dict[str, Any],
    anchor_bundle: ActionAnchorBundle,
    dataset_label: str,
) -> dict[str, Any]:
    dataset_spec = infer_dataset_runtime_spec(dataset_bundle)
    anchor_task = normalize_task_name(anchor_bundle.task)
    dataset_task = normalize_task_name(dataset_spec["task"])

    print(
        f"[compat] dataset={dataset_label} dataset_task={dataset_task or 'unknown'} "
        f"anchor_task={anchor_task or 'unknown'} dataset_action_dim={dataset_spec['action_dim']} "
        f"anchor_action_dim={anchor_bundle.action_dim} "
        f"dataset_action_chunk_horizon={dataset_spec['action_chunk_horizon']} "
        f"anchor_action_chunk_horizon={anchor_bundle.action_chunk_horizon} "
        f"dataset_action_chunk_dim={dataset_spec['action_chunk_dim']} "
        f"anchor_action_chunk_dim={anchor_bundle.action_chunk_dim} "
        f"dataset_receding_horizon={dataset_spec['receding_horizon']} "
        f"anchor_receding_horizon={anchor_bundle.receding_horizon} "
        f"dataset_action_block={dataset_spec['action_block']} "
        f"anchor_action_block={anchor_bundle.action_block}"
    )

    if dataset_task is not None and anchor_task is not None and dataset_task != anchor_task:
        raise ValueError(
            f"Anchor bundle task '{anchor_task}' does not match {dataset_label} task '{dataset_task}'."
        )
    if dataset_task is None or anchor_task is None:
        print(
            f"[warn] Could not fully verify task compatibility for {dataset_label}: "
            f"dataset_task={dataset_task}, anchor_task={anchor_task}."
        )

    comparable_fields = [
        ("action_dim", dataset_spec["action_dim"], anchor_bundle.action_dim),
        ("action_chunk_horizon", dataset_spec["action_chunk_horizon"], anchor_bundle.action_chunk_horizon),
        ("action_chunk_dim", dataset_spec["action_chunk_dim"], anchor_bundle.action_chunk_dim),
        ("receding_horizon", dataset_spec["receding_horizon"], anchor_bundle.receding_horizon),
        ("action_block", dataset_spec["action_block"], anchor_bundle.action_block),
    ]
    for field_name, dataset_value, anchor_value in comparable_fields:
        if dataset_value is None or anchor_value is None:
            continue
        if int(dataset_value) != int(anchor_value):
            raise ValueError(
                f"Anchor bundle {field_name}={anchor_value} does not match {dataset_label} {field_name}={dataset_value}."
            )
    return dataset_spec


def build_model_from_dataset_and_anchor_bundle(
    dataset_bundle: dict[str, Any],
    anchor_bundle: ActionAnchorBundle,
    args: argparse.Namespace,
) -> DiffusionPlannerModel:
    model_cfg = infer_model_config_from_dataset_and_anchor_bundle(
        dataset_bundle,
        anchor_bundle,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        activation=args.activation,
        timestep_embedding_dim=args.timestep_embedding_dim,
        fusion_num_layers=args.fusion_num_layers,
        num_train_steps=args.num_train_steps,
        truncation_steps=args.truncation_steps,
        start_timestep=args.start_timestep,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )
    return DiffusionPlannerModel.from_anchor_bundle(model_cfg, anchor_bundle)


def _get_plan_config_dict(source: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(source, dict):
        return None
    plan_config = source.get("plan_config")
    if isinstance(plan_config, dict):
        return plan_config
    return None


def infer_goal_loss_rollout_shape(
    *,
    args: argparse.Namespace,
    dataset_bundle: dict[str, Any],
    anchor_bundle: ActionAnchorBundle,
    plan_horizon: int,
) -> tuple[int, int]:
    """Infer rollout block shape for latent-only goal loss.

    returns:
        receding_horizon: R
        action_block: number of env-step actions per predictor block
    """
    receding_horizon = args.goal_loss_receding_horizon
    action_block = args.goal_loss_action_block

    if receding_horizon is None or action_block is None:
        if receding_horizon is None and anchor_bundle.receding_horizon is not None:
            receding_horizon = int(anchor_bundle.receding_horizon)
        if action_block is None and anchor_bundle.action_block is not None:
            action_block = int(anchor_bundle.action_block)

    if receding_horizon is None or action_block is None:
        candidate_plan_configs: list[dict[str, Any]] = []
        dataset_plan_config = _get_plan_config_dict(dataset_bundle.get("build_info", {}))
        if dataset_plan_config is not None:
            candidate_plan_configs.append(dataset_plan_config)
        anchor_source_build_info = anchor_bundle.metadata.get("source_build_info")
        anchor_plan_config = _get_plan_config_dict(anchor_source_build_info)
        if anchor_plan_config is not None:
            candidate_plan_configs.append(anchor_plan_config)

        for plan_config in candidate_plan_configs:
            if receding_horizon is None and "receding_horizon" in plan_config:
                receding_horizon = int(plan_config["receding_horizon"])
            if action_block is None and "action_block" in plan_config:
                action_block = int(plan_config["action_block"])
            if receding_horizon is not None and action_block is not None:
                break

    if receding_horizon is None and action_block is not None:
        if plan_horizon % int(action_block) != 0:
            raise ValueError(
                "Cannot infer goal-loss receding_horizon because plan_horizon is not divisible by action_block: "
                f"{plan_horizon} % {action_block} != 0."
            )
        receding_horizon = int(plan_horizon // int(action_block))
    if action_block is None and receding_horizon is not None:
        if plan_horizon % int(receding_horizon) != 0:
            raise ValueError(
                "Cannot infer goal-loss action_block because plan_horizon is not divisible by receding_horizon: "
                f"{plan_horizon} % {receding_horizon} != 0."
            )
        action_block = int(plan_horizon // int(receding_horizon))

    if receding_horizon is None or action_block is None:
        raise ValueError(
            "Could not infer latent rollout block shape for goal loss. Provide "
            "--goal-loss-receding-horizon and --goal-loss-action-block, or build the dataset with build_info.plan_config."
        )
    if int(receding_horizon) <= 0 or int(action_block) <= 0:
        raise ValueError(
            f"Goal-loss rollout shape must be positive, got receding_horizon={receding_horizon}, "
            f"action_block={action_block}."
        )
    if int(receding_horizon) * int(action_block) != int(plan_horizon):
        raise ValueError(
            "Goal-loss rollout shape does not match planner action chunk horizon: "
            f"{receding_horizon} * {action_block} != {plan_horizon}."
        )
    return int(receding_horizon), int(action_block)


def reshape_flat_actions_to_rollout_blocks(
    flat_actions: torch.Tensor,
    *,
    plan_horizon: int,
    action_dim: int,
    receding_horizon: int,
    action_block: int,
) -> torch.Tensor:
    """Convert flat action chunks to predictor rollout blocks.

    flat_actions:
        [B, action_chunk_dim] or [B, S, action_chunk_dim]

    returns:
        action_blocks: [B, S, receding_horizon, action_block * action_dim]
    """
    if not torch.is_tensor(flat_actions):
        raise TypeError(f"flat_actions must be a torch.Tensor, got {type(flat_actions)}.")
    squeezed_candidates = False
    if flat_actions.ndim == 2:
        flat_actions = flat_actions.unsqueeze(1)  # [B, 1, action_chunk_dim]
        squeezed_candidates = True
    if flat_actions.ndim != 3:
        raise ValueError(
            "flat_actions must have shape [B, D] or [B, S, D], "
            f"got {tuple(flat_actions.shape)}."
        )

    expected_dim = int(plan_horizon * action_dim)
    if int(flat_actions.shape[-1]) != expected_dim:
        raise ValueError(
            f"flat_actions width {flat_actions.shape[-1]} does not match plan_horizon * action_dim = {expected_dim}."
        )
    if int(receding_horizon * action_block) != int(plan_horizon):
        raise ValueError(
            "receding_horizon * action_block must equal plan_horizon for rollout blocks: "
            f"{receding_horizon} * {action_block} != {plan_horizon}."
        )

    candidate_steps = flat_actions.reshape(
        int(flat_actions.shape[0]),
        int(flat_actions.shape[1]),
        int(plan_horizon),
        int(action_dim),
    )  # [B, S, plan_horizon, action_dim]
    action_blocks = candidate_steps.reshape(
        int(flat_actions.shape[0]),
        int(flat_actions.shape[1]),
        int(receding_horizon),
        int(action_block * action_dim),
    )  # [B, S, receding_horizon, action_block * action_dim]
    if squeezed_candidates:
        return action_blocks  # [B, 1, receding_horizon, action_block * action_dim]
    return action_blocks


def load_frozen_world_model(policy_path: str, device: torch.device) -> torch.nn.Module:
    """Load a frozen LeWM world model for latent-only rollout goal loss."""
    world_model = swm.policy.AutoCostModel(policy_path)
    world_model = world_model.to(device)
    world_model.eval()
    world_model.requires_grad_(False)
    return world_model


def build_reconstruction_loss_fn(loss_name: str) -> nn.Module:
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss()
    if loss_name == "l1":
        return nn.L1Loss()
    raise ValueError(f"Unsupported reconstruction loss '{loss_name}'.")


def split_train_val(
    dataset: Dataset,
    val_split: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    if not (0.0 < val_split < 1.0):
        raise ValueError(f"val_split must be in (0, 1), got {val_split}.")
    dataset_len = len(dataset)
    val_len = max(1, int(math.floor(dataset_len * val_split)))
    train_len = dataset_len - val_len
    if train_len <= 0:
        raise ValueError(
            f"Dataset size {dataset_len} is too small for val_split={val_split}; train_len became {train_len}."
        )
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)
    return train_set, val_set


def build_dataloaders(
    train_set: Dataset,
    val_set: Dataset,
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def sample_timestep_grid(
    model: DiffusionPlannerModel,
    *,
    batch_size: int,
    device: torch.device,
    sampling: str,
) -> torch.Tensor:
    """Sample training timestep ids for all candidates.

    returns:
        timestep_grid: [B, K]
    """
    if sampling == "truncation":
        choices = model.schedule.truncation_timesteps.to(device=device)  # [T_trunc]
        random_ids = torch.randint(
            low=0,
            high=int(choices.shape[0]),
            size=(int(batch_size),),
            device=device,
        )  # [B]
        timestep_per_sample = choices.index_select(0, random_ids).view(int(batch_size), 1)  # [B, 1]
        return timestep_per_sample.expand(-1, model.num_anchors)  # [B, K]

    if sampling == "full":
        timestep_per_sample = torch.randint(
            low=0,
            high=model.num_train_steps,
            size=(int(batch_size), 1),
            device=device,
        )  # [B, 1]
        return timestep_per_sample.expand(-1, model.num_anchors)  # [B, K]

    raise ValueError(f"Unsupported timestep sampling mode '{sampling}'.")


def build_eval_timestep_grid(
    model: DiffusionPlannerModel,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Use the highest truncation timestep for deterministic validation."""
    timestep = int(model.schedule.truncation_timesteps[0].item())
    return torch.full(
        (int(batch_size), model.num_anchors),
        timestep,
        device=device,
        dtype=torch.long,
    )  # [B, K]


def assign_positive_anchors(
    teacher_plan: torch.Tensor,
    anchors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign the closest anchor to each teacher action chunk.

    teacher_plan: [B, action_chunk_dim]
    anchors: [K, action_chunk_dim]

    returns:
        positive_anchor_indices: [B]
        squared_distances: [B, K]
    """
    if teacher_plan.ndim != 2:
        raise ValueError(f"teacher_plan must have shape [B, D], got {tuple(teacher_plan.shape)}.")
    if anchors.ndim != 2:
        raise ValueError(f"anchors must have shape [K, D], got {tuple(anchors.shape)}.")
    if int(teacher_plan.shape[-1]) != int(anchors.shape[-1]):
        raise ValueError(
            f"teacher_plan dim {teacher_plan.shape[-1]} does not match anchor dim {anchors.shape[-1]}."
        )

    squared_distances = (teacher_plan.unsqueeze(1) - anchors.unsqueeze(0)).square().sum(dim=-1)  # [B, K]
    positive_anchor_indices = torch.argmin(squared_distances, dim=-1)  # [B]
    return positive_anchor_indices, squared_distances


def gather_positive_candidates(
    refined_actions: torch.Tensor,
    positive_anchor_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather the positive candidate action chunk from [B, K, D].

    returns:
        positive_candidates: [B, action_chunk_dim]
    """
    if refined_actions.ndim != 3:
        raise ValueError(f"refined_actions must have shape [B, K, D], got {tuple(refined_actions.shape)}.")
    if positive_anchor_indices.ndim != 1:
        raise ValueError(
            f"positive_anchor_indices must have shape [B], got {tuple(positive_anchor_indices.shape)}."
        )
    if int(refined_actions.shape[0]) != int(positive_anchor_indices.shape[0]):
        raise ValueError(
            "Batch size mismatch between refined_actions and positive_anchor_indices: "
            f"{refined_actions.shape[0]} != {positive_anchor_indices.shape[0]}."
        )

    gather_index = positive_anchor_indices.view(-1, 1, 1).expand(-1, 1, refined_actions.shape[-1])  # [B, 1, D]
    return refined_actions.gather(1, gather_index).squeeze(1)  # [B, D]


def gather_candidate_batch(
    refined_actions: torch.Tensor,
    candidate_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather multiple candidate action chunks from [B, K, D].

    refined_actions: [B, K, action_chunk_dim]
    candidate_indices: [B, M]
    returns:
        gathered_candidates: [B, M, action_chunk_dim]
    """
    if refined_actions.ndim != 3:
        raise ValueError(f"refined_actions must have shape [B, K, D], got {tuple(refined_actions.shape)}.")
    if candidate_indices.ndim != 2:
        raise ValueError(
            f"candidate_indices must have shape [B, M], got {tuple(candidate_indices.shape)}."
        )
    if int(refined_actions.shape[0]) != int(candidate_indices.shape[0]):
        raise ValueError(
            "Batch size mismatch between refined_actions and candidate_indices: "
            f"{refined_actions.shape[0]} != {candidate_indices.shape[0]}."
        )
    gather_index = candidate_indices.unsqueeze(-1).expand(-1, -1, refined_actions.shape[-1])  # [B, M, D]
    return refined_actions.gather(1, gather_index)  # [B, M, action_chunk_dim]


def select_goal_pool_candidate_indices(
    *,
    positive_anchor_indices: torch.Tensor,
    score_logits: torch.Tensor,
    squared_anchor_distances: torch.Tensor,
    topk: int,
    source: str,
) -> torch.Tensor:
    """Select a small candidate pool for softmin rollout goal loss.

    The positive candidate is always included. The remaining M - 1 entries are
    filled from either:
        - score logits ranking
        - nearest-anchor distance ranking

    Inputs:
        positive_anchor_indices: [B]
        score_logits: [B, K]
        squared_anchor_distances: [B, K]

    returns:
        candidate_indices: [B, M]
    """
    if positive_anchor_indices.ndim != 1:
        raise ValueError(
            f"positive_anchor_indices must have shape [B], got {tuple(positive_anchor_indices.shape)}."
        )
    if score_logits.ndim != 2:
        raise ValueError(f"score_logits must have shape [B, K], got {tuple(score_logits.shape)}.")
    if squared_anchor_distances.ndim != 2:
        raise ValueError(
            f"squared_anchor_distances must have shape [B, K], got {tuple(squared_anchor_distances.shape)}."
        )
    if int(score_logits.shape[0]) != int(positive_anchor_indices.shape[0]):
        raise ValueError(
            "Batch size mismatch between score_logits and positive_anchor_indices: "
            f"{score_logits.shape[0]} != {positive_anchor_indices.shape[0]}."
        )
    if tuple(score_logits.shape) != tuple(squared_anchor_distances.shape):
        raise ValueError(
            "score_logits and squared_anchor_distances must have matching [B, K] shape, "
            f"got {tuple(score_logits.shape)} and {tuple(squared_anchor_distances.shape)}."
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}.")

    batch_size, num_candidates = score_logits.shape
    effective_topk = min(int(topk), int(num_candidates))

    if source == "score":
        ranking = torch.argsort(score_logits.detach(), dim=-1, descending=True)  # [B, K]
    elif source == "nearest":
        ranking = torch.argsort(squared_anchor_distances.detach(), dim=-1, descending=False)  # [B, K]
    else:
        raise ValueError(f"Unsupported goal pool candidate source '{source}'.")

    selected_per_batch: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        positive_idx = int(positive_anchor_indices[batch_index].item())
        ordered = ranking[batch_index].tolist()
        chosen = [positive_idx]
        for candidate_idx in ordered:
            if int(candidate_idx) == positive_idx:
                continue
            chosen.append(int(candidate_idx))
            if len(chosen) >= effective_topk:
                break
        selected_per_batch.append(
            torch.as_tensor(chosen, device=score_logits.device, dtype=torch.long)
        )  # [M]

    return torch.stack(selected_per_batch, dim=0)  # [B, M]


def compute_reconstruction_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
) -> torch.Tensor:
    """Compute per-sample reconstruction error without reducing the batch dimension.

    prediction / target:
        [B, action_chunk_dim]
        [B, M, action_chunk_dim]

    returns:
        reconstruction_error:
            [B]
            [B, M]
    """
    if tuple(prediction.shape) != tuple(target.shape):
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} must match target shape {tuple(target.shape)}."
        )
    if loss_name == "smooth_l1":
        loss = F.smooth_l1_loss(prediction, target, reduction="none")
    elif loss_name == "l1":
        loss = F.l1_loss(prediction, target, reduction="none")
    else:
        raise ValueError(f"Unsupported reconstruction loss '{loss_name}'.")
    return loss.mean(dim=-1)


def build_soft_anchor_targets(
    squared_anchor_distances: torch.Tensor,
    *,
    temperature: float,
    topk: int | None = None,
) -> torch.Tensor:
    """Convert anchor distances into soft [B, K] targets for auxiliary supervision."""
    if squared_anchor_distances.ndim != 2:
        raise ValueError(
            "squared_anchor_distances must have shape [B, K], "
            f"got {tuple(squared_anchor_distances.shape)}."
        )
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    scores = -squared_anchor_distances / float(temperature)  # [B, K]
    if topk is not None:
        if topk <= 0:
            raise ValueError(f"topk must be positive when provided, got {topk}.")
        k = min(int(topk), int(scores.shape[-1]))
        topk_indices = torch.topk(scores, k=k, dim=-1, largest=True).indices  # [B, k]
        masked_scores = torch.full_like(scores, float("-inf"))
        masked_scores.scatter_(1, topk_indices, scores.gather(1, topk_indices))
        scores = masked_scores
    return torch.softmax(scores, dim=-1)  # [B, K]


def build_binary_anchor_targets(
    squared_anchor_distances: torch.Tensor,
    *,
    topk: int,
) -> torch.Tensor:
    """Build BCE positive/negative targets from nearest anchors.

    squared_anchor_distances:
        [B, K]

    returns:
        binary_targets: [B, K]
    """
    if squared_anchor_distances.ndim != 2:
        raise ValueError(
            "squared_anchor_distances must have shape [B, K], "
            f"got {tuple(squared_anchor_distances.shape)}."
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}.")

    effective_topk = min(int(topk), int(squared_anchor_distances.shape[-1]))
    positive_indices = torch.topk(
        squared_anchor_distances,
        k=effective_topk,
        dim=-1,
        largest=False,
    ).indices  # [B, topk]
    binary_targets = torch.zeros_like(squared_anchor_distances)  # [B, K]
    binary_targets.scatter_(1, positive_indices, 1.0)
    return binary_targets  # [B, K]


def compute_batch_losses(
    *,
    model: DiffusionPlannerModel,
    batch: dict[str, torch.Tensor],
    rec_loss_fn: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    timestep_grid: torch.Tensor,
    world_model: torch.nn.Module | None = None,
    goal_loss_receding_horizon: int | None = None,
    goal_loss_action_block: int | None = None,
    noise_override: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Compute the minimal diffusion planner losses.

    Inputs:
        batch["z_cur"]: [B, latent_dim]
        batch["z_goal"]: [B, latent_dim]
        batch["teacher_plan"]: [B, action_chunk_dim]
        timestep_grid: [B, K]

    Outputs:
        loss_dict["total_loss"]: []
        loss_dict["cls_loss"]: []
        loss_dict["rec_loss"]: []
        metrics: scalar python floats
    """
    z_cur = batch["z_cur"].to(device)  # [B, latent_dim]
    z_goal = batch["z_goal"].to(device)  # [B, latent_dim]
    teacher_plan = batch["teacher_plan"].to(device)  # [B, action_chunk_dim]
    batch_size = int(z_cur.shape[0])

    noisy_candidates, _ = model.initialize_noisy_candidates(
        batch_size=batch_size,
        device=device,
        dtype=z_cur.dtype,
        timesteps=timestep_grid,
        noise=noise_override,
    )  # [B, K, action_chunk_dim]
    outputs = model(
        z_cur,
        z_goal,
        noisy_candidates,
        timestep_grid,
    )
    refined_actions = outputs["refined_actions"]  # [B, K, action_chunk_dim]
    score_logits = outputs["score_logits"]  # [B, K]

    positive_anchor_indices, squared_anchor_distances = assign_positive_anchors(
        teacher_plan,
        model.anchors.to(device=device, dtype=teacher_plan.dtype),
    )  # [B], [B, K]
    positive_candidates = gather_positive_candidates(
        refined_actions,
        positive_anchor_indices,
    )  # [B, action_chunk_dim]

    ce_loss = torch.zeros((), device=device, dtype=z_cur.dtype)
    if args.cls_loss_type in {"ce", "ce_bce"}:
        ce_loss = F.cross_entropy(score_logits, positive_anchor_indices)

    binary_targets = torch.zeros_like(score_logits)  # [B, K]
    bce_loss = torch.zeros((), device=device, dtype=z_cur.dtype)
    if args.cls_loss_type in {"bce", "ce_bce"}:
        binary_targets = build_binary_anchor_targets(
            squared_anchor_distances.detach(),
            topk=int(args.bce_pos_topk),
        ).to(device=device, dtype=score_logits.dtype)  # [B, K]
        bce_loss = F.binary_cross_entropy_with_logits(
            score_logits,
            binary_targets,
        )

    cls_loss = args.cls_loss_weight * ce_loss + args.bce_weight * bce_loss
    rec_loss = rec_loss_fn(positive_candidates, teacher_plan)
    goal_pos_loss = torch.zeros((), device=device, dtype=z_cur.dtype)
    if args.goal_loss_weight > 0.0:
        if world_model is None:
            raise ValueError("--goal-loss-weight > 0 requires a loaded world_model.")
        if goal_loss_receding_horizon is None or goal_loss_action_block is None:
            raise ValueError("Goal-loss rollout shape must be resolved before computing losses.")
        positive_action_blocks = reshape_flat_actions_to_rollout_blocks(
            positive_candidates,
            plan_horizon=model.plan_horizon,
            action_dim=model.action_dim,
            receding_horizon=int(goal_loss_receding_horizon),
            action_block=int(goal_loss_action_block),
        )  # [B, 1, receding_horizon, action_block * action_dim]
        rollout_outputs = latent_rollout(
            world_model=world_model,
            z_context=z_cur,
            action_blocks=positive_action_blocks,
            history_size=int(args.goal_loss_history_size),
            return_sequence=False,
            freeze_world_model=True,
        )
        z_terminal = rollout_outputs["z_terminal"]  # [B, 1, latent_dim]
        if z_terminal.ndim != 3 or int(z_terminal.shape[1]) != 1:
            raise ValueError(
                "latent_rollout z_terminal for positive candidates must have shape [B, 1, latent_dim], "
                f"got {tuple(z_terminal.shape)}."
            )
        goal_pos_loss = F.mse_loss(
            z_terminal[:, 0, :],
            z_goal.detach(),
        )

    goal_pool_loss = torch.zeros((), device=device, dtype=z_cur.dtype)
    goal_pool_cost_min = torch.zeros((), device=device, dtype=z_cur.dtype)
    goal_pool_cost_mean = torch.zeros((), device=device, dtype=z_cur.dtype)
    if args.enable_goal_pool_loss:
        if world_model is None:
            raise ValueError("--enable-goal-pool-loss requires a loaded world_model.")
        if goal_loss_receding_horizon is None or goal_loss_action_block is None:
            raise ValueError("Goal-loss rollout shape must be resolved before computing losses.")
        goal_pool_indices = select_goal_pool_candidate_indices(
            positive_anchor_indices=positive_anchor_indices,
            score_logits=score_logits,
            squared_anchor_distances=squared_anchor_distances,
            topk=int(args.goal_pool_topk),
            source=str(args.goal_pool_candidate_source),
        )  # [B, M]
        selected_candidates = gather_candidate_batch(
            refined_actions,
            goal_pool_indices,
        )  # [B, M, action_chunk_dim]
        selected_action_blocks = reshape_flat_actions_to_rollout_blocks(
            selected_candidates,
            plan_horizon=model.plan_horizon,
            action_dim=model.action_dim,
            receding_horizon=int(goal_loss_receding_horizon),
            action_block=int(goal_loss_action_block),
        )  # [B, M, receding_horizon, action_block * action_dim]
        rollout_outputs = latent_rollout(
            world_model=world_model,
            z_context=z_cur,
            action_blocks=selected_action_blocks,
            history_size=int(args.goal_loss_history_size),
            return_sequence=False,
            freeze_world_model=True,
        )
        pool_terminal = rollout_outputs["z_terminal"]  # [B, M, latent_dim]
        if pool_terminal.ndim != 3:
            raise ValueError(
                "latent_rollout z_terminal for goal pool loss must have shape [B, M, latent_dim], "
                f"got {tuple(pool_terminal.shape)}."
            )
        goal_costs = (pool_terminal - z_goal.unsqueeze(1).detach()).square().mean(dim=-1)  # [B, M]
        goal_pool_cost_min = goal_costs.min(dim=-1).values.mean()
        goal_pool_cost_mean = goal_costs.mean()
        goal_pool_loss = (
            -float(args.goal_pool_tau)
            * torch.logsumexp(-goal_costs / float(args.goal_pool_tau), dim=1)
        ).mean()

    aux_rec_loss = torch.zeros((), device=device, dtype=z_cur.dtype)
    aux_topk = min(int(args.aux_rec_topk), model.num_anchors)
    if args.aux_rec_weight > 0.0 and aux_topk > 0:
        nearest_indices = torch.topk(
            squared_anchor_distances,
            k=aux_topk,
            dim=-1,
            largest=False,
        ).indices  # [B, topk]
        nearest_candidates = gather_candidate_batch(
            refined_actions,
            nearest_indices,
        )  # [B, topk, action_chunk_dim]
        teacher_plan_expanded = teacher_plan.unsqueeze(1).expand(-1, aux_topk, -1)  # [B, topk, action_chunk_dim]
        nearest_rec_error = compute_reconstruction_error(
            nearest_candidates,
            teacher_plan_expanded,
            loss_name=args.rec_loss,
        )  # [B, topk]
        aux_weights_full = build_soft_anchor_targets(
            squared_anchor_distances.detach(),
            temperature=float(args.aux_rec_temperature),
            topk=aux_topk,
        )  # [B, K]
        aux_weights = aux_weights_full.gather(1, nearest_indices)  # [B, topk]
        aux_rec_loss = (nearest_rec_error * aux_weights).sum(dim=-1).mean()

    score_ranking_loss = torch.zeros((), device=device, dtype=z_cur.dtype)
    if args.score_ranking_weight > 0.0:
        ranking_targets = build_soft_anchor_targets(
            squared_anchor_distances.detach(),
            temperature=float(args.score_ranking_temperature),
        )  # [B, K]
        log_probs = F.log_softmax(score_logits, dim=-1)  # [B, K]
        score_ranking_loss = -(ranking_targets * log_probs).sum(dim=-1).mean()

    total_loss = (
        cls_loss
        + args.rec_loss_weight * rec_loss
        + args.aux_rec_weight * aux_rec_loss
        + args.score_ranking_weight * score_ranking_loss
        + args.goal_loss_weight * goal_pos_loss
        + args.goal_pool_weight * goal_pool_loss
    )

    positive_scores = score_logits.gather(1, positive_anchor_indices.view(-1, 1)).squeeze(1)  # [B]
    pred_anchor_indices = torch.argmax(score_logits, dim=-1)  # [B]
    cls_acc = pred_anchor_indices.eq(positive_anchor_indices).float().mean()
    topk_anchor_l2 = torch.sqrt(
        torch.topk(
            squared_anchor_distances,
            k=max(1, aux_topk),
            dim=-1,
            largest=False,
        ).values.mean(dim=-1)
    ).mean()

    metrics = {
        "cls_loss": float(cls_loss.detach().item()),
        "ce_loss": float(ce_loss.detach().item()),
        "bce_loss": float(bce_loss.detach().item()),
        "rec_loss": float(rec_loss.detach().item()),
        "aux_rec_loss": float(aux_rec_loss.detach().item()),
        "score_ranking_loss": float(score_ranking_loss.detach().item()),
        "goal_pos_loss": float(goal_pos_loss.detach().item()),
        "goal_pool_loss": float(goal_pool_loss.detach().item()),
        "goal_loss": float((goal_pos_loss + goal_pool_loss).detach().item()),
        "goal_pool_cost_min": float(goal_pool_cost_min.detach().item()),
        "goal_pool_cost_mean": float(goal_pool_cost_mean.detach().item()),
        "total_loss": float(total_loss.detach().item()),
        "cls_acc": float(cls_acc.detach().item()),
        "positive_anchor_l2": float(torch.sqrt(squared_anchor_distances.min(dim=-1).values).mean().detach().item()),
        "topk_anchor_l2": float(topk_anchor_l2.detach().item()),
        "positive_score_mean": float(positive_scores.detach().mean().item()),
    }
    return {
        "total_loss": total_loss,
        "cls_loss": cls_loss,
        "ce_loss": ce_loss,
        "bce_loss": bce_loss,
        "rec_loss": rec_loss,
        "aux_rec_loss": aux_rec_loss,
        "score_ranking_loss": score_ranking_loss,
        "goal_pos_loss": goal_pos_loss,
        "goal_pool_loss": goal_pool_loss,
        "goal_loss": goal_pos_loss + goal_pool_loss,
    }, metrics


def train_one_epoch(
    model: DiffusionPlannerModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    rec_loss_fn: nn.Module,
    device: torch.device,
    epoch_idx: int,
    args: argparse.Namespace,
    world_model: torch.nn.Module | None = None,
    goal_loss_receding_horizon: int | None = None,
    goal_loss_action_block: int | None = None,
) -> dict[str, float]:
    model.train()
    total_cls_loss = 0.0
    total_ce_loss = 0.0
    total_bce_loss = 0.0
    total_rec_loss = 0.0
    total_aux_rec_loss = 0.0
    total_score_ranking_loss = 0.0
    total_goal_pos_loss = 0.0
    total_goal_pool_loss = 0.0
    total_goal_loss = 0.0
    total_goal_pool_cost_min = 0.0
    total_goal_pool_cost_mean = 0.0
    total_loss = 0.0
    total_cls_acc = 0.0
    total_anchor_l2 = 0.0
    total_topk_anchor_l2 = 0.0
    total_samples = 0

    for step_idx, batch in enumerate(loader):
        batch_size = int(batch["z_cur"].shape[0])
        timestep_grid = sample_timestep_grid(
            model,
            batch_size=batch_size,
            device=device,
            sampling=args.timestep_sampling,
        )  # [B, K]

        optimizer.zero_grad(set_to_none=True)
        loss_dict, metrics = compute_batch_losses(
            model=model,
            batch=batch,
            rec_loss_fn=rec_loss_fn,
            args=args,
            device=device,
            timestep_grid=timestep_grid,
            world_model=world_model,
            goal_loss_receding_horizon=goal_loss_receding_horizon,
            goal_loss_action_block=goal_loss_action_block,
        )
        loss_dict["total_loss"].backward()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
        optimizer.step()

        total_cls_loss += metrics["cls_loss"] * batch_size
        total_ce_loss += metrics["ce_loss"] * batch_size
        total_bce_loss += metrics["bce_loss"] * batch_size
        total_rec_loss += metrics["rec_loss"] * batch_size
        total_aux_rec_loss += metrics["aux_rec_loss"] * batch_size
        total_score_ranking_loss += metrics["score_ranking_loss"] * batch_size
        total_goal_pos_loss += metrics["goal_pos_loss"] * batch_size
        total_goal_pool_loss += metrics["goal_pool_loss"] * batch_size
        total_goal_loss += metrics["goal_loss"] * batch_size
        total_goal_pool_cost_min += metrics["goal_pool_cost_min"] * batch_size
        total_goal_pool_cost_mean += metrics["goal_pool_cost_mean"] * batch_size
        total_loss += metrics["total_loss"] * batch_size
        total_cls_acc += metrics["cls_acc"] * batch_size
        total_anchor_l2 += metrics["positive_anchor_l2"] * batch_size
        total_topk_anchor_l2 += metrics["topk_anchor_l2"] * batch_size
        total_samples += batch_size

        if args.log_every > 0 and (step_idx % args.log_every == 0 or step_idx == len(loader) - 1):
            unique_timesteps = torch.unique(timestep_grid.detach().cpu()).tolist()
            print(
                f"[train] epoch={epoch_idx:03d} step={step_idx:04d}/{len(loader):04d} "
                f"loss_preset={args.loss_preset} cls_loss_type={args.cls_loss_type} "
                f"batch_size={batch_size} total_loss={metrics['total_loss']:.6f} "
                f"cls_loss={metrics['cls_loss']:.6f} ce_loss={metrics['ce_loss']:.6f} "
                f"bce_loss={metrics['bce_loss']:.6f} rec_loss={metrics['rec_loss']:.6f} "
                f"aux_rec_loss={metrics['aux_rec_loss']:.6f} "
                f"score_ranking_loss={metrics['score_ranking_loss']:.6f} "
                f"goal_pos_loss={metrics['goal_pos_loss']:.6f} "
                f"goal_pool_loss={metrics['goal_pool_loss']:.6f} "
                f"goal_pool_cost_min={metrics['goal_pool_cost_min']:.6f} "
                f"goal_pool_cost_mean={metrics['goal_pool_cost_mean']:.6f} "
                f"cls_acc={metrics['cls_acc']:.4f} timesteps={unique_timesteps}"
            )

    return {
        "cls_loss": total_cls_loss / max(total_samples, 1),
        "ce_loss": total_ce_loss / max(total_samples, 1),
        "bce_loss": total_bce_loss / max(total_samples, 1),
        "rec_loss": total_rec_loss / max(total_samples, 1),
        "aux_rec_loss": total_aux_rec_loss / max(total_samples, 1),
        "score_ranking_loss": total_score_ranking_loss / max(total_samples, 1),
        "goal_pos_loss": total_goal_pos_loss / max(total_samples, 1),
        "goal_pool_loss": total_goal_pool_loss / max(total_samples, 1),
        "goal_loss": total_goal_loss / max(total_samples, 1),
        "goal_pool_cost_min": total_goal_pool_cost_min / max(total_samples, 1),
        "goal_pool_cost_mean": total_goal_pool_cost_mean / max(total_samples, 1),
        "loss": total_loss / max(total_samples, 1),
        "cls_acc": total_cls_acc / max(total_samples, 1),
        "positive_anchor_l2": total_anchor_l2 / max(total_samples, 1),
        "topk_anchor_l2": total_topk_anchor_l2 / max(total_samples, 1),
    }


@torch.inference_mode()
def evaluate(
    model: DiffusionPlannerModel,
    loader: DataLoader,
    rec_loss_fn: nn.Module,
    device: torch.device,
    split: str,
    args: argparse.Namespace,
    world_model: torch.nn.Module | None = None,
    goal_loss_receding_horizon: int | None = None,
    goal_loss_action_block: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_cls_loss = 0.0
    total_ce_loss = 0.0
    total_bce_loss = 0.0
    total_rec_loss = 0.0
    total_aux_rec_loss = 0.0
    total_score_ranking_loss = 0.0
    total_goal_pos_loss = 0.0
    total_goal_pool_loss = 0.0
    total_goal_loss = 0.0
    total_goal_pool_cost_min = 0.0
    total_goal_pool_cost_mean = 0.0
    total_loss = 0.0
    total_cls_acc = 0.0
    total_anchor_l2 = 0.0
    total_topk_anchor_l2 = 0.0
    total_samples = 0

    for batch in loader:
        batch_size = int(batch["z_cur"].shape[0])
        timestep_grid = build_eval_timestep_grid(
            model,
            batch_size=batch_size,
            device=device,
        )  # [B, K]

        _, metrics = compute_batch_losses(
            model=model,
            batch=batch,
            rec_loss_fn=rec_loss_fn,
            args=args,
            device=device,
            timestep_grid=timestep_grid,
            world_model=world_model,
            goal_loss_receding_horizon=goal_loss_receding_horizon,
            goal_loss_action_block=goal_loss_action_block,
            noise_override=torch.zeros(
                batch_size,
                model.num_anchors,
                model.action_chunk_dim,
                device=device,
                dtype=torch.float32,
            ),
        )

        total_cls_loss += metrics["cls_loss"] * batch_size
        total_ce_loss += metrics["ce_loss"] * batch_size
        total_bce_loss += metrics["bce_loss"] * batch_size
        total_rec_loss += metrics["rec_loss"] * batch_size
        total_aux_rec_loss += metrics["aux_rec_loss"] * batch_size
        total_score_ranking_loss += metrics["score_ranking_loss"] * batch_size
        total_goal_pos_loss += metrics["goal_pos_loss"] * batch_size
        total_goal_pool_loss += metrics["goal_pool_loss"] * batch_size
        total_goal_loss += metrics["goal_loss"] * batch_size
        total_goal_pool_cost_min += metrics["goal_pool_cost_min"] * batch_size
        total_goal_pool_cost_mean += metrics["goal_pool_cost_mean"] * batch_size
        total_loss += metrics["total_loss"] * batch_size
        total_cls_acc += metrics["cls_acc"] * batch_size
        total_anchor_l2 += metrics["positive_anchor_l2"] * batch_size
        total_topk_anchor_l2 += metrics["topk_anchor_l2"] * batch_size
        total_samples += batch_size

    return {
        f"{split}/cls_loss": total_cls_loss / max(total_samples, 1),
        f"{split}/ce_loss": total_ce_loss / max(total_samples, 1),
        f"{split}/bce_loss": total_bce_loss / max(total_samples, 1),
        f"{split}/rec_loss": total_rec_loss / max(total_samples, 1),
        f"{split}/aux_rec_loss": total_aux_rec_loss / max(total_samples, 1),
        f"{split}/score_ranking_loss": total_score_ranking_loss / max(total_samples, 1),
        f"{split}/goal_pos_loss": total_goal_pos_loss / max(total_samples, 1),
        f"{split}/goal_pool_loss": total_goal_pool_loss / max(total_samples, 1),
        f"{split}/goal_loss": total_goal_loss / max(total_samples, 1),
        f"{split}/goal_pool_cost_min": total_goal_pool_cost_min / max(total_samples, 1),
        f"{split}/goal_pool_cost_mean": total_goal_pool_cost_mean / max(total_samples, 1),
        f"{split}/loss": total_loss / max(total_samples, 1),
        f"{split}/cls_acc": total_cls_acc / max(total_samples, 1),
        f"{split}/positive_anchor_l2": total_anchor_l2 / max(total_samples, 1),
        f"{split}/topk_anchor_l2": total_topk_anchor_l2 / max(total_samples, 1),
    }


def save_training_summary(
    path: str | Path,
    *,
    args: argparse.Namespace,
    model: DiffusionPlannerModel,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_val_loss: float,
    anchor_bundle: ActionAnchorBundle,
) -> None:
    summary_path = Path(path).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "args": vars(args),
        "model_config": asdict(model.config),
        "anchor_bundle": {
            "num_anchors": int(anchor_bundle.num_anchors),
            "plan_horizon": int(anchor_bundle.plan_horizon),
            "action_chunk_horizon": int(anchor_bundle.action_chunk_horizon),
            "action_dim": int(anchor_bundle.action_dim),
            "action_chunk_dim": int(anchor_bundle.action_chunk_dim),
            "receding_horizon": anchor_bundle.receding_horizon,
            "action_block": anchor_bundle.action_block,
            "task": anchor_bundle.task,
            "dataset_path": anchor_bundle.dataset_path,
            "max_samples": anchor_bundle.max_samples,
            "fit_method": str(anchor_bundle.fit_method),
            "seed": anchor_bundle.seed,
            "metadata": dict(anchor_bundle.metadata),
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_val_loss": float(best_val_loss),
    }
    torch.save(summary, summary_path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    loss_feature_flags = apply_loss_preset(args)
    if (
        args.cls_loss_weight < 0.0
        or args.bce_weight < 0.0
        or args.rec_loss_weight < 0.0
        or args.aux_rec_weight < 0.0
        or args.score_ranking_weight < 0.0
        or args.goal_loss_weight < 0.0
        or args.goal_pool_weight < 0.0
    ):
        raise ValueError("Loss weights must be non-negative.")
    if args.goal_loss_history_size <= 0:
        raise ValueError(f"goal_loss_history_size must be positive, got {args.goal_loss_history_size}.")
    if (args.goal_loss_weight > 0.0 or args.enable_goal_pool_loss) and args.wm_policy in [None, "", "null"]:
        raise ValueError(
            "--goal-loss-weight > 0 or --enable-goal-pool-loss requires --wm-policy "
            "to load the frozen LeWM predictor."
        )
    if args.goal_pool_topk <= 0:
        raise ValueError(f"goal_pool_topk must be positive, got {args.goal_pool_topk}.")
    if args.bce_pos_topk <= 0:
        raise ValueError(f"bce_pos_topk must be positive, got {args.bce_pos_topk}.")
    if args.goal_pool_tau <= 0.0:
        raise ValueError(f"goal_pool_tau must be positive, got {args.goal_pool_tau}.")
    if args.aux_rec_topk <= 0:
        raise ValueError(f"aux_rec_topk must be positive, got {args.aux_rec_topk}.")
    if args.aux_rec_temperature <= 0.0:
        raise ValueError(
            f"aux_rec_temperature must be positive, got {args.aux_rec_temperature}."
        )
    if args.score_ranking_temperature <= 0.0:
        raise ValueError(
            "score_ranking_temperature must be positive, "
            f"got {args.score_ranking_temperature}."
        )
    classification_enabled = (
        (args.cls_loss_type in {"ce", "ce_bce"} and args.cls_loss_weight > 0.0)
        or (args.cls_loss_type in {"bce", "ce_bce"} and args.bce_weight > 0.0)
    )
    if (
        not classification_enabled
        and args.rec_loss_weight == 0.0
        and args.aux_rec_weight == 0.0
        and args.score_ranking_weight == 0.0
        and args.goal_loss_weight == 0.0
        and (not args.enable_goal_pool_loss or args.goal_pool_weight == 0.0)
    ):
        raise ValueError(
            "At least one of CE/BCE classification, rec_loss_weight, aux_rec_weight, "
            "score_ranking_weight, goal_loss_weight, or goal_pool_weight must be positive."
        )

    set_seed(args.seed)
    device = torch.device(args.device)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_bundle = load_dataset_bundle(args.dataset_path)
    anchor_bundle = load_anchor_bundle(args.anchor_bundle_path)
    train_dataset = DiffusionPlannerTensorDataset(train_bundle)
    train_dataset_spec = validate_anchor_dataset_compatibility(
        dataset_bundle=train_bundle,
        anchor_bundle=anchor_bundle,
        dataset_label="train_dataset",
    )

    if args.val_dataset_path is not None:
        val_bundle = load_dataset_bundle(args.val_dataset_path)
        val_dataset = DiffusionPlannerTensorDataset(val_bundle)
        validate_anchor_dataset_compatibility(
            dataset_bundle=val_bundle,
            anchor_bundle=anchor_bundle,
            dataset_label="val_dataset",
        )
        train_set = train_dataset
        val_set = val_dataset
    else:
        train_set, val_set = split_train_val(train_dataset, val_split=args.val_split, seed=args.seed)

    train_loader, val_loader = build_dataloaders(train_set, val_set, args=args)

    model = build_model_from_dataset_and_anchor_bundle(
        dataset_bundle=train_bundle,
        anchor_bundle=anchor_bundle,
        args=args,
    ).to(device)
    world_model = None
    goal_loss_receding_horizon = None
    goal_loss_action_block = None
    if args.goal_loss_weight > 0.0 or args.enable_goal_pool_loss:
        goal_loss_receding_horizon, goal_loss_action_block = infer_goal_loss_rollout_shape(
            args=args,
            dataset_bundle=train_bundle,
            anchor_bundle=anchor_bundle,
            plan_horizon=model.plan_horizon,
        )
        world_model = load_frozen_world_model(args.wm_policy, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rec_loss_fn = build_reconstruction_loss_fn(args.rec_loss)
    train_task = train_dataset_spec.get("task", "unknown")
    goal_loss_enabled = bool(args.goal_loss_weight > 0.0 or args.enable_goal_pool_loss)

    if train_task in {"tworoom", "reacher"} and args.loss_preset != "simple_bce":
        print(
            f"[warn] task={train_task} is using loss_preset={args.loss_preset}. "
            "simple_bce is the recommended task-agnostic baseline for new tasks."
        )

    print(
        f"[setup] train_samples={len(train_set)} val_samples={len(val_set)} "
        f"task={train_task} latent_dim={model.latent_dim} input_dim={model.input_dim} "
        f"action_chunk_horizon={model.action_chunk_horizon} "
        f"action_chunk_dim={model.action_chunk_dim} plan_horizon={model.plan_horizon} "
        f"action_dim={model.action_dim} num_anchors={model.num_anchors} "
        f"anchor_shape={tuple(anchor_bundle.anchors.shape)} "
        f"anchor_task={anchor_bundle.task} "
        f"anchor_action_chunk_horizon={anchor_bundle.action_chunk_horizon} "
        f"anchor_receding_horizon={anchor_bundle.receding_horizon} "
        f"anchor_action_block={anchor_bundle.action_block} "
        f"anchor_source_dataset={anchor_bundle.metadata.get('source_dataset', 'unknown')} "
        f"num_train_steps={model.num_train_steps} truncation_steps={model.truncation_steps} "
        f"loss_preset={args.loss_preset} cls_loss_type={args.cls_loss_type} "
        f"cls_loss_weight={args.cls_loss_weight} bce_weight={args.bce_weight} "
        f"bce_pos_topk={args.bce_pos_topk} rec_loss_weight={args.rec_loss_weight} "
        f"aux_rec_topk={args.aux_rec_topk} aux_rec_weight={args.aux_rec_weight} "
        f"score_ranking_weight={args.score_ranking_weight} goal_loss_weight={args.goal_loss_weight} "
        f"goal_loss_enabled={goal_loss_enabled} "
        f"enable_goal_pool_loss={args.enable_goal_pool_loss} goal_pool_weight={args.goal_pool_weight} "
        f"goal_pool_topk={args.goal_pool_topk} goal_pool_tau={args.goal_pool_tau} "
        f"goal_pool_candidate_source={args.goal_pool_candidate_source} "
        f"goal_loss_receding_horizon={goal_loss_receding_horizon} "
        f"goal_loss_action_block={goal_loss_action_block} "
        f"goal_loss_history_size={args.goal_loss_history_size}"
    )
    print(
        f"[loss-mode] loss_preset={args.loss_preset} cls_loss_type={args.cls_loss_type} "
        f"aux_rec={'enabled' if loss_feature_flags['aux_rec_enabled'] else 'disabled'} "
        f"score_rank={'enabled' if loss_feature_flags['score_rank_enabled'] else 'disabled'} "
        f"goal_pool={'enabled' if loss_feature_flags['goal_pool_enabled'] else 'disabled'} "
        f"wm_rank={'enabled' if loss_feature_flags['wm_rank_enabled'] else 'disabled'} "
        f"diversity={'enabled' if loss_feature_flags['diversity_enabled'] else 'disabled'}"
    )

    best_val_loss = float("inf")
    best_bundle_path = output_dir / "diffusion_planner_best_bundle.pt"
    last_bundle_path = output_dir / "diffusion_planner_last_bundle.pt"
    summary_path = output_dir / "diffusion_planner_train_summary.pt"

    for epoch_idx in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            rec_loss_fn=rec_loss_fn,
            device=device,
            epoch_idx=epoch_idx,
            args=args,
            world_model=world_model,
            goal_loss_receding_horizon=goal_loss_receding_horizon,
            goal_loss_action_block=goal_loss_action_block,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            rec_loss_fn=rec_loss_fn,
            device=device,
            split="val",
            args=args,
            world_model=world_model,
            goal_loss_receding_horizon=goal_loss_receding_horizon,
            goal_loss_action_block=goal_loss_action_block,
        )

        current_val_loss = float(val_metrics["val/loss"])
        print(
            f"[epoch] epoch={epoch_idx:03d} task={train_task} "
            f"action_chunk_horizon={model.action_chunk_horizon} "
            f"action_chunk_dim={model.action_chunk_dim} "
            f"loss_preset={args.loss_preset} cls_loss_type={args.cls_loss_type} "
            f"train_loss={train_metrics['loss']:.6f} train_cls_loss={train_metrics['cls_loss']:.6f} "
            f"train_ce_loss={train_metrics['ce_loss']:.6f} train_bce_loss={train_metrics['bce_loss']:.6f} "
            f"train_rec_loss={train_metrics['rec_loss']:.6f} "
            f"train_aux_rec_loss={train_metrics['aux_rec_loss']:.6f} "
            f"train_score_ranking_loss={train_metrics['score_ranking_loss']:.6f} "
            f"train_goal_pos_loss={train_metrics['goal_pos_loss']:.6f} "
            f"train_goal_pool_loss={train_metrics['goal_pool_loss']:.6f} "
            f"train_goal_pool_cost_min={train_metrics['goal_pool_cost_min']:.6f} "
            f"train_goal_pool_cost_mean={train_metrics['goal_pool_cost_mean']:.6f} "
            f"train_cls_acc={train_metrics['cls_acc']:.4f} "
            f"train_positive_anchor_l2={train_metrics['positive_anchor_l2']:.4f} "
            f"train_topk_anchor_l2={train_metrics['topk_anchor_l2']:.4f} "
            f"val_loss={val_metrics['val/loss']:.6f} val_cls_loss={val_metrics['val/cls_loss']:.6f} "
            f"val_ce_loss={val_metrics['val/ce_loss']:.6f} val_bce_loss={val_metrics['val/bce_loss']:.6f} "
            f"val_rec_loss={val_metrics['val/rec_loss']:.6f} "
            f"val_aux_rec_loss={val_metrics['val/aux_rec_loss']:.6f} "
            f"val_score_ranking_loss={val_metrics['val/score_ranking_loss']:.6f} "
            f"val_goal_pos_loss={val_metrics['val/goal_pos_loss']:.6f} "
            f"val_goal_pool_loss={val_metrics['val/goal_pool_loss']:.6f} "
            f"val_goal_pool_cost_min={val_metrics['val/goal_pool_cost_min']:.6f} "
            f"val_goal_pool_cost_mean={val_metrics['val/goal_pool_cost_mean']:.6f} "
            f"val_cls_acc={val_metrics['val/cls_acc']:.4f} "
            f"val_positive_anchor_l2={val_metrics['val/positive_anchor_l2']:.4f} "
            f"val_topk_anchor_l2={val_metrics['val/topk_anchor_l2']:.4f}"
        )

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            save_diffusion_planner_bundle(model, best_bundle_path)
            print(f"[save] best bundle updated: {best_bundle_path} (val_loss={best_val_loss:.6f})")

        save_diffusion_planner_bundle(model, last_bundle_path)
        save_training_summary(
            summary_path,
            args=args,
            model=model,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_val_loss=best_val_loss,
            anchor_bundle=anchor_bundle,
        )

    print(
        f"[done] best_val_loss={best_val_loss:.6f} "
        f"best_bundle={best_bundle_path} last_bundle={last_bundle_path}"
    )


if __name__ == "__main__":
    main()
