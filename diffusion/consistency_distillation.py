from __future__ import annotations

import argparse
import copy
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm as _tqdm

from diffusion.model import (
    DiffusionPlannerModel,
    load_diffusion_planner_bundle,
    save_diffusion_planner_bundle,
)
from diffusion.utils import add_noise_to_action_chunks, denoise_step_from_x0
from planners.latent_rollout import latent_rollout


def progress_iter(
    iterable: Iterable,
    *,
    desc: str,
    total: int | None = None,
    unit: str = "it",
    leave: bool = False,
):
    if not sys.stderr.isatty():
        return iterable
    return _tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave)


@dataclass
class ConsistencyDistillationConfig:
    """Loss weights and solver settings for consistency planner distillation.

    The student reuses the existing x0-prediction diffusion planner interface:
        C(tilde_u_i, i, z_cur, z_goal) -> u_0

    CTM-style distillation is implemented on the existing DDPM timestep grid:
        1. sample a noisy anchor state at timestep t,
        2. move it toward a lower-noise timestep u with the frozen teacher,
        3. match the student x0 prediction at t to the EMA-student x0 prediction at u.

    This keeps the module independent from the original diffusion training path while
    preserving the teacher-trajectory consistency idea from Consistency Policy.
    """

    ctm_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    goal_loss_weight: float = 0.0
    score_loss_weight: float = 0.0
    teacher_ode_steps: int = 2
    huber_delta: float = 0.0
    ema_decay: float = 0.999
    dsm_loss_weight: float = 0.0
    timestep_sampling: str = "uniform"


@dataclass(frozen=True)
class ConsistencyTimestepBatch:
    """Discrete CTM bridge timesteps on the planner's DDPM schedule."""

    start: torch.Tensor
    teacher_target: torch.Tensor
    clean_target: torch.Tensor


class ConsistencyPlannerTensorDataset(Dataset):
    """Minimal dataset wrapper for saved planner `.pt` bundles."""

    def __init__(self, dataset_bundle: dict[str, Any]):
        required_keys = {"z_cur", "z_goal", "teacher_plan"}
        missing = required_keys.difference(dataset_bundle.keys())
        if missing:
            raise KeyError(f"Dataset bundle is missing required keys: {sorted(missing)}.")

        self.z_cur = dataset_bundle["z_cur"].float()
        self.z_goal = dataset_bundle["z_goal"].float()
        self.teacher_plan = dataset_bundle["teacher_plan"].float()
        self.meta = dataset_bundle.get("meta", [{} for _ in range(int(self.z_cur.shape[0]))])
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
        if int(self.z_cur.shape[0]) != int(self.teacher_plan.shape[0]):
            raise ValueError(
                "Dataset sample count mismatch between z_cur and teacher_plan: "
                f"{self.z_cur.shape[0]} != {self.teacher_plan.shape[0]}."
            )

    def __len__(self) -> int:
        return int(self.z_cur.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "z_cur": self.z_cur[index].clone(),
            "z_goal": self.z_goal[index].clone(),
            "teacher_plan": self.teacher_plan[index].clone(),
            "meta": dict(self.meta[index]) if index < len(self.meta) else {},
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def update_ema_model(target: nn.Module, source: nn.Module, *, decay: float) -> None:
    """Update an EMA target model in place."""
    if not 0.0 <= float(decay) <= 1.0:
        raise ValueError(f"decay must be in [0, 1], got {decay}.")

    target_params = [param.data for param in target.parameters()]
    source_params = [param.data for param in source.parameters()]
    if len(target_params) != len(source_params):
        raise ValueError("target and source must have the same number of parameters.")

    torch._foreach_mul_(target_params, float(decay))
    torch._foreach_add_(target_params, source_params, alpha=1.0 - float(decay))


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval()
    model.requires_grad_(False)
    return model


def pseudo_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    delta: float = 0.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pseudo-Huber loss variant used by the Consistency Policy codebase.

    With delta=0 this reduces to MSE, matching their implementation.
    """
    if tuple(prediction.shape) != tuple(target.shape):
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} must match target shape {tuple(target.shape)}."
        )
    squared_error = (prediction - target).square()
    delta = float(delta)
    if delta == 0.0:
        loss = squared_error
        if weights is not None:
            weight_tensor = torch.as_tensor(weights, device=loss.device, dtype=loss.dtype)
            while weight_tensor.ndim < loss.ndim:
                weight_tensor = weight_tensor.unsqueeze(-1)
            loss = loss * weight_tensor
        return loss.mean()
    if delta < 0.0:
        delta = math.sqrt(math.prod(prediction.shape[1:])) * 0.00054
    loss = torch.sqrt(squared_error.square() + delta**2) - delta
    if weights is not None:
        weight_tensor = torch.as_tensor(weights, device=loss.device, dtype=loss.dtype)
        while weight_tensor.ndim < loss.ndim:
            weight_tensor = weight_tensor.unsqueeze(-1)
        loss = loss * weight_tensor
    return loss.mean()


def assign_positive_anchors(
    teacher_plan: torch.Tensor,
    anchors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if teacher_plan.ndim != 2:
        raise ValueError(f"teacher_plan must have shape [B, D], got {tuple(teacher_plan.shape)}.")
    if anchors.ndim != 2:
        raise ValueError(f"anchors must have shape [K, D], got {tuple(anchors.shape)}.")
    if int(teacher_plan.shape[-1]) != int(anchors.shape[-1]):
        raise ValueError(
            f"teacher_plan dim {teacher_plan.shape[-1]} does not match anchor dim {anchors.shape[-1]}."
        )
    squared_distances = (teacher_plan.unsqueeze(1) - anchors.unsqueeze(0)).square().sum(dim=-1)
    return torch.argmin(squared_distances, dim=-1), squared_distances


def gather_positive_candidates(
    candidates: torch.Tensor,
    positive_indices: torch.Tensor,
) -> torch.Tensor:
    if candidates.ndim != 3:
        raise ValueError(f"candidates must have shape [B, K, D], got {tuple(candidates.shape)}.")
    if positive_indices.ndim != 1:
        raise ValueError(f"positive_indices must have shape [B], got {tuple(positive_indices.shape)}.")
    gather_index = positive_indices.view(-1, 1, 1).expand(-1, 1, int(candidates.shape[-1]))
    return candidates.gather(1, gather_index).squeeze(1)


def sample_consistency_timesteps(
    *,
    batch_size: int,
    num_candidates: int,
    num_train_steps: int,
    teacher_ode_steps: int,
    device: torch.device,
    sampling: str = "uniform",
) -> ConsistencyTimestepBatch:
    """Sample start and teacher-target timesteps for CTM distillation.

    start >= teacher_target >= clean_target, where clean_target is always 0.
    All candidates in the same sample share the same timestep to match the existing
    diffusion planner training convention.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if num_candidates <= 0:
        raise ValueError(f"num_candidates must be positive, got {num_candidates}.")
    if num_train_steps <= 1:
        raise ValueError(f"num_train_steps must be greater than 1, got {num_train_steps}.")
    if teacher_ode_steps <= 0:
        raise ValueError(f"teacher_ode_steps must be positive, got {teacher_ode_steps}.")

    if sampling == "uniform":
        start_per_sample = torch.randint(
            low=1,
            high=int(num_train_steps),
            size=(int(batch_size), 1),
            device=device,
            dtype=torch.long,
        )
    elif sampling == "high_noise":
        min_step = max(1, int(num_train_steps // 2))
        start_per_sample = torch.randint(
            low=min_step,
            high=int(num_train_steps),
            size=(int(batch_size), 1),
            device=device,
            dtype=torch.long,
        )
    else:
        raise ValueError(f"Unsupported timestep sampling mode '{sampling}'.")

    max_jump = torch.clamp(start_per_sample, min=1, max=int(teacher_ode_steps))
    random_fraction = torch.rand_like(start_per_sample.float())
    jump = torch.floor(random_fraction * max_jump.float()).long() + 1
    teacher_target_per_sample = torch.clamp(start_per_sample - jump, min=0)
    clean_per_sample = torch.zeros_like(start_per_sample)

    return ConsistencyTimestepBatch(
        start=start_per_sample.expand(-1, int(num_candidates)).contiguous(),
        teacher_target=teacher_target_per_sample.expand(-1, int(num_candidates)).contiguous(),
        clean_target=clean_per_sample.expand(-1, int(num_candidates)).contiguous(),
    )


@torch.no_grad()
def teacher_denoise_to_timestep(
    *,
    teacher: DiffusionPlannerModel,
    z_cur: torch.Tensor,
    z_goal: torch.Tensor,
    noisy_candidates: torch.Tensor,
    start_timesteps: torch.Tensor,
    target_timesteps: torch.Tensor,
    max_steps: int,
) -> torch.Tensor:
    """Move noisy candidates from start_timesteps down to target_timesteps using teacher x0 predictions."""
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}.")
    current = noisy_candidates
    current_timesteps = start_timesteps.clone()
    schedule = teacher.schedule.to(device=current.device, dtype=current.dtype)

    for _ in range(int(max_steps)):
        remaining = current_timesteps - target_timesteps
        if bool(torch.all(remaining <= 0)):
            break
        next_timesteps = torch.maximum(current_timesteps - 1, target_timesteps)
        outputs = teacher(z_cur, z_goal, current, current_timesteps)
        x0_pred = outputs["refined_actions"]
        current = denoise_step_from_x0(
            current,
            x0_pred,
            timesteps=current_timesteps,
            next_timesteps=next_timesteps,
            schedule=schedule,
            eta=0.0,
            action_chunk_dim=teacher.action_chunk_dim,
        )
        current_timesteps = next_timesteps
    return current.detach()


def _compute_goal_loss_with_callable(
    world_model: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    z_cur: torch.Tensor,
    action_chunk: torch.Tensor,
    z_goal: torch.Tensor,
) -> torch.Tensor:
    z_pred = world_model(z_cur, action_chunk)
    if not torch.is_tensor(z_pred):
        raise TypeError("world_model(z_cur, action_chunk) must return a tensor for goal loss.")
    if tuple(z_pred.shape) != tuple(z_goal.shape):
        raise ValueError(
            f"world_model output shape {tuple(z_pred.shape)} must match z_goal {tuple(z_goal.shape)}."
        )
    return F.mse_loss(z_pred, z_goal.detach())


def compute_consistency_losses(
    *,
    student: DiffusionPlannerModel,
    teacher: DiffusionPlannerModel,
    ema_student: DiffusionPlannerModel,
    batch: dict[str, torch.Tensor],
    config: ConsistencyDistillationConfig,
    world_model: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    goal_rollout_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    timestep_batch: ConsistencyTimestepBatch | None = None,
    noise_override: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Compute CTM/action/goal distillation losses for one batch."""
    device = next(student.parameters()).device
    z_cur = batch["z_cur"].to(device)
    z_goal = batch["z_goal"].to(device)
    teacher_plan = batch["teacher_plan"].to(device)
    batch_size = int(z_cur.shape[0])

    if timestep_batch is None:
        timestep_batch = sample_consistency_timesteps(
            batch_size=batch_size,
            num_candidates=student.num_anchors,
            num_train_steps=student.num_train_steps,
            teacher_ode_steps=int(config.teacher_ode_steps),
            device=device,
            sampling=str(config.timestep_sampling),
        )
    start_timesteps = timestep_batch.start.to(device=device)
    target_timesteps = timestep_batch.teacher_target.to(device=device)

    noisy_candidates, _ = student.initialize_noisy_candidates(
        batch_size=batch_size,
        device=device,
        dtype=z_cur.dtype,
        timesteps=start_timesteps,
        noise=noise_override,
    )

    teacher_state_at_u = teacher_denoise_to_timestep(
        teacher=teacher,
        z_cur=z_cur,
        z_goal=z_goal,
        noisy_candidates=noisy_candidates,
        start_timesteps=start_timesteps,
        target_timesteps=target_timesteps,
        max_steps=int(config.teacher_ode_steps),
    )

    student_outputs = student(z_cur, z_goal, noisy_candidates, start_timesteps)
    student_x0 = student_outputs["refined_actions"]
    score_logits = student_outputs["score_logits"]

    with torch.no_grad():
        target_outputs = ema_student(z_cur, z_goal, teacher_state_at_u, target_timesteps)
        target_x0 = target_outputs["refined_actions"].detach()

    ctm_loss = pseudo_huber_loss(
        student_x0,
        target_x0,
        delta=float(config.huber_delta),
    )

    positive_indices, squared_anchor_distances = assign_positive_anchors(
        teacher_plan,
        student.anchors.to(device=device, dtype=teacher_plan.dtype),
    )
    positive_candidates = gather_positive_candidates(student_x0, positive_indices)
    action_loss = F.l1_loss(positive_candidates, teacher_plan.detach())

    score_loss = torch.zeros((), device=device, dtype=z_cur.dtype)
    if float(config.score_loss_weight) > 0.0:
        score_loss = F.cross_entropy(score_logits, positive_indices)

    goal_loss = torch.zeros((), device=device, dtype=z_cur.dtype)
    if float(config.goal_loss_weight) > 0.0:
        if goal_rollout_fn is not None:
            goal_loss = goal_rollout_fn(z_cur, positive_candidates, z_goal)
        elif world_model is not None:
            goal_loss = _compute_goal_loss_with_callable(world_model, z_cur, positive_candidates, z_goal)
        else:
            raise ValueError("goal_loss_weight > 0 requires world_model or goal_rollout_fn.")

    dsm_loss = torch.zeros((), device=device, dtype=z_cur.dtype)
    if float(config.dsm_loss_weight) > 0.0:
        clean_repeated = teacher_plan.unsqueeze(1).expand(-1, student.num_anchors, -1)
        clean_noisy = add_noise_to_action_chunks(
            clean_repeated,
            timesteps=start_timesteps,
            schedule=student.schedule.to(device=device, dtype=z_cur.dtype),
            noise=noise_override,
            action_chunk_dim=student.action_chunk_dim,
        )
        dsm_pred = student(z_cur, z_goal, clean_noisy, start_timesteps)["refined_actions"]
        dsm_loss = pseudo_huber_loss(
            dsm_pred,
            clean_repeated.detach(),
            delta=float(config.huber_delta),
        )

    total_loss = (
        float(config.ctm_loss_weight) * ctm_loss
        + float(config.action_loss_weight) * action_loss
        + float(config.goal_loss_weight) * goal_loss
        + float(config.score_loss_weight) * score_loss
        + float(config.dsm_loss_weight) * dsm_loss
    )

    positive_scores = score_logits.gather(1, positive_indices.view(-1, 1)).squeeze(1)
    pred_indices = torch.argmax(score_logits, dim=-1)
    metrics = {
        "total_loss": float(total_loss.detach().item()),
        "ctm_loss": float(ctm_loss.detach().item()),
        "action_loss": float(action_loss.detach().item()),
        "goal_loss": float(goal_loss.detach().item()),
        "score_loss": float(score_loss.detach().item()),
        "dsm_loss": float(dsm_loss.detach().item()),
        "score_acc": float(pred_indices.eq(positive_indices).float().mean().detach().item()),
        "positive_anchor_l2": float(torch.sqrt(squared_anchor_distances.min(dim=-1).values).mean().detach().item()),
        "positive_score_mean": float(positive_scores.detach().mean().item()),
        "start_timestep_mean": float(start_timesteps.float().mean().detach().item()),
        "target_timestep_mean": float(target_timesteps.float().mean().detach().item()),
    }
    return {
        "total_loss": total_loss,
        "ctm_loss": ctm_loss,
        "action_loss": action_loss,
        "goal_loss": goal_loss,
        "score_loss": score_loss,
        "dsm_loss": dsm_loss,
    }, metrics


class ConsistencyPlannerSampler:
    """One/few-step sampler for a distilled consistency planner."""

    def __init__(self, planner: DiffusionPlannerModel):
        self.planner = planner

    @torch.inference_mode()
    def sample(
        self,
        z_cur: torch.Tensor,
        z_goal: torch.Tensor,
        *,
        steps: int = 1,
        start_timestep: int | None = None,
        noise: torch.Tensor | None = None,
        noise_scale: float = 1.0,
        sampling_temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if int(steps) <= 0:
            raise ValueError(f"steps must be positive, got {steps}.")
        outputs = self.planner.generate_candidates(
            z_cur,
            z_goal,
            noise=noise,
            eta=0.0,
            truncation_steps=int(steps),
            start_timestep=start_timestep,
            noise_scale=float(noise_scale),
            sampling_temperature=float(sampling_temperature),
            return_intermediates=False,
        )
        return {
            "candidates": outputs["candidates"],
            "score_logits": outputs["score_logits"],
            "scores": outputs["scores"],
            "initial_noisy_candidates": outputs["initial_noisy_candidates"],
            "final_noisy_state": outputs["final_noisy_state"],
            "timesteps": outputs["truncation_timesteps"],
            "truncation_timesteps": outputs["truncation_timesteps"],
        }


def load_dataset_bundle(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset bundle not found: {dataset_path}")
    return torch.load(dataset_path, map_location="cpu")


def split_train_val(
    dataset: Dataset,
    val_split: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    if not 0.0 < float(val_split) < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}.")
    val_len = max(1, int(math.floor(len(dataset) * float(val_split))))
    train_len = len(dataset) - val_len
    if train_len <= 0:
        raise ValueError(f"Dataset size {len(dataset)} is too small for val_split={val_split}.")
    generator = torch.Generator().manual_seed(int(seed))
    return random_split(dataset, [train_len, val_len], generator=generator)


def build_dataloaders(
    train_set: Dataset,
    val_set: Dataset,
    *,
    batch_size: int,
    val_batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    return (
        DataLoader(
            train_set,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(num_workers),
            pin_memory=True,
            drop_last=False,
        ),
        DataLoader(
            val_set,
            batch_size=int(val_batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=True,
            drop_last=False,
        ),
    )


def reshape_flat_actions_to_rollout_blocks(
    flat_actions: torch.Tensor,
    *,
    plan_horizon: int,
    action_dim: int,
    receding_horizon: int,
    action_block: int,
) -> torch.Tensor:
    if flat_actions.ndim != 2:
        raise ValueError(f"flat_actions must have shape [B, D], got {tuple(flat_actions.shape)}.")
    if int(receding_horizon) * int(action_block) != int(plan_horizon):
        raise ValueError(
            "receding_horizon * action_block must equal plan_horizon: "
            f"{receding_horizon} * {action_block} != {plan_horizon}."
        )
    steps = flat_actions.reshape(int(flat_actions.shape[0]), int(plan_horizon), int(action_dim))
    return steps.reshape(
        int(flat_actions.shape[0]),
        1,
        int(receding_horizon),
        int(action_block * action_dim),
    )


def make_lewm_goal_rollout_fn(
    *,
    world_model: nn.Module,
    plan_horizon: int,
    action_dim: int,
    receding_horizon: int,
    action_block: int,
    history_size: int,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    def rollout_goal_loss(z_cur: torch.Tensor, action_chunk: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        action_blocks = reshape_flat_actions_to_rollout_blocks(
            action_chunk,
            plan_horizon=int(plan_horizon),
            action_dim=int(action_dim),
            receding_horizon=int(receding_horizon),
            action_block=int(action_block),
        )
        rollout = latent_rollout(
            world_model=world_model,
            z_context=z_cur,
            action_blocks=action_blocks,
            history_size=int(history_size),
            return_sequence=False,
            freeze_world_model=True,
        )
        z_terminal = rollout["z_terminal"]
        if z_terminal.ndim != 3 or int(z_terminal.shape[1]) != 1:
            raise ValueError(
                f"latent_rollout z_terminal must have shape [B, 1, latent_dim], got {tuple(z_terminal.shape)}."
            )
        return F.mse_loss(z_terminal[:, 0, :], z_goal.detach())

    return rollout_goal_loss


def load_frozen_world_model(policy_path: str, device: torch.device) -> nn.Module:
    import stable_worldmodel as swm

    world_model = swm.policy.AutoCostModel(policy_path)
    world_model = world_model.to(device)
    world_model.eval()
    world_model.requires_grad_(False)
    return world_model


def infer_rollout_shape(
    *,
    dataset_bundle: dict[str, Any],
    model: DiffusionPlannerModel,
    receding_horizon: int | None,
    action_block: int | None,
) -> tuple[int, int]:
    if receding_horizon is None:
        value = model.anchor_metadata.get("receding_horizon")
        if value is None:
            value = model.anchor_metadata.get("source_build_info", {}).get("plan_config", {}).get("receding_horizon")
        if value is None:
            value = dataset_bundle.get("build_info", {}).get("plan_config", {}).get("receding_horizon")
        receding_horizon = None if value is None else int(value)
    if action_block is None:
        value = model.anchor_metadata.get("action_block")
        if value is None:
            value = model.anchor_metadata.get("source_build_info", {}).get("plan_config", {}).get("action_block")
        if value is None:
            value = dataset_bundle.get("build_info", {}).get("plan_config", {}).get("action_block")
        action_block = None if value is None else int(value)

    if receding_horizon is None and action_block is not None:
        if model.plan_horizon % int(action_block) != 0:
            raise ValueError("Cannot infer receding_horizon from action_block.")
        receding_horizon = int(model.plan_horizon // int(action_block))
    if action_block is None and receding_horizon is not None:
        if model.plan_horizon % int(receding_horizon) != 0:
            raise ValueError("Cannot infer action_block from receding_horizon.")
        action_block = int(model.plan_horizon // int(receding_horizon))
    if receding_horizon is None or action_block is None:
        raise ValueError(
            "Could not infer goal rollout shape. Provide --goal-loss-receding-horizon and --goal-loss-action-block."
        )
    return int(receding_horizon), int(action_block)


def train_one_epoch(
    *,
    student: DiffusionPlannerModel,
    teacher: DiffusionPlannerModel,
    ema_student: DiffusionPlannerModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_idx: int,
    config: ConsistencyDistillationConfig,
    grad_clip_norm: float,
    log_every: int,
    goal_rollout_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, float]:
    student.train()
    teacher.eval()
    ema_student.eval()
    totals: dict[str, float] = {}
    total_samples = 0

    train_batches = progress_iter(
        loader,
        desc=f"consistency train epoch {epoch_idx:03d}",
        total=len(loader),
        unit="batch",
    )
    for step_idx, batch in enumerate(train_batches):
        batch_size = int(batch["z_cur"].shape[0])
        optimizer.zero_grad(set_to_none=True)
        loss_dict, metrics = compute_consistency_losses(
            student=student,
            teacher=teacher,
            ema_student=ema_student,
            batch=batch,
            config=config,
            goal_rollout_fn=goal_rollout_fn,
        )
        loss_dict["total_loss"].backward()
        if float(grad_clip_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=float(grad_clip_norm))
        optimizer.step()
        update_ema_model(ema_student, student, decay=float(config.ema_decay))

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        total_samples += batch_size

        if int(log_every) > 0 and (step_idx % int(log_every) == 0 or step_idx == len(loader) - 1):
            print(
                f"[consistency-train] epoch={epoch_idx:03d} step={step_idx:04d}/{len(loader):04d} "
                f"total_loss={metrics['total_loss']:.6f} ctm_loss={metrics['ctm_loss']:.6f} "
                f"action_loss={metrics['action_loss']:.6f} goal_loss={metrics['goal_loss']:.6f} "
                f"score_loss={metrics['score_loss']:.6f} dsm_loss={metrics['dsm_loss']:.6f} "
                f"score_acc={metrics['score_acc']:.4f} "
                f"start_t={metrics['start_timestep_mean']:.2f} target_t={metrics['target_timestep_mean']:.2f}"
            )

    return {key: value / max(total_samples, 1) for key, value in totals.items()}


@torch.inference_mode()
def evaluate(
    *,
    student: DiffusionPlannerModel,
    teacher: DiffusionPlannerModel,
    ema_student: DiffusionPlannerModel,
    loader: DataLoader,
    device: torch.device,
    config: ConsistencyDistillationConfig,
    goal_rollout_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    split: str = "val",
) -> dict[str, float]:
    student.eval()
    teacher.eval()
    ema_student.eval()
    totals: dict[str, float] = {}
    total_samples = 0

    val_batches = progress_iter(
        loader,
        desc=f"consistency {split}",
        total=len(loader),
        unit="batch",
    )
    for batch in val_batches:
        batch_size = int(batch["z_cur"].shape[0])
        timestep_batch = ConsistencyTimestepBatch(
            start=torch.full(
                (batch_size, student.num_anchors),
                student.num_train_steps - 1,
                device=device,
                dtype=torch.long,
            ),
            teacher_target=torch.clamp(
                torch.full(
                    (batch_size, student.num_anchors),
                    student.num_train_steps - 1 - int(config.teacher_ode_steps),
                    device=device,
                    dtype=torch.long,
                ),
                min=0,
            ),
            clean_target=torch.zeros(batch_size, student.num_anchors, device=device, dtype=torch.long),
        )
        _, metrics = compute_consistency_losses(
            student=student,
            teacher=teacher,
            ema_student=ema_student,
            batch=batch,
            config=config,
            goal_rollout_fn=goal_rollout_fn,
            timestep_batch=timestep_batch,
            noise_override=torch.zeros(
                batch_size,
                student.num_anchors,
                student.action_chunk_dim,
                device=device,
                dtype=torch.float32,
            ),
        )
        for key, value in metrics.items():
            totals[f"{split}/{key}"] = totals.get(f"{split}/{key}", 0.0) + float(value) * batch_size
        total_samples += batch_size

    return {key: value / max(total_samples, 1) for key, value in totals.items()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train_consistency_planner.py",
        description="Distill a multi-step diffusion planner into a one/few-step consistency planner.",
    )
    parser.add_argument("--dataset-path", required=True, help="Path to a built planner dataset `.pt` file.")
    parser.add_argument("--val-dataset-path", default=None, help="Optional validation dataset `.pt` file.")
    parser.add_argument("--teacher-bundle-path", required=True, help="Pretrained diffusion planner bundle.")
    parser.add_argument(
        "--student-init-bundle-path",
        default=None,
        help="Optional planner bundle used to initialize the student. Defaults to teacher warm-start.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for distilled planner bundles.")
    parser.add_argument("--wm-policy", default=None, help="Optional LeWM checkpoint path for goal consistency.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Train batch size.")
    parser.add_argument("--val-batch-size", type=int, default=256, help="Validation batch size.")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split when no val dataset is given.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping max norm.")
    parser.add_argument("--log-every", type=int, default=50, help="Train-step logging frequency.")
    parser.add_argument("--ctm-loss-weight", type=float, default=1.0, help="CTM trajectory consistency loss weight.")
    parser.add_argument("--action-loss-weight", type=float, default=1.0, help="Teacher action chunk L1 loss weight.")
    parser.add_argument("--goal-loss-weight", type=float, default=0.0, help="LeWM latent goal consistency loss weight.")
    parser.add_argument("--score-loss-weight", type=float, default=0.0, help="Positive anchor score CE loss weight.")
    parser.add_argument("--dsm-loss-weight", type=float, default=0.0, help="Optional clean action DSM loss weight.")
    parser.add_argument("--teacher-ode-steps", type=int, default=2, help="Teacher bridge steps from t to u.")
    parser.add_argument("--huber-delta", type=float, default=0.0, help="Pseudo-Huber delta. 0 means MSE.")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay for target student.")
    parser.add_argument(
        "--timestep-sampling",
        choices=["uniform", "high_noise"],
        default="uniform",
        help="Training timestep sampler.",
    )
    parser.add_argument("--goal-loss-history-size", type=int, default=3, help="History size for latent rollout.")
    parser.add_argument("--goal-loss-receding-horizon", type=int, default=None, help="Rollout block count.")
    parser.add_argument("--goal-loss-action-block", type=int, default=None, help="Env steps per rollout action block.")
    return parser.parse_args(argv)


def validate_training_args(args: argparse.Namespace) -> None:
    nonnegative = [
        args.ctm_loss_weight,
        args.action_loss_weight,
        args.goal_loss_weight,
        args.score_loss_weight,
        args.dsm_loss_weight,
        args.weight_decay,
    ]
    if any(float(value) < 0.0 for value in nonnegative):
        raise ValueError("Loss weights and weight_decay must be non-negative.")
    if args.ctm_loss_weight == 0.0 and args.action_loss_weight == 0.0 and args.goal_loss_weight == 0.0 and args.score_loss_weight == 0.0 and args.dsm_loss_weight == 0.0:
        raise ValueError("At least one loss weight must be positive.")
    if args.teacher_ode_steps <= 0:
        raise ValueError(f"teacher_ode_steps must be positive, got {args.teacher_ode_steps}.")
    if not 0.0 <= float(args.ema_decay) <= 1.0:
        raise ValueError(f"ema_decay must be in [0, 1], got {args.ema_decay}.")
    if args.goal_loss_weight > 0.0 and args.wm_policy in [None, "", "null"]:
        raise ValueError("--goal-loss-weight > 0 requires --wm-policy.")


def _none_if_null(value: Any) -> Any:
    if value in [None, "", "null", "None"]:
        return None
    return value


def build_args_from_config(cfg: DictConfig) -> argparse.Namespace:
    """Flatten the Hydra YAML config into the training args namespace."""
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Hydra config must resolve to a dictionary.")

    task = resolved.get("task", {})
    teacher = resolved.get("teacher", {})
    output = resolved.get("output", {})
    runtime = resolved.get("runtime", {})
    train_cfg = resolved.get("train", {})
    distill = resolved.get("distill", {})
    goal_loss = resolved.get("goal_loss", {})

    return argparse.Namespace(
        dataset_path=str(task["planner_dataset_path"]),
        val_dataset_path=_none_if_null(task.get("val_dataset_path")),
        teacher_bundle_path=str(teacher["bundle_path"]),
        student_init_bundle_path=_none_if_null(teacher.get("student_init_bundle_path")),
        output_dir=str(output["dir"]),
        wm_policy=_none_if_null(task.get("wm_policy")),
        device=str(runtime.get("device", "cuda")),
        seed=int(runtime.get("seed", 42)),
        epochs=int(train_cfg.get("epochs", 50)),
        batch_size=int(train_cfg.get("batch_size", 128)),
        val_batch_size=int(train_cfg.get("val_batch_size", 256)),
        val_split=float(train_cfg.get("val_split", 0.1)),
        num_workers=int(train_cfg.get("num_workers", 4)),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        grad_clip_norm=float(train_cfg.get("grad_clip_norm", 1.0)),
        log_every=int(train_cfg.get("log_every", 50)),
        ctm_loss_weight=float(distill.get("ctm_loss_weight", 1.0)),
        action_loss_weight=float(distill.get("action_loss_weight", 1.0)),
        goal_loss_weight=float(distill.get("goal_loss_weight", 0.0)),
        score_loss_weight=float(distill.get("score_loss_weight", 0.0)),
        dsm_loss_weight=float(distill.get("dsm_loss_weight", 0.0)),
        teacher_ode_steps=int(distill.get("teacher_ode_steps", 2)),
        huber_delta=float(distill.get("huber_delta", 0.0)),
        ema_decay=float(distill.get("ema_decay", 0.999)),
        timestep_sampling=str(distill.get("timestep_sampling", "uniform")),
        goal_loss_history_size=int(goal_loss.get("history_size", 3)),
        goal_loss_receding_horizon=_none_if_null(goal_loss.get("receding_horizon")),
        goal_loss_action_block=_none_if_null(goal_loss.get("action_block")),
    )


def save_training_summary(
    path: str | Path,
    *,
    args: argparse.Namespace,
    student: DiffusionPlannerModel,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    best_val_loss: float,
) -> None:
    summary_path = Path(path).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "args": vars(args),
            "distillation_config": {
                "ctm_loss_weight": float(args.ctm_loss_weight),
                "action_loss_weight": float(args.action_loss_weight),
                "goal_loss_weight": float(args.goal_loss_weight),
                "score_loss_weight": float(args.score_loss_weight),
                "dsm_loss_weight": float(args.dsm_loss_weight),
                "teacher_ode_steps": int(args.teacher_ode_steps),
                "huber_delta": float(args.huber_delta),
                "ema_decay": float(args.ema_decay),
                "timestep_sampling": str(args.timestep_sampling),
            },
            "model_config": asdict(student.config),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_val_loss": float(best_val_loss),
        },
        summary_path,
    )


def run_training(args: argparse.Namespace) -> None:
    validate_training_args(args)
    set_seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_bundle = load_dataset_bundle(args.dataset_path)
    dataset = ConsistencyPlannerTensorDataset(dataset_bundle)
    if args.val_dataset_path is None:
        train_set, val_set = split_train_val(dataset, val_split=args.val_split, seed=args.seed)
    else:
        train_set = dataset
        val_set = ConsistencyPlannerTensorDataset(load_dataset_bundle(args.val_dataset_path))
    train_loader, val_loader = build_dataloaders(
        train_set,
        val_set,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
    )

    teacher_bundle = load_diffusion_planner_bundle(args.teacher_bundle_path, map_location=device)
    teacher = teacher_bundle.instantiate_model(map_location=device).to(device)
    teacher.load_state_dict(teacher_bundle.model_state_dict)

    if args.student_init_bundle_path is None:
        student = copy.deepcopy(teacher).to(device)
    else:
        student_bundle = load_diffusion_planner_bundle(args.student_init_bundle_path, map_location=device)
        student = student_bundle.instantiate_model(map_location=device).to(device)
        student.load_state_dict(student_bundle.model_state_dict)
    freeze_model(teacher)
    student.requires_grad_(True)
    student.train()
    ema_student = copy.deepcopy(student).to(device)
    freeze_model(ema_student)

    config = ConsistencyDistillationConfig(
        ctm_loss_weight=float(args.ctm_loss_weight),
        action_loss_weight=float(args.action_loss_weight),
        goal_loss_weight=float(args.goal_loss_weight),
        score_loss_weight=float(args.score_loss_weight),
        dsm_loss_weight=float(args.dsm_loss_weight),
        teacher_ode_steps=int(args.teacher_ode_steps),
        huber_delta=float(args.huber_delta),
        ema_decay=float(args.ema_decay),
        timestep_sampling=str(args.timestep_sampling),
    )

    world_model = None
    goal_rollout_fn = None
    if args.goal_loss_weight > 0.0:
        receding_horizon, action_block = infer_rollout_shape(
            dataset_bundle=dataset_bundle,
            model=student,
            receding_horizon=args.goal_loss_receding_horizon,
            action_block=args.goal_loss_action_block,
        )
        world_model = load_frozen_world_model(args.wm_policy, device=device)
        goal_rollout_fn = make_lewm_goal_rollout_fn(
            world_model=world_model,
            plan_horizon=student.plan_horizon,
            action_dim=student.action_dim,
            receding_horizon=receding_horizon,
            action_block=action_block,
            history_size=int(args.goal_loss_history_size),
        )

    optimizer = torch.optim.AdamW(student.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_val_loss = float("inf")
    best_bundle_path = output_dir / "consistency_planner_best_bundle.pt"
    last_bundle_path = output_dir / "consistency_planner_last_bundle.pt"
    ema_bundle_path = output_dir / "consistency_planner_ema_bundle.pt"
    summary_path = output_dir / "consistency_planner_train_summary.pt"

    print(
        f"[consistency-setup] train_samples={len(train_set)} val_samples={len(val_set)} "
        f"latent_dim={student.latent_dim} action_chunk_dim={student.action_chunk_dim} "
        f"num_anchors={student.num_anchors} num_train_steps={student.num_train_steps} "
        f"teacher_ode_steps={config.teacher_ode_steps} ctm_loss_weight={config.ctm_loss_weight} "
        f"action_loss_weight={config.action_loss_weight} goal_loss_weight={config.goal_loss_weight} "
        f"score_loss_weight={config.score_loss_weight} dsm_loss_weight={config.dsm_loss_weight} "
        f"ema_decay={config.ema_decay}"
    )

    train_metrics: dict[str, float] = {}
    val_metrics: dict[str, float] = {}
    epoch_iter = progress_iter(
        range(1, int(args.epochs) + 1),
        desc="consistency epochs",
        total=int(args.epochs),
        unit="epoch",
        leave=True,
    )
    for epoch_idx in epoch_iter:
        train_metrics = train_one_epoch(
            student=student,
            teacher=teacher,
            ema_student=ema_student,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch_idx=epoch_idx,
            config=config,
            grad_clip_norm=float(args.grad_clip_norm),
            log_every=int(args.log_every),
            goal_rollout_fn=goal_rollout_fn,
        )
        val_metrics = evaluate(
            student=student,
            teacher=teacher,
            ema_student=ema_student,
            loader=val_loader,
            device=device,
            config=config,
            goal_rollout_fn=goal_rollout_fn,
            split="val",
        )
        current_val_loss = float(val_metrics["val/total_loss"])
        print(
            f"[consistency-epoch] epoch={epoch_idx:03d} "
            f"train_loss={train_metrics['total_loss']:.6f} train_ctm={train_metrics['ctm_loss']:.6f} "
            f"train_action={train_metrics['action_loss']:.6f} train_goal={train_metrics['goal_loss']:.6f} "
            f"val_loss={val_metrics['val/total_loss']:.6f} val_ctm={val_metrics['val/ctm_loss']:.6f} "
            f"val_action={val_metrics['val/action_loss']:.6f} val_goal={val_metrics['val/goal_loss']:.6f}"
        )
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            save_diffusion_planner_bundle(student, best_bundle_path)
            print(f"[save] best consistency bundle updated: {best_bundle_path} (val_loss={best_val_loss:.6f})")
        save_diffusion_planner_bundle(student, last_bundle_path)
        save_diffusion_planner_bundle(ema_student, ema_bundle_path)
        save_training_summary(
            summary_path,
            args=args,
            student=student,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_val_loss=best_val_loss,
        )

    print(
        f"[done] best_val_loss={best_val_loss:.6f} "
        f"best_bundle={best_bundle_path} last_bundle={last_bundle_path} ema_bundle={ema_bundle_path}"
    )


def hydra_main(cfg: DictConfig) -> None:
    args = build_args_from_config(cfg)
    run_training(args)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_training(args)


if __name__ == "__main__":
    main()
