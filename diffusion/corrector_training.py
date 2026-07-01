from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from diffusion.corrector import (
    ActionChunkCorrector,
    ActionChunkCorrectorConfig,
    save_corrector_bundle,
)
from planners.latent_rollout import latent_rollout


@dataclass
class CorrectorTrainingSpec:
    correction_interval: int = 5
    action_block: int | None = None
    noise_std: float = 0.05
    noise_prob: float = 1.0
    remainder_noise_std: float | None = None
    lambda_action: float = 1.0
    lambda_goal: float = 0.0
    lambda_smooth: float = 0.0
    seed: int = 42

    @property
    def effective_remainder_noise_std(self) -> float:
        if self.remainder_noise_std is None:
            return float(self.noise_std)
        return float(self.remainder_noise_std)


def load_dataset_bundle(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Corrector dataset bundle not found: {dataset_path}")
    bundle = torch.load(dataset_path, map_location="cpu")
    if not isinstance(bundle, dict):
        raise ValueError(f"Expected dataset bundle dict, got {type(bundle)}.")
    return bundle


def _infer_plan_horizon_action_dim(dataset_bundle: dict[str, Any]) -> tuple[int, int]:
    teacher_plan = dataset_bundle.get("teacher_plan")
    meta = dataset_bundle.get("meta", [])
    if not torch.is_tensor(teacher_plan) or teacher_plan.ndim != 2:
        raise ValueError("dataset_bundle['teacher_plan'] must have shape [N, action_chunk_dim].")
    meta0 = meta[0] if isinstance(meta, list) and len(meta) > 0 and isinstance(meta[0], dict) else {}
    plan_horizon = meta0.get("plan_horizon")
    action_dim = meta0.get("action_dim")
    if plan_horizon is None or action_dim is None:
        raise KeyError("dataset_bundle meta[0] must contain plan_horizon and action_dim.")
    plan_horizon = int(plan_horizon)
    action_dim = int(action_dim)
    if plan_horizon <= 0 or action_dim <= 0:
        raise ValueError(
            f"plan_horizon and action_dim must be positive, got {plan_horizon}, {action_dim}."
        )
    if int(teacher_plan.shape[-1]) != int(plan_horizon * action_dim):
        raise ValueError(
            "teacher_plan width does not match meta plan_horizon * action_dim: "
            f"{teacher_plan.shape[-1]} != {plan_horizon * action_dim}."
        )
    return plan_horizon, action_dim


def _infer_action_block(dataset_bundle: dict[str, Any], spec: CorrectorTrainingSpec) -> int:
    if spec.action_block is not None:
        action_block = int(spec.action_block)
        if action_block <= 0:
            raise ValueError(f"action_block must be positive, got {action_block}.")
        return action_block
    build_info = dataset_bundle.get("build_info", {})
    plan_config = build_info.get("plan_config", {}) if isinstance(build_info, dict) else {}
    if isinstance(plan_config, dict) and plan_config.get("action_block") is not None:
        action_block = int(plan_config["action_block"])
        if action_block <= 0:
            raise ValueError(f"action_block must be positive, got {action_block}.")
        return action_block
    return 1


def resolve_effective_correction_steps(
    *,
    correction_interval: int,
    action_block: int,
    plan_horizon: int,
) -> int:
    correction_interval = int(correction_interval)
    action_block = int(action_block)
    plan_horizon = int(plan_horizon)
    if correction_interval <= 0:
        raise ValueError(f"correction_interval must be positive, got {correction_interval}.")
    if action_block <= 0:
        raise ValueError(f"action_block must be positive, got {action_block}.")
    effective = int(math.ceil(correction_interval / action_block) * action_block)
    if effective <= 0 or effective >= plan_horizon:
        raise ValueError(
            "Effective correction split must leave at least one remaining action: "
            f"effective={effective}, plan_horizon={plan_horizon}."
        )
    return effective


class CorrectorTrainingDataset(Dataset):
    """Synthetic drift dataset for learned action-chunk correction."""

    def __init__(
        self,
        dataset_bundle: dict[str, Any],
        spec: CorrectorTrainingSpec,
    ):
        required = {"z_cur", "z_goal", "teacher_plan", "meta"}
        missing = required.difference(dataset_bundle.keys())
        if missing:
            raise KeyError(f"Corrector dataset bundle is missing required keys: {sorted(missing)}.")
        self.z_cur = dataset_bundle["z_cur"].float()
        self.z_goal = dataset_bundle["z_goal"].float()
        self.teacher_plan = dataset_bundle["teacher_plan"].float()
        self.meta = dataset_bundle["meta"]
        self.spec = spec
        if self.z_cur.ndim != 2 or self.z_goal.ndim != 2:
            raise ValueError("z_cur and z_goal must have shape [N, latent_dim].")
        if self.z_cur.shape != self.z_goal.shape:
            raise ValueError("z_cur and z_goal must have matching shape.")
        if self.teacher_plan.ndim != 2:
            raise ValueError("teacher_plan must have shape [N, action_chunk_dim].")
        if int(self.teacher_plan.shape[0]) != int(self.z_cur.shape[0]):
            raise ValueError("teacher_plan and latents must have matching sample count.")
        if len(self.meta) != int(self.z_cur.shape[0]):
            raise ValueError("meta length must match sample count.")

        self.plan_horizon, self.action_dim = _infer_plan_horizon_action_dim(dataset_bundle)
        self.action_block = _infer_action_block(dataset_bundle, spec)
        self.prefix_steps = resolve_effective_correction_steps(
            correction_interval=int(spec.correction_interval),
            action_block=self.action_block,
            plan_horizon=self.plan_horizon,
        )
        self.remain_horizon = int(self.plan_horizon - self.prefix_steps)

    @property
    def latent_dim(self) -> int:
        return int(self.z_cur.shape[-1])

    def __len__(self) -> int:
        return int(self.z_cur.shape[0])

    def _noise_like(self, value: torch.Tensor, *, index: int, salt: int, std: float) -> torch.Tensor:
        if std <= 0.0:
            return torch.zeros_like(value)
        generator = torch.Generator(device="cpu").manual_seed(int(self.spec.seed) + 1000003 * int(index) + salt)
        keep_generator = torch.Generator(device="cpu").manual_seed(int(self.spec.seed) + 2000003 * int(index) + salt)
        noise = torch.randn(value.shape, generator=generator, dtype=value.dtype) * float(std)
        if float(self.spec.noise_prob) >= 1.0:
            return noise
        if float(self.spec.noise_prob) <= 0.0:
            return torch.zeros_like(value)
        mask = (
            torch.rand(value.shape, generator=keep_generator, dtype=value.dtype)
            < float(self.spec.noise_prob)
        ).to(dtype=value.dtype)
        return noise * mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        plan = self.teacher_plan[index].reshape(self.plan_horizon, self.action_dim)
        clean_prefix = plan[: self.prefix_steps].clone()
        target_remain = plan[self.prefix_steps :].clone()
        noisy_prefix = clean_prefix + self._noise_like(
            clean_prefix,
            index=index,
            salt=17,
            std=float(self.spec.noise_std),
        )
        u_remain = target_remain + self._noise_like(
            target_remain,
            index=index,
            salt=31,
            std=float(self.spec.effective_remainder_noise_std),
        )
        return {
            "z_cur": self.z_cur[index].clone(),
            "z_goal": self.z_goal[index].clone(),
            "clean_prefix": clean_prefix,
            "noisy_prefix": noisy_prefix,
            "u_remain": u_remain,
            "target_remain": target_remain,
            "action_block": torch.tensor(int(self.action_block), dtype=torch.long),
        }


def action_steps_to_blocks(
    action_steps: torch.Tensor,
    *,
    action_block: int,
) -> torch.Tensor:
    if not torch.is_tensor(action_steps) or action_steps.ndim != 3:
        raise ValueError(
            f"action_steps must have shape [B, steps, action_dim], got {getattr(action_steps, 'shape', None)}."
        )
    if int(action_steps.shape[1]) % int(action_block) != 0:
        raise ValueError(
            f"action step count {action_steps.shape[1]} is not divisible by action_block={action_block}."
        )
    return action_steps.reshape(
        int(action_steps.shape[0]),
        int(action_steps.shape[1]) // int(action_block),
        int(action_steps.shape[2]) * int(action_block),
    )


def rollout_terminal(
    world_model: torch.nn.Module,
    z_start: torch.Tensor,
    action_steps: torch.Tensor,
    *,
    action_block: int,
    history_size: int,
) -> torch.Tensor:
    action_blocks = action_steps_to_blocks(action_steps, action_block=action_block)
    return latent_rollout(
        world_model=world_model,
        z_context=z_start,
        action_blocks=action_blocks,
        history_size=history_size,
        return_sequence=False,
        freeze_world_model=True,
    )["z_terminal"]


def smoothness_loss(action_steps: torch.Tensor) -> torch.Tensor:
    if int(action_steps.shape[1]) <= 1:
        return action_steps.new_zeros(())
    return (action_steps[:, 1:, :] - action_steps[:, :-1, :]).square().mean()


def compute_corrector_loss(
    model: ActionChunkCorrector,
    world_model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    spec: CorrectorTrainingSpec,
    *,
    history_size: int = 3,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    required = {"z_cur", "z_goal", "clean_prefix", "noisy_prefix", "u_remain", "target_remain"}
    missing = required.difference(batch.keys())
    if missing:
        raise KeyError(f"Corrector batch is missing required keys: {sorted(missing)}.")

    for parameter in world_model.parameters():
        parameter.requires_grad_(False)
    world_model.eval()

    z_cur = batch["z_cur"]
    z_goal = batch["z_goal"]
    clean_prefix = batch["clean_prefix"]
    noisy_prefix = batch["noisy_prefix"]
    u_remain = batch["u_remain"]
    target_remain = batch["target_remain"]
    if "action_block" in batch:
        action_block_values = batch["action_block"]
        if torch.is_tensor(action_block_values):
            flat_blocks = action_block_values.detach().reshape(-1).cpu()
            action_block = int(flat_blocks[0].item())
            if not torch.all(flat_blocks == action_block):
                raise ValueError("All samples in a corrector batch must use the same action_block.")
        else:
            action_block = int(action_block_values)
    else:
        action_block = int(spec.action_block or 1)

    with torch.no_grad():
        z_pred = rollout_terminal(
            world_model,
            z_cur,
            clean_prefix,
            action_block=action_block,
            history_size=history_size,
        ).detach()
        z_real_like = rollout_terminal(
            world_model,
            z_cur,
            noisy_prefix,
            action_block=action_block,
            history_size=history_size,
        ).detach()
        error_latent = z_real_like - z_pred

    u_corr = model(z_real_like, z_goal, error_latent, u_remain)
    action_loss = F.smooth_l1_loss(u_corr, target_remain)
    if float(spec.lambda_goal) != 0.0:
        z_corr = rollout_terminal(
            world_model,
            z_real_like,
            u_corr,
            action_block=action_block,
            history_size=history_size,
        )
        goal_loss = (z_corr - z_goal.detach()).square().mean()
    else:
        goal_loss = u_corr.new_zeros(())
    smooth_loss = smoothness_loss(u_corr)
    total = (
        float(spec.lambda_action) * action_loss
        + float(spec.lambda_goal) * goal_loss
        + float(spec.lambda_smooth) * smooth_loss
    )
    return total, {
        "loss": total.detach(),
        "action_loss": action_loss.detach(),
        "goal_loss": goal_loss.detach(),
        "smoothness_loss": smooth_loss.detach(),
    }


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train_one_epoch(
    model: ActionChunkCorrector,
    world_model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    spec: CorrectorTrainingSpec,
    *,
    device: torch.device,
    history_size: int,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {"loss": 0.0, "action_loss": 0.0, "goal_loss": 0.0, "smoothness_loss": 0.0}
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, terms = compute_corrector_loss(
            model,
            world_model,
            batch,
            spec,
            history_size=history_size,
        )
        loss.backward()
        optimizer.step()
        batch_size = int(batch["z_cur"].shape[0])
        count += batch_size
        for key in totals:
            totals[key] += float(terms[key].item()) * batch_size
    denom = max(count, 1)
    return {key: value / denom for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: ActionChunkCorrector,
    world_model: torch.nn.Module,
    loader: DataLoader,
    spec: CorrectorTrainingSpec,
    *,
    device: torch.device,
    history_size: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {"loss": 0.0, "action_loss": 0.0, "goal_loss": 0.0, "smoothness_loss": 0.0}
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        loss, terms = compute_corrector_loss(
            model,
            world_model,
            batch,
            spec,
            history_size=history_size,
        )
        batch_size = int(batch["z_cur"].shape[0])
        count += batch_size
        for key in totals:
            totals[key] += float(terms[key].item()) * batch_size
    denom = max(count, 1)
    return {key: value / denom for key, value in totals.items()}


def split_train_val(
    dataset: Dataset,
    *,
    val_split: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    if not (0.0 < float(val_split) < 1.0):
        raise ValueError(f"val_split must be in (0, 1), got {val_split}.")
    val_len = max(1, int(math.floor(len(dataset) * float(val_split))))
    train_len = len(dataset) - val_len
    if train_len <= 0:
        raise ValueError(f"Dataset too small for val_split={val_split}.")
    generator = torch.Generator().manual_seed(int(seed))
    return random_split(dataset, [train_len, val_len], generator=generator)


def limit_dataset_samples(
    dataset: Dataset,
    *,
    max_samples: int | None,
    seed: int,
) -> Dataset:
    if max_samples in [None, 0]:
        return dataset
    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError(f"max_samples must be positive when set, got {max_samples}.")
    if max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def load_frozen_world_model(policy_path: str, device: torch.device) -> torch.nn.Module:
    world_model = swm.policy.AutoCostModel(policy_path)
    world_model = world_model.to(device)
    world_model.eval()
    world_model.requires_grad_(False)
    return world_model


def hydra_main(cfg: DictConfig) -> None:
    print("[config]")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    device = torch.device(str(cfg.train.device))
    torch.manual_seed(int(cfg.train.seed))
    dataset_bundle = load_dataset_bundle(cfg.task.planner_dataset_path)
    spec = CorrectorTrainingSpec(
        correction_interval=int(cfg.corrective.correction_interval),
        action_block=None if cfg.corrective.action_block in [None, "null"] else int(cfg.corrective.action_block),
        noise_std=float(cfg.training.noise_std),
        noise_prob=float(cfg.training.noise_prob),
        remainder_noise_std=(
            None
            if cfg.training.remainder_noise_std in [None, "null"]
            else float(cfg.training.remainder_noise_std)
        ),
        lambda_action=float(cfg.training.lambda_action),
        lambda_goal=float(cfg.training.lambda_goal),
        lambda_smooth=float(cfg.training.lambda_smooth),
        seed=int(cfg.train.seed),
    )
    base_dataset = CorrectorTrainingDataset(dataset_bundle, spec)
    dataset: Dataset = base_dataset
    max_samples = cfg.train.get("max_samples", None)
    if max_samples not in [None, "", "null"]:
        dataset = limit_dataset_samples(
            base_dataset,
            max_samples=int(max_samples),
            seed=int(cfg.train.seed),
        )
    train_set, val_set = split_train_val(
        dataset,
        val_split=float(cfg.train.val_split),
        seed=int(cfg.train.seed),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg.train.val_batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    model = ActionChunkCorrector(
        ActionChunkCorrectorConfig(
            latent_dim=base_dataset.latent_dim,
            action_dim=base_dataset.action_dim,
            remain_horizon=base_dataset.remain_horizon,
            hidden_dim=int(cfg.model.hidden_dim),
            num_layers=int(cfg.model.num_layers),
            dropout=float(cfg.model.dropout),
            activation=str(cfg.model.activation),
            predict_residual=bool(cfg.model.predict_residual),
            residual_scale=float(cfg.model.residual_scale),
        )
    ).to(device)
    world_model = load_frozen_world_model(str(cfg.task.wm_policy), device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    output_dir = Path(str(cfg.task.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = output_dir / "corrector_best_bundle.pt"
    last_path = output_dir / "corrector_last_bundle.pt"
    history_size = int(cfg.training.history_size)
    metadata = {
        "task": str(cfg.task.name),
        "planner_dataset_path": str(cfg.task.planner_dataset_path),
        "wm_policy": str(cfg.task.wm_policy),
        "correction_interval": int(spec.correction_interval),
        "effective_prefix_steps": int(base_dataset.prefix_steps),
        "remain_horizon": int(base_dataset.remain_horizon),
        "action_block": int(base_dataset.action_block),
        "training_spec": asdict(spec),
    }

    for epoch in range(1, int(cfg.train.epochs) + 1):
        train_metrics = train_one_epoch(
            model,
            world_model,
            train_loader,
            optimizer,
            spec,
            device=device,
            history_size=history_size,
        )
        val_metrics = evaluate(
            model,
            world_model,
            val_loader,
            spec,
            device=device,
            history_size=history_size,
        )
        print(
            "[epoch] "
            f"{epoch}/{int(cfg.train.epochs)} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_action={val_metrics['action_loss']:.6f} "
            f"val_goal={val_metrics['goal_loss']:.6f}"
        )
        if float(val_metrics["loss"]) < best_val:
            best_val = float(val_metrics["loss"])
            save_corrector_bundle(model, best_path, metadata={**metadata, "best_val_loss": best_val})
            print(f"[save] best corrector updated: {best_path}")
        save_corrector_bundle(model, last_path, metadata={**metadata, "last_val_loss": float(val_metrics["loss"])})
    print(f"[done] best_bundle={best_path} last_bundle={last_path}")


__all__ = [
    "CorrectorTrainingDataset",
    "CorrectorTrainingSpec",
    "action_steps_to_blocks",
    "compute_corrector_loss",
    "evaluate",
    "hydra_main",
    "limit_dataset_samples",
    "load_dataset_bundle",
    "rollout_terminal",
    "train_one_epoch",
]
