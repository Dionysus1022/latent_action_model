from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data import Subset
from tqdm.auto import tqdm as _tqdm

from planners.gc_idm_model import GCIDMModel, GCIDMModelConfig, save_gc_idm_bundle


def progress_iter(iterable, *, desc: str, total: int | None = None, unit: str = "it", leave: bool = True):
    """Use tqdm only for interactive terminals; keep log files clean otherwise."""
    force_progress = os.environ.get("GC_IDM_FORCE_PROGRESS", "").strip().lower() in {"1", "true", "yes", "on"}
    if not force_progress and not sys.stderr.isatty():
        return iterable
    return _tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave)


class GCIDMTensorDataset(Dataset):
    """Dataset wrapper for GC-IDM latent/action supervision.

    Expected schema:
        z_cur: [N, latent_dim]
        z_goal: [N, latent_dim]
        horizon: [N]
        action: [N, action_dim]
        meta: list[dict]
    """

    def __init__(self, dataset_bundle: dict[str, Any]):
        required = {"z_cur", "z_goal", "horizon", "action", "meta"}
        missing = required.difference(dataset_bundle.keys())
        if missing:
            raise KeyError(f"Dataset bundle is missing required keys: {sorted(missing)}.")

        self.z_cur = dataset_bundle["z_cur"].float()
        self.z_goal = dataset_bundle["z_goal"].float()
        self.horizon = dataset_bundle["horizon"].long()
        self.action = dataset_bundle["action"].float()
        self.meta = dataset_bundle["meta"]
        self.build_info = dataset_bundle.get("build_info", {})

        if self.z_cur.ndim != 2:
            raise ValueError(f"z_cur must have shape [N, latent_dim], got {tuple(self.z_cur.shape)}.")
        if self.z_goal.ndim != 2:
            raise ValueError(f"z_goal must have shape [N, latent_dim], got {tuple(self.z_goal.shape)}.")
        if self.z_cur.shape != self.z_goal.shape:
            raise ValueError(
                f"z_cur and z_goal must have matching shape, got {tuple(self.z_cur.shape)} and {tuple(self.z_goal.shape)}."
            )
        if self.horizon.ndim != 1:
            raise ValueError(f"horizon must have shape [N], got {tuple(self.horizon.shape)}.")
        if self.action.ndim != 2:
            raise ValueError(f"action must have shape [N, action_dim], got {tuple(self.action.shape)}.")
        if int(self.z_cur.shape[0]) != int(self.action.shape[0]) or int(self.z_cur.shape[0]) != int(self.horizon.shape[0]):
            raise ValueError("z_cur, z_goal, horizon, and action must have the same sample count.")
        if len(self.meta) != int(self.z_cur.shape[0]):
            raise ValueError(f"meta length {len(self.meta)} must match sample count {self.z_cur.shape[0]}.")

    @property
    def latent_dim(self) -> int:
        return int(self.z_cur.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[-1])

    @property
    def max_horizon(self) -> int:
        return int(torch.max(self.horizon).item())

    def __len__(self) -> int:
        return int(self.z_cur.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "z_cur": self.z_cur[index].clone(),
            "z_goal": self.z_goal[index].clone(),
            "horizon": self.horizon[index].clone(),
            "action": self.action[index].clone(),
            "meta": dict(self.meta[index]),
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GC-IDM on frozen LeWM latent/action supervision.",
    )
    parser.add_argument("--dataset-path", required=True, help="Path to a GC-IDM dataset .pt bundle.")
    parser.add_argument("--output-dir", required=True, help="Directory for saved GC-IDM bundles.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--val-batch-size", type=int, default=1024)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--horizon-embedding-dim", type=int, default=64)
    parser.add_argument("--horizon-mlp-layers", type=int, default=2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--checkpoint-selection",
        choices=["last", "best"],
        default="last",
        help="Which checkpoint the default gc_idm_best_bundle.pt path should contain.",
    )
    parser.add_argument(
        "--sample-level-split",
        action="store_true",
        help="Use random sample-level validation split instead of episode-level split.",
    )
    parser.add_argument(
        "--disable-cosine-scheduler",
        action="store_true",
        help="Disable the paper-style cosine schedule from lr to lr/100.",
    )
    parser.add_argument(
        "--diagnose-horizon-buckets",
        action="store_true",
        help="Print validation MSE/R2 by horizon bucket after training.",
    )
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def load_dataset_bundle(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset bundle not found: {dataset_path}")
    return torch.load(dataset_path, map_location="cpu")


def split_dataset(
    dataset: GCIDMTensorDataset,
    *,
    val_split: float,
    seed: int,
    split_by_episode: bool = True,
) -> tuple[Dataset, Dataset]:
    if not (0.0 < float(val_split) < 1.0):
        raise ValueError(f"val_split must be in (0, 1), got {val_split}.")
    if split_by_episode:
        episode_ids = np.asarray([sample_meta.get("episode_id", sample_meta.get("episode", idx)) for idx, sample_meta in enumerate(dataset.meta)])
        unique_episodes = np.unique(episode_ids)
        if unique_episodes.shape[0] >= 2:
            rng = np.random.default_rng(int(seed))
            shuffled = np.array(unique_episodes, copy=True)
            rng.shuffle(shuffled)
            val_episode_count = max(1, int(round(unique_episodes.shape[0] * float(val_split))))
            val_episode_count = min(val_episode_count, unique_episodes.shape[0] - 1)
            val_episodes = set(shuffled[:val_episode_count].tolist())
            train_indices = [idx for idx, ep_id in enumerate(episode_ids.tolist()) if ep_id not in val_episodes]
            val_indices = [idx for idx, ep_id in enumerate(episode_ids.tolist()) if ep_id in val_episodes]
            if train_indices and val_indices:
                return Subset(dataset, train_indices), Subset(dataset, val_indices)

    val_size = max(1, int(round(len(dataset) * float(val_split))))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for the requested validation split.")
    generator = torch.Generator().manual_seed(int(seed))
    return random_split(dataset, [train_size, val_size], generator=generator)


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    lr: float,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    if int(epochs) <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}.")
    if float(lr) <= 0.0:
        raise ValueError(f"lr must be positive, got {lr}.")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(epochs),
        eta_min=float(lr) / 100.0,
    )


def evaluate(model: GCIDMModel, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_zero_loss = 0.0
    total_count = 0
    pred_sum = None
    pred_sq_sum = None
    action_sum = None
    action_sq_sum = None
    with torch.inference_mode():
        for batch in loader:
            z_cur = batch["z_cur"].to(device)
            z_goal = batch["z_goal"].to(device)
            horizon = batch["horizon"].to(device)
            action = batch["action"].to(device)
            pred = model(z_cur, z_goal, horizon)
            loss = F.mse_loss(pred, action, reduction="sum")
            total_loss += float(loss.detach().cpu())
            total_zero_loss += float(torch.sum(action * action).detach().cpu())
            total_count += int(action.numel())
            pred_batch_sum = pred.sum(dim=0).detach().cpu()
            pred_batch_sq_sum = (pred * pred).sum(dim=0).detach().cpu()
            action_batch_sum = action.sum(dim=0).detach().cpu()
            action_batch_sq_sum = (action * action).sum(dim=0).detach().cpu()
            if pred_sum is None:
                pred_sum = pred_batch_sum
                pred_sq_sum = pred_batch_sq_sum
                action_sum = action_batch_sum
                action_sq_sum = action_batch_sq_sum
            else:
                pred_sum += pred_batch_sum
                pred_sq_sum += pred_batch_sq_sum
                action_sum += action_batch_sum
                action_sq_sum += action_batch_sq_sum
    mse = total_loss / max(1, total_count)
    zero_mse = total_zero_loss / max(1, total_count)
    r2_vs_zero = 1.0 - mse / zero_mse if zero_mse > 0.0 else float("nan")
    if pred_sum is not None and pred_sq_sum is not None and action_sum is not None and action_sq_sum is not None:
        sample_count = max(1, total_count // int(action_sum.numel()))
        pred_mean = pred_sum / sample_count
        action_mean = action_sum / sample_count
        pred_var = torch.clamp(pred_sq_sum / sample_count - pred_mean * pred_mean, min=0.0)
        action_var = torch.clamp(action_sq_sum / sample_count - action_mean * action_mean, min=0.0)
        pred_std_mean = float(torch.sqrt(pred_var).mean())
        action_std_mean = float(torch.sqrt(action_var).mean())
    else:
        pred_std_mean = float("nan")
        action_std_mean = float("nan")
    return {
        "mse": mse,
        "zero_mse": zero_mse,
        "r2_vs_zero": r2_vs_zero,
        "pred_std_mean": pred_std_mean,
        "action_std_mean": action_std_mean,
    }


def evaluate_horizon_buckets(
    model: GCIDMModel,
    loader: DataLoader,
    device: torch.device,
    buckets: list[tuple[int, int]] | None = None,
) -> list[dict[str, float]]:
    buckets = buckets or [(1, 1), (2, 2), (3, 5), (6, 10), (11, 20), (21, 30), (31, 40), (41, 50)]
    model.eval()
    stats = {
        (int(lo), int(hi)): {"loss": 0.0, "zero_loss": 0.0, "count": 0}
        for lo, hi in buckets
    }
    with torch.inference_mode():
        for batch in loader:
            z_cur = batch["z_cur"].to(device)
            z_goal = batch["z_goal"].to(device)
            horizon = batch["horizon"].to(device)
            action = batch["action"].to(device)
            pred = model(z_cur, z_goal, horizon)
            per_sample_loss = torch.sum((pred - action) ** 2, dim=-1)
            per_sample_zero_loss = torch.sum(action * action, dim=-1)
            action_dim = int(action.shape[-1])
            for lo, hi in stats:
                mask = (horizon >= lo) & (horizon <= hi)
                if not bool(mask.any()):
                    continue
                stats[(lo, hi)]["loss"] += float(per_sample_loss[mask].sum().detach().cpu())
                stats[(lo, hi)]["zero_loss"] += float(per_sample_zero_loss[mask].sum().detach().cpu())
                stats[(lo, hi)]["count"] += int(mask.sum().detach().cpu()) * action_dim

    rows: list[dict[str, float]] = []
    for (lo, hi), values in stats.items():
        count = int(values["count"])
        if count <= 0:
            continue
        mse = float(values["loss"]) / count
        zero_mse = float(values["zero_loss"]) / count
        rows.append(
            {
                "horizon_lo": float(lo),
                "horizon_hi": float(hi),
                "mse": mse,
                "zero_mse": zero_mse,
                "r2_vs_zero": 1.0 - mse / zero_mse if zero_mse > 0.0 else float("nan"),
                "samples": float(count // max(1, action_dim)),
            }
        )
    return rows


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_bundle = load_dataset_bundle(args.dataset_path)
    dataset = GCIDMTensorDataset(dataset_bundle)
    train_dataset, val_dataset = split_dataset(
        dataset,
        val_split=float(args.val_split),
        seed=int(args.seed),
        split_by_episode=not bool(args.sample_level_split),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.val_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        drop_last=False,
    )

    model = GCIDMModel(
        GCIDMModelConfig(
            latent_dim=dataset.latent_dim,
            action_dim=dataset.action_dim,
            hidden_dim=int(args.hidden_dim),
            num_layers=int(args.num_layers),
            dropout=float(args.dropout),
            horizon_embedding_dim=int(args.horizon_embedding_dim),
            horizon_mlp_layers=int(args.horizon_mlp_layers),
            max_horizon=int(dataset.max_horizon),
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = None
    if not bool(args.disable_cosine_scheduler):
        scheduler = build_cosine_scheduler(optimizer, epochs=int(args.epochs), lr=float(args.lr))

    best_val = float("inf")
    best_path = output_dir / "gc_idm_best_bundle.pt"
    last_path = output_dir / "gc_idm_last_bundle.pt"
    history: list[dict[str, float]] = []

    epoch_iter = progress_iter(
        range(1, int(args.epochs) + 1),
        desc="gc-idm epochs",
        total=int(args.epochs),
        unit="epoch",
        leave=True,
    )
    for epoch in epoch_iter:
        model.train()
        running_loss = 0.0
        running_count = 0
        train_batches = progress_iter(
            train_loader,
            desc=f"gc-idm train {epoch}",
            total=len(train_loader),
            unit="batch",
            leave=False,
        )
        for step, batch in enumerate(train_batches, start=1):
            z_cur = batch["z_cur"].to(device)
            z_goal = batch["z_goal"].to(device)
            horizon = batch["horizon"].to(device)
            action = batch["action"].to(device)

            pred = model(z_cur, z_goal, horizon)
            loss = F.mse_loss(pred, action)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
            optimizer.step()

            running_loss += float(loss.detach().cpu()) * int(action.numel())
            running_count += int(action.numel())
            if int(args.log_every) > 0 and step % int(args.log_every) == 0:
                print(f"[train] epoch={epoch} step={step} mse={running_loss / max(1, running_count):.6f}")

        train_mse = running_loss / max(1, running_count)
        val_metrics = evaluate(model, val_loader, device)
        val_mse = float(val_metrics["mse"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_mse": train_mse,
                "val_mse": val_mse,
                "val_zero_mse": float(val_metrics["zero_mse"]),
                "val_r2_vs_zero": float(val_metrics["r2_vs_zero"]),
                "lr": current_lr,
            }
        )
        print(
            f"[epoch] epoch={epoch} train_mse={train_mse:.6f} "
            f"val_mse={val_mse:.6f} val_zero_mse={float(val_metrics['zero_mse']):.6f} "
            f"val_r2_vs_zero={float(val_metrics['r2_vs_zero']):.6f} "
            f"pred_std_mean={float(val_metrics['pred_std_mean']):.6f} "
            f"action_std_mean={float(val_metrics['action_std_mean']):.6f} "
            f"lr={current_lr:.8f}"
        )

        metadata = {
            "dataset_path": str(Path(args.dataset_path).expanduser().resolve()),
            "build_info": dataset_bundle.get("build_info", {}),
            "history": history,
            "training_hyperparameters": vars(args),
        }
        save_gc_idm_bundle(model, last_path, metadata=metadata)
        if str(args.checkpoint_selection) == "last":
            save_gc_idm_bundle(model, best_path, metadata={**metadata, "selected_checkpoint": "last"})
        if val_mse < best_val:
            best_val = val_mse
            if str(args.checkpoint_selection) == "best":
                save_gc_idm_bundle(
                    model,
                    best_path,
                    metadata={**metadata, "best_val_mse": best_val, "selected_checkpoint": "best"},
                )
            print(f"[save] best_val_mse={best_val:.6f} checkpoint_selection={args.checkpoint_selection}")
        if scheduler is not None:
            scheduler.step()

    if bool(args.diagnose_horizon_buckets):
        for row in evaluate_horizon_buckets(model, val_loader, device):
            print(
                "[horizon-bucket] "
                f"h={int(row['horizon_lo'])}-{int(row['horizon_hi'])} "
                f"samples={int(row['samples'])} mse={float(row['mse']):.6f} "
                f"zero_mse={float(row['zero_mse']):.6f} "
                f"r2_vs_zero={float(row['r2_vs_zero']):.6f}"
            )

    return {
        "best_bundle": str(best_path),
        "last_bundle": str(last_path),
        "best_val_mse": best_val,
        "history": history,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_training(args)
    print(f"[done] best_bundle={summary['best_bundle']} best_val_mse={summary['best_val_mse']:.6f}")


if __name__ == "__main__":
    main()
