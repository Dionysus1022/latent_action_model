from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class SinglePeakModelConfig:
    """Hyperparameters for the minimal single-peak planner model."""

    latent_dim: int
    plan_horizon: int
    action_dim: int
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.0
    activation: str = "gelu"

    @property
    def input_dim(self) -> int:
        return int(3 * self.latent_dim)

    @property
    def action_chunk_dim(self) -> int:
        return int(self.plan_horizon * self.action_dim)


def infer_model_config_from_dataset_bundle(dataset_bundle: dict[str, Any]) -> SinglePeakModelConfig:
    """Infer runtime-critical model dimensions from a built dataset bundle."""
    if "z_cur" not in dataset_bundle:
        raise KeyError("dataset_bundle must contain 'z_cur'.")
    if "teacher_plan" not in dataset_bundle:
        raise KeyError("dataset_bundle must contain 'teacher_plan'.")
    if "meta" not in dataset_bundle or len(dataset_bundle["meta"]) == 0:
        raise KeyError("dataset_bundle must contain a non-empty 'meta' list.")

    z_cur = dataset_bundle["z_cur"]
    teacher_plan = dataset_bundle["teacher_plan"]
    meta0 = dataset_bundle["meta"][0]

    if not torch.is_tensor(z_cur) or z_cur.ndim != 2:
        raise ValueError(
            f"dataset_bundle['z_cur'] must have shape [N, latent_dim], got {type(z_cur)} with shape {getattr(z_cur, 'shape', None)}."
        )
    if not torch.is_tensor(teacher_plan) or teacher_plan.ndim != 2:
        raise ValueError(
            "dataset_bundle['teacher_plan'] must have shape [N, action_chunk_dim], "
            f"got {type(teacher_plan)} with shape {getattr(teacher_plan, 'shape', None)}."
        )

    latent_dim = int(z_cur.shape[-1])
    plan_horizon = int(meta0["plan_horizon"])
    action_dim = int(meta0["action_dim"])
    action_chunk_dim = int(meta0["action_chunk_dim"])

    expected_chunk_dim = int(plan_horizon * action_dim)
    if action_chunk_dim != expected_chunk_dim:
        raise ValueError(
            f"action_chunk_dim mismatch in dataset metadata: {action_chunk_dim} != {expected_chunk_dim}."
        )
    if int(teacher_plan.shape[-1]) != action_chunk_dim:
        raise ValueError(
            f"teacher_plan width {teacher_plan.shape[-1]} does not match action_chunk_dim {action_chunk_dim}."
        )

    return SinglePeakModelConfig(
        latent_dim=latent_dim,
        plan_horizon=plan_horizon,
        action_dim=action_dim,
    )


class SinglePeakPlannerModel(nn.Module):
    """Minimal single-peak planner.

    Inputs:
        z_cur: [B, latent_dim] or [latent_dim]
        z_goal: [B, latent_dim] or [latent_dim]

    Internal feature:
        x = concat([z_cur, z_goal, z_goal - z_cur])  # [B, 3 * latent_dim]

    Output:
        u_hat: [B, action_chunk_dim] or [action_chunk_dim]
    """

    def __init__(self, config: SinglePeakModelConfig):
        super().__init__()
        self.config = config
        self.trunk = build_mlp_trunk(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.action_head = nn.Linear(config.hidden_dim, config.action_chunk_dim)

    @property
    def latent_dim(self) -> int:
        return self.config.latent_dim

    @property
    def plan_horizon(self) -> int:
        return self.config.plan_horizon

    @property
    def action_dim(self) -> int:
        return self.config.action_dim

    @property
    def action_chunk_dim(self) -> int:
        return self.config.action_chunk_dim

    @property
    def input_dim(self) -> int:
        return self.config.input_dim

    def encode_inputs(
        self,
        z_cur: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Build the planner input.

        z_cur: [B, latent_dim] or [latent_dim]
        z_goal: [B, latent_dim] or [latent_dim]
        x: [B, 3 * latent_dim]
        """
        z_cur, squeezed = ensure_batched_latents(z_cur, self.latent_dim, name="z_cur")
        z_goal, squeezed_goal = ensure_batched_latents(z_goal, self.latent_dim, name="z_goal")
        if squeezed != squeezed_goal:
            raise ValueError("z_cur and z_goal must both be batched or both be unbatched.")
        if z_cur.shape != z_goal.shape:
            raise ValueError(
                f"z_cur and z_goal must have matching shapes, got {tuple(z_cur.shape)} and {tuple(z_goal.shape)}."
            )

        z_delta = z_goal - z_cur  # [B, latent_dim]
        x = torch.cat([z_cur, z_goal, z_delta], dim=-1)  # [B, 3 * latent_dim]
        return x, squeezed

    def forward(
        self,
        z_cur: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a flattened action chunk.

        z_cur: [B, latent_dim] or [latent_dim]
        z_goal: [B, latent_dim] or [latent_dim]
        u_hat: [B, action_chunk_dim] or [action_chunk_dim]
        """
        x, squeezed = self.encode_inputs(z_cur, z_goal)  # [B, 3 * latent_dim]
        h = self.trunk(x)  # [B, hidden_dim]
        u_hat = self.action_head(h)  # [B, action_chunk_dim]
        if squeezed:
            return u_hat[0]  # [action_chunk_dim]
        return u_hat  # [B, action_chunk_dim]

    def forward_dict(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward using dataset-style keys.

        batch["z_cur"]: [B, latent_dim]
        batch["z_goal"]: [B, latent_dim]
        out["u_hat"]: [B, action_chunk_dim]
        """
        if "z_cur" not in batch or "z_goal" not in batch:
            raise KeyError("batch must contain 'z_cur' and 'z_goal'.")
        u_hat = self.forward(batch["z_cur"], batch["z_goal"])
        return {"u_hat": u_hat}


@dataclass
class SinglePeakPlannerBundle:
    """Portable save/load bundle for runtime and training reuse."""

    model_state_dict: dict[str, torch.Tensor]
    model_hyperparameters: dict[str, Any]
    latent_dim: int
    plan_horizon: int
    action_dim: int
    action_chunk_dim: int
    input_dim: int
    bundle_version: int = 1

    def instantiate_model(self, map_location: str | torch.device | None = None) -> SinglePeakPlannerModel:
        config = SinglePeakModelConfig(**self.model_hyperparameters)
        model = SinglePeakPlannerModel(config)
        state_dict = self.model_state_dict
        if map_location is not None:
            state_dict = {
                key: value.to(map_location) if torch.is_tensor(value) else value
                for key, value in state_dict.items()
            }
        model.load_state_dict(state_dict)
        return model

    def as_dict(self) -> dict[str, Any]:
        return {
            "bundle_version": int(self.bundle_version),
            "model_state_dict": self.model_state_dict,
            "model_hyperparameters": self.model_hyperparameters,
            "latent_dim": int(self.latent_dim),
            "plan_horizon": int(self.plan_horizon),
            "action_dim": int(self.action_dim),
            "action_chunk_dim": int(self.action_chunk_dim),
            "input_dim": int(self.input_dim),
        }


def build_mlp_trunk(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    activation: str,
) -> nn.Sequential:
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}.")
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}.")

    act_factory = get_activation_factory(activation)
    layers: list[nn.Module] = []
    in_dim = input_dim
    for layer_idx in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_factory())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim
    return nn.Sequential(*layers)


def get_activation_factory(name: str):
    activation_name = name.lower()
    if activation_name == "gelu":
        return nn.GELU
    if activation_name == "relu":
        return nn.ReLU
    if activation_name == "silu":
        return nn.SiLU
    raise ValueError(f"Unsupported activation '{name}'.")


def ensure_batched_latents(
    z: torch.Tensor,
    latent_dim: int,
    name: str,
) -> tuple[torch.Tensor, bool]:
    if not torch.is_tensor(z):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(z)}.")
    if z.ndim == 1:
        if int(z.shape[0]) != latent_dim:
            raise ValueError(
                f"{name} must have shape [latent_dim], got {tuple(z.shape)} for latent_dim={latent_dim}."
            )
        return z.unsqueeze(0), True  # [1, latent_dim]
    if z.ndim == 2:
        if int(z.shape[-1]) != latent_dim:
            raise ValueError(
                f"{name} must have shape [B, latent_dim], got {tuple(z.shape)} for latent_dim={latent_dim}."
            )
        return z, False  # [B, latent_dim]
    raise ValueError(
        f"{name} must have shape [latent_dim] or [B, latent_dim], got {tuple(z.shape)}."
    )


def build_single_peak_bundle(model: SinglePeakPlannerModel) -> SinglePeakPlannerBundle:
    config = model.config
    return SinglePeakPlannerBundle(
        model_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        model_hyperparameters=asdict(config),
        latent_dim=int(config.latent_dim),
        plan_horizon=int(config.plan_horizon),
        action_dim=int(config.action_dim),
        action_chunk_dim=int(config.action_chunk_dim),
        input_dim=int(config.input_dim),
    )


def save_single_peak_bundle(
    model: SinglePeakPlannerModel,
    path: str | Path,
) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = build_single_peak_bundle(model)
    torch.save(bundle.as_dict(), output_path)
    return output_path


def load_single_peak_bundle(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> SinglePeakPlannerBundle:
    bundle_dict = torch.load(Path(path).expanduser().resolve(), map_location=map_location)
    validate_bundle_dict(bundle_dict)
    return SinglePeakPlannerBundle(**bundle_dict)


def load_single_peak_model(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> SinglePeakPlannerModel:
    bundle = load_single_peak_bundle(path, map_location=map_location)
    model = bundle.instantiate_model()
    model.load_state_dict(bundle.model_state_dict)
    return model


def validate_bundle_dict(bundle_dict: dict[str, Any]) -> None:
    required_keys = {
        "bundle_version",
        "model_state_dict",
        "model_hyperparameters",
        "latent_dim",
        "plan_horizon",
        "action_dim",
        "action_chunk_dim",
        "input_dim",
    }
    missing = required_keys.difference(bundle_dict.keys())
    if missing:
        raise KeyError(f"Bundle is missing required keys: {sorted(missing)}.")

    model_hparams = bundle_dict["model_hyperparameters"]
    if not isinstance(model_hparams, dict):
        raise ValueError("bundle['model_hyperparameters'] must be a dict.")

    config = SinglePeakModelConfig(**model_hparams)
    if int(bundle_dict["latent_dim"]) != int(config.latent_dim):
        raise ValueError("bundle latent_dim does not match model_hyperparameters.")
    if int(bundle_dict["plan_horizon"]) != int(config.plan_horizon):
        raise ValueError("bundle plan_horizon does not match model_hyperparameters.")
    if int(bundle_dict["action_dim"]) != int(config.action_dim):
        raise ValueError("bundle action_dim does not match model_hyperparameters.")
    if int(bundle_dict["action_chunk_dim"]) != int(config.action_chunk_dim):
        raise ValueError("bundle action_chunk_dim does not match model_hyperparameters.")
    if int(bundle_dict["input_dim"]) != int(config.input_dim):
        raise ValueError("bundle input_dim does not match model_hyperparameters.")
