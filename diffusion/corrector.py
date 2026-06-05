from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from planners.single_peak_model import build_mlp_trunk, ensure_batched_latents


@dataclass
class ActionChunkCorrectorConfig:
    """Hyperparameters for learned corrective action-chunk repair."""

    latent_dim: int
    action_dim: int
    remain_horizon: int
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.1
    activation: str = "gelu"
    predict_residual: bool = True
    residual_scale: float = 1.0

    @property
    def remain_action_dim(self) -> int:
        return int(self.remain_horizon * self.action_dim)

    @property
    def input_dim(self) -> int:
        return int(3 * self.latent_dim + self.remain_action_dim)


class ActionChunkCorrector(nn.Module):
    """MLP corrector for the remaining part of an action chunk.

    Inputs:
        z_real: [B, latent_dim] or [latent_dim]
        z_goal: [B, latent_dim] or [latent_dim]
        error_latent: [B, latent_dim] or [latent_dim]
        u_remain: [B, remain_horizon, action_dim] or [remain_horizon, action_dim]

    Output:
        corrected remaining actions with the same shape as u_remain.
    """

    def __init__(self, config: ActionChunkCorrectorConfig):
        super().__init__()
        self.config = config
        self.trunk = build_mlp_trunk(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.action_head = nn.Linear(config.hidden_dim, config.remain_action_dim)

    @property
    def latent_dim(self) -> int:
        return int(self.config.latent_dim)

    @property
    def action_dim(self) -> int:
        return int(self.config.action_dim)

    @property
    def remain_horizon(self) -> int:
        return int(self.config.remain_horizon)

    @property
    def remain_action_dim(self) -> int:
        return int(self.config.remain_action_dim)

    def _validate_actions(self, u_remain: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if not torch.is_tensor(u_remain):
            raise TypeError(f"u_remain must be a torch.Tensor, got {type(u_remain)}.")
        if u_remain.ndim == 2:
            expected_shape = (self.remain_horizon, self.action_dim)
            if tuple(u_remain.shape) != expected_shape:
                raise ValueError(
                    f"u_remain must have shape {expected_shape}, got {tuple(u_remain.shape)}."
                )
            return u_remain.unsqueeze(0), True
        if u_remain.ndim == 3:
            expected_tail = (self.remain_horizon, self.action_dim)
            if tuple(u_remain.shape[1:]) != expected_tail:
                raise ValueError(
                    f"u_remain must have shape [B, {expected_tail[0]}, {expected_tail[1]}], "
                    f"got {tuple(u_remain.shape)}."
                )
            return u_remain, False
        raise ValueError(
            "u_remain must have shape [remain_horizon, action_dim] or "
            f"[B, remain_horizon, action_dim], got {tuple(u_remain.shape)}."
        )

    def encode_inputs(
        self,
        z_real: torch.Tensor,
        z_goal: torch.Tensor,
        error_latent: torch.Tensor,
        u_remain: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        z_real, squeezed_latent = ensure_batched_latents(z_real, self.latent_dim, name="z_real")
        z_goal, squeezed_goal = ensure_batched_latents(z_goal, self.latent_dim, name="z_goal")
        error_latent, squeezed_error = ensure_batched_latents(
            error_latent,
            self.latent_dim,
            name="error_latent",
        )
        u_remain, squeezed_action = self._validate_actions(u_remain)
        if not (squeezed_latent == squeezed_goal == squeezed_error == squeezed_action):
            raise ValueError("Corrector inputs must all be batched or all be unbatched.")
        batch_size = int(z_real.shape[0])
        if int(z_goal.shape[0]) != batch_size or int(error_latent.shape[0]) != batch_size:
            raise ValueError("Latent inputs must have matching batch sizes.")
        if int(u_remain.shape[0]) != batch_size:
            raise ValueError(
                f"u_remain batch size {u_remain.shape[0]} does not match latent batch size {batch_size}."
            )
        u_flat = u_remain.reshape(batch_size, self.remain_action_dim)
        x = torch.cat([z_real, z_goal, error_latent, u_flat], dim=-1)
        return x, u_remain, squeezed_action

    def forward(
        self,
        z_real: torch.Tensor,
        z_goal: torch.Tensor,
        error_latent: torch.Tensor,
        u_remain: torch.Tensor,
    ) -> torch.Tensor:
        x, normalized_remain, squeezed = self.encode_inputs(
            z_real,
            z_goal,
            error_latent,
            u_remain,
        )
        correction = self.action_head(self.trunk(x)).reshape(
            int(normalized_remain.shape[0]),
            self.remain_horizon,
            self.action_dim,
        )
        if bool(self.config.predict_residual):
            output = normalized_remain + float(self.config.residual_scale) * correction
        else:
            output = correction
        if squeezed:
            return output[0]
        return output


@dataclass
class CorrectorBundle:
    """Portable save/load bundle for ActionChunkCorrector."""

    model_state_dict: dict[str, torch.Tensor]
    model_hyperparameters: dict[str, Any]
    latent_dim: int
    remain_horizon: int
    action_dim: int
    remain_action_dim: int
    input_dim: int
    metadata: dict[str, Any]
    bundle_version: int = 1

    @property
    def config(self) -> ActionChunkCorrectorConfig:
        return ActionChunkCorrectorConfig(**self.model_hyperparameters)

    def instantiate_model(
        self,
        map_location: str | torch.device | None = None,
    ) -> ActionChunkCorrector:
        model = ActionChunkCorrector(self.config)
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
            "remain_horizon": int(self.remain_horizon),
            "action_dim": int(self.action_dim),
            "remain_action_dim": int(self.remain_action_dim),
            "input_dim": int(self.input_dim),
            "metadata": dict(self.metadata),
        }


def build_corrector_bundle(
    model: ActionChunkCorrector,
    *,
    metadata: dict[str, Any] | None = None,
) -> CorrectorBundle:
    config = model.config
    return CorrectorBundle(
        model_state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        model_hyperparameters=asdict(config),
        latent_dim=int(config.latent_dim),
        remain_horizon=int(config.remain_horizon),
        action_dim=int(config.action_dim),
        remain_action_dim=int(config.remain_action_dim),
        input_dim=int(config.input_dim),
        metadata=dict(metadata or {}),
    )


def save_corrector_bundle(
    model: ActionChunkCorrector,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_corrector_bundle(model, metadata=metadata).as_dict(), output_path)


def load_corrector_bundle(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> CorrectorBundle:
    bundle_dict = torch.load(Path(path).expanduser().resolve(), map_location=map_location)
    validate_corrector_bundle_dict(bundle_dict)
    return CorrectorBundle(**bundle_dict)


def validate_corrector_bundle_dict(bundle_dict: dict[str, Any]) -> None:
    required = {
        "bundle_version",
        "model_state_dict",
        "model_hyperparameters",
        "latent_dim",
        "remain_horizon",
        "action_dim",
        "remain_action_dim",
        "input_dim",
        "metadata",
    }
    missing = required.difference(bundle_dict.keys())
    if missing:
        raise KeyError(f"Corrector bundle is missing required keys: {sorted(missing)}.")
    config = ActionChunkCorrectorConfig(**bundle_dict["model_hyperparameters"])
    if int(bundle_dict["latent_dim"]) != int(config.latent_dim):
        raise ValueError("Corrector bundle latent_dim does not match model_hyperparameters.")
    if int(bundle_dict["remain_horizon"]) != int(config.remain_horizon):
        raise ValueError("Corrector bundle remain_horizon does not match model_hyperparameters.")
    if int(bundle_dict["action_dim"]) != int(config.action_dim):
        raise ValueError("Corrector bundle action_dim does not match model_hyperparameters.")
    if int(bundle_dict["remain_action_dim"]) != int(config.remain_action_dim):
        raise ValueError("Corrector bundle remain_action_dim does not match model_hyperparameters.")
    if int(bundle_dict["input_dim"]) != int(config.input_dim):
        raise ValueError("Corrector bundle input_dim does not match model_hyperparameters.")
    if not isinstance(bundle_dict["metadata"], dict):
        raise ValueError("Corrector bundle metadata must be a dict.")


__all__ = [
    "ActionChunkCorrector",
    "ActionChunkCorrectorConfig",
    "CorrectorBundle",
    "build_corrector_bundle",
    "load_corrector_bundle",
    "save_corrector_bundle",
    "validate_corrector_bundle_dict",
]
