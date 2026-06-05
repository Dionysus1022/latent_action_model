from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class GCIDMModelConfig:
    """GC-IDM hyperparameters matching the paper defaults."""

    latent_dim: int
    action_dim: int
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.1
    activation: str = "gelu"
    horizon_embedding_dim: int = 64
    horizon_mlp_layers: int = 2
    max_horizon: int = 50

    @property
    def input_dim(self) -> int:
        return int(2 * self.latent_dim)


class AdaLNZero(nn.Module):
    """Zero-initialized additive horizon modulation before the action head."""

    def __init__(self, cond_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta = nn.Linear(cond_dim, hidden_dim)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, hidden: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        return self.norm(hidden) * (1.0 + gamma) + beta


class GCIDMModel(nn.Module):
    """Goal-conditioned inverse dynamics model.

    Paper mapping:
        gc-idm_psi(z_t, z_goal, h_t) -> a_t

    The remaining horizon is normalized/clamped to [0, 1], sinusoidally
    encoded, passed through a small MLP, and injected with AdaLN-Zero.
    """

    def __init__(self, config: GCIDMModelConfig):
        super().__init__()
        self.config = config
        self.backbone = build_mlp_trunk(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.horizon_mlp = build_condition_mlp(
            input_dim=config.horizon_embedding_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.horizon_mlp_layers,
            activation=config.activation,
        )
        self.adaln_zero = AdaLNZero(config.hidden_dim, config.hidden_dim)
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)
        self._init_weights()

    @property
    def latent_dim(self) -> int:
        return int(self.config.latent_dim)

    @property
    def action_dim(self) -> int:
        return int(self.config.action_dim)

    @property
    def max_horizon(self) -> int:
        return int(self.config.max_horizon)

    def forward(
        self,
        z_cur: torch.Tensor,
        z_goal: torch.Tensor,
        horizon: torch.Tensor | int | float,
    ) -> torch.Tensor:
        z_cur, squeezed = ensure_batched_latents(z_cur, self.latent_dim, name="z_cur")
        z_goal, squeezed_goal = ensure_batched_latents(z_goal, self.latent_dim, name="z_goal")
        if squeezed != squeezed_goal:
            raise ValueError("z_cur and z_goal must both be batched or both be unbatched.")
        if z_cur.shape != z_goal.shape:
            raise ValueError(
                f"z_cur and z_goal must have matching shapes, got {tuple(z_cur.shape)} and {tuple(z_goal.shape)}."
            )
        horizon_tensor = ensure_batched_horizon(
            horizon,
            batch_size=int(z_cur.shape[0]),
            device=z_cur.device,
            dtype=z_cur.dtype,
        )
        x = torch.cat([z_cur, z_goal], dim=-1)
        hidden = self.backbone(x)
        cond = self.horizon_mlp(
            sinusoidal_horizon_embedding(
                horizon_tensor,
                max_horizon=self.max_horizon,
                embedding_dim=int(self.config.horizon_embedding_dim),
            )
        )
        hidden = self.adaln_zero(hidden, cond)
        action = self.action_head(hidden)
        if squeezed:
            return action[0]
        return action

    def forward_dict(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key in ["z_cur", "z_goal", "horizon"]:
            if key not in batch:
                raise KeyError(f"batch must contain '{key}'.")
        return {
            "action": self.forward(
                batch["z_cur"],
                batch["z_goal"],
                batch["horizon"],
            )
        }

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.action_head.weight, mean=0.0, std=0.01)
        if self.action_head.bias is not None:
            nn.init.zeros_(self.action_head.bias)
        nn.init.zeros_(self.adaln_zero.gamma.weight)
        nn.init.zeros_(self.adaln_zero.gamma.bias)
        nn.init.zeros_(self.adaln_zero.beta.weight)
        nn.init.zeros_(self.adaln_zero.beta.bias)


@dataclass
class GCIDMBundle:
    model_state_dict: dict[str, torch.Tensor]
    model_hyperparameters: dict[str, Any]
    latent_dim: int
    action_dim: int
    max_horizon: int
    metadata: dict[str, Any] = field(default_factory=dict)
    bundle_version: int = 1

    def instantiate_model(self, map_location: str | torch.device | None = None) -> GCIDMModel:
        config = GCIDMModelConfig(**self.model_hyperparameters)
        model = GCIDMModel(config)
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
            "model_hyperparameters": dict(self.model_hyperparameters),
            "latent_dim": int(self.latent_dim),
            "action_dim": int(self.action_dim),
            "max_horizon": int(self.max_horizon),
            "metadata": dict(self.metadata),
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
    in_dim = int(input_dim)
    for _ in range(int(num_layers)):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(act_factory())
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        in_dim = int(hidden_dim)
    return nn.Sequential(*layers)


def build_condition_mlp(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    activation: str,
) -> nn.Sequential:
    if num_layers <= 0:
        raise ValueError(f"horizon_mlp_layers must be positive, got {num_layers}.")
    act_factory = get_activation_factory(activation)
    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    for layer_idx in range(int(num_layers)):
        layers.append(nn.Linear(in_dim, hidden_dim))
        if layer_idx < int(num_layers) - 1:
            layers.append(act_factory())
        in_dim = int(hidden_dim)
    return nn.Sequential(*layers)


def get_activation_factory(name: str):
    activation_name = str(name).lower()
    if activation_name == "gelu":
        return nn.GELU
    if activation_name == "relu":
        return nn.ReLU
    if activation_name == "silu":
        return nn.SiLU
    raise ValueError(f"Unsupported activation '{name}'.")


def sinusoidal_horizon_embedding(
    horizon: torch.Tensor,
    *,
    max_horizon: int,
    embedding_dim: int,
) -> torch.Tensor:
    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")
    if max_horizon <= 0:
        raise ValueError(f"max_horizon must be positive, got {max_horizon}.")
    if horizon.ndim != 1:
        raise ValueError(f"horizon must have shape [B], got {tuple(horizon.shape)}.")

    normalized = torch.clamp(horizon.float(), min=1.0, max=float(max_horizon)) / float(max_horizon)
    half_dim = int(math.ceil(embedding_dim / 2))
    frequencies = torch.exp(
        torch.arange(half_dim, device=horizon.device, dtype=normalized.dtype)
        * (-math.log(10000.0) / max(half_dim - 1, 1))
    )
    angles = normalized[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if int(embedding.shape[-1]) > embedding_dim:
        embedding = embedding[:, :embedding_dim]
    if int(embedding.shape[-1]) < embedding_dim:
        pad = embedding.new_zeros(int(embedding.shape[0]), embedding_dim - int(embedding.shape[-1]))
        embedding = torch.cat([embedding, pad], dim=-1)
    return embedding


def ensure_batched_latents(
    z: torch.Tensor,
    latent_dim: int,
    name: str,
) -> tuple[torch.Tensor, bool]:
    if not torch.is_tensor(z):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(z)}.")
    if z.ndim == 1:
        if int(z.shape[0]) != int(latent_dim):
            raise ValueError(
                f"{name} must have shape [latent_dim], got {tuple(z.shape)} for latent_dim={latent_dim}."
            )
        return z.unsqueeze(0), True
    if z.ndim == 2:
        if int(z.shape[-1]) != int(latent_dim):
            raise ValueError(
                f"{name} must have shape [B, latent_dim], got {tuple(z.shape)} for latent_dim={latent_dim}."
            )
        return z, False
    raise ValueError(f"{name} must have shape [latent_dim] or [B, latent_dim], got {tuple(z.shape)}.")


def ensure_batched_horizon(
    horizon: torch.Tensor | int | float,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(horizon):
        horizon_tensor = horizon.to(device=device, dtype=dtype)
    else:
        horizon_tensor = torch.tensor(horizon, device=device, dtype=dtype)
    if horizon_tensor.ndim == 0:
        horizon_tensor = horizon_tensor.expand(batch_size)
    if horizon_tensor.ndim != 1:
        raise ValueError(f"horizon must have shape [B] or be scalar, got {tuple(horizon_tensor.shape)}.")
    if int(horizon_tensor.shape[0]) != int(batch_size):
        raise ValueError(
            f"horizon batch size must match z batch size: {horizon_tensor.shape[0]} != {batch_size}."
        )
    return horizon_tensor


def build_gc_idm_bundle(
    model: GCIDMModel,
    *,
    metadata: dict[str, Any] | None = None,
) -> GCIDMBundle:
    config = model.config
    return GCIDMBundle(
        model_state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        model_hyperparameters=asdict(config),
        latent_dim=int(config.latent_dim),
        action_dim=int(config.action_dim),
        max_horizon=int(config.max_horizon),
        metadata=dict(metadata or {}),
    )


def save_gc_idm_bundle(
    model: GCIDMModel,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = build_gc_idm_bundle(model, metadata=metadata)
    torch.save(bundle.as_dict(), output_path)
    return output_path


def load_gc_idm_bundle(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> GCIDMBundle:
    bundle_dict = torch.load(Path(path).expanduser().resolve(), map_location=map_location)
    validate_bundle_dict(bundle_dict)
    return GCIDMBundle(**bundle_dict)


def load_gc_idm_model(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> GCIDMModel:
    bundle = load_gc_idm_bundle(path, map_location=map_location)
    return bundle.instantiate_model(map_location=map_location)


def validate_bundle_dict(bundle_dict: dict[str, Any]) -> None:
    required_keys = {
        "bundle_version",
        "model_state_dict",
        "model_hyperparameters",
        "latent_dim",
        "action_dim",
        "max_horizon",
        "metadata",
    }
    missing = required_keys.difference(bundle_dict.keys())
    if missing:
        raise KeyError(f"Bundle is missing required keys: {sorted(missing)}.")

    model_hparams = bundle_dict["model_hyperparameters"]
    if not isinstance(model_hparams, dict):
        raise ValueError("bundle['model_hyperparameters'] must be a dict.")
    config = GCIDMModelConfig(**model_hparams)
    if int(bundle_dict["latent_dim"]) != int(config.latent_dim):
        raise ValueError("bundle latent_dim does not match model_hyperparameters.")
    if int(bundle_dict["action_dim"]) != int(config.action_dim):
        raise ValueError("bundle action_dim does not match model_hyperparameters.")
    if int(bundle_dict["max_horizon"]) != int(config.max_horizon):
        raise ValueError("bundle max_horizon does not match model_hyperparameters.")
