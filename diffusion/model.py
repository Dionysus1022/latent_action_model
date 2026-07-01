from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from diffusion.anchors import ActionAnchorBundle, validate_anchor_tensor
from diffusion.utils import (
    TruncatedDiffusionSchedule,
    add_noise_to_anchors,
    broadcast_timesteps,
    build_truncation_timesteps,
    build_truncated_diffusion_schedule,
    denoise_step_from_x0,
    get_timestep_embedding,
    validate_action_chunk_tensor,
    validate_diffusion_schedule,
)
from planners.single_peak_model import (
    SinglePeakModelConfig,
    build_mlp_trunk,
    ensure_batched_latents,
    get_activation_factory,
)


@dataclass
class DiffusionPlannerModelConfig:
    """Hyperparameters for the minimal anchor-conditioned truncated diffusion planner."""

    latent_dim: int
    plan_horizon: int
    action_dim: int
    num_anchors: int
    hidden_dim: int = 512
    num_layers: int = 3
    dropout: float = 0.0
    activation: str = "gelu"
    timestep_embedding_dim: int = 128
    fusion_num_layers: int = 2
    denoiser_type: str = "mlp"
    dit_num_layers: int = 4
    dit_num_heads: int = 4
    dit_mlp_ratio: float = 4.0
    num_train_steps: int = 16
    truncation_steps: int = 4
    start_timestep: int | None = None
    beta_schedule: str = "linear"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    score_head_type: str = "linear"
    score_head_hidden_dim: int | None = None
    score_head_num_layers: int = 2

    @property
    def input_dim(self) -> int:
        return int(3 * self.latent_dim)

    @property
    def action_chunk_dim(self) -> int:
        return int(self.plan_horizon * self.action_dim)

    @property
    def action_chunk_horizon(self) -> int:
        return int(self.plan_horizon)


def _maybe_get_nested(source: dict[str, Any] | None, path: list[str]) -> Any:
    current: Any = source
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _maybe_positive_int(value: Any) -> int | None:
    if value in [None, "", "null"]:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"Expected a positive integer, got {parsed}.")
    return parsed


def infer_diffusion_dataset_config(
    dataset_bundle: dict[str, Any],
    *,
    anchor_bundle: ActionAnchorBundle | None = None,
) -> SinglePeakModelConfig:
    """Infer runtime-critical dataset dimensions for diffusion training.

    Expected:
        z_cur: [N, latent_dim]
        teacher_plan: [N, action_chunk_dim]
    """
    if "z_cur" not in dataset_bundle:
        raise KeyError("dataset_bundle must contain 'z_cur'.")
    if "teacher_plan" not in dataset_bundle:
        raise KeyError("dataset_bundle must contain 'teacher_plan'.")

    z_cur = dataset_bundle["z_cur"]
    teacher_plan = dataset_bundle["teacher_plan"]
    build_info = dataset_bundle.get("build_info", {})
    meta = dataset_bundle.get("meta", [])
    meta0 = meta[0] if isinstance(meta, list) and len(meta) > 0 and isinstance(meta[0], dict) else {}

    if not torch.is_tensor(z_cur) or z_cur.ndim != 2:
        raise ValueError(
            f"dataset_bundle['z_cur'] must have shape [N, latent_dim], got {type(z_cur)} with shape {getattr(z_cur, 'shape', None)}."
        )
    if not torch.is_tensor(teacher_plan) or teacher_plan.ndim != 2:
        raise ValueError(
            "dataset_bundle['teacher_plan'] must have shape [N, action_chunk_dim], "
            f"got {type(teacher_plan)} with shape {getattr(teacher_plan, 'shape', None)}."
        )

    latent_dim = int(z_cur.shape[-1])  # [N, latent_dim]
    action_chunk_dim = int(teacher_plan.shape[-1])  # [N, action_chunk_dim]

    action_dim = _maybe_positive_int(
        build_info.get("action_dim")
        or _maybe_get_nested(build_info, ["task_spec", "action_dim"])
        or meta0.get("action_dim")
        or (anchor_bundle.action_dim if anchor_bundle is not None else None)
    )
    plan_horizon = _maybe_positive_int(
        build_info.get("action_chunk_horizon")
        or _maybe_get_nested(build_info, ["task_spec", "action_chunk_horizon"])
        or meta0.get("plan_horizon")
        or (anchor_bundle.action_chunk_horizon if anchor_bundle is not None else None)
        or (anchor_bundle.plan_horizon if anchor_bundle is not None else None)
    )

    if action_dim is None and plan_horizon is not None:
        if action_chunk_dim % int(plan_horizon) != 0:
            raise KeyError(
                "Could not infer dataset action_dim: teacher_plan width is not divisible by action_chunk_horizon: "
                f"{action_chunk_dim} % {plan_horizon} != 0."
            )
        action_dim = int(action_chunk_dim // int(plan_horizon))
    if plan_horizon is None and action_dim is not None:
        if action_chunk_dim % int(action_dim) != 0:
            raise KeyError(
                "Could not infer dataset action_chunk_horizon: teacher_plan width is not divisible by action_dim: "
                f"{action_chunk_dim} % {action_dim} != 0."
            )
        plan_horizon = int(action_chunk_dim // int(action_dim))

    if action_dim is None:
        raise KeyError(
            "Could not infer dataset action_dim from build_info/meta/anchor bundle. "
            "Rebuild the planner dataset with metadata or use a compatible anchor bundle."
        )
    if plan_horizon is None:
        raise KeyError(
            "Could not infer dataset action_chunk_horizon from build_info/meta/anchor bundle. "
            "Rebuild the planner dataset with metadata or use a compatible anchor bundle."
        )

    expected_chunk_dim = int(plan_horizon * action_dim)
    if expected_chunk_dim != action_chunk_dim:
        raise ValueError(
            "Dataset action chunk shape mismatch: "
            f"action_chunk_horizon * action_dim = {plan_horizon} * {action_dim} = {expected_chunk_dim}, "
            f"but teacher_plan width is {action_chunk_dim}."
        )

    return SinglePeakModelConfig(
        latent_dim=latent_dim,
        plan_horizon=plan_horizon,
        action_dim=action_dim,
    )


def infer_model_config_from_dataset_and_anchor_bundle(
    dataset_bundle: dict[str, Any],
    anchor_bundle: ActionAnchorBundle,
    *,
    hidden_dim: int = 512,
    num_layers: int = 3,
    dropout: float = 0.0,
    activation: str = "gelu",
    timestep_embedding_dim: int = 128,
    fusion_num_layers: int = 2,
    denoiser_type: str = "mlp",
    dit_num_layers: int = 4,
    dit_num_heads: int = 4,
    dit_mlp_ratio: float = 4.0,
    num_train_steps: int = 16,
    truncation_steps: int = 4,
    start_timestep: int | None = None,
    beta_schedule: str = "linear",
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    score_head_type: str = "linear",
    score_head_hidden_dim: int | None = None,
    score_head_num_layers: int = 2,
) -> DiffusionPlannerModelConfig:
    """Infer diffusion planner dimensions from a dataset bundle and an anchor bundle."""
    dataset_cfg = infer_diffusion_dataset_config(dataset_bundle, anchor_bundle=anchor_bundle)
    if int(anchor_bundle.plan_horizon) != int(dataset_cfg.plan_horizon):
        raise ValueError(
            f"Anchor bundle plan_horizon {anchor_bundle.plan_horizon} does not match dataset plan_horizon {dataset_cfg.plan_horizon}."
        )
    if int(anchor_bundle.action_chunk_horizon) != int(dataset_cfg.plan_horizon):
        raise ValueError(
            "Anchor bundle action_chunk_horizon does not match dataset action_chunk_horizon: "
            f"{anchor_bundle.action_chunk_horizon} != {dataset_cfg.plan_horizon}."
        )
    if int(anchor_bundle.action_dim) != int(dataset_cfg.action_dim):
        raise ValueError(
            f"Anchor bundle action_dim {anchor_bundle.action_dim} does not match dataset action_dim {dataset_cfg.action_dim}."
        )
    if int(anchor_bundle.action_chunk_dim) != int(dataset_cfg.action_chunk_dim):
        raise ValueError(
            "Anchor bundle action_chunk_dim does not match dataset action_chunk_dim: "
            f"{anchor_bundle.action_chunk_dim} != {dataset_cfg.action_chunk_dim}."
        )

    return DiffusionPlannerModelConfig(
        latent_dim=int(dataset_cfg.latent_dim),
        plan_horizon=int(dataset_cfg.plan_horizon),
        action_dim=int(dataset_cfg.action_dim),
        num_anchors=int(anchor_bundle.num_anchors),
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        dropout=float(dropout),
        activation=str(activation),
        timestep_embedding_dim=int(timestep_embedding_dim),
        fusion_num_layers=int(fusion_num_layers),
        denoiser_type=str(denoiser_type),
        dit_num_layers=int(dit_num_layers),
        dit_num_heads=int(dit_num_heads),
        dit_mlp_ratio=float(dit_mlp_ratio),
        num_train_steps=int(num_train_steps),
        truncation_steps=int(truncation_steps),
        start_timestep=None if start_timestep is None else int(start_timestep),
        beta_schedule=str(beta_schedule),
        beta_start=float(beta_start),
        beta_end=float(beta_end),
        score_head_type=str(score_head_type),
        score_head_hidden_dim=None if score_head_hidden_dim is None else int(score_head_hidden_dim),
        score_head_num_layers=int(score_head_num_layers),
    )


def ensure_batched_candidates(
    noisy_candidates: torch.Tensor,
    *,
    num_candidates: int,
    action_chunk_dim: int,
    name: str = "noisy_candidates",
) -> tuple[torch.Tensor, bool]:
    """Normalize candidate actions to [B, K, action_chunk_dim].

    noisy_candidates:
        [K, action_chunk_dim]
        [B, K, action_chunk_dim]
    """
    candidates = validate_action_chunk_tensor(
        noisy_candidates,
        name=name,
        action_chunk_dim=action_chunk_dim,
    )
    if candidates.ndim == 2:
        if int(candidates.shape[0]) != int(num_candidates):
            raise ValueError(
                f"{name} candidate dim {candidates.shape[0]} does not match num_candidates {num_candidates}."
            )
        return candidates.unsqueeze(0), True  # [1, K, action_chunk_dim]
    if candidates.ndim == 3:
        if int(candidates.shape[1]) != int(num_candidates):
            raise ValueError(
                f"{name} candidate dim {candidates.shape[1]} does not match num_candidates {num_candidates}."
            )
        return candidates, False  # [B, K, action_chunk_dim]
    raise ValueError(
        f"{name} must have shape [K, action_chunk_dim] or [B, K, action_chunk_dim], got {tuple(candidates.shape)}."
    )


class DiTActionBlock(nn.Module):
    """Transformer block for action-token denoising with latent cross-attention."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")
        act_factory = get_activation_factory(activation)
        mlp_hidden_dim = int(round(float(hidden_dim) * float(mlp_ratio)))
        if mlp_hidden_dim <= 0:
            raise ValueError(f"mlp_ratio must produce a positive hidden size, got {mlp_ratio}.")
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            act_factory(),
            nn.Dropout(float(dropout)),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, action_tokens: torch.Tensor, condition_tokens: torch.Tensor) -> torch.Tensor:
        self_out, _ = self.self_attn(
            self.self_norm(action_tokens),
            self.self_norm(action_tokens),
            self.self_norm(action_tokens),
            need_weights=False,
        )
        action_tokens = action_tokens + self_out
        cross_out, _ = self.cross_attn(
            self.cross_norm(action_tokens),
            condition_tokens,
            condition_tokens,
            need_weights=False,
        )
        action_tokens = action_tokens + cross_out
        action_tokens = action_tokens + self.mlp(self.mlp_norm(action_tokens))
        return action_tokens


class DiffusionPlannerModel(nn.Module):
    """Minimal anchor-conditioned truncated diffusion planner.

    Inputs:
        z_cur: [B, latent_dim] or [latent_dim]
        z_goal: [B, latent_dim] or [latent_dim]
        noisy_candidates: [B, K, action_chunk_dim] or [K, action_chunk_dim]
        timesteps: [B, K], [K], or scalar

    Internal:
        x = concat([z_cur, z_goal, z_goal - z_cur])  # [B, 3 * latent_dim]

    Outputs:
        refined_actions / x0_pred: [B, K, action_chunk_dim] or [K, action_chunk_dim]
        score_logits / scores: [B, K] or [K]
    """

    def __init__(
        self,
        config: DiffusionPlannerModelConfig,
        anchors: torch.Tensor,
        *,
        anchor_fit_method: str = "manual",
        anchor_seed: int | None = None,
        anchor_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        validated_anchors = validate_anchor_tensor(
            anchors,
            plan_horizon=config.plan_horizon,
            action_dim=config.action_dim,
            num_anchors=config.num_anchors,
        )  # [K, action_chunk_dim]
        self.register_buffer("anchors", validated_anchors.clone(), persistent=False)

        self.anchor_fit_method = str(anchor_fit_method)
        self.anchor_seed = None if anchor_seed is None else int(anchor_seed)
        self.anchor_metadata = dict(anchor_metadata or {})

        schedule = build_truncated_diffusion_schedule(
            num_train_steps=config.num_train_steps,
            truncation_steps=config.truncation_steps,
            start_timestep=config.start_timestep,
            beta_schedule=config.beta_schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        )
        self._register_schedule_buffers(schedule)

        self.denoiser_type = str(config.denoiser_type).lower().strip()
        if self.denoiser_type not in {"mlp", "dit"}:
            raise ValueError(f"Unsupported denoiser_type '{config.denoiser_type}'. Expected mlp or dit.")
        act_factory = get_activation_factory(config.activation)
        self.condition_trunk = build_mlp_trunk(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.timestep_encoder = nn.Sequential(
            nn.Linear(config.timestep_embedding_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            act_factory(),
        )
        if self.denoiser_type == "mlp":
            self.candidate_encoder = nn.Sequential(
                nn.Linear(config.action_chunk_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                act_factory(),
            )
            self.fusion_trunk = build_mlp_trunk(
                input_dim=3 * config.hidden_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.fusion_num_layers,
                dropout=config.dropout,
                activation=config.activation,
            )
            self.action_head = nn.Linear(config.hidden_dim, config.action_chunk_dim)
            self.step_action_encoder = None
            self.action_pos_embedding = None
            self.condition_token_encoder = None
            self.dit_blocks = None
            self.dit_final_norm = None
            self.step_action_head = None
        else:
            self.candidate_encoder = None
            self.fusion_trunk = None
            self.action_head = None
            self.step_action_encoder = nn.Sequential(
                nn.Linear(config.action_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                act_factory(),
            )
            self.action_pos_embedding = nn.Parameter(
                torch.zeros(1, config.plan_horizon, config.hidden_dim)
            )
            self.condition_token_encoder = nn.Sequential(
                nn.Linear(config.latent_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                act_factory(),
            )
            self.dit_blocks = nn.ModuleList(
                [
                    DiTActionBlock(
                        hidden_dim=config.hidden_dim,
                        num_heads=int(config.dit_num_heads),
                        mlp_ratio=float(config.dit_mlp_ratio),
                        dropout=float(config.dropout),
                        activation=config.activation,
                    )
                    for _ in range(int(config.dit_num_layers))
                ]
            )
            self.dit_final_norm = nn.LayerNorm(config.hidden_dim)
            self.step_action_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.score_head = self._build_score_head(config)

    @staticmethod
    def _build_score_head(config: DiffusionPlannerModelConfig) -> nn.Module:
        score_head_type = str(config.score_head_type).lower().strip()
        if score_head_type == "linear":
            return nn.Linear(config.hidden_dim, 1)
        if score_head_type != "mlp":
            raise ValueError(
                f"Unsupported score_head_type '{config.score_head_type}'. Expected linear or mlp."
            )
        hidden_dim = (
            int(config.score_head_hidden_dim)
            if config.score_head_hidden_dim is not None
            else int(config.hidden_dim)
        )
        if hidden_dim <= 0:
            raise ValueError(f"score_head_hidden_dim must be positive, got {hidden_dim}.")
        if int(config.score_head_num_layers) <= 0:
            raise ValueError(
                f"score_head_num_layers must be positive, got {config.score_head_num_layers}."
            )
        act_factory = get_activation_factory(config.activation)
        layers: list[nn.Module] = [nn.LayerNorm(config.hidden_dim)]
        in_dim = int(config.hidden_dim)
        for _ in range(int(config.score_head_num_layers)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_factory())
            if float(config.dropout) > 0.0:
                layers.append(nn.Dropout(float(config.dropout)))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        return nn.Sequential(*layers)

    def _register_schedule_buffers(self, schedule: TruncatedDiffusionSchedule) -> None:
        validate_diffusion_schedule(schedule)
        self.register_buffer("schedule_betas", schedule.betas.clone(), persistent=False)
        self.register_buffer("schedule_alphas", schedule.alphas.clone(), persistent=False)
        self.register_buffer("schedule_alpha_bars", schedule.alpha_bars.clone(), persistent=False)
        self.register_buffer("schedule_alpha_bars_prev", schedule.alpha_bars_prev.clone(), persistent=False)
        self.register_buffer("schedule_sqrt_alpha_bars", schedule.sqrt_alpha_bars.clone(), persistent=False)
        self.register_buffer(
            "schedule_sqrt_one_minus_alpha_bars",
            schedule.sqrt_one_minus_alpha_bars.clone(),
            persistent=False,
        )
        self.register_buffer(
            "schedule_posterior_variance",
            schedule.posterior_variance.clone(),
            persistent=False,
        )
        self.register_buffer(
            "schedule_truncation_timesteps",
            schedule.truncation_timesteps.clone(),
            persistent=False,
        )

    @classmethod
    def from_anchor_bundle(
        cls,
        config: DiffusionPlannerModelConfig,
        anchor_bundle: ActionAnchorBundle,
    ) -> "DiffusionPlannerModel":
        return cls(
            config=config,
            anchors=anchor_bundle.anchors,
            anchor_fit_method=anchor_bundle.fit_method,
            anchor_seed=anchor_bundle.seed,
            anchor_metadata=anchor_bundle.metadata,
        )

    @property
    def latent_dim(self) -> int:
        return int(self.config.latent_dim)

    @property
    def plan_horizon(self) -> int:
        return int(self.config.plan_horizon)

    @property
    def action_chunk_horizon(self) -> int:
        return int(self.config.action_chunk_horizon)

    @property
    def action_dim(self) -> int:
        return int(self.config.action_dim)

    @property
    def action_chunk_dim(self) -> int:
        return int(self.config.action_chunk_dim)

    @property
    def num_anchors(self) -> int:
        return int(self.config.num_anchors)

    @property
    def input_dim(self) -> int:
        return int(self.config.input_dim)

    @property
    def num_train_steps(self) -> int:
        return int(self.config.num_train_steps)

    @property
    def truncation_steps(self) -> int:
        return int(self.config.truncation_steps)

    @property
    def schedule(self) -> TruncatedDiffusionSchedule:
        return TruncatedDiffusionSchedule(
            betas=self.schedule_betas,
            alphas=self.schedule_alphas,
            alpha_bars=self.schedule_alpha_bars,
            alpha_bars_prev=self.schedule_alpha_bars_prev,
            sqrt_alpha_bars=self.schedule_sqrt_alpha_bars,
            sqrt_one_minus_alpha_bars=self.schedule_sqrt_one_minus_alpha_bars,
            posterior_variance=self.schedule_posterior_variance,
            truncation_timesteps=self.schedule_truncation_timesteps,
            num_train_steps=int(self.config.num_train_steps),
            truncation_steps=int(self.config.truncation_steps),
            beta_schedule=str(self.config.beta_schedule),
            beta_start=float(self.config.beta_start),
            beta_end=float(self.config.beta_end),
        )

    def resolve_inference_truncation_timesteps(
        self,
        *,
        truncation_steps: int | None = None,
        start_timestep: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Resolve the reverse-diffusion timestep grid used at inference.

        returns:
            truncation_timesteps: [T_runtime] in descending order
        """
        default_timesteps = self.schedule.truncation_timesteps
        if truncation_steps is None and start_timestep is None:
            return default_timesteps.to(device=device)

        runtime_truncation_steps = (
            int(self.truncation_steps) if truncation_steps is None else int(truncation_steps)
        )
        if runtime_truncation_steps <= 0:
            raise ValueError(
                f"truncation_steps must be positive at inference, got {runtime_truncation_steps}."
            )

        runtime_start_timestep = (
            int(default_timesteps[0].item()) if start_timestep is None else int(start_timestep)
        )
        return build_truncation_timesteps(
            num_train_steps=self.num_train_steps,
            truncation_steps=runtime_truncation_steps,
            start_timestep=runtime_start_timestep,
            device=device,
        )  # [T_runtime]

    def encode_condition(
        self,
        z_cur: torch.Tensor,
        z_goal: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Encode goal-conditioned planner state.

        z_cur: [B, latent_dim] or [latent_dim]
        z_goal: [B, latent_dim] or [latent_dim]
        returns:
            condition: [B, hidden_dim]
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
        condition = self.condition_trunk(x)  # [B, hidden_dim]
        return condition, squeezed

    def expand_anchors(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Expand anchor library to [B, K, action_chunk_dim]."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        anchors = self.anchors.to(device=device, dtype=dtype)
        return anchors.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, action_chunk_dim]

    def make_timestep_grid(
        self,
        *,
        batch_size: int,
        timestep: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create a constant timestep grid [B, K]."""
        return torch.full(
            (int(batch_size), self.num_anchors),
            int(timestep),
            device=device,
            dtype=torch.long,
        )  # [B, K]

    def initialize_noisy_candidates(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        timesteps: int | torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample noisy candidates around anchors.

        returns:
            noisy_candidates: [B, K, action_chunk_dim]
            timestep_grid: [B, K]
        """
        anchors = self.expand_anchors(batch_size, device=device, dtype=dtype)  # [B, K, action_chunk_dim]
        if timesteps is None:
            timesteps = int(self.schedule.truncation_timesteps[0].item())
        timestep_grid = broadcast_timesteps(
            timesteps,
            target_shape=(int(batch_size), self.num_anchors),
            num_train_steps=self.num_train_steps,
            device=device,
        )  # [B, K]
        noisy_candidates = add_noise_to_anchors(
            anchors,
            timesteps=timestep_grid,
            schedule=self.schedule.to(device=device, dtype=dtype),
            noise=noise,
            action_chunk_dim=self.action_chunk_dim,
        )  # [B, K, action_chunk_dim]
        return noisy_candidates, timestep_grid

    def _forward_mlp_denoiser(
        self,
        *,
        condition: torch.Tensor,
        candidates: torch.Tensor,
        timestep_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(condition.shape[0])
        condition_features = condition.unsqueeze(1).expand(-1, self.num_anchors, -1)  # [B, K, hidden_dim]
        candidate_features = self.candidate_encoder(
            candidates.reshape(batch_size * self.num_anchors, self.action_chunk_dim)
        ).reshape(batch_size, self.num_anchors, -1)  # [B, K, hidden_dim]
        timestep_embedding = get_timestep_embedding(
            timestep_grid,
            embedding_dim=self.config.timestep_embedding_dim,
            device=candidates.device,
            dtype=candidates.dtype,
        )  # [B, K, timestep_embedding_dim]
        timestep_features = self.timestep_encoder(
            timestep_embedding.reshape(batch_size * self.num_anchors, self.config.timestep_embedding_dim)
        ).reshape(batch_size, self.num_anchors, -1)  # [B, K, hidden_dim]

        fused = torch.cat(
            [condition_features, candidate_features, timestep_features],
            dim=-1,
        )  # [B, K, 3 * hidden_dim]
        fused = self.fusion_trunk(
            fused.reshape(batch_size * self.num_anchors, 3 * self.config.hidden_dim)
        ).reshape(batch_size, self.num_anchors, -1)  # [B, K, hidden_dim]

        refined_actions = self.action_head(
            fused.reshape(batch_size * self.num_anchors, self.config.hidden_dim)
        ).reshape(batch_size, self.num_anchors, self.action_chunk_dim)  # [B, K, action_chunk_dim]
        score_logits = self.score_head(
            fused.reshape(batch_size * self.num_anchors, self.config.hidden_dim)
        ).reshape(batch_size, self.num_anchors)  # [B, K]
        return refined_actions, score_logits

    def _forward_dit_denoiser(
        self,
        *,
        z_cur: torch.Tensor,
        z_goal: torch.Tensor,
        candidates: torch.Tensor,
        timestep_grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(z_cur.shape[0])
        action_steps = candidates.reshape(
            batch_size,
            self.num_anchors,
            self.plan_horizon,
            self.action_dim,
        )  # [B, K, H, A]
        flat_steps = action_steps.reshape(batch_size * self.num_anchors, self.plan_horizon, self.action_dim)
        action_tokens = self.step_action_encoder(
            flat_steps.reshape(batch_size * self.num_anchors * self.plan_horizon, self.action_dim)
        ).reshape(batch_size * self.num_anchors, self.plan_horizon, self.config.hidden_dim)
        action_tokens = action_tokens + self.action_pos_embedding.to(device=action_tokens.device, dtype=action_tokens.dtype)

        timestep_embedding = get_timestep_embedding(
            timestep_grid,
            embedding_dim=self.config.timestep_embedding_dim,
            device=candidates.device,
            dtype=candidates.dtype,
        )  # [B, K, timestep_embedding_dim]
        timestep_tokens = self.timestep_encoder(
            timestep_embedding.reshape(batch_size * self.num_anchors, self.config.timestep_embedding_dim)
        ).reshape(batch_size * self.num_anchors, 1, self.config.hidden_dim)
        action_tokens = action_tokens + timestep_tokens

        latent_tokens = torch.stack(
            [z_cur, z_goal, z_goal - z_cur],
            dim=1,
        )  # [B, 3, latent_dim]
        condition_tokens = self.condition_token_encoder(
            latent_tokens.reshape(batch_size * 3, self.latent_dim)
        ).reshape(batch_size, 3, self.config.hidden_dim)  # [B, 3, hidden_dim]
        condition_tokens = condition_tokens.unsqueeze(1).expand(
            batch_size,
            self.num_anchors,
            3,
            self.config.hidden_dim,
        ).reshape(batch_size * self.num_anchors, 3, self.config.hidden_dim)

        for block in self.dit_blocks:
            action_tokens = block(action_tokens, condition_tokens)
        action_tokens = self.dit_final_norm(action_tokens)
        refined_steps = self.step_action_head(
            action_tokens.reshape(batch_size * self.num_anchors * self.plan_horizon, self.config.hidden_dim)
        ).reshape(batch_size, self.num_anchors, self.plan_horizon, self.action_dim)
        refined_actions = refined_steps.reshape(batch_size, self.num_anchors, self.action_chunk_dim)
        pooled = action_tokens.mean(dim=1).reshape(batch_size, self.num_anchors, self.config.hidden_dim)
        score_logits = self.score_head(
            pooled.reshape(batch_size * self.num_anchors, self.config.hidden_dim)
        ).reshape(batch_size, self.num_anchors)
        return refined_actions, score_logits

    def forward(
        self,
        z_cur: torch.Tensor,
        z_goal: torch.Tensor,
        noisy_candidates: torch.Tensor,
        timesteps: int | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict x0-style refined actions and candidate scores.

        z_cur: [B, latent_dim] or [latent_dim]
        z_goal: [B, latent_dim] or [latent_dim]
        noisy_candidates: [B, K, action_chunk_dim] or [K, action_chunk_dim]
        timesteps: scalar, [K], or [B, K]

        out["refined_actions"] / out["x0_pred"]: [B, K, action_chunk_dim] or [K, action_chunk_dim]
        out["score_logits"] / out["scores"]: [B, K] or [K]
        """
        condition, squeezed_latents = self.encode_condition(z_cur, z_goal)  # [B, hidden_dim]
        candidates, squeezed_candidates = ensure_batched_candidates(
            noisy_candidates,
            num_candidates=self.num_anchors,
            action_chunk_dim=self.action_chunk_dim,
            name="noisy_candidates",
        )  # [B, K, action_chunk_dim]
        if squeezed_latents != squeezed_candidates:
            raise ValueError(
                "Latents and noisy_candidates must both be batched or both be unbatched."
            )

        batch_size = int(condition.shape[0])
        if int(candidates.shape[0]) != batch_size:
            raise ValueError(
                f"Candidate batch size {candidates.shape[0]} does not match latent batch size {batch_size}."
            )

        timestep_grid = broadcast_timesteps(
            timesteps,
            target_shape=(batch_size, self.num_anchors),
            num_train_steps=self.num_train_steps,
            device=candidates.device,
        )  # [B, K]

        if self.denoiser_type == "mlp":
            refined_actions, score_logits = self._forward_mlp_denoiser(
                condition=condition,
                candidates=candidates,
                timestep_grid=timestep_grid,
            )
        else:
            z_cur_batched, _ = ensure_batched_latents(z_cur, self.latent_dim, name="z_cur")
            z_goal_batched, _ = ensure_batched_latents(z_goal, self.latent_dim, name="z_goal")
            refined_actions, score_logits = self._forward_dit_denoiser(
                z_cur=z_cur_batched,
                z_goal=z_goal_batched,
                candidates=candidates,
                timestep_grid=timestep_grid,
            )

        expanded_anchors = self.expand_anchors(
            batch_size,
            device=candidates.device,
            dtype=candidates.dtype,
        )  # [B, K, action_chunk_dim]

        if squeezed_latents:
            return {
                "refined_actions": refined_actions[0],  # [K, action_chunk_dim]
                "x0_pred": refined_actions[0],  # [K, action_chunk_dim]
                "candidates": refined_actions[0],  # [K, action_chunk_dim]
                "score_logits": score_logits[0],  # [K]
                "scores": score_logits[0],  # [K]
                "noisy_candidates": candidates[0],  # [K, action_chunk_dim]
                "timesteps": timestep_grid[0],  # [K]
                "anchors": expanded_anchors[0],  # [K, action_chunk_dim]
            }

        return {
            "refined_actions": refined_actions,  # [B, K, action_chunk_dim]
            "x0_pred": refined_actions,  # [B, K, action_chunk_dim]
            "candidates": refined_actions,  # [B, K, action_chunk_dim]
            "score_logits": score_logits,  # [B, K]
            "scores": score_logits,  # [B, K]
            "noisy_candidates": candidates,  # [B, K, action_chunk_dim]
            "timesteps": timestep_grid,  # [B, K]
            "anchors": expanded_anchors,  # [B, K, action_chunk_dim]
        }

    def forward_dict(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward using dataset-style keys."""
        required = {"z_cur", "z_goal", "noisy_candidates", "timesteps"}
        missing = required.difference(batch.keys())
        if missing:
            raise KeyError(f"batch is missing required keys: {sorted(missing)}.")
        return self.forward(
            batch["z_cur"],
            batch["z_goal"],
            batch["noisy_candidates"],
            batch["timesteps"],
        )

    @torch.inference_mode()
    def generate_candidates(
        self,
        z_cur: torch.Tensor,
        z_goal: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        eta: float = 0.0,
        truncation_steps: int | None = None,
        start_timestep: int | None = None,
        noise_scale: float = 1.0,
        sampling_temperature: float = 1.0,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Run truncated diffusion from noisy anchors to final candidate chunks.

        returns:
            final["candidates"]: [B, K, action_chunk_dim] or [K, action_chunk_dim]
            final["score_logits"]: [B, K] or [K]
        """
        if noise_scale < 0.0:
            raise ValueError(f"noise_scale must be non-negative, got {noise_scale}.")
        if sampling_temperature < 0.0:
            raise ValueError(
                f"sampling_temperature must be non-negative, got {sampling_temperature}."
            )

        condition, squeezed = self.encode_condition(z_cur, z_goal)  # [B, hidden_dim]
        batch_size = int(condition.shape[0])
        device = condition.device
        dtype = condition.dtype
        runtime_schedule = self.schedule.to(device=device, dtype=dtype)
        runtime_truncation_timesteps = self.resolve_inference_truncation_timesteps(
            truncation_steps=truncation_steps,
            start_timestep=start_timestep,
            device=device,
        )  # [T_runtime]
        initial_timestep = int(runtime_truncation_timesteps[0].item())

        initial_noise = noise
        if initial_noise is None:
            initial_noise = torch.randn(
                batch_size,
                self.num_anchors,
                self.action_chunk_dim,
                device=device,
                dtype=dtype,
            )  # [B, K, action_chunk_dim]
            initial_noise = initial_noise * float(noise_scale) * float(sampling_temperature)

        current, _ = self.initialize_noisy_candidates(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            timesteps=initial_timestep,
            noise=initial_noise,
        )  # [B, K, action_chunk_dim]
        initial_noisy = current.clone()

        intermediates: list[torch.Tensor] = []
        last_outputs: dict[str, torch.Tensor] | None = None
        timestep_values = runtime_truncation_timesteps.detach().cpu().tolist()
        step_pairs = [
            (int(timestep_values[idx]), int(timestep_values[idx + 1]))
            for idx in range(len(timestep_values) - 1)
        ]
        if len(timestep_values) > 0:
            step_pairs.append((int(timestep_values[-1]), -1))

        for current_t, next_t in step_pairs:
            timestep_grid = self.make_timestep_grid(
                batch_size=batch_size,
                timestep=current_t,
                device=device,
            )  # [B, K]
            last_outputs = self.forward(z_cur, z_goal, current, timestep_grid)
            x0_pred = last_outputs["refined_actions"]
            if x0_pred.ndim == 2:
                x0_pred = x0_pred.unsqueeze(0)  # [1, K, action_chunk_dim]

            next_timestep_grid = torch.full(
                (batch_size, self.num_anchors),
                int(next_t),
                device=device,
                dtype=torch.long,
            )  # [B, K]
            reverse_noise = None
            if eta > 0.0:
                reverse_noise = torch.randn_like(current) * float(sampling_temperature)
            current = denoise_step_from_x0(
                current,
                x0_pred,
                timesteps=timestep_grid,
                next_timesteps=next_timestep_grid,
                schedule=runtime_schedule,
                eta=eta,
                noise=reverse_noise,
                action_chunk_dim=self.action_chunk_dim,
            )  # [B, K, action_chunk_dim]
            if return_intermediates:
                intermediates.append(current.detach().clone())

        if last_outputs is None:
            raise RuntimeError("generate_candidates() did not execute any diffusion step.")

        outputs: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "refined_actions": last_outputs["refined_actions"],
            "x0_pred": last_outputs["x0_pred"],
            "candidates": last_outputs["candidates"],
            "score_logits": last_outputs["score_logits"],
            "scores": last_outputs["scores"],
            "initial_noisy_candidates": initial_noisy[0] if squeezed else initial_noisy,
            "final_noisy_state": current[0] if squeezed else current,
            "truncation_timesteps": runtime_truncation_timesteps.detach().clone(),
        }
        if return_intermediates:
            outputs["intermediates"] = [item[0] if squeezed else item for item in intermediates]
        return outputs


@dataclass
class DiffusionPlannerBundle:
    """Portable save/load bundle for the truncated diffusion planner."""

    model_state_dict: dict[str, torch.Tensor]
    model_hyperparameters: dict[str, Any]
    latent_dim: int
    plan_horizon: int
    action_dim: int
    action_chunk_dim: int
    num_anchors: int
    input_dim: int
    num_train_steps: int
    truncation_steps: int
    start_timestep: int | None
    beta_schedule: str
    beta_start: float
    beta_end: float
    truncation_timesteps: torch.Tensor
    anchors: torch.Tensor
    anchor_fit_method: str
    anchor_seed: int | None
    anchor_metadata: dict[str, Any]
    bundle_version: int = 1

    def instantiate_model(
        self,
        map_location: str | torch.device | None = None,
    ) -> DiffusionPlannerModel:
        config = DiffusionPlannerModelConfig(**self.model_hyperparameters)
        anchors = self.anchors
        if map_location is not None and torch.is_tensor(anchors):
            anchors = anchors.to(map_location)
        model = DiffusionPlannerModel(
            config=config,
            anchors=anchors,
            anchor_fit_method=self.anchor_fit_method,
            anchor_seed=self.anchor_seed,
            anchor_metadata=self.anchor_metadata,
        )
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
            "num_anchors": int(self.num_anchors),
            "input_dim": int(self.input_dim),
            "num_train_steps": int(self.num_train_steps),
            "truncation_steps": int(self.truncation_steps),
            "start_timestep": None if self.start_timestep is None else int(self.start_timestep),
            "beta_schedule": str(self.beta_schedule),
            "beta_start": float(self.beta_start),
            "beta_end": float(self.beta_end),
            "truncation_timesteps": self.truncation_timesteps,
            "anchors": self.anchors,
            "anchor_fit_method": str(self.anchor_fit_method),
            "anchor_seed": None if self.anchor_seed is None else int(self.anchor_seed),
            "anchor_metadata": dict(self.anchor_metadata),
        }


def build_diffusion_planner_bundle(
    model: DiffusionPlannerModel,
) -> DiffusionPlannerBundle:
    config = model.config
    schedule = model.schedule
    return DiffusionPlannerBundle(
        model_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        model_hyperparameters=asdict(config),
        latent_dim=int(config.latent_dim),
        plan_horizon=int(config.plan_horizon),
        action_dim=int(config.action_dim),
        action_chunk_dim=int(config.action_chunk_dim),
        num_anchors=int(config.num_anchors),
        input_dim=int(config.input_dim),
        num_train_steps=int(config.num_train_steps),
        truncation_steps=int(config.truncation_steps),
        start_timestep=None if config.start_timestep is None else int(config.start_timestep),
        beta_schedule=str(config.beta_schedule),
        beta_start=float(config.beta_start),
        beta_end=float(config.beta_end),
        truncation_timesteps=schedule.truncation_timesteps.detach().cpu().clone(),
        anchors=model.anchors.detach().cpu().clone(),
        anchor_fit_method=str(model.anchor_fit_method),
        anchor_seed=None if model.anchor_seed is None else int(model.anchor_seed),
        anchor_metadata=dict(model.anchor_metadata),
    )


def save_diffusion_planner_bundle(
    model: DiffusionPlannerModel,
    path: str | Path,
) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = build_diffusion_planner_bundle(model)
    torch.save(bundle.as_dict(), output_path)
    return output_path


def load_diffusion_planner_bundle(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> DiffusionPlannerBundle:
    bundle_dict = torch.load(Path(path).expanduser().resolve(), map_location=map_location)
    validate_bundle_dict(bundle_dict)
    bundle = DiffusionPlannerBundle(**bundle_dict)
    bundle.anchors = validate_anchor_tensor(
        bundle.anchors,
        plan_horizon=bundle.plan_horizon,
        action_dim=bundle.action_dim,
        num_anchors=bundle.num_anchors,
    )  # [K, action_chunk_dim]
    return bundle


def load_diffusion_planner_model(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> DiffusionPlannerModel:
    bundle = load_diffusion_planner_bundle(path, map_location=map_location)
    model = bundle.instantiate_model(map_location=map_location)
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
        "num_anchors",
        "input_dim",
        "num_train_steps",
        "truncation_steps",
        "start_timestep",
        "beta_schedule",
        "beta_start",
        "beta_end",
        "truncation_timesteps",
        "anchors",
        "anchor_fit_method",
        "anchor_seed",
        "anchor_metadata",
    }
    missing = required_keys.difference(bundle_dict.keys())
    if missing:
        raise KeyError(f"Bundle is missing required keys: {sorted(missing)}.")

    model_hparams = bundle_dict["model_hyperparameters"]
    if not isinstance(model_hparams, dict):
        raise ValueError("bundle['model_hyperparameters'] must be a dict.")

    config = DiffusionPlannerModelConfig(**model_hparams)
    if int(bundle_dict["latent_dim"]) != int(config.latent_dim):
        raise ValueError("bundle latent_dim does not match model_hyperparameters.")
    if int(bundle_dict["plan_horizon"]) != int(config.plan_horizon):
        raise ValueError("bundle plan_horizon does not match model_hyperparameters.")
    if int(bundle_dict["action_dim"]) != int(config.action_dim):
        raise ValueError("bundle action_dim does not match model_hyperparameters.")
    if int(bundle_dict["action_chunk_dim"]) != int(config.action_chunk_dim):
        raise ValueError("bundle action_chunk_dim does not match model_hyperparameters.")
    if int(bundle_dict["num_anchors"]) != int(config.num_anchors):
        raise ValueError("bundle num_anchors does not match model_hyperparameters.")
    if int(bundle_dict["input_dim"]) != int(config.input_dim):
        raise ValueError("bundle input_dim does not match model_hyperparameters.")
    if int(bundle_dict["num_train_steps"]) != int(config.num_train_steps):
        raise ValueError("bundle num_train_steps does not match model_hyperparameters.")
    if int(bundle_dict["truncation_steps"]) != int(config.truncation_steps):
        raise ValueError("bundle truncation_steps does not match model_hyperparameters.")
    if bundle_dict["start_timestep"] != config.start_timestep:
        raise ValueError("bundle start_timestep does not match model_hyperparameters.")
    if str(bundle_dict["beta_schedule"]) != str(config.beta_schedule):
        raise ValueError("bundle beta_schedule does not match model_hyperparameters.")
    if float(bundle_dict["beta_start"]) != float(config.beta_start):
        raise ValueError("bundle beta_start does not match model_hyperparameters.")
    if float(bundle_dict["beta_end"]) != float(config.beta_end):
        raise ValueError("bundle beta_end does not match model_hyperparameters.")

    anchors = validate_anchor_tensor(
        bundle_dict["anchors"],
        plan_horizon=int(bundle_dict["plan_horizon"]),
        action_dim=int(bundle_dict["action_dim"]),
        num_anchors=int(bundle_dict["num_anchors"]),
    )  # [K, action_chunk_dim]
    if int(anchors.shape[-1]) != int(bundle_dict["action_chunk_dim"]):
        raise ValueError(
            f"Anchor width {anchors.shape[-1]} does not match action_chunk_dim {bundle_dict['action_chunk_dim']}."
        )

    schedule = build_truncated_diffusion_schedule(
        num_train_steps=int(bundle_dict["num_train_steps"]),
        truncation_steps=int(bundle_dict["truncation_steps"]),
        start_timestep=bundle_dict["start_timestep"],
        beta_schedule=str(bundle_dict["beta_schedule"]),
        beta_start=float(bundle_dict["beta_start"]),
        beta_end=float(bundle_dict["beta_end"]),
    )
    bundle_truncation_timesteps = torch.as_tensor(bundle_dict["truncation_timesteps"]).long()
    if bundle_truncation_timesteps.ndim != 1:
        raise ValueError(
            f"bundle['truncation_timesteps'] must be 1D, got {tuple(bundle_truncation_timesteps.shape)}."
        )
    if not torch.equal(bundle_truncation_timesteps.cpu(), schedule.truncation_timesteps.cpu()):
        raise ValueError(
            "bundle truncation_timesteps do not match the schedule reconstructed from model_hyperparameters: "
            f"{bundle_truncation_timesteps.tolist()} != {schedule.truncation_timesteps.tolist()}."
        )

    if not isinstance(bundle_dict["anchor_fit_method"], str):
        raise ValueError("bundle['anchor_fit_method'] must be a string.")
    if bundle_dict["anchor_seed"] is not None and not isinstance(bundle_dict["anchor_seed"], int):
        raise ValueError("bundle['anchor_seed'] must be an int or None.")
    if not isinstance(bundle_dict["anchor_metadata"], dict):
        raise ValueError("bundle['anchor_metadata'] must be a dict.")
