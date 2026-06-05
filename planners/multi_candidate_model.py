from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from planners.action_anchors import (
    ActionAnchorBundle,
    validate_anchor_tensor,
)
from planners.single_peak_model import (
    build_mlp_trunk,
    ensure_batched_latents,
    infer_model_config_from_dataset_bundle as infer_single_peak_dataset_config,
)


@dataclass
class MultiCandidateModelConfig:
    """Hyperparameters for the minimal anchor-based multi-candidate planner."""

    latent_dim: int
    plan_horizon: int
    action_dim: int
    num_anchors: int
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


def infer_model_config_from_dataset_and_anchor_bundle(
    dataset_bundle: dict[str, Any],
    anchor_bundle: ActionAnchorBundle,
) -> MultiCandidateModelConfig:
    """Infer runtime-critical model dimensions from a dataset bundle and an anchor bundle."""
    dataset_cfg = infer_single_peak_dataset_config(dataset_bundle)
    if int(anchor_bundle.plan_horizon) != int(dataset_cfg.plan_horizon):
        raise ValueError(
            f"Anchor bundle plan_horizon {anchor_bundle.plan_horizon} does not match dataset plan_horizon {dataset_cfg.plan_horizon}."
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

    return MultiCandidateModelConfig(
        latent_dim=int(dataset_cfg.latent_dim),
        plan_horizon=int(dataset_cfg.plan_horizon),
        action_dim=int(dataset_cfg.action_dim),
        num_anchors=int(anchor_bundle.num_anchors),
    )


class MultiCandidatePlannerModel(nn.Module):
    """Anchor-based multi-candidate planner.

    Inputs:
        z_cur: [B, latent_dim] or [latent_dim]
        z_goal: [B, latent_dim] or [latent_dim]

    Internal feature:
        x = concat([z_cur, z_goal, z_goal - z_cur])  # [B, 3 * latent_dim]

    Outputs:
        score_logits: [B, K] or [K]
        residual: [B, K, action_chunk_dim] or [K, action_chunk_dim]
        candidates: [B, K, action_chunk_dim] or [K, action_chunk_dim]
        anchors: [B, K, action_chunk_dim] or [K, action_chunk_dim]
    """

    def __init__(
        self,
        config: MultiCandidateModelConfig,
        anchors: torch.Tensor,
        *,
        anchor_fit_method: str = "manual",
        anchor_seed: int | None = None,
        anchor_metadata: dict[str, Any] | None = None,
    ):
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

        self.trunk = build_mlp_trunk(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.score_head = nn.Linear(config.hidden_dim, config.num_anchors)
        self.residual_head = nn.Linear(
            config.hidden_dim,
            config.num_anchors * config.action_chunk_dim,
        )

    @classmethod
    def from_anchor_bundle(
        cls,
        config: MultiCandidateModelConfig,
        anchor_bundle: ActionAnchorBundle,
    ) -> "MultiCandidatePlannerModel":
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
    ) -> dict[str, torch.Tensor]:
        """Predict multi-candidate action chunks and scores.

        z_cur: [B, latent_dim] or [latent_dim]
        z_goal: [B, latent_dim] or [latent_dim]

        out["score_logits"]: [B, K] or [K]
        out["residual"]: [B, K, action_chunk_dim] or [K, action_chunk_dim]
        out["candidates"]: [B, K, action_chunk_dim] or [K, action_chunk_dim]
        out["anchors"]: [B, K, action_chunk_dim] or [K, action_chunk_dim]
        """
        x, squeezed = self.encode_inputs(z_cur, z_goal)  # [B, 3 * latent_dim]
        batch_size = int(x.shape[0])

        h = self.trunk(x)  # [B, hidden_dim]
        score_logits = self.score_head(h)  # [B, K]
        residual = self.residual_head(h).reshape(
            batch_size,
            self.num_anchors,
            self.action_chunk_dim,
        )  # [B, K, action_chunk_dim]
        expanded_anchors = self.anchors.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )  # [B, K, action_chunk_dim]
        candidates = expanded_anchors + residual  # [B, K, action_chunk_dim]

        if squeezed:
            return {
                "score_logits": score_logits[0],  # [K]
                "scores": score_logits[0],  # [K]
                "residual": residual[0],  # [K, action_chunk_dim]
                "candidates": candidates[0],  # [K, action_chunk_dim]
                "anchors": expanded_anchors[0],  # [K, action_chunk_dim]
            }

        return {
            "score_logits": score_logits,  # [B, K]
            "scores": score_logits,  # [B, K]
            "residual": residual,  # [B, K, action_chunk_dim]
            "candidates": candidates,  # [B, K, action_chunk_dim]
            "anchors": expanded_anchors,  # [B, K, action_chunk_dim]
        }

    def forward_dict(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward using dataset-style keys.

        batch["z_cur"]: [B, latent_dim]
        batch["z_goal"]: [B, latent_dim]
        out["candidates"]: [B, K, action_chunk_dim]
        out["score_logits"]: [B, K]
        """
        if "z_cur" not in batch or "z_goal" not in batch:
            raise KeyError("batch must contain 'z_cur' and 'z_goal'.")
        return self.forward(batch["z_cur"], batch["z_goal"])


@dataclass
class MultiCandidatePlannerBundle:
    """Portable save/load bundle for multi-candidate planner reuse."""

    model_state_dict: dict[str, torch.Tensor]
    model_hyperparameters: dict[str, Any]
    latent_dim: int
    plan_horizon: int
    action_dim: int
    action_chunk_dim: int
    num_anchors: int
    input_dim: int
    anchors: torch.Tensor
    anchor_fit_method: str
    anchor_seed: int | None
    anchor_metadata: dict[str, Any]
    bundle_version: int = 1

    def instantiate_model(
        self,
        map_location: str | torch.device | None = None,
    ) -> MultiCandidatePlannerModel:
        config = MultiCandidateModelConfig(**self.model_hyperparameters)
        anchors = self.anchors
        if map_location is not None and torch.is_tensor(anchors):
            anchors = anchors.to(map_location)
        model = MultiCandidatePlannerModel(
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
            "anchors": self.anchors,
            "anchor_fit_method": str(self.anchor_fit_method),
            "anchor_seed": None if self.anchor_seed is None else int(self.anchor_seed),
            "anchor_metadata": dict(self.anchor_metadata),
        }


def build_multi_candidate_bundle(
    model: MultiCandidatePlannerModel,
) -> MultiCandidatePlannerBundle:
    config = model.config
    return MultiCandidatePlannerBundle(
        model_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        model_hyperparameters=asdict(config),
        latent_dim=int(config.latent_dim),
        plan_horizon=int(config.plan_horizon),
        action_dim=int(config.action_dim),
        action_chunk_dim=int(config.action_chunk_dim),
        num_anchors=int(config.num_anchors),
        input_dim=int(config.input_dim),
        anchors=model.anchors.detach().cpu().clone(),
        anchor_fit_method=str(model.anchor_fit_method),
        anchor_seed=None if model.anchor_seed is None else int(model.anchor_seed),
        anchor_metadata=dict(model.anchor_metadata),
    )


def save_multi_candidate_bundle(
    model: MultiCandidatePlannerModel,
    path: str | Path,
) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = build_multi_candidate_bundle(model)
    torch.save(bundle.as_dict(), output_path)
    return output_path


def load_multi_candidate_bundle(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> MultiCandidatePlannerBundle:
    bundle_dict = torch.load(Path(path).expanduser().resolve(), map_location=map_location)
    validate_bundle_dict(bundle_dict)
    bundle = MultiCandidatePlannerBundle(**bundle_dict)
    bundle.anchors = validate_anchor_tensor(
        bundle.anchors,
        plan_horizon=bundle.plan_horizon,
        action_dim=bundle.action_dim,
        num_anchors=bundle.num_anchors,
    )  # [K, action_chunk_dim]
    return bundle


def load_multi_candidate_model(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> MultiCandidatePlannerModel:
    bundle = load_multi_candidate_bundle(path, map_location=map_location)
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

    config = MultiCandidateModelConfig(**model_hparams)
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
    if not isinstance(bundle_dict["anchor_fit_method"], str):
        raise ValueError("bundle['anchor_fit_method'] must be a string.")
    if bundle_dict["anchor_seed"] is not None and not isinstance(bundle_dict["anchor_seed"], int):
        raise ValueError("bundle['anchor_seed'] must be an int or None.")
    if not isinstance(bundle_dict["anchor_metadata"], dict):
        raise ValueError("bundle['anchor_metadata'] must be a dict.")


def select_top_candidate(
    candidates: torch.Tensor,
    score_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the top-scoring candidate action chunk.

    candidates:
        [B, K, action_chunk_dim] or [K, action_chunk_dim]
    score_logits:
        [B, K] or [K]

    returns:
        best_candidates: [B, action_chunk_dim] or [action_chunk_dim]
        best_indices: [B] or []
    """
    if not torch.is_tensor(candidates) or not torch.is_tensor(score_logits):
        raise TypeError("candidates and score_logits must be torch.Tensor values.")

    squeezed = False
    if candidates.ndim == 2:
        candidates = candidates.unsqueeze(0)  # [1, K, action_chunk_dim]
        squeezed = True
    if score_logits.ndim == 1:
        score_logits = score_logits.unsqueeze(0)  # [1, K]
        squeezed = True

    if candidates.ndim != 3:
        raise ValueError(f"candidates must have shape [B, K, D] or [K, D], got {tuple(candidates.shape)}.")
    if score_logits.ndim != 2:
        raise ValueError(f"score_logits must have shape [B, K] or [K], got {tuple(score_logits.shape)}.")
    if candidates.shape[:2] != score_logits.shape:
        raise ValueError(
            f"candidates batch/candidate dims {tuple(candidates.shape[:2])} do not match score_logits {tuple(score_logits.shape)}."
        )

    best_indices = torch.argmax(score_logits, dim=-1)  # [B]
    gather_index = best_indices.view(-1, 1, 1).expand(-1, 1, candidates.shape[-1])  # [B, 1, D]
    best_candidates = candidates.gather(1, gather_index).squeeze(1)  # [B, D]

    if squeezed:
        return best_candidates[0], best_indices[0]
    return best_candidates, best_indices
