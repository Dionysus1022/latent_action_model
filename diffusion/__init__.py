"""Diffusion planner training and runtime package."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DiffusionPlannerBundle",
    "DiffusionPlannerModel",
    "DiffusionPlannerModelConfig",
    "DiffusionPlannerPolicy",
    "load_diffusion_planner_bundle",
    "load_diffusion_planner_model",
    "save_diffusion_planner_bundle",
]

_MODEL_EXPORTS = {
    "DiffusionPlannerBundle",
    "DiffusionPlannerModel",
    "DiffusionPlannerModelConfig",
    "load_diffusion_planner_bundle",
    "load_diffusion_planner_model",
    "save_diffusion_planner_bundle",
}

_POLICY_EXPORTS = {"DiffusionPlannerPolicy"}


def __getattr__(name: str) -> Any:
    if name in _MODEL_EXPORTS:
        from diffusion import model

        value = getattr(model, name)
        globals()[name] = value
        return value
    if name in _POLICY_EXPORTS:
        from diffusion import policy

        value = getattr(policy, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
