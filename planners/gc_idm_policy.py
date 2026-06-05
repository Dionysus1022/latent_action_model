from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from stable_worldmodel.policy import BasePolicy

from planners.gc_idm_model import GCIDMBundle, GCIDMModel, load_gc_idm_bundle


class GCIDMPolicy(BasePolicy):
    """Closed-loop GC-IDM policy from the paper.

    Runtime:
        1. Encode goal observation once.
        2. At every env step, re-encode the current observation.
        3. Predict and apply one single-step action.
    """

    def __init__(
        self,
        world_model: torch.nn.Module,
        planner: GCIDMModel,
        *,
        goal_offset_steps: int,
        eval_budget: int,
        process: dict[str, Any] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        planner_bundle: GCIDMBundle | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.type = "gc_idm"
        self.world_model = world_model
        planner_device = next(world_model.parameters()).device
        self.planner = planner.to(planner_device).eval()
        self.goal_offset_steps = int(goal_offset_steps)
        self.eval_budget = int(eval_budget)
        self.process = process or {}
        self.transform = transform or {}
        self.planner_bundle = planner_bundle

        self._goal_latent: torch.Tensor | None = None
        self._step_index = 0
        self._num_replans = 0
        self._planning_time_total_sec = 0.0
        self._env_action_shape: tuple[int, ...] | None = None
        self._validate_runtime_contracts()

    @classmethod
    def from_bundle(
        cls,
        *,
        bundle_path: str | Path,
        world_model: torch.nn.Module,
        goal_offset_steps: int,
        eval_budget: int,
        process: dict[str, Any] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        map_location: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> "GCIDMPolicy":
        bundle = load_gc_idm_bundle(bundle_path, map_location=map_location)
        planner = bundle.instantiate_model()
        planner.load_state_dict(bundle.model_state_dict)
        planner.eval()
        return cls(
            world_model=world_model,
            planner=planner,
            goal_offset_steps=goal_offset_steps,
            eval_budget=eval_budget,
            process=process,
            transform=transform,
            planner_bundle=bundle,
            **kwargs,
        )

    @property
    def latent_dim(self) -> int:
        return int(self.planner.latent_dim)

    @property
    def action_dim(self) -> int:
        return int(self.planner.action_dim)

    @property
    def max_horizon(self) -> int:
        return int(self.planner.max_horizon)

    @property
    def flatten_receding_horizon(self) -> int:
        return 1

    def set_env(self, env: Any) -> None:
        super().set_env(env)
        self.reset()
        env_action_shape = tuple(getattr(env.action_space, "shape", ()))
        if len(env_action_shape) == 0:
            raise ValueError("env.action_space.shape must be defined for GCIDMPolicy.")

        n_envs = int(getattr(env, "num_envs", 1))
        batched_action_dim = int(np.prod(env_action_shape))
        if batched_action_dim == self.action_dim:
            per_env_action_dim = batched_action_dim
        elif n_envs > 1 and batched_action_dim % n_envs == 0:
            per_env_action_dim = batched_action_dim // n_envs
        else:
            per_env_action_dim = batched_action_dim

        if per_env_action_dim != self.action_dim:
            raise ValueError(
                "Planner action_dim does not match the inferred per-env action dim: "
                f"{self.action_dim} != {per_env_action_dim} "
                f"(env.action_space.shape={env_action_shape}, num_envs={n_envs})."
            )
        self._env_action_shape = env_action_shape

    def reset(self) -> None:
        self._goal_latent = None
        self._step_index = 0
        self._num_replans = 0
        self._planning_time_total_sec = 0.0

    def get_action(self, info_dict: dict[str, Any], **kwargs: Any) -> np.ndarray:
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "pixels" in info_dict, "'pixels' must be provided in info_dict"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        prepared_info = self._prepare_info(dict(info_dict))
        action = self.predict_action(prepared_info)
        self._num_replans += 1
        self._step_index += 1

        target_shape = self._env_action_shape or tuple(self.env.action_space.shape)
        action_np = action.reshape(*target_shape).detach().cpu().numpy()
        if "action" in self.process:
            action_np = self.process["action"].inverse_transform(action_np)
        return action_np

    @torch.inference_mode()
    def predict_action(self, prepared_info: dict[str, torch.Tensor]) -> torch.Tensor:
        start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start is not None:
            start.record()

        z_cur = self.encode_current(prepared_info)
        z_goal = self.encode_goal_once(prepared_info)
        remaining = max(1, int(self.eval_budget) - int(self._step_index))
        horizon = torch.full(
            (int(z_cur.shape[0]),),
            min(remaining, self.max_horizon),
            dtype=z_cur.dtype,
            device=z_cur.device,
        )
        action = self.planner(z_cur, z_goal, horizon)

        if end is not None:
            end.record()
            torch.cuda.synchronize()
            self._planning_time_total_sec += float(start.elapsed_time(end)) / 1000.0
        return action.detach().cpu().float()

    @torch.inference_mode()
    def encode_current(self, prepared_info: dict[str, torch.Tensor]) -> torch.Tensor:
        if "pixels" not in prepared_info:
            raise KeyError("'pixels' must be present in prepared_info.")
        pixels = prepared_info["pixels"]
        if not torch.is_tensor(pixels):
            raise TypeError("prepared_info['pixels'] must be a torch.Tensor.")
        if pixels.ndim != 5:
            raise ValueError(f"prepared_info['pixels'] must have shape [B, T, C, H, W], got {tuple(pixels.shape)}.")
        device = next(self.world_model.parameters()).device
        encoded = self.world_model.encode({"pixels": pixels.to(device)})
        if "emb" not in encoded:
            raise KeyError("world_model.encode(...) must return a dict containing 'emb'.")
        z_cur = encoded["emb"][:, -1].detach()
        self._validate_latent(z_cur, name="z_cur")
        return z_cur

    @torch.inference_mode()
    def encode_goal_once(self, prepared_info: dict[str, torch.Tensor]) -> torch.Tensor:
        if self._goal_latent is not None:
            return self._goal_latent
        if "goal" not in prepared_info:
            raise KeyError("'goal' must be present in prepared_info.")
        goal_pixels = prepared_info["goal"]
        if not torch.is_tensor(goal_pixels):
            raise TypeError("prepared_info['goal'] must be a torch.Tensor.")
        if goal_pixels.ndim != 5:
            raise ValueError(f"prepared_info['goal'] must have shape [B, T, C, H, W], got {tuple(goal_pixels.shape)}.")
        device = next(self.world_model.parameters()).device
        encoded = self.world_model.encode({"pixels": goal_pixels.to(device)})
        if "emb" not in encoded:
            raise KeyError("world_model.encode(...) must return a dict containing 'emb'.")
        z_goal = encoded["emb"][:, -1].detach()
        self._validate_latent(z_goal, name="z_goal")
        self._goal_latent = z_goal
        return z_goal

    def _validate_latent(self, z: torch.Tensor, *, name: str) -> None:
        if z.ndim != 2:
            raise ValueError(f"{name} must have shape [B, latent_dim], got {tuple(z.shape)}.")
        if int(z.shape[-1]) != self.latent_dim:
            raise ValueError(
                f"{name} latent dim {z.shape[-1]} does not match planner latent_dim {self.latent_dim}."
            )

    def _validate_runtime_contracts(self) -> None:
        if not hasattr(self.world_model, "encode"):
            raise TypeError("world_model must expose an encode(info_dict) method.")
        if not isinstance(self.planner, GCIDMModel):
            raise TypeError(f"planner must be a GCIDMModel, got {type(self.planner)}.")
        if self.eval_budget <= 0:
            raise ValueError(f"eval_budget must be positive, got {self.eval_budget}.")
        if self.goal_offset_steps <= 0:
            raise ValueError(f"goal_offset_steps must be positive, got {self.goal_offset_steps}.")
