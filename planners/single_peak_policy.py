from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from stable_worldmodel import PlanConfig
from stable_worldmodel.policy import BasePolicy

from planners.single_peak_data import unflatten_action_chunk
from planners.single_peak_model import (
    SinglePeakPlannerBundle,
    SinglePeakPlannerModel,
    load_single_peak_bundle,
)


class SinglePeakPolicy(BasePolicy):
    """Runtime wrapper for the minimal single-peak planner.

    Runtime flow:
        1. Preprocess raw env info via BasePolicy._prepare_info.
        2. Encode current obs and goal obs with the world model encoder.
        3. Predict one flattened action chunk with the single-peak planner.
        4. Reshape the chunk to [plan_horizon, action_dim].
        5. Push per-step actions into an internal action buffer.
        6. Pop one [num_envs, action_dim] action on each get_action call.

    Shapes:
        prepared_info["pixels"]: [num_envs, history, C, H, W]
        prepared_info["goal"]: [num_envs, history, C, H, W]
        z_cur: [num_envs, latent_dim]
        z_goal: [num_envs, latent_dim]
        u_hat: [num_envs, action_chunk_dim]
        plan: [num_envs, plan_horizon, action_dim]
        buffered action: [num_envs, action_dim]
        get_action(...) return: [num_envs, action_dim]
    """

    def __init__(
        self,
        world_model: torch.nn.Module,
        planner: SinglePeakPlannerModel,
        config: PlanConfig | None = None,
        process: dict[str, Any] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        planner_bundle: SinglePeakPlannerBundle | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.type = "single_peak"
        self.world_model = world_model
        planner_device = next(world_model.parameters()).device
        self.planner = planner.to(planner_device).eval()
        self.cfg = config
        self.process = process or {}
        self.transform = transform or {}
        self.planner_bundle = planner_bundle

        self._action_buffer: deque[torch.Tensor] = deque(maxlen=self.plan_horizon)
        self._last_plan: torch.Tensor | None = None
        self._num_replans = 0
        self._env_action_shape: tuple[int, ...] | None = None

        self._validate_runtime_contracts()

    @classmethod
    def from_bundle(
        cls,
        *,
        bundle_path: str | Path,
        world_model: torch.nn.Module,
        config: PlanConfig | None = None,
        process: dict[str, Any] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        map_location: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> "SinglePeakPolicy":
        bundle = load_single_peak_bundle(bundle_path, map_location=map_location)
        planner = bundle.instantiate_model()
        planner.load_state_dict(bundle.model_state_dict)
        planner.eval()
        return cls(
            world_model=world_model,
            planner=planner,
            config=config,
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
    def plan_horizon(self) -> int:
        return int(self.planner.plan_horizon)

    @property
    def action_chunk_dim(self) -> int:
        return int(self.planner.action_chunk_dim)

    @property
    def flatten_receding_horizon(self) -> int:
        """Receding horizon in environment steps.

        For the single-peak planner, one planner call predicts exactly one
        executable chunk of length plan_horizon. If a PlanConfig is provided,
        its flattened receding horizon must match the trained chunk length.
        """
        return int(self.plan_horizon)

    def set_env(self, env: Any) -> None:
        """Associate the policy with an environment and validate action shape."""
        super().set_env(env)
        self.reset()

        env_action_shape = tuple(getattr(env.action_space, "shape", ()))
        if len(env_action_shape) == 0:
            raise ValueError("env.action_space.shape must be defined for SinglePeakPolicy.")

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
        """Clear runtime state so the next call re-plans from scratch."""
        self._action_buffer = deque(maxlen=self.plan_horizon)
        self._last_plan = None
        self._num_replans = 0

    def get_action(self, info_dict: dict[str, Any], **kwargs: Any) -> np.ndarray:
        """Return one environment-step action for each env.

        Input:
            info_dict["pixels"]: raw or processed obs history
            info_dict["goal"]: raw or processed goal history

        Output:
            action: [num_envs, action_dim] as numpy
        """
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "pixels" in info_dict, "'pixels' must be provided in info_dict"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        prepared_info = self._prepare_info(dict(info_dict))

        if len(self._action_buffer) == 0:
            plan = self.plan_actions(prepared_info)  # [num_envs, plan_horizon, action_dim]
            self._last_plan = plan
            self._num_replans += 1
            self._action_buffer.extend(plan.transpose(0, 1))

        action = self._action_buffer.popleft()  # [num_envs, action_dim]
        target_shape = self._env_action_shape or tuple(self.env.action_space.shape)
        action = action.reshape(*target_shape)
        action = action.detach().cpu().numpy()

        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)

        return action

    @torch.inference_mode()
    def plan_actions(self, prepared_info: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict one action chunk and reshape it into environment steps.

        prepared_info["pixels"]: [num_envs, history, C, H, W]
        prepared_info["goal"]: [num_envs, history, C, H, W]
        returns: [num_envs, plan_horizon, action_dim]
        """
        z_cur, z_goal = self.encode_current_goal(prepared_info)  # [B, latent_dim], [B, latent_dim]
        u_hat = self.planner(z_cur, z_goal)  # [B, action_chunk_dim]

        if u_hat.ndim != 2:
            raise ValueError(f"Planner output must have shape [B, action_chunk_dim], got {tuple(u_hat.shape)}.")
        if int(u_hat.shape[-1]) != self.action_chunk_dim:
            raise ValueError(
                f"Planner output width {u_hat.shape[-1]} does not match action_chunk_dim {self.action_chunk_dim}."
            )

        plan = torch.stack(
            [
                unflatten_action_chunk(
                    u_hat[env_index].detach().cpu().float(),
                    plan_horizon=self.plan_horizon,
                    action_dim=self.action_dim,
                )
                for env_index in range(int(u_hat.shape[0]))
            ],
            dim=0,
        )  # [num_envs, plan_horizon, action_dim]
        return plan

    @torch.inference_mode()
    def encode_current_goal(
        self,
        prepared_info: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode current obs and goal obs with the world model encoder.

        prepared_info["pixels"]: [num_envs, history, C, H, W]
        prepared_info["goal"]: [num_envs, history, C, H, W]
        z_cur: [num_envs, latent_dim]
        z_goal: [num_envs, latent_dim]
        """
        if "pixels" not in prepared_info:
            raise KeyError("'pixels' must be present in prepared_info.")
        if "goal" not in prepared_info:
            raise KeyError("'goal' must be present in prepared_info.")

        pixels = prepared_info["pixels"]
        goal_pixels = prepared_info["goal"]
        if not torch.is_tensor(pixels) or not torch.is_tensor(goal_pixels):
            raise TypeError("prepared_info['pixels'] and prepared_info['goal'] must be torch.Tensor values.")
        if pixels.ndim != 5:
            raise ValueError(f"prepared_info['pixels'] must have shape [B, T, C, H, W], got {tuple(pixels.shape)}.")
        if goal_pixels.ndim != 5:
            raise ValueError(f"prepared_info['goal'] must have shape [B, T, C, H, W], got {tuple(goal_pixels.shape)}.")

        device = next(self.world_model.parameters()).device
        current_encoded = self.world_model.encode({"pixels": pixels.to(device)})
        goal_encoded = self.world_model.encode({"pixels": goal_pixels.to(device)})

        if "emb" not in current_encoded or "emb" not in goal_encoded:
            raise KeyError("world_model.encode(...) must return a dict containing 'emb'.")

        z_cur = current_encoded["emb"][:, -1].detach()  # [num_envs, latent_dim]
        z_goal = goal_encoded["emb"][:, -1].detach()  # [num_envs, latent_dim]

        if z_cur.shape != z_goal.shape:
            raise ValueError(
                f"Current and goal latent shapes must match, got {tuple(z_cur.shape)} and {tuple(z_goal.shape)}."
            )
        if z_cur.ndim != 2:
            raise ValueError(f"Encoded latents must have shape [B, latent_dim], got {tuple(z_cur.shape)}.")
        if int(z_cur.shape[-1]) != self.latent_dim:
            raise ValueError(
                f"Encoded latent dim {z_cur.shape[-1]} does not match planner latent_dim {self.latent_dim}."
            )

        return z_cur, z_goal

    def _validate_runtime_contracts(self) -> None:
        if not hasattr(self.world_model, "encode"):
            raise TypeError("world_model must expose an encode(info_dict) method.")
        if not isinstance(self.planner, SinglePeakPlannerModel):
            raise TypeError(f"planner must be a SinglePeakPlannerModel, got {type(self.planner)}.")
        if self.cfg is not None:
            expected_plan_horizon = int(self.cfg.receding_horizon * self.cfg.action_block)
            if expected_plan_horizon != self.plan_horizon:
                raise ValueError(
                    "Single-peak planner chunk length does not match PlanConfig receding horizon: "
                    f"{self.plan_horizon} != {expected_plan_horizon} "
                    f"(receding_horizon={self.cfg.receding_horizon}, action_block={self.cfg.action_block})."
                )
