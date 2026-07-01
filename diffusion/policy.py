from __future__ import annotations

import time
import math
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from stable_worldmodel import PlanConfig
from stable_worldmodel.policy import BasePolicy

from diffusion.corrector import (
    ActionChunkCorrector,
    CorrectorBundle,
    load_corrector_bundle,
)
from diffusion.model import (
    DiffusionPlannerBundle,
    DiffusionPlannerModel,
    load_diffusion_planner_bundle,
)
from diffusion.prediction_error import (
    compute_prediction_error,
    compute_trigger_error,
    resolve_prediction_error_check,
    summarize_prediction_error_records,
)
from planners.latent_rollout import latent_rollout
from planners.single_peak_data import clone_info_dict, unflatten_action_chunk


TASK_ALIASES = {
    "pusht": "pusht",
    "tworoom": "tworoom",
    "two-room": "tworoom",
    "two_room": "tworoom",
    "reacher": "reacher",
    "researcher": "reacher",
}


@dataclass(frozen=True)
class DiffusionRuntimeSpec:
    task: str | None
    block_horizon: int
    receding_horizon: int
    action_block: int
    goal_offset_steps: int | None
    eval_budget: int | None


@dataclass(frozen=True)
class PredictionErrorCheckpoint:
    step: int
    plan_index: int
    rollout_blocks: int
    z_real: torch.Tensor
    z_pred: torch.Tensor
    z_goal: torch.Tensor
    errors: torch.Tensor
    max_error: float
    mean_error: float


def normalize_task_name(task_name: str | None) -> str | None:
    if task_name in [None, "", "null"]:
        return None
    normalized = str(task_name).strip().lower()
    return TASK_ALIASES.get(normalized, normalized)


def maybe_get_nested(source: dict[str, Any] | None, path: list[str]) -> Any:
    current: Any = source
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def maybe_positive_int(value: Any) -> int | None:
    if value in [None, "", "null"]:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"Expected a positive integer, got {parsed}.")
    return parsed


class DiffusionPlannerPolicy(BasePolicy):
    """Runtime wrapper for the anchor-conditioned truncated diffusion planner.

    Runtime flow:
        1. Preprocess raw env info via BasePolicy._prepare_info.
        2. Encode current obs and goal obs with the world model encoder.
        3. Run truncated diffusion from anchors to generate K candidate action chunks.
        4. Convert candidate chunks from [B, K, D] to world-model rollout format.
        5. Score K candidates via world_model.get_cost(...).
        6. Select the lowest-cost candidate per env.
        7. Reshape the selected chunk to [plan_horizon, action_dim].
        8. Push only the first runtime_execute_steps per-step actions into an internal action buffer.
        9. Pop one [num_envs, action_dim] action on each get_action call.

    Shapes:
        prepared_info["pixels"]: [num_envs, history, C, H, W]
        prepared_info["goal"]: [num_envs, history, C, H, W]
        prepared_info["action"]: [num_envs, history, action_dim] if present
        z_cur: [num_envs, latent_dim]
        z_goal: [num_envs, latent_dim]
        generated["candidates"]: [num_envs, K, action_chunk_dim]
        candidate_steps: [num_envs, K, plan_horizon, action_dim]
        candidate_blocks: [num_envs, K, receding_horizon, action_block * action_dim]
        world_model_costs: [num_envs, K]
        selected_candidates: [num_envs, action_chunk_dim]
        selected_plan: [num_envs, plan_horizon, action_dim]
        executed_plan: [num_envs, runtime_execute_steps, action_dim]
        buffered action: [num_envs, action_dim]
        get_action(...) return: [num_envs, action_dim]
    """

    def __init__(
        self,
        world_model: torch.nn.Module,
        planner: DiffusionPlannerModel,
        config: PlanConfig | None = None,
        process: dict[str, Any] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
        planner_bundle: DiffusionPlannerBundle | None = None,
        diffusion_eta: float = 0.0,
        num_candidates: int | None = None,
        truncation_steps: int | None = None,
        start_timestep: int | None = None,
        noise_scale: float = 1.0,
        sampling_temperature: float = 1.0,
        selection_mode: str = "wm_only",
        score_topk: int | None = None,
        goal_offset_steps: int | None = None,
        eval_budget: int | None = None,
        runtime_execute_steps: int | None = None,
        corrective_enabled: bool = False,
        corrective_mode: str = "none",
        corrective_correction_interval: int = 2,
        corrective_error_threshold: float = 0.5,
        corrective_trigger_stat: str = "max",
        corrective_trigger_quantile: float = 0.9,
        corrective_trigger_scope: str = "per_env",
        corrective_error_metric: str = "l2",
        corrective_log_prediction_error: bool = False,
        corrector: ActionChunkCorrector | None = None,
        corrector_bundle: CorrectorBundle | None = None,
        corrector_path: str | Path | None = None,
        refinement_enabled: bool = False,
        refinement_steps: int = 1,
        refinement_step_size: float = 0.03,
        refinement_topk: int | None = None,
        refinement_goal_weight: float = 1.0,
        refinement_prior_weight: float = 0.05,
        refinement_smoothness_weight: float = 0.005,
        refinement_grad_clip_norm: float | None = None,
        rerank_delta_weight: float = 0.0,
        rerank_jerk_weight: float = 0.0,
        rerank_action_l2_weight: float = 0.0,
        rerank_clip_weight: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.type = "diffusion"
        self.world_model = world_model
        planner_device = next(world_model.parameters()).device
        self.planner = planner.to(planner_device).eval()
        self.cfg = config
        self.process = process or {}
        self.transform = transform or {}
        self.planner_bundle = planner_bundle
        self.diffusion_eta = float(diffusion_eta)
        self.requested_num_candidates = (
            None if num_candidates is None else int(num_candidates)
        )
        self.proposal_truncation_steps = (
            None if truncation_steps is None else int(truncation_steps)
        )
        self.proposal_start_timestep = (
            None if start_timestep is None else int(start_timestep)
        )
        self.proposal_noise_scale = float(noise_scale)
        self.proposal_sampling_temperature = float(sampling_temperature)
        self.selection_mode = str(selection_mode).lower().strip()
        self.score_topk = None if score_topk is None else int(score_topk)
        self.requested_goal_offset_steps = None if goal_offset_steps is None else int(goal_offset_steps)
        self.requested_eval_budget = None if eval_budget is None else int(eval_budget)
        self.requested_runtime_execute_steps = (
            None if runtime_execute_steps is None else int(runtime_execute_steps)
        )
        self.corrective_enabled = bool(corrective_enabled)
        self.corrective_mode = str(corrective_mode).lower().strip()
        self.corrective_correction_interval = int(corrective_correction_interval)
        self.corrective_error_threshold = float(corrective_error_threshold)
        self.corrective_trigger_stat = str(corrective_trigger_stat).lower().strip()
        self.corrective_trigger_quantile = float(corrective_trigger_quantile)
        self.corrective_trigger_scope = str(corrective_trigger_scope).lower().strip()
        self.corrective_error_metric = str(corrective_error_metric).lower().strip()
        self.corrective_log_prediction_error = bool(corrective_log_prediction_error)
        self.corrector = None if corrector is None else corrector.to(planner_device).eval()
        self.corrector_bundle = corrector_bundle
        self.corrector_path = None if corrector_path in [None, "", "null"] else str(corrector_path)
        self.refinement_enabled = bool(refinement_enabled)
        self.refinement_steps = int(refinement_steps)
        self.refinement_step_size = float(refinement_step_size)
        self.refinement_topk = None if refinement_topk is None else int(refinement_topk)
        self.refinement_goal_weight = float(refinement_goal_weight)
        self.refinement_prior_weight = float(refinement_prior_weight)
        self.refinement_smoothness_weight = float(refinement_smoothness_weight)
        self.refinement_grad_clip_norm = (
            None if refinement_grad_clip_norm is None else float(refinement_grad_clip_norm)
        )
        self.rerank_delta_weight = float(rerank_delta_weight)
        self.rerank_jerk_weight = float(rerank_jerk_weight)
        self.rerank_action_l2_weight = float(rerank_action_l2_weight)
        self.rerank_clip_weight = float(rerank_clip_weight)
        self.runtime_spec = self._resolve_runtime_spec(config=config)

        self._action_buffer: deque[torch.Tensor] = deque(maxlen=self.runtime_execute_steps)
        self._last_plan: torch.Tensor | None = None
        self._last_executed_plan: torch.Tensor | None = None
        self._last_candidates: torch.Tensor | None = None
        self._last_model_score_logits: torch.Tensor | None = None
        self._last_world_model_costs: torch.Tensor | None = None
        self._last_selected_indices: torch.Tensor | None = None
        self._last_selected_wm_costs: torch.Tensor | None = None
        self._last_selected_model_scores: torch.Tensor | None = None
        self._last_has_finite_costs: torch.Tensor | None = None
        self._last_finite_candidate_mask: torch.Tensor | None = None
        self._last_fallback_to_model_score: torch.Tensor | None = None
        self._last_initial_noisy_candidates: torch.Tensor | None = None
        self._last_final_noisy_state: torch.Tensor | None = None
        self._last_unrefined_candidates: torch.Tensor | None = None
        self._last_refined_candidates: torch.Tensor | None = None
        self._last_refinement_cost_before: float | None = None
        self._last_refinement_cost_after: float | None = None
        self._last_refinement_goal_cost_before: float | None = None
        self._last_refinement_goal_cost_after: float | None = None
        self._last_refinement_delta_norm: float | None = None
        self._last_refinement_candidate_count = 0
        self._last_refinement_steps = 0
        self._last_truncation_timesteps: torch.Tensor | None = None
        self._last_plan_start_latent: torch.Tensor | None = None
        self._current_plan: torch.Tensor | None = None
        self._current_plan_start_latent: torch.Tensor | None = None
        self._current_plan_index = 0
        self._current_plan_executed_steps = 0
        self._logged_prediction_error_steps: set[int] = set()
        self._checked_prediction_error_steps: set[int] = set()
        self._prediction_error_records: list[dict[str, float | int | str]] = []
        self._corrective_check_count = 0
        self._corrective_replan_count = 0
        self._corrective_replan_error_records: list[dict[str, float | int | str]] = []
        self._corrective_correction_count = 0
        self._corrective_correction_norms: list[float] = []
        self._corrective_action_delta_norms: list[float] = []
        self._corrective_correction_time_total_sec = 0.0
        self._num_replans = 0
        self._planning_time_total_sec = 0.0
        self._generation_time_total_sec = 0.0
        self._scoring_time_total_sec = 0.0
        self._selection_time_total_sec = 0.0
        self._refinement_time_total_sec = 0.0
        self._wm_scoring_time_total_sec = 0.0
        self._wm_rollout_time_total_sec = 0.0
        self._wm_goal_encode_time_total_sec = 0.0
        self._wm_criterion_time_total_sec = 0.0
        self._wm_scoring_call_count = 0
        self._wm_rollout_candidate_count = 0
        self._wm_rollout_block_count = 0
        self._wm_refinement_rollout_time_total_sec = 0.0
        self._wm_refinement_rollout_call_count = 0
        self._wm_refinement_rollout_candidate_count = 0
        self._wm_refinement_rollout_block_count = 0
        self._env_action_shape: tuple[int, ...] | None = None
        self._action_low: np.ndarray | None = None
        self._action_high: np.ndarray | None = None
        self._action_dtype: np.dtype | None = None

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
        diffusion_eta: float = 0.0,
        num_candidates: int | None = None,
        truncation_steps: int | None = None,
        start_timestep: int | None = None,
        noise_scale: float = 1.0,
        sampling_temperature: float = 1.0,
        selection_mode: str = "wm_only",
        score_topk: int | None = None,
        goal_offset_steps: int | None = None,
        eval_budget: int | None = None,
        runtime_execute_steps: int | None = None,
        corrective_enabled: bool = False,
        corrective_mode: str = "none",
        corrective_correction_interval: int = 2,
        corrective_error_threshold: float = 0.5,
        corrective_trigger_stat: str = "max",
        corrective_trigger_quantile: float = 0.9,
        corrective_trigger_scope: str = "per_env",
        corrective_error_metric: str = "l2",
        corrective_log_prediction_error: bool = False,
        corrector_path: str | Path | None = None,
        refinement_enabled: bool = False,
        refinement_steps: int = 1,
        refinement_step_size: float = 0.03,
        refinement_topk: int | None = None,
        refinement_goal_weight: float = 1.0,
        refinement_prior_weight: float = 0.05,
        refinement_smoothness_weight: float = 0.005,
        refinement_grad_clip_norm: float | None = None,
        rerank_delta_weight: float = 0.0,
        rerank_jerk_weight: float = 0.0,
        rerank_action_l2_weight: float = 0.0,
        rerank_clip_weight: float = 0.0,
        **kwargs: Any,
    ) -> "DiffusionPlannerPolicy":
        bundle = load_diffusion_planner_bundle(bundle_path, map_location=map_location)
        planner = bundle.instantiate_model(map_location=map_location)
        planner.load_state_dict(bundle.model_state_dict)
        planner.eval()
        corrector_bundle = None
        corrector = None
        if corrector_path not in [None, "", "null"]:
            corrector_bundle = load_corrector_bundle(corrector_path, map_location=map_location)
            corrector = corrector_bundle.instantiate_model(map_location=map_location)
            corrector.eval()
        return cls(
            world_model=world_model,
            planner=planner,
            config=config,
            process=process,
            transform=transform,
            planner_bundle=bundle,
            diffusion_eta=diffusion_eta,
            num_candidates=num_candidates,
            truncation_steps=truncation_steps,
            start_timestep=start_timestep,
            noise_scale=noise_scale,
            sampling_temperature=sampling_temperature,
            selection_mode=selection_mode,
            score_topk=score_topk,
            goal_offset_steps=goal_offset_steps,
            eval_budget=eval_budget,
            runtime_execute_steps=runtime_execute_steps,
            corrective_enabled=corrective_enabled,
            corrective_mode=corrective_mode,
            corrective_correction_interval=corrective_correction_interval,
            corrective_error_threshold=corrective_error_threshold,
            corrective_trigger_stat=corrective_trigger_stat,
            corrective_trigger_quantile=corrective_trigger_quantile,
            corrective_trigger_scope=corrective_trigger_scope,
            corrective_error_metric=corrective_error_metric,
            corrective_log_prediction_error=corrective_log_prediction_error,
            corrector=corrector,
            corrector_bundle=corrector_bundle,
            corrector_path=corrector_path,
            refinement_enabled=refinement_enabled,
            refinement_steps=refinement_steps,
            refinement_step_size=refinement_step_size,
            refinement_topk=refinement_topk,
            refinement_goal_weight=refinement_goal_weight,
            refinement_prior_weight=refinement_prior_weight,
            refinement_smoothness_weight=refinement_smoothness_weight,
            refinement_grad_clip_norm=refinement_grad_clip_norm,
            rerank_delta_weight=rerank_delta_weight,
            rerank_jerk_weight=rerank_jerk_weight,
            rerank_action_l2_weight=rerank_action_l2_weight,
            rerank_clip_weight=rerank_clip_weight,
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
    def num_candidates(self) -> int:
        return int(self.planner.num_anchors)

    @property
    def base_num_candidates(self) -> int:
        return int(self.planner.num_anchors)

    @property
    def proposal_rounds(self) -> int:
        if self.requested_num_candidates is None:
            return 1
        return int(self.requested_num_candidates // self.base_num_candidates)

    @property
    def effective_num_candidates(self) -> int:
        return int(self.base_num_candidates * self.proposal_rounds)

    @property
    def block_horizon(self) -> int:
        return int(self.runtime_spec.block_horizon)

    @property
    def action_chunk_horizon(self) -> int:
        return int(self.plan_horizon)

    @property
    def runtime_execute_steps(self) -> int:
        if self.requested_runtime_execute_steps is None:
            return int(self.action_chunk_horizon)
        return int(self.requested_runtime_execute_steps)

    @property
    def runtime_truncation_steps(self) -> int:
        if self.proposal_truncation_steps is None:
            return int(self.planner.truncation_steps)
        return int(self.proposal_truncation_steps)

    @property
    def runtime_start_timestep(self) -> int:
        default_start = int(self.planner.schedule.truncation_timesteps[0].item())
        if self.proposal_start_timestep is None:
            return default_start
        return int(self.proposal_start_timestep)

    @property
    def flatten_receding_horizon(self) -> int:
        """Executed replan interval in environment steps."""
        return int(self.runtime_execute_steps)

    @property
    def receding_horizon(self) -> int:
        return int(self.runtime_spec.receding_horizon)

    @property
    def action_block(self) -> int:
        return int(self.runtime_spec.action_block)

    @property
    def blocked_action_dim(self) -> int:
        return int(self.action_block * self.action_dim)

    @property
    def task(self) -> str | None:
        return self.runtime_spec.task

    @property
    def goal_offset_steps(self) -> int | None:
        return self.runtime_spec.goal_offset_steps

    @property
    def eval_budget(self) -> int | None:
        return self.runtime_spec.eval_budget

    @property
    def action_clip_range(self) -> tuple[float | None, float | None]:
        if self._action_low is None or self._action_high is None:
            return None, None
        return float(np.nanmin(self._action_low)), float(np.nanmax(self._action_high))

    def set_env(self, env: Any) -> None:
        """Associate the policy with an environment and validate action shape."""
        super().set_env(env)
        self.reset()

        env_action_shape = tuple(getattr(env.action_space, "shape", ()))
        if len(env_action_shape) == 0:
            raise ValueError("env.action_space.shape must be defined for DiffusionPlannerPolicy.")

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

        action_space = getattr(env, "action_space", None)
        if hasattr(action_space, "low") and hasattr(action_space, "high"):
            self._action_low = np.asarray(action_space.low)
            self._action_high = np.asarray(action_space.high)
            self._action_dtype = np.dtype(getattr(action_space, "dtype", np.float32))
        else:
            self._action_low = None
            self._action_high = None
            self._action_dtype = None

    def reset(self) -> None:
        """Clear runtime state so the next call re-plans from scratch."""
        self._action_buffer = deque(maxlen=self.runtime_execute_steps)
        self._last_plan = None
        self._last_executed_plan = None
        self._last_candidates = None
        self._last_model_score_logits = None
        self._last_world_model_costs = None
        self._last_selected_indices = None
        self._last_selected_wm_costs = None
        self._last_selected_model_scores = None
        self._last_has_finite_costs = None
        self._last_finite_candidate_mask = None
        self._last_fallback_to_model_score = None
        self._last_initial_noisy_candidates = None
        self._last_final_noisy_state = None
        self._last_score_topk_indices = None
        self._last_score_topk_world_model_costs = None
        self._last_score_topk_model_scores = None
        self._last_unrefined_candidates = None
        self._last_refined_candidates = None
        self._last_refinement_cost_before = None
        self._last_refinement_cost_after = None
        self._last_refinement_goal_cost_before = None
        self._last_refinement_goal_cost_after = None
        self._last_refinement_delta_norm = None
        self._last_refinement_candidate_count = 0
        self._last_refinement_steps = 0
        self._last_truncation_timesteps = None
        self._last_plan_start_latent = None
        self._current_plan = None
        self._current_plan_start_latent = None
        self._current_plan_index = 0
        self._current_plan_executed_steps = 0
        self._logged_prediction_error_steps = set()
        self._checked_prediction_error_steps = set()
        self._prediction_error_records = []
        self._corrective_check_count = 0
        self._corrective_replan_count = 0
        self._corrective_replan_error_records = []
        self._corrective_correction_count = 0
        self._corrective_correction_norms = []
        self._corrective_action_delta_norms = []
        self._corrective_correction_time_total_sec = 0.0
        self._num_replans = 0
        self._planning_time_total_sec = 0.0
        self._generation_time_total_sec = 0.0
        self._scoring_time_total_sec = 0.0
        self._selection_time_total_sec = 0.0
        self._refinement_time_total_sec = 0.0
        self._wm_scoring_time_total_sec = 0.0
        self._wm_rollout_time_total_sec = 0.0
        self._wm_goal_encode_time_total_sec = 0.0
        self._wm_criterion_time_total_sec = 0.0
        self._wm_scoring_call_count = 0
        self._wm_rollout_candidate_count = 0
        self._wm_rollout_block_count = 0
        self._wm_refinement_rollout_time_total_sec = 0.0
        self._wm_refinement_rollout_call_count = 0
        self._wm_refinement_rollout_candidate_count = 0
        self._wm_refinement_rollout_block_count = 0

    @staticmethod
    def _sync_cuda_if_available() -> None:
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass

    def get_action(self, info_dict: dict[str, Any], **kwargs: Any) -> np.ndarray:
        """Return one environment-step action for each env."""
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "pixels" in info_dict, "'pixels' must be provided in info_dict"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        prepared_info = self._prepare_info(dict(info_dict))
        planned_this_call = False

        self._maybe_handle_corrective_checkpoint(prepared_info)

        if len(self._action_buffer) == 0:
            plan = self.plan_actions(prepared_info)  # [num_envs, plan_horizon, action_dim]
            self._last_plan = plan
            executed_plan = plan[:, : self.runtime_execute_steps, :]  # [num_envs, runtime_execute_steps, action_dim]
            self._last_executed_plan = executed_plan
            self._current_plan = plan.detach()
            self._current_plan_start_latent = (
                None if self._last_plan_start_latent is None else self._last_plan_start_latent.detach()
            )
            self._current_plan_index += 1
            self._current_plan_executed_steps = 0
            self._logged_prediction_error_steps = set()
            self._checked_prediction_error_steps = set()
            self._num_replans += 1
            self._action_buffer.extend(executed_plan.transpose(0, 1))
            planned_this_call = True

        action = self._action_buffer.popleft()  # [num_envs, action_dim]
        self._current_plan_executed_steps += 1
        target_shape = self._env_action_shape or tuple(self.env.action_space.shape)
        action = action.reshape(*target_shape)
        action = action.detach().cpu().numpy()

        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)

        preclip_min, preclip_max = self._compute_action_range(action)
        clipped_any = False
        if self._action_low is not None and self._action_high is not None:
            clipped_action = np.clip(action, self._action_low, self._action_high)
            clipped_any = not np.allclose(clipped_action, action)
            action = clipped_action

        if self._action_dtype is not None:
            action = action.astype(self._action_dtype, copy=False)

        postclip_min, postclip_max = self._compute_action_range(action)
        if planned_this_call or clipped_any:
            print(
                "[diffusion-action] "
                f"mode={self.selection_mode} "
                f"preclip_min={preclip_min:.6f} preclip_max={preclip_max:.6f} "
                f"postclip_min={postclip_min:.6f} postclip_max={postclip_max:.6f} "
                f"clipped={int(clipped_any)}"
            )

        return action

    def get_prediction_error_records(self) -> list[dict[str, float | int | str]]:
        """Return a copy of raw prediction-error checkpoint records."""
        return [dict(record) for record in self._prediction_error_records]

    def get_prediction_error_summary(
        self,
        episode_successes: Any,
    ) -> dict[str, float | int]:
        return summarize_prediction_error_records(
            self._prediction_error_records,
            episode_successes,
        )

    @torch.inference_mode()
    def _maybe_log_prediction_error(
        self,
        prepared_info: dict[str, torch.Tensor],
    ) -> None:
        if not self.corrective_log_prediction_error:
            return
        checkpoint = self._compute_prediction_error_checkpoint(prepared_info)
        if checkpoint is None:
            return
        self._record_prediction_error_checkpoint(checkpoint)

    @torch.inference_mode()
    def _maybe_handle_corrective_checkpoint(
        self,
        prepared_info: dict[str, torch.Tensor],
    ) -> None:
        checkpoint = self._compute_prediction_error_checkpoint(prepared_info)
        if checkpoint is None:
            return

        if self.corrective_log_prediction_error:
            self._record_prediction_error_checkpoint(checkpoint)

        if self.corrective_mode not in {"replan", "learned"}:
            return

        self._corrective_check_count += 1
        trigger_error = compute_trigger_error(
            checkpoint.errors,
            stat=self.corrective_trigger_stat,
            quantile=self.corrective_trigger_quantile,
        )
        env_indices = self._resolve_corrective_replan_env_indices(
            checkpoint,
            trigger_error=trigger_error,
        )
        if len(env_indices) == 0:
            return

        record = {
            "step": int(checkpoint.step),
            "plan_index": int(checkpoint.plan_index),
            "rollout_blocks": int(checkpoint.rollout_blocks),
            "max_error": float(checkpoint.max_error),
            "mean_error": float(checkpoint.mean_error),
            "trigger_error": float(trigger_error),
            "trigger_stat": self.corrective_trigger_stat,
            "trigger_quantile": float(self.corrective_trigger_quantile),
            "trigger_scope": self.corrective_trigger_scope,
            "threshold": float(self.corrective_error_threshold),
            "env_indices": [int(index) for index in env_indices],
            "env_count": int(len(env_indices)),
            "metric": self.corrective_error_metric,
            "mode": self.corrective_mode,
        }

        if self.corrective_mode == "learned":
            self._apply_learned_correction(checkpoint, env_indices, record=record)
            return

        self._corrective_replan_count += 1
        self._corrective_replan_error_records.append(record)
        if self.corrective_trigger_scope == "batch":
            self._discard_current_plan_for_corrective_replan()
            return
        self._replan_selected_envs(prepared_info, env_indices)

    @torch.inference_mode()
    def _apply_learned_correction(
        self,
        checkpoint: PredictionErrorCheckpoint,
        env_indices: list[int],
        *,
        record: dict[str, Any],
    ) -> None:
        if self.corrector is None:
            raise ValueError("corrective.mode=learned requires a loaded corrector.")
        if len(env_indices) == 0:
            return
        if self._current_plan is None:
            return
        if len(self._action_buffer) == 0:
            return

        correction_start = time.perf_counter()
        remaining_steps = int(len(self._action_buffer))
        if hasattr(self.corrector, "remain_horizon"):
            expected_steps = int(getattr(self.corrector, "remain_horizon"))
            if remaining_steps != expected_steps:
                return

        buffer_steps = list(self._action_buffer)
        u_remain = torch.stack(
            [step.detach().clone() for step in buffer_steps],
            dim=1,
        )  # [B, remaining_steps, action_dim]
        selected_cpu = torch.as_tensor(env_indices, dtype=torch.long, device=u_remain.device)
        selected_device = selected_cpu.to(device=checkpoint.z_real.device)
        first_parameter = next(self.corrector.parameters(), None)
        if first_parameter is None:
            corrector_device = u_remain.device
            corrector_dtype = u_remain.dtype
        else:
            corrector_device = first_parameter.device
            corrector_dtype = first_parameter.dtype
        z_real = checkpoint.z_real.index_select(0, selected_device).to(
            device=corrector_device,
            dtype=corrector_dtype,
        )
        z_goal = checkpoint.z_goal.index_select(0, selected_device).to(
            device=corrector_device,
            dtype=corrector_dtype,
        )
        error_latent = (checkpoint.z_real - checkpoint.z_pred).index_select(0, selected_device).to(
            device=corrector_device,
            dtype=corrector_dtype,
        )
        original = u_remain.index_select(0, selected_cpu)
        corrected = self.corrector(
            z_real,
            z_goal,
            error_latent,
            original.to(device=corrector_device, dtype=corrector_dtype),
        )
        if tuple(corrected.shape) != tuple(original.shape):
            raise ValueError(
                "Corrector output shape must match selected remainder shape: "
                f"{tuple(corrected.shape)} != {tuple(original.shape)}."
            )
        corrected = corrected.to(device=u_remain.device, dtype=u_remain.dtype)
        for step_index in range(remaining_steps):
            updated = buffer_steps[step_index].clone()
            updated[env_indices] = corrected[:, step_index, :]
            buffer_steps[step_index] = updated
            plan_step = int(self._current_plan_executed_steps) + step_index
            if plan_step < int(self._current_plan.shape[1]):
                self._current_plan[env_indices, plan_step, :] = corrected[:, step_index, :].to(
                    device=self._current_plan.device,
                    dtype=self._current_plan.dtype,
                )
        self._action_buffer = deque(buffer_steps, maxlen=self.runtime_execute_steps)

        delta = corrected - original
        correction_norm = float(
            torch.linalg.vector_norm(corrected.reshape(int(corrected.shape[0]), -1), ord=2, dim=-1)
            .mean()
            .item()
        )
        delta_norm = float(
            torch.linalg.vector_norm(delta.reshape(int(delta.shape[0]), -1), ord=2, dim=-1)
            .mean()
            .item()
        )
        elapsed = time.perf_counter() - correction_start
        self._corrective_correction_count += 1
        self._corrective_correction_norms.append(correction_norm)
        self._corrective_action_delta_norms.append(delta_norm)
        self._corrective_correction_time_total_sec += float(elapsed)
        self._corrective_replan_error_records.append(
            {
                **record,
                "correction_norm": correction_norm,
                "action_delta_norm": delta_norm,
                "correction_time_sec": float(elapsed),
            }
        )

    @torch.inference_mode()
    def _maybe_trigger_corrective_replan(
        self,
        prepared_info: dict[str, torch.Tensor],
    ) -> bool:
        previous_count = int(getattr(self, "_corrective_replan_count", 0))
        self._maybe_handle_corrective_checkpoint(prepared_info)
        return int(getattr(self, "_corrective_replan_count", 0)) > previous_count

    def _discard_current_plan_for_corrective_replan(self) -> None:
        self._action_buffer.clear()
        self._current_plan = None
        self._current_plan_start_latent = None
        self._current_plan_executed_steps = 0
        self._logged_prediction_error_steps = set()
        self._checked_prediction_error_steps = set()

    def _resolve_corrective_replan_env_indices(
        self,
        checkpoint: PredictionErrorCheckpoint,
        *,
        trigger_error: float,
    ) -> list[int]:
        if self.corrective_trigger_scope == "batch":
            if trigger_error <= self.corrective_error_threshold:
                return []
            return list(range(int(checkpoint.errors.numel())))
        if self.corrective_trigger_scope != "per_env":
            raise ValueError(
                "corrective_trigger_scope must be one of per_env or batch, "
                f"got '{self.corrective_trigger_scope}'."
            )

        triggered = torch.nonzero(
            checkpoint.errors.detach().reshape(-1) > self.corrective_error_threshold,
            as_tuple=False,
        ).reshape(-1)
        return [int(index) for index in triggered.detach().cpu().tolist()]

    @torch.inference_mode()
    def _replan_selected_envs(
        self,
        prepared_info: dict[str, torch.Tensor],
        env_indices: list[int],
    ) -> None:
        if len(env_indices) == 0:
            return
        if self._current_plan is None:
            return
        if len(self._action_buffer) == 0:
            return

        replan_plan = self.plan_actions(prepared_info)
        self._last_plan = replan_plan
        executed_plan = replan_plan[:, : self.runtime_execute_steps, :].detach()
        self._last_executed_plan = executed_plan
        if self._last_plan_start_latent is not None and self._current_plan_start_latent is not None:
            self._current_plan_start_latent[env_indices] = self._last_plan_start_latent.detach()[env_indices]
        self._current_plan[env_indices] = replan_plan.detach()[env_indices]

        remaining_steps = min(int(len(self._action_buffer)), int(executed_plan.shape[1]))
        replacement = executed_plan[:, :remaining_steps, :].transpose(0, 1)
        buffer_steps = list(self._action_buffer)
        for step_index in range(remaining_steps):
            updated = buffer_steps[step_index].clone()
            updated[env_indices] = replacement[step_index, env_indices]
            buffer_steps[step_index] = updated
        self._action_buffer = deque(buffer_steps, maxlen=self.runtime_execute_steps)
        self._num_replans += 1

    @torch.inference_mode()
    def _compute_prediction_error_checkpoint(
        self,
        prepared_info: dict[str, torch.Tensor],
    ) -> PredictionErrorCheckpoint | None:
        if not self.corrective_enabled:
            return None
        if self.corrective_mode not in {"none", "replan", "learned"}:
            raise ValueError(
                "corrective_mode must be one of none, replan, learned, "
                f"got '{self.corrective_mode}'."
            )
        if self._current_plan is None or self._current_plan_start_latent is None:
            return None

        rollout_blocks = resolve_prediction_error_check(
            prefix_steps=self._current_plan_executed_steps,
            action_block=self.action_block,
            correction_interval=self.corrective_correction_interval,
        )
        if rollout_blocks is None:
            return None
        if self._current_plan_executed_steps in self._checked_prediction_error_steps:
            return None

        z_real, z_goal = self.encode_current_goal(prepared_info)
        prefix = self._current_plan[
            :,
            : self._current_plan_executed_steps,
            :,
        ].to(device=z_real.device, dtype=z_real.dtype)
        action_blocks = prefix.reshape(
            int(prefix.shape[0]),
            int(rollout_blocks),
            self.blocked_action_dim,
        )
        rollout = latent_rollout(
            world_model=self.world_model,
            z_context=self._current_plan_start_latent.to(
                device=z_real.device,
                dtype=z_real.dtype,
            ),
            action_blocks=action_blocks,
            history_size=int(prepared_info["pixels"].shape[1]),
            return_sequence=False,
            freeze_world_model=True,
        )
        errors = compute_prediction_error(
            z_real,
            rollout["z_terminal"],
            metric=self.corrective_error_metric,
        )
        detached_errors = errors.detach()
        self._checked_prediction_error_steps.add(int(self._current_plan_executed_steps))
        return PredictionErrorCheckpoint(
            step=int(self._current_plan_executed_steps),
            plan_index=int(self._current_plan_index),
            rollout_blocks=int(rollout_blocks),
            z_real=z_real.detach(),
            z_pred=rollout["z_terminal"].detach(),
            z_goal=z_goal.detach(),
            errors=detached_errors,
            max_error=float(torch.max(detached_errors).item()),
            mean_error=float(torch.mean(detached_errors).item()),
        )

    def _record_prediction_error_checkpoint(
        self,
        checkpoint: PredictionErrorCheckpoint,
    ) -> None:
        for env_index, value in enumerate(checkpoint.errors.detach().cpu().tolist()):
            self._prediction_error_records.append(
                {
                    "env_index": int(env_index),
                    "step": int(checkpoint.step),
                    "plan_index": int(checkpoint.plan_index),
                    "rollout_blocks": int(checkpoint.rollout_blocks),
                    "error": float(value),
                    "metric": self.corrective_error_metric,
                }
            )
        self._logged_prediction_error_steps.add(int(checkpoint.step))

    @torch.inference_mode()
    def plan_actions(self, prepared_info: dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate K candidates, rerank them with the world model, and reshape the winner.

        returns:
            selected_plan: [num_envs, plan_horizon, action_dim]
        """
        planning_start = time.perf_counter()
        generation_start = time.perf_counter()
        generation = self.generate_candidates(prepared_info)
        generation_time = time.perf_counter() - generation_start
        candidates = generation["candidates"]  # [B, K, action_chunk_dim]
        model_scores = generation["score_logits"]  # [B, K]

        scoring_start = time.perf_counter()
        if self.selection_mode == "score_topk_wm":
            refined_candidates = candidates
            topk_candidates, topk_scores, topk_world_model_costs, topk_indices = (
                self.score_prefilter_candidates_with_world_model(
                    prepared_info,
                    candidates,
                    model_scores,
                )
            )
            world_model_costs = self._expand_score_topk_costs(
                model_scores=model_scores,
                topk_indices=topk_indices,
                topk_world_model_costs=topk_world_model_costs,
            )
            refinement_time = 0.0
            scoring_time = time.perf_counter() - scoring_start

            selection_start = time.perf_counter()
            topk_selected, topk_selected_indices, fallback_mask = self.select_best_candidates(
                topk_candidates,
                topk_world_model_costs,
                topk_scores,
            )
            selected_indices = topk_indices.gather(1, topk_selected_indices.view(-1, 1)).squeeze(1)
            selected_candidates = topk_selected
            selection_time = time.perf_counter() - selection_start
        else:
            initial_world_model_costs = self.score_candidates_with_world_model(
                prepared_info,
                candidates,
            )  # [B, K]
            scoring_time = time.perf_counter() - scoring_start

            refinement_start = time.perf_counter()
            refined_candidates = self.refine_candidates_with_world_model(
                prepared_info,
                candidates,
                world_model_costs=initial_world_model_costs,
                model_scores=model_scores,
            )
            refinement_time = time.perf_counter() - refinement_start
            if refined_candidates is candidates or torch.equal(refined_candidates, candidates):
                world_model_costs = initial_world_model_costs
            else:
                rescoring_start = time.perf_counter()
                world_model_costs = self.score_candidates_with_world_model(
                    prepared_info,
                    refined_candidates,
                )  # [B, K]
                scoring_time += time.perf_counter() - rescoring_start

            selection_start = time.perf_counter()
            selected_candidates, selected_indices, fallback_mask = self.select_best_candidates(
                refined_candidates,
                world_model_costs,
                model_scores,
            )  # [B, action_chunk_dim], [B], [B]
            selection_time = time.perf_counter() - selection_start

        selected_wm_costs = world_model_costs.gather(1, selected_indices.view(-1, 1)).squeeze(1)  # [B]
        selected_model_scores = model_scores.gather(1, selected_indices.view(-1, 1)).squeeze(1)  # [B]
        finite_mask = torch.isfinite(world_model_costs)  # [B, K]
        has_finite = finite_mask.any(dim=-1)  # [B]

        self._last_candidates = refined_candidates.detach().cpu().float()  # [B, K, action_chunk_dim]
        self._last_unrefined_candidates = candidates.detach().cpu().float()  # [B, K, action_chunk_dim]
        self._last_refined_candidates = refined_candidates.detach().cpu().float()  # [B, K, action_chunk_dim]
        self._last_model_score_logits = model_scores.detach().cpu().float()  # [B, K]
        self._last_world_model_costs = world_model_costs.detach().cpu().float()  # [B, K]
        self._last_selected_indices = selected_indices.detach().cpu().long()  # [B]
        self._last_selected_wm_costs = selected_wm_costs.detach().cpu().float()  # [B]
        self._last_selected_model_scores = selected_model_scores.detach().cpu().float()  # [B]
        self._last_has_finite_costs = has_finite.detach().cpu().bool()  # [B]
        self._last_finite_candidate_mask = finite_mask.detach().cpu().bool()  # [B, K]
        self._last_fallback_to_model_score = fallback_mask.detach().cpu().bool()  # [B]
        self._last_initial_noisy_candidates = generation["initial_noisy_candidates"].detach().cpu().float()
        self._last_final_noisy_state = generation["final_noisy_state"].detach().cpu().float()
        self._last_truncation_timesteps = generation["truncation_timesteps"].detach().cpu().long()
        self.log_rerank_summary(
            selected_indices=selected_indices,
            selected_wm_costs=selected_wm_costs,
            selected_model_scores=selected_model_scores,
            world_model_costs=world_model_costs,
            fallback_mask=fallback_mask,
        )

        plan = torch.stack(
            [
                unflatten_action_chunk(
                    selected_candidates[env_index].detach().cpu().float(),
                    plan_horizon=self.plan_horizon,
                    action_dim=self.action_dim,
                )
                for env_index in range(int(selected_candidates.shape[0]))
            ],
            dim=0,
        )  # [num_envs, plan_horizon, action_dim]
        planning_time = time.perf_counter() - planning_start
        self._planning_time_total_sec += planning_time
        self._generation_time_total_sec += generation_time
        self._refinement_time_total_sec += refinement_time
        self._scoring_time_total_sec += scoring_time
        self._selection_time_total_sec += selection_time
        return plan

    @torch.inference_mode()
    def generate_candidates(
        self,
        prepared_info: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run truncated diffusion to produce candidate action chunks.

        returns:
            out["candidates"]: [num_envs, K, action_chunk_dim]
            out["score_logits"]: [num_envs, K]
            out["initial_noisy_candidates"]: [num_envs, K, action_chunk_dim]
            out["final_noisy_state"]: [num_envs, K, action_chunk_dim]
        """
        z_cur, z_goal = self.encode_current_goal(prepared_info)  # [B, latent_dim], [B, latent_dim]
        self._last_plan_start_latent = z_cur.detach()
        round_candidates: list[torch.Tensor] = []
        round_scores: list[torch.Tensor] = []
        round_initial_noisy: list[torch.Tensor] = []
        round_final_noisy: list[torch.Tensor] = []
        truncation_timesteps: torch.Tensor | None = None

        for _round_idx in range(self.proposal_rounds):
            outputs = self.planner.generate_candidates(
                z_cur,
                z_goal,
                eta=self.diffusion_eta,
                truncation_steps=self.proposal_truncation_steps,
                start_timestep=self.proposal_start_timestep,
                noise_scale=self.proposal_noise_scale,
                sampling_temperature=self.proposal_sampling_temperature,
                return_intermediates=False,
            )
            candidates = outputs["candidates"]  # [B, K_base, action_chunk_dim]
            score_logits = outputs["score_logits"]  # [B, K_base]

            if candidates.ndim != 3:
                raise ValueError(f"candidates must have shape [B, K, D], got {tuple(candidates.shape)}.")
            if score_logits.ndim != 2:
                raise ValueError(f"score_logits must have shape [B, K], got {tuple(score_logits.shape)}.")
            if tuple(candidates.shape[:2]) != tuple(score_logits.shape):
                raise ValueError(
                    f"Candidate leading dims {tuple(candidates.shape[:2])} do not match score_logits {tuple(score_logits.shape)}."
                )
            if int(candidates.shape[1]) != self.base_num_candidates:
                raise ValueError(
                    f"Candidate dim {candidates.shape[1]} does not match base_num_candidates {self.base_num_candidates}."
                )
            if int(candidates.shape[-1]) != self.action_chunk_dim:
                raise ValueError(
                    f"Candidate width {candidates.shape[-1]} does not match action_chunk_dim {self.action_chunk_dim}."
                )

            round_candidates.append(candidates)
            round_scores.append(score_logits)
            round_initial_noisy.append(outputs["initial_noisy_candidates"])
            round_final_noisy.append(outputs["final_noisy_state"])
            if truncation_timesteps is None:
                truncation_timesteps = outputs["truncation_timesteps"]

        if truncation_timesteps is None:
            raise RuntimeError("Diffusion proposal generation produced no timestep schedule.")

        all_candidates = torch.cat(round_candidates, dim=1)  # [B, K_eff, action_chunk_dim]
        all_scores = torch.cat(round_scores, dim=1)  # [B, K_eff]
        all_initial_noisy = torch.cat(round_initial_noisy, dim=1)  # [B, K_eff, action_chunk_dim]
        all_final_noisy = torch.cat(round_final_noisy, dim=1)  # [B, K_eff, action_chunk_dim]

        if int(all_candidates.shape[1]) != self.effective_num_candidates:
            raise ValueError(
                f"Effective candidate dim {all_candidates.shape[1]} does not match configured effective_num_candidates {self.effective_num_candidates}."
            )

        return {
            "candidates": all_candidates,  # [B, K_eff, action_chunk_dim]
            "score_logits": all_scores,  # [B, K_eff]
            "scores": all_scores,  # [B, K_eff]
            "initial_noisy_candidates": all_initial_noisy,  # [B, K_eff, action_chunk_dim]
            "final_noisy_state": all_final_noisy,  # [B, K_eff, action_chunk_dim]
            "truncation_timesteps": truncation_timesteps.detach().clone(),  # [T_runtime]
        }

    def _candidate_smoothness_cost(self, candidates: torch.Tensor) -> torch.Tensor:
        """Mean squared adjacent action-step difference for [B, K, action_chunk_dim]."""
        candidate_steps = candidates.reshape(
            int(candidates.shape[0]),
            int(candidates.shape[1]),
            self.plan_horizon,
            self.action_dim,
        )
        if int(candidate_steps.shape[2]) <= 1:
            return torch.zeros((), device=candidates.device, dtype=candidates.dtype)
        diffs = candidate_steps[:, :, 1:, :] - candidate_steps[:, :, :-1, :]
        return diffs.square().mean()

    def compute_rerank_penalty(self, candidates: torch.Tensor) -> torch.Tensor:
        """Return per-candidate action quality penalty for reranking.

        candidates: [B, K, action_chunk_dim]
        returns: [B, K]
        """
        if candidates.ndim != 3:
            raise ValueError(f"candidates must have shape [B, K, D], got {tuple(candidates.shape)}.")
        candidate_steps = candidates.reshape(
            int(candidates.shape[0]),
            int(candidates.shape[1]),
            self.plan_horizon,
            self.action_dim,
        )
        penalty = torch.zeros(
            int(candidates.shape[0]),
            int(candidates.shape[1]),
            dtype=candidates.dtype,
            device=candidates.device,
        )
        if self.rerank_action_l2_weight > 0.0:
            penalty = penalty + float(self.rerank_action_l2_weight) * candidate_steps.square().mean(dim=(2, 3))
        if self.rerank_delta_weight > 0.0 and int(candidate_steps.shape[2]) > 1:
            delta = candidate_steps[:, :, 1:, :] - candidate_steps[:, :, :-1, :]
            penalty = penalty + float(self.rerank_delta_weight) * delta.square().mean(dim=(2, 3))
        if self.rerank_jerk_weight > 0.0 and int(candidate_steps.shape[2]) > 2:
            jerk = candidate_steps[:, :, 2:, :] - 2.0 * candidate_steps[:, :, 1:-1, :] + candidate_steps[:, :, :-2, :]
            penalty = penalty + float(self.rerank_jerk_weight) * jerk.square().mean(dim=(2, 3))
        if self.rerank_clip_weight > 0.0 and self._action_low is not None and self._action_high is not None:
            low = torch.as_tensor(self._action_low, dtype=candidates.dtype, device=candidates.device)
            high = torch.as_tensor(self._action_high, dtype=candidates.dtype, device=candidates.device)
            low = low.reshape(-1)[-self.action_dim :].view(1, 1, 1, self.action_dim)
            high = high.reshape(-1)[-self.action_dim :].view(1, 1, 1, self.action_dim)
            lower_excess = torch.clamp(low - candidate_steps, min=0.0)
            upper_excess = torch.clamp(candidate_steps - high, min=0.0)
            clip_excess = lower_excess.square() + upper_excess.square()
            penalty = penalty + float(self.rerank_clip_weight) * clip_excess.mean(dim=(2, 3))
        return penalty

    def _refinement_cost(
        self,
        *,
        z_cur: torch.Tensor,
        z_goal: torch.Tensor,
        candidates: torch.Tensor,
        initial_candidates: torch.Tensor,
        history_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_blocks = self.flatten_candidates_to_action_blocks(candidates)
        self._sync_cuda_if_available()
        rollout_start = time.perf_counter()
        try:
            rollout = latent_rollout(
                world_model=self.world_model,
                z_context=z_cur,
                action_blocks=action_blocks,
                history_size=history_size,
                return_sequence=False,
                freeze_world_model=True,
            )
        finally:
            self._sync_cuda_if_available()
            self._wm_refinement_rollout_time_total_sec += time.perf_counter() - rollout_start
            self._wm_refinement_rollout_call_count += 1
            self._wm_refinement_rollout_candidate_count += int(action_blocks.shape[0]) * int(action_blocks.shape[1])
            self._wm_refinement_rollout_block_count += (
                int(action_blocks.shape[0])
                * int(action_blocks.shape[1])
                * int(action_blocks.shape[2])
            )
        z_terminal = rollout["z_terminal"]
        if z_terminal.ndim != 3:
            raise ValueError(
                "latent_rollout z_terminal must have shape [B, K, latent_dim], "
                f"got {tuple(z_terminal.shape)}."
            )
        goal_cost = (z_terminal - z_goal.unsqueeze(1).detach()).square().mean()
        prior_cost = (candidates - initial_candidates.detach()).square().mean()
        smoothness_cost = self._candidate_smoothness_cost(candidates)
        total_cost = (
            self.refinement_goal_weight * goal_cost
            + self.refinement_prior_weight * prior_cost
            + self.refinement_smoothness_weight * smoothness_cost
        )
        return total_cost, goal_cost, prior_cost, smoothness_cost

    def refine_candidates_with_world_model(
        self,
        prepared_info: dict[str, torch.Tensor],
        candidates: torch.Tensor,
        *,
        world_model_costs: torch.Tensor | None = None,
        model_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Refine action candidates by differentiating through frozen LeWM dynamics."""
        self._last_refinement_cost_before = None
        self._last_refinement_cost_after = None
        self._last_refinement_goal_cost_before = None
        self._last_refinement_goal_cost_after = None
        self._last_refinement_delta_norm = None
        self._last_refinement_candidate_count = 0
        self._last_refinement_steps = 0

        if not self.refinement_enabled:
            return candidates
        if self.refinement_steps <= 0:
            return candidates
        if self.refinement_step_size <= 0.0:
            return candidates
        if self.refinement_goal_weight == 0.0 and self.refinement_prior_weight == 0.0 and self.refinement_smoothness_weight == 0.0:
            return candidates

        with torch.inference_mode(False):
            return self._refine_candidates_with_world_model_grad(
                prepared_info,
                candidates,
                world_model_costs=world_model_costs,
                model_scores=model_scores,
            )

    @torch.enable_grad()
    def _refine_candidates_with_world_model_grad(
        self,
        prepared_info: dict[str, torch.Tensor],
        candidates: torch.Tensor,
        *,
        world_model_costs: torch.Tensor | None = None,
        model_scores: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(candidates) or candidates.ndim != 3:
            raise ValueError(f"candidates must have shape [B, K, D], got {getattr(candidates, 'shape', None)}.")
        if not torch.is_tensor(model_scores) or model_scores.ndim != 2:
            raise ValueError(f"model_scores must have shape [B, K], got {getattr(model_scores, 'shape', None)}.")
        if tuple(model_scores.shape) != tuple(candidates.shape[:2]):
            raise ValueError(
                f"model_scores shape {tuple(model_scores.shape)} does not match candidates {tuple(candidates.shape[:2])}."
            )
        if world_model_costs is not None:
            if not torch.is_tensor(world_model_costs) or world_model_costs.ndim != 2:
                raise ValueError(
                    "world_model_costs must have shape [B, K] when provided, "
                    f"got {getattr(world_model_costs, 'shape', None)}."
                )
            if tuple(world_model_costs.shape) != tuple(candidates.shape[:2]):
                raise ValueError(
                    "world_model_costs shape "
                    f"{tuple(world_model_costs.shape)} does not match candidates {tuple(candidates.shape[:2])}."
                )

        z_cur, z_goal = self.encode_current_goal(prepared_info)
        z_cur = z_cur.detach().clone()
        z_goal = z_goal.detach().clone()
        history_size = int(prepared_info["pixels"].shape[1])
        candidates = candidates.detach().clone()
        candidate_count = int(candidates.shape[1])
        if self.refinement_topk is None:
            selected_indices = torch.arange(candidate_count, device=candidates.device).view(1, -1)
            selected_indices = selected_indices.expand(int(candidates.shape[0]), -1)
        else:
            topk = min(int(self.refinement_topk), candidate_count)
            if topk <= 0:
                return candidates
            if world_model_costs is None:
                raise ValueError("refinement_topk requires world_model_costs for WM-only candidate preselection.")
            finite_mask = torch.isfinite(world_model_costs)
            safe_costs = torch.where(
                finite_mask,
                world_model_costs,
                torch.full_like(world_model_costs, float("inf")),
            )
            selected_indices = torch.topk(safe_costs.detach(), k=topk, dim=-1, largest=False).indices

        gather_index = selected_indices.unsqueeze(-1).expand(-1, -1, int(candidates.shape[-1]))
        initial_selected = candidates.gather(1, gather_index).detach()
        refined = initial_selected.clone().detach().to(device=z_cur.device, dtype=z_cur.dtype)
        refined.requires_grad_(True)

        with torch.no_grad():
            total_before, goal_before, _, _ = self._refinement_cost(
                z_cur=z_cur,
                z_goal=z_goal,
                candidates=refined.detach(),
                initial_candidates=initial_selected.to(device=z_cur.device, dtype=z_cur.dtype),
                history_size=history_size,
            )
            self._last_refinement_cost_before = float(total_before.detach().item())
            self._last_refinement_goal_cost_before = float(goal_before.detach().item())

        for _step_idx in range(int(self.refinement_steps)):
            total_cost, _, _, _ = self._refinement_cost(
                z_cur=z_cur,
                z_goal=z_goal,
                candidates=refined,
                initial_candidates=initial_selected.to(device=z_cur.device, dtype=z_cur.dtype),
                history_size=history_size,
            )
            grad = torch.autograd.grad(total_cost, refined, retain_graph=False, create_graph=False)[0]
            if self.refinement_grad_clip_norm is not None:
                grad_norm = torch.linalg.vector_norm(grad.reshape(int(grad.shape[0]), -1), ord=2, dim=-1)
                scale = (float(self.refinement_grad_clip_norm) / grad_norm.clamp(min=1e-12)).clamp(max=1.0)
                while scale.ndim < grad.ndim:
                    scale = scale.unsqueeze(-1)
                grad = grad * scale
            with torch.no_grad():
                refined = refined - float(self.refinement_step_size) * grad
            refined = refined.detach().requires_grad_(True)

        refined_detached = refined.detach()
        with torch.no_grad():
            total_after, goal_after, _, _ = self._refinement_cost(
                z_cur=z_cur,
                z_goal=z_goal,
                candidates=refined_detached,
                initial_candidates=initial_selected.to(device=z_cur.device, dtype=z_cur.dtype),
                history_size=history_size,
            )
            self._last_refinement_cost_after = float(total_after.detach().item())
            self._last_refinement_goal_cost_after = float(goal_after.detach().item())
            self._last_refinement_delta_norm = float(
                torch.linalg.vector_norm((refined_detached - initial_selected.to(device=z_cur.device, dtype=z_cur.dtype)).reshape(int(refined_detached.shape[0]), -1), ord=2, dim=-1)
                .mean()
                .detach()
                .item()
            )

        output = candidates.detach().clone().to(device=refined_detached.device, dtype=refined_detached.dtype)
        output.scatter_(1, gather_index.to(device=output.device), refined_detached)
        self._last_refinement_candidate_count = int(selected_indices.numel())
        self._last_refinement_steps = int(self.refinement_steps)
        return output.to(device=candidates.device, dtype=candidates.dtype)

    @torch.inference_mode()
    def score_candidates_with_world_model(
        self,
        prepared_info: dict[str, torch.Tensor],
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidate action chunks with the world model goal cost.

        prepared_info["pixels"]: [B, history, C, H, W]
        prepared_info["goal"]: [B, history, C, H, W]
        candidates: [B, K, action_chunk_dim]

        world-model scoring path:
            candidate_steps: [B, K, plan_horizon, action_dim]
            candidate_blocks: [B, K, receding_horizon, action_block * action_dim]
            world_model.get_cost(...) -> [B, K]
        """
        self._sync_cuda_if_available()
        scoring_start = time.perf_counter()
        scoring_info = self.prepare_info_for_scoring(prepared_info)
        candidate_blocks = self.flatten_candidates_to_action_blocks(candidates)  # [B, K, R, block_action_dim]
        expanded_info = self.expand_prepared_info_for_candidates(
            scoring_info,
            num_candidates=int(candidates.shape[1]),
        )
        try:
            world_model_costs = self.compute_world_model_costs_from_rollout(
                expanded_info=expanded_info,
                candidate_blocks=candidate_blocks,
            )
        finally:
            self._sync_cuda_if_available()
            self._wm_scoring_time_total_sec += time.perf_counter() - scoring_start
            self._wm_scoring_call_count += 1
            self._wm_rollout_candidate_count += int(candidate_blocks.shape[0]) * int(candidate_blocks.shape[1])
            self._wm_rollout_block_count += (
                int(candidate_blocks.shape[0])
                * int(candidate_blocks.shape[1])
                * int(candidate_blocks.shape[2])
            )
        if not torch.is_tensor(world_model_costs):
            raise TypeError(
                "World-model reranking must return a torch.Tensor, "
                f"got {type(world_model_costs)}."
            )
        if world_model_costs.ndim != 2:
            raise ValueError(f"world_model_costs must have shape [B, K], got {tuple(world_model_costs.shape)}.")
        if tuple(world_model_costs.shape) != tuple(candidates.shape[:2]):
            raise ValueError(
                f"world_model_costs shape {tuple(world_model_costs.shape)} does not match candidates {tuple(candidates.shape[:2])}."
            )
        return world_model_costs

    @torch.inference_mode()
    def compute_world_model_costs_from_rollout(
        self,
        *,
        expanded_info: dict[str, Any],
        candidate_blocks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute goal cost with the repository's real JEPA rollout contract.

        expanded_info:
            pixels: [B, K, history, C, H, W]
            goal: [B, K, history, C, H, W]
            action: [B, K, history, action_dim]
        candidate_blocks:
            [B, K, receding_horizon, action_block * action_dim]

        returns:
            world_model_costs: [B, K]
        """
        if not hasattr(self.world_model, "rollout") or not hasattr(self.world_model, "criterion"):
            if not hasattr(self.world_model, "get_cost"):
                raise TypeError(
                    "world_model must expose either rollout(...)+criterion(...) "
                    "or get_cost(...) for diffusion reranking."
                )
            self._sync_cuda_if_available()
            rollout_start = time.perf_counter()
            costs = self.world_model.get_cost(clone_info_dict(expanded_info), candidate_blocks)
            self._sync_cuda_if_available()
            self._wm_rollout_time_total_sec += time.perf_counter() - rollout_start
            return costs

        device = next(self.world_model.parameters()).device
        rollout_info = clone_info_dict(expanded_info)
        for key, value in rollout_info.items():
            if torch.is_tensor(value):
                rollout_info[key] = value.to(device)

        candidate_blocks = candidate_blocks.to(device)

        # Mirror JEPA.get_cost(...), but expand goal embeddings across K candidates
        # before criterion() so shapes match:
        # goal_emb: [B, K, history, latent_dim]
        goal_info = {key: value[:, 0] for key, value in rollout_info.items() if torch.is_tensor(value)}
        if "goal" not in goal_info:
            raise KeyError("expanded_info must contain a 'goal' tensor for world-model scoring.")
        goal_info["pixels"] = goal_info["goal"]  # [B, history, C, H, W]

        for key in list(goal_info.keys()):
            if key.startswith("goal_"):
                goal_info[key[len("goal_") :]] = goal_info.pop(key)

        goal_info.pop("action", None)
        self._sync_cuda_if_available()
        goal_encode_start = time.perf_counter()
        encoded_goal = self.world_model.encode(goal_info)
        self._sync_cuda_if_available()
        self._wm_goal_encode_time_total_sec += time.perf_counter() - goal_encode_start
        if "emb" not in encoded_goal:
            raise KeyError("world_model.encode(goal_info) must return a dict containing 'emb'.")

        goal_emb = encoded_goal["emb"]  # [B, history, latent_dim]
        if goal_emb.ndim != 3:
            raise ValueError(
                "Encoded goal embeddings must have shape [B, history, latent_dim], "
                f"got {tuple(goal_emb.shape)}."
            )

        goal_emb = goal_emb.unsqueeze(1).expand(
            int(candidate_blocks.shape[0]),
            int(candidate_blocks.shape[1]),
            int(goal_emb.shape[1]),
            int(goal_emb.shape[2]),
        )  # [B, K, history, latent_dim]

        rollout_info["goal_emb"] = goal_emb
        self._sync_cuda_if_available()
        rollout_start = time.perf_counter()
        rollout_outputs = self.world_model.rollout(rollout_info, candidate_blocks)
        self._sync_cuda_if_available()
        self._wm_rollout_time_total_sec += time.perf_counter() - rollout_start
        if "predicted_emb" not in rollout_outputs:
            raise KeyError("world_model.rollout(...) must return a dict containing 'predicted_emb'.")
        self._sync_cuda_if_available()
        criterion_start = time.perf_counter()
        costs = self.world_model.criterion(rollout_outputs)
        self._sync_cuda_if_available()
        self._wm_criterion_time_total_sec += time.perf_counter() - criterion_start
        return costs

    @torch.inference_mode()
    def encode_current_goal(
        self,
        prepared_info: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode current obs and goal obs with the world model encoder."""
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

    def prepare_info_for_scoring(
        self,
        prepared_info: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Ensure prepared_info satisfies the world-model scoring prerequisites."""
        scoring_info = clone_info_dict(prepared_info)
        if "pixels" not in scoring_info or "goal" not in scoring_info:
            raise KeyError("prepared_info must contain 'pixels' and 'goal' for scoring.")

        pixels = scoring_info["pixels"]
        if not torch.is_tensor(pixels) or pixels.ndim != 5:
            raise ValueError(
                f"prepared_info['pixels'] must have shape [B, history, C, H, W], got {getattr(pixels, 'shape', None)}."
            )
        history_size = int(pixels.shape[1])
        if self.receding_horizon < history_size:
            raise ValueError(
                "world-model scoring requires receding_horizon >= observation history size, "
                f"got receding_horizon={self.receding_horizon}, history_size={history_size}."
            )

        if "action" not in scoring_info:
            # JEPA.get_cost() expects an action history key to exist so it can be copied into the
            # goal dict before rollout. If the environment did not provide it, fall back to zeros.
            scoring_info["action"] = torch.zeros(
                int(pixels.shape[0]),
                history_size,
                self.action_dim,
                dtype=torch.float32,
                device=pixels.device,
            )  # [B, history, action_dim]
        else:
            action = scoring_info["action"]
            if not torch.is_tensor(action) or action.ndim != 3:
                raise ValueError(
                    f"prepared_info['action'] must have shape [B, history, action_dim], got {getattr(action, 'shape', None)}."
                )
            if int(action.shape[0]) != int(pixels.shape[0]) or int(action.shape[1]) != history_size:
                raise ValueError(
                    "prepared_info['action'] batch/history dims must match prepared_info['pixels']: "
                    f"{tuple(action.shape[:2])} != {(int(pixels.shape[0]), history_size)}."
                )
            if int(action.shape[-1]) != self.action_dim:
                raise ValueError(
                    f"prepared_info['action'] dim {action.shape[-1]} does not match action_dim {self.action_dim}."
                )
        return scoring_info

    def expand_prepared_info_for_candidates(
        self,
        prepared_info: dict[str, Any],
        *,
        num_candidates: int,
    ) -> dict[str, Any]:
        """Expand prepared_info from [B, ...] to [B, K, ...] like CEM solver does."""
        if num_candidates <= 0:
            raise ValueError(f"num_candidates must be positive, got {num_candidates}.")

        expanded: dict[str, Any] = {}
        for key, value in prepared_info.items():
            if torch.is_tensor(value):
                expanded[key] = value.unsqueeze(1).expand(value.shape[0], num_candidates, *value.shape[1:])
            elif isinstance(value, np.ndarray):
                expanded[key] = np.repeat(value[:, None, ...], num_candidates, axis=1)
            else:
                expanded[key] = deepcopy(value)
        return expanded

    def flatten_candidates_to_action_blocks(
        self,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Convert flat action chunks to world-model rollout blocks.

        candidates: [B, K, action_chunk_dim]
        returns:
            candidate_blocks: [B, K, receding_horizon, action_block * action_dim]
        """
        if not torch.is_tensor(candidates) or candidates.ndim != 3:
            raise ValueError(f"candidates must have shape [B, K, D], got {getattr(candidates, 'shape', None)}.")
        if int(candidates.shape[-1]) != self.action_chunk_dim:
            raise ValueError(
                f"Candidate width {candidates.shape[-1]} does not match action_chunk_dim {self.action_chunk_dim}."
            )

        candidate_steps = candidates.reshape(
            int(candidates.shape[0]),
            int(candidates.shape[1]),
            self.plan_horizon,
            self.action_dim,
        )  # [B, K, plan_horizon, action_dim]
        candidate_blocks = candidate_steps.reshape(
            int(candidates.shape[0]),
            int(candidates.shape[1]),
            self.receding_horizon,
            self.blocked_action_dim,
        )  # [B, K, receding_horizon, action_block * action_dim]
        return candidate_blocks

    def normalize_candidate_values(
        self,
        values: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Normalize [B, K] candidate values row-wise for hybrid selection."""
        if values.ndim != 2:
            raise ValueError(f"values must have shape [B, K], got {tuple(values.shape)}.")
        if valid_mask is None:
            valid_mask = torch.ones_like(values, dtype=torch.bool)
        if valid_mask.shape != values.shape:
            raise ValueError(
                f"valid_mask shape {tuple(valid_mask.shape)} does not match values {tuple(values.shape)}."
            )

        mask_f = valid_mask.to(dtype=values.dtype)
        safe_values = torch.where(valid_mask, values, torch.zeros_like(values))
        count = mask_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mean = safe_values.sum(dim=-1, keepdim=True) / count
        centered = torch.where(valid_mask, values - mean, torch.zeros_like(values))
        var = centered.square().sum(dim=-1, keepdim=True) / count
        std = torch.sqrt(var).clamp(min=1e-6)
        normalized = centered / std
        return torch.where(valid_mask, normalized, torch.zeros_like(normalized))

    def compute_hybrid_selection_values(
        self,
        world_model_costs: torch.Tensor,
        model_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Blend normalized WM cost and model score into a single [B, K] value."""
        finite_mask = torch.isfinite(world_model_costs)  # [B, K]
        normalized_cost = self.normalize_candidate_values(world_model_costs, valid_mask=finite_mask)  # [B, K]
        normalized_scores = self.normalize_candidate_values(model_scores)  # [B, K]
        hybrid_values = normalized_scores - normalized_cost  # [B, K]
        return torch.where(
            finite_mask,
            hybrid_values,
            torch.full_like(hybrid_values, float("-inf")),
        )  # [B, K]

    def _resolve_score_topk(self, num_candidates: int) -> int:
        if self.score_topk is None:
            return min(16, int(num_candidates))
        return min(int(self.score_topk), int(num_candidates))

    def score_prefilter_candidates_with_world_model(
        self,
        prepared_info: dict[str, torch.Tensor],
        candidates: torch.Tensor,
        model_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score only the score-head top-k candidates with the world model.

        returns:
            topk_candidates: [B, topk, action_chunk_dim]
            topk_scores: [B, topk]
            topk_world_model_costs: [B, topk]
            topk_indices: [B, topk], original candidate indices
        """
        if candidates.ndim != 3:
            raise ValueError(f"candidates must have shape [B, K, D], got {tuple(candidates.shape)}.")
        if model_scores.ndim != 2:
            raise ValueError(f"model_scores must have shape [B, K], got {tuple(model_scores.shape)}.")
        if tuple(candidates.shape[:2]) != tuple(model_scores.shape):
            raise ValueError(
                f"candidates leading dims {tuple(candidates.shape[:2])} do not match model_scores {tuple(model_scores.shape)}."
            )
        topk = self._resolve_score_topk(int(candidates.shape[1]))
        if topk <= 0:
            raise ValueError(f"score_topk must be positive when using score_topk_wm, got {topk}.")

        topk_indices = torch.topk(model_scores, k=topk, dim=-1, largest=True).indices  # [B, topk]
        gather_index = topk_indices.unsqueeze(-1).expand(-1, -1, int(candidates.shape[-1]))
        topk_candidates = candidates.gather(1, gather_index)  # [B, topk, D]
        topk_scores = model_scores.gather(1, topk_indices)  # [B, topk]
        topk_world_model_costs = self.score_candidates_with_world_model(
            prepared_info,
            topk_candidates,
        )  # [B, topk]

        self._last_score_topk_indices = topk_indices.detach().cpu().long()
        self._last_score_topk_world_model_costs = topk_world_model_costs.detach().cpu().float()
        self._last_score_topk_model_scores = topk_scores.detach().cpu().float()
        return topk_candidates, topk_scores, topk_world_model_costs, topk_indices

    def _expand_score_topk_costs(
        self,
        *,
        model_scores: torch.Tensor,
        topk_indices: torch.Tensor | None,
        topk_world_model_costs: torch.Tensor | None,
    ) -> torch.Tensor:
        if topk_indices is None or topk_world_model_costs is None:
            raise ValueError("score_topk_wm requires topk indices and topk world-model costs.")
        topk_indices = topk_indices.to(device=model_scores.device)
        topk_world_model_costs = topk_world_model_costs.to(device=model_scores.device, dtype=model_scores.dtype)
        full_costs = torch.full_like(model_scores, float("inf"))
        full_costs.scatter_(1, topk_indices, topk_world_model_costs)
        return full_costs

    def score_and_select_candidates(
        self,
        prepared_info: dict[str, torch.Tensor],
        candidates: torch.Tensor,
        model_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.selection_mode != "score_topk_wm":
            world_model_costs = self.score_candidates_with_world_model(prepared_info, candidates)
            return self.select_best_candidates(candidates, world_model_costs, model_scores)

        topk_candidates, topk_scores, topk_world_model_costs, topk_indices = (
            self.score_prefilter_candidates_with_world_model(
                prepared_info,
                candidates,
                model_scores,
            )
        )
        selected_topk, selected_topk_indices, fallback_mask = self.select_best_candidates(
            topk_candidates,
            topk_world_model_costs,
            topk_scores,
        )
        selected_indices = topk_indices.gather(1, selected_topk_indices.view(-1, 1)).squeeze(1)
        return selected_topk, selected_indices, fallback_mask

    def select_best_candidates(
        self,
        candidates: torch.Tensor,
        world_model_costs: torch.Tensor,
        model_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select the lowest-cost candidate without using the deprecated score head for wm_only.

        returns:
            selected_candidates: [B, action_chunk_dim]
            selected_indices: [B]
            fallback_to_model_score: [B]
        """
        if candidates.ndim != 3:
            raise ValueError(f"candidates must have shape [B, K, D], got {tuple(candidates.shape)}.")
        if world_model_costs.ndim != 2 or model_scores.ndim != 2:
            raise ValueError("world_model_costs and model_scores must have shape [B, K].")
        if tuple(candidates.shape[:2]) != tuple(world_model_costs.shape):
            raise ValueError(
                f"candidates leading dims {tuple(candidates.shape[:2])} do not match world_model_costs {tuple(world_model_costs.shape)}."
            )
        if tuple(candidates.shape[:2]) != tuple(model_scores.shape):
            raise ValueError(
                f"candidates leading dims {tuple(candidates.shape[:2])} do not match model_scores {tuple(model_scores.shape)}."
            )

        finite_mask = torch.isfinite(world_model_costs)  # [B, K]
        has_finite = finite_mask.any(dim=-1)  # [B]
        rerank_penalty = self.compute_rerank_penalty(candidates)
        penalized_world_model_costs = world_model_costs + rerank_penalty
        penalized_world_model_costs = torch.where(
            torch.isfinite(world_model_costs),
            penalized_world_model_costs,
            world_model_costs,
        )
        safe_costs = torch.where(
            finite_mask,
            penalized_world_model_costs,
            torch.full_like(penalized_world_model_costs, float("inf")),
        )  # [B, K]

        best_by_cost = torch.argmin(safe_costs, dim=-1)  # [B]
        best_by_score = torch.argmax(model_scores, dim=-1)  # [B]
        if self.selection_mode in {"wm_only", "score_topk_wm"}:
            selected_indices = torch.where(has_finite, best_by_cost, torch.zeros_like(best_by_cost))  # [B]
            fallback_to_model_score = torch.zeros_like(best_by_cost, dtype=torch.bool)  # [B]
        elif self.selection_mode == "score_only":
            selected_indices = best_by_score  # [B]
            fallback_to_model_score = torch.ones_like(best_by_score, dtype=torch.bool)  # [B]
        elif self.selection_mode == "hybrid":
            hybrid_values = self.compute_hybrid_selection_values(penalized_world_model_costs, model_scores)  # [B, K]
            best_by_hybrid = torch.argmax(hybrid_values, dim=-1)  # [B]
            selected_indices = torch.where(has_finite, best_by_hybrid, best_by_score)  # [B]
            fallback_to_model_score = ~has_finite  # [B]
        else:
            raise ValueError(f"Unsupported diffusion selection_mode '{self.selection_mode}'.")

        gather_index = selected_indices.view(-1, 1, 1).expand(-1, 1, candidates.shape[-1])  # [B, 1, D]
        selected_candidates = candidates.gather(1, gather_index).squeeze(1)  # [B, action_chunk_dim]
        return selected_candidates, selected_indices, fallback_to_model_score

    def log_rerank_summary(
        self,
        *,
        selected_indices: torch.Tensor,
        selected_wm_costs: torch.Tensor,
        selected_model_scores: torch.Tensor,
        world_model_costs: torch.Tensor,
        fallback_mask: torch.Tensor,
    ) -> None:
        """Print compact rerank diagnostics once per planning call."""
        finite_mask = torch.isfinite(world_model_costs)  # [B, K]
        has_finite = finite_mask.any(dim=-1)  # [B]
        finite_candidate_rate = float(finite_mask.float().mean().item())
        all_bad_env_rate = float((~has_finite).float().mean().item())
        fallback_rate = float(fallback_mask.float().mean().item())

        selected_index_first = int(selected_indices[0].item()) if selected_indices.numel() > 0 else -1
        selected_wm_cost_first = (
            float(selected_wm_costs[0].item()) if selected_wm_costs.numel() > 0 else float("nan")
        )
        selected_model_score_first = (
            float(selected_model_scores[0].item()) if selected_model_scores.numel() > 0 else float("nan")
        )

        finite_selected_wm = selected_wm_costs[torch.isfinite(selected_wm_costs)]
        selected_wm_cost_mean = (
            float(finite_selected_wm.mean().item()) if finite_selected_wm.numel() > 0 else float("nan")
        )
        selected_model_score_mean = (
            float(selected_model_scores.mean().item()) if selected_model_scores.numel() > 0 else float("nan")
        )

        goal_offset = "unknown" if self.goal_offset_steps is None else str(self.goal_offset_steps)
        print(
            "[diffusion-rerank] "
            f"mode={self.selection_mode} goal_offset={goal_offset} "
            f"block_horizon={self.block_horizon} action_chunk_horizon={self.action_chunk_horizon} "
            f"runtime_execute_steps={self.runtime_execute_steps} "
            f"replan_interval={self.flatten_receding_horizon} "
            f"base_num_candidates={self.base_num_candidates} "
            f"proposal_rounds={self.proposal_rounds} "
            f"num_candidates={int(world_model_costs.shape[1])} "
            f"denoise_steps={self.runtime_truncation_steps} "
            f"start_timestep={self.runtime_start_timestep} "
            f"eta={self.diffusion_eta:.4f} "
            f"noise_scale={self.proposal_noise_scale:.4f} "
            f"temperature={self.proposal_sampling_temperature:.4f} "
            f"finite_candidate_rate={finite_candidate_rate:.4f} "
            f"all_bad_env_rate={all_bad_env_rate:.4f} fallback_rate={fallback_rate:.4f}"
        )
        print(
            "[diffusion-rerank] "
            f"selected_index_first={selected_index_first} "
            f"selected_wm_cost_first={selected_wm_cost_first:.6f} "
            f"selected_model_score_first={selected_model_score_first:.6f} "
            f"selected_wm_cost_mean={selected_wm_cost_mean:.6f} "
            f"selected_model_score_mean={selected_model_score_mean:.6f}"
        )

    def _compute_action_range(self, action: np.ndarray) -> tuple[float, float]:
        if action.size == 0:
            return float("nan"), float("nan")
        return float(np.nanmin(action)), float(np.nanmax(action))

    def _resolve_runtime_spec(
        self,
        *,
        config: PlanConfig | None,
    ) -> DiffusionRuntimeSpec:
        anchor_metadata = {}
        if self.planner_bundle is not None:
            anchor_metadata = dict(self.planner_bundle.anchor_metadata or {})
        elif hasattr(self.planner, "anchor_metadata"):
            anchor_metadata = dict(getattr(self.planner, "anchor_metadata", {}) or {})

        source_build_info = anchor_metadata.get("source_build_info", {})
        source_plan_config = maybe_get_nested(source_build_info, ["plan_config"]) or {}

        task = normalize_task_name(
            anchor_metadata.get("task")
            or source_build_info.get("task")
        )

        cfg_block_horizon = maybe_positive_int(getattr(config, "horizon", None)) if config is not None else None
        cfg_receding_horizon = (
            maybe_positive_int(getattr(config, "receding_horizon", None)) if config is not None else None
        )
        cfg_action_block = maybe_positive_int(getattr(config, "action_block", None)) if config is not None else None

        bundle_receding_horizon = maybe_positive_int(
            anchor_metadata.get("receding_horizon")
            or source_plan_config.get("receding_horizon")
            or maybe_get_nested(source_build_info, ["task_spec", "receding_horizon"])
        )
        bundle_action_block = maybe_positive_int(
            anchor_metadata.get("action_block")
            or source_plan_config.get("action_block")
            or maybe_get_nested(source_build_info, ["task_spec", "action_block"])
        )
        bundle_goal_offset_steps = maybe_positive_int(
            source_build_info.get("goal_offset_steps")
            or maybe_get_nested(source_build_info, ["task_spec", "goal_offset_steps"])
        )
        bundle_eval_budget = maybe_positive_int(
            source_build_info.get("eval_budget")
            or maybe_get_nested(source_build_info, ["task_spec", "eval_budget"])
        )

        receding_horizon = cfg_receding_horizon or bundle_receding_horizon
        action_block = cfg_action_block or bundle_action_block
        if receding_horizon is None or action_block is None:
            raise ValueError(
                "Could not infer diffusion runtime rollout shape. "
                "Provide eval plan_config.{receding_horizon,action_block} or use a diffusion bundle "
                "whose anchor metadata contains source_build_info.plan_config."
            )
        if cfg_receding_horizon is not None and bundle_receding_horizon is not None:
            if int(cfg_receding_horizon) != int(bundle_receding_horizon):
                raise ValueError(
                    "Eval config receding_horizon does not match diffusion bundle metadata: "
                    f"{cfg_receding_horizon} != {bundle_receding_horizon}."
                )
        if cfg_action_block is not None and bundle_action_block is not None:
            if int(cfg_action_block) != int(bundle_action_block):
                raise ValueError(
                    "Eval config action_block does not match diffusion bundle metadata: "
                    f"{cfg_action_block} != {bundle_action_block}."
                )

        expected_plan_horizon = int(receding_horizon * action_block)
        if expected_plan_horizon != self.action_chunk_horizon:
            raise ValueError(
                "Diffusion planner chunk length does not match resolved rollout shape: "
                f"{self.action_chunk_horizon} != {expected_plan_horizon} "
                f"(receding_horizon={receding_horizon}, action_block={action_block})."
            )

        goal_offset_steps = self.requested_goal_offset_steps or bundle_goal_offset_steps
        eval_budget = self.requested_eval_budget or bundle_eval_budget

        if cfg_block_horizon is None:
            cfg_block_horizon = int(receding_horizon)

        return DiffusionRuntimeSpec(
            task=task,
            block_horizon=int(cfg_block_horizon),
            receding_horizon=int(receding_horizon),
            action_block=int(action_block),
            goal_offset_steps=None if goal_offset_steps is None else int(goal_offset_steps),
            eval_budget=None if eval_budget is None else int(eval_budget),
        )

    def _validate_runtime_contracts(self) -> None:
        if not hasattr(self.world_model, "encode"):
            raise TypeError("world_model must expose an encode(info_dict) method.")
        if not (
            hasattr(self.world_model, "get_cost")
            or (hasattr(self.world_model, "rollout") and hasattr(self.world_model, "criterion"))
        ):
            raise TypeError(
                "world_model must expose get_cost(info_dict, action_candidates) or "
                "rollout(info_dict, action_candidates)+criterion(info_dict) for reranking."
            )
        if not isinstance(self.planner, DiffusionPlannerModel):
            raise TypeError(f"planner must be a DiffusionPlannerModel, got {type(self.planner)}.")

        expected_plan_horizon = int(self.receding_horizon * self.action_block)
        if expected_plan_horizon != self.plan_horizon:
            raise ValueError(
                "Diffusion planner chunk length does not match resolved receding horizon: "
                f"{self.plan_horizon} != {expected_plan_horizon} "
                f"(receding_horizon={self.receding_horizon}, action_block={self.action_block})."
            )
        if self.base_num_candidates <= 0:
            raise ValueError(
                f"base_num_candidates must be positive, got {self.base_num_candidates}."
            )
        if self.runtime_execute_steps <= 0:
            raise ValueError(
                f"runtime_execute_steps must be positive, got {self.runtime_execute_steps}."
            )
        if self.runtime_execute_steps > self.action_chunk_horizon:
            raise ValueError(
                "runtime_execute_steps cannot exceed the full action chunk horizon: "
                f"{self.runtime_execute_steps} > {self.action_chunk_horizon}."
            )
        if self.requested_num_candidates is not None:
            if self.requested_num_candidates < self.base_num_candidates:
                raise ValueError(
                    "requested num_candidates must be >= the anchor count so every anchor remains represented, "
                    f"got requested_num_candidates={self.requested_num_candidates}, "
                    f"base_num_candidates={self.base_num_candidates}."
                )
            if self.requested_num_candidates % self.base_num_candidates != 0:
                raise ValueError(
                    "requested num_candidates must be a positive multiple of the anchor count for "
                    "uniform per-anchor resampling, got "
                    f"requested_num_candidates={self.requested_num_candidates}, "
                    f"base_num_candidates={self.base_num_candidates}."
                )
        if self.diffusion_eta < 0.0:
            raise ValueError(f"diffusion_eta must be non-negative, got {self.diffusion_eta}.")
        if self.proposal_noise_scale < 0.0:
            raise ValueError(
                f"proposal_noise_scale must be non-negative, got {self.proposal_noise_scale}."
            )
        if self.proposal_sampling_temperature < 0.0:
            raise ValueError(
                "proposal_sampling_temperature must be non-negative, "
                f"got {self.proposal_sampling_temperature}."
            )
        if self.runtime_truncation_steps <= 0:
            raise ValueError(
                f"runtime truncation_steps must be positive, got {self.runtime_truncation_steps}."
            )
        if self.selection_mode not in {"wm_only", "hybrid", "score_only", "score_topk_wm"}:
            raise ValueError(
                "selection_mode must be one of {'wm_only', 'hybrid', 'score_only', 'score_topk_wm'}, "
                f"got '{self.selection_mode}'."
            )
        if self.score_topk is not None:
            if self.score_topk <= 0:
                raise ValueError(f"score_topk must be positive when set, got {self.score_topk}.")
            if self.score_topk > self.effective_num_candidates:
                raise ValueError(
                    "score_topk cannot exceed effective_num_candidates: "
                    f"{self.score_topk} > {self.effective_num_candidates}."
                )
        if self.refinement_steps < 0:
            raise ValueError(f"refinement_steps must be non-negative, got {self.refinement_steps}.")
        if self.refinement_step_size < 0.0:
            raise ValueError(
                f"refinement_step_size must be non-negative, got {self.refinement_step_size}."
            )
        if self.refinement_topk is not None:
            if self.refinement_topk <= 0:
                raise ValueError(f"refinement_topk must be positive when set, got {self.refinement_topk}.")
            if self.refinement_topk > self.effective_num_candidates:
                raise ValueError(
                    "refinement_topk cannot exceed effective_num_candidates: "
                    f"{self.refinement_topk} > {self.effective_num_candidates}."
                )
        if self.refinement_goal_weight < 0.0:
            raise ValueError(f"refinement_goal_weight must be non-negative, got {self.refinement_goal_weight}.")
        if self.refinement_prior_weight < 0.0:
            raise ValueError(f"refinement_prior_weight must be non-negative, got {self.refinement_prior_weight}.")
        if self.refinement_smoothness_weight < 0.0:
            raise ValueError(
                "refinement_smoothness_weight must be non-negative, "
                f"got {self.refinement_smoothness_weight}."
            )
        if self.refinement_grad_clip_norm is not None and self.refinement_grad_clip_norm <= 0.0:
            raise ValueError(
                "refinement_grad_clip_norm must be positive when set, "
                f"got {self.refinement_grad_clip_norm}."
            )
        if self.rerank_delta_weight < 0.0:
            raise ValueError(f"rerank_delta_weight must be non-negative, got {self.rerank_delta_weight}.")
        if self.rerank_jerk_weight < 0.0:
            raise ValueError(f"rerank_jerk_weight must be non-negative, got {self.rerank_jerk_weight}.")
        if self.rerank_action_l2_weight < 0.0:
            raise ValueError(
                f"rerank_action_l2_weight must be non-negative, got {self.rerank_action_l2_weight}."
            )
        if self.rerank_clip_weight < 0.0:
            raise ValueError(f"rerank_clip_weight must be non-negative, got {self.rerank_clip_weight}.")
        self._validate_learned_corrector_contract()

    def _expected_corrector_remain_horizon(self) -> int:
        effective_correction_steps = int(
            math.ceil(self.corrective_correction_interval / self.action_block)
            * self.action_block
        )
        return int(self.runtime_execute_steps - effective_correction_steps)

    def _validate_learned_corrector_contract(self) -> None:
        if not self.corrective_enabled or self.corrective_mode != "learned":
            return
        if self.corrector is None:
            raise ValueError("corrective.mode=learned requires a loaded corrector.")
        if not hasattr(self.corrector, "remain_horizon"):
            raise ValueError("Loaded corrector must expose remain_horizon.")
        expected = self._expected_corrector_remain_horizon()
        if expected <= 0:
            raise ValueError(
                "corrective.mode=learned requires runtime_execute_steps to exceed "
                f"the effective correction checkpoint, got expected remain_horizon={expected}."
            )
        actual = int(getattr(self.corrector, "remain_horizon"))
        if actual != expected:
            raise ValueError(
                "Corrector remain_horizon does not match eval corrective horizon: "
                f"{actual} != {expected}. "
                "Train the corrector with the same correction_interval/action_block "
                "and use matching corrective.execute_horizon."
            )
