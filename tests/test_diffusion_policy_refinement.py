import types
import unittest

import torch

from diffusion.policy import DiffusionPlannerPolicy, DiffusionRuntimeSpec
from planners.latent_rollout import latent_rollout


class LinearActionWorldModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.action_encoder = torch.nn.Identity()

    def encode(self, info):
        pixels = info["pixels"]
        return {"emb": pixels[:, :, :, 0, 0].float()}

    def predict(self, emb, act_emb):
        return emb[:, -1:, :] + act_emb[:, -1:, :]


def make_refinement_policy(*, enabled: bool):
    policy = DiffusionPlannerPolicy.__new__(DiffusionPlannerPolicy)
    policy.world_model = LinearActionWorldModel()
    policy.planner = types.SimpleNamespace(
        latent_dim=1,
        plan_horizon=2,
        action_dim=1,
        action_chunk_dim=2,
        num_anchors=2,
    )
    policy.runtime_spec = DiffusionRuntimeSpec(
        task="fake",
        block_horizon=2,
        receding_horizon=2,
        action_block=1,
        goal_offset_steps=2,
        eval_budget=2,
    )
    policy.refinement_enabled = enabled
    policy.refinement_steps = 2
    policy.refinement_step_size = 0.2
    policy.refinement_topk = None
    policy.refinement_goal_weight = 1.0
    policy.refinement_prior_weight = 0.0
    policy.refinement_smoothness_weight = 0.0
    policy.refinement_grad_clip_norm = None
    policy.score_topk = None
    policy.rerank_delta_weight = 0.0
    policy.rerank_jerk_weight = 0.0
    policy.rerank_action_l2_weight = 0.0
    policy.rerank_clip_weight = 0.0
    policy._last_refinement_cost_before = None
    policy._last_refinement_cost_after = None
    policy._last_refinement_goal_cost_before = None
    policy._last_refinement_goal_cost_after = None
    policy._last_refinement_delta_norm = None
    policy._last_refinement_candidate_count = 0
    policy._last_refinement_steps = 0
    policy._last_score_topk_indices = None
    policy._last_score_topk_world_model_costs = None
    policy._last_score_topk_model_scores = None
    policy._refinement_time_total_sec = 0.0
    policy._wm_scoring_time_total_sec = 0.0
    policy._wm_rollout_time_total_sec = 0.0
    policy._wm_goal_encode_time_total_sec = 0.0
    policy._wm_criterion_time_total_sec = 0.0
    policy._wm_scoring_call_count = 0
    policy._wm_rollout_candidate_count = 0
    policy._wm_rollout_block_count = 0
    policy._wm_refinement_rollout_time_total_sec = 0.0
    policy._wm_refinement_rollout_call_count = 0
    policy._wm_refinement_rollout_candidate_count = 0
    policy._wm_refinement_rollout_block_count = 0
    return policy


def goal_cost(policy, prepared_info, candidates):
    z_cur, z_goal = policy.encode_current_goal(prepared_info)
    action_blocks = policy.flatten_candidates_to_action_blocks(candidates)
    rollout = latent_rollout(
        world_model=policy.world_model,
        z_context=z_cur,
        action_blocks=action_blocks,
        history_size=int(prepared_info["pixels"].shape[1]),
        return_sequence=False,
        freeze_world_model=True,
    )
    return (rollout["z_terminal"] - z_goal.unsqueeze(1)).square().mean()


class DiffusionPolicyRefinementTest(unittest.TestCase):
    def test_refinement_disabled_returns_candidates_unchanged(self):
        policy = make_refinement_policy(enabled=False)
        prepared_info = {
            "pixels": torch.zeros(1, 1, 1, 1, 1),
            "goal": torch.zeros(1, 1, 1, 1, 1),
        }
        candidates = torch.tensor([[[1.0, 1.0], [0.5, 0.5]]])

        refined = policy.refine_candidates_with_world_model(
            prepared_info,
            candidates,
            model_scores=torch.tensor([[0.1, 0.2]]),
        )

        self.assertTrue(torch.equal(refined, candidates))
        self.assertEqual(policy._last_refinement_candidate_count, 0)

    def test_refinement_enabled_decreases_goal_cost(self):
        policy = make_refinement_policy(enabled=True)
        prepared_info = {
            "pixels": torch.zeros(1, 1, 1, 1, 1),
            "goal": torch.zeros(1, 1, 1, 1, 1),
        }
        candidates = torch.tensor([[[1.0, 1.0], [0.5, 0.5]]])

        before = goal_cost(policy, prepared_info, candidates)
        refined = policy.refine_candidates_with_world_model(
            prepared_info,
            candidates,
            model_scores=torch.tensor([[0.1, 0.2]]),
        )
        after = goal_cost(policy, prepared_info, refined)

        self.assertLess(float(after), float(before))
        self.assertGreater(policy._last_refinement_delta_norm, 0.0)
        self.assertEqual(policy._last_refinement_candidate_count, 2)
        self.assertEqual(policy._last_refinement_steps, 2)

    def test_refinement_runs_inside_inference_mode(self):
        policy = make_refinement_policy(enabled=True)
        prepared_info = {
            "pixels": torch.zeros(1, 1, 1, 1, 1),
            "goal": torch.zeros(1, 1, 1, 1, 1),
        }

        with torch.inference_mode():
            candidates = torch.tensor([[[1.0, 1.0], [0.5, 0.5]]])
            refined = policy.refine_candidates_with_world_model(
                prepared_info,
                candidates,
                model_scores=torch.tensor([[0.1, 0.2]]),
            )

        self.assertLess(float(goal_cost(policy, prepared_info, refined)), float(goal_cost(policy, prepared_info, candidates)))

    def test_wm_only_selection_does_not_fallback_to_score_when_costs_are_invalid(self):
        policy = make_refinement_policy(enabled=False)
        policy.selection_mode = "wm_only"
        candidates = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
        world_model_costs = torch.tensor([[float("nan"), float("nan")]])
        model_scores = torch.tensor([[0.1, 10.0]])

        selected, selected_indices, fallback_mask = policy.select_best_candidates(
            candidates,
            world_model_costs,
            model_scores,
        )

        self.assertEqual(selected_indices.tolist(), [0])
        self.assertFalse(bool(fallback_mask.item()))
        self.assertTrue(torch.equal(selected, candidates[:, 0, :]))

    def test_refinement_topk_uses_lowest_world_model_cost_not_score(self):
        policy = make_refinement_policy(enabled=True)
        policy.refinement_topk = 1
        prepared_info = {
            "pixels": torch.zeros(1, 1, 1, 1, 1),
            "goal": torch.zeros(1, 1, 1, 1, 1),
        }
        candidates = torch.tensor([[[2.0, 2.0], [0.5, 0.5]]])
        world_model_costs = torch.tensor([[16.0, 1.0]])
        model_scores = torch.tensor([[100.0, -100.0]])

        refined = policy.refine_candidates_with_world_model(
            prepared_info,
            candidates,
            world_model_costs=world_model_costs,
            model_scores=model_scores,
        )

        self.assertTrue(torch.equal(refined[:, 0, :], candidates[:, 0, :]))
        self.assertFalse(torch.equal(refined[:, 1, :], candidates[:, 1, :]))
        self.assertEqual(policy._last_refinement_candidate_count, 1)

    def test_smoothness_penalty_can_override_lower_world_model_cost(self):
        policy = make_refinement_policy(enabled=False)
        policy.selection_mode = "wm_only"
        policy.planner.plan_horizon = 4
        policy.planner.action_dim = 1
        policy.planner.action_chunk_dim = 4
        policy.rerank_delta_weight = 1.0
        policy.rerank_jerk_weight = 1.0
        candidates = torch.tensor([[[0.0, 2.0, -2.0, 2.0], [0.2, 0.2, 0.2, 0.2]]])
        world_model_costs = torch.tensor([[0.0, 1.0]])
        model_scores = torch.tensor([[0.0, 0.0]])

        selected, selected_indices, _ = policy.select_best_candidates(
            candidates,
            world_model_costs,
            model_scores,
        )

        self.assertEqual(selected_indices.tolist(), [1])
        self.assertTrue(torch.equal(selected, candidates[:, 1, :]))

    def test_clip_penalty_prefers_in_range_candidate(self):
        policy = make_refinement_policy(enabled=False)
        policy.selection_mode = "wm_only"
        policy.planner.plan_horizon = 2
        policy.planner.action_dim = 1
        policy.planner.action_chunk_dim = 2
        policy.rerank_clip_weight = 10.0
        policy._action_low = torch.tensor([-1.0]).numpy()
        policy._action_high = torch.tensor([1.0]).numpy()
        candidates = torch.tensor([[[1.5, 1.5], [0.9, 0.9]]])
        world_model_costs = torch.tensor([[0.0, 1.0]])
        model_scores = torch.tensor([[0.0, 0.0]])

        _, selected_indices, _ = policy.select_best_candidates(
            candidates,
            world_model_costs,
            model_scores,
        )

        self.assertEqual(selected_indices.tolist(), [1])

    def test_score_topk_wm_scores_only_score_prefiltered_candidates(self):
        policy = make_refinement_policy(enabled=False)
        policy.selection_mode = "score_topk_wm"
        policy.score_topk = 3
        policy._last_score_topk_indices = None
        policy._last_score_topk_world_model_costs = None
        policy._last_score_topk_model_scores = None
        prepared_info = {
            "pixels": torch.zeros(1, 1, 1, 1, 1),
            "goal": torch.zeros(1, 1, 1, 1, 1),
        }
        candidates = torch.arange(8, dtype=torch.float32).view(1, 8, 1).repeat(1, 1, 2)
        model_scores = torch.tensor([[0.1, 9.0, 0.2, 7.0, 0.3, 8.0, 0.4, 0.5]])
        seen_candidates = []

        def fake_score_candidates(prepared, score_candidates):
            seen_candidates.append(score_candidates.detach().clone())
            self.assertEqual(tuple(score_candidates.shape), (1, 3, 2))
            # score top3 original indices are [1, 5, 3]; choose original index 5.
            return torch.tensor([[5.0, 1.0, 3.0]])

        policy.score_candidates_with_world_model = fake_score_candidates

        selected, selected_indices, fallback_mask = policy.score_and_select_candidates(
            prepared_info,
            candidates,
            model_scores,
        )

        self.assertEqual(selected_indices.tolist(), [5])
        self.assertFalse(bool(fallback_mask.item()))
        self.assertTrue(torch.equal(selected, candidates[:, 5, :]))
        self.assertEqual(policy._last_score_topk_indices.tolist(), [[1, 5, 3]])
        self.assertEqual(len(seen_candidates), 1)


if __name__ == "__main__":
    unittest.main()
