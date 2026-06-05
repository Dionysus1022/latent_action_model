import types
import unittest
from collections import deque

import numpy as np
import torch

from diffusion.policy import DiffusionPlannerPolicy, DiffusionRuntimeSpec
from diffusion.prediction_error import compute_trigger_error


class FakeWorldModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.action_encoder = torch.nn.Identity()

    def encode(self, info):
        pixels = info["pixels"]
        return {"emb": pixels[:, :, :, 0, 0].float()}

    def predict(self, emb, act_emb):
        return emb[:, -1:, :] + act_emb[:, -1:, :]


def make_policy_for_prediction_error_test():
    policy = DiffusionPlannerPolicy.__new__(DiffusionPlannerPolicy)
    policy.world_model = FakeWorldModel()
    policy.planner = types.SimpleNamespace(
        latent_dim=2,
        plan_horizon=2,
        action_dim=2,
        num_anchors=1,
    )
    policy.runtime_spec = DiffusionRuntimeSpec(
        task="fake",
        block_horizon=2,
        receding_horizon=2,
        action_block=1,
        goal_offset_steps=2,
        eval_budget=2,
    )
    policy.corrective_enabled = True
    policy.corrective_mode = "none"
    policy.corrective_log_prediction_error = True
    policy.corrective_error_metric = "l2"
    policy.corrective_correction_interval = 1
    policy.corrective_error_threshold = 1.0
    policy.corrective_trigger_stat = "max"
    policy.corrective_trigger_quantile = 0.9
    policy.corrective_trigger_scope = "per_env"
    policy.corrector = None
    policy.requested_runtime_execute_steps = None
    policy.selection_mode = "wm_only"
    policy._corrective_correction_count = 0
    policy._corrective_correction_norms = []
    policy._corrective_action_delta_norms = []
    policy._corrective_correction_time_total_sec = 0.0
    policy._corrective_replan_count = 0
    policy._corrective_check_count = 0
    policy._corrective_replan_error_records = []
    policy._prediction_error_records = []
    policy._logged_prediction_error_steps = set()
    policy._checked_prediction_error_steps = set()
    policy._num_replans = 0
    policy._current_plan = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )
    policy._current_plan_start_latent = torch.zeros(2, 2)
    policy._current_plan_index = 1
    policy._current_plan_executed_steps = 1
    policy._action_buffer = deque(
        [
            torch.tensor([[9.0, 9.0], [9.0, 9.0]]),
            torch.tensor([[8.0, 8.0], [8.0, 8.0]]),
        ]
    )
    policy._env_action_shape = (2, 2)
    policy._action_low = None
    policy._action_high = None
    policy._action_dtype = None
    policy.process = {}
    policy.env = types.SimpleNamespace(
        action_space=types.SimpleNamespace(shape=(2, 2)),
    )
    return policy


class DiffusionPolicyPredictionErrorTest(unittest.TestCase):
    def test_compute_trigger_error_supports_max_mean_and_quantile(self):
        errors = torch.tensor([1.0, 1.0, 1.0, 20.0])

        self.assertAlmostEqual(compute_trigger_error(errors, stat="max"), 20.0)
        self.assertAlmostEqual(compute_trigger_error(errors, stat="mean"), 5.75)
        self.assertAlmostEqual(
            compute_trigger_error(errors, stat="quantile", quantile=0.75),
            1.0,
        )

    def test_quantile_trigger_ignores_single_outlier(self):
        errors = torch.tensor([1.0] * 49 + [20.0])

        trigger_error = compute_trigger_error(
            errors,
            stat="quantile",
            quantile=0.9,
        )

        self.assertAlmostEqual(trigger_error, 1.0)

    def test_policy_records_prediction_error_at_correction_boundary(self):
        policy = make_policy_for_prediction_error_test()
        prepared_info = {
            "pixels": torch.tensor(
                [
                    [[[[1.25]], [[0.0]]]],
                    [[[[3.0]], [[0.0]]]],
                ]
            ),
            "goal": torch.zeros(2, 1, 2, 1, 1),
        }

        policy._maybe_log_prediction_error(prepared_info)

        self.assertEqual(len(policy._prediction_error_records), 2)
        self.assertEqual(policy._prediction_error_records[0]["env_index"], 0)
        self.assertEqual(policy._prediction_error_records[0]["step"], 1)
        self.assertAlmostEqual(policy._prediction_error_records[0]["error"], 0.25)
        self.assertAlmostEqual(policy._prediction_error_records[1]["error"], 2.0)

    def test_policy_does_not_record_when_corrective_disabled(self):
        policy = make_policy_for_prediction_error_test()
        policy.corrective_enabled = False
        prepared_info = {
            "pixels": torch.tensor(
                [
                    [[[[1.25]], [[0.0]]]],
                    [[[[3.0]], [[0.0]]]],
                ]
            ),
            "goal": torch.zeros(2, 1, 2, 1, 1),
        }

        policy._maybe_log_prediction_error(prepared_info)

        self.assertEqual(policy._prediction_error_records, [])

    def test_policy_does_not_record_when_prediction_error_logging_disabled(self):
        policy = make_policy_for_prediction_error_test()
        policy.corrective_log_prediction_error = False
        prepared_info = {
            "pixels": torch.tensor(
                [
                    [[[[1.25]], [[0.0]]]],
                    [[[[3.0]], [[0.0]]]],
                ]
            ),
            "goal": torch.zeros(2, 1, 2, 1, 1),
        }

        policy._maybe_log_prediction_error(prepared_info)

        self.assertEqual(policy._prediction_error_records, [])

    def test_policy_prediction_error_summary_splits_success_and_failure(self):
        policy = make_policy_for_prediction_error_test()
        policy._prediction_error_records = [
            {"env_index": 0, "step": 1, "error": 0.25},
            {"env_index": 1, "step": 1, "error": 2.0},
        ]

        summary = policy.get_prediction_error_summary([True, False])

        self.assertAlmostEqual(summary["successful_prediction_error_mean"], 0.25)
        self.assertAlmostEqual(summary["failed_prediction_error_mean"], 2.0)
        self.assertAlmostEqual(summary["prediction_error_fail_minus_success"], 1.75)

    def test_policy_replans_only_triggered_env_when_error_exceeds_threshold(self):
        policy = make_policy_for_prediction_error_test()
        policy.corrective_mode = "replan"
        policy.corrective_error_threshold = 1.0
        replanned = []

        def fake_plan_actions(prepared_info):
            replanned.append(prepared_info)
            policy._last_plan_start_latent = torch.zeros(2, 2)
            return torch.tensor(
                [
                    [[0.1, 0.2], [0.3, 0.4]],
                    [[0.5, 0.6], [0.7, 0.8]],
                ]
            )

        policy.plan_actions = fake_plan_actions
        prepared_info = {
            "pixels": torch.tensor(
                [
                    [[[[1.25]], [[0.0]]]],
                    [[[[3.0]], [[0.0]]]],
                ]
            ),
            "goal": torch.zeros(2, 1, 2, 1, 1),
        }

        should_replan = policy._maybe_trigger_corrective_replan(prepared_info)

        self.assertTrue(should_replan)
        self.assertEqual(len(replanned), 1)
        self.assertEqual(len(policy._action_buffer), 2)
        buffer_steps = list(policy._action_buffer)
        self.assertTrue(torch.allclose(buffer_steps[0][0], torch.tensor([9.0, 9.0])))
        self.assertTrue(torch.allclose(buffer_steps[1][0], torch.tensor([8.0, 8.0])))
        self.assertTrue(torch.allclose(buffer_steps[0][1], torch.tensor([0.5, 0.6])))
        self.assertTrue(torch.allclose(buffer_steps[1][1], torch.tensor([0.7, 0.8])))
        self.assertIsNotNone(policy._current_plan)
        self.assertIsNotNone(policy._current_plan_start_latent)
        self.assertEqual(policy._current_plan_executed_steps, 1)
        self.assertEqual(policy._corrective_check_count, 1)
        self.assertEqual(policy._corrective_replan_count, 1)
        self.assertEqual(len(policy._corrective_replan_error_records), 1)
        self.assertAlmostEqual(
            policy._corrective_replan_error_records[0]["max_error"],
            2.0,
        )
        self.assertEqual(policy._corrective_replan_error_records[0]["env_indices"], [1])

    def test_policy_logs_and_replans_from_same_checkpoint_once(self):
        policy = make_policy_for_prediction_error_test()
        policy.corrective_mode = "replan"
        policy.corrective_error_threshold = 1.0
        policy.corrective_log_prediction_error = True

        def fake_plan_actions(prepared_info):
            policy._last_plan_start_latent = torch.zeros(2, 2)
            return torch.tensor(
                [
                    [[0.1, 0.2], [0.3, 0.4]],
                    [[0.5, 0.6], [0.7, 0.8]],
                ]
            )

        policy.plan_actions = fake_plan_actions
        prepared_info = {
            "pixels": torch.tensor(
                [
                    [[[[1.25]], [[0.0]]]],
                    [[[[3.0]], [[0.0]]]],
                ]
            ),
            "goal": torch.zeros(2, 1, 2, 1, 1),
        }

        policy._maybe_handle_corrective_checkpoint(prepared_info)
        policy._maybe_handle_corrective_checkpoint(prepared_info)

        self.assertEqual(len(policy._prediction_error_records), 2)
        self.assertEqual(policy._corrective_check_count, 1)
        self.assertEqual(policy._corrective_replan_count, 1)

    def test_policy_keeps_remaining_chunk_when_error_is_below_threshold(self):
        policy = make_policy_for_prediction_error_test()
        policy.corrective_mode = "replan"
        policy.corrective_error_threshold = 3.0
        prepared_info = {
            "pixels": torch.tensor(
                [
                    [[[[1.25]], [[0.0]]]],
                    [[[[3.0]], [[0.0]]]],
                ]
            ),
            "goal": torch.zeros(2, 1, 2, 1, 1),
        }

        should_replan = policy._maybe_trigger_corrective_replan(prepared_info)

        self.assertFalse(should_replan)
        self.assertEqual(len(policy._action_buffer), 2)
        self.assertIsNotNone(policy._current_plan)
        self.assertEqual(policy._current_plan_executed_steps, 1)
        self.assertEqual(policy._corrective_check_count, 1)
        self.assertEqual(policy._corrective_replan_count, 0)

    def test_policy_batch_quantile_trigger_does_not_replan_for_single_outlier(self):
        policy = make_policy_for_prediction_error_test()
        policy.corrective_mode = "replan"
        policy.corrective_error_threshold = 1.5
        policy.corrective_trigger_stat = "quantile"
        policy.corrective_trigger_quantile = 0.75
        policy.corrective_trigger_scope = "batch"
        policy._current_plan = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0]],
            ]
        )
        policy._current_plan_start_latent = torch.zeros(4, 2)
        policy._action_buffer = deque(
            [
                torch.tensor(
                    [
                        [9.0, 9.0],
                        [9.0, 9.0],
                        [9.0, 9.0],
                        [9.0, 9.0],
                    ]
                )
            ]
        )
        prepared_info = {
            "pixels": torch.tensor(
                [
                    [[[[2.0]], [[0.0]]]],
                    [[[[2.0]], [[0.0]]]],
                    [[[[2.0]], [[0.0]]]],
                    [[[[21.0]], [[0.0]]]],
                ]
            ),
            "goal": torch.zeros(4, 1, 2, 1, 1),
        }

        should_replan = policy._maybe_trigger_corrective_replan(prepared_info)

        self.assertFalse(should_replan)
        self.assertEqual(policy._corrective_check_count, 1)
        self.assertEqual(policy._corrective_replan_count, 0)
        self.assertEqual(len(policy._action_buffer), 1)

    def test_get_action_replans_only_triggered_env_from_current_observation(self):
        policy = make_policy_for_prediction_error_test()
        policy.corrective_mode = "replan"
        policy.corrective_error_threshold = 1.0
        policy.runtime_spec = DiffusionRuntimeSpec(
            task="fake",
            block_horizon=2,
            receding_horizon=2,
            action_block=1,
            goal_offset_steps=2,
            eval_budget=2,
        )
        planned_actions = []

        def fake_plan_actions(prepared_info):
            planned_actions.append(prepared_info)
            policy._last_plan_start_latent = torch.zeros(2, 2)
            return torch.tensor(
                [
                    [[0.1, 0.2], [0.3, 0.4]],
                    [[0.5, 0.6], [0.7, 0.8]],
                ]
            )

        policy._prepare_info = lambda info: info
        policy.plan_actions = fake_plan_actions
        info = {
            "pixels": torch.tensor(
                [
                    [[[[1.25]], [[0.0]]]],
                    [[[[3.0]], [[0.0]]]],
                ]
            ),
            "goal": torch.zeros(2, 1, 2, 1, 1),
        }

        action = policy.get_action(info)

        self.assertEqual(len(planned_actions), 1)
        self.assertEqual(policy._corrective_replan_count, 1)
        self.assertTrue(
            np.allclose(
                action,
                np.asarray([[9.0, 9.0], [0.5, 0.6]], dtype=np.float32),
            )
        )
        self.assertEqual(policy._current_plan_index, 1)

    def test_learned_mode_requires_loaded_corrector_when_triggered(self):
        policy = make_policy_for_prediction_error_test()
        policy.corrective_mode = "learned"
        policy.corrective_error_threshold = 1.0
        prepared_info = {
            "pixels": torch.tensor(
                [
                    [[[[1.25]], [[0.0]]]],
                    [[[[3.0]], [[0.0]]]],
                ]
            ),
            "goal": torch.zeros(2, 1, 2, 1, 1),
        }

        with self.assertRaisesRegex(ValueError, "corrective.mode=learned requires a loaded corrector"):
            policy._maybe_handle_corrective_checkpoint(prepared_info)

    def test_learned_mode_corrects_only_triggered_env_remainder(self):
        class AddOneCorrector(torch.nn.Module):
            def forward(self, z_real, z_goal, error_latent, u_remain):
                return u_remain + 1.0

        policy = make_policy_for_prediction_error_test()
        policy.corrective_mode = "learned"
        policy.corrective_error_threshold = 1.0
        policy.corrector = AddOneCorrector()
        prepared_info = {
            "pixels": torch.tensor(
                [
                    [[[[1.25]], [[0.0]]]],
                    [[[[3.0]], [[0.0]]]],
                ]
            ),
            "goal": torch.zeros(2, 1, 2, 1, 1),
        }

        policy._maybe_handle_corrective_checkpoint(prepared_info)

        self.assertEqual(policy._corrective_check_count, 1)
        self.assertEqual(policy._corrective_correction_count, 1)
        self.assertEqual(policy._corrective_replan_count, 0)
        buffer_steps = list(policy._action_buffer)
        self.assertTrue(torch.allclose(buffer_steps[0][0], torch.tensor([9.0, 9.0])))
        self.assertTrue(torch.allclose(buffer_steps[1][0], torch.tensor([8.0, 8.0])))
        self.assertTrue(torch.allclose(buffer_steps[0][1], torch.tensor([10.0, 10.0])))
        self.assertTrue(torch.allclose(buffer_steps[1][1], torch.tensor([9.0, 9.0])))
        self.assertTrue(torch.allclose(policy._current_plan[0], torch.tensor([[1.0, 0.0], [1.0, 0.0]])))
        self.assertTrue(torch.allclose(policy._current_plan[1], torch.tensor([[1.0, 0.0], [10.0, 10.0]])))
        self.assertEqual(len(policy._corrective_action_delta_norms), 1)
        self.assertGreater(policy._corrective_action_delta_norms[0], 0.0)

    def test_learned_mode_rejects_corrector_with_wrong_remainder_horizon(self):
        policy = make_policy_for_prediction_error_test()
        policy.corrective_mode = "learned"
        policy.corrective_correction_interval = 1
        policy.corrector = types.SimpleNamespace(remain_horizon=3)

        with self.assertRaisesRegex(ValueError, "Corrector remain_horizon"):
            policy._validate_learned_corrector_contract()


if __name__ == "__main__":
    unittest.main()
