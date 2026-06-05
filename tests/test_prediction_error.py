import math
import unittest

import torch

from diffusion.prediction_error import (
    compute_prediction_error,
    resolve_prediction_error_check,
    summarize_prediction_error_records,
)


class PredictionErrorTest(unittest.TestCase):
    def test_compute_l2_error_for_flat_latents(self):
        z_real = torch.tensor([[3.0, 4.0], [1.0, 1.0]])
        z_pred = torch.tensor([[0.0, 0.0], [1.0, 3.0]])

        error = compute_prediction_error(z_real, z_pred, metric="l2")

        self.assertEqual(tuple(error.shape), (2,))
        self.assertTrue(torch.allclose(error, torch.tensor([5.0, 2.0])))

    def test_compute_error_mean_pools_token_latents(self):
        z_real = torch.tensor(
            [
                [[2.0, 0.0], [4.0, 2.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ]
        )
        z_pred = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[2.0, 1.0], [2.0, 1.0]],
            ]
        )

        error = compute_prediction_error(z_real, z_pred, metric="l2")

        self.assertTrue(torch.allclose(error, torch.tensor([math.sqrt(10.0), 1.0])))

    def test_compute_mse_and_cosine_errors(self):
        z_real = torch.tensor([[2.0, 0.0], [1.0, 0.0]])
        z_pred = torch.tensor([[0.0, 0.0], [0.0, 1.0]])

        mse = compute_prediction_error(z_real, z_pred, metric="mse")
        cosine = compute_prediction_error(z_real, z_pred, metric="cosine")

        self.assertTrue(torch.allclose(mse, torch.tensor([2.0, 1.0])))
        self.assertTrue(torch.allclose(cosine, torch.tensor([1.0, 1.0])))

    def test_compute_error_detaches_from_inputs(self):
        z_real = torch.tensor([[1.0, 2.0]], requires_grad=True)
        z_pred = torch.tensor([[0.0, 0.0]], requires_grad=True)

        error = compute_prediction_error(z_real, z_pred, metric="l2")

        self.assertFalse(error.requires_grad)

    def test_summarize_prediction_error_records_splits_success_and_failure(self):
        records = [
            {"env_index": 0, "step": 5, "error": 0.2},
            {"env_index": 0, "step": 10, "error": 0.4},
            {"env_index": 1, "step": 5, "error": 1.5},
            {"env_index": 1, "step": 10, "error": 2.5},
            {"env_index": 2, "step": 5, "error": 0.6},
        ]
        successes = [True, False, True]

        summary = summarize_prediction_error_records(records, successes)

        self.assertEqual(summary["prediction_error_count"], 5)
        self.assertAlmostEqual(summary["prediction_error_mean"], 1.04)
        self.assertAlmostEqual(summary["prediction_error_max"], 2.5)
        self.assertEqual(summary["successful_episode_count"], 2)
        self.assertEqual(summary["failed_episode_count"], 1)
        self.assertAlmostEqual(summary["successful_prediction_error_mean"], 0.45)
        self.assertAlmostEqual(summary["failed_prediction_error_mean"], 2.0)
        self.assertAlmostEqual(summary["prediction_error_fail_minus_success"], 1.55)
        self.assertAlmostEqual(summary["prediction_error_fail_success_ratio"], 2.0 / 0.45)
        self.assertEqual(summary["prediction_error_failed_higher"], 1.0)
        self.assertEqual(summary["prediction_error_episode_mean_count"], 3)

    def test_resolve_prediction_error_check_uses_action_block_boundaries(self):
        self.assertIsNone(
            resolve_prediction_error_check(
                prefix_steps=2,
                action_block=5,
                correction_interval=2,
            )
        )
        self.assertEqual(
            resolve_prediction_error_check(
                prefix_steps=5,
                action_block=5,
                correction_interval=2,
            ),
            1,
        )
        self.assertEqual(
            resolve_prediction_error_check(
                prefix_steps=10,
                action_block=5,
                correction_interval=7,
            ),
            2,
        )

    def test_summarize_prediction_error_records_handles_missing_failure_group(self):
        records = [
            {"env_index": 0, "step": 5, "error": 0.2},
            {"env_index": 1, "step": 5, "error": 0.3},
        ]
        successes = [True, True]

        summary = summarize_prediction_error_records(records, successes)

        self.assertEqual(summary["successful_episode_count"], 2)
        self.assertEqual(summary["failed_episode_count"], 0)
        self.assertTrue(math.isnan(summary["failed_prediction_error_mean"]))
        self.assertTrue(math.isnan(summary["prediction_error_fail_minus_success"]))


if __name__ == "__main__":
    unittest.main()
