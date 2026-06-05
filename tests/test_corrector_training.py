import types
import unittest

import torch

from diffusion.corrector import ActionChunkCorrector, ActionChunkCorrectorConfig
from diffusion.corrector_training import (
    CorrectorTrainingDataset,
    CorrectorTrainingSpec,
    compute_corrector_loss,
    limit_dataset_samples,
)


class FakeWorldModel(torch.nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.action_encoder = torch.nn.Identity()
        self.latent_dim = latent_dim

    def predict(self, emb, act_emb):
        action = act_emb[:, -1:, :]
        if int(action.shape[-1]) > self.latent_dim:
            action = action[:, :, : self.latent_dim]
        if int(action.shape[-1]) < self.latent_dim:
            pad = torch.zeros(
                int(action.shape[0]),
                1,
                self.latent_dim - int(action.shape[-1]),
                device=action.device,
                dtype=action.dtype,
            )
            action = torch.cat([action, pad], dim=-1)
        return emb[:, -1:, :] + action


class StrictBlockedWorldModel(FakeWorldModel):
    def __init__(self, latent_dim: int, expected_action_width: int):
        super().__init__(latent_dim=latent_dim)
        self.expected_action_width = expected_action_width

    def predict(self, emb, act_emb):
        if int(act_emb.shape[-1]) != self.expected_action_width:
            raise ValueError(
                f"expected action width {self.expected_action_width}, got {int(act_emb.shape[-1])}"
            )
        return emb[:, -1:, :] + act_emb[:, -1:, :]


def make_tiny_bundle():
    return {
        "z_cur": torch.zeros(4, 2),
        "z_goal": torch.ones(4, 2),
        "teacher_plan": torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0, 2.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 2.0, 1.0],
                [0.0, 2.0, 1.0, 2.0, 2.0, 2.0],
                [0.0, 3.0, 1.0, 3.0, 2.0, 3.0],
            ]
        ),
        "meta": [
            {
                "plan_horizon": 3,
                "action_dim": 2,
                "action_chunk_dim": 6,
                "episode_id": 0,
                "step": index,
            }
            for index in range(4)
        ],
        "build_info": {"plan_config": {"receding_horizon": 3, "action_block": 1}},
    }


def make_blocked_bundle():
    return {
        "z_cur": torch.zeros(3, 4),
        "z_goal": torch.ones(3, 4),
        "teacher_plan": torch.tensor(
            [
                [0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 1.5, 0.5],
                [0.0, 1.0, 0.5, 1.5, 1.0, 1.0, 1.5, 1.5],
                [0.0, 2.0, 0.5, 2.5, 1.0, 2.0, 1.5, 2.5],
            ]
        ),
        "meta": [
            {
                "plan_horizon": 4,
                "action_dim": 2,
                "action_chunk_dim": 8,
            }
            for _ in range(3)
        ],
        "build_info": {"plan_config": {"receding_horizon": 2, "action_block": 2}},
    }


class CorrectorTrainingTest(unittest.TestCase):
    def test_dataset_builds_noisy_drift_sample_without_identity_target(self):
        spec = CorrectorTrainingSpec(
            correction_interval=1,
            action_block=1,
            noise_std=0.25,
            noise_prob=1.0,
            seed=7,
        )
        dataset = CorrectorTrainingDataset(make_tiny_bundle(), spec)

        sample = dataset[0]

        self.assertEqual(tuple(sample["z_cur"].shape), (2,))
        self.assertEqual(tuple(sample["z_goal"].shape), (2,))
        self.assertEqual(tuple(sample["clean_prefix"].shape), (1, 2))
        self.assertEqual(tuple(sample["noisy_prefix"].shape), (1, 2))
        self.assertEqual(tuple(sample["u_remain"].shape), (2, 2))
        self.assertEqual(tuple(sample["target_remain"].shape), (2, 2))
        self.assertFalse(torch.allclose(sample["noisy_prefix"], sample["clean_prefix"]))
        self.assertFalse(torch.allclose(sample["u_remain"], sample["target_remain"]))

    def test_compute_corrector_loss_returns_finite_terms_and_gradients(self):
        spec = CorrectorTrainingSpec(
            correction_interval=1,
            action_block=1,
            noise_std=0.1,
            noise_prob=1.0,
            lambda_action=1.0,
            lambda_goal=0.25,
            lambda_smooth=0.1,
            seed=3,
        )
        dataset = CorrectorTrainingDataset(make_tiny_bundle(), spec)
        batch = {
            key: torch.stack([dataset[0][key], dataset[1][key]], dim=0)
            for key in [
                "z_cur",
                "z_goal",
                "clean_prefix",
                "noisy_prefix",
                "u_remain",
                "target_remain",
            ]
        }
        model = ActionChunkCorrector(
            ActionChunkCorrectorConfig(
                latent_dim=2,
                action_dim=2,
                remain_horizon=2,
                hidden_dim=16,
                num_layers=2,
                dropout=0.0,
            )
        )
        world_model = FakeWorldModel(latent_dim=2)
        loss, terms = compute_corrector_loss(
            model,
            world_model,
            batch,
            spec,
            history_size=1,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(terms["action_loss"]), 0.0)
        self.assertGreaterEqual(float(terms["goal_loss"]), 0.0)
        self.assertGreaterEqual(float(terms["smoothness_loss"]), 0.0)
        loss.backward()
        grad_norm = sum(
            float(parameter.grad.abs().sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in world_model.parameters())
        )

    def test_dataset_infers_spec_dimensions_from_bundle(self):
        spec = CorrectorTrainingSpec(
            correction_interval=1,
            action_block=None,
            noise_std=0.1,
            seed=1,
        )

        dataset = CorrectorTrainingDataset(make_tiny_bundle(), spec)

        self.assertEqual(dataset.plan_horizon, 3)
        self.assertEqual(dataset.action_dim, 2)
        self.assertEqual(dataset.action_block, 1)
        self.assertEqual(dataset.remain_horizon, 2)

    def test_limit_dataset_samples_is_deterministic(self):
        dataset = CorrectorTrainingDataset(
            make_tiny_bundle(),
            CorrectorTrainingSpec(correction_interval=1, action_block=1),
        )

        unlimited = limit_dataset_samples(dataset, max_samples=None, seed=42)
        limited_a = limit_dataset_samples(dataset, max_samples=2, seed=42)
        limited_b = limit_dataset_samples(dataset, max_samples=2, seed=42)

        self.assertIs(unlimited, dataset)
        self.assertEqual(len(limited_a), 2)
        self.assertEqual(limited_a.indices, limited_b.indices)

    def test_loss_uses_dataset_inferred_action_block(self):
        spec = CorrectorTrainingSpec(
            correction_interval=2,
            action_block=None,
            noise_std=0.1,
            lambda_action=1.0,
            lambda_goal=0.25,
            seed=5,
        )
        dataset = CorrectorTrainingDataset(make_blocked_bundle(), spec)
        batch = {
            key: torch.stack([dataset[0][key], dataset[1][key]], dim=0)
            for key in [
                "z_cur",
                "z_goal",
                "clean_prefix",
                "noisy_prefix",
                "u_remain",
                "target_remain",
            ]
        }
        batch["action_block"] = torch.tensor([dataset.action_block, dataset.action_block])
        model = ActionChunkCorrector(
            ActionChunkCorrectorConfig(
                latent_dim=4,
                action_dim=2,
                remain_horizon=2,
                hidden_dim=16,
                num_layers=2,
                dropout=0.0,
            )
        )
        world_model = StrictBlockedWorldModel(latent_dim=4, expected_action_width=4)

        loss, _ = compute_corrector_loss(
            model,
            world_model,
            batch,
            spec,
            history_size=1,
        )

        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
