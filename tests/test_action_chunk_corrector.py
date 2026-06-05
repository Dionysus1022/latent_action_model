import tempfile
import unittest
from pathlib import Path

import torch

from diffusion.corrector import (
    ActionChunkCorrector,
    ActionChunkCorrectorConfig,
    load_corrector_bundle,
    save_corrector_bundle,
)


class ActionChunkCorrectorTest(unittest.TestCase):
    def test_forward_returns_remaining_action_shape_for_batch_sizes(self):
        config = ActionChunkCorrectorConfig(
            latent_dim=4,
            action_dim=2,
            remain_horizon=3,
            hidden_dim=16,
            num_layers=2,
            dropout=0.0,
            predict_residual=False,
        )
        model = ActionChunkCorrector(config)

        for batch_size in [1, 5]:
            z_real = torch.randn(batch_size, 4)
            z_goal = torch.randn(batch_size, 4)
            error_latent = torch.randn(batch_size, 4)
            u_remain = torch.randn(batch_size, 3, 2)

            output = model(z_real, z_goal, error_latent, u_remain)

            self.assertEqual(tuple(output.shape), (batch_size, 3, 2))

    def test_zero_residual_starts_from_input_remainder(self):
        config = ActionChunkCorrectorConfig(
            latent_dim=4,
            action_dim=2,
            remain_horizon=3,
            hidden_dim=16,
            num_layers=1,
            dropout=0.0,
            predict_residual=True,
            residual_scale=0.5,
        )
        model = ActionChunkCorrector(config)
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        u_remain = torch.randn(2, 3, 2)

        output = model(
            torch.randn(2, 4),
            torch.randn(2, 4),
            torch.randn(2, 4),
            u_remain,
        )

        self.assertTrue(torch.allclose(output, u_remain))

    def test_bundle_round_trip_restores_config_and_weights(self):
        config = ActionChunkCorrectorConfig(
            latent_dim=4,
            action_dim=2,
            remain_horizon=3,
            hidden_dim=16,
            num_layers=2,
            dropout=0.0,
            predict_residual=True,
            residual_scale=1.0,
        )
        model = ActionChunkCorrector(config)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corrector.pt"

            save_corrector_bundle(
                model,
                path,
                metadata={"task": "fake", "correction_interval": 5},
            )
            bundle = load_corrector_bundle(path)
            restored = bundle.instantiate_model()

        restored.load_state_dict(bundle.model_state_dict)
        self.assertEqual(bundle.config.latent_dim, 4)
        self.assertEqual(bundle.metadata["task"], "fake")
        self.assertEqual(bundle.metadata["correction_interval"], 5)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.allclose(value, restored.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
