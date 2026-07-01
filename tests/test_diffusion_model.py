import tempfile
import unittest
from pathlib import Path

import torch

from diffusion.model import (
    DiffusionPlannerModel,
    DiffusionPlannerModelConfig,
    load_diffusion_planner_bundle,
    save_diffusion_planner_bundle,
)


class DiffusionModelBundleCompatibilityTest(unittest.TestCase):
    def test_loads_legacy_mlp_denoiser_metadata(self):
        config = DiffusionPlannerModelConfig(
            latent_dim=2,
            plan_horizon=2,
            action_dim=1,
            num_anchors=2,
            hidden_dim=8,
            num_layers=1,
            fusion_num_layers=1,
        )
        anchors = torch.zeros(2, 2)
        model = DiffusionPlannerModel(config, anchors)

        with tempfile.TemporaryDirectory() as tmp:
            path = save_diffusion_planner_bundle(model, Path(tmp) / "bundle.pt")
            bundle_dict = torch.load(path, map_location="cpu")
            bundle_dict["model_hyperparameters"]["denoiser_type"] = "mlp"
            bundle_dict["model_hyperparameters"]["dit_num_layers"] = 4
            bundle_dict["model_hyperparameters"]["dit_num_heads"] = 4
            bundle_dict["model_hyperparameters"]["dit_mlp_ratio"] = 4.0
            torch.save(bundle_dict, path)

            bundle = load_diffusion_planner_bundle(path)
            restored = bundle.instantiate_model()

        self.assertIsInstance(restored, DiffusionPlannerModel)
        self.assertFalse(hasattr(restored.config, "denoiser_type"))

    def test_rejects_removed_dit_denoiser_metadata(self):
        config = DiffusionPlannerModelConfig(
            latent_dim=2,
            plan_horizon=2,
            action_dim=1,
            num_anchors=2,
            hidden_dim=8,
            num_layers=1,
            fusion_num_layers=1,
        )
        anchors = torch.zeros(2, 2)
        model = DiffusionPlannerModel(config, anchors)

        with tempfile.TemporaryDirectory() as tmp:
            path = save_diffusion_planner_bundle(model, Path(tmp) / "bundle.pt")
            bundle_dict = torch.load(path, map_location="cpu")
            bundle_dict["model_hyperparameters"]["denoiser_type"] = "dit"
            torch.save(bundle_dict, path)

            with self.assertRaisesRegex(ValueError, "only supports the mainline MLP diffusion denoiser"):
                load_diffusion_planner_bundle(path)


if __name__ == "__main__":
    unittest.main()
