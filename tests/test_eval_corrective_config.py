import unittest

from omegaconf import OmegaConf

from eval import resolve_corrective_config
from eval import resolve_diffusion_refinement_config
from eval import resolve_diffusion_runtime_execute_steps


class EvalCorrectiveConfigTest(unittest.TestCase):
    def test_missing_corrective_config_defaults_to_no_logging(self):
        cfg = OmegaConf.create({})

        corrective = resolve_corrective_config(cfg)

        self.assertFalse(corrective["enabled"])
        self.assertEqual(corrective["mode"], "none")
        self.assertFalse(corrective["logging"]["log_prediction_error"])

    def test_resolves_prediction_error_logging_config(self):
        cfg = OmegaConf.create(
            {
                "corrective": {
                    "enabled": True,
                    "mode": "none",
                    "corrector_path": "/tmp/unused_corrector.pt",
                    "correction_interval": 5,
                    "execute_horizon": 4,
                    "error_threshold": 3.5,
                    "trigger_stat": "quantile",
                    "trigger_quantile": 0.9,
                    "trigger_scope": "per_env",
                    "error_metric": "mse",
                    "logging": {"log_prediction_error": True},
                }
            }
        )

        corrective = resolve_corrective_config(cfg)

        self.assertTrue(corrective["enabled"])
        self.assertEqual(corrective["mode"], "none")
        self.assertEqual(corrective["corrector_path"], "/tmp/unused_corrector.pt")
        self.assertEqual(corrective["correction_interval"], 5)
        self.assertEqual(corrective["execute_horizon"], 4)
        self.assertAlmostEqual(corrective["error_threshold"], 3.5)
        self.assertEqual(corrective["trigger_stat"], "quantile")
        self.assertAlmostEqual(corrective["trigger_quantile"], 0.9)
        self.assertEqual(corrective["trigger_scope"], "per_env")
        self.assertEqual(corrective["error_metric"], "mse")
        self.assertTrue(corrective["logging"]["log_prediction_error"])

    def test_disabled_corrective_execute_horizon_does_not_override_diffusion_runtime(self):
        cfg = OmegaConf.create(
            {
                "corrective": {
                    "enabled": False,
                    "mode": "none",
                    "execute_horizon": 4,
                }
            }
        )
        corrective = resolve_corrective_config(cfg)

        runtime_execute_steps = resolve_diffusion_runtime_execute_steps(
            None,
            corrective,
        )

        self.assertIsNone(runtime_execute_steps)

    def test_enabled_corrective_execute_horizon_overrides_diffusion_runtime(self):
        cfg = OmegaConf.create(
            {
                "corrective": {
                    "enabled": True,
                    "mode": "none",
                    "execute_horizon": 4,
                }
            }
        )
        corrective = resolve_corrective_config(cfg)

        runtime_execute_steps = resolve_diffusion_runtime_execute_steps(
            8,
            corrective,
        )

        self.assertEqual(runtime_execute_steps, 4)

    def test_learned_mode_resolves_corrector_path(self):
        cfg = OmegaConf.create(
            {
                "corrective": {
                    "enabled": True,
                    "mode": "learned",
                    "corrector_path": "/data/ykz/pusht/corrector_best_bundle.pt",
                }
            }
        )

        corrective = resolve_corrective_config(cfg)

        self.assertTrue(corrective["enabled"])
        self.assertEqual(corrective["mode"], "learned")
        self.assertEqual(
            corrective["corrector_path"],
            "/data/ykz/pusht/corrector_best_bundle.pt",
        )

    def test_enabled_learned_mode_requires_corrector_path(self):
        cfg = OmegaConf.create(
            {
                "corrective": {
                    "enabled": True,
                    "mode": "learned",
                    "corrector_path": None,
                }
            }
        )

        with self.assertRaisesRegex(ValueError, "corrective.corrector_path"):
            resolve_corrective_config(cfg)


class EvalDiffusionRefinementConfigTest(unittest.TestCase):
    def test_missing_refinement_config_defaults_to_disabled(self):
        cfg = OmegaConf.create({})

        refinement = resolve_diffusion_refinement_config(cfg)

        self.assertFalse(refinement["enabled"])
        self.assertEqual(refinement["steps"], 1)
        self.assertIsNone(refinement["topk"])

    def test_resolves_refinement_config(self):
        cfg = OmegaConf.create(
            {
                "diffusion_refinement": {
                    "enabled": True,
                    "steps": 2,
                    "step_size": 0.05,
                    "topk": 8,
                    "goal_weight": 1.0,
                    "prior_weight": 0.05,
                    "smoothness_weight": 0.005,
                    "grad_clip_norm": 0.25,
                }
            }
        )

        refinement = resolve_diffusion_refinement_config(cfg)

        self.assertTrue(refinement["enabled"])
        self.assertEqual(refinement["steps"], 2)
        self.assertEqual(refinement["topk"], 8)
        self.assertAlmostEqual(refinement["step_size"], 0.05)
        self.assertAlmostEqual(refinement["prior_weight"], 0.05)
        self.assertAlmostEqual(refinement["smoothness_weight"], 0.005)
        self.assertAlmostEqual(refinement["grad_clip_norm"], 0.25)


if __name__ == "__main__":
    unittest.main()
