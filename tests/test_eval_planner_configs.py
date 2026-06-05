import unittest
from pathlib import Path


class EvalPlannerConfigTests(unittest.TestCase):
    def test_legacy_config_names_are_normalized_to_profiles(self) -> None:
        from eval import normalize_eval_cli_args

        cases = {
            "cube_mpc": ["cube", "mpc"],
            "pusht_diffusion": ["pusht", "diffusion"],
            "reacher_consistency": ["reacher", "consistency"],
            "tworoom_diffusion": ["tworoom", "diffusion"],
            "pusht_diffusion_corrective": ["pusht", "corrective_learned"],
        }

        for legacy_name, (task_name, profile_name) in cases.items():
            with self.subTest(legacy_name=legacy_name):
                self.assertEqual(
                    normalize_eval_cli_args(["eval.py", "--config-name", legacy_name]),
                    ["eval.py", "--config-name", task_name, f"eval_profile={profile_name}"],
                )
                self.assertEqual(
                    normalize_eval_cli_args(["eval.py", f"--config-name={legacy_name}"]),
                    ["eval.py", f"--config-name={task_name}", f"eval_profile={profile_name}"],
                )

    def test_legacy_yaml_files_are_archived_outside_eval_root(self) -> None:
        config_dir = Path(__file__).resolve().parents[1] / "config" / "eval"
        legacy_files = {
            "cube_mpc.yaml",
            "cube_diffusion.yaml",
            "cube_consistency.yaml",
            "pusht_mpc.yaml",
            "pusht_diffusion.yaml",
            "pusht_consistency.yaml",
            "pusht_diffusion_corrective.yaml",
            "reacher_mpc.yaml",
            "reacher_diffusion.yaml",
            "reacher_consistency.yaml",
            "tworoom_mpc.yaml",
            "tworoom_diffusion.yaml",
            "tworoom_consistency.yaml",
        }

        for filename in legacy_files:
            with self.subTest(filename=filename):
                self.assertFalse((config_dir / filename).exists())
                self.assertTrue((config_dir / "legacy" / filename).exists())

    def test_archived_legacy_yaml_files_still_compose_at_global_package(self) -> None:
        import hydra

        config_dir = str(Path(__file__).resolve().parents[1] / "config" / "eval")
        with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = hydra.compose(config_name="legacy/pusht_diffusion")

        self.assertNotIn("legacy", cfg)
        self.assertEqual(cfg.task, "pusht")
        self.assertEqual(cfg.eval_profile, "diffusion")

    def test_legacy_task_flag_maps_to_mpc_config(self) -> None:
        from eval import normalize_eval_cli_args

        normalized = normalize_eval_cli_args(["eval.py", "--task", "tworoom"])

        self.assertEqual(
            normalized,
            ["eval.py", "--config-name", "tworoom", "eval_profile=mpc"],
        )

    def test_legacy_task_flag_preserves_explicit_eval_profile(self) -> None:
        from eval import normalize_eval_cli_args

        normalized = normalize_eval_cli_args(
            ["eval.py", "--task", "pusht", "eval_profile=diffusion", "eval.num_eval=10"]
        )

        self.assertEqual(
            normalized,
            [
                "eval.py",
                "--config-name",
                "pusht",
                "eval_profile=diffusion",
                "eval.num_eval=10",
            ],
        )

    def test_task_base_configs_do_not_select_a_planner(self) -> None:
        import hydra

        config_dir = str(Path(__file__).resolve().parents[1] / "config" / "eval")
        with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
            for task in ["cube", "pusht", "reacher", "tworoom"]:
                cfg = hydra.compose(config_name=task)
                self.assertNotIn("planner_type", cfg)
                self.assertNotIn("policy", cfg)
                self.assertNotIn("solver", cfg)
                self.assertNotIn("diffusion_bundle", cfg)

    def test_mpc_configs_are_concrete_lewm_cem_configs(self) -> None:
        import hydra
        from eval import resolve_eval_profile_config

        config_dir = str(Path(__file__).resolve().parents[1] / "config" / "eval")
        expected_policy = {
            "cube": "/data/ykz/cube/lewm_epoch_27",
            "pusht": "/data/ykz/pusht/lewm_epoch_100",
            "reacher": "/data/ykz/reacher/lewm_epoch_29",
            "tworoom": "/data/ykz/tworoom/lewm_epoch_67",
        }

        with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
            for config_name, policy_path in expected_policy.items():
                cfg = resolve_eval_profile_config(
                    hydra.compose(config_name=config_name, overrides=["eval_profile=mpc"])
                )
                self.assertEqual(cfg.planner_type, "mpc")
                self.assertEqual(cfg.policy, policy_path)
                self.assertEqual(cfg.solver._target_, "stable_worldmodel.solver.CEMSolver")
                self.assertEqual(cfg.solver.num_samples, 300)
                self.assertEqual(cfg.solver.n_steps, 30)
                self.assertEqual(cfg.solver.topk, 30)
                self.assertEqual(cfg.plan_config.horizon, 5)
                self.assertEqual(cfg.plan_config.receding_horizon, 5)
                self.assertEqual(cfg.plan_config.action_block, 5)
                self.assertNotIn("diffusion_bundle", cfg)

    def test_diffusion_configs_are_concrete_bundle_configs_without_solver(self) -> None:
        import hydra
        from eval import resolve_eval_profile_config

        config_dir = str(Path(__file__).resolve().parents[1] / "config" / "eval")
        expected = {
            "cube": (
                "/data/ykz/cube/lewm_epoch_27",
                "/data/ykz/cube/diffusion_pipeline/cube_diffusion_k128_200000/diffusion_planner_best_bundle.pt",
            ),
            "pusht": (
                "/data/ykz/pusht/lewm_epoch_100",
                "/data/ykz/pusht/diffusion_pipeline/pusht_diffusion_k128_200000/diffusion_planner_best_bundle.pt",
            ),
            "reacher": (
                "/data/ykz/reacher/lewm_epoch_29",
                "/data/ykz/reacher/diffusion_pipeline/reacher_diffusion_k128_200000/diffusion_planner_best_bundle.pt",
            ),
            "tworoom": (
                "/data/ykz/tworoom/lewm_epoch_67",
                "/data/ykz/tworoom/diffusion_pipeline/tworoom_diffusion_k128_200000/diffusion_planner_best_bundle.pt",
            ),
        }

        with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
            for config_name, (policy_path, bundle_path) in expected.items():
                cfg = resolve_eval_profile_config(
                    hydra.compose(config_name=config_name, overrides=["eval_profile=diffusion"])
                )
                self.assertEqual(cfg.planner_type, "diffusion")
                self.assertEqual(cfg.policy, policy_path)
                self.assertEqual(cfg.diffusion_bundle, bundle_path)
                self.assertEqual(cfg.diffusion_selection_mode, "wm_only")
                self.assertEqual(cfg.diffusion_num_candidates, 128)
                self.assertIsNone(cfg.diffusion_truncation_steps)
                self.assertEqual(
                    bool(cfg.diffusion_refinement.enabled),
                    config_name == "pusht",
                )
                self.assertEqual(cfg.diffusion_refinement.topk, 16)
                self.assertEqual(cfg.diffusion_refinement.steps, 1)
                self.assertAlmostEqual(cfg.diffusion_refinement.step_size, 0.03)
                self.assertAlmostEqual(cfg.diffusion_refinement.grad_clip_norm, 1.0)
                self.assertNotIn("solver", cfg)

    def test_pusht_corrective_diffusion_config_sets_phase3_defaults(self) -> None:
        import hydra
        from eval import resolve_eval_profile_config

        config_dir = str(Path(__file__).resolve().parents[1] / "config" / "eval")
        with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = resolve_eval_profile_config(
                hydra.compose(
                    config_name="pusht",
                    overrides=["eval_profile=corrective_learned"],
                )
            )

        self.assertEqual(cfg.planner_type, "diffusion")
        self.assertEqual(cfg.policy, "/data/ykz/pusht/lewm_epoch_100")
        self.assertEqual(
            cfg.diffusion_bundle,
            "/data/ykz/pusht/diffusion_pipeline/pusht_diffusion_k128_200000/diffusion_planner_best_bundle.pt",
        )
        self.assertTrue(cfg.corrective.enabled)
        self.assertEqual(cfg.corrective.mode, "learned")
        self.assertEqual(
            cfg.corrective.corrector_path,
            "/data/ykz/pusht/diffusion_pipeline/pusht_corrector_ci5/corrector_best_bundle.pt",
        )
        self.assertTrue(cfg.corrective.logging.log_prediction_error)
        self.assertEqual(cfg.corrective.correction_interval, 5)
        self.assertEqual(cfg.corrective.execute_horizon, 25)
        self.assertEqual(cfg.corrective.trigger_scope, "per_env")
        self.assertAlmostEqual(cfg.corrective.error_threshold, 5.0)

    def test_pusht_eval_profiles_resolve_to_concrete_planner_configs(self) -> None:
        import hydra
        from eval import resolve_eval_profile_config

        config_dir = str(Path(__file__).resolve().parents[1] / "config" / "eval")
        with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
            diffusion_cfg = hydra.compose(
                config_name="pusht",
                overrides=["eval_profile=diffusion"],
            )
            corrective_cfg = hydra.compose(
                config_name="pusht",
                overrides=["eval_profile=corrective_learned"],
            )

        diffusion = resolve_eval_profile_config(diffusion_cfg)
        corrective = resolve_eval_profile_config(corrective_cfg)

        self.assertEqual(diffusion.planner_type, "diffusion")
        self.assertEqual(diffusion.policy, "/data/ykz/pusht/lewm_epoch_100")
        self.assertEqual(
            diffusion.diffusion_bundle,
            "/data/ykz/pusht/diffusion_pipeline/pusht_diffusion_k128_200000/diffusion_planner_best_bundle.pt",
        )
        self.assertFalse(diffusion.corrective.enabled)
        self.assertEqual(corrective.planner_type, "diffusion")
        self.assertTrue(corrective.corrective.enabled)
        self.assertEqual(corrective.corrective.mode, "learned")
        self.assertEqual(corrective.corrective.correction_interval, 5)
        self.assertEqual(corrective.corrective.execute_horizon, 25)
        self.assertAlmostEqual(corrective.corrective.error_threshold, 5.0)

    def test_profile_overrides_keep_explicit_null_default_values(self) -> None:
        import hydra
        from eval import (
            resolve_corrective_config,
            resolve_diffusion_refinement_config,
            resolve_eval_profile_config,
        )

        config_dir = str(Path(__file__).resolve().parents[1] / "config" / "eval")
        with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = hydra.compose(
                config_name="pusht",
                overrides=[
                    "eval_profile=corrective_learned",
                    "corrective.error_threshold=0.5",
                    "diffusion_refinement.enabled=true",
                ],
            )
            base_cfg = hydra.compose(config_name="pusht")

        resolved = resolve_eval_profile_config(cfg)
        corrective = resolve_corrective_config(base_cfg)
        refinement = resolve_diffusion_refinement_config(base_cfg)

        self.assertTrue(resolved.corrective.enabled)
        self.assertAlmostEqual(resolved.corrective.error_threshold, 0.5)
        self.assertTrue(resolved.diffusion_refinement.enabled)
        self.assertEqual(corrective["mode"], "none")
        self.assertFalse(corrective["enabled"])
        self.assertFalse(refinement["enabled"])

    def test_profile_scalar_diffusion_overrides_use_normal_hydra_syntax(self) -> None:
        import hydra
        from eval import resolve_eval_profile_config

        config_dir = str(Path(__file__).resolve().parents[1] / "config" / "eval")
        with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = hydra.compose(
                config_name="cube",
                overrides=[
                    "eval_profile=diffusion",
                    "diffusion_num_candidates=64",
                    "diffusion_truncation_steps=1",
                ],
            )

        resolved = resolve_eval_profile_config(cfg)

        self.assertEqual(resolved.diffusion_num_candidates, 64)
        self.assertEqual(resolved.diffusion_truncation_steps, 1)

    def test_diffusion_refinement_can_be_enabled_as_a_diffusion_override(self) -> None:
        import hydra
        from eval import resolve_eval_profile_config

        config_dir = str(Path(__file__).resolve().parents[1] / "config" / "eval")
        with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = resolve_eval_profile_config(hydra.compose(
                config_name="tworoom",
                overrides=["eval_profile=diffusion", "diffusion_refinement.enabled=true"],
            ))
            self.assertEqual(cfg.planner_type, "diffusion")
            self.assertEqual(cfg.diffusion_selection_mode, "wm_only")
            self.assertTrue(cfg.diffusion_refinement.enabled)
            self.assertEqual(cfg.diffusion_refinement.topk, 16)
            self.assertEqual(cfg.diffusion_refinement.steps, 1)
            self.assertAlmostEqual(cfg.diffusion_refinement.step_size, 0.03)
            self.assertAlmostEqual(cfg.diffusion_refinement.grad_clip_norm, 1.0)
            self.assertNotIn("solver", cfg)

    def test_gc_idm_configs_are_concrete_bundle_configs_without_solver(self) -> None:
        import hydra
        from eval import resolve_eval_profile_config

        config_dir = str(Path(__file__).resolve().parents[1] / "config" / "eval")
        expected = {
            "cube": (
                "/data/ykz/cube/lewm_epoch_27",
                "/data/ykz/cube/gc_idm/gc_idm_best_bundle.pt",
            ),
            "pusht": (
                "/data/ykz/pusht/lewm_epoch_100",
                "/data/ykz/pusht/gc_idm/gc_idm_best_bundle.pt",
            ),
            "reacher": (
                "/data/ykz/reacher/lewm_epoch_29",
                "/data/ykz/reacher/gc_idm/gc_idm_best_bundle.pt",
            ),
            "tworoom": (
                "/data/ykz/tworoom/lewm_epoch_67",
                "/data/ykz/tworoom/gc_idm/gc_idm_best_bundle.pt",
            ),
        }

        with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
            for config_name, (policy_path, bundle_path) in expected.items():
                cfg = resolve_eval_profile_config(
                    hydra.compose(config_name=config_name, overrides=["eval_profile=gc_idm"])
                )
                self.assertEqual(cfg.planner_type, "gc_idm")
                self.assertEqual(cfg.policy, policy_path)
                self.assertEqual(cfg.gc_idm_bundle, bundle_path)
                self.assertNotIn("solver", cfg)

    def test_gc_idm_bundle_can_be_overridden_with_normal_hydra_syntax(self) -> None:
        import hydra
        from eval import resolve_eval_profile_config

        config_dir = str(Path(__file__).resolve().parents[1] / "config" / "eval")
        with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = resolve_eval_profile_config(
                hydra.compose(
                    config_name="reacher",
                    overrides=[
                        "eval_profile=gc_idm",
                        "gc_idm_bundle=/tmp/gc_idm_best_bundle.pt",
                    ],
                )
            )

        self.assertEqual(cfg.planner_type, "gc_idm")
        self.assertEqual(cfg.gc_idm_bundle, "/tmp/gc_idm_best_bundle.pt")

    def test_planning_stats_solver_records_solver_calls(self) -> None:
        from eval import PlanningStatsSolver, read_planning_stat

        class FakeSolver:
            def __init__(self) -> None:
                self.configure_args = None
                self.solve_calls = 0
                self.extra_value = "proxied"

            def configure(self, **kwargs):
                self.configure_args = kwargs
                return "configured"

            @property
            def action_dim(self) -> int:
                return 3

            @property
            def n_envs(self) -> int:
                return 5

            @property
            def horizon(self) -> int:
                return 7

            def solve(self, info_dict, init_action=None):
                self.solve_calls += 1
                return {"actions": self.solve_calls, "init_action": init_action}

        solver = PlanningStatsSolver(FakeSolver())

        self.assertEqual(solver.configure(action_space="box", n_envs=5, config="cfg"), "configured")
        self.assertEqual(solver.action_dim, 3)
        self.assertEqual(solver.n_envs, 5)
        self.assertEqual(solver.horizon, 7)
        self.assertEqual(solver.extra_value, "proxied")
        self.assertEqual(solver({"pixels": "obs"}, init_action="warm"), {"actions": 1, "init_action": "warm"})
        self.assertEqual(solver.solve({"pixels": "obs"}), {"actions": 2, "init_action": None})
        self.assertEqual(solver._num_replans, 2)
        self.assertGreaterEqual(solver._planning_time_total_sec, 0.0)

        class FakePolicy:
            pass

        policy = FakePolicy()
        policy.solver = solver
        self.assertEqual(read_planning_stat(policy, "_num_replans"), 2)
        self.assertEqual(
            read_planning_stat(policy, "_planning_time_total_sec"),
            solver._planning_time_total_sec,
        )


if __name__ == "__main__":
    unittest.main()
