from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class ComparisonExperimentPipelineTests(unittest.TestCase):
    def test_build_run_matrix_has_expected_108_runs(self) -> None:
        from scripts.run_comparison_experiments import build_run_matrix, default_experiment_spec

        runs = build_run_matrix(default_experiment_spec())

        self.assertEqual(len(runs), 108)
        self.assertEqual({run.task for run in runs}, {"cube", "pusht", "reacher", "tworoom"})
        self.assertEqual({run.method for run in runs}, {"mpc_cem", "gc_idm", "ours_full"})
        self.assertEqual({run.seed for run in runs}, {42, 43, 44})
        self.assertEqual({run.repeat for run in runs}, {0, 1, 2})

    def test_ours_full_commands_include_task_specific_overrides(self) -> None:
        from scripts.run_comparison_experiments import EvalRun, build_eval_command, default_experiment_spec

        spec = default_experiment_spec()
        cube_command = build_eval_command(
            EvalRun(task="cube", method="ours_full", seed=42, repeat=0),
            spec=spec,
            python_bin=Path("/repo/.venv/bin/python"),
        )
        reacher_command = build_eval_command(
            EvalRun(task="reacher", method="ours_full", seed=42, repeat=0),
            spec=spec,
            python_bin=Path("/repo/.venv/bin/python"),
        )
        pusht_command = build_eval_command(
            EvalRun(task="pusht", method="ours_full", seed=42, repeat=0),
            spec=spec,
            python_bin=Path("/repo/.venv/bin/python"),
        )

        self.assertIn("+dataset_h5=/data/ykz/cube/cube_single_expert.h5", cube_command)
        self.assertIn("diffusion_selection_mode=wm_only", cube_command)
        self.assertIn("diffusion_refinement.enabled=true", cube_command)
        self.assertIn("eval_profile=diffusion", reacher_command)
        self.assertIn("diffusion_selection_mode=wm_only", reacher_command)
        self.assertIn("diffusion_refinement.enabled=true", reacher_command)
        self.assertIn("+dataset_h5=/data/ykz/reacher/reacher.h5", reacher_command)
        self.assertIn("eval_profile=corrective_learned", pusht_command)
        self.assertIn("diffusion_selection_mode=wm_only", pusht_command)
        self.assertIn("diffusion_refinement.enabled=true", pusht_command)
        self.assertIn("+dataset_h5=/data/ykz/pusht/pusht_expert_train.h5", pusht_command)
        self.assertIn("seed=42", pusht_command)
        self.assertIn("trajectory_quality.enabled=true", pusht_command)
        self.assertIn("trajectory_quality.save_video=false", pusht_command)

    def test_every_run_uses_explicit_dataset_h5(self) -> None:
        from scripts.run_comparison_experiments import build_eval_command, build_run_matrix, default_experiment_spec

        spec = default_experiment_spec()
        for run in build_run_matrix(spec):
            command = build_eval_command(run, spec=spec, python_bin=Path("/repo/.venv/bin/python"))
            self.assertTrue(
                any(arg.startswith("+dataset_h5=") for arg in command),
                f"missing dataset override for {run}",
            )

    def test_parse_eval_log_extracts_core_metrics(self) -> None:
        from scripts.run_comparison_experiments import parse_eval_log_text

        metrics = parse_eval_log_text(
            """
[planner] type=diffusion task=pusht config=pusht policy=/data/ykz/pusht/lewm_epoch_100 bundle=/data/ykz/pusht/diffusion_pipeline/pusht_diffusion_k128_200000/diffusion_planner_best_bundle.pt goal_offset=25 eval_budget=50 block_horizon=5 receding_horizon=5 action_block=5 action_chunk_horizon=25 action_chunk_dim=50 runtime_execute_steps=25 replan_interval=25 action_dim=2 base_num_candidates=128 num_candidates=128 proposal_rounds=1 denoise_steps=4 start_timestep=15 eta=0.0000 noise_scale=1.0000 temperature=1.0000 selection_mode=wm_only refinement_enabled=1 refinement_steps=1 refinement_step_size=0.030000 refinement_topk=16
[planner-runtime] planner_type=diffusion task=pusht config=pusht policy=/data/ykz/pusht/lewm_epoch_100 diffusion_bundle=/data/ykz/pusht/diffusion_pipeline/pusht_diffusion_k128_200000/diffusion_planner_best_bundle.pt goal_offset=25 eval_budget=50 block_horizon=5 receding_horizon=5 action_block=5 action_chunk_horizon=25 action_dim=2 action_chunk_dim=50 num_candidates=128 selection_mode=wm_only action_clip_low=-1.0 action_clip_high=1.0
[refinement] enabled=1 steps=1 step_size=0.030000 topk=16 goal_weight=1.000000 prior_weight=0.050000 smoothness_weight=0.005000 grad_clip_norm=1.0
[corrective] mode=learned logging_prediction_error=1 correction_interval=5 effective_error_interval=5 effective_execute_horizon=25 action_block=5 error_threshold=5.000000 trigger_stat=max trigger_quantile=0.900000 trigger_scope=per_env error_metric=l2 corrector_path=/data/ykz/pusht/diffusion_pipeline/pusht_corrector_ci5/corrector_best_bundle.pt
[summary] success_rate=94.0000
[summary] episode_successes=1,0,1
[summary] evaluation_time=38.2629s
[trajectory-quality] final_goal_distance_mean=0.012345
[trajectory-quality] min_goal_distance_mean=0.001234
[trajectory-quality] path_length_mean=1.500000
[trajectory-quality] straight_line_ratio_mean=1.200000
[trajectory-quality] action_l2_mean_mean=0.700000
[trajectory-quality] action_delta_l2_mean_mean=0.400000
[trajectory-quality] action_jerk_l2_mean_mean=0.300000
[trajectory-quality] steps_to_success_mean=12.000000
[planner-stats] global_planning_calls=2
[planner-stats] effective_replans_per_episode=2
[planner-stats] planning_time_total_sec=6.069811
[planner-stats] avg_planning_time_sec=3.034905
[planner-stats] avg_generation_time_sec=0.249123
[planner-stats] refinement_time_total_sec=0.010000
[planner-stats] avg_refinement_time_sec=0.005000
[planner-stats] avg_scoring_time_sec=2.609443
[planner-stats] avg_selection_time_sec=0.008418
[corrective-stats] corrective_check_count=9
[corrective-stats] corrective_replan_count=1
[corrective-stats] corrective_replan_rate=0.500000
[corrective-stats] corrective_correction_count=1
[corrective-stats] mean_prediction_error_before_replan=16.795120 max_prediction_error_before_replan=16.795120
[corrective-stats] mean_correction_norm=0.100000 mean_action_delta_norm=0.200000
[corrective-stats] correction_time_total_sec=0.004000
[corrective-stats] avg_correction_time_sec=0.002000
[corrective-summary] prediction_error_count=450 episode_mean_count=50 mean=2.315576 max=19.738548
[corrective-summary] success_mean=2.142556 failure_mean=5.026225 fail_minus_success=2.883669 fail_success_ratio=2.345901 cohens_d=2.543583
[refinement-summary] candidate_count=16 steps=1 cost_before=8.000000 cost_after=7.000000 goal_before=6.000000 goal_after=5.000000 delta_norm=0.123000
[diffusion-rerank] mode=wm_only goal_offset=25 block_horizon=5 action_chunk_horizon=25 runtime_execute_steps=25 replan_interval=25 base_num_candidates=128 proposal_rounds=1 num_candidates=128 denoise_steps=4 start_timestep=15 eta=0.0000 noise_scale=1.0000 temperature=1.0000 finite_candidate_rate=1.0000 all_bad_env_rate=0.0000 fallback_rate=0.0000
[diffusion-rerank] selected_index_first=97 selected_wm_cost_first=17.719501 selected_model_score_first=-4.194184 selected_wm_cost_mean=67.065857 selected_model_score_mean=-6.907038
"""
        )

        self.assertEqual(metrics["planner_type"], "diffusion")
        self.assertEqual(metrics["policy"], "/data/ykz/pusht/lewm_epoch_100")
        self.assertEqual(
            metrics["diffusion_bundle"],
            "/data/ykz/pusht/diffusion_pipeline/pusht_diffusion_k128_200000/diffusion_planner_best_bundle.pt",
        )
        self.assertEqual(metrics["diffusion_selection_mode"], "wm_only")
        self.assertEqual(metrics["success_rate"], 94.0)
        self.assertEqual(metrics["episode_successes"], "1,0,1")
        self.assertEqual(metrics["evaluation_time_sec"], 38.2629)
        self.assertEqual(metrics["global_planning_calls"], 2)
        self.assertEqual(metrics["effective_replans_per_episode"], 2)
        self.assertEqual(metrics["diffusion_num_candidates"], 128)
        self.assertEqual(metrics["diffusion_truncation_steps"], 4)
        self.assertEqual(metrics["diffusion_runtime_execute_steps"], 25)
        self.assertEqual(metrics["diffusion_start_timestep"], 15)
        self.assertEqual(metrics["diffusion_refinement_enabled"], 1)
        self.assertEqual(metrics["diffusion_refinement_steps"], 1)
        self.assertEqual(metrics["diffusion_refinement_topk"], 16)
        self.assertAlmostEqual(metrics["avg_generation_time_sec"], 0.249123)
        self.assertAlmostEqual(metrics["avg_scoring_time_sec"], 2.609443)
        self.assertAlmostEqual(metrics["avg_selection_time_sec"], 0.008418)
        self.assertAlmostEqual(metrics["refinement_time_total_sec"], 0.01)
        self.assertAlmostEqual(metrics["avg_refinement_time_sec"], 0.005)
        self.assertAlmostEqual(metrics["action_l2_mean_mean"], 0.7)
        self.assertAlmostEqual(metrics["action_delta_l2_mean_mean"], 0.4)
        self.assertEqual(metrics["corrective_correction_count"], 1)
        self.assertEqual(metrics["corrective_check_count"], 9)
        self.assertEqual(metrics["corrective_replan_count"], 1)
        self.assertEqual(metrics["corrective_mode"], "learned")
        self.assertEqual(metrics["corrective_trigger_scope"], "per_env")
        self.assertEqual(
            metrics["corrector_path"],
            "/data/ykz/pusht/diffusion_pipeline/pusht_corrector_ci5/corrector_best_bundle.pt",
        )
        self.assertAlmostEqual(metrics["prediction_error_mean"], 2.315576)
        self.assertAlmostEqual(metrics["prediction_error_max"], 19.738548)
        self.assertAlmostEqual(metrics["successful_prediction_error_mean"], 2.142556)
        self.assertAlmostEqual(metrics["failed_prediction_error_mean"], 5.026225)
        self.assertAlmostEqual(metrics["prediction_error_fail_success_ratio"], 2.345901)
        self.assertAlmostEqual(metrics["prediction_error_cohens_d_fail_vs_success"], 2.543583)
        self.assertAlmostEqual(metrics["mean_prediction_error_before_replan"], 16.795120)
        self.assertAlmostEqual(metrics["max_prediction_error_before_replan"], 16.795120)
        self.assertAlmostEqual(metrics["mean_correction_norm"], 0.1)
        self.assertAlmostEqual(metrics["mean_action_delta_norm"], 0.2)
        self.assertAlmostEqual(metrics["correction_time_total_sec"], 0.004)
        self.assertAlmostEqual(metrics["finite_candidate_rate"], 1.0)
        self.assertAlmostEqual(metrics["fallback_rate"], 0.0)
        self.assertAlmostEqual(metrics["all_bad_env_rate"], 0.0)
        self.assertAlmostEqual(metrics["refinement_cost_before"], 8.0)
        self.assertAlmostEqual(metrics["refinement_cost_after"], 7.0)
        self.assertAlmostEqual(metrics["refinement_delta_norm"], 0.123)
        self.assertAlmostEqual(metrics["selected_wm_cost_mean"], 67.065857)
        self.assertAlmostEqual(metrics["selected_model_score_mean"], -6.907038)

    def test_raw_columns_include_required_experiment_metrics(self) -> None:
        from scripts.run_comparison_experiments import RAW_COLUMNS

        required = {
            "planner_type",
            "policy",
            "diffusion_bundle",
            "dataset_h5",
            "success_rate",
            "episode_successes",
            "evaluation_time_sec",
            "planning_time_total_sec",
            "avg_planning_time_sec",
            "global_planning_calls",
            "effective_replans_per_episode",
            "final_goal_distance_mean",
            "min_goal_distance_mean",
            "steps_to_success_mean",
            "path_length_mean",
            "straight_line_ratio_mean",
            "action_l2_mean_mean",
            "action_delta_l2_mean_mean",
            "action_jerk_l2_mean_mean",
            "diffusion_selection_mode",
            "diffusion_runtime_execute_steps",
            "diffusion_num_candidates",
            "diffusion_truncation_steps",
            "diffusion_start_timestep",
            "avg_generation_time_sec",
            "avg_scoring_time_sec",
            "avg_selection_time_sec",
            "refinement_time_total_sec",
            "avg_refinement_time_sec",
            "finite_candidate_rate",
            "fallback_rate",
            "all_bad_env_rate",
            "selected_wm_cost_mean",
            "selected_model_score_mean",
            "corrective_check_count",
            "corrective_replan_count",
            "corrective_replan_rate",
            "corrective_correction_count",
            "prediction_error_mean",
            "prediction_error_max",
            "successful_prediction_error_mean",
            "failed_prediction_error_mean",
            "prediction_error_cohens_d_fail_vs_success",
            "mean_prediction_error_before_replan",
            "max_prediction_error_before_replan",
            "mean_correction_norm",
            "mean_action_delta_norm",
            "correction_time_total_sec",
            "avg_correction_time_sec",
            "command",
            "log_path",
        }

        self.assertFalse(required - set(RAW_COLUMNS))

    def test_dry_run_writes_logs_and_result_template(self) -> None:
        from scripts.run_comparison_experiments import (
            ExperimentSpec,
            TaskSpec,
            run_experiments,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = ExperimentSpec(
                tasks={
                    "reacher": TaskSpec(
                        config_name="reacher",
                        dataset_h5="/data/ykz/reacher/reacher.h5",
                        ours_overrides=("eval_profile=diffusion", "diffusion_refinement.enabled=true"),
                    )
                },
                methods=("mpc_cem", "ours_full"),
                seeds=(42,),
                repeats=(0, 1),
                eval_num_eval=5,
            )
            summary = run_experiments(
                spec=spec,
                output_root=root / "runs",
                result_path=root / "result.md",
                python_bin=Path("/repo/.venv/bin/python"),
                dry_run=True,
                force=True,
                progress=False,
            )

            self.assertEqual(summary["total_runs"], 4)
            self.assertEqual(summary["completed_runs"], 4)
            self.assertTrue((root / "result.md").exists())
            self.assertIn("reacher", (root / "result.md").read_text(encoding="utf-8"))
            self.assertEqual(len(list((root / "runs").glob("*/*.log"))), 4)

    def test_real_run_does_not_skip_existing_dry_run_rows(self) -> None:
        from scripts.run_comparison_experiments import (
            ExperimentSpec,
            TaskSpec,
            run_experiments,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = ExperimentSpec(
                tasks={
                    "reacher": TaskSpec(
                        config_name="reacher",
                        dataset_h5="/data/ykz/reacher/reacher.h5",
                        ours_overrides=("eval_profile=diffusion", "diffusion_refinement.enabled=true"),
                    )
                },
                methods=("ours_full",),
                seeds=(42,),
                repeats=(0,),
                eval_num_eval=5,
            )
            run_experiments(
                spec=spec,
                output_root=root / "runs",
                result_path=root / "result.md",
                python_bin=Path("/repo/.venv/bin/python"),
                dry_run=True,
                force=True,
                progress=False,
            )

            summary = run_experiments(
                spec=spec,
                output_root=root / "runs",
                result_path=root / "result.md",
                python_bin=Path("/repo/.venv/bin/python"),
                dry_run=False,
                force=False,
                progress=False,
                runner=lambda command, log_path: log_path.write_text(
                    "[summary] success_rate=88.0000\n[summary] evaluation_time=1.0000s\n",
                    encoding="utf-8",
                ),
            )

            self.assertEqual(summary["completed_runs"], 1)
            self.assertEqual(summary["skipped_runs"], 0)
            self.assertIn("88", (root / "result.md").read_text(encoding="utf-8"))

    def test_resume_reruns_when_command_changes(self) -> None:
        from scripts.run_comparison_experiments import (
            ExperimentSpec,
            TaskSpec,
            run_experiments,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_spec = ExperimentSpec(
                tasks={
                    "reacher": TaskSpec(
                        config_name="reacher",
                        dataset_h5="/data/ykz/reacher/reacher.h5",
                        ours_overrides=("eval_profile=diffusion", "diffusion_refinement.enabled=true"),
                    )
                },
                methods=("ours_full",),
                seeds=(42,),
                repeats=(0,),
                eval_num_eval=5,
            )
            changed_spec = ExperimentSpec(
                tasks=base_spec.tasks,
                methods=base_spec.methods,
                seeds=base_spec.seeds,
                repeats=base_spec.repeats,
                eval_num_eval=6,
            )
            run_experiments(
                spec=base_spec,
                output_root=root / "runs",
                result_path=root / "result.md",
                python_bin=Path("/repo/.venv/bin/python"),
                dry_run=True,
                force=True,
                progress=False,
            )

            summary = run_experiments(
                spec=changed_spec,
                output_root=root / "runs",
                result_path=root / "result.md",
                python_bin=Path("/repo/.venv/bin/python"),
                dry_run=True,
                force=False,
                progress=False,
            )

            raw_csv = (root / "runs" / "raw_runs.csv").read_text(encoding="utf-8")
            self.assertEqual(summary["completed_runs"], 1)
            self.assertEqual(raw_csv.count("reacher,ours_full"), 1)
            self.assertIn("eval.num_eval=6", raw_csv)

    def test_default_ours_full_overrides_resolve_to_enabled_optimizations(self) -> None:
        import hydra

        from eval import resolve_eval_profile_config
        from scripts.run_comparison_experiments import build_eval_command, default_experiment_spec

        config_dir = str(Path(__file__).resolve().parents[1] / "config" / "eval")
        spec = default_experiment_spec()
        with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
            for run in [r for r in __import__("scripts.run_comparison_experiments", fromlist=["build_run_matrix"]).build_run_matrix(spec) if r.method == "ours_full" and r.seed == 42 and r.repeat == 0]:
                command = build_eval_command(run, spec=spec, python_bin=Path("/repo/.venv/bin/python"))
                config_name = command[command.index("--config-name") + 1]
                overrides = [arg for arg in command[command.index(config_name) + 1 :]]
                cfg = resolve_eval_profile_config(hydra.compose(config_name=config_name, overrides=overrides))
                self.assertEqual(cfg.planner_type, "diffusion")
                self.assertEqual(cfg.diffusion_selection_mode, "wm_only")
                self.assertTrue(bool(cfg.diffusion_refinement.enabled), run)
                self.assertEqual(cfg.diffusion_refinement.steps, 1)
                self.assertEqual(cfg.diffusion_refinement.topk, 16)
                if run.task == "pusht":
                    self.assertTrue(bool(cfg.corrective.enabled))
                    self.assertIn(str(cfg.corrective.mode), {"learned", "replan"})


if __name__ == "__main__":
    unittest.main()
