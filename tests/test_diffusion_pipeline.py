from __future__ import annotations

import tempfile
import unittest
import subprocess
from pathlib import Path


class DiffusionConfigValidationTests(unittest.TestCase):
    def test_normalize_task_name_aliases(self) -> None:
        from diffusion.config_schema import normalize_task_name

        self.assertEqual(normalize_task_name("two-room"), "tworoom")
        self.assertEqual(normalize_task_name("two_room"), "tworoom")
        self.assertEqual(normalize_task_name("researcher"), "reacher")
        self.assertEqual(normalize_task_name("cube"), "cube")

    def test_resolve_policy_path_accepts_prefix_or_ckpt_file(self) -> None:
        from diffusion.config_schema import resolve_policy_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / "lewm_epoch_26_object.ckpt"
            ckpt.write_bytes(b"checkpoint")
            self.assertEqual(resolve_policy_path(root / "lewm_epoch_26"), root / "lewm_epoch_26")
            self.assertEqual(resolve_policy_path(ckpt), ckpt)

    def test_resolve_policy_path_rejects_missing_path(self) -> None:
        from diffusion.config_schema import resolve_policy_path

        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "lewm_epoch_100"
            with self.assertRaisesRegex(FileNotFoundError, "Missing world model policy"):
                resolve_policy_path(missing)


class DiffusionPipelineTests(unittest.TestCase):
    def _base_cfg(self, root: Path):
        from omegaconf import OmegaConf

        raw_h5 = root / "cube_single_expert.h5"
        raw_h5.write_bytes(b"h5")
        policy = root / "lewm_epoch_26_object.ckpt"
        policy.write_bytes(b"ckpt")
        return OmegaConf.create(
            {
                "task": {
                    "name": "cube",
                    "eval_config": "cube",
                    "data_root": str(root),
                    "dataset_name": "cube_single_expert",
                    "raw_h5": str(raw_h5),
                    "split_train_h5": str(root / "splits" / "cube_single_expert_train" / "cube_single_expert.h5"),
                    "split_val_h5": str(root / "splits" / "cube_single_expert_val" / "cube_single_expert.h5"),
                    "split_train_root": str(root / "splits" / "cube_single_expert_train"),
                    "split_val_root": str(root / "splits" / "cube_single_expert_val"),
                    "planner_dataset_path": str(root / "diffusion_pipeline" / "planner_dataset.pt"),
                    "anchor_bundle_path": str(root / "diffusion_pipeline" / "action_anchors.pt"),
                    "train_output_dir": str(root / "diffusion_pipeline" / "train"),
                    "wm_policy": str(root / "lewm_epoch_26"),
                    "episode_key": "auto",
                    "download_if_missing": False,
                    "download": {"url": None, "archive_type": "none", "output": None},
                },
                "pipeline": {
                    "device": "cpu",
                    "seed": 42,
                    "use_raw_dataset": True,
                    "train_ratio": 0.8,
                    "num_samples": 16,
                    "build_batch_size": 4,
                    "force_split": False,
                    "force_dataset": False,
                    "force_anchors": False,
                    "force_train": False,
                    "dry_run": False,
                },
                "anchors": {"num_anchors": 8, "max_samples": 16, "max_iter": 10},
                "train": {
                    "epochs": 2,
                    "batch_size": 4,
                    "val_batch_size": 4,
                    "num_workers": 0,
                    "loss_preset": "simple_bce",
                    "lr": 0.001,
                    "weight_decay": 0.0001,
                    "val_split": 0.1,
                    "log_every": 10,
                },
            }
        )

    def test_stage_commands_are_argument_lists(self) -> None:
        from diffusion.pipeline import build_stage_commands

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(Path(tmp))
            commands = build_stage_commands(cfg)
            self.assertIsInstance(commands["split_hdf5"], list)
            self.assertIn("--input-h5", commands["split_hdf5"])
            self.assertIn("--wm-policy", commands["build_dataset"])
            self.assertIn("--dataset-h5", commands["build_dataset"])
            self.assertIn(str(Path(cfg.task.raw_h5)), commands["build_dataset"])
            self.assertIn("--loss-preset", commands["train"])
            self.assertNotIsInstance(commands["train"], str)

    def test_raw_dataset_mode_skips_split_and_builds_dataset_from_raw_h5(self) -> None:
        from diffusion.pipeline import DiffusionPipeline

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(Path(tmp))
            cfg.pipeline.use_raw_dataset = True
            cfg.pipeline.dry_run = True
            pipeline = DiffusionPipeline(cfg, runner=lambda stage, argv, log_path: None)
            summary = pipeline.run()
            self.assertEqual(summary["stages"][1]["stage"], "split_hdf5")
            self.assertEqual(summary["stages"][1]["status"], "skipped")
            self.assertEqual(summary["stages"][1]["reason"], "pipeline.use_raw_dataset=true")
            build_command = summary["stages"][2]["command"]
            dataset_h5_index = build_command.index("--dataset-h5") + 1
            self.assertEqual(build_command[dataset_h5_index], str(Path(cfg.task.raw_h5)))
            self.assertNotIn(str(Path(cfg.task.split_train_h5)), build_command)

    def test_split_mode_builds_dataset_from_split_train_h5(self) -> None:
        from diffusion.pipeline import DiffusionPipeline

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(Path(tmp))
            cfg.pipeline.use_raw_dataset = False
            cfg.pipeline.dry_run = True
            pipeline = DiffusionPipeline(cfg, runner=lambda stage, argv, log_path: None)
            summary = pipeline.run()
            self.assertEqual(summary["stages"][1]["stage"], "split_hdf5")
            self.assertEqual(summary["stages"][1]["reason"], "dry_run")
            build_command = summary["stages"][2]["command"]
            dataset_h5_index = build_command.index("--dataset-h5") + 1
            self.assertEqual(build_command[dataset_h5_index], str(Path(cfg.task.split_train_h5)))
            self.assertNotIn(str(Path(cfg.task.raw_h5)), build_command[dataset_h5_index + 1 :])

    def test_existing_outputs_skip_stages_when_force_false(self) -> None:
        from diffusion.pipeline import DiffusionPipeline

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(Path(tmp))
            for key in ["split_train_h5", "split_val_h5", "planner_dataset_path", "anchor_bundle_path"]:
                Path(cfg.task[key]).parent.mkdir(parents=True, exist_ok=True)
                Path(cfg.task[key]).write_bytes(b"artifact")
            best = Path(cfg.task.train_output_dir) / "diffusion_planner_best_bundle.pt"
            best.parent.mkdir(parents=True, exist_ok=True)
            best.write_bytes(b"bundle")
            pipeline = DiffusionPipeline(cfg, runner=lambda stage, argv, log_path: None)
            summary = pipeline.run()
            self.assertEqual([stage["status"] for stage in summary["stages"]], ["skipped"] * 5)

    def test_force_dataset_rebuilds_only_dataset_stage(self) -> None:
        from diffusion.pipeline import DiffusionPipeline

        calls: list[str] = []

        def runner(stage: str, argv: list[str], log_path: Path) -> None:
            calls.append(stage)
            if stage == "build_dataset":
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                Path(log_path).write_text("ran\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(Path(tmp))
            cfg.pipeline.force_dataset = True
            for key in ["split_train_h5", "split_val_h5", "planner_dataset_path", "anchor_bundle_path"]:
                Path(cfg.task[key]).parent.mkdir(parents=True, exist_ok=True)
                Path(cfg.task[key]).write_bytes(b"artifact")
            best = Path(cfg.task.train_output_dir) / "diffusion_planner_best_bundle.pt"
            best.parent.mkdir(parents=True, exist_ok=True)
            best.write_bytes(b"bundle")
            pipeline = DiffusionPipeline(cfg, runner=runner)
            summary = pipeline.run()
            self.assertEqual(calls, ["build_dataset"])
            self.assertEqual(summary["stages"][2]["status"], "completed")

    def test_missing_raw_h5_without_download_fails_cleanly(self) -> None:
        from diffusion.pipeline import DiffusionPipeline

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(Path(tmp))
            Path(cfg.task.raw_h5).unlink()
            pipeline = DiffusionPipeline(cfg, runner=lambda stage, argv, log_path: None)
            with self.assertRaisesRegex(FileNotFoundError, "Missing raw HDF5"):
                pipeline.run()

    def test_pipeline_reports_stage_progress_events(self) -> None:
        from diffusion.pipeline import DiffusionPipeline

        class RecordingProgress:
            def __init__(self) -> None:
                self.events: list[tuple[str, str, str]] = []

            def start(self, stages: list[str]) -> None:
                self.events.append(("start", ",".join(stages), ""))

            def stage_started(self, stage: str, total: int | None, log_path: Path | None) -> None:
                self.events.append(("started", stage, "" if log_path is None else log_path.name))

            def stage_progress(
                self,
                stage: str,
                completed: int | None = None,
                total: int | None = None,
                description: str | None = None,
            ) -> None:
                self.events.append(("progress", stage, str(completed)))

            def stage_finished(self, stage: str, status: str, reason: str | None = None) -> None:
                self.events.append(("finished", stage, status))

            def finish(self) -> None:
                self.events.append(("finish", "", ""))

        calls: list[str] = []

        def runner(stage: str, argv: list[str], log_path: Path) -> None:
            calls.append(stage)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("ran\n", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._base_cfg(Path(tmp))
            reporter = RecordingProgress()
            pipeline = DiffusionPipeline(cfg, runner=runner, progress_reporter=reporter)
            summary = pipeline.run()

            self.assertEqual(calls, ["build_dataset", "build_anchors", "train"])
            self.assertEqual([stage["stage"] for stage in summary["stages"]], [
                "prepare_data",
                "split_hdf5",
                "build_dataset",
                "build_anchors",
                "train",
            ])
            self.assertIn(("start", "prepare_data,split_hdf5,build_dataset,build_anchors,train", ""), reporter.events)
            self.assertIn(("finished", "prepare_data", "skipped"), reporter.events)
            self.assertIn(("finished", "split_hdf5", "skipped"), reporter.events)
            self.assertIn(("finished", "build_dataset", "completed"), reporter.events)
            self.assertIn(("finished", "build_anchors", "completed"), reporter.events)
            self.assertIn(("finished", "train", "completed"), reporter.events)
            self.assertEqual(reporter.events[-1], ("finish", "", ""))

    def test_parse_stage_progress_lines(self) -> None:
        from diffusion.pipeline import parse_stage_progress_line

        dataset_update = parse_stage_progress_line(
            "build_dataset",
            "[batch] rows 384:512 (size=128, first_row=384, last_row=511)",
            default_total=1000,
        )
        self.assertEqual(dataset_update.completed, 512)
        self.assertEqual(dataset_update.total, 1000)
        self.assertEqual(dataset_update.description, "rows 512/1000")

        train_step_update = parse_stage_progress_line(
            "train",
            "[train] epoch=003 step=0010/0100 batch_size=64 total_loss=1.000000",
            default_total=8,
        )
        self.assertEqual(train_step_update.completed, 211)
        self.assertEqual(train_step_update.total, 800)
        self.assertEqual(train_step_update.description, "epoch 3/8 step 11/100")

        train_epoch_update = parse_stage_progress_line(
            "train",
            "[epoch] epoch=004 task=cube train_loss=0.500000 val_loss=0.400000",
            default_total=8,
        )
        self.assertEqual(train_epoch_update.completed, 4)
        self.assertEqual(train_epoch_update.total, 8)
        self.assertEqual(train_epoch_update.description, "epoch 4/8")

    def test_default_progress_reporter_is_silent_for_non_tty_output(self) -> None:
        from diffusion.pipeline import NullProgressReporter, default_progress_reporter

        self.assertIsInstance(default_progress_reporter(is_terminal=False), NullProgressReporter)


class DiffusionDatasetBuilderPathTests(unittest.TestCase):
    def test_explicit_dataset_h5_defaults_to_file_stem_and_parent_cache_dir(self) -> None:
        from planners.build_single_peak_dataset import resolve_explicit_dataset_source

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h5_path = root / "cube_single_expert.h5"
            h5_path.write_bytes(b"h5")
            dataset_name, cache_dir = resolve_explicit_dataset_source(str(h5_path))
            self.assertEqual(dataset_name, "cube_single_expert")
            self.assertEqual(cache_dir, root)

    def test_explicit_dataset_h5_supports_nested_dataset_name(self) -> None:
        from planners.build_single_peak_dataset import resolve_explicit_dataset_source

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h5_path = root / "dmc" / "reacher_random.h5"
            h5_path.parent.mkdir(parents=True)
            h5_path.write_bytes(b"h5")
            dataset_name, cache_dir = resolve_explicit_dataset_source(
                str(h5_path),
                dataset_name="dmc/reacher_random",
            )
            self.assertEqual(dataset_name, "dmc/reacher_random")
            self.assertEqual(cache_dir, root)


class DiffusionEvalPathTests(unittest.TestCase):
    def test_eval_explicit_dataset_h5_defaults_to_file_stem_and_parent_cache_dir(self) -> None:
        from eval import resolve_explicit_dataset_source

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h5_path = root / "reacher.h5"
            h5_path.write_bytes(b"h5")
            dataset_name, cache_dir = resolve_explicit_dataset_source(
                str(h5_path),
                dataset_name="dmc/reacher_random",
            )
            self.assertEqual(dataset_name, "reacher")
            self.assertEqual(cache_dir, root)


class LegacyDiffusionPipelineScriptTests(unittest.TestCase):
    def test_legacy_script_dry_run_uses_raw_dataset_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_h5 = root / "pusht_expert_train.h5"
            input_h5.write_bytes(b"h5")
            wm_policy = root / "lewm_epoch_100"
            cmd = [
                "bash",
                "scripts/run_diffusion_pipeline.sh",
                "--task",
                "pusht",
                "--input-h5",
                str(input_h5),
                "--wm-policy",
                str(wm_policy),
                "--dry-run",
            ]
            completed = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True, text=True, capture_output=True)
            self.assertNotIn("[run:split]", completed.stdout)
            self.assertIn("--dataset-h5", completed.stdout)
            self.assertIn(str(input_h5), completed.stdout)
            self.assertIn(f"dataset_h5={input_h5}", completed.stdout)


class DiffusionHydraConfigTests(unittest.TestCase):
    def test_cube_config_embeds_default_policy(self) -> None:
        import hydra

        config_dir = str(Path(__file__).resolve().parents[1] / "config" / "diffusion")
        with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = hydra.compose(config_name="train", overrides=["task=cube", "pipeline.device=cpu"])
        self.assertEqual(cfg.task.name, "cube")
        self.assertEqual(cfg.task.wm_policy, "/data/ykz/cube/lewm_epoch_27")
        self.assertEqual(cfg.task.raw_h5, "/data/ykz/cube/cube_single_expert.h5")


if __name__ == "__main__":
    unittest.main()
