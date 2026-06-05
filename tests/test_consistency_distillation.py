import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import torch
from omegaconf import OmegaConf

from diffusion.model import DiffusionPlannerModel, DiffusionPlannerModelConfig, save_diffusion_planner_bundle


def make_planner(num_train_steps: int = 8) -> DiffusionPlannerModel:
    config = DiffusionPlannerModelConfig(
        latent_dim=3,
        plan_horizon=2,
        action_dim=2,
        num_anchors=3,
        hidden_dim=16,
        num_layers=2,
        timestep_embedding_dim=8,
        fusion_num_layers=1,
        num_train_steps=num_train_steps,
        truncation_steps=2,
        beta_end=1e-2,
    )
    anchors = torch.zeros(config.num_anchors, config.action_chunk_dim)
    return DiffusionPlannerModel(config, anchors)


class LinearLatentWorldModel(torch.nn.Module):
    def __init__(self, latent_dim: int, action_chunk_dim: int):
        super().__init__()
        self.proj = torch.nn.Linear(action_chunk_dim, latent_dim, bias=False)

    def forward(self, z_cur: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        return z_cur + self.proj(action_chunk)


class ConsistencyDistillationTests(unittest.TestCase):
    def test_default_teacher_warm_start_keeps_student_trainable(self) -> None:
        from diffusion.consistency_distillation import parse_args, run_training

        teacher = make_planner()
        dataset = {
            "z_cur": torch.randn(6, teacher.latent_dim),
            "z_goal": torch.randn(6, teacher.latent_dim),
            "teacher_plan": torch.randn(6, teacher.action_chunk_dim),
        }

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            dataset_path = tmp / "planner_dataset.pt"
            teacher_bundle_path = tmp / "teacher_bundle.pt"
            output_dir = tmp / "consistency"
            torch.save(dataset, dataset_path)
            save_diffusion_planner_bundle(teacher, teacher_bundle_path)

            args = parse_args(
                [
                    "--dataset-path",
                    str(dataset_path),
                    "--teacher-bundle-path",
                    str(teacher_bundle_path),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cpu",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "3",
                    "--val-batch-size",
                    "3",
                    "--val-split",
                    "0.5",
                    "--num-workers",
                    "0",
                    "--teacher-ode-steps",
                    "1",
                    "--log-every",
                    "0",
                ]
            )

            run_training(args)

            self.assertTrue((output_dir / "consistency_planner_best_bundle.pt").exists())

    def test_pseudo_huber_delta_zero_has_finite_zero_error_gradient(self) -> None:
        from diffusion.consistency_distillation import pseudo_huber_loss

        prediction = torch.zeros(4, requires_grad=True)
        target = torch.zeros(4)

        loss = pseudo_huber_loss(prediction, target, delta=0.0)
        loss.backward()

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue(torch.allclose(prediction.grad, torch.zeros_like(prediction.grad)))

    def test_sampled_teacher_bridge_always_moves_to_lower_timestep(self) -> None:
        from diffusion.consistency_distillation import sample_consistency_timesteps

        timesteps = sample_consistency_timesteps(
            batch_size=64,
            num_candidates=3,
            num_train_steps=8,
            teacher_ode_steps=2,
            device=torch.device("cpu"),
            sampling="uniform",
        )

        self.assertTrue(torch.all(timesteps.teacher_target < timesteps.start))
        self.assertTrue(torch.all(timesteps.teacher_target >= 0))

    def test_ema_update_moves_target_toward_student(self) -> None:
        from diffusion.consistency_distillation import update_ema_model

        source = make_planner()
        target = make_planner()
        for param in source.parameters():
            param.data.fill_(2.0)
        for param in target.parameters():
            param.data.fill_(0.0)

        update_ema_model(target, source, decay=0.25)

        target_param = next(target.parameters())
        self.assertTrue(torch.allclose(target_param, torch.full_like(target_param, 1.5)))

    def test_consistency_loss_backpropagates_through_student_only(self) -> None:
        from diffusion.consistency_distillation import ConsistencyDistillationConfig, compute_consistency_losses

        student = make_planner()
        teacher = make_planner()
        ema_student = make_planner()
        for param in teacher.parameters():
            param.requires_grad_(False)
        for param in ema_student.parameters():
            param.requires_grad_(False)

        batch = {
            "z_cur": torch.randn(4, student.latent_dim),
            "z_goal": torch.randn(4, student.latent_dim),
            "teacher_plan": torch.randn(4, student.action_chunk_dim),
        }
        world_model = LinearLatentWorldModel(student.latent_dim, student.action_chunk_dim)
        cfg = ConsistencyDistillationConfig(
            ctm_loss_weight=1.0,
            action_loss_weight=0.3,
            goal_loss_weight=0.2,
            score_loss_weight=0.1,
            teacher_ode_steps=2,
        )

        loss_dict, metrics = compute_consistency_losses(
            student=student,
            teacher=teacher,
            ema_student=ema_student,
            batch=batch,
            config=cfg,
            world_model=world_model,
        )
        loss_dict["total_loss"].backward()

        self.assertGreater(float(metrics["total_loss"]), 0.0)
        self.assertIn("ctm_loss", metrics)
        self.assertTrue(any(param.grad is not None for param in student.parameters()))
        self.assertTrue(all(param.grad is None for param in teacher.parameters()))
        self.assertTrue(all(param.grad is None for param in ema_student.parameters()))

    def test_one_step_sampler_returns_candidate_bundle(self) -> None:
        from diffusion.consistency_distillation import ConsistencyPlannerSampler

        student = make_planner()
        sampler = ConsistencyPlannerSampler(student)
        z_cur = torch.randn(2, student.latent_dim)
        z_goal = torch.randn(2, student.latent_dim)

        output = sampler.sample(z_cur, z_goal, steps=1, noise_scale=0.0)

        self.assertEqual(tuple(output["candidates"].shape), (2, student.num_anchors, student.action_chunk_dim))
        self.assertEqual(tuple(output["score_logits"].shape), (2, student.num_anchors))
        self.assertEqual(output["timesteps"].numel(), 1)

    def test_training_entrypoint_is_separate_from_diffusion_train(self) -> None:
        import train_consistency_planner

        self.assertTrue(callable(train_consistency_planner.main))
        self.assertNotEqual(train_consistency_planner.main, types.SimpleNamespace)

    def test_progress_iterator_is_disabled_for_non_tty_output(self) -> None:
        from diffusion.consistency_distillation import progress_iter

        values = [1, 2, 3]

        with mock.patch("diffusion.consistency_distillation.sys.stderr.isatty", return_value=False):
            wrapped = progress_iter(values, desc="train")

        self.assertIs(wrapped, values)

    def test_progress_iterator_uses_tqdm_for_tty_output(self) -> None:
        from diffusion.consistency_distillation import progress_iter

        calls = []

        def fake_tqdm(iterable, **kwargs):
            calls.append((iterable, kwargs))
            return ("wrapped", iterable)

        values = [1, 2, 3]
        with (
            mock.patch("diffusion.consistency_distillation.sys.stderr.isatty", return_value=True),
            mock.patch("diffusion.consistency_distillation._tqdm", fake_tqdm),
        ):
            wrapped = progress_iter(values, desc="train", total=3, unit="batch")

        self.assertEqual(wrapped, ("wrapped", values))
        self.assertEqual(calls[0][0], values)
        self.assertEqual(calls[0][1]["desc"], "train")
        self.assertEqual(calls[0][1]["total"], 3)
        self.assertEqual(calls[0][1]["unit"], "batch")

    def test_hydra_config_maps_to_training_args(self) -> None:
        import hydra
        from diffusion.consistency_distillation import build_args_from_config

        config_dir = str(Path(__file__).resolve().parents[1] / "config" / "consistency")
        with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
            cfg = hydra.compose(
                config_name="train",
                overrides=[
                    "task=pusht",
                    "runtime.device=cpu",
                    "runtime.seed=7",
                    "train.epochs=3",
                    "distill.teacher_ode_steps=1",
                ],
            )

        args = build_args_from_config(cfg)

        self.assertEqual(args.dataset_path, cfg.task.planner_dataset_path)
        self.assertEqual(args.teacher_bundle_path, cfg.teacher.bundle_path)
        self.assertEqual(args.output_dir, cfg.output.dir)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.teacher_ode_steps, 1)

    def test_hydra_task_defaults_match_diffusion_pipeline_outputs(self) -> None:
        import hydra
        from diffusion.consistency_distillation import build_args_from_config

        config_dir = str(Path(__file__).resolve().parents[1] / "config" / "consistency")
        expected = {
            "cube": (
                "/data/ykz/cube/diffusion_pipeline/cube_diffusion_k128_200000/diffusion_planner_best_bundle.pt",
                "/data/ykz/cube/lewm_epoch_27",
            ),
            "pusht": (
                "/data/ykz/pusht/diffusion_pipeline/pusht_diffusion_k128_200000/diffusion_planner_best_bundle.pt",
                "/data/ykz/pusht/lewm_epoch_100",
            ),
            "reacher": (
                "/data/ykz/reacher/diffusion_pipeline/reacher_diffusion_k128_200000/diffusion_planner_best_bundle.pt",
                "/data/ykz/reacher/lewm_epoch_20",
            ),
            "tworoom": (
                "/data/ykz/tworoom/diffusion_pipeline/tworoom_diffusion_k128_200000/diffusion_planner_best_bundle.pt",
                "/data/ykz/tworoom/lewm_epoch_67",
            ),
        }
        with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
            for task, (bundle_path, wm_policy) in expected.items():
                cfg = hydra.compose(config_name="train", overrides=[f"task={task}"])
                args = build_args_from_config(cfg)
                self.assertEqual(args.teacher_bundle_path, bundle_path)
                self.assertEqual(args.wm_policy, wm_policy)

    def test_consistency_eval_configs_force_one_step_runtime(self) -> None:
        import hydra
        from eval import resolve_eval_profile_config

        config_dir = str(Path(__file__).resolve().parents[1] / "config" / "eval")
        expected = {
            "cube": (
                "/data/ykz/cube/lewm_epoch_27",
                "/data/ykz/cube/diffusion_pipeline/consistency/consistency_planner_best_bundle.pt",
            ),
            "pusht": (
                "/data/ykz/pusht/lewm_epoch_100",
                "/data/ykz/pusht/diffusion_pipeline/consistency/consistency_planner_best_bundle.pt",
            ),
            "reacher": (
                "/data/ykz/reacher/lewm_epoch_20",
                "/data/ykz/reacher/diffusion_pipeline/consistency/consistency_planner_best_bundle.pt",
            ),
            "tworoom": (
                "/data/ykz/tworoom/lewm_epoch_67",
                "/data/ykz/tworoom/diffusion_pipeline/consistency/consistency_planner_best_bundle.pt",
            ),
        }

        with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
            for config_name, (policy_path, bundle_path) in expected.items():
                cfg = resolve_eval_profile_config(
                    hydra.compose(
                        config_name=config_name,
                        overrides=["eval_profile=consistency"],
                    )
                )
                self.assertEqual(cfg.planner_type, "diffusion")
                self.assertEqual(cfg.policy, policy_path)
                self.assertEqual(cfg.diffusion_bundle, bundle_path)
                self.assertEqual(cfg.diffusion_selection_mode, "wm_only")
                self.assertEqual(cfg.diffusion_num_candidates, 128)
                self.assertEqual(cfg.diffusion_truncation_steps, 1)

    def test_hydra_entrypoint_accepts_resolved_config(self) -> None:
        from diffusion.consistency_distillation import hydra_main

        cfg = OmegaConf.create(
            {
                "task": {
                    "planner_dataset_path": "/tmp/planner_dataset.pt",
                    "wm_policy": "/tmp/lewm_policy",
                },
                "teacher": {
                    "bundle_path": "/tmp/teacher_bundle.pt",
                    "student_init_bundle_path": None,
                },
                "output": {"dir": "/tmp/consistency_output"},
                "runtime": {"device": "cpu", "seed": 1},
                "train": {
                    "epochs": 1,
                    "batch_size": 2,
                    "val_batch_size": 2,
                    "val_split": 0.25,
                    "num_workers": 0,
                    "lr": 1e-4,
                    "weight_decay": 1e-4,
                    "grad_clip_norm": 1.0,
                    "log_every": 1,
                },
                "distill": {
                    "ctm_loss_weight": 1.0,
                    "action_loss_weight": 1.0,
                    "goal_loss_weight": 0.0,
                    "score_loss_weight": 0.0,
                    "dsm_loss_weight": 0.0,
                    "teacher_ode_steps": 2,
                    "huber_delta": 0.0,
                    "ema_decay": 0.999,
                    "timestep_sampling": "uniform",
                },
                "goal_loss": {
                    "history_size": 3,
                    "receding_horizon": None,
                    "action_block": None,
                },
            }
        )

        with self.assertRaises(FileNotFoundError):
            hydra_main(cfg)


if __name__ == "__main__":
    unittest.main()
