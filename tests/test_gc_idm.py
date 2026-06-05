import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.utils.data import Subset


class GCIDMModelTests(unittest.TestCase):
    def test_model_predicts_single_step_action_with_horizon_conditioning(self) -> None:
        from planners.gc_idm_model import GCIDMModel, GCIDMModelConfig

        cfg = GCIDMModelConfig(
            latent_dim=4,
            action_dim=2,
            hidden_dim=512,
            num_layers=3,
            horizon_embedding_dim=64,
            horizon_mlp_layers=2,
        )
        model = GCIDMModel(cfg)
        z_cur = torch.randn(5, 4)
        z_goal = torch.randn(5, 4)
        horizon = torch.tensor([1, 2, 3, 4, 5])

        action = model(z_cur, z_goal, horizon)

        self.assertEqual(tuple(action.shape), (5, 2))
        self.assertEqual(cfg.input_dim, 8)
        self.assertTrue(hasattr(model, "adaln_zero"))

    def test_model_uses_paper_output_head_initialization(self) -> None:
        from planners.gc_idm_model import GCIDMModel, GCIDMModelConfig

        torch.manual_seed(0)
        model = GCIDMModel(GCIDMModelConfig(latent_dim=4, action_dim=2, hidden_dim=512))

        self.assertTrue(hasattr(model.adaln_zero, "norm"))
        self.assertLess(float(model.action_head.weight.std()), 0.02)
        self.assertTrue(torch.allclose(model.action_head.bias, torch.zeros_like(model.action_head.bias)))
        self.assertTrue(torch.allclose(model.adaln_zero.gamma.weight, torch.zeros_like(model.adaln_zero.gamma.weight)))

    def test_bundle_round_trips_model_config_and_weights(self) -> None:
        from planners.gc_idm_model import (
            GCIDMModel,
            GCIDMModelConfig,
            load_gc_idm_bundle,
            save_gc_idm_bundle,
        )

        model = GCIDMModel(GCIDMModelConfig(latent_dim=3, action_dim=1, hidden_dim=16)).eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gc_idm_bundle.pt"
            save_gc_idm_bundle(model, path, metadata={"task": "fake"})
            bundle = load_gc_idm_bundle(path)

        self.assertEqual(bundle.latent_dim, 3)
        self.assertEqual(bundle.action_dim, 1)
        self.assertEqual(bundle.metadata["task"], "fake")
        restored = bundle.instantiate_model().eval()
        z_cur = torch.randn(2, 3)
        z_goal = torch.randn(2, 3)
        horizon = torch.ones(2, dtype=torch.long)
        self.assertTrue(torch.allclose(model(z_cur, z_goal, horizon), restored(z_cur, z_goal, horizon)))


class GCIDMPolicyTests(unittest.TestCase):
    def test_policy_recomputes_action_every_step_and_outputs_single_action(self) -> None:
        from planners.gc_idm_model import GCIDMModel, GCIDMModelConfig
        from planners.gc_idm_policy import GCIDMPolicy

        class FakeWorldModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros(()))
                self.encode_calls = 0

            def encode(self, info):
                self.encode_calls += 1
                pixels = info["pixels"].float()
                return {"emb": pixels[:, :, :, 0, 0]}

        class FakeEnv:
            num_envs = 2

            class ActionSpace:
                shape = (2, 1)

            action_space = ActionSpace()

        model = GCIDMModel(GCIDMModelConfig(latent_dim=3, action_dim=1, hidden_dim=16))
        world_model = FakeWorldModel()
        policy = GCIDMPolicy(
            world_model=world_model,
            planner=model,
            goal_offset_steps=5,
            eval_budget=10,
        )
        policy.set_env(FakeEnv())

        info = {
            "pixels": np.zeros((2, 1, 3, 1, 1), dtype=np.float32),
            "goal": np.ones((2, 1, 3, 1, 1), dtype=np.float32),
        }
        first = policy.get_action(info)
        second = policy.get_action(info)

        self.assertEqual(first.shape, (2, 1))
        self.assertEqual(second.shape, (2, 1))
        self.assertEqual(policy._num_replans, 2)
        self.assertEqual(world_model.encode_calls, 3)

    def test_policy_conditions_on_remaining_eval_budget(self) -> None:
        from planners.gc_idm_model import GCIDMModel, GCIDMModelConfig
        from planners.gc_idm_policy import GCIDMPolicy

        class FakeWorldModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros(()))

            def encode(self, info):
                pixels = info["pixels"].float()
                return {"emb": pixels[:, :, :, 0, 0]}

        class RecordingPlanner(GCIDMModel):
            def __init__(self) -> None:
                super().__init__(GCIDMModelConfig(latent_dim=3, action_dim=1, hidden_dim=16, max_horizon=50))
                self.recorded_horizons = []

            def forward(self, z_cur, z_goal, horizon):
                self.recorded_horizons.append(horizon.detach().cpu().clone())
                return torch.zeros((z_cur.shape[0], self.action_dim), dtype=z_cur.dtype, device=z_cur.device)

        class FakeEnv:
            num_envs = 2

            class ActionSpace:
                shape = (2, 1)

            action_space = ActionSpace()

        planner = RecordingPlanner()
        policy = GCIDMPolicy(
            world_model=FakeWorldModel(),
            planner=planner,
            goal_offset_steps=5,
            eval_budget=10,
        )
        policy.set_env(FakeEnv())

        info = {
            "pixels": np.zeros((2, 1, 3, 1, 1), dtype=np.float32),
            "goal": np.ones((2, 1, 3, 1, 1), dtype=np.float32),
        }
        policy.get_action(info)
        policy.get_action(info)

        self.assertEqual([h.tolist() for h in planner.recorded_horizons], [[10, 10], [9, 9]])


class GCIDMDatasetTests(unittest.TestCase):
    def test_training_defaults_match_gc_idm_paper(self) -> None:
        from planners.gc_idm_training import build_cosine_scheduler, parse_args

        args = parse_args(["--dataset-path", "/tmp/data.pt", "--output-dir", "/tmp/gc_idm"])
        param = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW([param], lr=float(args.lr), weight_decay=float(args.weight_decay))
        scheduler = build_cosine_scheduler(optimizer, epochs=int(args.epochs), lr=float(args.lr))

        self.assertEqual(args.epochs, 50)
        self.assertEqual(args.batch_size, 1024)
        self.assertEqual(args.val_batch_size, 1024)
        self.assertEqual(args.lr, 1e-3)
        self.assertEqual(args.weight_decay, 1e-4)
        self.assertEqual(args.dropout, 0.1)
        self.assertEqual(scheduler.eta_min, 1e-5)
        self.assertEqual(args.checkpoint_selection, "last")

    def test_episode_split_keeps_whole_episodes_disjoint(self) -> None:
        from planners.gc_idm_training import GCIDMTensorDataset, split_dataset

        dataset = GCIDMTensorDataset(
            {
                "z_cur": torch.randn(8, 3),
                "z_goal": torch.randn(8, 3),
                "horizon": torch.ones(8, dtype=torch.long),
                "action": torch.randn(8, 2),
                "meta": [
                    {"episode_id": 0},
                    {"episode_id": 0},
                    {"episode_id": 1},
                    {"episode_id": 1},
                    {"episode_id": 2},
                    {"episode_id": 2},
                    {"episode_id": 3},
                    {"episode_id": 3},
                ],
            }
        )

        train_dataset, val_dataset = split_dataset(dataset, val_split=0.25, seed=1, split_by_episode=True)

        self.assertIsInstance(train_dataset, Subset)
        self.assertIsInstance(val_dataset, Subset)
        train_eps = {dataset.meta[i]["episode_id"] for i in train_dataset.indices}
        val_eps = {dataset.meta[i]["episode_id"] for i in val_dataset.indices}
        self.assertTrue(train_eps)
        self.assertTrue(val_eps)
        self.assertTrue(train_eps.isdisjoint(val_eps))

    def test_dataset_wrapper_exposes_horizon_and_single_step_action(self) -> None:
        from planners.gc_idm_training import GCIDMTensorDataset

        dataset = GCIDMTensorDataset(
            {
                "z_cur": torch.randn(4, 3),
                "z_goal": torch.randn(4, 3),
                "horizon": torch.tensor([1, 2, 3, 4]),
                "action": torch.randn(4, 2),
                "meta": [{"i": i} for i in range(4)],
            }
        )

        sample = dataset[2]

        self.assertEqual(tuple(sample["z_cur"].shape), (3,))
        self.assertEqual(tuple(sample["action"].shape), (2,))
        self.assertEqual(int(sample["horizon"]), 3)
        self.assertEqual(sample["meta"]["i"], 2)

    def test_progress_iterator_is_disabled_for_non_tty_output(self) -> None:
        from planners.gc_idm_training import progress_iter

        values = [1, 2, 3]

        with mock.patch("planners.gc_idm_training.sys.stderr.isatty", return_value=False):
            wrapped = progress_iter(values, desc="train")

        self.assertIs(wrapped, values)

    def test_progress_iterator_uses_tqdm_for_tty_output(self) -> None:
        from planners.gc_idm_training import progress_iter

        calls = []

        def fake_tqdm(iterable, **kwargs):
            calls.append((iterable, kwargs))
            return ("wrapped", iterable)

        values = [1, 2, 3]
        with (
            mock.patch("planners.gc_idm_training.sys.stderr.isatty", return_value=True),
            mock.patch("planners.gc_idm_training._tqdm", fake_tqdm),
        ):
            wrapped = progress_iter(values, desc="gc-idm", total=3, unit="batch")

        self.assertEqual(wrapped, ("wrapped", values))
        self.assertEqual(calls[0][0], values)
        self.assertEqual(calls[0][1]["desc"], "gc-idm")
        self.assertEqual(calls[0][1]["total"], 3)
        self.assertEqual(calls[0][1]["unit"], "batch")


class GCIDMBuilderTests(unittest.TestCase):
    def test_batch_loader_uses_horizon_as_exact_goal_offset(self) -> None:
        from scripts.build_gc_idm_dataset import load_gc_idm_batch

        class FakeDataset:
            def load_chunk(self, episodes_idx, start_steps, end_steps):
                return [
                    {
                        "pixels": torch.arange(6, dtype=torch.float32).view(6, 1, 1, 1),
                        "action": torch.arange(12, dtype=torch.float32).view(6, 2),
                    }
                ]

        class TaskSpec:
            pixels_key = "pixels"
            action_key = "action"

        batch = load_gc_idm_batch(
            dataset=FakeDataset(),
            task_spec=TaskSpec(),
            episodes_idx=np.array([0]),
            start_steps=np.array([0]),
            horizons=np.array([3]),
        )

        self.assertEqual(float(batch["pixels"][0, 0, 0, 0, 0]), 0.0)
        self.assertEqual(float(batch["goal"][0, 0, 0, 0, 0]), 3.0)
        self.assertEqual(batch["action"][0, 0].tolist(), [0.0, 1.0])

    def test_world_model_builder_enables_eval_encoder_options(self) -> None:
        from scripts.build_gc_idm_dataset import configure_gc_idm_world_model_for_encoding

        class FakeModel:
            pass

        model = FakeModel()
        configured = configure_gc_idm_world_model_for_encoding(model)

        self.assertIs(configured, model)
        self.assertTrue(model.interpolate_pos_encoding)


if __name__ == "__main__":
    unittest.main()
