import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from planners.latent_rollout import latent_rollout
from diffusion.model import (
    DiffusionPlannerModel,
    DiffusionPlannerModelConfig,
    load_diffusion_planner_bundle,
    save_diffusion_planner_bundle,
)
from diffusion.train import (
    apply_loss_preset,
    configure_trainable_parameters,
    compute_batch_losses,
    initialize_model_from_bundle,
    parse_args,
    reshape_flat_actions_to_rollout_blocks,
)


def make_dit_model() -> DiffusionPlannerModel:
    config = DiffusionPlannerModelConfig(
        latent_dim=5,
        plan_horizon=4,
        action_dim=2,
        num_anchors=3,
        hidden_dim=16,
        num_layers=2,
        timestep_embedding_dim=8,
        fusion_num_layers=1,
        denoiser_type="dit",
        dit_num_layers=2,
        dit_num_heads=4,
        dit_mlp_ratio=2.0,
        num_train_steps=8,
        truncation_steps=2,
        beta_end=1e-2,
    )
    anchors = torch.zeros(config.num_anchors, config.action_chunk_dim)
    return DiffusionPlannerModel(config, anchors)


def make_loss_args(**overrides) -> SimpleNamespace:
    values = dict(
        cls_loss_type="bce",
        cls_loss_weight=0.0,
        bce_weight=1.0,
        bce_pos_topk=1,
        rec_loss_weight=1.0,
        goal_loss_weight=0.0,
        enable_goal_pool_loss=False,
        goal_pool_topk=1,
        goal_pool_candidate_source="nearest",
        goal_loss_history_size=1,
        goal_pool_tau=1.0,
        goal_pool_weight=0.0,
        aux_rec_topk=1,
        aux_rec_weight=0.0,
        aux_rec_temperature=1.0,
        rec_loss="smooth_l1",
        score_ranking_weight=0.0,
        score_ranking_temperature=1.0,
        wm_score_ranking_weight=0.0,
        wm_score_ranking_temperature=1.0,
        wm_score_target_mode="softmax",
        wm_score_candidate_source="single_step",
        wm_score_regression_normalize=True,
        wm_score_ce_temperature=1.0,
        wm_score_topk_margin_k=16,
        wm_score_topk_margin=0.1,
        wm_score_topk_margin_weight=0.0,
        diversity_weight=0.0,
        wm_rank_weight=0.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class CriterionWorldModel(torch.nn.Module):
    def __init__(self, action_dim: int, latent_dim: int):
        super().__init__()
        self.action_encoder = torch.nn.Linear(action_dim, latent_dim, bias=False)
        with torch.no_grad():
            self.action_encoder.weight.fill_(0.25)

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        del emb
        return act_emb

    def criterion(self, info_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]
        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
        return F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))


class DiffusionDiTModelTest(unittest.TestCase):
    def test_mlp_score_head_forward_and_freeze_only_trains_score_head(self):
        config = DiffusionPlannerModelConfig(
            latent_dim=5,
            plan_horizon=4,
            action_dim=2,
            num_anchors=3,
            hidden_dim=16,
            num_layers=2,
            timestep_embedding_dim=8,
            fusion_num_layers=1,
            denoiser_type="dit",
            dit_num_layers=2,
            dit_num_heads=4,
            dit_mlp_ratio=2.0,
            num_train_steps=8,
            truncation_steps=2,
            beta_end=1e-2,
            score_head_type="mlp",
            score_head_hidden_dim=32,
            score_head_num_layers=2,
        )
        model = DiffusionPlannerModel(
            config,
            torch.zeros(config.num_anchors, config.action_chunk_dim),
        )
        z_cur = torch.randn(2, model.latent_dim)
        z_goal = torch.randn(2, model.latent_dim)
        noisy = torch.randn(2, model.num_anchors, model.action_chunk_dim)
        timesteps = torch.full((2, model.num_anchors), 7, dtype=torch.long)

        out = model(z_cur, z_goal, noisy, timesteps)
        trainable = configure_trainable_parameters(model, freeze_non_score_head=True)
        trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]

        self.assertEqual(tuple(out["score_logits"].shape), (2, 3))
        self.assertIsInstance(model.score_head, torch.nn.Sequential)
        self.assertGreater(sum(parameter.numel() for parameter in trainable), 17)
        self.assertTrue(trainable_names)
        self.assertTrue(all(name.startswith("score_head.") for name in trainable_names))

    def test_mlp_score_head_can_initialize_main_weights_from_linear_score_bundle(self):
        linear_model = make_dit_model()
        with TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "linear_bundle.pt"
            save_diffusion_planner_bundle(linear_model, bundle_path)
            mlp_config = DiffusionPlannerModelConfig(
                **{
                    **linear_model.config.__dict__,
                    "score_head_type": "mlp",
                    "score_head_hidden_dim": 32,
                    "score_head_num_layers": 2,
                }
            )
            mlp_model = DiffusionPlannerModel(
                mlp_config,
                linear_model.anchors.detach().clone(),
            )

            initialize_model_from_bundle(
                mlp_model,
                bundle_path,
                device=torch.device("cpu"),
            )

        self.assertTrue(
            torch.allclose(
                mlp_model.step_action_head.weight,
                linear_model.step_action_head.weight,
            )
        )
        self.assertIsInstance(mlp_model.score_head, torch.nn.Sequential)

    def test_dit_forward_and_generation_shapes_match_mlp_contract(self):
        model = make_dit_model()
        z_cur = torch.randn(2, model.latent_dim)
        z_goal = torch.randn(2, model.latent_dim)
        noisy = torch.randn(2, model.num_anchors, model.action_chunk_dim)
        timesteps = torch.full((2, model.num_anchors), 7, dtype=torch.long)

        out = model(z_cur, z_goal, noisy, timesteps)
        generated = model.generate_candidates(z_cur, z_goal, truncation_steps=2)

        self.assertEqual(tuple(out["refined_actions"].shape), (2, 3, 8))
        self.assertEqual(tuple(out["score_logits"].shape), (2, 3))
        self.assertEqual(tuple(generated["candidates"].shape), (2, 3, 8))
        self.assertEqual(tuple(generated["score_logits"].shape), (2, 3))

    def test_dit_bundle_round_trips(self):
        model = make_dit_model()

        with TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "dit_bundle.pt"
            save_diffusion_planner_bundle(model, bundle_path)
            restored = load_diffusion_planner_bundle(bundle_path).instantiate_model()

        self.assertEqual(restored.config.denoiser_type, "dit")
        self.assertEqual(restored.config.dit_num_layers, 2)
        self.assertEqual(restored.config.dit_num_heads, 4)
        self.assertIsInstance(restored, DiffusionPlannerModel)

    def test_dit_compute_loss_batch_backpropagates(self):
        model = make_dit_model()
        batch = {
            "z_cur": torch.randn(4, model.latent_dim),
            "z_goal": torch.randn(4, model.latent_dim),
            "teacher_plan": torch.randn(4, model.action_chunk_dim),
        }
        args = make_loss_args()
        timestep_grid = torch.full((4, model.num_anchors), 7, dtype=torch.long)

        losses, _ = compute_batch_losses(
            model=model,
            batch=batch,
            rec_loss_fn=torch.nn.SmoothL1Loss(),
            args=args,
            device=torch.device("cpu"),
            timestep_grid=timestep_grid,
        )

        self.assertTrue(torch.isfinite(losses["total_loss"]))
        losses["total_loss"].backward()
        grad_norm = sum(
            float(parameter.grad.abs().sum().item())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

    def test_wm_score_ranking_loss_uses_frozen_world_model_cost_targets(self):
        model = make_dit_model()
        batch = {
            "z_cur": torch.randn(2, model.latent_dim),
            "z_goal": torch.randn(2, model.latent_dim),
            "teacher_plan": torch.randn(2, model.action_chunk_dim),
        }
        args = make_loss_args(
            bce_weight=0.0,
            rec_loss_weight=0.0,
            wm_score_ranking_weight=1.0,
            wm_score_ranking_temperature=0.5,
        )
        timestep_grid = torch.full((2, model.num_anchors), 7, dtype=torch.long)
        wm_costs = torch.tensor(
            [
                [3.0, 1.0, 2.0],
                [0.5, 4.0, 2.0],
            ],
            dtype=torch.float32,
        )

        losses, metrics = compute_batch_losses(
            model=model,
            batch=batch,
            rec_loss_fn=torch.nn.SmoothL1Loss(),
            args=args,
            device=torch.device("cpu"),
            timestep_grid=timestep_grid,
            wm_candidate_costs=wm_costs,
        )

        expected_targets = torch.softmax(-wm_costs / args.wm_score_ranking_temperature, dim=-1)
        self.assertGreater(float(metrics["wm_score_ranking_loss"]), 0.0)
        self.assertTrue(torch.isfinite(losses["wm_score_ranking_loss"]))
        self.assertTrue(torch.isfinite(losses["total_loss"]))
        self.assertGreaterEqual(metrics["wm_score_best_target_index_acc"], 0.0)
        self.assertLessEqual(metrics["wm_score_best_target_index_acc"], 1.0)
        self.assertEqual(expected_targets.argmax(dim=-1).tolist(), [1, 0])

    def test_wm_score_regression_loss_directly_fits_negative_cost(self):
        model = make_dit_model()
        batch = {
            "z_cur": torch.randn(2, model.latent_dim),
            "z_goal": torch.randn(2, model.latent_dim),
            "teacher_plan": torch.randn(2, model.action_chunk_dim),
        }
        args = make_loss_args(
            bce_weight=0.0,
            rec_loss_weight=0.0,
            wm_score_ranking_weight=1.0,
            wm_score_target_mode="neg_cost",
            wm_score_regression_normalize=True,
        )
        timestep_grid = torch.full((2, model.num_anchors), 7, dtype=torch.long)
        noise = torch.zeros(2, model.num_anchors, model.action_chunk_dim)
        wm_costs = torch.tensor(
            [
                [3.0, 1.0, 2.0],
                [0.5, 4.0, 2.0],
            ],
            dtype=torch.float32,
        )

        noisy_candidates, _ = model.initialize_noisy_candidates(
            batch_size=2,
            device=torch.device("cpu"),
            dtype=batch["z_cur"].dtype,
            timesteps=timestep_grid,
            noise=noise,
        )
        score_logits = model(
            batch["z_cur"],
            batch["z_goal"],
            noisy_candidates,
            timestep_grid,
        )["score_logits"]
        expected_scores = F.layer_norm(score_logits, normalized_shape=(model.num_anchors,))
        expected_targets = F.layer_norm(-wm_costs, normalized_shape=(model.num_anchors,))
        expected_loss = F.mse_loss(expected_scores, expected_targets)

        losses, metrics = compute_batch_losses(
            model=model,
            batch=batch,
            rec_loss_fn=torch.nn.SmoothL1Loss(),
            args=args,
            device=torch.device("cpu"),
            timestep_grid=timestep_grid,
            noise_override=noise,
            wm_candidate_costs=wm_costs,
        )

        self.assertTrue(torch.allclose(losses["wm_score_ranking_loss"], expected_loss))
        self.assertTrue(torch.allclose(losses["total_loss"], expected_loss))
        self.assertAlmostEqual(metrics["wm_score_ranking_loss"], float(expected_loss.item()))

    def test_wm_score_argmin_ce_topk_margin_targets_best_cost_candidate(self):
        model = make_dit_model()
        batch = {
            "z_cur": torch.randn(2, model.latent_dim),
            "z_goal": torch.randn(2, model.latent_dim),
            "teacher_plan": torch.randn(2, model.action_chunk_dim),
        }
        args = make_loss_args(
            bce_weight=0.0,
            rec_loss_weight=0.0,
            wm_score_ranking_weight=1.0,
            wm_score_target_mode="argmin_ce_topk_margin",
            wm_score_ce_temperature=0.5,
            wm_score_topk_margin_k=2,
            wm_score_topk_margin=0.25,
            wm_score_topk_margin_weight=0.75,
        )
        timestep_grid = torch.full((2, model.num_anchors), 7, dtype=torch.long)
        noise = torch.zeros(2, model.num_anchors, model.action_chunk_dim)
        wm_costs = torch.tensor(
            [
                [3.0, 1.0, 2.0],
                [0.5, 4.0, 2.0],
            ],
            dtype=torch.float32,
        )

        noisy_candidates, _ = model.initialize_noisy_candidates(
            batch_size=2,
            device=torch.device("cpu"),
            dtype=batch["z_cur"].dtype,
            timesteps=timestep_grid,
            noise=noise,
        )
        score_logits = model(
            batch["z_cur"],
            batch["z_goal"],
            noisy_candidates,
            timestep_grid,
        )["score_logits"]
        best_indices = torch.argmin(wm_costs, dim=-1)
        ce_loss = F.cross_entropy(score_logits / args.wm_score_ce_temperature, best_indices)
        best_scores = score_logits.gather(1, best_indices.view(-1, 1)).squeeze(1)
        negative_scores = score_logits.masked_fill(
            F.one_hot(best_indices, num_classes=model.num_anchors).bool(),
            float("-inf"),
        )
        boundary = torch.topk(
            negative_scores,
            k=min(args.wm_score_topk_margin_k, model.num_anchors - 1),
            dim=-1,
        ).values[:, -1]
        margin_loss = F.relu(args.wm_score_topk_margin - best_scores + boundary).mean()
        expected_loss = ce_loss + args.wm_score_topk_margin_weight * margin_loss
        topk_indices = torch.topk(
            score_logits,
            k=min(args.wm_score_topk_margin_k, model.num_anchors),
            dim=-1,
        ).indices
        expected_topk_acc = topk_indices.eq(best_indices.view(-1, 1)).any(dim=-1).float().mean()

        losses, metrics = compute_batch_losses(
            model=model,
            batch=batch,
            rec_loss_fn=torch.nn.SmoothL1Loss(),
            args=args,
            device=torch.device("cpu"),
            timestep_grid=timestep_grid,
            noise_override=noise,
            wm_candidate_costs=wm_costs,
        )

        self.assertTrue(torch.allclose(losses["wm_score_ranking_loss"], expected_loss))
        self.assertTrue(torch.allclose(losses["total_loss"], expected_loss))
        self.assertAlmostEqual(metrics["wm_score_ranking_loss"], float(expected_loss.item()))
        self.assertAlmostEqual(
            metrics["wm_score_best_target_topk_acc"],
            float(expected_topk_acc.item()),
        )
        self.assertEqual(best_indices.tolist(), [1, 0])

    def test_wm_score_topk_margin_preset_sets_argmin_ce_defaults(self):
        args = parse_args(
            [
                "--dataset-path",
                "dataset.pt",
                "--anchor-bundle-path",
                "anchors.pt",
                "--output-dir",
                "out",
                "--loss-preset",
                "wm_score_topk_margin",
            ]
        )

        flags = apply_loss_preset(args)

        self.assertEqual(args.cls_loss_type, "bce")
        self.assertEqual(args.cls_loss_weight, 0.0)
        self.assertEqual(args.bce_weight, 0.0)
        self.assertEqual(args.rec_loss_weight, 0.0)
        self.assertEqual(args.aux_rec_weight, 0.0)
        self.assertEqual(args.score_ranking_weight, 0.0)
        self.assertFalse(args.enable_goal_pool_loss)
        self.assertEqual(args.goal_pool_weight, 0.0)
        self.assertEqual(args.wm_score_target_mode, "argmin_ce_topk_margin")
        self.assertEqual(args.wm_score_candidate_source, "inference")
        self.assertEqual(args.wm_score_ranking_weight, 1.0)
        self.assertEqual(args.wm_score_topk_margin_k, 16)
        self.assertGreater(args.wm_score_topk_margin_weight, 0.0)
        self.assertTrue(flags["wm_rank_enabled"])

    def test_parse_args_accepts_mlp_score_head_options(self):
        args = parse_args(
            [
                "--dataset-path",
                "dataset.pt",
                "--anchor-bundle-path",
                "anchors.pt",
                "--output-dir",
                "out",
                "--score-head-type",
                "mlp",
                "--score-head-hidden-dim",
                "32",
                "--score-head-num-layers",
                "2",
            ]
        )

        self.assertEqual(args.score_head_type, "mlp")
        self.assertEqual(args.score_head_hidden_dim, 32)
        self.assertEqual(args.score_head_num_layers, 2)

    def test_wm_score_cost_uses_world_model_criterion_scale(self):
        model = make_dit_model()
        world_model = CriterionWorldModel(action_dim=model.action_dim, latent_dim=model.latent_dim)
        batch = {
            "z_cur": torch.zeros(2, model.latent_dim),
            "z_goal": torch.randn(2, model.latent_dim),
            "teacher_plan": torch.randn(2, model.action_chunk_dim),
        }
        args = make_loss_args(
            bce_weight=0.0,
            rec_loss_weight=0.0,
            wm_score_ranking_weight=1.0,
            wm_score_target_mode="softmax",
        )
        timestep_grid = torch.full((2, model.num_anchors), 7, dtype=torch.long)
        noise = torch.zeros(2, model.num_anchors, model.action_chunk_dim)

        noisy_candidates, _ = model.initialize_noisy_candidates(
            batch_size=2,
            device=torch.device("cpu"),
            dtype=batch["z_cur"].dtype,
            timesteps=timestep_grid,
            noise=noise,
        )
        refined_actions = model(
            batch["z_cur"],
            batch["z_goal"],
            noisy_candidates,
            timestep_grid,
        )["refined_actions"]
        action_blocks = reshape_flat_actions_to_rollout_blocks(
            refined_actions.detach(),
            plan_horizon=model.plan_horizon,
            action_dim=model.action_dim,
            receding_horizon=model.plan_horizon,
            action_block=1,
        )
        rollout = latent_rollout(
            world_model=world_model,
            z_context=batch["z_cur"],
            action_blocks=action_blocks,
            history_size=1,
            return_sequence=False,
            freeze_world_model=True,
        )
        expected_costs = world_model.criterion(
            {
                "predicted_emb": rollout["z_terminal"].unsqueeze(2),
                "goal_emb": batch["z_goal"].unsqueeze(1).unsqueeze(2).expand(
                    2,
                    model.num_anchors,
                    1,
                    model.latent_dim,
                ),
            }
        )

        _, metrics = compute_batch_losses(
            model=model,
            batch=batch,
            rec_loss_fn=torch.nn.SmoothL1Loss(),
            args=args,
            device=torch.device("cpu"),
            timestep_grid=timestep_grid,
            world_model=world_model,
            goal_loss_receding_horizon=model.plan_horizon,
            goal_loss_action_block=1,
            noise_override=noise,
        )

        self.assertAlmostEqual(metrics["wm_score_cost_mean"], float(expected_costs.mean().item()))

    def test_inference_wm_score_training_reuses_validation_noise_override(self):
        model = make_dit_model()
        batch = {
            "z_cur": torch.randn(2, model.latent_dim),
            "z_goal": torch.randn(2, model.latent_dim),
            "teacher_plan": torch.randn(2, model.action_chunk_dim),
        }
        args = make_loss_args(
            bce_weight=0.0,
            rec_loss_weight=0.0,
            wm_score_ranking_weight=1.0,
            wm_score_target_mode="neg_cost",
            wm_score_candidate_source="inference",
        )
        timestep_grid = torch.full((2, model.num_anchors), 7, dtype=torch.long)
        noise = torch.randn(2, model.num_anchors, model.action_chunk_dim)
        wm_costs = torch.tensor(
            [
                [3.0, 1.0, 2.0],
                [0.5, 4.0, 2.0],
            ],
            dtype=torch.float32,
        )
        captured: dict[str, torch.Tensor | None] = {}

        def fake_forward_inference_candidate_scores(
            model_arg: DiffusionPlannerModel,
            z_cur_arg: torch.Tensor,
            z_goal_arg: torch.Tensor,
            *,
            noise: torch.Tensor | None = None,
            eta: float = 0.0,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del model_arg, z_cur_arg, z_goal_arg, eta
            captured["noise"] = noise
            return (
                torch.zeros(2, model.num_anchors, model.action_chunk_dim),
                torch.zeros(2, model.num_anchors),
            )

        with patch(
            "diffusion.train.forward_inference_candidate_scores",
            side_effect=fake_forward_inference_candidate_scores,
        ):
            compute_batch_losses(
                model=model,
                batch=batch,
                rec_loss_fn=torch.nn.SmoothL1Loss(),
                args=args,
                device=torch.device("cpu"),
                timestep_grid=timestep_grid,
                noise_override=noise,
                wm_candidate_costs=wm_costs,
            )

        self.assertIs(captured["noise"], noise)


if __name__ == "__main__":
    unittest.main()
