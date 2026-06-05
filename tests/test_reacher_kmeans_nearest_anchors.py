from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch


class ReacherKMeansNearestAnchorTests(unittest.TestCase):
    def test_selects_real_teacher_chunks_near_kmeans_centroids(self) -> None:
        from scripts.build_reacher_kmeans_nearest_anchors import fit_reacher_kmeans_nearest_anchors

        teacher_plan = torch.tensor(
            [
                [10.0, 0.0],
                [12.0, 0.0],
                [-10.0, 0.0],
                [-12.0, 0.0],
                [0.0, 10.0],
                [0.0, 12.0],
            ],
            dtype=torch.float32,
        )

        result = fit_reacher_kmeans_nearest_anchors(
            teacher_plan,
            num_anchors=3,
            seed=0,
            max_iter=50,
        )

        self.assertEqual(tuple(result.anchors.shape), (3, 2))
        teacher_rows = {tuple(row.tolist()) for row in teacher_plan}
        anchor_rows = {tuple(row.tolist()) for row in result.anchors}
        self.assertTrue(anchor_rows.issubset(teacher_rows))
        self.assertEqual(result.fit_method, "kmeans_nearest_real_sample")
        self.assertIn("centroid_norm_mean", result.metadata)
        self.assertIn("real_anchor_norm_mean", result.metadata)
        self.assertIn("centroid_to_real_l2_mean", result.metadata)
        self.assertGreater(
            result.metadata["centroid_to_real_l2_mean"],
            0.0,
        )

    def test_cli_writes_standard_anchor_bundle_for_reacher_dataset(self) -> None:
        from diffusion.anchors import load_anchor_bundle
        from scripts.build_reacher_kmeans_nearest_anchors import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "reacher_planner_dataset.pt"
            output_path = root / "reacher_action_anchors_k3_kmeans_nearest.pt"
            teacher_plan = torch.tensor(
                [
                    [10.0, 0.0],
                    [12.0, 0.0],
                    [-10.0, 0.0],
                    [-12.0, 0.0],
                    [0.0, 10.0],
                    [0.0, 12.0],
                ],
                dtype=torch.float32,
            )
            torch.save(
                {
                    "teacher_plan": teacher_plan,
                    "meta": [
                        {
                            "task": "reacher",
                            "plan_horizon": 1,
                            "action_dim": 2,
                            "action_chunk_dim": 2,
                        }
                        for _ in range(int(teacher_plan.shape[0]))
                    ],
                    "build_info": {
                        "task": "reacher",
                        "action_dim": 2,
                        "action_chunk_horizon": 1,
                        "action_chunk_dim": 2,
                        "plan_config": {"receding_horizon": 1, "action_block": 1},
                    },
                },
                dataset_path,
            )

            main(
                [
                    "--dataset-path",
                    str(dataset_path),
                    "--output-path",
                    str(output_path),
                    "--num-anchors",
                    "3",
                    "--max-samples",
                    "6",
                    "--seed",
                    "0",
                    "--max-iter",
                    "50",
                ]
            )

            bundle = load_anchor_bundle(output_path)
            self.assertEqual(bundle.task, "reacher")
            self.assertEqual(bundle.fit_method, "kmeans_nearest_real_sample")
            self.assertEqual(tuple(bundle.anchors.shape), (3, 2))
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
