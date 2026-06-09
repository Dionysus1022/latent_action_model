import numpy as np
import torch

from trajectory_quality import (
    compute_latent_monotonicity,
    compute_task_goal_distances,
    compute_trajectory_quality,
)


def test_compute_trajectory_quality_reports_path_and_action_smoothness():
    states = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [6.0, 0.0]],
        ],
        dtype=np.float32,
    )
    goal_distances = np.array([[6.0, 5.0, 3.0, 0.0]], dtype=np.float32)
    actions = np.array(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
        ],
        dtype=np.float32,
    )
    successes = np.array([[False, False, False, True]])

    summary, per_episode = compute_trajectory_quality(
        states=states,
        actions=actions,
        goal_distances=goal_distances,
        successes_by_step=successes,
    )

    assert np.isclose(per_episode["path_length"][0], 6.0)
    assert np.isclose(per_episode["straight_line_distance"][0], 6.0)
    assert np.isclose(per_episode["straight_line_ratio"][0], 1.0)
    assert np.isclose(per_episode["final_goal_distance"][0], 0.0)
    assert np.isclose(per_episode["min_goal_distance"][0], 0.0)
    assert np.isclose(per_episode["action_l2_mean"][0], 2.5)
    assert np.isclose(per_episode["action_delta_l2_mean"][0], 1.0)
    assert np.isclose(per_episode["action_jerk_l2_mean"][0], 0.0)
    assert per_episode["steps_to_success"][0] == 3
    assert np.isclose(summary["path_length_mean"], 6.0)
    assert np.isclose(summary["steps_to_success_mean"], 3.0)


def test_compute_trajectory_quality_ignores_padding_after_termination():
    states = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [100.0, 100.0], [200.0, 200.0]],
        ],
        dtype=np.float32,
    )
    goal_distances = np.array([[1.0, 0.0, 99.0, 199.0]], dtype=np.float32)
    actions = np.ones((1, 4, 2), dtype=np.float32)
    successes = np.array([[False, True, False, False]])

    summary, per_episode = compute_trajectory_quality(
        states=states,
        actions=actions,
        goal_distances=goal_distances,
        successes_by_step=successes,
        truncate_after_success=True,
    )

    assert np.isclose(per_episode["path_length"][0], 1.0)
    assert np.isclose(per_episode["final_goal_distance"][0], 0.0)
    assert np.isclose(per_episode["min_goal_distance"][0], 0.0)
    assert per_episode["effective_length"][0] == 2
    assert np.isclose(summary["path_length_mean"], 1.0)


def test_compute_latent_monotonicity_reports_fraction_of_goal_direct_steps():
    latents = np.array(
        [
            [[3.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            [[3.0, 0.0], [4.0, 0.0], [2.0, 0.0], [1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    goal = np.zeros((2, 2), dtype=np.float32)

    summary, per_episode = compute_latent_monotonicity(latents=latents, goal_latents=goal)

    assert np.allclose(per_episode["latent_monotonicity"], [1.0, 0.0])
    assert np.allclose(per_episode["latent_monotonic_step_fraction"], [1.0, 2.0 / 3.0])
    assert np.isclose(summary["latent_monotonicity_mean"], 0.5)
    assert np.isclose(summary["latent_monotonic_step_fraction_mean"], (1.0 + 2.0 / 3.0) / 2.0)
    assert np.allclose(per_episode["final_latent_goal_distance"], [0.0, 1.0])


def test_compute_task_goal_distances_for_reacher_and_tworoom():
    reacher_state = {
        "qpos": np.array([[[0.0, 0.0], [0.5, 0.0]]], dtype=np.float32),
        "goal_qpos": np.array([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32),
    }
    reacher = compute_task_goal_distances("reacher", reacher_state)
    assert np.allclose(reacher["goal_distance"], [[1.0, 0.5]])

    tworoom_state = {
        "proprio": np.array([[[0.0, 0.0], [3.0, 4.0]]], dtype=np.float32),
        "goal_proprio": np.array([[[0.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
    }
    tworoom = compute_task_goal_distances("tworoom", tworoom_state)
    assert np.allclose(tworoom["goal_distance"], [[0.0, 5.0]])


def test_compute_task_goal_distances_for_pusht_and_cube():
    pusht_state = {
        "state": np.array(
            [[[0.0, 0.0, 0.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0, np.pi]]],
            dtype=np.float32,
        ),
        "goal_state": np.array(
            [[[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -np.pi]]],
            dtype=np.float32,
        ),
    }
    pusht = compute_task_goal_distances("pusht", pusht_state)
    assert np.allclose(pusht["pusht_pos_distance"], [[0.0, 5.0]])
    assert np.allclose(pusht["pusht_angle_distance"], [[0.0, 0.0]], atol=1e-6)

    cube_state = {
        "privileged/block_0_pos": np.array(
            [[[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]]],
            dtype=np.float32,
        ),
        "goal_privileged_block_0_pos": np.array(
            [[[0.0, 0.0, 0.0], [1.0, 2.0, 5.0]]],
            dtype=np.float32,
        ),
    }
    cube = compute_task_goal_distances("cube", cube_state)
    assert np.allclose(cube["goal_distance"], [[0.0, 3.0]])


class _FakeDataset:
    column_names = ["pixels", "action", "proprio"]

    def load_chunk(self, episodes_idx, start, end):
        chunks = []
        for _ep, _start, _end in zip(episodes_idx, start, end):
            chunks.append(
                {
                    "pixels": torch.zeros((3, 3, 2, 2), dtype=torch.uint8),
                    "action": torch.zeros((2, 2), dtype=torch.float32),
                    "proprio": torch.tensor(
                        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                        dtype=torch.float32,
                    ),
                }
            )
        return chunks


class _FakeUnwrapped:
    _autoreset_envs = None

    def __init__(self, envs):
        self.envs = envs


class _FakeEnv:
    @property
    def unwrapped(self):
        return self

    def _set_state(self, state):
        self.state = np.asarray(state, dtype=np.float32)

    def _set_goal_state(self, goal_state):
        self.goal_state = np.asarray(goal_state, dtype=np.float32)


class _FakeVectorEnv:
    def __init__(self, envs):
        self.unwrapped = _FakeUnwrapped(envs)


class _FakeWorld:
    num_envs = 1

    def __init__(self):
        self.envs = _FakeVectorEnv([_FakeEnv()])
        self.step_index = 0
        self.terminateds = np.array([False])
        self.infos = {}

    def reset(self, seed=None, options=None):
        del seed, options
        self.step_index = 0
        self.infos = {
            "pixels": np.zeros((1, 1, 2, 2, 3), dtype=np.uint8),
            "action": np.zeros((1, 1, 2), dtype=np.float32),
            "proprio": np.zeros((1, 1, 2), dtype=np.float32),
        }

    def step(self):
        self.step_index += 1
        position = np.array([[float(self.step_index), 0.0]], dtype=np.float32)
        self.infos = {
            "pixels": np.zeros((1, 1, 2, 2, 3), dtype=np.uint8),
            "action": np.ones((1, 1, 2), dtype=np.float32),
            "proprio": position[:, None, :],
            "goal_proprio": np.array([[[2.0, 0.0]]], dtype=np.float32),
        }
        self.terminateds = np.array([self.step_index >= 2])


def test_eval_trajectory_quality_loop_records_metrics_without_real_env(tmp_path):
    from eval import run_evaluation_with_trajectory_quality

    metrics, quality = run_evaluation_with_trajectory_quality(
        world=_FakeWorld(),
        dataset=_FakeDataset(),
        episodes_idx=[0],
        start_steps=[0],
        goal_offset_steps=2,
        eval_budget=3,
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "proprio"}}},
            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_proprio"}}},
        ],
        video_path=tmp_path,
        task="tworoom",
        quality_cfg={
            "enabled": True,
            "save_npz": False,
            "truncate_after_success": True,
            "save_video": False,
        },
    )

    assert metrics["success_rate"] == 100.0
    assert metrics["episode_successes"].tolist() == [True]
    assert np.isclose(quality["summary"]["final_goal_distance_mean"], 0.0)
    assert np.isclose(quality["summary"]["path_length_mean"], 1.0)


def test_sample_eval_episode_starts_uses_episode_metadata_only():
    from eval import sample_eval_episode_starts

    class Dataset:
        lengths = np.array([3, 5, 7], dtype=np.int64)
        offsets = np.array([0, 3, 8], dtype=np.int64)

        def get_col_data(self, _key):
            raise AssertionError("sampling should not scan full dataset columns")

    episodes, starts, valid_count = sample_eval_episode_starts(
        dataset=Dataset(),
        ep_indices=np.array([10, 11, 12], dtype=np.int64),
        goal_offset_steps=2,
        num_eval=3,
        seed=0,
    )

    assert valid_count == 9
    assert episodes.shape == (3,)
    assert starts.shape == (3,)
    for episode, start in zip(episodes, starts):
        episode_pos = {10: 0, 11: 1, 12: 2}[int(episode)]
        assert 0 <= int(start) <= int(Dataset.lengths[episode_pos] - 3)
