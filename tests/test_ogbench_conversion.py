import subprocess
from pathlib import Path

import h5py
import numpy as np


def test_convert_ogbench_npz_to_hdf5_builds_episode_metadata(tmp_path: Path) -> None:
    input_npz = tmp_path / "toy.npz"
    output_h5 = tmp_path / "toy.h5"
    observations = np.arange(7 * 3, dtype=np.float32).reshape(7, 3)
    actions = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
    qpos = observations[:, :2]
    terminals = np.array([False, False, True, False, False, False, True])

    np.savez(
        input_npz,
        observations=observations,
        actions=actions,
        terminals=terminals,
        qpos=qpos,
    )

    subprocess.run(
        [
            "./.venv/bin/python",
            "scripts/convert_ogbench_npz_to_hdf5.py",
            "--input-npz",
            str(input_npz),
            "--output-h5",
            str(output_h5),
            "--dataset-name",
            "toy-ogbench",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    with h5py.File(output_h5, "r") as handle:
        assert handle.attrs["source_format"] == "ogbench_npz"
        assert handle.attrs["dataset_name"] == "toy-ogbench"
        assert handle.attrs["transition_policy"] == "ogbench_regular"
        np.testing.assert_array_equal(
            handle["observation"][:],
            np.vstack([observations[0:2], observations[3:6]]),
        )
        np.testing.assert_array_equal(
            handle["next_observation"][:],
            np.vstack([observations[1:3], observations[4:7]]),
        )
        np.testing.assert_array_equal(handle["action"][:], actions)
        np.testing.assert_array_equal(handle["qpos"][:], np.vstack([qpos[0:2], qpos[3:6]]))
        np.testing.assert_array_equal(handle["ep_len"][:], np.array([2, 3]))
        np.testing.assert_array_equal(handle["ep_offset"][:], np.array([0, 2]))
        np.testing.assert_array_equal(handle["ep_idx"][:], np.array([0, 0, 1, 1, 1]))
        np.testing.assert_array_equal(handle["step_idx"][:], np.array([0, 1, 0, 1, 2]))
        np.testing.assert_array_equal(handle["terminated"][:], np.array([False, True, False, False, True]))
        np.testing.assert_array_equal(handle["truncated"][:], np.zeros(5, dtype=bool))
        np.testing.assert_array_equal(handle["success"][:], np.array([False, True, False, False, True]))

        expected_target = np.vstack(
            [
                np.repeat(observations[2:3], 2, axis=0),
                np.repeat(observations[6:7], 3, axis=0),
            ]
        )
        np.testing.assert_array_equal(handle["target"][:], expected_target)


def test_convert_visual_ogbench_npz_writes_pixels_schema(tmp_path: Path) -> None:
    input_npz = tmp_path / "visual.npz"
    output_h5 = tmp_path / "visual.h5"
    observations = np.arange(6 * 4 * 4 * 3, dtype=np.uint8).reshape(6, 4, 4, 3)
    actions = np.arange(6 * 5, dtype=np.float32).reshape(6, 5)
    terminals = np.array([False, False, True, False, False, True])
    qvel = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)

    np.savez(
        input_npz,
        observations=observations,
        actions=actions,
        terminals=terminals,
        qvel=qvel,
    )

    subprocess.run(
        [
            "./.venv/bin/python",
            "scripts/convert_ogbench_npz_to_hdf5.py",
            "--input-npz",
            str(input_npz),
            "--output-h5",
            str(output_h5),
            "--dataset-name",
            "visual-toy",
            "--observation-output-key",
            "pixels",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    with h5py.File(output_h5, "r") as handle:
        assert handle.attrs["observation_output_key"] == "pixels"
        assert handle["pixels"].dtype == np.uint8
        np.testing.assert_array_equal(
            handle["pixels"][:],
            np.concatenate([observations[0:2], observations[3:5]], axis=0),
        )
        np.testing.assert_array_equal(
            handle["next_pixels"][:],
            np.concatenate([observations[1:3], observations[4:6]], axis=0),
        )
        np.testing.assert_array_equal(
            handle["target"][:],
            np.concatenate(
                [
                    np.repeat(observations[2:3], 2, axis=0),
                    np.repeat(observations[5:6], 2, axis=0),
                ],
                axis=0,
            ),
        )
        np.testing.assert_array_equal(handle["action"][:], actions[[0, 1, 3, 4]])
        np.testing.assert_array_equal(handle["qvel"][:], qvel[[0, 1, 3, 4]])
        assert "observation" not in handle
        assert "next_observation" not in handle


def test_convert_visual_ogbench_npz_can_store_target_indices_without_repeated_images(
    tmp_path: Path,
) -> None:
    input_npz = tmp_path / "visual_index.npz"
    output_h5 = tmp_path / "visual_index.h5"
    observations = np.arange(6 * 4 * 4 * 3, dtype=np.uint8).reshape(6, 4, 4, 3)
    actions = np.arange(6 * 5, dtype=np.float32).reshape(6, 5)
    terminals = np.array([False, False, True, False, False, True])

    np.savez(
        input_npz,
        observations=observations,
        actions=actions,
        terminals=terminals,
    )

    subprocess.run(
        [
            "./.venv/bin/python",
            "scripts/convert_ogbench_npz_to_hdf5.py",
            "--input-npz",
            str(input_npz),
            "--output-h5",
            str(output_h5),
            "--dataset-name",
            "visual-toy",
            "--observation-output-key",
            "pixels",
            "--target-mode",
            "index",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    with h5py.File(output_h5, "r") as handle:
        assert handle.attrs["target_storage"] == "index"
        assert "target" not in handle
        np.testing.assert_array_equal(handle["target_index"][:], np.array([2, 2, 5, 5]))


def test_convert_visual_ogbench_npz_can_skip_next_pixels(tmp_path: Path) -> None:
    input_npz = tmp_path / "visual_skip_next.npz"
    output_h5 = tmp_path / "visual_skip_next.h5"
    observations = np.arange(6 * 4 * 4 * 3, dtype=np.uint8).reshape(6, 4, 4, 3)
    actions = np.arange(6 * 5, dtype=np.float32).reshape(6, 5)
    terminals = np.array([False, False, True, False, False, True])

    np.savez(
        input_npz,
        observations=observations,
        actions=actions,
        terminals=terminals,
    )

    subprocess.run(
        [
            "./.venv/bin/python",
            "scripts/convert_ogbench_npz_to_hdf5.py",
            "--input-npz",
            str(input_npz),
            "--output-h5",
            str(output_h5),
            "--dataset-name",
            "visual-toy",
            "--observation-output-key",
            "pixels",
            "--target-mode",
            "index",
            "--next-observation-mode",
            "skip",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    with h5py.File(output_h5, "r") as handle:
        assert handle.attrs["next_observation_storage"] == "skip"
        assert "next_pixels" not in handle
        np.testing.assert_array_equal(
            handle["pixels"][:],
            np.concatenate([observations[0:2], observations[3:5]], axis=0),
        )


def test_convert_ogbench_npz_can_cache_npz_members_for_mmap_reads(tmp_path: Path) -> None:
    input_npz = tmp_path / "cached.npz"
    output_h5 = tmp_path / "cached.h5"
    cache_dir = tmp_path / "cache"
    observations = np.arange(7 * 3, dtype=np.float32).reshape(7, 3)
    actions = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
    terminals = np.array([False, False, True, False, False, False, True])

    np.savez_compressed(
        input_npz,
        observations=observations,
        actions=actions,
        terminals=terminals,
    )

    subprocess.run(
        [
            "./.venv/bin/python",
            "scripts/convert_ogbench_npz_to_hdf5.py",
            "--input-npz",
            str(input_npz),
            "--output-h5",
            str(output_h5),
            "--dataset-name",
            "cached-toy",
            "--npz-cache-dir",
            str(cache_dir),
            "--target-mode",
            "index",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    assert (cache_dir / "observations.npy").exists()
    assert (cache_dir / "actions.npy").exists()
    assert (cache_dir / "terminals.npy").exists()
    with h5py.File(output_h5, "r") as handle:
        assert handle.attrs["npz_cache_dir"] == str(cache_dir)
        np.testing.assert_array_equal(
            handle["observation"][:],
            np.vstack([observations[0:2], observations[3:6]]),
        )
        np.testing.assert_array_equal(handle["target_index"][:], np.array([2, 2, 6, 6, 6]))


def test_convert_ogbench_npz_refreshes_cache_when_source_changes_with_same_shape(
    tmp_path: Path,
) -> None:
    input_npz = tmp_path / "refresh.npz"
    output_h5_a = tmp_path / "refresh_a.h5"
    output_h5_b = tmp_path / "refresh_b.h5"
    cache_dir = tmp_path / "cache"
    observations_a = np.arange(7 * 3, dtype=np.float32).reshape(7, 3)
    observations_b = observations_a + 100.0
    actions = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
    terminals = np.array([False, False, True, False, False, False, True])

    np.savez_compressed(
        input_npz,
        observations=observations_a,
        actions=actions,
        terminals=terminals,
    )
    subprocess.run(
        [
            "./.venv/bin/python",
            "scripts/convert_ogbench_npz_to_hdf5.py",
            "--input-npz",
            str(input_npz),
            "--output-h5",
            str(output_h5_a),
            "--dataset-name",
            "refresh-toy",
            "--npz-cache-dir",
            str(cache_dir),
            "--target-mode",
            "index",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    np.savez_compressed(
        input_npz,
        observations=observations_b,
        actions=actions,
        terminals=terminals,
    )
    subprocess.run(
        [
            "./.venv/bin/python",
            "scripts/convert_ogbench_npz_to_hdf5.py",
            "--input-npz",
            str(input_npz),
            "--output-h5",
            str(output_h5_b),
            "--dataset-name",
            "refresh-toy",
            "--npz-cache-dir",
            str(cache_dir),
            "--target-mode",
            "index",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    with h5py.File(output_h5_b, "r") as handle:
        np.testing.assert_array_equal(
            handle["observation"][:],
            np.vstack([observations_b[0:2], observations_b[3:6]]),
        )
