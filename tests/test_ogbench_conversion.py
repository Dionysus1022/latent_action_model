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
