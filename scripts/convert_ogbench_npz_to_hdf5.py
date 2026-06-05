#!/usr/bin/env python3
"""Convert an OGBench .npz dataset into the root-level HDF5 schema used here."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--output-h5", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of transitions to process per HDF5 write chunk.",
    )
    return parser.parse_args()


def validate_npz_payload(payload: np.lib.npyio.NpzFile) -> None:
    required = {"observations", "actions", "terminals"}
    missing = sorted(required - set(payload.files))
    if missing:
        raise ValueError(f"Missing required OGBench fields: {missing}")

    num_states = payload["observations"].shape[0]
    if payload["terminals"].shape[0] != num_states:
        raise ValueError(
            f"Field 'terminals' has {payload['terminals'].shape[0]} rows, expected {num_states}."
        )
    action_rows = payload["actions"].shape[0]
    if action_rows not in (num_states, num_states - int(np.asarray(payload["terminals"], dtype=bool).sum())):
        raise ValueError(
            "Field 'actions' must have either one row per raw state or one row per valid transition; "
            f"got actions={action_rows}, observations={num_states}."
        )
    for key in ("qpos", "qvel", "button_states"):
        if key in payload.files and payload[key].shape[0] != num_states:
            raise ValueError(
                f"Field {key!r} has {payload[key].shape[0]} rows, expected {num_states}."
            )


def episode_ends_from_terminals(terminals: np.ndarray) -> np.ndarray:
    terminals = np.asarray(terminals, dtype=bool).reshape(-1)
    if terminals.size == 0:
        raise ValueError("Dataset is empty.")

    ends = np.flatnonzero(terminals)
    if ends.size == 0 or int(ends[-1]) != terminals.size - 1:
        ends = np.concatenate([ends, np.array([terminals.size - 1], dtype=np.int64)])
    return ends.astype(np.int64)


def raw_episode_lengths_and_offsets(
    num_steps: int, episode_ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.concatenate([np.array([0], dtype=np.int64), episode_ends[:-1] + 1])
    lengths = episode_ends - starts + 1
    if int(lengths.sum()) != int(num_steps):
        raise AssertionError("Episode lengths do not cover the dataset.")
    return lengths.astype(np.int64), starts.astype(np.int64)


def create_dataset_from_array(
    handle: h5py.File,
    name: str,
    array: np.ndarray,
    indices: np.ndarray | None = None,
    compression: str | None = "gzip",
) -> h5py.Dataset:
    if indices is not None:
        array = array[indices]
    chunks = True if array.ndim > 0 else None
    return handle.create_dataset(name, data=array, chunks=chunks, compression=compression)


def build_transition_indices(
    raw_episode_lengths: np.ndarray,
    raw_episode_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transition_lengths = raw_episode_lengths - 1
    if np.any(transition_lengths <= 0):
        raise ValueError("Every raw episode must contain at least two states.")
    transition_offsets = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(transition_lengths[:-1], dtype=np.int64)]
    )

    observation_indices = np.concatenate(
        [
            np.arange(int(start), int(start + length - 1), dtype=np.int64)
            for start, length in zip(raw_episode_offsets, raw_episode_lengths)
        ]
    )
    next_observation_indices = observation_indices + 1
    return (
        observation_indices,
        next_observation_indices,
        transition_lengths.astype(np.int64),
        transition_offsets.astype(np.int64),
    )


def write_per_step_metadata(
    handle: h5py.File,
    episode_lengths: np.ndarray,
    episode_offsets: np.ndarray,
    transition_terminals: np.ndarray,
    observations: np.ndarray,
    target_indices: np.ndarray,
    chunk_size: int,
) -> None:
    num_steps = int(transition_terminals.shape[0])
    num_episodes = int(episode_lengths.shape[0])

    ep_idx_ds = handle.create_dataset("ep_idx", shape=(num_steps,), dtype=np.int64, chunks=True, compression="gzip")
    step_idx_ds = handle.create_dataset(
        "step_idx", shape=(num_steps,), dtype=np.int64, chunks=True, compression="gzip"
    )
    truncated_ds = handle.create_dataset(
        "truncated", shape=(num_steps,), dtype=np.bool_, chunks=True, compression="gzip"
    )
    success_ds = handle.create_dataset("success", shape=(num_steps,), dtype=np.bool_, chunks=True, compression="gzip")
    target_ds = handle.create_dataset(
        "target",
        shape=(num_steps, *observations.shape[1:]),
        dtype=observations.dtype,
        chunks=True,
        compression="gzip",
    )

    for episode_id in range(num_episodes):
        start = int(episode_offsets[episode_id])
        length = int(episode_lengths[episode_id])
        end = start + length
        ep_idx_ds[start:end] = episode_id
        step_idx_ds[start:end] = np.arange(length, dtype=np.int64)
        truncated_ds[start:end] = False
        success_ds[start:end] = transition_terminals[start:end]

        final_observation = observations[target_indices[episode_id] : target_indices[episode_id] + 1]
        for chunk_start in range(start, end, chunk_size):
            chunk_end = min(end, chunk_start + chunk_size)
            target_ds[chunk_start:chunk_end] = final_observation


def convert_ogbench_npz_to_hdf5(
    input_npz: Path,
    output_h5: Path,
    dataset_name: str,
    chunk_size: int = 100_000,
) -> None:
    if chunk_size <= 0:
        raise ValueError(f"--chunk-size must be positive, got {chunk_size}.")
    if input_npz.suffix != ".npz":
        raise ValueError(f"--input-npz must point to a .npz file, got {input_npz}.")
    if output_h5.suffix != ".h5":
        raise ValueError(f"--output-h5 must point to a .h5 file, got {output_h5}.")

    input_npz = input_npz.expanduser()
    output_h5 = output_h5.expanduser()
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with np.load(input_npz, mmap_mode="r") as payload:
        validate_npz_payload(payload)
        observations = payload["observations"]
        actions = payload["actions"]
        terminals = np.asarray(payload["terminals"], dtype=bool).reshape(-1)

        episode_ends = episode_ends_from_terminals(terminals)
        raw_episode_lengths, raw_episode_offsets = raw_episode_lengths_and_offsets(
            int(observations.shape[0]), episode_ends
        )
        (
            observation_indices,
            next_observation_indices,
            episode_lengths,
            episode_offsets,
        ) = build_transition_indices(raw_episode_lengths, raw_episode_offsets)
        transition_terminals = np.zeros(int(observation_indices.shape[0]), dtype=bool)
        transition_terminals[episode_offsets + episode_lengths - 1] = True

        tmp_h5 = output_h5.with_suffix(output_h5.suffix + ".tmp")
        if tmp_h5.exists():
            tmp_h5.unlink()
        with h5py.File(tmp_h5, "w") as handle:
            handle.attrs["source_format"] = "ogbench_npz"
            handle.attrs["source_path"] = str(input_npz)
            handle.attrs["dataset_name"] = dataset_name
            handle.attrs["created_at_utc"] = datetime.now(timezone.utc).isoformat()
            handle.attrs["target_policy"] = "episode_final_observation"
            handle.attrs["transition_policy"] = "ogbench_regular"

            create_dataset_from_array(handle, "observation", observations, indices=observation_indices)
            create_dataset_from_array(
                handle,
                "next_observation",
                observations,
                indices=next_observation_indices,
            )
            if actions.shape[0] == observations.shape[0]:
                create_dataset_from_array(handle, "action", actions, indices=observation_indices)
            else:
                create_dataset_from_array(handle, "action", actions)
            create_dataset_from_array(handle, "terminated", transition_terminals)
            create_dataset_from_array(handle, "ep_len", episode_lengths, compression=None)
            create_dataset_from_array(handle, "ep_offset", episode_offsets, compression=None)
            for key in ("qpos", "qvel", "button_states"):
                if key in payload.files:
                    create_dataset_from_array(handle, key, payload[key], indices=observation_indices)
            write_per_step_metadata(
                handle,
                episode_lengths=episode_lengths,
                episode_offsets=episode_offsets,
                transition_terminals=transition_terminals,
                observations=observations,
                target_indices=episode_ends,
                chunk_size=chunk_size,
            )

        tmp_h5.replace(output_h5)


def main() -> int:
    args = parse_args()
    convert_ogbench_npz_to_hdf5(
        input_npz=args.input_npz,
        output_h5=args.output_h5,
        dataset_name=args.dataset_name,
        chunk_size=args.chunk_size,
    )
    print(f"[convert] wrote {args.output_h5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
