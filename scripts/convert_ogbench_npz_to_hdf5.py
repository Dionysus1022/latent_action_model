#!/usr/bin/env python3
"""Convert an OGBench .npz dataset into the root-level HDF5 schema used here."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - tqdm is available in the project env.
    _tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", required=True, type=Path)
    parser.add_argument("--output-h5", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--observation-output-key",
        choices=("observation", "pixels"),
        default="observation",
        help="HDF5 key to use for OGBench observations.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of transitions to process per HDF5 write chunk.",
    )
    parser.add_argument(
        "--target-mode",
        choices=("repeated", "index", "none"),
        default="repeated",
        help=(
            "How to store episode-final targets. 'repeated' writes a per-transition "
            "target array for compatibility; 'index' stores only target_index; "
            "'none' skips target storage."
        ),
    )
    parser.add_argument(
        "--target-chunk-size",
        type=int,
        default=5_000,
        help="Rows per write when --target-mode repeated materializes target observations.",
    )
    parser.add_argument(
        "--npz-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for extracting .npz members to .npy files before conversion. "
            "Use this for large compressed visual NPZ files so arrays can be memory-mapped."
        ),
    )
    parser.add_argument(
        "--next-observation-mode",
        choices=("write", "skip"),
        default="write",
        help=(
            "Whether to materialize next_observation/next_pixels. Existing LeWM HDF5 "
            "datasets do not require this key; skipping saves time and disk for visual data."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars and stage messages.",
    )
    return parser.parse_args()


def print_stage(stage: int, total: int, message: str, *, enabled: bool) -> None:
    if enabled:
        print(f"[stage {stage:02d}/{total:02d}] {message}", flush=True)


def progress_iter(
    iterable,
    *,
    total: int,
    desc: str,
    unit: str,
    enabled: bool,
):
    if enabled and _tqdm is not None:
        return _tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)
    return iterable


class ProgressCounter:
    def __init__(
        self,
        *,
        total: int,
        desc: str,
        unit: str,
        enabled: bool,
        unit_scale: bool = False,
        unit_divisor: int = 1000,
    ) -> None:
        self._bar = None
        if enabled and _tqdm is not None:
            self._bar = _tqdm(
                total=total,
                desc=desc,
                unit=unit,
                unit_scale=unit_scale,
                unit_divisor=unit_divisor,
                dynamic_ncols=True,
            )

    def update(self, value: int) -> None:
        if self._bar is not None:
            self._bar.update(value)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()

    def __enter__(self) -> "ProgressCounter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class CachedNpzPayload:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self._arrays = arrays
        self.files = list(arrays.keys())

    def __getitem__(self, key: str) -> np.ndarray:
        return self._arrays[key]


def npz_array_members(input_npz: Path) -> dict[str, str]:
    with ZipFile(input_npz) as archive:
        members: dict[str, str] = {}
        for info in archive.infolist():
            filename = Path(info.filename).name
            if filename.endswith(".npy"):
                members[filename[:-4]] = info.filename
        return members


def is_valid_npy_cache(path: Path, expected_size: int) -> bool:
    if not path.exists() or int(path.stat().st_size) != int(expected_size):
        return False
    try:
        np.load(path, mmap_mode="r")
    except Exception:
        return False
    return True


def cache_metadata_for_member(input_npz: Path, info) -> dict[str, int | str]:
    stat = input_npz.stat()
    return {
        "source_path": str(input_npz),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "member_name": str(info.filename),
        "member_crc": int(info.CRC),
        "member_file_size": int(info.file_size),
        "member_compress_size": int(info.compress_size),
    }


def read_json_file(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def is_valid_member_cache(path: Path, metadata_path: Path, expected_metadata: dict) -> bool:
    if not is_valid_npy_cache(path, int(expected_metadata["member_file_size"])):
        return False
    return read_json_file(metadata_path) == expected_metadata


def extract_npz_member_to_cache(
    archive: ZipFile,
    input_npz: Path,
    member_name: str,
    output_path: Path,
    *,
    progress: bool,
) -> None:
    info = archive.getinfo(member_name)
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    expected_metadata = cache_metadata_for_member(input_npz, info)
    if is_valid_member_cache(output_path, metadata_path, expected_metadata):
        if progress:
            print(f"[npz-cache] reuse {output_path}", flush=True)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_metadata_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    if tmp_metadata_path.exists():
        tmp_metadata_path.unlink()
    if progress:
        print(
            f"[npz-cache] extract {member_name} -> {output_path} "
            f"({info.file_size / 1024**3:.2f}GiB)",
            flush=True,
        )

    with archive.open(info) as source, tmp_path.open("wb") as target:
        with ProgressCounter(
            total=int(info.file_size),
            desc=f"cache {Path(member_name).stem}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            enabled=progress,
        ) as counter:
            while True:
                chunk = source.read(16 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                counter.update(len(chunk))
    with tmp_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(expected_metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(output_path)
    tmp_metadata_path.replace(metadata_path)


def cache_npz_members(
    input_npz: Path,
    cache_dir: Path,
    *,
    progress: bool,
) -> dict[str, Path]:
    members = npz_array_members(input_npz)
    required = {"observations", "actions", "terminals"}
    missing = sorted(required - set(members))
    if missing:
        raise ValueError(f"Missing required OGBench fields: {missing}")

    keys_to_cache = [
        key
        for key in ("observations", "actions", "terminals", "qpos", "qvel", "button_states")
        if key in members
    ]
    paths: dict[str, Path] = {}
    with ZipFile(input_npz) as archive:
        for key in keys_to_cache:
            output_path = cache_dir / f"{key}.npy"
            extract_npz_member_to_cache(
                archive,
                input_npz,
                members[key],
                output_path,
                progress=progress,
            )
            paths[key] = output_path
    return paths


@contextmanager
def open_array_payload(input_npz: Path, cache_dir: Path | None, *, progress: bool):
    if cache_dir is None:
        with np.load(input_npz, mmap_mode="r") as payload:
            yield payload, None
        return

    cached_paths = cache_npz_members(input_npz, cache_dir, progress=progress)
    arrays = {
        key: np.load(path, mmap_mode="r")
        for key, path in cached_paths.items()
    }
    yield CachedNpzPayload(arrays), cache_dir


def validate_npz_payload(payload) -> None:
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
    chunk_size: int = 100_000,
    progress: bool = True,
) -> h5py.Dataset:
    if array.ndim == 0:
        return handle.create_dataset(name, data=array, compression=compression)

    if indices is not None:
        indices = np.asarray(indices, dtype=np.int64)
        output_shape = (int(indices.shape[0]), *array.shape[1:])
    else:
        output_shape = array.shape

    dataset = handle.create_dataset(
        name,
        shape=output_shape,
        dtype=array.dtype,
        chunks=True,
        compression=compression,
    )
    num_rows = int(output_shape[0])
    num_chunks = math.ceil(num_rows / chunk_size) if num_rows else 0
    chunk_starts = range(0, num_rows, chunk_size)
    for start in progress_iter(
        chunk_starts,
        total=num_chunks,
        desc=f"write {name}",
        unit="chunk",
        enabled=progress,
    ):
        end = min(num_rows, start + chunk_size)
        if indices is None:
            dataset[start:end] = array[start:end]
        else:
            dataset[start:end] = array[indices[start:end]]
    return dataset


def row_chunk_shape(
    num_rows: int,
    row_shape: tuple[int, ...],
    dtype: np.dtype,
    *,
    preferred_rows: int = 4_096,
    max_chunk_bytes: int = 4 * 1024 * 1024,
) -> tuple[int, ...]:
    row_elements = int(np.prod(row_shape, dtype=np.int64)) if row_shape else 1
    row_bytes = max(1, row_elements * np.dtype(dtype).itemsize)
    rows_by_bytes = max(1, max_chunk_bytes // row_bytes)
    rows = max(1, min(int(num_rows), int(preferred_rows), int(rows_by_bytes)))
    return (rows, *row_shape)


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
    progress: bool = True,
    target_mode: str = "repeated",
    target_chunk_size: int = 5_000,
) -> None:
    num_steps = int(transition_terminals.shape[0])
    num_episodes = int(episode_lengths.shape[0])
    if target_mode not in ("repeated", "index", "none"):
        raise ValueError(f"Unsupported target_mode: {target_mode!r}.")
    if target_chunk_size <= 0:
        raise ValueError(f"target_chunk_size must be positive, got {target_chunk_size}.")

    ep_idx_ds = handle.create_dataset("ep_idx", shape=(num_steps,), dtype=np.int64, chunks=True, compression="gzip")
    step_idx_ds = handle.create_dataset(
        "step_idx", shape=(num_steps,), dtype=np.int64, chunks=True, compression="gzip"
    )
    truncated_ds = handle.create_dataset(
        "truncated", shape=(num_steps,), dtype=np.bool_, chunks=True, compression="gzip"
    )
    success_ds = handle.create_dataset("success", shape=(num_steps,), dtype=np.bool_, chunks=True, compression="gzip")
    target_index_ds = None
    target_indices_per_step = None
    if target_mode != "none":
        target_index_ds = handle.create_dataset(
            "target_index",
            shape=(num_steps,),
            dtype=np.int64,
            chunks=True,
            compression="gzip",
        )
        target_indices_per_step = np.empty(num_steps, dtype=np.int64)

    with ProgressCounter(total=num_steps, desc="write metadata", unit="row", enabled=progress) as counter:
        for episode_id in range(num_episodes):
            start = int(episode_offsets[episode_id])
            length = int(episode_lengths[episode_id])
            end = start + length
            ep_idx_ds[start:end] = episode_id
            step_idx_ds[start:end] = np.arange(length, dtype=np.int64)
            truncated_ds[start:end] = False
            success_ds[start:end] = transition_terminals[start:end]
            if target_indices_per_step is not None:
                target_indices_per_step[start:end] = int(target_indices[episode_id])
            counter.update(length)

    if target_index_ds is not None and target_indices_per_step is not None:
        target_index_ds[:] = target_indices_per_step

    if target_mode != "repeated":
        return

    target_ds = handle.create_dataset(
        "target",
        shape=(num_steps, *observations.shape[1:]),
        dtype=observations.dtype,
        chunks=row_chunk_shape(num_steps, tuple(observations.shape[1:]), observations.dtype),
        compression="gzip",
    )
    write_rows = min(int(chunk_size), int(target_chunk_size))
    num_chunks = math.ceil(num_steps / write_rows) if num_steps else 0
    chunk_starts = range(0, num_steps, write_rows)
    for chunk_start in progress_iter(
        chunk_starts,
        total=num_chunks,
        desc="write target",
        unit="chunk",
        enabled=progress,
    ):
        chunk_end = min(num_steps, chunk_start + write_rows)
        target_ds[chunk_start:chunk_end] = observations[target_indices_per_step[chunk_start:chunk_end]]


def log_payload_summary(
    *,
    dataset_name: str,
    observations: np.ndarray,
    actions: np.ndarray,
    episode_lengths: np.ndarray,
    observation_indices: np.ndarray,
    output_h5: Path,
    progress: bool,
) -> None:
    if not progress:
        return
    print(
        "[convert] "
        f"dataset={dataset_name} "
        f"observations={observations.shape}/{observations.dtype} "
        f"actions={actions.shape}/{actions.dtype} "
        f"episodes={episode_lengths.shape[0]} "
        f"transitions={observation_indices.shape[0]} "
        f"output={output_h5}",
        flush=True,
    )


def convert_ogbench_npz_to_hdf5(
    input_npz: Path,
    output_h5: Path,
    dataset_name: str,
    observation_output_key: str = "observation",
    chunk_size: int = 100_000,
    progress: bool = True,
    target_mode: str = "repeated",
    target_chunk_size: int = 5_000,
    npz_cache_dir: Path | None = None,
    next_observation_mode: str = "write",
) -> None:
    if chunk_size <= 0:
        raise ValueError(f"--chunk-size must be positive, got {chunk_size}.")
    if observation_output_key not in ("observation", "pixels"):
        raise ValueError(
            "observation_output_key must be either 'observation' or 'pixels', "
            f"got {observation_output_key!r}."
        )
    if target_mode not in ("repeated", "index", "none"):
        raise ValueError(f"target_mode must be one of repeated/index/none, got {target_mode!r}.")
    if target_chunk_size <= 0:
        raise ValueError(f"--target-chunk-size must be positive, got {target_chunk_size}.")
    if next_observation_mode not in ("write", "skip"):
        raise ValueError(
            "next_observation_mode must be either 'write' or 'skip', "
            f"got {next_observation_mode!r}."
        )
    if input_npz.suffix != ".npz":
        raise ValueError(f"--input-npz must point to a .npz file, got {input_npz}.")
    if output_h5.suffix != ".h5":
        raise ValueError(f"--output-h5 must point to a .h5 file, got {output_h5}.")

    input_npz = input_npz.expanduser()
    output_h5 = output_h5.expanduser()
    if npz_cache_dir is not None:
        npz_cache_dir = npz_cache_dir.expanduser()
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    print_stage(1, 6, f"open npz: {input_npz}", enabled=progress)
    with open_array_payload(input_npz, npz_cache_dir, progress=progress) as (payload, active_cache_dir):
        print_stage(2, 6, "validate payload and build episode indices", enabled=progress)
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

        log_payload_summary(
            dataset_name=dataset_name,
            observations=observations,
            actions=actions,
            episode_lengths=episode_lengths,
            observation_indices=observation_indices,
            output_h5=output_h5,
            progress=progress,
        )

        tmp_h5 = output_h5.with_suffix(output_h5.suffix + ".tmp")
        if tmp_h5.exists():
            print_stage(3, 6, f"remove stale tmp file: {tmp_h5}", enabled=progress)
            tmp_h5.unlink()
        else:
            print_stage(3, 6, f"prepare tmp file: {tmp_h5}", enabled=progress)
        with h5py.File(tmp_h5, "w") as handle:
            handle.attrs["source_format"] = "ogbench_npz"
            handle.attrs["source_path"] = str(input_npz)
            handle.attrs["dataset_name"] = dataset_name
            handle.attrs["created_at_utc"] = datetime.now(timezone.utc).isoformat()
            handle.attrs["target_policy"] = "episode_final_observation"
            handle.attrs["target_storage"] = target_mode
            handle.attrs["transition_policy"] = "ogbench_regular"
            handle.attrs["observation_output_key"] = observation_output_key
            handle.attrs["next_observation_storage"] = next_observation_mode
            if active_cache_dir is not None:
                handle.attrs["npz_cache_dir"] = str(active_cache_dir)

            next_observation_output_key = (
                "next_pixels" if observation_output_key == "pixels" else "next_observation"
            )
            print_stage(4, 6, "write transition arrays", enabled=progress)
            create_dataset_from_array(
                handle,
                observation_output_key,
                observations,
                indices=observation_indices,
                chunk_size=chunk_size,
                progress=progress,
            )
            if next_observation_mode == "write":
                create_dataset_from_array(
                    handle,
                    next_observation_output_key,
                    observations,
                    indices=next_observation_indices,
                    chunk_size=chunk_size,
                    progress=progress,
                )
            if actions.shape[0] == observations.shape[0]:
                create_dataset_from_array(
                    handle,
                    "action",
                    actions,
                    indices=observation_indices,
                    chunk_size=chunk_size,
                    progress=progress,
                )
            else:
                create_dataset_from_array(
                    handle,
                    "action",
                    actions,
                    chunk_size=chunk_size,
                    progress=progress,
                )
            create_dataset_from_array(
                handle,
                "terminated",
                transition_terminals,
                chunk_size=chunk_size,
                progress=progress,
            )
            create_dataset_from_array(
                handle,
                "ep_len",
                episode_lengths,
                compression=None,
                chunk_size=chunk_size,
                progress=progress,
            )
            create_dataset_from_array(
                handle,
                "ep_offset",
                episode_offsets,
                compression=None,
                chunk_size=chunk_size,
                progress=progress,
            )
            for key in ("qpos", "qvel", "button_states"):
                if key in payload.files:
                    create_dataset_from_array(
                        handle,
                        key,
                        payload[key],
                        indices=observation_indices,
                        chunk_size=chunk_size,
                        progress=progress,
                    )
            print_stage(5, 6, "write episode metadata and targets", enabled=progress)
            write_per_step_metadata(
                handle,
                episode_lengths=episode_lengths,
                episode_offsets=episode_offsets,
                transition_terminals=transition_terminals,
                observations=observations,
                target_indices=episode_ends,
                chunk_size=chunk_size,
                progress=progress,
                target_mode=target_mode,
                target_chunk_size=target_chunk_size,
            )

        print_stage(6, 6, f"finalize h5: {output_h5}", enabled=progress)
        tmp_h5.replace(output_h5)


def main() -> int:
    args = parse_args()
    convert_ogbench_npz_to_hdf5(
        input_npz=args.input_npz,
        output_h5=args.output_h5,
        dataset_name=args.dataset_name,
        observation_output_key=args.observation_output_key,
        chunk_size=args.chunk_size,
        progress=not bool(args.no_progress),
        target_mode=args.target_mode,
        target_chunk_size=args.target_chunk_size,
        npz_cache_dir=args.npz_cache_dir,
        next_observation_mode=args.next_observation_mode,
    )
    print(f"[convert] wrote {args.output_h5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
