#!/usr/bin/env python3
"""Split an HDF5 dataset into train/test files by full episodes.

Supports two storage layouts:
1. Top-level groups where each group is an episode/trajectory.
2. Array-style storage with transition-level datasets and an explicit episode index.

For array-style storage, the output keeps episodes contiguous and rewrites
episode-level metadata like `ep_idx`, `ep_len`, and `ep_offset` so the split
files stay compatible with stable_worldmodel.data.HDF5Dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HDF5_PLUGIN_PATH_USED: str | None = None
try:
    import hdf5plugin  # type: ignore

    plugin_path = str(Path(hdf5plugin.PLUGINS_PATH))
    existing_plugin_path = os.environ.get("HDF5_PLUGIN_PATH", "")
    existing_entries = [entry for entry in existing_plugin_path.split(os.pathsep) if entry]
    valid_existing_entries = [
        entry for entry in existing_entries if entry != plugin_path and Path(entry).exists()
    ]
    os.environ["HDF5_PLUGIN_PATH"] = os.pathsep.join([plugin_path, *valid_existing_entries])
    _HDF5_PLUGIN_PATH_USED = os.environ["HDF5_PLUGIN_PATH"]
except Exception:
    hdf5plugin = None  # type: ignore

import h5py
import numpy as np


AUTO_EPISODE_KEY_CANDIDATES = (
    "episode_idx",
    "ep_idx",
    "episode_id",
    "traj_idx",
    "trajectory_idx",
)
EPISODE_LENGTH_BASENAMES = {"ep_len", "episode_len", "traj_len", "trajectory_len"}
EPISODE_OFFSET_BASENAMES = {"ep_offset", "episode_offset", "traj_offset", "trajectory_offset"}


@dataclass(frozen=True)
class SplitAssignment:
    train_episode_ids: list[Any]
    test_episode_ids: list[Any]


@dataclass(frozen=True)
class ArraySplitPlan:
    split_name: str
    selected_original_episode_ids: list[Any]
    selected_episode_positions: np.ndarray
    selected_transition_indices: np.ndarray
    original_to_new_episode_id: dict[Any, int]
    remapped_transition_episode_ids: np.ndarray
    remapped_episode_level_ids: np.ndarray
    episode_lengths: np.ndarray
    episode_offsets: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5", required=True, type=Path)
    parser.add_argument("--output-train-h5", required=True, type=Path)
    parser.add_argument("--output-test-h5", required=True, type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--episode-key",
        type=str,
        default="auto",
        help="Episode-index dataset path/name for array-style HDF5. Default: auto.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def normalize_h5_path(path: str) -> str:
    normalized = path.strip("/")
    return normalized


def dataset_paths(handle: h5py.File) -> list[str]:
    paths: list[str] = []

    def visitor(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            paths.append(name)

    handle.visititems(visitor)
    return sorted(paths)


def to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def to_jsonable(value: Any) -> Any:
    value = to_python_scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def detect_storage_mode(handle: h5py.File) -> str:
    root_keys = list(handle.keys())
    if root_keys and all(isinstance(handle[key], h5py.Group) for key in root_keys):
        return "episode_groups"
    return "array"


def resolve_episode_key_path(handle: h5py.File, episode_key: str) -> str:
    paths = dataset_paths(handle)
    basename_to_paths: dict[str, list[str]] = {}
    for path in paths:
        basename_to_paths.setdefault(Path(path).name, []).append(path)

    if episode_key != "auto":
        normalized = normalize_h5_path(episode_key)
        if normalized in paths:
            return normalized
        basename_matches = basename_to_paths.get(normalized, [])
        if len(basename_matches) == 1:
            return basename_matches[0]
        if len(basename_matches) > 1:
            raise ValueError(
                f"--episode-key={episode_key!r} is ambiguous. Matching datasets: {basename_matches}. "
                "Please pass the full HDF5 dataset path."
            )
        raise ValueError(
            f"--episode-key={episode_key!r} not found in input HDF5. "
            f"Available datasets include: {paths[:20]}{'...' if len(paths) > 20 else ''}"
        )

    candidate_matches: list[str] = []
    for candidate in AUTO_EPISODE_KEY_CANDIDATES:
        candidate_matches.extend(basename_to_paths.get(candidate, []))

    candidate_matches = sorted(set(candidate_matches))
    if len(candidate_matches) == 1:
        return candidate_matches[0]
    if len(candidate_matches) > 1:
        raise ValueError(
            "Could not reliably infer the episode key automatically. "
            f"Found multiple candidates: {candidate_matches}. "
            "Please specify --episode-key explicitly."
        )
    raise ValueError(
        "Could not reliably infer episode boundaries for array-style HDF5. "
        f"Tried auto candidates {list(AUTO_EPISODE_KEY_CANDIDATES)} but found none. "
        "Please specify --episode-key explicitly."
    )


def stable_unique(values: np.ndarray) -> list[Any]:
    ordered: list[Any] = []
    seen: set[Any] = set()
    for value in values.tolist():
        py_value = to_python_scalar(value)
        if py_value in seen:
            continue
        seen.add(py_value)
        ordered.append(py_value)
    return ordered


def assign_episode_split(episode_ids: list[Any], train_ratio: float, seed: int) -> SplitAssignment:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"--train-ratio must be in (0, 1), got {train_ratio}.")
    if len(episode_ids) < 2:
        raise ValueError(
            "Need at least 2 episodes to produce non-empty train/test splits, "
            f"but found {len(episode_ids)}."
        )

    shuffled = list(episode_ids)
    random.Random(seed).shuffle(shuffled)

    num_train = int(round(len(shuffled) * train_ratio))
    num_train = max(1, min(len(shuffled) - 1, num_train))

    train_ids = shuffled[:num_train]
    test_ids = shuffled[num_train:]
    return SplitAssignment(train_episode_ids=train_ids, test_episode_ids=test_ids)


def validate_split_assignment(
    episode_ids: list[Any],
    assignment: SplitAssignment,
) -> None:
    train_set = set(assignment.train_episode_ids)
    test_set = set(assignment.test_episode_ids)
    if train_set & test_set:
        raise AssertionError("Train/test episode ids overlap.")
    if len(assignment.train_episode_ids) + len(assignment.test_episode_ids) != len(episode_ids):
        raise AssertionError("Train/test episode counts do not sum to the total.")
    if train_set | test_set != set(episode_ids):
        raise AssertionError("Train/test episode ids do not cover the full episode set.")


def build_array_split_plan(
    *,
    split_name: str,
    row_episode_ids: np.ndarray,
    ordered_episode_ids: list[Any],
    selected_episode_ids: list[Any],
) -> ArraySplitPlan:
    selected_episode_id_set = set(selected_episode_ids)
    selected_episode_positions = np.array(
        [index for index, episode_id in enumerate(ordered_episode_ids) if episode_id in selected_episode_id_set],
        dtype=np.int64,
    )
    selected_episode_ids_in_file_order = [ordered_episode_ids[index] for index in selected_episode_positions.tolist()]
    original_to_new_episode_id = {
        episode_id: new_id for new_id, episode_id in enumerate(selected_episode_ids_in_file_order)
    }

    selected_transition_indices = np.flatnonzero(
        np.isin(row_episode_ids, np.array(selected_episode_ids_in_file_order, dtype=row_episode_ids.dtype))
    ).astype(np.int64)
    remapped_transition_episode_ids = np.array(
        [original_to_new_episode_id[to_python_scalar(row_episode_ids[index])] for index in selected_transition_indices],
        dtype=np.int64,
    )

    episode_lengths = np.array(
        [
            int(np.sum(row_episode_ids[selected_transition_indices] == episode_id))
            for episode_id in selected_episode_ids_in_file_order
        ],
        dtype=np.int64,
    )
    episode_offsets = np.zeros_like(episode_lengths)
    if episode_lengths.size > 0:
        episode_offsets[1:] = np.cumsum(episode_lengths[:-1], dtype=np.int64)

    return ArraySplitPlan(
        split_name=split_name,
        selected_original_episode_ids=selected_episode_ids_in_file_order,
        selected_episode_positions=selected_episode_positions,
        selected_transition_indices=selected_transition_indices,
        original_to_new_episode_id=original_to_new_episode_id,
        remapped_transition_episode_ids=remapped_transition_episode_ids,
        remapped_episode_level_ids=np.arange(len(selected_episode_ids_in_file_order), dtype=np.int64),
        episode_lengths=episode_lengths,
        episode_offsets=episode_offsets,
    )


def maybe_adjust_chunks(chunks: tuple[int, ...] | None, new_shape: tuple[int, ...]) -> tuple[int, ...] | None:
    if chunks is None:
        return None
    if len(chunks) != len(new_shape):
        return chunks
    adjusted = []
    for chunk_dim, shape_dim in zip(chunks, new_shape):
        if shape_dim <= 0:
            adjusted.append(max(1, int(chunk_dim)))
        else:
            adjusted.append(max(1, min(int(chunk_dim), int(shape_dim))))
    return tuple(adjusted)


def dataset_create_kwargs(src: h5py.Dataset, new_shape: tuple[int, ...]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "dtype": src.dtype,
    }
    chunks = maybe_adjust_chunks(src.chunks, new_shape)
    if chunks is not None:
        kwargs["chunks"] = chunks
    compression = src.compression
    if compression is not None and str(compression).lower() != "unknown":
        kwargs["compression"] = compression
        if src.compression_opts is not None:
            kwargs["compression_opts"] = src.compression_opts
    elif compression is not None:
        print(
            "[compression-warning] "
            f"dataset={src.name} uses unsupported compression={compression!r}; "
            "writing the split output without compression for this dataset."
        )
    if src.shuffle:
        kwargs["shuffle"] = src.shuffle
    if src.fletcher32:
        kwargs["fletcher32"] = src.fletcher32
    if src.scaleoffset is not None:
        kwargs["scaleoffset"] = src.scaleoffset
    fillvalue = getattr(src, "fillvalue", None)
    if fillvalue is not None:
        kwargs["fillvalue"] = fillvalue
    return kwargs


def copy_dataset_full(src: h5py.Dataset, dst_group: h5py.Group, name: str) -> None:
    data = src[()]
    dst_dataset = dst_group.create_dataset(name, data=data, **dataset_create_kwargs(src, tuple(src.shape)))
    copy_attrs(src.attrs, dst_dataset.attrs)


def write_indexed_dataset(
    src: h5py.Dataset,
    dst_group: h5py.Group,
    name: str,
    indices: np.ndarray,
    remap_values: np.ndarray | None = None,
) -> None:
    output_shape = (len(indices),) + tuple(src.shape[1:])
    dst_dataset = dst_group.create_dataset(name, shape=output_shape, **dataset_create_kwargs(src, output_shape))
    copy_attrs(src.attrs, dst_dataset.attrs)
    if len(indices) == 0:
        return

    chunk_rows = src.chunks[0] if src.chunks is not None and len(src.chunks) > 0 else 1024
    chunk_rows = max(1, int(chunk_rows))
    for start in range(0, len(indices), chunk_rows):
        end = min(len(indices), start + chunk_rows)
        selection = indices[start:end]
        try:
            batch = src[selection]
        except OSError as exc:
            raise OSError(
                f"Failed to read dataset {src.name!r} from the input HDF5. "
                f"This often means its compression plugin is unavailable "
                f"(compression={src.compression!r}, HDF5_PLUGIN_PATH={os.environ.get('HDF5_PLUGIN_PATH')!r}). "
                "Try ensuring the same Python environment has `hdf5plugin` installed, "
                "or set HDF5_PLUGIN_PATH to a valid plugin directory."
            ) from exc
        if remap_values is not None:
            batch = remap_values[start:end].astype(src.dtype, copy=False)
        dst_dataset[start:end] = batch


def should_remap_transition_episode_dataset(
    *,
    path: str,
    dataset: h5py.Dataset,
    episode_key_path: str,
    row_episode_ids: np.ndarray,
) -> bool:
    if path == episode_key_path:
        return True
    if dataset.ndim != 1 or int(dataset.shape[0]) != int(row_episode_ids.shape[0]):
        return False
    if Path(path).name not in AUTO_EPISODE_KEY_CANDIDATES:
        return False
    return np.array_equal(dataset[:], row_episode_ids)


def should_remap_episode_level_id_dataset(
    *,
    path: str,
    dataset: h5py.Dataset,
    ordered_episode_ids: list[Any],
) -> bool:
    if dataset.ndim != 1 or int(dataset.shape[0]) != len(ordered_episode_ids):
        return False
    if Path(path).name not in AUTO_EPISODE_KEY_CANDIDATES:
        return False
    if dataset.dtype.kind not in ("i", "u"):
        return False
    return np.array_equal(dataset[:], np.array(ordered_episode_ids, dtype=dataset.dtype))


def copy_array_tree(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    *,
    split_plan: ArraySplitPlan,
    num_rows: int,
    num_episodes: int,
    episode_key_path: str,
    ordered_episode_ids: list[Any],
    row_episode_ids: np.ndarray,
    path_prefix: str = "",
) -> None:
    copy_attrs(src_group.attrs, dst_group.attrs)
    for name, item in src_group.items():
        current_path = f"{path_prefix}/{name}" if path_prefix else name
        if isinstance(item, h5py.Group):
            child_group = dst_group.create_group(name)
            copy_array_tree(
                item,
                child_group,
                split_plan=split_plan,
                num_rows=num_rows,
                num_episodes=num_episodes,
                episode_key_path=episode_key_path,
                ordered_episode_ids=ordered_episode_ids,
                row_episode_ids=row_episode_ids,
                path_prefix=current_path,
            )
            continue

        if item.ndim > 0 and int(item.shape[0]) == num_rows:
            remap_values = None
            if should_remap_transition_episode_dataset(
                path=current_path,
                dataset=item,
                episode_key_path=episode_key_path,
                row_episode_ids=row_episode_ids,
            ):
                remap_values = split_plan.remapped_transition_episode_ids
            write_indexed_dataset(
                item,
                dst_group,
                name,
                indices=split_plan.selected_transition_indices,
                remap_values=remap_values,
            )
            continue

        if item.ndim > 0 and int(item.shape[0]) == num_episodes:
            basename = Path(current_path).name
            if basename in EPISODE_LENGTH_BASENAMES:
                data = split_plan.episode_lengths.astype(item.dtype, copy=False)
                dst_dataset = dst_group.create_dataset(
                    name,
                    data=data,
                    **dataset_create_kwargs(item, tuple(data.shape)),
                )
                copy_attrs(item.attrs, dst_dataset.attrs)
                continue
            if basename in EPISODE_OFFSET_BASENAMES:
                data = split_plan.episode_offsets.astype(item.dtype, copy=False)
                dst_dataset = dst_group.create_dataset(
                    name,
                    data=data,
                    **dataset_create_kwargs(item, tuple(data.shape)),
                )
                copy_attrs(item.attrs, dst_dataset.attrs)
                continue
            if should_remap_episode_level_id_dataset(
                path=current_path,
                dataset=item,
                ordered_episode_ids=ordered_episode_ids,
            ):
                data = split_plan.remapped_episode_level_ids.astype(item.dtype, copy=False)
                dst_dataset = dst_group.create_dataset(
                    name,
                    data=data,
                    **dataset_create_kwargs(item, tuple(data.shape)),
                )
                copy_attrs(item.attrs, dst_dataset.attrs)
                continue

            write_indexed_dataset(
                item,
                dst_group,
                name,
                indices=split_plan.selected_episode_positions,
            )
            continue

        copy_dataset_full(item, dst_group, name)


def ensure_parent_dirs(*paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def copy_episode_groups(
    src: h5py.File,
    dst_path: Path,
    selected_group_names: list[str],
) -> None:
    with h5py.File(dst_path, "w") as dst:
        copy_attrs(src.attrs, dst.attrs)
        for group_name in selected_group_names:
            src.copy(src[group_name], dst, name=group_name)


def write_array_split(
    src: h5py.File,
    dst_path: Path,
    *,
    split_plan: ArraySplitPlan,
    num_rows: int,
    num_episodes: int,
    episode_key_path: str,
    ordered_episode_ids: list[Any],
    row_episode_ids: np.ndarray,
) -> None:
    with h5py.File(dst_path, "w") as dst:
        copy_array_tree(
            src,
            dst,
            split_plan=split_plan,
            num_rows=num_rows,
            num_episodes=num_episodes,
            episode_key_path=episode_key_path,
            ordered_episode_ids=ordered_episode_ids,
            row_episode_ids=row_episode_ids,
        )


def root_attrs_equal(src: h5py.File, dst: h5py.File) -> bool:
    if set(src.attrs.keys()) != set(dst.attrs.keys()):
        return False
    for key in src.attrs.keys():
        if not np.array_equal(np.asarray(src.attrs[key]), np.asarray(dst.attrs[key])):
            return False
    return True


def sanity_check_group_outputs(
    src_path: Path,
    train_path: Path,
    test_path: Path,
    assignment: SplitAssignment,
) -> None:
    with h5py.File(src_path, "r") as src, h5py.File(train_path, "r") as train_h5, h5py.File(
        test_path, "r"
    ) as test_h5:
        if not root_attrs_equal(src, train_h5) or not root_attrs_equal(src, test_h5):
            raise AssertionError("Root attrs were not preserved in the output HDF5 files.")
        if set(train_h5.keys()) != set(assignment.train_episode_ids):
            raise AssertionError("Train HDF5 does not contain the expected episode groups.")
        if set(test_h5.keys()) != set(assignment.test_episode_ids):
            raise AssertionError("Test HDF5 does not contain the expected episode groups.")


def sanity_check_array_outputs(
    src_path: Path,
    train_path: Path,
    test_path: Path,
    *,
    episode_key_path: str,
    train_plan: ArraySplitPlan,
    test_plan: ArraySplitPlan,
) -> None:
    with h5py.File(src_path, "r") as src, h5py.File(train_path, "r") as train_h5, h5py.File(
        test_path, "r"
    ) as test_h5:
        if not root_attrs_equal(src, train_h5) or not root_attrs_equal(src, test_h5):
            raise AssertionError("Root attrs were not preserved in the output HDF5 files.")

        train_episode_idx = train_h5[episode_key_path][:]
        test_episode_idx = test_h5[episode_key_path][:]
        if not np.array_equal(
            np.unique(train_episode_idx),
            np.arange(len(train_plan.selected_original_episode_ids), dtype=train_episode_idx.dtype),
        ):
            raise AssertionError("Train episode index dataset was not remapped to contiguous ids.")
        if not np.array_equal(
            np.unique(test_episode_idx),
            np.arange(len(test_plan.selected_original_episode_ids), dtype=test_episode_idx.dtype),
        ):
            raise AssertionError("Test episode index dataset was not remapped to contiguous ids.")
        if "ep_len" in train_h5 and train_h5["ep_len"].shape[0] != len(train_plan.selected_original_episode_ids):
            raise AssertionError("Train ep_len does not match the number of train episodes.")
        if "ep_len" in test_h5 and test_h5["ep_len"].shape[0] != len(test_plan.selected_original_episode_ids):
            raise AssertionError("Test ep_len does not match the number of test episodes.")


def build_split_info(
    *,
    input_h5: Path,
    output_train_h5: Path,
    output_test_h5: Path,
    train_ratio: float,
    seed: int,
    assignment: SplitAssignment,
) -> dict[str, Any]:
    return {
        "input_h5": str(input_h5),
        "train_h5": str(output_train_h5),
        "test_h5": str(output_test_h5),
        "train_ratio": train_ratio,
        "seed": seed,
        "num_episodes_total": len(assignment.train_episode_ids) + len(assignment.test_episode_ids),
        "num_episodes_train": len(assignment.train_episode_ids),
        "num_episodes_test": len(assignment.test_episode_ids),
        "train_episode_ids": [to_jsonable(value) for value in assignment.train_episode_ids],
        "test_episode_ids": [to_jsonable(value) for value in assignment.test_episode_ids],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def print_summary(
    *,
    storage_mode: str,
    input_h5: Path,
    episode_key_path: str | None,
    assignment: SplitAssignment,
    dry_run: bool,
    split_info_path: Path,
) -> None:
    print(f"[mode] storage_mode={storage_mode}")
    print(f"[input] {input_h5}")
    if episode_key_path is not None:
        print(f"[episode-key] {episode_key_path}")
    print(
        "[split] "
        f"train_episodes={len(assignment.train_episode_ids)} "
        f"test_episodes={len(assignment.test_episode_ids)} "
        f"total={len(assignment.train_episode_ids) + len(assignment.test_episode_ids)}"
    )
    print(f"[split] train_episode_ids={assignment.train_episode_ids}")
    print(f"[split] test_episode_ids={assignment.test_episode_ids}")
    if dry_run:
        print("[dry-run] no files will be written.")
    else:
        print(f"[sidecar] {split_info_path}")


def main() -> None:
    args = parse_args()
    input_h5 = args.input_h5.expanduser().resolve()
    output_train_h5 = args.output_train_h5.expanduser().resolve()
    output_test_h5 = args.output_test_h5.expanduser().resolve()
    split_info_path = output_train_h5.parent / "split_info.json"

    if not input_h5.exists():
        raise FileNotFoundError(f"Input HDF5 not found: {input_h5}")
    if input_h5 == output_train_h5 or input_h5 == output_test_h5:
        raise ValueError("Input HDF5 path must be different from both output paths.")
    if output_train_h5 == output_test_h5:
        raise ValueError("Train/test output paths must be different.")
    if _HDF5_PLUGIN_PATH_USED is not None:
        print(f"[hdf5-plugin] using HDF5_PLUGIN_PATH={_HDF5_PLUGIN_PATH_USED}")

    with h5py.File(input_h5, "r") as src:
        storage_mode = detect_storage_mode(src)
        if storage_mode == "episode_groups":
            episode_ids = list(src.keys())
            assignment = assign_episode_split(episode_ids, args.train_ratio, args.seed)
            validate_split_assignment(episode_ids, assignment)

            print_summary(
                storage_mode=storage_mode,
                input_h5=input_h5,
                episode_key_path=None,
                assignment=assignment,
                dry_run=args.dry_run,
                split_info_path=split_info_path,
            )
            if args.dry_run:
                return

            ensure_parent_dirs(output_train_h5, output_test_h5)
            copy_episode_groups(src, output_train_h5, assignment.train_episode_ids)
            copy_episode_groups(src, output_test_h5, assignment.test_episode_ids)
            sanity_check_group_outputs(input_h5, output_train_h5, output_test_h5, assignment)

        else:
            episode_key_path = resolve_episode_key_path(src, args.episode_key)
            row_episode_ids = src[episode_key_path][:]
            if row_episode_ids.ndim != 1:
                raise ValueError(
                    f"Episode key dataset {episode_key_path!r} must be 1D, got shape {tuple(row_episode_ids.shape)}."
                )
            if row_episode_ids.dtype.kind not in ("i", "u"):
                raise ValueError(
                    f"Episode key dataset {episode_key_path!r} must use an integer dtype, "
                    f"got {row_episode_ids.dtype}."
                )

            ordered_episode_ids = stable_unique(row_episode_ids)
            assignment = assign_episode_split(ordered_episode_ids, args.train_ratio, args.seed)
            validate_split_assignment(ordered_episode_ids, assignment)

            train_plan = build_array_split_plan(
                split_name="train",
                row_episode_ids=row_episode_ids,
                ordered_episode_ids=ordered_episode_ids,
                selected_episode_ids=assignment.train_episode_ids,
            )
            test_plan = build_array_split_plan(
                split_name="test",
                row_episode_ids=row_episode_ids,
                ordered_episode_ids=ordered_episode_ids,
                selected_episode_ids=assignment.test_episode_ids,
            )

            print_summary(
                storage_mode=storage_mode,
                input_h5=input_h5,
                episode_key_path=episode_key_path,
                assignment=assignment,
                dry_run=args.dry_run,
                split_info_path=split_info_path,
            )
            if args.dry_run:
                return

            ensure_parent_dirs(output_train_h5, output_test_h5)
            write_array_split(
                src,
                output_train_h5,
                split_plan=train_plan,
                num_rows=int(row_episode_ids.shape[0]),
                num_episodes=len(ordered_episode_ids),
                episode_key_path=episode_key_path,
                ordered_episode_ids=ordered_episode_ids,
                row_episode_ids=row_episode_ids,
            )
            write_array_split(
                src,
                output_test_h5,
                split_plan=test_plan,
                num_rows=int(row_episode_ids.shape[0]),
                num_episodes=len(ordered_episode_ids),
                episode_key_path=episode_key_path,
                ordered_episode_ids=ordered_episode_ids,
                row_episode_ids=row_episode_ids,
            )
            sanity_check_array_outputs(
                input_h5,
                output_train_h5,
                output_test_h5,
                episode_key_path=episode_key_path,
                train_plan=train_plan,
                test_plan=test_plan,
            )

    split_info = build_split_info(
        input_h5=input_h5,
        output_train_h5=output_train_h5,
        output_test_h5=output_test_h5,
        train_ratio=args.train_ratio,
        seed=args.seed,
        assignment=assignment,
    )
    split_info_path.write_text(json.dumps(split_info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] wrote train={output_train_h5} test={output_test_h5}")
    print(f"[done] wrote sidecar={split_info_path}")


if __name__ == "__main__":
    main()
