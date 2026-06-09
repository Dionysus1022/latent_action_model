"""Local dataset path adapters for this workspace."""

from __future__ import annotations

from pathlib import Path


DEFAULT_DATASET_H5 = {
    "tworoom": Path("/data/ykz/tworoom/tworoom.h5"),
    "pusht": Path("/data/ykz/pusht/pusht_expert_train.h5"),
    "cube": Path("/data/ykz/cube/cube_single_expert.h5"),
    "reacher": Path("/data/ykz/reacher/reacher.h5"),
}


def resolve_dataset_h5(task: str, dataset_h5: str | None = None) -> Path:
    """Resolve the HDF5 file used for a task.

    Passing dataset_h5 overrides the workspace defaults. The returned path is
    suitable for stable_worldmodel.data.HDF5Dataset via name=stem/cache_dir=parent.
    """
    if dataset_h5 is None:
        try:
            path = DEFAULT_DATASET_H5[task]
        except KeyError as exc:
            raise KeyError(
                f"No default dataset path configured for task '{task}'. "
                f"Known tasks: {sorted(DEFAULT_DATASET_H5)}"
            ) from exc
    else:
        path = Path(dataset_h5)

    path = path.expanduser().resolve()
    if path.suffix != ".h5":
        raise ValueError(f"Expected an .h5 dataset path, got: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Dataset HDF5 file does not exist: {path}")
    return path


def hdf5_name_and_cache_dir(path: Path) -> tuple[str, str]:
    return path.stem, str(path.parent)
