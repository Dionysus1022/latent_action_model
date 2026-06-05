from __future__ import annotations

from pathlib import Path
from typing import Any


TASK_ALIASES = {
    "cube": "cube",
    "pusht": "pusht",
    "tworoom": "tworoom",
    "two-room": "tworoom",
    "two_room": "tworoom",
    "reacher": "reacher",
    "researcher": "reacher",
}


def normalize_task_name(task_name: str | None) -> str | None:
    if task_name in [None, "", "null"]:
        return None
    normalized = str(task_name).strip().lower()
    return TASK_ALIASES.get(normalized, normalized)


def as_path(value: Any) -> Path:
    return Path(str(value)).expanduser()


def exists_as_policy_reference(path: Path) -> bool:
    if path.exists():
        return True
    if path.with_name(f"{path.name}_object.ckpt").exists():
        return True
    if path.with_suffix(".ckpt").exists():
        return True
    return False


def resolve_policy_path(value: str | Path) -> Path:
    path = as_path(value)
    if exists_as_policy_reference(path):
        return path
    raise FileNotFoundError(
        "Missing world model policy:\n"
        f"  {path}\n\n"
        "Place the checkpoint there or override:\n"
        "  task.wm_policy=/path/to/lewm_epoch_xx"
    )


def require_not_old_data_root(path: str | Path) -> None:
    if "/data/yuekangzhou" in str(path):
        raise ValueError(
            f"Path still points at old data root /data/yuekangzhou: {path}. "
            "Use /data/ykz or override the task config."
        )
