from __future__ import annotations

import subprocess
import sys
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from omegaconf import DictConfig, OmegaConf

from diffusion.config_schema import normalize_task_name, require_not_old_data_root, resolve_policy_path

StageRunner = Callable[[str, list[str], Path], None]
STAGE_ORDER = ["prepare_data", "split_hdf5", "build_dataset", "build_anchors", "train"]

DATASET_BATCH_RE = re.compile(r"\[batch\]\s+rows\s+(?P<start>\d+):(?P<end>\d+)")
TRAIN_STEP_RE = re.compile(r"\[train\].*?epoch=(?P<epoch>\d+)\s+step=(?P<step>\d+)/(?P<steps>\d+)")
TRAIN_EPOCH_RE = re.compile(r"\[epoch\].*?epoch=(?P<epoch>\d+)")


@dataclass(frozen=True)
class StageProgressUpdate:
    completed: int | None = None
    total: int | None = None
    description: str | None = None


class ProgressReporter(Protocol):
    def start(self, stages: list[str]) -> None:
        ...

    def stage_started(self, stage: str, total: int | None, log_path: Path | None) -> None:
        ...

    def stage_progress(
        self,
        stage: str,
        completed: int | None = None,
        total: int | None = None,
        description: str | None = None,
    ) -> None:
        ...

    def stage_finished(self, stage: str, status: str, reason: str | None = None) -> None:
        ...

    def finish(self) -> None:
        ...


class NullProgressReporter:
    def start(self, stages: list[str]) -> None:
        return None

    def stage_started(self, stage: str, total: int | None, log_path: Path | None) -> None:
        return None

    def stage_progress(
        self,
        stage: str,
        completed: int | None = None,
        total: int | None = None,
        description: str | None = None,
    ) -> None:
        return None

    def stage_finished(self, stage: str, status: str, reason: str | None = None) -> None:
        return None

    def finish(self) -> None:
        return None


class RichStageProgressReporter:
    """Render one live progress line for the currently running pipeline stage."""

    def __init__(self) -> None:
        self._progress = None
        self._task_id = None
        self._stages: list[str] = []
        self._current_total: int | None = None
        self._active_stage: str | None = None

    def start(self, stages: list[str]) -> None:
        self._stages = list(stages)
        try:
            from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
        except ImportError:
            self._progress = None
            return

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.fields[detail]}"),
            TimeElapsedColumn(),
            transient=False,
        )
        self._progress.start()

    def stage_started(self, stage: str, total: int | None, log_path: Path | None) -> None:
        if self._progress is None:
            return
        self._current_total = None if total is None else int(total)
        self._active_stage = stage
        description = self._format_stage_description(stage, "running")
        detail = self._format_progress_detail(0, self._current_total, "running")
        if self._task_id is None:
            self._task_id = self._progress.add_task(
                description,
                total=self._current_total,
                completed=0,
                detail=detail,
            )
            return
        self._progress.reset(
            self._task_id,
            total=self._current_total,
            completed=0,
            description=description,
            detail=detail,
        )

    def stage_progress(
        self,
        stage: str,
        completed: int | None = None,
        total: int | None = None,
        description: str | None = None,
    ) -> None:
        if self._progress is None or self._task_id is None:
            return
        effective_completed = completed
        if total is not None:
            parsed_total = int(total)
            if self._current_total is not None and parsed_total < self._current_total:
                effective_completed = None
            else:
                self._current_total = parsed_total
        detail = self._format_progress_detail(completed, self._current_total, description)
        update_kwargs: dict[str, Any] = {
            "description": self._format_stage_description(stage, "running"),
            "detail": detail,
        }
        if effective_completed is not None:
            update_kwargs["completed"] = int(effective_completed)
        if self._current_total is not None:
            update_kwargs["total"] = self._current_total
        self._progress.update(self._task_id, **update_kwargs)

    def stage_finished(self, stage: str, status: str, reason: str | None = None) -> None:
        if self._progress is None:
            return
        description = self._format_stage_description(stage, status)
        detail = str(reason) if reason else status
        stage_was_active = self._active_stage == stage
        if self._task_id is None:
            self._task_id = self._progress.add_task(
                description,
                total=1,
                completed=1,
                detail=detail,
            )
            self._active_stage = None
            self._current_total = None
            return
        total = self._current_total if stage_was_active and self._current_total is not None else 1
        completed = total if status in {"completed", "skipped"} else 0
        self._progress.update(
            self._task_id,
            description=description,
            total=total,
            completed=completed,
            detail=detail,
        )
        self._active_stage = None
        self._current_total = None

    def finish(self) -> None:
        if self._progress is not None:
            self._progress.stop()

    def _format_stage_description(self, stage: str, status: str) -> str:
        try:
            index = self._stages.index(stage) + 1
            prefix = f"[{index}/{len(self._stages)}]"
        except ValueError:
            prefix = "[?/?]"
        return f"{prefix} {stage} {status}"

    @staticmethod
    def _format_progress_detail(
        completed: int | None,
        total: int | None,
        description: str | None,
    ) -> str:
        if description:
            return description
        if completed is not None and total is not None:
            return f"{completed}/{total}"
        if total is not None:
            return f"0/{total}"
        return "running"


def default_progress_reporter(is_terminal: bool | None = None) -> ProgressReporter:
    if is_terminal is None:
        is_terminal = sys.stderr.isatty()
    if not bool(is_terminal):
        return NullProgressReporter()
    return RichStageProgressReporter()


def parse_stage_progress_line(
    stage: str,
    line: str,
    *,
    default_total: int | None = None,
) -> StageProgressUpdate | None:
    if stage == "build_dataset":
        match = DATASET_BATCH_RE.search(line)
        if match is None:
            return None
        completed = int(match.group("end"))
        total = None if default_total is None else int(default_total)
        if total is not None:
            completed = min(completed, total)
        description = f"rows {completed}/{total}" if total is not None else f"rows {completed}"
        return StageProgressUpdate(completed=completed, total=total, description=description)

    if stage == "train":
        step_match = TRAIN_STEP_RE.search(line)
        if step_match is not None:
            epoch = int(step_match.group("epoch"))
            step_index = int(step_match.group("step"))
            steps_per_epoch = int(step_match.group("steps"))
            step_completed = step_index + 1
            epochs_total = None if default_total is None else int(default_total)
            total = None if epochs_total is None else epochs_total * steps_per_epoch
            completed = (epoch - 1) * steps_per_epoch + step_completed
            if total is not None:
                completed = min(completed, total)
            description = (
                f"epoch {epoch}/{epochs_total} step {step_completed}/{steps_per_epoch}"
                if epochs_total is not None
                else f"epoch {epoch} step {step_completed}/{steps_per_epoch}"
            )
            return StageProgressUpdate(completed=completed, total=total, description=description)

        epoch_match = TRAIN_EPOCH_RE.search(line)
        if epoch_match is not None:
            epoch = int(epoch_match.group("epoch"))
            total = None if default_total is None else int(default_total)
            description = f"epoch {epoch}/{total}" if total is not None else f"epoch {epoch}"
            return StageProgressUpdate(completed=epoch, total=total, description=description)

    return None


def _path(value: Any) -> Path:
    return Path(str(value)).expanduser()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _best_bundle_path(cfg: DictConfig) -> Path:
    return _path(cfg.task.train_output_dir) / "diffusion_planner_best_bundle.pt"


def _use_raw_dataset(cfg: DictConfig) -> bool:
    return bool(cfg.pipeline.get("use_raw_dataset", False))


def _planner_dataset_h5(cfg: DictConfig) -> Path:
    if _use_raw_dataset(cfg):
        return _path(cfg.task.raw_h5)
    return _path(cfg.task.split_train_h5)


def _planner_dataset_cache_dir(cfg: DictConfig) -> Path:
    if _use_raw_dataset(cfg):
        return _path(cfg.task.data_root)
    return _path(cfg.task.split_train_root)


def build_stage_commands(cfg: DictConfig) -> dict[str, list[str]]:
    python = sys.executable
    planner_dataset_h5 = _planner_dataset_h5(cfg)
    planner_dataset_cache_dir = _planner_dataset_cache_dir(cfg)
    return {
        "split_hdf5": [
            python,
            "-u",
            "scripts/split_hdf5_by_episode.py",
            "--input-h5",
            str(_path(cfg.task.raw_h5)),
            "--output-train-h5",
            str(_path(cfg.task.split_train_h5)),
            "--output-test-h5",
            str(_path(cfg.task.split_val_h5)),
            "--train-ratio",
            str(cfg.pipeline.train_ratio),
            "--seed",
            str(cfg.pipeline.seed),
            "--episode-key",
            str(cfg.task.episode_key),
        ],
        "build_dataset": [
            python,
            "-u",
            "-m",
            "diffusion.dataset_builder",
            "--mode",
            "build",
            "--task",
            str(normalize_task_name(cfg.task.name)),
            "--wm-policy",
            str(_path(cfg.task.wm_policy)),
            "--dataset-name",
            str(cfg.task.dataset_name),
            "--dataset-h5",
            str(planner_dataset_h5),
            "--label-source",
            "trajectory",
            "--num-samples",
            str(cfg.pipeline.num_samples),
            "--output-path",
            str(_path(cfg.task.planner_dataset_path)),
            "--batch-size",
            str(cfg.pipeline.build_batch_size),
            "--seed",
            str(cfg.pipeline.seed),
            "--device",
            str(cfg.pipeline.device),
            "--on-error",
            "skip",
            "--cache-dir",
            str(planner_dataset_cache_dir),
        ],
        "build_anchors": [
            python,
            "-u",
            "-m",
            "diffusion.anchor_builder",
            "--mode",
            "build",
            "--dataset-path",
            str(_path(cfg.task.planner_dataset_path)),
            "--output-path",
            str(_path(cfg.task.anchor_bundle_path)),
            "--num-anchors",
            str(cfg.anchors.num_anchors),
            "--seed",
            str(cfg.pipeline.seed),
            "--max-samples",
            str(cfg.anchors.max_samples),
            "--on-error",
            "skip",
            "--max-iter",
            str(cfg.anchors.max_iter),
            "--task",
            str(normalize_task_name(cfg.task.name)),
        ],
        "train": [
            python,
            "-u",
            "-m",
            "diffusion.train",
            "--dataset-path",
            str(_path(cfg.task.planner_dataset_path)),
            "--anchor-bundle-path",
            str(_path(cfg.task.anchor_bundle_path)),
            "--wm-policy",
            str(_path(cfg.task.wm_policy)),
            "--output-dir",
            str(_path(cfg.task.train_output_dir)),
            "--seed",
            str(cfg.pipeline.seed),
            "--device",
            str(cfg.pipeline.device),
            "--epochs",
            str(cfg.train.epochs),
            "--batch-size",
            str(cfg.train.batch_size),
            "--val-batch-size",
            str(cfg.train.val_batch_size),
            "--lr",
            str(cfg.train.lr),
            "--weight-decay",
            str(cfg.train.weight_decay),
            "--val-split",
            str(cfg.train.val_split),
            "--num-workers",
            str(cfg.train.num_workers),
            "--log-every",
            str(cfg.train.log_every),
            "--loss-preset",
            str(cfg.train.loss_preset),
        ],
    }


def default_runner(
    stage: str,
    argv: list[str],
    log_path: Path,
    *,
    progress_callback: Callable[[StageProgressUpdate], None] | None = None,
    default_total: int | None = None,
) -> None:
    _ensure_parent(log_path)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[stage] {stage}\n")
        log_file.write(f"[command] {' '.join(argv)}\n")
        log_file.flush()
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is not None:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                update = parse_stage_progress_line(stage, line, default_total=default_total)
                if update is not None and progress_callback is not None:
                    progress_callback(update)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, argv)


class DiffusionPipeline:
    def __init__(
        self,
        cfg: DictConfig,
        runner: StageRunner | None = None,
        progress_reporter: ProgressReporter | None = None,
    ) -> None:
        self.cfg = cfg
        self.runner = runner
        self.progress_reporter = progress_reporter or default_progress_reporter()
        self.commands = build_stage_commands(cfg)
        self.log_dir = _path(cfg.task.train_output_dir) / "pipeline_logs"
        self.summary_path = _path(cfg.task.train_output_dir) / "pipeline_summary.yaml"

    def validate(self) -> None:
        for value in [
            self.cfg.task.data_root,
            self.cfg.task.raw_h5,
            self.cfg.task.planner_dataset_path,
            self.cfg.task.anchor_bundle_path,
            self.cfg.task.train_output_dir,
            self.cfg.task.wm_policy,
        ]:
            require_not_old_data_root(value)
        if not _use_raw_dataset(self.cfg):
            for value in [
                self.cfg.task.split_train_h5,
                self.cfg.task.split_val_h5,
            ]:
                require_not_old_data_root(value)
        if normalize_task_name(self.cfg.task.name) is None:
            raise ValueError("task.name must be set.")
        resolve_policy_path(self.cfg.task.wm_policy)
        if str(self.cfg.pipeline.device) == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("pipeline.device=cuda but CUDA is not available.")
        _path(self.cfg.task.train_output_dir).mkdir(parents=True, exist_ok=True)

    def prepare_data(self) -> dict[str, Any]:
        raw_h5 = _path(self.cfg.task.raw_h5)
        if raw_h5.exists():
            return {
                "stage": "prepare_data",
                "status": "skipped",
                "reason": "raw_h5 exists",
                "output": str(raw_h5),
            }
        if not bool(self.cfg.task.download_if_missing):
            raise FileNotFoundError(
                "Missing raw HDF5:\n"
                f"  {raw_h5}\n\n"
                "Download the dataset or set task.download_if_missing=true when download metadata is configured."
            )
        download = self.cfg.task.get("download", {})
        url = download.get("url", None)
        if url in [None, "", "null"]:
            raise FileNotFoundError(
                "Missing raw HDF5 and no download URL is configured:\n"
                f"  {raw_h5}"
            )
        return self._run_download(url=str(url), raw_h5=raw_h5)

    def _run_download(self, url: str, raw_h5: Path) -> dict[str, Any]:
        output = self.cfg.task.download.get("output", None) or str(raw_h5)
        archive_type = str(self.cfg.task.download.get("archive_type", "none"))
        download_path = _path(output)
        _ensure_parent(download_path)
        if archive_type == "none":
            argv = ["wget", "-c", "-O", str(raw_h5), url]
        else:
            argv = ["wget", "-c", "-O", str(download_path), url]
        if bool(self.cfg.pipeline.dry_run):
            return {"stage": "prepare_data", "status": "skipped", "reason": "dry_run", "command": argv}
        self._run_command("prepare_data", argv, self.log_dir / "prepare_data.log")
        if archive_type == "h5.zst":
            self._run_command(
                "prepare_data",
                ["zstd", "-d", "-f", str(download_path), "-o", str(raw_h5)],
                self.log_dir / "prepare_data_extract.log",
            )
        elif archive_type == "tar.zst":
            self._run_command(
                "prepare_data",
                ["tar", "--zstd", "-xvf", str(download_path), "-C", str(_path(self.cfg.task.data_root))],
                self.log_dir / "prepare_data_extract.log",
            )
        elif archive_type != "none":
            raise ValueError(f"Unsupported archive_type: {archive_type}")
        return {"stage": "prepare_data", "status": "completed", "output": str(raw_h5)}

    def _run_or_skip(self, stage: str, outputs: list[Path], force: bool) -> dict[str, Any]:
        if all(path.exists() for path in outputs) and not force:
            return {
                "stage": stage,
                "status": "skipped",
                "reason": "outputs exist",
                "outputs": [str(path) for path in outputs],
            }
        if bool(self.cfg.pipeline.dry_run):
            return {
                "stage": stage,
                "status": "skipped",
                "reason": "dry_run",
                "command": self.commands[stage],
            }
        for output in outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
        self._run_command(stage, self.commands[stage], self.log_dir / f"{stage}.log")
        return {"stage": stage, "status": "completed", "outputs": [str(path) for path in outputs]}

    def split_hdf5(self) -> dict[str, Any]:
        if _use_raw_dataset(self.cfg):
            return {
                "stage": "split_hdf5",
                "status": "skipped",
                "reason": "pipeline.use_raw_dataset=true",
                "outputs": [str(_path(self.cfg.task.raw_h5))],
            }
        return self._run_or_skip(
            "split_hdf5",
            [_path(self.cfg.task.split_train_h5), _path(self.cfg.task.split_val_h5)],
            bool(self.cfg.pipeline.force_split),
        )

    def _stage_total(self, stage: str) -> int | None:
        if stage == "build_dataset":
            return int(self.cfg.pipeline.num_samples)
        if stage == "train":
            return int(self.cfg.train.epochs)
        return None

    def _run_command(self, stage: str, argv: list[str], log_path: Path) -> None:
        stage_total = self._stage_total(stage)
        self.progress_reporter.stage_started(stage, stage_total, log_path)

        def report_progress(update: StageProgressUpdate) -> None:
            self.progress_reporter.stage_progress(
                stage,
                completed=update.completed,
                total=update.total,
                description=update.description,
            )

        if self.runner is None:
            default_runner(
                stage,
                argv,
                log_path,
                progress_callback=report_progress,
                default_total=stage_total,
            )
            return
        self.runner(stage, argv, log_path)

    def _run_stage_by_name(self, stage: str) -> dict[str, Any]:
        if stage == "prepare_data":
            return self.prepare_data()
        if stage == "split_hdf5":
            return self.split_hdf5()
        if stage == "build_dataset":
            return self._run_or_skip(
                "build_dataset",
                [_path(self.cfg.task.planner_dataset_path)],
                bool(self.cfg.pipeline.force_dataset),
            )
        if stage == "build_anchors":
            return self._run_or_skip(
                "build_anchors",
                [_path(self.cfg.task.anchor_bundle_path)],
                bool(self.cfg.pipeline.force_anchors),
            )
        if stage == "train":
            return self._run_or_skip(
                "train",
                [_best_bundle_path(self.cfg)],
                bool(self.cfg.pipeline.force_train),
            )
        raise ValueError(f"Unsupported pipeline stage: {stage}")

    def run(self) -> dict[str, Any]:
        self.validate()
        started_at = _now()
        stages = []
        self.progress_reporter.start(list(STAGE_ORDER))
        try:
            for stage in STAGE_ORDER:
                try:
                    stage_result = self._run_stage_by_name(stage)
                except Exception as exc:
                    self.progress_reporter.stage_finished(stage, "failed", reason=str(exc))
                    raise
                self.progress_reporter.stage_finished(
                    stage_result["stage"],
                    stage_result["status"],
                    reason=stage_result.get("reason"),
                )
                stages.append(stage_result)
        finally:
            self.progress_reporter.finish()
        summary = {
            "started_at": started_at,
            "ended_at": _now(),
            "task": str(normalize_task_name(self.cfg.task.name)),
            "best_bundle": str(_best_bundle_path(self.cfg)),
            "stages": stages,
            "resolved_config": OmegaConf.to_container(self.cfg, resolve=True),
        }
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(OmegaConf.to_yaml(OmegaConf.create(summary)), encoding="utf-8")
        return summary


def run_pipeline(cfg: DictConfig) -> dict[str, Any]:
    return DiffusionPipeline(cfg).run()
