#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

SUMMARY_RE = re.compile(r"^\[summary\]\s+(?P<key>[a-zA-Z0-9_]+)=(?P<value>[^ ]+)s?$")
TRAJECTORY_RE = re.compile(r"^\[trajectory-quality\]\s+(?P<key>[a-zA-Z0-9_]+)=(?P<value>[-+0-9.eE]+)$")
PLANNER_RE = re.compile(r"^\[planner-stats\]\s+(?P<key>[a-zA-Z0-9_]+)=(?P<value>[-+0-9.eE]+)$")
CORRECTIVE_RE = re.compile(r"^\[corrective-stats\]\s+(?P<key>[a-zA-Z0-9_]+)=(?P<value>[-+0-9.eE]+)$")
KEY_VALUE_RE = re.compile(r"(?P<key>[a-zA-Z0-9_]+)=(?P<value>[^ ]+)")


@dataclass(frozen=True)
class TaskSpec:
    config_name: str
    dataset_h5: str
    ours_overrides: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentSpec:
    tasks: dict[str, TaskSpec]
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    repeats: tuple[int, ...]
    eval_num_eval: int = 50
    eval_goal_offset_steps: int = 25
    eval_budget: int = 50
    trajectory_quality: bool = True
    save_video: bool = False


@dataclass(frozen=True, order=True)
class EvalRun:
    task: str
    method: str
    seed: int
    repeat: int

    @property
    def run_id(self) -> str:
        return f"{self.task}/{self.method}_seed{self.seed}_repeat{self.repeat}"


def default_experiment_spec() -> ExperimentSpec:
    return ExperimentSpec(
        tasks={
            "cube": TaskSpec(
                config_name="cube",
                dataset_h5="/data/ykz/cube/cube_single_expert.h5",
                ours_overrides=(
                    "eval_profile=diffusion",
                    "diffusion_selection_mode=wm_only",
                    "diffusion_refinement.enabled=true",
                ),
            ),
            "pusht": TaskSpec(
                config_name="pusht",
                dataset_h5="/data/ykz/pusht/pusht_expert_train.h5",
                ours_overrides=(
                    "eval_profile=corrective_learned",
                    "diffusion_selection_mode=wm_only",
                    "diffusion_refinement.enabled=true",
                ),
            ),
            "reacher": TaskSpec(
                config_name="reacher",
                dataset_h5="/data/ykz/reacher/reacher.h5",
                ours_overrides=(
                    "eval_profile=diffusion",
                    "diffusion_selection_mode=wm_only",
                    "diffusion_refinement.enabled=true",
                ),
            ),
            "tworoom": TaskSpec(
                config_name="tworoom",
                dataset_h5="/data/ykz/tworoom/tworoom.h5",
                ours_overrides=(
                    "eval_profile=diffusion",
                    "diffusion_selection_mode=wm_only",
                    "diffusion_refinement.enabled=true",
                ),
            ),
        },
        methods=("mpc_cem", "gc_idm", "ours_full"),
        seeds=(42, 43, 44),
        repeats=(0, 1, 2),
    )


def build_run_matrix(spec: ExperimentSpec) -> list[EvalRun]:
    return [
        EvalRun(task=task, method=method, seed=seed, repeat=repeat)
        for task in spec.tasks
        for method in spec.methods
        for seed in spec.seeds
        for repeat in spec.repeats
    ]


def build_eval_command(run: EvalRun, *, spec: ExperimentSpec, python_bin: Path) -> list[str]:
    if run.task not in spec.tasks:
        raise KeyError(f"Unknown task '{run.task}'.")
    task = spec.tasks[run.task]
    command = [
        str(python_bin),
        str(REPO_ROOT / "eval.py"),
        "--config-name",
        task.config_name,
    ]
    if run.method == "mpc_cem":
        command.append("eval_profile=mpc")
    elif run.method == "gc_idm":
        command.append("eval_profile=gc_idm")
    elif run.method == "ours_full":
        command.extend(task.ours_overrides)
    else:
        raise ValueError(f"Unsupported method '{run.method}'.")

    command.extend(
        [
            f"seed={int(run.seed)}",
            f"eval.num_eval={int(spec.eval_num_eval)}",
            f"eval.goal_offset_steps={int(spec.eval_goal_offset_steps)}",
            f"eval.eval_budget={int(spec.eval_budget)}",
            f"trajectory_quality.enabled={str(bool(spec.trajectory_quality)).lower()}",
            f"trajectory_quality.save_video={str(bool(spec.save_video)).lower()}",
        ]
    )
    if not task.dataset_h5:
        raise ValueError(f"Task '{run.task}' must define an explicit dataset_h5.")
    command.append(f"+dataset_h5={task.dataset_h5}")
    return command


def _parse_number(value: str) -> int | float | str:
    cleaned = value.strip().rstrip(",")
    numeric_candidate = cleaned[:-1] if cleaned.endswith("s") else cleaned
    try:
        number = float(numeric_candidate)
    except ValueError:
        return cleaned
    if math.isfinite(number) and number.is_integer():
        return int(number)
    return number


def _set_metric(metrics: dict[str, Any], key: str, value: str) -> None:
    if key == "evaluation_time":
        key = "evaluation_time_sec"
    if key == "denoise_steps":
        key = "diffusion_truncation_steps"
    elif key == "runtime_execute_steps":
        key = "diffusion_runtime_execute_steps"
    elif key == "num_candidates":
        key = "diffusion_num_candidates"
    elif key == "selection_mode":
        key = "diffusion_selection_mode"
    elif key == "start_timestep":
        key = "diffusion_start_timestep"
    elif key == "bundle":
        key = "diffusion_bundle"
    metrics[key] = _parse_number(value)


def parse_eval_log_text(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for regex in (SUMMARY_RE, TRAJECTORY_RE, PLANNER_RE, CORRECTIVE_RE):
            match = regex.match(line)
            if match is not None:
                _set_metric(metrics, match.group("key"), match.group("value"))
                break
        else:
            if line.startswith("[planner]") or line.startswith("[planner-runtime]"):
                pairs = dict(KEY_VALUE_RE.findall(line))
                for key, value in pairs.items():
                    if key in {
                        "type",
                        "planner_type",
                        "policy",
                        "bundle",
                        "diffusion_bundle",
                        "selection_mode",
                        "num_candidates",
                        "denoise_steps",
                        "runtime_execute_steps",
                        "start_timestep",
                        "refinement_enabled",
                        "refinement_steps",
                        "refinement_topk",
                        "refinement_step_size",
                    }:
                        metric_key = "planner_type" if key == "type" else key
                        if metric_key == "refinement_enabled":
                            metric_key = "diffusion_refinement_enabled"
                        elif metric_key == "refinement_steps":
                            metric_key = "diffusion_refinement_steps"
                        elif metric_key == "refinement_topk":
                            metric_key = "diffusion_refinement_topk"
                        elif metric_key == "refinement_step_size":
                            metric_key = "diffusion_refinement_step_size"
                        _set_metric(metrics, metric_key, value)
            elif line.startswith("[refinement]"):
                pairs = dict(KEY_VALUE_RE.findall(line))
                mapping = {
                    "enabled": "diffusion_refinement_enabled",
                    "steps": "diffusion_refinement_steps",
                    "step_size": "diffusion_refinement_step_size",
                    "topk": "diffusion_refinement_topk",
                    "goal_weight": "diffusion_refinement_goal_weight",
                    "prior_weight": "diffusion_refinement_prior_weight",
                    "smoothness_weight": "diffusion_refinement_smoothness_weight",
                    "grad_clip_norm": "diffusion_refinement_grad_clip_norm",
                }
                for key, metric_key in mapping.items():
                    if key in pairs:
                        _set_metric(metrics, metric_key, pairs[key])
            elif line.startswith("[corrective]"):
                pairs = dict(KEY_VALUE_RE.findall(line))
                mapping = {
                    "mode": "corrective_mode",
                    "correction_interval": "corrective_correction_interval",
                    "effective_error_interval": "corrective_effective_error_interval",
                    "effective_execute_horizon": "corrective_effective_execute_horizon",
                    "error_threshold": "corrective_error_threshold",
                    "trigger_stat": "corrective_trigger_stat",
                    "trigger_quantile": "corrective_trigger_quantile",
                    "trigger_scope": "corrective_trigger_scope",
                    "error_metric": "corrective_error_metric",
                    "corrector_path": "corrector_path",
                }
                for key, metric_key in mapping.items():
                    if key in pairs:
                        _set_metric(metrics, metric_key, pairs[key])
            elif line.startswith("[corrective-stats]"):
                for key, value in KEY_VALUE_RE.findall(line):
                    _set_metric(metrics, key, value)
            elif line.startswith("[corrective-summary]"):
                pairs = dict(KEY_VALUE_RE.findall(line))
                if "prediction_error_count" in pairs:
                    _set_metric(metrics, "prediction_error_count", pairs["prediction_error_count"])
                if "episode_mean_count" in pairs:
                    _set_metric(metrics, "prediction_error_episode_mean_count", pairs["episode_mean_count"])
                if "mean" in pairs:
                    _set_metric(metrics, "prediction_error_mean", pairs["mean"])
                if "max" in pairs:
                    _set_metric(metrics, "prediction_error_max", pairs["max"])
                if "cohens_d" in pairs:
                    _set_metric(metrics, "prediction_error_cohens_d_fail_vs_success", pairs["cohens_d"])
                if "success_mean" in pairs:
                    _set_metric(metrics, "successful_prediction_error_mean", pairs["success_mean"])
                if "failure_mean" in pairs:
                    _set_metric(metrics, "failed_prediction_error_mean", pairs["failure_mean"])
                if "fail_minus_success" in pairs:
                    _set_metric(metrics, "prediction_error_fail_minus_success", pairs["fail_minus_success"])
                if "fail_success_ratio" in pairs:
                    _set_metric(metrics, "prediction_error_fail_success_ratio", pairs["fail_success_ratio"])
            elif line.startswith("[refinement-summary]"):
                pairs = dict(KEY_VALUE_RE.findall(line))
                mapping = {
                    "candidate_count": "refinement_candidate_count",
                    "steps": "refinement_steps_observed",
                    "cost_before": "refinement_cost_before",
                    "cost_after": "refinement_cost_after",
                    "goal_before": "refinement_goal_before",
                    "goal_after": "refinement_goal_after",
                    "delta_norm": "refinement_delta_norm",
                }
                for key, metric_key in mapping.items():
                    if key in pairs:
                        _set_metric(metrics, metric_key, pairs[key])
            elif line.startswith("[diffusion-rerank]"):
                for key, value in KEY_VALUE_RE.findall(line):
                    if key in {
                        "mode",
                        "num_candidates",
                        "denoise_steps",
                        "runtime_execute_steps",
                        "start_timestep",
                        "finite_candidate_rate",
                        "all_bad_env_rate",
                        "fallback_rate",
                        "selected_wm_cost_first",
                        "selected_model_score_first",
                        "selected_wm_cost_mean",
                        "selected_model_score_mean",
                    }:
                        metric_key = "diffusion_selection_mode" if key == "mode" else key
                        _set_metric(metrics, metric_key, value)
    return metrics


def parse_eval_log_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return parse_eval_log_text(path.read_text(encoding="utf-8", errors="replace"))


class ProgressPrinter:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.started_at = time.monotonic()

    def update(self, index: int, total: int, run: EvalRun, status: str) -> None:
        if not self.enabled:
            return
        width = 28
        filled = int(width * index / max(total, 1))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.monotonic() - self.started_at
        print(
            f"\r[{bar}] {index:03d}/{total:03d} "
            f"{status} task={run.task} method={run.method} seed={run.seed} repeat={run.repeat} "
            f"elapsed={elapsed:.1f}s",
            end="",
            flush=True,
        )
        if index >= total or status in {"failed", "skipped"}:
            print(flush=True)


def _mean(values: Iterable[Any]) -> float:
    finite = [float(value) for value in values if value not in [None, ""]]
    return float("nan") if not finite else sum(finite) / len(finite)


def _std(values: Iterable[Any]) -> float:
    finite = [float(value) for value in values if value not in [None, ""]]
    if len(finite) <= 1:
        return 0.0 if len(finite) == 1 else float("nan")
    mean = sum(finite) / len(finite)
    return math.sqrt(sum((value - mean) ** 2 for value in finite) / (len(finite) - 1))


def _format_cell(value: Any) -> str:
    if value in [None, ""]:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6f}"
    return str(value)


RAW_COLUMNS = [
    "task",
    "method",
    "seed",
    "repeat",
    "planner_type",
    "policy",
    "dataset_h5",
    "diffusion_bundle",
    "diffusion_selection_mode",
    "success_rate",
    "episode_successes",
    "evaluation_time_sec",
    "planning_time_total_sec",
    "avg_planning_time_sec",
    "global_planning_calls",
    "effective_replans_per_episode",
    "final_goal_distance_mean",
    "min_goal_distance_mean",
    "steps_to_success_mean",
    "path_length_mean",
    "straight_line_ratio_mean",
    "action_l2_mean_mean",
    "action_delta_l2_mean_mean",
    "action_jerk_l2_mean_mean",
    "diffusion_runtime_execute_steps",
    "diffusion_num_candidates",
    "diffusion_truncation_steps",
    "diffusion_start_timestep",
    "diffusion_refinement_enabled",
    "diffusion_refinement_steps",
    "diffusion_refinement_step_size",
    "diffusion_refinement_topk",
    "diffusion_refinement_goal_weight",
    "diffusion_refinement_prior_weight",
    "diffusion_refinement_smoothness_weight",
    "diffusion_refinement_grad_clip_norm",
    "avg_generation_time_sec",
    "avg_scoring_time_sec",
    "avg_selection_time_sec",
    "refinement_time_total_sec",
    "avg_refinement_time_sec",
    "finite_candidate_rate",
    "all_bad_env_rate",
    "fallback_rate",
    "selected_wm_cost_first",
    "selected_model_score_first",
    "selected_wm_cost_mean",
    "selected_model_score_mean",
    "refinement_candidate_count",
    "refinement_steps_observed",
    "refinement_cost_before",
    "refinement_cost_after",
    "refinement_goal_before",
    "refinement_goal_after",
    "refinement_delta_norm",
    "corrective_mode",
    "corrective_correction_interval",
    "corrective_effective_error_interval",
    "corrective_effective_execute_horizon",
    "corrective_error_threshold",
    "corrective_trigger_stat",
    "corrective_trigger_quantile",
    "corrective_trigger_scope",
    "corrective_error_metric",
    "corrector_path",
    "corrective_check_count",
    "corrective_replan_count",
    "corrective_replan_rate",
    "corrective_correction_count",
    "mean_prediction_error_before_replan",
    "max_prediction_error_before_replan",
    "mean_correction_norm",
    "mean_action_delta_norm",
    "correction_time_total_sec",
    "avg_correction_time_sec",
    "prediction_error_count",
    "prediction_error_episode_mean_count",
    "prediction_error_mean",
    "prediction_error_max",
    "successful_prediction_error_mean",
    "failed_prediction_error_mean",
    "prediction_error_fail_minus_success",
    "prediction_error_fail_success_ratio",
    "prediction_error_cohens_d_fail_vs_success",
    "log_path",
    "command",
    "returncode",
    "dry_run",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in keys), []).append(row)
    return grouped


def build_seed_summary_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for (task, method, seed), rows in sorted(_group_rows(raw_rows, ("task", "method", "seed")).items()):
        output.append(
            {
                "task": task,
                "method": method,
                "seed": seed,
                "success_rate_mean": _mean(row.get("success_rate") for row in rows),
                "success_rate_std": _std(row.get("success_rate") for row in rows),
                "evaluation_time_sec_mean": _mean(row.get("evaluation_time_sec") for row in rows),
                "evaluation_time_sec_std": _std(row.get("evaluation_time_sec") for row in rows),
                "planning_time_total_sec_mean": _mean(row.get("planning_time_total_sec") for row in rows),
                "planning_time_total_sec_std": _std(row.get("planning_time_total_sec") for row in rows),
                "action_l2_mean_mean": _mean(row.get("action_l2_mean_mean") for row in rows),
                "action_l2_mean_std": _std(row.get("action_l2_mean_mean") for row in rows),
                "action_delta_l2_mean_mean": _mean(row.get("action_delta_l2_mean_mean") for row in rows),
                "action_delta_l2_mean_std": _std(row.get("action_delta_l2_mean_mean") for row in rows),
                "action_jerk_l2_mean_mean": _mean(row.get("action_jerk_l2_mean_mean") for row in rows),
                "action_jerk_l2_mean_std": _std(row.get("action_jerk_l2_mean_mean") for row in rows),
                "final_goal_distance_mean": _mean(row.get("final_goal_distance_mean") for row in rows),
                "min_goal_distance_mean": _mean(row.get("min_goal_distance_mean") for row in rows),
                "steps_to_success_mean": _mean(row.get("steps_to_success_mean") for row in rows),
                "path_length_mean": _mean(row.get("path_length_mean") for row in rows),
                "straight_line_ratio_mean": _mean(row.get("straight_line_ratio_mean") for row in rows),
            }
        )
    return output


def build_final_summary_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    grouped = _group_rows(seed_rows, ("task", "method"))
    mpc_time_by_task = {
        task: _mean(row.get("evaluation_time_sec_mean") for row in rows)
        for (task, method), rows in grouped.items()
        if method == "mpc_cem"
    }
    mpc_planning_by_task = {
        task: _mean(row.get("planning_time_total_sec_mean") for row in rows)
        for (task, method), rows in grouped.items()
        if method == "mpc_cem"
    }
    for (task, method), rows in sorted(grouped.items()):
        eval_time = _mean(row.get("evaluation_time_sec_mean") for row in rows)
        planning_time = _mean(row.get("planning_time_total_sec_mean") for row in rows)
        mpc_time = mpc_time_by_task.get(task, float("nan"))
        mpc_planning = mpc_planning_by_task.get(task, float("nan"))
        output.append(
            {
                "task": task,
                "method": method,
                "success_rate_mean": _mean(row.get("success_rate_mean") for row in rows),
                "success_rate_std": _std(row.get("success_rate_mean") for row in rows),
                "evaluation_time_sec_mean": eval_time,
                "evaluation_time_sec_std": _std(row.get("evaluation_time_sec_mean") for row in rows),
                "planning_time_total_sec_mean": planning_time,
                "planning_time_total_sec_std": _std(row.get("planning_time_total_sec_mean") for row in rows),
                "speedup_vs_mpc": (mpc_time / eval_time) if eval_time and math.isfinite(mpc_time) else "",
                "planning_speedup_vs_mpc": (
                    mpc_planning / planning_time
                    if planning_time and math.isfinite(mpc_planning)
                    else ""
                ),
                "final_goal_distance_mean": _mean(row.get("final_goal_distance_mean") for row in rows),
                "min_goal_distance_mean": _mean(row.get("min_goal_distance_mean") for row in rows),
                "steps_to_success_mean": _mean(row.get("steps_to_success_mean") for row in rows),
                "path_length_mean": _mean(row.get("path_length_mean") for row in rows),
                "straight_line_ratio_mean": _mean(row.get("straight_line_ratio_mean") for row in rows),
                "action_l2_mean_mean": _mean(row.get("action_l2_mean_mean") for row in rows),
                "action_delta_l2_mean_mean": _mean(row.get("action_delta_l2_mean_mean") for row in rows),
                "action_jerk_l2_mean_mean": _mean(row.get("action_jerk_l2_mean_mean") for row in rows),
            }
        )
    return output


def write_result_markdown(
    *,
    path: Path,
    raw_rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Comparison Experiment Results",
        "",
        "本文档由 `scripts/run_comparison_experiments.py` 生成或更新。",
        "",
        "## Raw Runs",
        "",
        "| Task | Method | Seed | Repeat | Success Rate | Eval Time Sec | Planning Total Sec | Avg Planning Sec | Calls | Replans/Ep | Final Goal Dist | Min Goal Dist | Steps To Success | Path Length | Straight Ratio | Action L2 | Action Delta | Action Jerk | Log Path |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in raw_rows:
        lines.append(
            "| "
            + " | ".join(
                _format_cell(row.get(key))
                for key in [
                    "task",
                    "method",
                    "seed",
                    "repeat",
                    "success_rate",
                    "evaluation_time_sec",
                    "planning_time_total_sec",
                    "avg_planning_time_sec",
                    "global_planning_calls",
                    "effective_replans_per_episode",
                    "final_goal_distance_mean",
                    "min_goal_distance_mean",
                    "steps_to_success_mean",
                    "path_length_mean",
                    "straight_line_ratio_mean",
                    "action_l2_mean_mean",
                    "action_delta_l2_mean_mean",
                    "action_jerk_l2_mean_mean",
                    "log_path",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Seed Summary",
            "",
            "| Task | Method | Seed | Success Rate Mean | Success Rate Std | Eval Time Mean | Eval Time Std | Planning Time Mean | Planning Time Std | Action Jerk Mean | Action Jerk Std |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in seed_rows:
        lines.append(
            "| "
            + " | ".join(
                _format_cell(row.get(key))
                for key in [
                    "task",
                    "method",
                    "seed",
                    "success_rate_mean",
                    "success_rate_std",
                    "evaluation_time_sec_mean",
                    "evaluation_time_sec_std",
                    "planning_time_total_sec_mean",
                    "planning_time_total_sec_std",
                    "action_jerk_l2_mean_mean",
                    "action_jerk_l2_mean_std",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Final Summary",
            "",
            "| Task | Method | Success Rate Mean | Success Rate Std | Eval Time Mean | Eval Time Std | Planning Time Mean | Planning Time Std | Speedup vs MPC | Planning Speedup vs MPC | Final Goal Dist Mean | Min Goal Dist Mean | Steps To Success | Path Length | Straight Ratio | Action L2 | Action Delta | Action Jerk |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in final_rows:
        lines.append(
            "| "
            + " | ".join(
                _format_cell(row.get(key))
                for key in [
                    "task",
                    "method",
                    "success_rate_mean",
                    "success_rate_std",
                    "evaluation_time_sec_mean",
                    "evaluation_time_sec_std",
                    "planning_time_total_sec_mean",
                    "planning_time_total_sec_std",
                    "speedup_vs_mpc",
                    "planning_speedup_vs_mpc",
                    "final_goal_distance_mean",
                    "min_goal_distance_mean",
                    "steps_to_success_mean",
                    "path_length_mean",
                    "straight_line_ratio_mean",
                    "action_l2_mean_mean",
                    "action_delta_l2_mean_mean",
                    "action_jerk_l2_mean_mean",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Commands",
            "",
            "完整命令保存在 CSV 的 `command` 字段；每次 run 的 stdout/stderr 保存在对应 `log_path`。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_existing_raw_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_is_complete_for_command(row: dict[str, Any], command: list[str]) -> bool:
    return (
        row.get("command") == " ".join(command)
        and str(row.get("returncode", "")) == "0"
        and str(row.get("dry_run", "")).lower() not in {"1", "true", "yes"}
    )


def run_experiments(
    *,
    spec: ExperimentSpec,
    output_root: Path,
    result_path: Path,
    python_bin: Path,
    dry_run: bool = False,
    force: bool = False,
    progress: bool = True,
    runner: Callable[[list[str], Path], object] | None = None,
) -> dict[str, int]:
    output_root.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    raw_csv = output_root / "raw_runs.csv"
    seed_csv = output_root / "seed_summary.csv"
    final_csv = output_root / "final_summary.csv"
    raw_rows = _load_existing_raw_rows(raw_csv) if not force else []
    existing = {
        (row["task"], row["method"], str(row["seed"]), str(row["repeat"])): row
        for row in raw_rows
    }
    runs = build_run_matrix(spec)
    printer = ProgressPrinter(enabled=progress)
    completed = 0
    skipped = 0

    for index, run in enumerate(runs, start=1):
        key = (run.task, run.method, str(run.seed), str(run.repeat))
        command = build_eval_command(run, spec=spec, python_bin=python_bin)
        log_path = output_root / run.task / f"{run.method}_seed{run.seed}_repeat{run.repeat}.log"
        existing_row = existing.get(key)
        if existing_row is not None and not force and _row_is_complete_for_command(existing_row, command):
            skipped += 1
            printer.update(index, len(runs), run, "skipped")
            continue

        printer.update(index - 1, len(runs), run, "running")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if existing_row is not None:
            raw_rows = [
                row for row in raw_rows
                if (row["task"], row["method"], str(row["seed"]), str(row["repeat"])) != key
            ]

        if dry_run:
            log_path.write_text(
                "\n".join(
                    [
                        "[dry-run] command:",
                        " ".join(command),
                        "[summary] success_rate=0.0000",
                        "[summary] evaluation_time=0.0000s",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            returncode = 0
        elif runner is not None:
            runner(command, log_path)
            returncode = 0
        else:
            with log_path.open("w", encoding="utf-8") as handle:
                process = subprocess.Popen(
                    command,
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    handle.write(line)
                    handle.flush()
                    if line.startswith("[summary]") or line.startswith("[planner-stats]"):
                        print(line.rstrip(), flush=True)
                returncode = process.wait()

        metrics = parse_eval_log_file(log_path)
        row: dict[str, Any] = {
            "task": run.task,
            "method": run.method,
            "seed": run.seed,
            "repeat": run.repeat,
            "dataset_h5": spec.tasks[run.task].dataset_h5,
            "log_path": str(log_path),
            "command": " ".join(command),
            "returncode": returncode,
            "dry_run": int(bool(dry_run)),
        }
        row.update(metrics)
        raw_rows.append(row)
        completed += 1
        _write_outputs(
            raw_rows=raw_rows,
            raw_csv=raw_csv,
            seed_csv=seed_csv,
            final_csv=final_csv,
            result_path=result_path,
        )
        if returncode != 0:
            printer.update(index, len(runs), run, "failed")
            raise subprocess.CalledProcessError(returncode, command)
        printer.update(index, len(runs), run, "done")

    seed_rows = build_seed_summary_rows(raw_rows)
    final_rows = build_final_summary_rows(seed_rows)
    return {
        "total_runs": len(runs),
        "completed_runs": completed,
        "skipped_runs": skipped,
        "raw_rows": len(raw_rows),
        "seed_rows": len(seed_rows),
        "final_rows": len(final_rows),
    }


def _write_outputs(
    *,
    raw_rows: list[dict[str, Any]],
    raw_csv: Path,
    seed_csv: Path,
    final_csv: Path,
    result_path: Path,
) -> None:
    seed_rows = build_seed_summary_rows(raw_rows)
    final_rows = build_final_summary_rows(seed_rows)
    _write_csv(raw_csv, raw_rows, RAW_COLUMNS)
    _write_csv(
        seed_csv,
        seed_rows,
        [
            "task",
            "method",
            "seed",
            "success_rate_mean",
            "success_rate_std",
            "evaluation_time_sec_mean",
            "evaluation_time_sec_std",
            "planning_time_total_sec_mean",
            "planning_time_total_sec_std",
            "action_jerk_l2_mean_mean",
            "action_jerk_l2_mean_std",
            "action_l2_mean_mean",
            "action_l2_mean_std",
            "action_delta_l2_mean_mean",
            "action_delta_l2_mean_std",
            "final_goal_distance_mean",
            "min_goal_distance_mean",
            "steps_to_success_mean",
            "path_length_mean",
            "straight_line_ratio_mean",
        ],
    )
    _write_csv(
        final_csv,
        final_rows,
        [
            "task",
            "method",
            "success_rate_mean",
            "success_rate_std",
            "evaluation_time_sec_mean",
            "evaluation_time_sec_std",
            "planning_time_total_sec_mean",
            "planning_time_total_sec_std",
            "speedup_vs_mpc",
            "planning_speedup_vs_mpc",
            "final_goal_distance_mean",
            "min_goal_distance_mean",
            "steps_to_success_mean",
            "path_length_mean",
            "straight_line_ratio_mean",
            "action_l2_mean_mean",
            "action_delta_l2_mean_mean",
            "action_jerk_l2_mean_mean",
        ],
    )
    write_result_markdown(
        path=result_path,
        raw_rows=raw_rows,
        seed_rows=seed_rows,
        final_rows=final_rows,
    )


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 4-task comparison experiment matrix.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/comparison_experiments"))
    parser.add_argument("--result-path", type=Path, default=Path("result.md"))
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--tasks", default="cube,pusht,reacher,tworoom")
    parser.add_argument("--methods", default="mpc_cem,gc_idm,ours_full")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--repeats", default="0,1,2")
    parser.add_argument("--eval-num-eval", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run rows already present in raw_runs.csv.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--pusht-ours-profile", choices=["corrective_learned", "corrective_replan"], default="corrective_learned")
    parser.add_argument("--cube-dataset-h5", default="/data/ykz/cube/cube_single_expert.h5")
    parser.add_argument("--pusht-dataset-h5", default="/data/ykz/pusht/pusht_expert_train.h5")
    parser.add_argument("--reacher-dataset-h5", default="/data/ykz/reacher/reacher.h5")
    parser.add_argument("--tworoom-dataset-h5", default="/data/ykz/tworoom/tworoom.h5")
    return parser.parse_args(argv)


def spec_from_args(args: argparse.Namespace) -> ExperimentSpec:
    default = default_experiment_spec()
    selected_tasks = parse_csv_strings(args.tasks)
    selected_methods = parse_csv_strings(args.methods)
    pusht_ours = (
        f"eval_profile={args.pusht_ours_profile}",
        "diffusion_selection_mode=wm_only",
        "diffusion_refinement.enabled=true",
    )
    task_specs = {
        "cube": TaskSpec(
            config_name="cube",
            dataset_h5=str(args.cube_dataset_h5),
            ours_overrides=default.tasks["cube"].ours_overrides,
        ),
        "pusht": TaskSpec(
            config_name="pusht",
            dataset_h5=str(args.pusht_dataset_h5),
            ours_overrides=pusht_ours,
        ),
        "reacher": TaskSpec(
            config_name="reacher",
            dataset_h5=str(args.reacher_dataset_h5),
            ours_overrides=default.tasks["reacher"].ours_overrides,
        ),
        "tworoom": TaskSpec(
            config_name="tworoom",
            dataset_h5=str(args.tworoom_dataset_h5),
            ours_overrides=default.tasks["tworoom"].ours_overrides,
        ),
    }
    missing_tasks = set(selected_tasks) - set(task_specs)
    if missing_tasks:
        raise ValueError(f"Unsupported tasks: {sorted(missing_tasks)}")
    return ExperimentSpec(
        tasks={task: task_specs[task] for task in selected_tasks},
        methods=selected_methods,
        seeds=parse_csv_ints(args.seeds),
        repeats=parse_csv_ints(args.repeats),
        eval_num_eval=int(args.eval_num_eval),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = spec_from_args(args)
    summary = run_experiments(
        spec=spec,
        output_root=args.output_root,
        result_path=args.result_path,
        python_bin=args.python_bin,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        progress=not bool(args.no_progress),
    )
    print(f"[comparison-summary] {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
