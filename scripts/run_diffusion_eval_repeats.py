#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"
EVAL_PY = REPO_ROOT / "eval.py"


@dataclass(frozen=True)
class TaskEvalConfig:
    task: str
    config_name: str
    policy: str
    diffusion_bundle: str
    cache_dir: str


DEFAULT_TASKS: dict[str, TaskEvalConfig] = {
    "pusht": TaskEvalConfig(
        task="pusht",
        config_name="pusht.yaml",
        policy="/data/yuekangzhou/pusht/lewm_epoch_100",
        diffusion_bundle="/data/yuekangzhou/diffusion_runs/pusht_diffusion_200k_simple_bce_k128_splittrain/diffusion_planner_best_bundle.pt",
        cache_dir="/data/yuekangzhou/pusht/splits/pusht_expert_train_test",
    ),
    "tworoom": TaskEvalConfig(
        task="tworoom",
        config_name="tworoom.yaml",
        policy="/data/yuekangzhou/tworoom/lewm_epoch_66",
        diffusion_bundle="/data/yuekangzhou/diffusion_runs/tworoom_diffusion_200k_simple_bce_k128_splittrain/diffusion_planner_best_bundle.pt",
        cache_dir="/data/yuekangzhou/tworoom/splits/tworoom_test",
    ),
}


SUCCESS_RE = re.compile(r"^\[summary\]\s+success_rate=([0-9.]+)\s*$", re.MULTILINE)
TIME_RE = re.compile(r"^\[summary\]\s+evaluation_time=([0-9.]+)s\s*$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated diffusion-policy evaluations for the split-test PushT and TwoRoom tasks, "
            "then save success-rate results to a JSON file for plotting."
        )
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["pusht", "tworoom"],
        choices=sorted(DEFAULT_TASKS.keys()),
        help="Tasks to evaluate. Default: pusht tworoom",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repeated eval runs per task. Default: 10",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="First seed to use. Seeds are [start_seed, start_seed + repeats). Default: 0",
    )
    parser.add_argument(
        "--eval-num-eval",
        type=int,
        default=50,
        help="Hydra override for eval.num_eval. Default: 50",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=128,
        help="Hydra override for diffusion_num_candidates. Default: 128",
    )
    parser.add_argument(
        "--truncation-steps",
        type=int,
        default=4,
        help="Hydra override for diffusion_truncation_steps. Default: 4",
    )
    parser.add_argument(
        "--start-timestep",
        type=int,
        default=15,
        help="Hydra override for diffusion_start_timestep. Default: 15",
    )
    parser.add_argument(
        "--selection-mode",
        default="wm_only",
        help="Hydra override for diffusion_selection_mode. Default: wm_only",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="Hydra override for diffusion_eta. Default: 0.0",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Hydra override for diffusion_noise_scale. Default: 1.0",
    )
    parser.add_argument(
        "--sampling-temperature",
        type=float,
        default=1.0,
        help="Hydra override for diffusion_sampling_temperature. Default: 1.0",
    )
    parser.add_argument(
        "--log-root",
        default="/data/yuekangzhou/diffusion_runs/repeat_eval_logs",
        help="Directory for per-run logs and the result JSON.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional explicit JSON output path. Default: /home/yuekangzhou/lewm-diffusion/pusht_tworoom_10runs.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    return parser.parse_args()


def ensure_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def build_eval_command(cfg: TaskEvalConfig, args: argparse.Namespace, seed: int) -> list[str]:
    return [
        str(PYTHON_BIN),
        str(EVAL_PY),
        f"--config-name={cfg.config_name}",
        "planner_type=diffusion",
        f"policy={cfg.policy}",
        f"diffusion_bundle={cfg.diffusion_bundle}",
        f"diffusion_selection_mode={args.selection_mode}",
        f"diffusion_num_candidates={args.num_candidates}",
        f"diffusion_truncation_steps={args.truncation_steps}",
        f"diffusion_start_timestep={args.start_timestep}",
        f"diffusion_eta={args.eta}",
        f"diffusion_noise_scale={args.noise_scale}",
        f"diffusion_sampling_temperature={args.sampling_temperature}",
        f"cache_dir={cfg.cache_dir}",
        f"eval.num_eval={args.eval_num_eval}",
        f"seed={seed}",
    ]


def parse_summary(stdout: str) -> tuple[float, float]:
    success_match = SUCCESS_RE.search(stdout)
    time_match = TIME_RE.search(stdout)
    if success_match is None:
        raise ValueError("Could not parse [summary] success_rate=... from eval output.")
    if time_match is None:
        raise ValueError("Could not parse [summary] evaluation_time=...s from eval output.")
    return float(success_match.group(1)), float(time_match.group(1))


def maybe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def build_summary(runs: list[dict[str, Any]]) -> dict[str, float]:
    success_rates = [float(run["success_rate"]) for run in runs]
    evaluation_times = [float(run["evaluation_time_sec"]) for run in runs]
    return {
        "mean_success_rate": float(statistics.mean(success_rates)),
        "std_success_rate": maybe_std(success_rates),
        "mean_evaluation_time_sec": float(statistics.mean(evaluation_times)),
        "std_evaluation_time_sec": maybe_std(evaluation_times),
        "num_runs": float(len(runs)),
    }


def main() -> int:
    args = parse_args()
    ensure_positive("repeats", args.repeats)
    ensure_positive("eval-num-eval", args.eval_num_eval)
    ensure_positive("num-candidates", args.num_candidates)
    ensure_positive("truncation-steps", args.truncation_steps)
    ensure_positive("start-timestep", args.start_timestep)

    log_root = Path(args.log_root).expanduser()
    output_json = (
        Path(args.output_json).expanduser()
        if args.output_json
        else Path("/home/yuekangzhou/lewm-diffusion/pusht_tworoom_10runs.json")
    )

    result: dict[str, Any] = {
        "meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "repo_root": str(REPO_ROOT),
            "repeats": int(args.repeats),
            "start_seed": int(args.start_seed),
            "eval_num_eval": int(args.eval_num_eval),
            "selection_mode": str(args.selection_mode),
            "num_candidates": int(args.num_candidates),
            "truncation_steps": int(args.truncation_steps),
            "start_timestep": int(args.start_timestep),
            "eta": float(args.eta),
            "noise_scale": float(args.noise_scale),
            "sampling_temperature": float(args.sampling_temperature),
        },
        "tasks": {},
    }

    if args.dry_run:
        print(f"[dry-run] repo_root={REPO_ROOT}")
        print(f"[dry-run] log_root={log_root}")
        print(f"[dry-run] output_json={output_json}")

    for task_name in args.tasks:
        cfg = DEFAULT_TASKS[task_name]
        task_runs: list[dict[str, Any]] = []
        task_log_dir = log_root / task_name
        if not args.dry_run:
            task_log_dir.mkdir(parents=True, exist_ok=True)

        print(f"[task] {task_name}")
        for offset in range(args.repeats):
            seed = int(args.start_seed + offset)
            command = build_eval_command(cfg, args, seed)
            log_path = task_log_dir / f"seed_{seed}.log"
            command_text = " ".join(command)

            if args.dry_run:
                print(f"[dry-run] task={task_name} seed={seed} cmd={command_text}")
                continue

            print(f"[run] task={task_name} seed={seed}")
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            log_path.write_text(completed.stdout, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Eval failed for task={task_name} seed={seed} with exit code {completed.returncode}. "
                    f"See {log_path}."
                )

            success_rate, evaluation_time = parse_summary(completed.stdout)
            task_runs.append(
                {
                    "seed": seed,
                    "success_rate": success_rate,
                    "evaluation_time_sec": evaluation_time,
                    "log_path": str(log_path),
                }
            )
            print(
                f"[done] task={task_name} seed={seed} "
                f"success_rate={success_rate:.4f} evaluation_time_sec={evaluation_time:.4f}"
            )

        if not args.dry_run:
            result["tasks"][task_name] = {
                "config": asdict(cfg),
                "runs": task_runs,
                "summary": build_summary(task_runs),
            }

    if args.dry_run:
        return 0

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[saved] {output_json}")

    for task_name, task_result in result["tasks"].items():
        summary = task_result["summary"]
        print(
            f"[summary] task={task_name} "
            f"mean_success_rate={summary['mean_success_rate']:.4f} "
            f"std_success_rate={summary['std_success_rate']:.4f} "
            f"mean_evaluation_time_sec={summary['mean_evaluation_time_sec']:.4f} "
            f"std_evaluation_time_sec={summary['std_evaluation_time_sec']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
