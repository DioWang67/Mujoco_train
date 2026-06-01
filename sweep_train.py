"""Randomized Sedon curriculum sweep runner.

This script samples a small number of reward/control configurations, runs a
short PPO sanity training for each one, evaluates saved checkpoints, and writes
a ranked CSV summary. It is intentionally simple: the goal is to automate the
current local iteration loop, not to replace a full experiment platform.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sedon_baseline.env import (
    CONFIG_OVERRIDES_ENV,
    SedonStandingConfig,
    SedonStandingEnv,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = REPO_ROOT / "artifacts" / "sedon_sweeps"
MAX_EPISODE_STEPS = 400

SEARCH_SPACE: dict[str, list[float]] = {
    "action_joint_delta_scale": [0.03, 0.05, 0.06],
    "gait_hip_roll_amp": [0.0, 0.005, 0.01, 0.02],
    "upright_weight": [4.0, 5.0, 6.0],
    "height_weight": [3.0, 3.5, 4.0],
    "low_forward_velocity_penalty_weight": [2.0, 3.0, 4.0],
    "progress_reward_weight": [1.5, 3.0, 5.0],
}


@dataclass(frozen=True)
class EvalMetrics:
    """Aggregated checkpoint evaluation metrics."""

    model_name: str
    score: float
    passed: bool
    promote_pass: bool
    fall_rate: float
    mean_length: float
    mean_forward_vel: float
    mean_final_base_x: float
    mean_final_upright: float
    mean_final_base_z: float


@dataclass(frozen=True)
class GaitMetrics:
    """Metrics from a zero-action rollout before PPO training."""

    survived: bool
    steps: int
    mean_forward_vel: float
    final_base_x: float
    final_base_z: float
    final_upright: float


def _sample_configs(trials: int, seed: int) -> list[dict[str, float]]:
    """Sample unique random configurations from the small search space."""
    rng = random.Random(seed)
    configs: list[dict[str, float]] = []
    seen: set[str] = set()
    max_attempts = trials * 20
    for _ in range(max_attempts):
        config = {
            key: rng.choice(values)
            for key, values in SEARCH_SPACE.items()
        }
        signature = json.dumps(config, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        configs.append(config)
        if len(configs) >= trials:
            break
    return configs


def _load_configs_file(path: Path) -> list[dict[str, float]]:
    """Load explicit sweep configurations from a JSON file."""
    try:
        raw_configs = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FileNotFoundError(f"--configs-file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"--configs-file is not valid JSON: {path}") from exc

    if not isinstance(raw_configs, list) or not raw_configs:
        raise ValueError("--configs-file must contain a non-empty JSON array.")

    configs: list[dict[str, float]] = []
    for index, raw_config in enumerate(raw_configs, start=1):
        if not isinstance(raw_config, dict):
            raise ValueError(f"Config #{index} must be a JSON object.")
        config: dict[str, float] = {}
        for key, value in raw_config.items():
            if not isinstance(key, str):
                raise ValueError(f"Config #{index} contains a non-string key.")
            if not isinstance(value, int | float):
                raise ValueError(
                    f"Config #{index} value for {key!r} must be numeric."
                )
            config[key] = float(value)
        configs.append(config)
    return configs


def _build_trial_env(
    base_env: dict[str, str],
    *,
    run_root: Path,
    config: dict[str, float],
) -> dict[str, str]:
    """Return subprocess environment for one isolated trial."""
    env = dict(base_env)
    env[CONFIG_OVERRIDES_ENV] = json.dumps(config, sort_keys=True)
    env["MUJOCO_TRAIN_LAYOUT_ROOT"] = str(run_root)
    env["MUJOCO_TRAIN_PROJECT_SLUG"] = "sedon"
    return env


def _with_termination_thresholds(
    config: dict[str, float],
    *,
    min_base_height: float,
    min_upright: float,
) -> dict[str, float]:
    """Return config with explicit termination thresholds."""
    return {
        **config,
        "min_base_height": min_base_height,
        "min_upright": min_upright,
    }


def _build_eval_env(
    train_env: dict[str, str],
    *,
    eval_config: dict[str, float],
) -> dict[str, str]:
    """Return subprocess environment for strict evaluation."""
    env = dict(train_env)
    env[CONFIG_OVERRIDES_ENV] = json.dumps(eval_config, sort_keys=True)
    return env


def _run_command(
    command: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
    timeout_seconds: int | None,
) -> int:
    """Run a subprocess and capture stdout/stderr into a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        try:
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            log_file.write(
                f"\n[timeout] command exceeded {timeout_seconds} seconds.\n"
            )
            return 124


def _zero_action_rollout(config_overrides: dict[str, float]) -> GaitMetrics:
    """Run fixed-gait zero-action rollout for prefiltering."""
    base_config = SedonStandingConfig()
    reward_config = SedonStandingConfig(
        **{**base_config.__dict__, **config_overrides}
    )
    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=reward_config)
    try:
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float64)
        forward_sum = 0.0
        last_info: dict[str, Any] = {}
        steps = 0
        for step in range(MAX_EPISODE_STEPS):
            _, _, terminated, truncated, last_info = env.step(action)
            forward_sum += float(last_info.get("forward_velocity", 0.0))
            steps = step + 1
            if terminated or truncated:
                break
        return GaitMetrics(
            survived=steps >= MAX_EPISODE_STEPS,
            steps=steps,
            mean_forward_vel=forward_sum / max(1, steps),
            final_base_x=float(last_info.get("base_x_position", 0.0)),
            final_base_z=float(last_info.get("base_height", 0.0)),
            final_upright=float(last_info.get("upright", 0.0)),
        )
    finally:
        env.close()


def _candidate_checkpoints(models_root: Path) -> list[tuple[str, Path, Path]]:
    """Return model/VecNormalize pairs to evaluate for one trial."""
    candidates = [
        ("latest", models_root / "latest_model.zip", models_root / "vecnorm.pkl"),
        ("best", models_root / "best" / "best_model.zip", models_root / "best" / "vecnorm.pkl"),
        (
            "forward_best",
            models_root / "forward_best" / "forward_best_model.zip",
            models_root / "forward_best" / "vecnorm.pkl",
        ),
        (
            "stable_forward_best",
            models_root / "stable_forward_best" / "stable_forward_best_model.zip",
            models_root / "stable_forward_best" / "vecnorm.pkl",
        ),
    ]
    return [
        (name, model_path, vecnorm_path)
        for name, model_path, vecnorm_path in candidates
        if model_path.is_file() and vecnorm_path.is_file()
    ]


def _parse_eval_csv(csv_path: Path, model_name: str) -> EvalMetrics:
    """Parse Sedon eval CSV and compute sweep score."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Eval CSV has no rows: {csv_path}")

    lengths = [float(row["length"]) for row in rows]
    final_base_x = [float(row["final_base_x"]) for row in rows]
    final_base_z = [float(row["final_base_height"]) for row in rows]
    final_uprights = [float(row["final_upright"]) for row in rows]
    forward_vels = [float(row["mean_forward_velocity"]) for row in rows]
    falls = [
        length < MAX_EPISODE_STEPS
        or base_z < 0.34
        or upright < 0.75
        for length, base_z, upright in zip(lengths, final_base_z, final_uprights)
    ]
    fall_rate = 100.0 * sum(falls) / max(1, len(falls))
    mean_length = float(np.mean(lengths))
    mean_forward_vel = float(np.mean(forward_vels))
    mean_final_base_x = float(np.mean(final_base_x))
    mean_final_upright = float(np.mean(final_uprights))
    mean_final_base_z = float(np.mean(final_base_z))
    score = (
        mean_forward_vel * 60.0
        + mean_final_base_x * 15.0
        + mean_final_upright * 20.0
        + mean_final_base_z * 10.0
        + mean_length * 0.05
        - fall_rate * 3.0
    )
    if mean_forward_vel <= 0.005 or mean_final_base_x <= 0.05:
        score -= 50.0
    passed = (
        fall_rate < 60.0
        and mean_length > 300.0
        and mean_forward_vel > 0.015
        and mean_final_upright > 0.72
        and mean_final_base_z > 0.34
    )
    promote_pass = (
        fall_rate <= 20.0
        and mean_length >= 380.0
        and mean_forward_vel >= 0.010
        and mean_final_base_x >= 0.18
        and mean_final_upright >= 0.80
        and mean_final_base_z >= 0.37
    )
    return EvalMetrics(
        model_name=model_name,
        score=score,
        passed=passed,
        promote_pass=promote_pass,
        fall_rate=fall_rate,
        mean_length=mean_length,
        mean_forward_vel=mean_forward_vel,
        mean_final_base_x=mean_final_base_x,
        mean_final_upright=mean_final_upright,
        mean_final_base_z=mean_final_base_z,
    )


def _evaluate_trial(
    *,
    env: dict[str, str],
    run_dir: Path,
    models_root: Path,
    episodes: int,
    timeout_seconds: int,
) -> EvalMetrics | None:
    """Evaluate all available checkpoints and return the best by score."""
    metrics: list[EvalMetrics] = []
    for model_name, model_path, vecnorm_path in _candidate_checkpoints(models_root):
        out_csv = run_dir / f"eval_{model_name}.csv"
        command = [
            sys.executable,
            "eval.py",
            "--project",
            "sedon",
            "--episodes",
            str(episodes),
            "--model-path",
            str(model_path),
            "--vecnorm-path",
            str(vecnorm_path),
            "--out-csv",
            str(out_csv),
        ]
        exit_code = _run_command(
            command,
            env=env,
            log_path=run_dir / f"eval_{model_name}.log",
            timeout_seconds=timeout_seconds,
        )
        if exit_code != 0:
            continue
        metrics.append(_parse_eval_csv(out_csv, model_name))
    if not metrics:
        return None
    checkpoint_priority = {
        "stable_forward_best": 4,
        "forward_best": 3,
        "best": 2,
        "latest": 1,
    }
    return max(
        metrics,
        key=lambda item: (
            item.promote_pass,
            item.passed,
            checkpoint_priority.get(item.model_name, 0),
            item.score,
        ),
    )


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write ranked sweep results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "config_id",
        "status",
        "best_model",
        "score",
        "passed",
        "promote_pass",
        "fall_rate",
        "mean_length",
        "mean_forward_vel",
        "mean_final_base_x",
        "mean_final_upright",
        "mean_final_base_z",
        "gait_steps",
        "gait_mean_forward_vel",
        "gait_final_base_x",
        "config_json",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows ordered by promotion gates and score."""
    ranked = sorted(
        rows,
        key=lambda row: (
            str(row.get("promote_pass", "")).lower() == "true",
            str(row.get("passed", "")).lower() == "true",
            float(row.get("score") or -1e9),
        ),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _record_row(
    rows: list[dict[str, Any]],
    summary_path: Path,
    row: dict[str, Any],
) -> None:
    """Append one result row and persist the current ranked summary."""
    rows.append(row)
    _write_summary(summary_path, _rank_rows(rows))


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--configs-file", type=Path)
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--action-std", type=float, default=0.05)
    parser.add_argument("--reset-noise-scale", type=float, default=0.0)
    parser.add_argument("--min-gait-steps", type=int, default=60)
    parser.add_argument("--train-min-upright", type=float, default=0.70)
    parser.add_argument("--train-min-base-height", type=float, default=0.32)
    parser.add_argument("--eval-min-upright", type=float, default=0.75)
    parser.add_argument("--eval-min-base-height", type=float, default=0.34)
    parser.add_argument("--train-timeout-seconds", type=int, default=900)
    parser.add_argument("--eval-timeout-seconds", type=int, default=300)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run randomized Sedon training sweep."""
    args = build_parser().parse_args(argv)
    if args.trials <= 0:
        raise ValueError("--trials must be positive.")
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive.")
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive.")
    if args.train_timeout_seconds <= 0:
        raise ValueError("--train-timeout-seconds must be positive.")
    if args.eval_timeout_seconds <= 0:
        raise ValueError("--eval-timeout-seconds must be positive.")

    if args.configs_file is not None:
        configs = _load_configs_file(args.configs_file)
    else:
        configs = _sample_configs(args.trials, args.seed)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sweep_root = args.out_root / f"sweep_{timestamp}"
    summary_path = sweep_root / "summary.csv"
    rows: list[dict[str, Any]] = []
    base_env = os.environ.copy()

    print(f"sweep_root: {sweep_root}")
    for index, config in enumerate(configs, start=1):
        config_id = f"run_{index:02d}"
        run_dir = sweep_root / config_id
        print(f"\n==> {config_id}: {json.dumps(config, sort_keys=True)}")
        if args.dry_run:
            train_config = _with_termination_thresholds(
                config,
                min_base_height=args.train_min_base_height,
                min_upright=args.train_min_upright,
            )
            _record_row(
                rows,
                summary_path,
                {
                    "config_id": config_id,
                    "status": "dry_run",
                    "config_json": json.dumps(train_config, sort_keys=True),
                    "run_dir": str(run_dir),
                }
            )
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        train_config = _with_termination_thresholds(
            config,
            min_base_height=args.train_min_base_height,
            min_upright=args.train_min_upright,
        )
        eval_config = _with_termination_thresholds(
            config,
            min_base_height=args.eval_min_base_height,
            min_upright=args.eval_min_upright,
        )
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "sampled": config,
                    "train": train_config,
                    "eval": eval_config,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        try:
            gait = _zero_action_rollout(train_config)
        except Exception as exc:  # noqa: BLE001 - sweep should continue on bad configs.
            print(f"[skip] zero-action rollout failed: {exc}")
            _record_row(
                rows,
                summary_path,
                {
                    "config_id": config_id,
                    "status": f"gait_error: {exc}",
                    "config_json": json.dumps(train_config, sort_keys=True),
                    "run_dir": str(run_dir),
                }
            )
            continue

        if gait.steps < args.min_gait_steps:
            print(f"[skip] gait prefilter failed: steps={gait.steps}")
            _record_row(
                rows,
                summary_path,
                {
                    "config_id": config_id,
                    "status": "gait_prefilter_failed",
                    "gait_steps": gait.steps,
                    "gait_mean_forward_vel": gait.mean_forward_vel,
                    "gait_final_base_x": gait.final_base_x,
                    "config_json": json.dumps(train_config, sort_keys=True),
                    "run_dir": str(run_dir),
                }
            )
            continue

        trial_env = _build_trial_env(base_env, run_root=run_dir, config=train_config)
        eval_env = _build_eval_env(trial_env, eval_config=eval_config)
        train_command = [
            sys.executable,
            "train.py",
            "--project",
            "sedon",
            "--n-envs",
            str(args.n_envs),
            "--total-timesteps",
            str(args.timesteps),
            "--reset-noise-scale",
            str(args.reset_noise_scale),
            "--action-std",
            str(args.action_std),
        ]
        exit_code = _run_command(
            train_command,
            env=trial_env,
            log_path=run_dir / "train.log",
            timeout_seconds=args.train_timeout_seconds,
        )
        if exit_code != 0:
            _record_row(
                rows,
                summary_path,
                {
                    "config_id": config_id,
                    "status": f"train_failed:{exit_code}",
                    "gait_steps": gait.steps,
                    "gait_mean_forward_vel": gait.mean_forward_vel,
                    "gait_final_base_x": gait.final_base_x,
                    "config_json": json.dumps(train_config, sort_keys=True),
                    "run_dir": str(run_dir),
                }
            )
            continue

        models_root = run_dir / "runs" / "sedon" / "models" / "sedon"
        best_metrics = _evaluate_trial(
            env=eval_env,
            run_dir=run_dir,
            models_root=models_root,
            episodes=args.episodes,
            timeout_seconds=args.eval_timeout_seconds,
        )
        if best_metrics is None:
            _record_row(
                rows,
                summary_path,
                {
                    "config_id": config_id,
                    "status": "eval_failed",
                    "gait_steps": gait.steps,
                    "gait_mean_forward_vel": gait.mean_forward_vel,
                    "gait_final_base_x": gait.final_base_x,
                    "config_json": json.dumps(train_config, sort_keys=True),
                    "run_dir": str(run_dir),
                }
            )
            continue

        _record_row(
            rows,
            summary_path,
            {
                "config_id": config_id,
                "status": "ok",
                "best_model": best_metrics.model_name,
                "score": best_metrics.score,
                "passed": best_metrics.passed,
                "promote_pass": best_metrics.promote_pass,
                "fall_rate": best_metrics.fall_rate,
                "mean_length": best_metrics.mean_length,
                "mean_forward_vel": best_metrics.mean_forward_vel,
                "mean_final_base_x": best_metrics.mean_final_base_x,
                "mean_final_upright": best_metrics.mean_final_upright,
                "mean_final_base_z": best_metrics.mean_final_base_z,
                "gait_steps": gait.steps,
                "gait_mean_forward_vel": gait.mean_forward_vel,
                "gait_final_base_x": gait.final_base_x,
                "config_json": json.dumps(train_config, sort_keys=True),
                "run_dir": str(run_dir),
            }
        )

    ranked = _rank_rows(rows)
    _write_summary(summary_path, ranked)

    print(f"\nsummary: {summary_path}")
    print("rank config_id score fall_rate mean_fwd final_x upright promote passed")
    for row in ranked[: min(10, len(ranked))]:
        print(
            f"{row.get('rank', '')!s:>4} {row.get('config_id', ''):<9} "
            f"{float(row.get('score') or 0.0):>7.2f} "
            f"{float(row.get('fall_rate') or 0.0):>8.1f} "
            f"{float(row.get('mean_forward_vel') or 0.0):>8.3f} "
            f"{float(row.get('mean_final_base_x') or 0.0):>7.3f} "
            f"{float(row.get('mean_final_upright') or 0.0):>7.3f} "
            f"{row.get('promote_pass', '')} "
            f"{row.get('passed', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
