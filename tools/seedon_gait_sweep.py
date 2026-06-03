"""Sweep deterministic Seedon gait-prior parameters without PPO training."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from seedon_baseline.env import SeedonStandingConfig, SeedonStandingEnv

DEFAULT_OUT_CSV = Path("reports") / "seedon_gait_sweep.csv"


@dataclass(frozen=True)
class GaitCandidate:
    """One deterministic gait-prior parameter set."""

    period: int
    hip_roll_amp: float
    right_hip_roll_sign: float
    left_hip_roll_sign: float
    hip_pitch_amp: float
    knee_pitch_amp: float
    ankle_pitch_amp: float


@dataclass(frozen=True)
class GaitSweepResult:
    """Metrics collected from one deterministic gait rollout."""

    rank: int
    survived: bool
    steps: int
    total_reward: float
    final_base_x: float
    mean_forward_velocity: float
    max_forward_velocity: float
    final_base_height: float
    final_upright: float
    min_base_height: float
    min_upright: float
    period: int
    hip_roll_amp: float
    right_hip_roll_sign: float
    left_hip_roll_sign: float
    hip_pitch_amp: float
    knee_pitch_amp: float
    ankle_pitch_amp: float


def _parse_float_list(raw_value: str, *, option_name: str) -> list[float]:
    """Parse a comma-separated float list."""
    try:
        values = [float(item.strip()) for item in raw_value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{option_name} must be a comma-separated float list."
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{option_name} must not be empty.")
    return values


def _parse_int_list(raw_value: str, *, option_name: str) -> list[int]:
    """Parse a comma-separated positive integer list."""
    try:
        values = [int(item.strip()) for item in raw_value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{option_name} must be a comma-separated integer list."
        ) from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{option_name} must contain positive integers.")
    return values


def _build_candidates(
    periods: list[int],
    hip_roll_amps: list[float],
    right_hip_roll_sign: float,
    left_hip_roll_sign: float,
    hip_amps: list[float],
    knee_amps: list[float],
    ankle_ratios: list[float],
) -> list[GaitCandidate]:
    """Return the full parameter grid for deterministic gait simulation."""
    candidates: list[GaitCandidate] = []
    for period in periods:
        for hip_roll_amp in hip_roll_amps:
            for hip_amp in hip_amps:
                for knee_amp in knee_amps:
                    for ankle_ratio in ankle_ratios:
                        candidates.append(
                            GaitCandidate(
                                period=period,
                                hip_roll_amp=hip_roll_amp,
                                right_hip_roll_sign=right_hip_roll_sign,
                                left_hip_roll_sign=left_hip_roll_sign,
                                hip_pitch_amp=hip_amp,
                                knee_pitch_amp=knee_amp,
                                ankle_pitch_amp=knee_amp * ankle_ratio,
                            )
                        )
    return candidates


def _rollout_candidate(
    candidate: GaitCandidate,
    *,
    base_config: SeedonStandingConfig,
    seeds: list[int],
    max_steps: int,
    reset_noise_scale: float,
) -> GaitSweepResult:
    """Run zero-action rollouts for a gait candidate and aggregate metrics."""
    results = [
        _rollout_candidate_once(
            candidate,
            base_config=base_config,
            seed=seed,
            max_steps=max_steps,
            reset_noise_scale=reset_noise_scale,
        )
        for seed in seeds
    ]
    return GaitSweepResult(
        rank=0,
        survived=all(result.survived for result in results),
        steps=min(result.steps for result in results),
        total_reward=float(np.mean([result.total_reward for result in results])),
        final_base_x=float(np.mean([result.final_base_x for result in results])),
        mean_forward_velocity=float(
            np.mean([result.mean_forward_velocity for result in results])
        ),
        max_forward_velocity=float(np.max([result.max_forward_velocity for result in results])),
        final_base_height=float(np.mean([result.final_base_height for result in results])),
        final_upright=float(np.mean([result.final_upright for result in results])),
        min_base_height=float(np.min([result.min_base_height for result in results])),
        min_upright=float(np.min([result.min_upright for result in results])),
        period=candidate.period,
        hip_roll_amp=candidate.hip_roll_amp,
        right_hip_roll_sign=candidate.right_hip_roll_sign,
        left_hip_roll_sign=candidate.left_hip_roll_sign,
        hip_pitch_amp=candidate.hip_pitch_amp,
        knee_pitch_amp=candidate.knee_pitch_amp,
        ankle_pitch_amp=candidate.ankle_pitch_amp,
    )


def _rollout_candidate_once(
    candidate: GaitCandidate,
    *,
    base_config: SeedonStandingConfig,
    seed: int,
    max_steps: int,
    reset_noise_scale: float,
) -> GaitSweepResult:
    """Run one zero-action rollout for a gait candidate."""
    reward_config = replace(
        base_config,
        gait_cycle_steps=candidate.period,
        gait_hip_roll_amp=candidate.hip_roll_amp,
        gait_right_hip_roll_sign=candidate.right_hip_roll_sign,
        gait_left_hip_roll_sign=candidate.left_hip_roll_sign,
        gait_hip_pitch_amp=candidate.hip_pitch_amp,
        gait_knee_pitch_amp=candidate.knee_pitch_amp,
        gait_ankle_pitch_amp=candidate.ankle_pitch_amp,
    )
    env = SeedonStandingEnv(
        reset_noise_scale=reset_noise_scale,
        reward_config=reward_config,
    )
    try:
        env.reset(seed=seed)
        action = np.zeros(env.action_space.shape, dtype=np.float64)
        total_reward = 0.0
        forward_velocity_sum = 0.0
        max_forward_velocity = -np.inf
        min_base_height = np.inf
        min_upright = np.inf
        last_info: dict = {}
        steps = 0
        for step in range(max_steps):
            _, reward, terminated, truncated, last_info = env.step(action)
            total_reward += float(reward)
            forward_velocity = float(last_info.get("forward_velocity", 0.0))
            base_height = float(last_info.get("base_height", np.nan))
            upright = float(last_info.get("upright", np.nan))
            forward_velocity_sum += forward_velocity
            max_forward_velocity = max(max_forward_velocity, forward_velocity)
            min_base_height = min(min_base_height, base_height)
            min_upright = min(min_upright, upright)
            steps = step + 1
            if terminated or truncated:
                break
        survived = steps >= max_steps
        return GaitSweepResult(
            rank=0,
            survived=survived,
            steps=steps,
            total_reward=total_reward,
            final_base_x=float(last_info.get("base_x_position", np.nan)),
            mean_forward_velocity=forward_velocity_sum / max(1, steps),
            max_forward_velocity=float(max_forward_velocity),
            final_base_height=float(last_info.get("base_height", np.nan)),
            final_upright=float(last_info.get("upright", np.nan)),
            min_base_height=float(min_base_height),
            min_upright=float(min_upright),
            period=candidate.period,
            hip_roll_amp=candidate.hip_roll_amp,
            right_hip_roll_sign=candidate.right_hip_roll_sign,
            left_hip_roll_sign=candidate.left_hip_roll_sign,
            hip_pitch_amp=candidate.hip_pitch_amp,
            knee_pitch_amp=candidate.knee_pitch_amp,
            ankle_pitch_amp=candidate.ankle_pitch_amp,
        )
    finally:
        env.close()


def _rank_results(results: list[GaitSweepResult]) -> list[GaitSweepResult]:
    """Sort results by stable forward progress and assign ranks."""
    def sort_key(result: GaitSweepResult) -> tuple:
        smooth_forward = (
            result.survived
            and 0.005 <= result.mean_forward_velocity <= 0.08
            and result.max_forward_velocity <= 0.35
            and result.final_base_x > 0.04
            and result.min_base_height >= 0.34
            and result.min_upright >= 0.75
            and result.final_base_height >= 0.38
            and result.final_upright >= 0.85
        )
        overspeed = max(0.0, result.max_forward_velocity - 0.35)
        return (
            smooth_forward,
            result.survived,
            -overspeed,
            result.final_upright,
            result.final_base_height,
            result.mean_forward_velocity,
            result.final_base_x,
            result.steps,
            result.total_reward,
        )

    ranked = sorted(
        results,
        key=sort_key,
        reverse=True,
    )
    return [replace(result, rank=index + 1) for index, result in enumerate(ranked)]


def _write_csv(path: Path, results: list[GaitSweepResult]) -> None:
    """Write sweep results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GaitSweepResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def build_parser() -> argparse.ArgumentParser:
    """Build the gait sweep CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400, help="Max rollout steps.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--seed-count",
        type=int,
        default=1,
        help="Number of consecutive seeds to evaluate for every candidate.",
    )
    parser.add_argument("--top", type=int, default=20, help="Number of rows to print.")
    parser.add_argument(
        "--reset-noise-scale",
        type=float,
        default=0.0,
        help="Uniform reset noise applied to actuated joints.",
    )
    parser.add_argument(
        "--periods",
        type=lambda value: _parse_int_list(value, option_name="--periods"),
        default="40,60,80",
        help="Comma-separated gait cycle lengths in RL steps.",
    )
    parser.add_argument(
        "--hip-roll-amps",
        type=lambda value: _parse_float_list(value, option_name="--hip-roll-amps"),
        default="0.0",
        help="Comma-separated hip roll amplitudes for lateral weight shift.",
    )
    parser.add_argument(
        "--right-hip-roll-sign",
        type=float,
        default=1.0,
        help="Direction multiplier for right hip roll gait offsets.",
    )
    parser.add_argument(
        "--left-hip-roll-sign",
        type=float,
        default=-1.0,
        help="Direction multiplier for left hip roll gait offsets.",
    )
    parser.add_argument(
        "--hip-amps",
        type=lambda value: _parse_float_list(value, option_name="--hip-amps"),
        default="-0.03,-0.04,-0.05,-0.06,-0.07",
        help="Comma-separated hip pitch amplitudes. Use --hip-amps=-0.03,-0.06.",
    )
    parser.add_argument(
        "--knee-amps",
        type=lambda value: _parse_float_list(value, option_name="--knee-amps"),
        default="0.045,0.06,0.075,0.09,0.105",
        help="Comma-separated knee pitch amplitudes.",
    )
    parser.add_argument(
        "--ankle-ratios",
        type=lambda value: _parse_float_list(value, option_name="--ankle-ratios"),
        default="0.4,0.5,0.6",
        help="Comma-separated ankle/knee amplitude ratios.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help="CSV path for full sweep results.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run deterministic Seedon gait-prior parameter sweep."""
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.top <= 0:
        raise ValueError("--top must be positive.")
    if args.seed_count <= 0:
        raise ValueError("--seed-count must be positive.")
    if args.reset_noise_scale < 0.0:
        raise ValueError("--reset-noise-scale must be non-negative.")

    candidates = _build_candidates(
        periods=args.periods,
        hip_roll_amps=args.hip_roll_amps,
        right_hip_roll_sign=args.right_hip_roll_sign,
        left_hip_roll_sign=args.left_hip_roll_sign,
        hip_amps=args.hip_amps,
        knee_amps=args.knee_amps,
        ankle_ratios=args.ankle_ratios,
    )
    base_config = SeedonStandingConfig()
    seeds = [args.seed + offset for offset in range(args.seed_count)]
    results = [
        _rollout_candidate(
            candidate,
            base_config=base_config,
            seeds=seeds,
            max_steps=args.steps,
            reset_noise_scale=args.reset_noise_scale,
        )
        for candidate in candidates
    ]
    ranked = _rank_results(results)
    _write_csv(args.out_csv, ranked)

    print(f"candidates: {len(candidates)}")
    print(f"csv       : {args.out_csv}")
    print(
        "rank survived steps mean_fwd max_fwd final_x base_z upright total_reward "
        "period roll hip knee ankle"
    )
    for result in ranked[: args.top]:
        print(
            f"{result.rank:>4} {str(result.survived):>8} {result.steps:>5} "
            f"{result.mean_forward_velocity:>8.4f} "
            f"{result.max_forward_velocity:>7.3f} {result.final_base_x:>7.4f} "
            f"{result.final_base_height:>6.3f} {result.final_upright:>7.3f} "
            f"{result.total_reward:>12.1f} {result.period:>6} "
            f"{result.hip_roll_amp:>6.3f} "
            f"{result.hip_pitch_amp:>7.3f} {result.knee_pitch_amp:>6.3f} "
            f"{result.ankle_pitch_amp:>7.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
