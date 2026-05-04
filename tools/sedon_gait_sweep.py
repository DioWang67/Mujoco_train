"""Sweep deterministic Sedon gait-prior parameters without PPO training."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from sedon_baseline.env import SedonStandingConfig, SedonStandingEnv

DEFAULT_OUT_CSV = Path("reports") / "sedon_gait_sweep.csv"


@dataclass(frozen=True)
class GaitCandidate:
    """One deterministic gait-prior parameter set."""

    period: int
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
    final_base_height: float
    final_upright: float
    period: int
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
    hip_amps: list[float],
    knee_amps: list[float],
    ankle_ratios: list[float],
) -> list[GaitCandidate]:
    """Return the full parameter grid for deterministic gait simulation."""
    candidates: list[GaitCandidate] = []
    for period in periods:
        for hip_amp in hip_amps:
            for knee_amp in knee_amps:
                for ankle_ratio in ankle_ratios:
                    candidates.append(
                        GaitCandidate(
                            period=period,
                            hip_pitch_amp=hip_amp,
                            knee_pitch_amp=knee_amp,
                            ankle_pitch_amp=knee_amp * ankle_ratio,
                        )
                    )
    return candidates


def _rollout_candidate(
    candidate: GaitCandidate,
    *,
    base_config: SedonStandingConfig,
    seed: int,
    max_steps: int,
) -> GaitSweepResult:
    """Run one deterministic zero-action rollout for a gait candidate."""
    reward_config = replace(
        base_config,
        gait_cycle_steps=candidate.period,
        gait_hip_pitch_amp=candidate.hip_pitch_amp,
        gait_knee_pitch_amp=candidate.knee_pitch_amp,
        gait_ankle_pitch_amp=candidate.ankle_pitch_amp,
    )
    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=reward_config)
    try:
        env.reset(seed=seed)
        action = np.zeros(env.action_space.shape, dtype=np.float64)
        total_reward = 0.0
        forward_velocity_sum = 0.0
        last_info: dict = {}
        steps = 0
        for step in range(max_steps):
            _, reward, terminated, truncated, last_info = env.step(action)
            total_reward += float(reward)
            forward_velocity_sum += float(last_info.get("forward_velocity", 0.0))
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
            final_base_height=float(last_info.get("base_height", np.nan)),
            final_upright=float(last_info.get("upright", np.nan)),
            period=candidate.period,
            hip_pitch_amp=candidate.hip_pitch_amp,
            knee_pitch_amp=candidate.knee_pitch_amp,
            ankle_pitch_amp=candidate.ankle_pitch_amp,
        )
    finally:
        env.close()


def _rank_results(results: list[GaitSweepResult]) -> list[GaitSweepResult]:
    """Sort results by stable forward progress and assign ranks."""
    ranked = sorted(
        results,
        key=lambda result: (
            result.survived,
            result.mean_forward_velocity,
            result.final_base_x,
            result.steps,
            result.total_reward,
        ),
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
    parser.add_argument("--top", type=int, default=20, help="Number of rows to print.")
    parser.add_argument(
        "--periods",
        type=lambda value: _parse_int_list(value, option_name="--periods"),
        default="40,60,80",
        help="Comma-separated gait cycle lengths in RL steps.",
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
    """Run deterministic Sedon gait-prior parameter sweep."""
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.top <= 0:
        raise ValueError("--top must be positive.")

    candidates = _build_candidates(
        periods=args.periods,
        hip_amps=args.hip_amps,
        knee_amps=args.knee_amps,
        ankle_ratios=args.ankle_ratios,
    )
    base_config = SedonStandingConfig()
    results = [
        _rollout_candidate(
            candidate,
            base_config=base_config,
            seed=args.seed + index,
            max_steps=args.steps,
        )
        for index, candidate in enumerate(candidates)
    ]
    ranked = _rank_results(results)
    _write_csv(args.out_csv, ranked)

    print(f"candidates: {len(candidates)}")
    print(f"csv       : {args.out_csv}")
    print(
        "rank survived steps mean_fwd final_x total_reward "
        "period hip knee ankle"
    )
    for result in ranked[: args.top]:
        print(
            f"{result.rank:>4} {str(result.survived):>8} {result.steps:>5} "
            f"{result.mean_forward_velocity:>8.4f} {result.final_base_x:>7.4f} "
            f"{result.total_reward:>12.1f} {result.period:>6} "
            f"{result.hip_pitch_amp:>7.3f} {result.knee_pitch_amp:>6.3f} "
            f"{result.ankle_pitch_amp:>7.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
