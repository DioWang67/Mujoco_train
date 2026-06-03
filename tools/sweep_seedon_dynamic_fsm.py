"""Exhaustive dynamic-step FSM sweep with live progress logging.

This tool searches a small deterministic FSM gait grid aimed at short
single-support "dynamic stepping" without PPO training. It prints live
progress, flushes output every update, and checkpoints CSV rows during the
sweep so local runs no longer look stuck.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from seedon_baseline.env import JOINT_NAMES, SeedonStandingEnv
from tools.seedon_debug_common import (
    DEBUG_OUT_DIR,
    RELAXED_FOOT_SIZE,
    apply_foot_size_override,
    contact_pairs,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "dynamic_fsm_sweep.csv"


@dataclass(frozen=True)
class DynamicFsmCandidate:
    """One symmetric dynamic stepping candidate."""

    lift_steps: int
    lower_steps: int
    double_support_steps: int
    swing_cap: float
    right_swing_scale: float
    left_swing_scale: float


@dataclass(frozen=True)
class DynamicRolloutMetrics:
    """Metrics collected from one zero-action rollout."""

    steps_requested: int
    steps_completed: int
    terminated: bool
    truncated: bool
    min_z: float
    min_upright: float
    max_abs_fwd: float
    right_only: int
    left_only: int
    both: int
    none: int
    single_support: int
    base_floor_steps: int


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=18)
    parser.add_argument("--checkpoint-every", type=int, default=36)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def _iter_candidates() -> list[DynamicFsmCandidate]:
    """Return the full user-requested dynamic-step search grid."""
    candidates: list[DynamicFsmCandidate] = []
    for values in itertools.product(
        [60, 80, 100],
        [40, 60, 80],
        [100, 120, 160],
        [0.70, 0.75, 0.80],
        [0.6, 0.7, 0.8],
        [0.6, 0.7, 0.8],
    ):
        candidates.append(
            DynamicFsmCandidate(
                lift_steps=values[0],
                lower_steps=values[1],
                double_support_steps=values[2],
                swing_cap=values[3],
                right_swing_scale=values[4],
                left_swing_scale=values[5],
            )
        )
    return candidates


def _make_fsm(candidate: DynamicFsmCandidate):
    """Return an in-memory replacement for ``_fsm_gait_joint_offsets``."""

    def _fsm(self: SeedonStandingEnv) -> np.ndarray:
        total_steps = 2 * (
            candidate.lift_steps
            + candidate.lower_steps
            + candidate.double_support_steps
        )
        phase_step = self._gait_step % total_steps
        offsets = np.zeros(len(JOINT_NAMES), dtype=np.float64)

        support_roll = self._reward_config.gait_hip_roll_amp
        hip_pitch = self._reward_config.gait_hip_pitch_amp
        knee_pitch = self._reward_config.gait_knee_pitch_amp
        ankle_pitch = self._reward_config.gait_ankle_pitch_amp
        right_support_roll_scale = 1.2
        left_support_roll_scale = 0.7

        def capped(value: float) -> float:
            return min(value, candidate.swing_cap)

        def apply_right_swing(swing: float) -> None:
            scaled_swing = candidate.right_swing_scale * swing
            offsets[6] += support_roll * right_support_roll_scale * swing
            offsets[2] += hip_pitch * scaled_swing
            offsets[3] += knee_pitch * scaled_swing
            offsets[4] += ankle_pitch * scaled_swing

        def apply_left_swing(swing: float) -> None:
            scaled_swing = candidate.left_swing_scale * swing
            offsets[1] += -(support_roll * left_support_roll_scale) * swing
            offsets[7] += hip_pitch * scaled_swing
            offsets[8] += knee_pitch * scaled_swing
            offsets[9] += ankle_pitch * scaled_swing

        if phase_step < candidate.lift_steps:
            s = capped(self._smoothstep((phase_step + 1) / candidate.lift_steps))
            apply_right_swing(s)
            return offsets
        phase_step -= candidate.lift_steps

        if phase_step < candidate.lower_steps:
            s = capped(1.0 - self._smoothstep((phase_step + 1) / candidate.lower_steps))
            apply_right_swing(s)
            return offsets
        phase_step -= candidate.lower_steps

        if phase_step < candidate.double_support_steps:
            return offsets
        phase_step -= candidate.double_support_steps

        if phase_step < candidate.lift_steps:
            s = capped(self._smoothstep((phase_step + 1) / candidate.lift_steps))
            apply_left_swing(s)
            return offsets
        phase_step -= candidate.lift_steps

        if phase_step < candidate.lower_steps:
            s = capped(1.0 - self._smoothstep((phase_step + 1) / candidate.lower_steps))
            apply_left_swing(s)
            return offsets

        return offsets

    return _fsm


def _rollout_candidate(
    candidate: DynamicFsmCandidate,
    *,
    steps: int,
    seed: int,
) -> DynamicRolloutMetrics:
    """Run one relaxed-foot zero-action rollout for a candidate."""
    original = SeedonStandingEnv._fsm_gait_joint_offsets
    SeedonStandingEnv._fsm_gait_joint_offsets = _make_fsm(candidate)
    env = SeedonStandingEnv(reset_noise_scale=0.0)
    try:
        env.reset(seed=seed)
        apply_foot_size_override(env.model, env.data, RELAXED_FOOT_SIZE)
        action = np.zeros(env.action_space.shape, dtype=np.float64)
        contact_state_counter: Counter[str] = Counter()
        min_z = float("inf")
        min_upright = float("inf")
        max_abs_fwd = 0.0
        base_floor_steps = 0
        terminated = False
        truncated = False
        steps_completed = 0

        for step in range(1, steps + 1):
            _, _, terminated, truncated, info = env.step(action)
            pairs = contact_pairs(env.model, env.data)
            pair_sets = [set((name_a, name_b)) for name_a, name_b, _ in pairs]
            floor_r = {"floor", "R_foot_collision"} in pair_sets
            floor_l = {"floor", "L_foot_collision"} in pair_sets
            floor_base = {"floor", "base_proxy"} in pair_sets
            if floor_base:
                base_floor_steps += 1

            if floor_r and floor_l:
                contact_state = "both"
            elif floor_r:
                contact_state = "right_only"
            elif floor_l:
                contact_state = "left_only"
            else:
                contact_state = "none"
            contact_state_counter.update([contact_state])

            min_z = min(min_z, float(info["base_height"]))
            min_upright = min(min_upright, float(info["upright"]))
            max_abs_fwd = max(max_abs_fwd, abs(float(info["forward_velocity"])))
            steps_completed = step
            if terminated or truncated:
                break

        return DynamicRolloutMetrics(
            steps_requested=steps,
            steps_completed=steps_completed,
            terminated=terminated,
            truncated=truncated,
            min_z=min_z,
            min_upright=min_upright,
            max_abs_fwd=max_abs_fwd,
            right_only=contact_state_counter["right_only"],
            left_only=contact_state_counter["left_only"],
            both=contact_state_counter["both"],
            none=contact_state_counter["none"],
            single_support=(
                contact_state_counter["right_only"] + contact_state_counter["left_only"]
            ),
            base_floor_steps=base_floor_steps,
        )
    finally:
        env.close()
        SeedonStandingEnv._fsm_gait_joint_offsets = original


def _passes_acceptance(metrics: DynamicRolloutMetrics) -> bool:
    """Return whether rollout metrics satisfy the user-specified target."""
    return (
        not metrics.terminated
        and metrics.min_z > 0.40
        and metrics.min_upright > 0.95
        and metrics.max_abs_fwd < 0.35
        and metrics.single_support > 0
        and metrics.none <= 5
        and metrics.base_floor_steps == 0
    )


def _best_rank_key(metrics: DynamicRolloutMetrics) -> tuple[float, ...]:
    """Return the ranking key used for the current best overall candidate."""
    return (
        float(_passes_acceptance(metrics)),
        float(not metrics.terminated),
        -metrics.max_abs_fwd,
        metrics.min_upright,
        metrics.min_z,
        float(metrics.single_support),
        -float(metrics.none),
    )


def _support_rank_key(metrics: DynamicRolloutMetrics) -> tuple[float, ...]:
    """Return the ranking key used for the best single-support candidate."""
    return (
        float(metrics.single_support > 0),
        float(not metrics.terminated),
        float(metrics.single_support),
        -metrics.max_abs_fwd,
        metrics.min_upright,
        metrics.min_z,
        -float(metrics.none),
    )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write or refresh the CSV checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _log_progress(
    *,
    index: int,
    total: int,
    started_at: float,
    best_candidate: DynamicFsmCandidate | None,
    best_metrics: DynamicRolloutMetrics | None,
    best_support_candidate: DynamicFsmCandidate | None,
    best_support_metrics: DynamicRolloutMetrics | None,
) -> None:
    """Print one flushed progress snapshot."""
    elapsed = max(1e-9, time.time() - started_at)
    rate = index / elapsed
    eta_seconds = int((total - index) / rate) if rate > 0.0 else 0
    minutes, seconds = divmod(eta_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    print(
        f"{index}/{total} rate={rate:.2f}/s eta={hours:02d}:{minutes:02d}:{seconds:02d}",
        flush=True,
    )
    if best_candidate is not None and best_metrics is not None:
        print(
            "  best_overall "
            f"pass={_passes_acceptance(best_metrics)} "
            f"term={best_metrics.terminated} "
            f"min_z={best_metrics.min_z:.4f} "
            f"min_u={best_metrics.min_upright:.4f} "
            f"maxf={best_metrics.max_abs_fwd:.4f} "
            f"support={best_metrics.right_only}/{best_metrics.left_only} "
            f"none={best_metrics.none} "
            f"params={json.dumps(asdict(best_candidate), sort_keys=True)}",
            flush=True,
        )
    if best_support_candidate is not None and best_support_metrics is not None:
        print(
            "  best_single_support "
            f"term={best_support_metrics.terminated} "
            f"min_z={best_support_metrics.min_z:.4f} "
            f"min_u={best_support_metrics.min_upright:.4f} "
            f"maxf={best_support_metrics.max_abs_fwd:.4f} "
            f"support={best_support_metrics.right_only}/{best_support_metrics.left_only} "
            f"none={best_support_metrics.none} "
            f"params={json.dumps(asdict(best_support_candidate), sort_keys=True)}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    """Run the exhaustive dynamic-step sweep with live progress output."""
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive.")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive.")

    candidates = _iter_candidates()
    total = len(candidates)
    rows: list[dict[str, object]] = []
    started_at = time.time()
    best_candidate: DynamicFsmCandidate | None = None
    best_metrics: DynamicRolloutMetrics | None = None
    best_support_candidate: DynamicFsmCandidate | None = None
    best_support_metrics: DynamicRolloutMetrics | None = None
    pass_count = 0

    print(f"search_space: {total} candidates", flush=True)
    print(f"csv: {args.out_csv}", flush=True)

    for index, candidate in enumerate(candidates, start=1):
        metrics = _rollout_candidate(candidate, steps=args.steps, seed=args.seed)
        row = {
            **asdict(candidate),
            **asdict(metrics),
            "passed": _passes_acceptance(metrics),
        }
        rows.append(row)

        if best_metrics is None or _best_rank_key(metrics) > _best_rank_key(best_metrics):
            best_candidate = candidate
            best_metrics = metrics
        if (
            best_support_metrics is None
            or _support_rank_key(metrics) > _support_rank_key(best_support_metrics)
        ):
            best_support_candidate = candidate
            best_support_metrics = metrics
        if row["passed"]:
            pass_count += 1
            print(
                "  pass_found "
                f"index={index} params={json.dumps(asdict(candidate), sort_keys=True)} "
                f"metrics={json.dumps(asdict(metrics), sort_keys=True)}",
                flush=True,
            )

        if index % args.print_every == 0 or row["passed"] or index == total:
            _log_progress(
                index=index,
                total=total,
                started_at=started_at,
                best_candidate=best_candidate,
                best_metrics=best_metrics,
                best_support_candidate=best_support_candidate,
                best_support_metrics=best_support_metrics,
            )
        if index % args.checkpoint_every == 0 or index == total:
            _write_rows(args.out_csv, rows)

    if best_candidate is None or best_metrics is None:
        raise RuntimeError("Sweep produced no candidates.")

    print("\npass_count:", pass_count, flush=True)
    print("best_params:", flush=True)
    print(json.dumps(asdict(best_candidate), indent=2, sort_keys=True), flush=True)
    print("\nbest_metrics:", flush=True)
    print(json.dumps(asdict(best_metrics), indent=2, sort_keys=True), flush=True)
    if best_support_candidate is not None and best_support_metrics is not None:
        print("\nbest_single_support_params:", flush=True)
        print(json.dumps(asdict(best_support_candidate), indent=2, sort_keys=True), flush=True)
        print("\nbest_single_support_metrics:", flush=True)
        print(json.dumps(asdict(best_support_metrics), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
