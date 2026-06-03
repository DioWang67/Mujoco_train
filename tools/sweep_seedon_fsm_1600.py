"""Sweep a relaxed-foot 1600-step robust Seedon FSM gait without PPO training.

This tool searches the full deterministic FSM gait parameter space in-memory,
prints the best overall candidate, the best candidate that produces bilateral
single-support, the best passing candidate when available, renders a drop-in
``_fsm_gait_joint_offsets()`` implementation, and runs a final
800/1600/2400 robustness check.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import textwrap
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from seedon_baseline.env import JOINT_NAMES, SeedonStandingEnv
from tools.seedon_debug_common import (
    DEBUG_OUT_DIR,
    RELAXED_FOOT_SIZE,
    apply_foot_size_override,
    contact_pairs,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "fsm_1600_sweep.csv"


@dataclass(frozen=True)
class FsmCandidate:
    """One full-FSM gait parameter set."""

    right_swing_scale: float
    left_swing_scale: float
    right_support_roll_scale: float
    left_support_roll_scale: float
    right_lift_steps: int
    left_lift_steps: int
    right_lower_steps: int
    left_lower_steps: int
    double_support_steps: int
    swing_cap: float


@dataclass(frozen=True)
class RolloutMetrics:
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
    parser.add_argument("--samples", type=int, default=360)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-steps", type=int, default=1600)
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument(
        "--local-neighbors",
        type=int,
        default=120,
        help="Extra local-search samples generated around the best bilateral-support candidate.",
    )
    parser.add_argument(
        "--robustness-steps",
        type=str,
        default="800,1600,2400",
        help="Comma-separated rollout lengths for the final robustness check.",
    )
    return parser


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


def _candidate_from_dict(raw: dict[str, float | int]) -> FsmCandidate:
    """Return a typed candidate from a plain mapping."""
    return FsmCandidate(
        right_swing_scale=float(raw["right_swing_scale"]),
        left_swing_scale=float(raw["left_swing_scale"]),
        right_support_roll_scale=float(raw["right_support_roll_scale"]),
        left_support_roll_scale=float(raw["left_support_roll_scale"]),
        right_lift_steps=int(raw["right_lift_steps"]),
        left_lift_steps=int(raw["left_lift_steps"]),
        right_lower_steps=int(raw["right_lower_steps"]),
        left_lower_steps=int(raw["left_lower_steps"]),
        double_support_steps=int(raw["double_support_steps"]),
        swing_cap=float(raw["swing_cap"]),
    )


def _make_fsm(candidate: FsmCandidate):
    """Return an in-memory replacement for ``_fsm_gait_joint_offsets``."""

    def _fsm(self: SeedonStandingEnv) -> np.ndarray:
        total_steps = (
            candidate.right_lift_steps
            + candidate.right_lower_steps
            + candidate.double_support_steps
            + candidate.left_lift_steps
            + candidate.left_lower_steps
            + candidate.double_support_steps
        )
        phase_step = self._gait_step % total_steps
        offsets = np.zeros(len(JOINT_NAMES), dtype=np.float64)

        support_roll = self._reward_config.gait_hip_roll_amp
        hip_pitch = self._reward_config.gait_hip_pitch_amp
        knee_pitch = self._reward_config.gait_knee_pitch_amp
        ankle_pitch = self._reward_config.gait_ankle_pitch_amp

        def capped(value: float) -> float:
            return min(value, candidate.swing_cap)

        def apply_right_swing(swing: float) -> None:
            scaled_swing = candidate.right_swing_scale * swing
            offsets[6] += support_roll * candidate.right_support_roll_scale * swing
            offsets[2] += hip_pitch * scaled_swing
            offsets[3] += knee_pitch * scaled_swing
            offsets[4] += ankle_pitch * scaled_swing

        def apply_left_swing(swing: float) -> None:
            scaled_swing = candidate.left_swing_scale * swing
            offsets[1] += -(support_roll * candidate.left_support_roll_scale) * swing
            offsets[7] += hip_pitch * scaled_swing
            offsets[8] += knee_pitch * scaled_swing
            offsets[9] += ankle_pitch * scaled_swing

        if phase_step < candidate.right_lift_steps:
            s = capped(self._smoothstep((phase_step + 1) / candidate.right_lift_steps))
            apply_right_swing(s)
            return offsets
        phase_step -= candidate.right_lift_steps

        if phase_step < candidate.right_lower_steps:
            s = capped(1.0 - self._smoothstep((phase_step + 1) / candidate.right_lower_steps))
            apply_right_swing(s)
            return offsets
        phase_step -= candidate.right_lower_steps

        if phase_step < candidate.double_support_steps:
            return offsets
        phase_step -= candidate.double_support_steps

        if phase_step < candidate.left_lift_steps:
            s = capped(self._smoothstep((phase_step + 1) / candidate.left_lift_steps))
            apply_left_swing(s)
            return offsets
        phase_step -= candidate.left_lift_steps

        if phase_step < candidate.left_lower_steps:
            s = capped(1.0 - self._smoothstep((phase_step + 1) / candidate.left_lower_steps))
            apply_left_swing(s)
            return offsets

        return offsets

    return _fsm


def _rollout_candidate(
    candidate: FsmCandidate,
    *,
    steps: int,
    seed: int,
    relaxed_foot: bool,
) -> RolloutMetrics:
    """Run one zero-action rollout for a candidate."""
    original = SeedonStandingEnv._fsm_gait_joint_offsets
    SeedonStandingEnv._fsm_gait_joint_offsets = _make_fsm(candidate)
    env = SeedonStandingEnv(reset_noise_scale=0.0)
    try:
        env.reset(seed=seed)
        if relaxed_foot:
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

        return RolloutMetrics(
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


def _passes_acceptance(metrics: RolloutMetrics) -> bool:
    """Return whether rollout metrics satisfy the 1600-step relaxed-foot target."""
    return (
        not metrics.terminated
        and metrics.min_z > 0.40
        and metrics.min_upright > 0.95
        and metrics.max_abs_fwd < 0.30
        and metrics.right_only > 0
        and metrics.left_only > 0
        and metrics.base_floor_steps == 0
    )


def _has_bilateral_support(metrics: RolloutMetrics) -> bool:
    """Return whether both single-support directions were observed."""
    return metrics.right_only > 0 and metrics.left_only > 0


def _rank_key(metrics: RolloutMetrics) -> tuple[float, ...]:
    """Return the user-requested ranking key."""
    return (
        float(not metrics.terminated),
        -metrics.max_abs_fwd,
        metrics.min_upright,
        metrics.min_z,
        float(metrics.single_support),
    )


def _support_rank_key(metrics: RolloutMetrics) -> tuple[float, ...]:
    """Return a ranking key that favors bilateral-support candidates."""
    return (
        float(_has_bilateral_support(metrics)),
        float(not metrics.terminated),
        float(metrics.steps_completed),
        -metrics.max_abs_fwd,
        metrics.min_upright,
        metrics.min_z,
        float(metrics.single_support),
    )


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write full sweep rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_function(candidate: FsmCandidate) -> str:
    """Render a drop-in ``_fsm_gait_joint_offsets`` implementation."""
    function_body = f"""
    def _fsm_gait_joint_offsets(self) -> np.ndarray:
        \"\"\"Return FSM gait with explicit lift/lower phases.\"\"\"
        right_lift_steps = {candidate.right_lift_steps}
        right_lower_steps = {candidate.right_lower_steps}
        left_lift_steps = {candidate.left_lift_steps}
        left_lower_steps = {candidate.left_lower_steps}
        double_support_steps = {candidate.double_support_steps}

        right_swing_scale = {candidate.right_swing_scale}
        left_swing_scale = {candidate.left_swing_scale}
        right_support_roll_scale = {candidate.right_support_roll_scale}
        left_support_roll_scale = {candidate.left_support_roll_scale}
        swing_cap = {candidate.swing_cap}

        total_steps = (
            right_lift_steps
            + right_lower_steps
            + double_support_steps
            + left_lift_steps
            + left_lower_steps
            + double_support_steps
        )

        phase_step = self._gait_step % total_steps
        offsets = np.zeros(len(JOINT_NAMES), dtype=np.float64)

        support_roll = self._reward_config.gait_hip_roll_amp
        hip_pitch = self._reward_config.gait_hip_pitch_amp
        knee_pitch = self._reward_config.gait_knee_pitch_amp
        ankle_pitch = self._reward_config.gait_ankle_pitch_amp

        def capped(value: float) -> float:
            return min(value, swing_cap)

        def apply_right_swing(swing: float) -> None:
            scaled_swing = right_swing_scale * swing
            offsets[6] += support_roll * right_support_roll_scale * swing
            offsets[2] += hip_pitch * scaled_swing
            offsets[3] += knee_pitch * scaled_swing
            offsets[4] += ankle_pitch * scaled_swing

        def apply_left_swing(swing: float) -> None:
            scaled_swing = left_swing_scale * swing
            offsets[1] += -(support_roll * left_support_roll_scale) * swing
            offsets[7] += hip_pitch * scaled_swing
            offsets[8] += knee_pitch * scaled_swing
            offsets[9] += ankle_pitch * scaled_swing

        if phase_step < right_lift_steps:
            s = capped(self._smoothstep((phase_step + 1) / right_lift_steps))
            apply_right_swing(s)
            return offsets
        phase_step -= right_lift_steps

        if phase_step < right_lower_steps:
            s = capped(1.0 - self._smoothstep((phase_step + 1) / right_lower_steps))
            apply_right_swing(s)
            return offsets
        phase_step -= right_lower_steps

        if phase_step < double_support_steps:
            return offsets
        phase_step -= double_support_steps

        if phase_step < left_lift_steps:
            s = capped(self._smoothstep((phase_step + 1) / left_lift_steps))
            apply_left_swing(s)
            return offsets
        phase_step -= left_lift_steps

        if phase_step < left_lower_steps:
            s = capped(1.0 - self._smoothstep((phase_step + 1) / left_lower_steps))
            apply_left_swing(s)
            return offsets

        return offsets
    """
    return textwrap.dedent(function_body).strip()


def _iter_samples(count: int, *, rng: random.Random) -> list[FsmCandidate]:
    """Return seeded random full-FSM samples in a narrow stability-focused band."""
    choices = {
        "right_swing_scale": [0.9, 1.0, 1.1],
        "left_swing_scale": [0.8, 0.9, 1.0],
        "right_support_roll_scale": [0.8, 1.0],
        "left_support_roll_scale": [0.6, 0.8, 1.0],
        "right_lift_steps": [220, 240, 260],
        "left_lift_steps": [200, 220, 240],
        "right_lower_steps": [180, 200, 220, 240],
        "left_lower_steps": [140, 160, 180, 200],
        "double_support_steps": [20, 40, 60, 80],
        "swing_cap": [0.56, 0.64, 0.68, 0.72],
    }
    seeded = [
        FsmCandidate(
            right_swing_scale=0.9,
            left_swing_scale=1.0,
            right_support_roll_scale=0.8,
            left_support_roll_scale=0.8,
            right_lift_steps=240,
            left_lift_steps=220,
            right_lower_steps=200,
            left_lower_steps=140,
            double_support_steps=0,
            swing_cap=0.72,
        ),
        FsmCandidate(
            right_swing_scale=1.0,
            left_swing_scale=0.9,
            right_support_roll_scale=0.8,
            left_support_roll_scale=0.8,
            right_lift_steps=240,
            left_lift_steps=220,
            right_lower_steps=220,
            left_lower_steps=160,
            double_support_steps=40,
            swing_cap=0.64,
        ),
        FsmCandidate(
            right_swing_scale=1.0,
            left_swing_scale=0.9,
            right_support_roll_scale=1.2,
            left_support_roll_scale=0.7,
            right_lift_steps=180,
            left_lift_steps=200,
            right_lower_steps=140,
            left_lower_steps=160,
            double_support_steps=0,
            swing_cap=0.76,
        ),
    ]
    samples: list[FsmCandidate] = []
    seen: set[tuple[object, ...]] = set()
    for candidate in seeded:
        key = tuple(asdict(candidate).values())
        if key not in seen:
            seen.add(key)
            samples.append(candidate)
    while len(samples) < count:
        raw_candidate = {
            name: rng.choice(values) for name, values in choices.items()
        }
        candidate = _candidate_from_dict(raw_candidate)
        key = tuple(asdict(candidate).values())
        if key in seen:
            continue
        seen.add(key)
        samples.append(candidate)
    return samples


def _neighbor_samples(
    center: FsmCandidate,
    *,
    count: int,
    rng: random.Random,
) -> list[FsmCandidate]:
    """Return local-search candidates near a bilateral-support center."""
    if count <= 0:
        return []

    float_neighbors = {
        "right_swing_scale": [0.1, 0.0, -0.1],
        "left_swing_scale": [0.1, 0.0, -0.1],
        "right_support_roll_scale": [0.2, 0.0],
        "left_support_roll_scale": [0.2, 0.0, -0.2],
        "swing_cap": [0.04, 0.0, -0.04, -0.08],
    }
    int_neighbors = {
        "right_lift_steps": [20, 0, -20],
        "left_lift_steps": [20, 0, -20],
        "right_lower_steps": [40, 20, 0, -20],
        "left_lower_steps": [40, 20, 0, -20],
        "double_support_steps": [80, 60, 40, 20, 0],
    }
    int_bounds = {
        "right_lift_steps": (180, 280),
        "left_lift_steps": (160, 260),
        "right_lower_steps": (140, 260),
        "left_lower_steps": (120, 220),
        "double_support_steps": (0, 120),
    }
    float_bounds = {
        "right_swing_scale": (0.8, 1.1),
        "left_swing_scale": (0.7, 1.0),
        "right_support_roll_scale": (0.8, 1.2),
        "left_support_roll_scale": (0.4, 1.0),
        "swing_cap": (0.56, 0.72),
    }

    samples: list[FsmCandidate] = []
    seen: set[tuple[object, ...]] = set()
    center_dict = asdict(center)
    while len(samples) < count:
        candidate = center_dict.copy()
        for name, deltas in float_neighbors.items():
            value = float(candidate[name]) + rng.choice(deltas)
            lo, hi = float_bounds[name]
            candidate[name] = round(float(np.clip(value, lo, hi)), 2)
        for name, deltas in int_neighbors.items():
            value = int(candidate[name]) + rng.choice(deltas)
            lo, hi = int_bounds[name]
            candidate[name] = int(np.clip(value, lo, hi))
        typed = _candidate_from_dict(candidate)
        key = tuple(asdict(typed).values())
        if key in seen:
            continue
        seen.add(key)
        samples.append(typed)
    return samples


def main(argv: list[str] | None = None) -> int:
    """Run the relaxed-foot 1600-step FSM sweep and print the best result."""
    args = build_parser().parse_args(argv)
    if args.samples <= 0:
        raise ValueError("--samples must be positive.")
    if args.search_steps <= 0:
        raise ValueError("--search-steps must be positive.")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive.")
    robustness_steps = _parse_int_list(
        args.robustness_steps,
        option_name="--robustness-steps",
    )

    rng = random.Random(args.seed)
    samples = _iter_samples(args.samples, rng=rng)
    print("search_mode: narrow_local_stability_search")
    print(
        "search_focus: center on bilateral single-support gait and trade some amplitude "
        "for more lower-phase damping and double-support recovery"
    )

    best_candidate: FsmCandidate | None = None
    best_metrics: RolloutMetrics | None = None
    best_support_candidate: FsmCandidate | None = None
    best_support_metrics: RolloutMetrics | None = None
    best_passing_candidate: FsmCandidate | None = None
    best_passing_metrics: RolloutMetrics | None = None
    rows: list[dict[str, object]] = []

    def evaluate(candidate: FsmCandidate, index: int, total: int) -> None:
        nonlocal best_candidate
        nonlocal best_metrics
        nonlocal best_support_candidate
        nonlocal best_support_metrics
        nonlocal best_passing_candidate
        nonlocal best_passing_metrics
        metrics = _rollout_candidate(
            candidate,
            steps=args.search_steps,
            seed=args.seed,
            relaxed_foot=True,
        )
        row = {
            **asdict(candidate),
            **asdict(metrics),
            "passed": _passes_acceptance(metrics),
        }
        rows.append(row)
        if best_metrics is None or _rank_key(metrics) > _rank_key(best_metrics):
            best_candidate = candidate
            best_metrics = metrics
        if (
            best_support_metrics is None
            or _support_rank_key(metrics) > _support_rank_key(best_support_metrics)
        ):
            best_support_candidate = candidate
            best_support_metrics = metrics
        if _passes_acceptance(metrics) and (
            best_passing_metrics is None
            or _rank_key(metrics) > _rank_key(best_passing_metrics)
        ):
            best_passing_candidate = candidate
            best_passing_metrics = metrics
        if index % args.print_every == 0 or row["passed"]:
            print(
                f"{index}/{total} pass={row['passed']} "
                f"done={metrics.steps_completed} term={metrics.terminated} "
                f"maxf={metrics.max_abs_fwd:.3f} minu={metrics.min_upright:.3f} "
                f"minz={metrics.min_z:.3f} support={metrics.right_only}/{metrics.left_only} "
                f"ds={candidate.double_support_steps} cap={candidate.swing_cap}"
            )

    for index, candidate in enumerate(samples, start=1):
        evaluate(candidate, index, len(samples))

    if best_support_candidate is not None:
        neighbors = _neighbor_samples(
            best_support_candidate,
            count=args.local_neighbors,
            rng=rng,
        )
        offset = len(samples)
        for index, candidate in enumerate(neighbors, start=1):
            evaluate(candidate, offset + index, offset + len(neighbors))

    _write_rows(args.out_csv, rows)
    if best_candidate is None or best_metrics is None:
        raise RuntimeError("Sweep produced no candidates.")

    print(f"\ncsv: {args.out_csv}")
    print("best_overall_candidate_json:")
    print(json.dumps(asdict(best_candidate), indent=2, sort_keys=True))
    print("\nbest_overall_metrics_json:")
    print(json.dumps(asdict(best_metrics), indent=2, sort_keys=True))
    print(f"\nbest_overall_passes_acceptance: {_passes_acceptance(best_metrics)}")

    if best_support_candidate is not None and best_support_metrics is not None:
        print("\nbest_bilateral_support_candidate_json:")
        print(json.dumps(asdict(best_support_candidate), indent=2, sort_keys=True))
        print("\nbest_bilateral_support_metrics_json:")
        print(json.dumps(asdict(best_support_metrics), indent=2, sort_keys=True))
        print(
            "\nbest_bilateral_support_passes_acceptance: "
            f"{_passes_acceptance(best_support_metrics)}"
        )

    if best_passing_candidate is not None and best_passing_metrics is not None:
        chosen_candidate = best_passing_candidate
        chosen_metrics = best_passing_metrics
        print("\nbest_passing_candidate_json:")
        print(json.dumps(asdict(best_passing_candidate), indent=2, sort_keys=True))
        print("\nbest_passing_metrics_json:")
        print(json.dumps(asdict(best_passing_metrics), indent=2, sort_keys=True))
    else:
        chosen_candidate = best_support_candidate or best_candidate
        chosen_metrics = best_support_metrics or best_metrics

    print("\n_best_fsm_function:")
    print(_render_function(chosen_candidate))

    print("\nrobustness_check:")
    for steps in robustness_steps:
        relaxed_metrics = _rollout_candidate(
            chosen_candidate,
            steps=steps,
            seed=args.seed,
            relaxed_foot=True,
        )
        default_metrics = _rollout_candidate(
            chosen_candidate,
            steps=steps,
            seed=args.seed,
            relaxed_foot=False,
        )
        print(
            "  relaxed "
            f"steps={steps} term={relaxed_metrics.terminated} "
            f"min_z={relaxed_metrics.min_z:.4f} "
            f"min_upright={relaxed_metrics.min_upright:.4f} "
            f"max_abs_fwd={relaxed_metrics.max_abs_fwd:.4f} "
            f"support={relaxed_metrics.right_only}/{relaxed_metrics.left_only} "
            f"base={relaxed_metrics.base_floor_steps}"
        )
        print(
            "  default "
            f"steps={steps} term={default_metrics.terminated} "
            f"min_z={default_metrics.min_z:.4f} "
            f"min_upright={default_metrics.min_upright:.4f} "
            f"max_abs_fwd={default_metrics.max_abs_fwd:.4f} "
            f"support={default_metrics.right_only}/{default_metrics.left_only} "
            f"base={default_metrics.base_floor_steps}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
