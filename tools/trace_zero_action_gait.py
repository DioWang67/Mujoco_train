"""Trace Sedon zero-action gait and contact pairs over one rollout."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from sedon_baseline.env import SedonStandingConfig, SedonStandingEnv, load_sedon_config_from_env
from tools.sedon_debug_common import (
    BASE_PROXY_GEOM,
    DEBUG_OUT_DIR,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RELAXED_FOOT_SIZE,
    RIGHT_FOOT_GEOM,
    apply_foot_size_override,
    contact_pairs,
    count_contacts,
    is_base_floor_contact,
    is_expected_floor_contact,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "zero_action_trace.csv"


def _parse_optional_range(raw_value: str | None, *, option_name: str) -> tuple[float, float] | None:
    """Parse a `lower,upper` string into an optional float tuple."""
    if raw_value is None:
        return None
    parts = [part.strip() for part in raw_value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"{option_name} must be 'lower,upper'.")
    lower, upper = float(parts[0]), float(parts[1])
    if lower > upper:
        raise argparse.ArgumentTypeError(f"{option_name} lower must be <= upper.")
    return (lower, upper)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument(
        "--relaxed-foot",
        action="store_true",
        help=f"Temporarily use relaxed Sedon foot collision half-size {RELAXED_FOOT_SIZE}.",
    )
    parser.add_argument(
        "--right-knee-safe-range",
        type=lambda value: _parse_optional_range(
            value,
            option_name="--right-knee-safe-range",
        ),
        default=None,
        help="Optional right-knee soft-safe qpos range as 'lower,upper'.",
    )
    parser.add_argument(
        "--left-knee-safe-range",
        type=lambda value: _parse_optional_range(
            value,
            option_name="--left-knee-safe-range",
        ),
        default=None,
        help="Optional left-knee soft-safe qpos range as 'lower,upper'.",
    )
    return parser


def _contact_flags(pairs: list[tuple[str, str, float]]) -> dict[str, bool]:
    """Return named contact flags for the current step."""
    pair_sets = [set((name_a, name_b)) for name_a, name_b, _ in pairs]
    return {
        "floor_r_foot": {FLOOR_GEOM, RIGHT_FOOT_GEOM} in pair_sets,
        "floor_l_foot": {FLOOR_GEOM, LEFT_FOOT_GEOM} in pair_sets,
        "floor_base_proxy": {FLOOR_GEOM, BASE_PROXY_GEOM} in pair_sets,
    }


def _unexpected_contacts(pairs: list[tuple[str, str, float]]) -> list[str]:
    """Return non-foot-floor contact pair names."""
    unexpected: list[str] = []
    for name_a, name_b, _ in pairs:
        if is_expected_floor_contact(name_a, name_b):
            continue
        unexpected.append(f"{name_a}<->{name_b}")
    return unexpected


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write trace rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "base_x",
        "delta_x",
        "base_z",
        "upright",
        "forward_velocity",
        "left_foot_bottom_z",
        "right_foot_bottom_z",
        "left_contact",
        "right_contact",
        "contact_state",
        "feet_near_floor",
        "foot_flatness",
        "floor_r_foot_contact",
        "floor_l_foot_contact",
        "floor_base_proxy_contact",
        "foot_size_x",
        "foot_size_y",
        "foot_size_z",
        "right_knee_qpos",
        "left_knee_qpos",
        "right_knee_safe_violation",
        "left_knee_safe_violation",
        "knee_safe_violation_sum",
        "unexpected_contacts",
        "contact_pairs",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    """Run zero-action Sedon rollout and write contact/pose diagnostics."""
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive.")

    base_config = load_sedon_config_from_env()
    overrides: dict[str, float] = {}
    if args.right_knee_safe_range is not None:
        overrides["right_knee_safe_lower"] = args.right_knee_safe_range[0]
        overrides["right_knee_safe_upper"] = args.right_knee_safe_range[1]
    if args.left_knee_safe_range is not None:
        overrides["left_knee_safe_lower"] = args.left_knee_safe_range[0]
        overrides["left_knee_safe_upper"] = args.left_knee_safe_range[1]
    reward_config = SedonStandingConfig(**{**base_config.__dict__, **overrides})
    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=reward_config)
    rows: list[dict[str, object]] = []
    pair_counter: Counter[tuple[str, str]] = Counter()
    unexpected_counter: Counter[str] = Counter()
    contact_state_counter: Counter[str] = Counter()
    base_floor_steps = 0
    first_base_floor_step: int | None = None
    previous_x = 0.0
    terminated = False
    truncated = False
    min_base_z = float("inf")
    min_upright = float("inf")
    max_forward_velocity = -float("inf")
    max_abs_forward_velocity = 0.0
    right_knee_violation_steps = 0
    left_knee_violation_steps = 0
    foot_size = RELAXED_FOOT_SIZE if args.relaxed_foot else None
    try:
        env.reset(seed=args.seed)
        if foot_size is not None:
            apply_foot_size_override(env.model, env.data, foot_size)
            print(f"relaxed_foot_size: {foot_size}")
        previous_x = float(env.data.qpos[0])
        action = np.zeros(env.action_space.shape, dtype=np.float64)
        print(
            "step base_x delta_x base_z upright fwd_v "
            "L_foot_z R_foot_z contact_state feet flat"
        )
        for step in range(1, args.steps + 1):
            _, _, terminated, truncated, info = env.step(action)
            foot_bottoms = env._foot_bottom_heights()  # debug tool; uses env internals.
            pairs = contact_pairs(env.model, env.data)
            pair_counts = count_contacts(pairs)
            pair_counter.update(pair_counts)
            flags = _contact_flags(pairs)
            unexpected = _unexpected_contacts(pairs)
            unexpected_counter.update(unexpected)
            if flags["floor_base_proxy"]:
                base_floor_steps += 1
                if first_base_floor_step is None:
                    first_base_floor_step = step

            base_x = float(info.get("base_x_position", env.data.qpos[0]))
            delta_x = base_x - previous_x
            previous_x = base_x
            if flags["floor_r_foot"] and flags["floor_l_foot"]:
                contact_state = "both"
            elif flags["floor_r_foot"]:
                contact_state = "right_only"
            elif flags["floor_l_foot"]:
                contact_state = "left_only"
            else:
                contact_state = "none"
            contact_state_counter.update([contact_state])
            base_z = float(info.get("base_height", np.nan))
            upright = float(info.get("upright", np.nan))
            forward_velocity = float(info.get("forward_velocity", np.nan))
            min_base_z = min(min_base_z, base_z)
            min_upright = min(min_upright, upright)
            max_forward_velocity = max(max_forward_velocity, forward_velocity)
            max_abs_forward_velocity = max(max_abs_forward_velocity, abs(forward_velocity))
            right_knee_violation = float(info.get("right_knee_safe_violation", 0.0))
            left_knee_violation = float(info.get("left_knee_safe_violation", 0.0))
            right_knee_violation_steps += int(right_knee_violation > 0.0)
            left_knee_violation_steps += int(left_knee_violation > 0.0)
            row = {
                "step": step,
                "base_x": base_x,
                "delta_x": delta_x,
                "base_z": base_z,
                "upright": upright,
                "forward_velocity": forward_velocity,
                "left_foot_bottom_z": float(foot_bottoms[1]),
                "right_foot_bottom_z": float(foot_bottoms[0]),
                "left_contact": flags["floor_l_foot"],
                "right_contact": flags["floor_r_foot"],
                "contact_state": contact_state,
                "feet_near_floor": int(info.get("feet_near_floor", 0)),
                "foot_flatness": float(info.get("foot_flatness", np.nan)),
                "floor_r_foot_contact": flags["floor_r_foot"],
                "floor_l_foot_contact": flags["floor_l_foot"],
                "floor_base_proxy_contact": flags["floor_base_proxy"],
                "foot_size_x": foot_size[0] if foot_size is not None else "",
                "foot_size_y": foot_size[1] if foot_size is not None else "",
                "foot_size_z": foot_size[2] if foot_size is not None else "",
                "right_knee_qpos": float(info.get("right_knee_qpos", np.nan)),
                "left_knee_qpos": float(info.get("left_knee_qpos", np.nan)),
                "right_knee_safe_violation": right_knee_violation,
                "left_knee_safe_violation": left_knee_violation,
                "knee_safe_violation_sum": float(info.get("knee_safe_violation_sum", 0.0)),
                "unexpected_contacts": ";".join(unexpected),
                "contact_pairs": ";".join(
                    f"{name_a}<->{name_b}:{distance:.6f}"
                    for name_a, name_b, distance in pairs
                ),
            }
            rows.append(row)

            if step == 1 or step % args.print_every == 0 or terminated or truncated:
                contact_state_label = contact_state
                if flags["floor_base_proxy"]:
                    contact_state_label = f"{contact_state}+base"
                print(
                    f"{step:>4} {base_x:>7.4f} {delta_x:>7.4f} "
                    f"{row['base_z']:>6.3f} {row['upright']:>7.3f} "
                    f"{row['forward_velocity']:>7.3f} "
                    f"{row['left_foot_bottom_z']:>8.4f} "
                    f"{row['right_foot_bottom_z']:>8.4f} "
                    f"{contact_state_label:>10} "
                    f"{row['feet_near_floor']:>4} {row['foot_flatness']:>6.3f}"
                )
            if terminated or truncated:
                break
    finally:
        env.close()

    _write_rows(args.out_csv, rows)

    print(f"\ncsv: {args.out_csv}")
    print(f"steps: {len(rows)} terminated={terminated} truncated={truncated}")
    if rows:
        print(
            "final: "
            f"x={rows[-1]['base_x']:.4f} z={rows[-1]['base_z']:.4f} "
            f"upright={rows[-1]['upright']:.4f} "
            f"fwd={rows[-1]['forward_velocity']:.4f}"
        )
        print(
            "range: "
            f"min_z={min_base_z:.4f} min_upright={min_upright:.4f} "
            f"max_fwd={max_forward_velocity:.4f} "
            f"max_abs_fwd={max_abs_forward_velocity:.4f}"
        )
    print("\ncontact state counts")
    if not contact_state_counter:
        print("  none")
    for state, count in contact_state_counter.most_common():
        print(f"  {state}: {count}")
    print("\ncontact pair counts")
    if not pair_counter:
        print("  none")
    for (name_a, name_b), count in pair_counter.most_common():
        warning = ""
        if is_base_floor_contact(name_a, name_b):
            warning = " WARNING_BASE_FLOOR"
        elif not is_expected_floor_contact(name_a, name_b):
            warning = " UNEXPECTED"
        print(f"  {name_a} <-> {name_b}: {count}{warning}")

    print("\nunexpected contact counts")
    if not unexpected_counter:
        print("  none")
    for pair_name, count in unexpected_counter.most_common():
        print(f"  {pair_name}: {count}")

    if base_floor_steps:
        print(
            f"\nwarning: base_proxy touched floor for {base_floor_steps} steps; "
            f"first step={first_base_floor_step}"
        )
    else:
        print("\nbase_proxy floor contact: none")
    print("\nknee safe-range violations")
    print(f"  right_steps: {right_knee_violation_steps}")
    print(f"  left_steps : {left_knee_violation_steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
