"""Trace Sedon COM-shift curriculum behavior under zero action."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from sedon_baseline.env import (
    SedonStandingEnv,
    load_sedon_config_from_env,
)
from tools.sedon_debug_common import (
    DEBUG_OUT_DIR,
    RELAXED_FOOT_SIZE,
    apply_foot_size_override,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "com_shift_trace.csv"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--relaxed-foot", action="store_true")
    return parser


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write trace rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    """Run the zero-action COM-shift trace."""
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive.")

    reward_config = load_sedon_config_from_env()
    if reward_config.task_mode.lower() != "com_shift":
        reward_config = replace(reward_config, task_mode="com_shift")

    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=reward_config)
    rows: list[dict[str, object]] = []
    phase_counter: Counter[str] = Counter()
    terminated = False
    truncated = False
    min_base_z = float("inf")
    min_upright = float("inf")
    max_abs_fwd = 0.0
    try:
        env.reset(seed=args.seed)
        if args.relaxed_foot:
            apply_foot_size_override(env.model, env.data, RELAXED_FOOT_SIZE)
            print(f"relaxed_foot_size: {RELAXED_FOOT_SIZE}")
        action = np.zeros(env.action_space.shape, dtype=np.float64)
        print(
            "step phase support base_y target_y base_z upright fwd_v "
            "support_contact swing_contact right_contact left_contact"
        )
        for step in range(1, args.steps + 1):
            _, _, terminated, truncated, info = env.step(action)
            phase_name = str(info["phase_name"])
            phase_counter.update([phase_name])
            base_z = float(info["base_height"])
            upright = float(info["upright"])
            fwd_v = float(info["forward_velocity"])
            min_base_z = min(min_base_z, base_z)
            min_upright = min(min_upright, upright)
            max_abs_fwd = max(max_abs_fwd, abs(fwd_v))
            row = {
                "step": step,
                "phase_name": phase_name,
                "support_side": str(info["support_side"]),
                "base_y_position": float(info["base_y_position"]),
                "desired_base_y": float(info["desired_base_y"]),
                "base_height": base_z,
                "upright": upright,
                "forward_velocity": fwd_v,
                "support_contact": bool(info["support_contact"]),
                "swing_contact": bool(info["swing_contact"]),
                "right_contact": bool(info["right_contact"]),
                "left_contact": bool(info["left_contact"]),
                "base_proxy_floor_contact": bool(info["base_proxy_floor_contact"]),
                "support_foot_bottom_z": float(info["support_foot_bottom_z"]),
                "swing_foot_bottom_z": float(info["swing_foot_bottom_z"]),
                "reward_total": float(info["reward_total"]),
                "reward_lateral_target": float(info.get("reward_lateral_target", np.nan)),
                "reward_lateral_progress": float(info.get("reward_lateral_progress", np.nan)),
                "reward_progress_gate": float(info.get("reward_progress_gate", np.nan)),
                "reward_both_contact_reward": float(
                    info.get("reward_both_contact_reward", np.nan)
                ),
                "reward_single_contact_penalty": float(
                    info.get("reward_single_contact_penalty", np.nan)
                ),
                "reward_no_contact_penalty": float(
                    info.get("reward_no_contact_penalty", np.nan)
                ),
                "reward_forward_velocity_abs_penalty": float(
                    info.get("reward_forward_velocity_abs_penalty", np.nan)
                ),
                "reward_forward_displacement_penalty": float(
                    info.get("reward_forward_displacement_penalty", np.nan)
                ),
                "reward_support_contact": float(info.get("reward_support_contact", np.nan)),
                "reward_double_support_contact": float(
                    info.get("reward_double_support_contact", np.nan)
                ),
                "reward_swing_unload": float(info.get("reward_swing_unload", np.nan)),
            }
            rows.append(row)
            if step == 1 or step % args.print_every == 0 or terminated or truncated:
                print(
                    f"{step:>4} {phase_name:>18} {str(info['support_side']):>7} "
                    f"{row['base_y_position']:>7.4f} {row['desired_base_y']:>8.4f} "
                    f"{base_z:>6.3f} {upright:>7.3f} {fwd_v:>7.3f} "
                    f"{str(row['support_contact']):>15} {str(row['swing_contact']):>13} "
                    f"{str(row['right_contact']):>13} {str(row['left_contact']):>12}"
                )
            if terminated or truncated:
                break
    finally:
        env.close()

    _write_rows(args.out_csv, rows)
    print(f"\ncsv: {args.out_csv}")
    print(f"steps: {len(rows)} terminated={terminated} truncated={truncated}")
    print(
        "range: "
        f"min_z={min_base_z:.4f} min_upright={min_upright:.4f} "
        f"max_abs_fwd={max_abs_fwd:.4f}"
    )
    print("\nphase counts")
    for phase_name, count in phase_counter.most_common():
        print(f"  {phase_name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
