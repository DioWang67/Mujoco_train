"""Map Sedon foot bottom height over hip/knee/ankle qpos without dynamics."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import mujoco
import numpy as np

from sedon_baseline.env import JOINT_NAMES, SedonStandingConfig, SedonStandingEnv
from tools.sedon_debug_common import DEBUG_OUT_DIR


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "kinematic_foot_height_map.csv"
DEFAULT_HIP_VALUES = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
DEFAULT_KNEE_VALUES = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
DEFAULT_ANKLE_VALUES = (-0.2, -0.1, 0.0, 0.1, 0.2)
FOOT_INDEX_BY_LEG = {"right": 0, "left": 1}
PITCH_JOINT_INDICES = {
    "right": (2, 3, 4),
    "left": (7, 8, 9),
}


def _parse_float_list(raw_value: str) -> list[float]:
    """Parse comma-separated float values."""
    values = [float(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    return values


def _joint_range(env: SedonStandingEnv, joint_index: int) -> tuple[float, float]:
    """Return the MuJoCo range for a Sedon actuated joint index."""
    joint_id = env._joint_ids[joint_index]
    return (
        float(env.model.jnt_range[joint_id][0]),
        float(env.model.jnt_range[joint_id][1]),
    )


def _in_joint_range(env: SedonStandingEnv, joint_index: int, value: float) -> bool:
    """Return whether a qpos value is inside the joint range."""
    lower, upper = _joint_range(env, joint_index)
    return lower <= value <= upper


def _set_joint_position(env: SedonStandingEnv, joint_index: int, value: float) -> None:
    """Set one actuated joint qpos by Sedon joint index."""
    joint_id = env._joint_ids[joint_index]
    qpos_address = env.model.jnt_qposadr[joint_id]
    env.data.qpos[qpos_address] = value


def _row_for_pose(
    env: SedonStandingEnv,
    *,
    lifted_leg: str,
    hip_pitch: float,
    knee_pitch: float,
    ankle_pitch: float,
    nominal_lifted_foot_bottom_z: float,
    nominal_support_foot_bottom_z: float,
) -> dict[str, object]:
    """Set one kinematic pose, forward it, and return measurements."""
    qpos = env.init_qpos.copy()
    qvel = env.init_qvel.copy()
    env._set_base_pose(qpos)
    qvel[:] = 0.0
    env.set_state(qpos, qvel)

    hip_index, knee_index, ankle_index = PITCH_JOINT_INDICES[lifted_leg]
    _set_joint_position(env, hip_index, hip_pitch)
    _set_joint_position(env, knee_index, knee_pitch)
    _set_joint_position(env, ankle_index, ankle_pitch)
    mujoco.mj_forward(env.model, env.data)

    lifted_foot_index = FOOT_INDEX_BY_LEG[lifted_leg]
    support_foot_index = 1 - lifted_foot_index
    foot_bottoms = env._foot_bottom_heights()
    lifted_foot_bottom_z = float(foot_bottoms[lifted_foot_index])
    support_foot_bottom_z = float(foot_bottoms[support_foot_index])
    return {
        "lifted_leg": lifted_leg,
        "hip_pitch": hip_pitch,
        "knee_pitch": knee_pitch,
        "ankle_pitch": ankle_pitch,
        "lifted_foot_bottom_z": lifted_foot_bottom_z,
        "support_foot_bottom_z": support_foot_bottom_z,
        "clearance_gain": lifted_foot_bottom_z - nominal_lifted_foot_bottom_z,
        "support_foot_delta_z": support_foot_bottom_z - nominal_support_foot_bottom_z,
        "base_height": env._base_height(),
        "upright": env._base_upright(),
        "hip_in_range": _in_joint_range(env, hip_index, hip_pitch),
        "knee_in_range": _in_joint_range(env, knee_index, knee_pitch),
        "ankle_in_range": _in_joint_range(env, ankle_index, ankle_pitch),
    }


def _nominal_foot_bottoms(env: SedonStandingEnv) -> np.ndarray:
    """Return nominal foot bottom heights after reset."""
    env.reset(seed=42)
    return env._foot_bottom_heights()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write kinematic map rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hip-values", type=_parse_float_list, default=list(DEFAULT_HIP_VALUES))
    parser.add_argument("--knee-values", type=_parse_float_list, default=list(DEFAULT_KNEE_VALUES))
    parser.add_argument("--ankle-values", type=_parse_float_list, default=list(DEFAULT_ANKLE_VALUES))
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the kinematic foot-height map diagnostic."""
    args = build_parser().parse_args(argv)
    if args.top <= 0:
        raise ValueError("--top must be positive.")

    env = SedonStandingEnv(
        reset_noise_scale=0.0,
        reward_config=SedonStandingConfig(gait_mode="fsm"),
    )
    try:
        nominal_bottoms = _nominal_foot_bottoms(env)
        rows: list[dict[str, object]] = []
        for lifted_leg in ("right", "left"):
            lifted_foot_index = FOOT_INDEX_BY_LEG[lifted_leg]
            support_foot_index = 1 - lifted_foot_index
            for hip_pitch, knee_pitch, ankle_pitch in itertools.product(
                args.hip_values,
                args.knee_values,
                args.ankle_values,
            ):
                rows.append(
                    _row_for_pose(
                        env,
                        lifted_leg=lifted_leg,
                        hip_pitch=hip_pitch,
                        knee_pitch=knee_pitch,
                        ankle_pitch=ankle_pitch,
                        nominal_lifted_foot_bottom_z=float(nominal_bottoms[lifted_foot_index]),
                        nominal_support_foot_bottom_z=float(nominal_bottoms[support_foot_index]),
                    )
                )
    finally:
        env.close()

    rows.sort(
        key=lambda row: (
            float(row["lifted_foot_bottom_z"]),
            -abs(float(row["support_foot_delta_z"])),
        ),
        reverse=True,
    )
    _write_csv(args.out_csv, rows)

    print(
        "rank leg hip knee ankle lifted_z clearance support_z "
        "support_delta base_z upright in_range"
    )
    for rank, row in enumerate(rows[: args.top], start=1):
        in_range = bool(row["hip_in_range"] and row["knee_in_range"] and row["ankle_in_range"])
        print(
            f"{rank:>4} {row['lifted_leg']:>5} "
            f"{float(row['hip_pitch']):>5.2f} "
            f"{float(row['knee_pitch']):>5.2f} "
            f"{float(row['ankle_pitch']):>6.2f} "
            f"{float(row['lifted_foot_bottom_z']):>8.5f} "
            f"{float(row['clearance_gain']):>9.5f} "
            f"{float(row['support_foot_bottom_z']):>9.5f} "
            f"{float(row['support_foot_delta_z']):>13.5f} "
            f"{float(row['base_height']):>6.3f} "
            f"{float(row['upright']):>7.3f} "
            f"{in_range}"
        )

    passing = [
        row
        for row in rows
        if float(row["lifted_foot_bottom_z"]) > 0.005
        and abs(float(row["support_foot_delta_z"])) <= 0.003
        and bool(row["hip_in_range"] and row["knee_in_range"] and row["ankle_in_range"])
    ]
    print(f"\nCSV: {args.out_csv}")
    print(f"rows: {len(rows)}")
    print(f"foot_z_gt_5mm_with_support_near_nominal: {len(passing)}")
    if passing:
        best = passing[0]
        print(
            "best_pass: "
            f"leg={best['lifted_leg']} hip={float(best['hip_pitch']):.3f} "
            f"knee={float(best['knee_pitch']):.3f} "
            f"ankle={float(best['ankle_pitch']):.3f} "
            f"lifted_z={float(best['lifted_foot_bottom_z']):.5f} "
            f"clearance={float(best['clearance_gain']):.5f}"
        )
    else:
        print("best_pass: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
