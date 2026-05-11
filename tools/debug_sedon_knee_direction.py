"""Diagnose Sedon leg bend direction, foot collision placement, and safe knee ranges."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from sedon_baseline.env import (
    LEFT_KNEE_JOINT_INDEX,
    RIGHT_KNEE_JOINT_INDEX,
    SedonStandingConfig,
    SedonStandingEnv,
)
from tools.sedon_debug_common import DEBUG_OUT_DIR, LEFT_FOOT_GEOM, RIGHT_FOOT_GEOM, snapshot_geom


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "knee_direction_diagnostic.csv"
DEFAULT_OFFSETS = (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3)
LEG_SPECS = {
    "right": {
        "hip_index": 2,
        "knee_index": RIGHT_KNEE_JOINT_INDEX,
        "ankle_index": 4,
        "hip_body": "R_link_hip_pitch",
        "knee_body": "R_link_knee_pitch",
        "ankle_body": "R_link_ankle_pitch",
        "foot_geom": RIGHT_FOOT_GEOM,
    },
    "left": {
        "hip_index": 7,
        "knee_index": LEFT_KNEE_JOINT_INDEX,
        "ankle_index": 9,
        "hip_body": "L_link_hip_pitch",
        "knee_body": "L_link_knee_pitch",
        "ankle_body": "L_link_ankle_pitch",
        "foot_geom": LEFT_FOOT_GEOM,
    },
}
PITCH_JOINT_GROUPS = {
    "hip_pitch": "hip_index",
    "knee_pitch": "knee_index",
    "ankle_pitch": "ankle_index",
}


@dataclass(frozen=True)
class DirectionProbeRow:
    """Measured result for one leg/joint/offset diagnostic pose."""

    leg: str
    joint_name: str
    offset: float
    qpos_value: float
    in_joint_range: bool
    distance_to_lower: float
    distance_to_upper: float
    hip_x: float
    hip_z: float
    knee_x: float
    knee_z: float
    ankle_x: float
    ankle_z: float
    foot_x: float
    foot_y: float
    foot_z: float
    foot_bottom_z: float
    foot_flatness: float
    leg_reach_delta: float
    knee_bend_cross_xz: float
    bend_direction: str
    inferred_label: str


def _parse_float_list(raw_value: str) -> list[float]:
    """Parse comma-separated float values."""
    values = [float(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float offset.")
    return values


def _body_position(env: SedonStandingEnv, body_name: str) -> np.ndarray:
    """Return the world position for one named body."""
    return env.data.xpos[env._body_id(body_name)].copy()


def _sagittal_reach(hip_pos: np.ndarray, ankle_pos: np.ndarray) -> float:
    """Return x/z planar hip-to-ankle distance."""
    return float(np.linalg.norm(ankle_pos[[0, 2]] - hip_pos[[0, 2]]))


def _knee_bend_cross_xz(hip_pos: np.ndarray, knee_pos: np.ndarray, ankle_pos: np.ndarray) -> float:
    """Return signed sagittal-plane bend direction at the knee."""
    thigh = hip_pos[[0, 2]] - knee_pos[[0, 2]]
    shank = ankle_pos[[0, 2]] - knee_pos[[0, 2]]
    return float(thigh[0] * shank[1] - thigh[1] * shank[0])


def _joint_range(env: SedonStandingEnv, joint_index: int) -> tuple[float, float]:
    """Return qpos lower/upper bounds for one actuated joint index."""
    return env._joint_range(joint_index)


def _distance_to_limits(value: float, lower: float, upper: float) -> tuple[float, float]:
    """Return distances to lower/upper limits."""
    return float(value - lower), float(upper - value)


def _bend_direction_label(cross_xz: float, tolerance: float = 1e-6) -> str:
    """Return a readable bend-direction label from signed knee cross product."""
    if cross_xz > tolerance:
        return "positive_cross"
    if cross_xz < -tolerance:
        return "negative_cross"
    return "near_straight"


def _classify_pose(*, leg_reach_delta: float, cross_xz: float, expected_cross_sign: int | None) -> str:
    """Infer whether one pose looks like expected fold or reverse fold."""
    if abs(cross_xz) <= 1e-6:
        return "near_straight"
    if expected_cross_sign is None:
        return "unknown"
    bend_sign = 1 if cross_xz > 0.0 else -1
    if bend_sign != expected_cross_sign:
        return "reverse_fold"
    if leg_reach_delta < 0.0:
        return "expected_q_fold"
    return "expected_but_extending"


def _set_joint_position(env: SedonStandingEnv, joint_index: int, value: float) -> None:
    """Set one joint qpos directly on the MuJoCo state."""
    joint_id = env._joint_ids[joint_index]
    qpos_address = env.model.jnt_qposadr[joint_id]
    env.data.qpos[qpos_address] = value


def _probe_joint_direction(
    env: SedonStandingEnv,
    *,
    leg: str,
    joint_name: str,
    offset: float,
    nominal_joint_positions: np.ndarray,
    nominal_hip_pos: np.ndarray,
    nominal_ankle_pos: np.ndarray,
    expected_cross_sign: int | None,
) -> DirectionProbeRow:
    """Apply one isolated joint offset and measure resulting geometry."""
    spec = LEG_SPECS[leg]
    joint_index = spec[PITCH_JOINT_GROUPS[joint_name]]
    qpos = env.init_qpos.copy()
    qvel = env.init_qvel.copy()
    env._set_base_pose(qpos)
    qvel[:] = 0.0
    env.set_state(qpos, qvel)

    for key in ("hip_index", "knee_index", "ankle_index"):
        index = spec[key]
        _set_joint_position(env, index, float(nominal_joint_positions[index]))
    base_value = float(nominal_joint_positions[joint_index])
    qpos_value = base_value + offset
    _set_joint_position(env, joint_index, qpos_value)
    mujoco.mj_forward(env.model, env.data)

    hip_pos = _body_position(env, spec["hip_body"])
    knee_pos = _body_position(env, spec["knee_body"])
    ankle_pos = _body_position(env, spec["ankle_body"])
    foot = snapshot_geom(env.model, env.data, name=spec["foot_geom"])
    lower, upper = _joint_range(env, joint_index)
    leg_reach_delta = _sagittal_reach(hip_pos, ankle_pos) - _sagittal_reach(
        nominal_hip_pos,
        nominal_ankle_pos,
    )
    cross_xz = _knee_bend_cross_xz(hip_pos, knee_pos, ankle_pos)
    return DirectionProbeRow(
        leg=leg,
        joint_name=joint_name,
        offset=offset,
        qpos_value=qpos_value,
        in_joint_range=lower <= qpos_value <= upper,
        distance_to_lower=_distance_to_limits(qpos_value, lower, upper)[0],
        distance_to_upper=_distance_to_limits(qpos_value, lower, upper)[1],
        hip_x=float(hip_pos[0]),
        hip_z=float(hip_pos[2]),
        knee_x=float(knee_pos[0]),
        knee_z=float(knee_pos[2]),
        ankle_x=float(ankle_pos[0]),
        ankle_z=float(ankle_pos[2]),
        foot_x=float(foot.position[0]),
        foot_y=float(foot.position[1]),
        foot_z=float(foot.position[2]),
        foot_bottom_z=float(foot.bottom_z),
        foot_flatness=float(foot.flatness if foot.flatness is not None else np.nan),
        leg_reach_delta=leg_reach_delta,
        knee_bend_cross_xz=cross_xz,
        bend_direction=_bend_direction_label(cross_xz),
        inferred_label=_classify_pose(
            leg_reach_delta=leg_reach_delta,
            cross_xz=cross_xz,
            expected_cross_sign=expected_cross_sign,
        ),
    )


def _expected_knee_cross_sign(rows: list[DirectionProbeRow]) -> int | None:
    """Infer the expected knee bend sign from isolated knee sweeps."""
    knee_rows = [
        row
        for row in rows
        if row.joint_name == "knee_pitch" and row.offset != 0.0 and abs(row.knee_bend_cross_xz) > 1e-6
    ]
    if not knee_rows:
        return None
    ranked = sorted(
        knee_rows,
        key=lambda row: (-row.leg_reach_delta, row.foot_bottom_z),
    )
    best = ranked[0]
    return 1 if best.knee_bend_cross_xz > 0.0 else -1


def _safe_range_suggestion(
    env: SedonStandingEnv,
    *,
    leg: str,
    expected_cross_sign: int | None,
) -> tuple[float, float] | None:
    """Suggest a one-sided knee range that blocks the inferred reverse fold."""
    if expected_cross_sign is None:
        return None
    knee_index = LEG_SPECS[leg]["knee_index"]
    lower, upper = _joint_range(env, knee_index)
    if expected_cross_sign > 0:
        return (0.0, upper)
    return (lower, 0.0)


def _write_rows(path: Path, rows: list[DirectionProbeRow]) -> None:
    """Write detailed diagnostic rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(DirectionProbeRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offsets",
        type=_parse_float_list,
        default=list(DEFAULT_OFFSETS),
        help="Comma-separated joint offsets to probe, e.g. -0.3,-0.2,-0.1,0.1,0.2,0.3",
    )
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Sedon leg-direction diagnostic."""
    args = build_parser().parse_args(argv)
    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=SedonStandingConfig(gait_mode="fsm"))
    try:
        env.reset(seed=42)
        nominal_joint_positions = env._joint_positions().copy()
        nominal_body_positions = {
            leg: {
                "hip": _body_position(env, spec["hip_body"]),
                "ankle": _body_position(env, spec["ankle_body"]),
            }
            for leg, spec in LEG_SPECS.items()
        }
        rows: list[DirectionProbeRow] = []
        expected_signs: dict[str, int | None] = {}
        for leg in ("right", "left"):
            provisional_rows = [
                _probe_joint_direction(
                    env,
                    leg=leg,
                    joint_name="knee_pitch",
                    offset=offset,
                    nominal_joint_positions=nominal_joint_positions,
                    nominal_hip_pos=nominal_body_positions[leg]["hip"],
                    nominal_ankle_pos=nominal_body_positions[leg]["ankle"],
                    expected_cross_sign=None,
                )
                for offset in args.offsets
            ]
            expected_signs[leg] = _expected_knee_cross_sign(provisional_rows)
            rows.extend(provisional_rows)
            for joint_name in ("hip_pitch", "ankle_pitch"):
                for offset in args.offsets:
                    rows.append(
                        _probe_joint_direction(
                            env,
                            leg=leg,
                            joint_name=joint_name,
                            offset=offset,
                            nominal_joint_positions=nominal_joint_positions,
                            nominal_hip_pos=nominal_body_positions[leg]["hip"],
                            nominal_ankle_pos=nominal_body_positions[leg]["ankle"],
                            expected_cross_sign=expected_signs[leg],
                        )
                    )
        # Re-label knee rows with the inferred expected cross sign.
        rows = [
            row
            if row.joint_name != "knee_pitch"
            else DirectionProbeRow(
                **{
                    **row.__dict__,
                    "inferred_label": _classify_pose(
                        leg_reach_delta=row.leg_reach_delta,
                        cross_xz=row.knee_bend_cross_xz,
                        expected_cross_sign=expected_signs[row.leg],
                    ),
                }
            )
            for row in rows
        ]

        _write_rows(args.out_csv, rows)
        print("leg joint offset qpos foot_bottom_z foot_x foot_y foot_z reach_delta cross label")
        for row in rows:
            if row.joint_name != "knee_pitch":
                continue
            print(
                f"{row.leg:>5} {row.joint_name:>10} {row.offset:>6.2f} {row.qpos_value:>6.3f} "
                f"{row.foot_bottom_z:>12.5f} {row.foot_x:>7.4f} {row.foot_y:>7.4f} {row.foot_z:>7.4f} "
                f"{row.leg_reach_delta:>10.5f} {row.knee_bend_cross_xz:>10.5f} {row.inferred_label}"
            )

        print("\nSafe knee range suggestions")
        for leg in ("right", "left"):
            suggestion = _safe_range_suggestion(env, leg=leg, expected_cross_sign=expected_signs[leg])
            knee_index = LEG_SPECS[leg]["knee_index"]
            actual_range = _joint_range(env, knee_index)
            print(
                f"  {leg}: model_range={actual_range} "
                f"expected_cross_sign={expected_signs[leg]} "
                f"suggested_safe_range={suggestion}"
            )

        print("\nFoot collision snapshots")
        env.reset(seed=42)
        for leg, spec in LEG_SPECS.items():
            foot = snapshot_geom(env.model, env.data, name=spec["foot_geom"])
            body_name = spec["ankle_body"]
            body_pos = _body_position(env, body_name)
            print(
                f"  {leg}: geom={spec['foot_geom']} body={body_name} "
                f"geom_pos={np.round(foot.position, 4).tolist()} "
                f"body_pos={np.round(body_pos, 4).tolist()} "
                f"bottom_z={foot.bottom_z:.5f} flatness={foot.flatness:.5f}"
            )
        print(f"\nCSV: {args.out_csv}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
