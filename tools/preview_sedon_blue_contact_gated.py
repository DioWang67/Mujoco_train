"""Preview a Blue-like contact-gated Sedon gait without PPO training.

The controller keeps Sedon in conservative double support unless floor-contact
load and lateral COM shift show that the intended support foot is actually
carrying weight. This avoids the older pure open-loop "time to lift" behavior.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import mujoco
import numpy as np

from sedon_baseline.env import SedonStandingEnv, load_sedon_config_from_env
from tools.sedon_debug_common import (
    BASE_PROXY_GEOM,
    DEBUG_OUT_DIR,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    contact_pairs,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "blue_contact_gated_preview.csv"
HIP_ROLL_INDEX = {"right": 1, "left": 6}
HIP_PITCH_INDEX = {"right": 2, "left": 7}
KNEE_PITCH_INDEX = {"right": 3, "left": 8}
ANKLE_PITCH_INDEX = {"right": 4, "left": 9}


class Phase(str, Enum):
    """High-level phases for the contact-gated preview controller."""

    DOUBLE_SUPPORT_STAND = "double_support_stand"
    SHIFT_TO_LEFT = "shift_to_left"
    LIFT_RIGHT_IF_UNLOADED = "lift_right_if_unloaded"
    PLACE_RIGHT = "place_right"
    SHIFT_TO_RIGHT = "shift_to_right"
    LIFT_LEFT_IF_UNLOADED = "lift_left_if_unloaded"
    PLACE_LEFT = "place_left"


@dataclass
class ControllerState:
    """Mutable state for the preview finite-state machine."""

    phase: Phase = Phase.DOUBLE_SUPPORT_STAND
    phase_step: int = 0
    next_shift_side: str = "left"
    entered_left_support: bool = False
    entered_right_support: bool = False


@dataclass(frozen=True)
class SensorSnapshot:
    """One control-step sensor snapshot used for gating and logging."""

    com_y: float
    base_roll: float
    base_z: float
    upright: float
    left_foot_y: float
    right_foot_y: float
    left_foot_z: float
    right_foot_z: float
    left_foot_z_delta: float
    right_foot_z_delta: float
    left_contact_count: int
    right_contact_count: int
    left_normal_force: float
    right_normal_force: float
    left_force_ratio: float
    right_force_ratio: float
    left_contact: bool
    right_contact: bool
    base_proxy_contact: bool
    terminated: bool

    @property
    def center_y(self) -> float:
        """Return the midpoint between the two foot contact boxes."""
        return 0.5 * (self.left_foot_y + self.right_foot_y)


@dataclass(frozen=True)
class PreviewSummary:
    """Compact rollout summary for console output."""

    entered_left_support: bool
    entered_right_support: bool
    any_swing_foot_lift: bool
    phase_steps: dict[str, int]
    max_abs_com_y_delta: float
    max_foot_z_delta: float
    terminated_step: int | None


def _overall_com(env: SedonStandingEnv) -> np.ndarray:
    """Return whole-body COM in world coordinates."""
    masses = env.model.body_mass
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Model has no positive body mass.")
    return np.sum(env.data.xipos * masses[:, None], axis=0) / total_mass


def _quat_to_roll(quat: np.ndarray) -> float:
    """Return base roll angle in radians from a MuJoCo quaternion."""
    w, x, y, z = [float(value) for value in quat]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    return float(math.atan2(sinr_cosp, cosr_cosp))


def _foot_floor_load(env: SedonStandingEnv, side: str) -> tuple[int, float]:
    """Return floor-contact count and summed normal force for one foot."""
    foot_geom_name = LEFT_FOOT_GEOM if side == "left" else RIGHT_FOOT_GEOM
    contact_count = 0
    normal_force_sum = 0.0
    wrench = np.zeros(6, dtype=np.float64)
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        name_a = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        name_b = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        if {name_a, name_b} != {FLOOR_GEOM, foot_geom_name}:
            continue
        mujoco.mj_contactForce(env.model, env.data, contact_index, wrench)
        contact_count += 1
        normal_force_sum += abs(float(wrench[0]))
    return contact_count, normal_force_sum


def _contact_flags(env: SedonStandingEnv) -> tuple[bool, bool, bool]:
    """Return left, right, and base-proxy floor-contact flags."""
    left = False
    right = False
    base = False
    for name_a, name_b, _ in contact_pairs(env.model, env.data):
        pair = {name_a, name_b}
        if pair == {FLOOR_GEOM, LEFT_FOOT_GEOM}:
            left = True
        elif pair == {FLOOR_GEOM, RIGHT_FOOT_GEOM}:
            right = True
        elif pair == {FLOOR_GEOM, BASE_PROXY_GEOM}:
            base = True
    return left, right, base


def _smooth_alpha(env: SedonStandingEnv, step_count: int, ramp_steps: int) -> float:
    """Return a clipped smooth interpolation value."""
    if ramp_steps <= 0:
        return 1.0
    alpha = min(1.0, step_count / ramp_steps)
    return float(env._smoothstep(alpha))


def _apply_support_roll(target: np.ndarray, support_side: str, magnitude: float) -> None:
    """Apply symmetric hip-roll offsets that bias load onto one foot."""
    if support_side == "left":
        target[HIP_ROLL_INDEX["right"]] += magnitude
        target[HIP_ROLL_INDEX["left"]] -= magnitude
        return
    if support_side == "right":
        target[HIP_ROLL_INDEX["right"]] -= magnitude
        target[HIP_ROLL_INDEX["left"]] += magnitude
        return
    raise ValueError(f"Unsupported support side: {support_side}")


def _apply_swing_lift(
    target: np.ndarray,
    swing_side: str,
    *,
    hip_pitch_delta: float,
    knee_pitch_delta: float,
    ankle_pitch_delta: float,
) -> None:
    """Apply a small swing-leg micro-lift target."""
    target[HIP_PITCH_INDEX[swing_side]] += hip_pitch_delta
    target[KNEE_PITCH_INDEX[swing_side]] += knee_pitch_delta
    target[ANKLE_PITCH_INDEX[swing_side]] += ankle_pitch_delta


def _apply_shift_unload_pose(
    target: np.ndarray,
    *,
    support_side: str,
    swing_side: str,
    scale: float,
    shift_support_knee_delta: float,
    shift_support_ankle_delta: float,
    shift_unload_hip_pitch_delta: float,
    shift_unload_knee_pitch_delta: float,
    shift_unload_ankle_pitch_delta: float,
) -> None:
    """Apply a conservative pre-lift pose while both feet remain on the floor."""
    target[KNEE_PITCH_INDEX[support_side]] += shift_support_knee_delta * scale
    target[ANKLE_PITCH_INDEX[support_side]] += shift_support_ankle_delta * scale
    target[HIP_PITCH_INDEX[swing_side]] += shift_unload_hip_pitch_delta * scale
    target[KNEE_PITCH_INDEX[swing_side]] += shift_unload_knee_pitch_delta * scale
    target[ANKLE_PITCH_INDEX[swing_side]] += shift_unload_ankle_pitch_delta * scale


def _build_target(
    env: SedonStandingEnv,
    state: ControllerState,
    args: argparse.Namespace,
) -> np.ndarray:
    """Return the joint target for the current contact-gated phase."""
    target = env._nominal_joint_qpos.copy()
    shift_ramp_steps = min(args.max_shift_steps, args.shift_ramp_steps)

    if state.phase == Phase.DOUBLE_SUPPORT_STAND:
        return env._apply_safe_joint_target_clamps(target)

    if state.phase == Phase.SHIFT_TO_LEFT:
        alpha = _smooth_alpha(env, state.phase_step + 1, shift_ramp_steps)
        roll = args.shift_roll * alpha
        _apply_support_roll(target, "left", roll)
        _apply_shift_unload_pose(
            target,
            support_side="left",
            swing_side="right",
            scale=alpha,
            shift_support_knee_delta=args.shift_support_left_knee_delta,
            shift_support_ankle_delta=args.shift_support_left_ankle_delta,
            shift_unload_hip_pitch_delta=args.shift_unload_right_hip_pitch_delta,
            shift_unload_knee_pitch_delta=args.shift_unload_right_knee_pitch_delta,
            shift_unload_ankle_pitch_delta=args.shift_unload_right_ankle_pitch_delta,
        )
        return env._apply_safe_joint_target_clamps(target)

    if state.phase == Phase.LIFT_RIGHT_IF_UNLOADED:
        _apply_support_roll(target, "left", args.shift_roll)
        _apply_shift_unload_pose(
            target,
            support_side="left",
            swing_side="right",
            scale=1.0,
            shift_support_knee_delta=args.shift_support_left_knee_delta,
            shift_support_ankle_delta=args.shift_support_left_ankle_delta,
            shift_unload_hip_pitch_delta=args.shift_unload_right_hip_pitch_delta,
            shift_unload_knee_pitch_delta=args.shift_unload_right_knee_pitch_delta,
            shift_unload_ankle_pitch_delta=args.shift_unload_right_ankle_pitch_delta,
        )
        lift_alpha = _smooth_alpha(env, state.phase_step + 1, args.max_lift_steps)
        _apply_swing_lift(
            target,
            "right",
            hip_pitch_delta=args.swing_hip_pitch_delta * lift_alpha,
            knee_pitch_delta=args.swing_knee_pitch_delta * lift_alpha,
            ankle_pitch_delta=args.swing_ankle_pitch_delta * lift_alpha,
        )
        return env._apply_safe_joint_target_clamps(target)

    if state.phase == Phase.PLACE_RIGHT:
        place_alpha = 1.0 - _smooth_alpha(env, state.phase_step + 1, args.max_place_steps)
        _apply_support_roll(target, "left", args.shift_roll * place_alpha)
        _apply_shift_unload_pose(
            target,
            support_side="left",
            swing_side="right",
            scale=place_alpha,
            shift_support_knee_delta=args.shift_support_left_knee_delta,
            shift_support_ankle_delta=args.shift_support_left_ankle_delta,
            shift_unload_hip_pitch_delta=args.shift_unload_right_hip_pitch_delta,
            shift_unload_knee_pitch_delta=args.shift_unload_right_knee_pitch_delta,
            shift_unload_ankle_pitch_delta=args.shift_unload_right_ankle_pitch_delta,
        )
        _apply_swing_lift(
            target,
            "right",
            hip_pitch_delta=args.swing_hip_pitch_delta * place_alpha,
            knee_pitch_delta=args.swing_knee_pitch_delta * place_alpha,
            ankle_pitch_delta=args.swing_ankle_pitch_delta * place_alpha,
        )
        return env._apply_safe_joint_target_clamps(target)

    if state.phase == Phase.SHIFT_TO_RIGHT:
        alpha = _smooth_alpha(env, state.phase_step + 1, shift_ramp_steps)
        roll = args.shift_roll * alpha
        _apply_support_roll(target, "right", roll)
        _apply_shift_unload_pose(
            target,
            support_side="right",
            swing_side="left",
            scale=alpha,
            shift_support_knee_delta=args.shift_support_right_knee_delta,
            shift_support_ankle_delta=args.shift_support_right_ankle_delta,
            shift_unload_hip_pitch_delta=args.shift_unload_left_hip_pitch_delta,
            shift_unload_knee_pitch_delta=args.shift_unload_left_knee_pitch_delta,
            shift_unload_ankle_pitch_delta=args.shift_unload_left_ankle_pitch_delta,
        )
        return env._apply_safe_joint_target_clamps(target)

    if state.phase == Phase.LIFT_LEFT_IF_UNLOADED:
        _apply_support_roll(target, "right", args.shift_roll)
        _apply_shift_unload_pose(
            target,
            support_side="right",
            swing_side="left",
            scale=1.0,
            shift_support_knee_delta=args.shift_support_right_knee_delta,
            shift_support_ankle_delta=args.shift_support_right_ankle_delta,
            shift_unload_hip_pitch_delta=args.shift_unload_left_hip_pitch_delta,
            shift_unload_knee_pitch_delta=args.shift_unload_left_knee_pitch_delta,
            shift_unload_ankle_pitch_delta=args.shift_unload_left_ankle_pitch_delta,
        )
        lift_alpha = _smooth_alpha(env, state.phase_step + 1, args.max_lift_steps)
        _apply_swing_lift(
            target,
            "left",
            hip_pitch_delta=args.swing_hip_pitch_delta * lift_alpha,
            knee_pitch_delta=args.swing_knee_pitch_delta * lift_alpha,
            ankle_pitch_delta=args.swing_ankle_pitch_delta * lift_alpha,
        )
        return env._apply_safe_joint_target_clamps(target)

    if state.phase == Phase.PLACE_LEFT:
        place_alpha = 1.0 - _smooth_alpha(env, state.phase_step + 1, args.max_place_steps)
        _apply_support_roll(target, "right", args.shift_roll * place_alpha)
        _apply_shift_unload_pose(
            target,
            support_side="right",
            swing_side="left",
            scale=place_alpha,
            shift_support_knee_delta=args.shift_support_right_knee_delta,
            shift_support_ankle_delta=args.shift_support_right_ankle_delta,
            shift_unload_hip_pitch_delta=args.shift_unload_left_hip_pitch_delta,
            shift_unload_knee_pitch_delta=args.shift_unload_left_knee_pitch_delta,
            shift_unload_ankle_pitch_delta=args.shift_unload_left_ankle_pitch_delta,
        )
        _apply_swing_lift(
            target,
            "left",
            hip_pitch_delta=args.swing_hip_pitch_delta * place_alpha,
            knee_pitch_delta=args.swing_knee_pitch_delta * place_alpha,
            ankle_pitch_delta=args.swing_ankle_pitch_delta * place_alpha,
        )
        return env._apply_safe_joint_target_clamps(target)

    raise ValueError(f"Unsupported phase: {state.phase}")


def _sample_snapshot(
    env: SedonStandingEnv,
    *,
    initial_left_foot_z: float,
    initial_right_foot_z: float,
) -> SensorSnapshot:
    """Capture the sensor values used by the phase gates."""
    left_contact_count, left_normal_force = _foot_floor_load(env, "left")
    right_contact_count, right_normal_force = _foot_floor_load(env, "right")
    total_normal_force = left_normal_force + right_normal_force
    if total_normal_force > 1e-9:
        left_force_ratio = left_normal_force / total_normal_force
        right_force_ratio = right_normal_force / total_normal_force
    else:
        left_force_ratio = 0.0
        right_force_ratio = 0.0

    left_contact, right_contact, base_proxy_contact = _contact_flags(env)
    foot_bottoms = env._foot_bottom_heights()
    left_foot_geom_id = env._geom_id(LEFT_FOOT_GEOM)
    right_foot_geom_id = env._geom_id(RIGHT_FOOT_GEOM)
    com_y = float(_overall_com(env)[1])
    base_z = env._base_height()
    upright = env._base_upright()
    observation = env._get_obs()
    terminated = env._is_terminated(base_z, upright, observation)

    return SensorSnapshot(
        com_y=com_y,
        base_roll=_quat_to_roll(env.data.xquat[env._base_body_id]),
        base_z=float(base_z),
        upright=float(upright),
        left_foot_y=float(env.data.geom_xpos[left_foot_geom_id][1]),
        right_foot_y=float(env.data.geom_xpos[right_foot_geom_id][1]),
        left_foot_z=float(foot_bottoms[1]),
        right_foot_z=float(foot_bottoms[0]),
        left_foot_z_delta=float(foot_bottoms[1] - initial_left_foot_z),
        right_foot_z_delta=float(foot_bottoms[0] - initial_right_foot_z),
        left_contact_count=left_contact_count,
        right_contact_count=right_contact_count,
        left_normal_force=float(left_normal_force),
        right_normal_force=float(right_normal_force),
        left_force_ratio=float(left_force_ratio),
        right_force_ratio=float(right_force_ratio),
        left_contact=left_contact,
        right_contact=right_contact,
        base_proxy_contact=base_proxy_contact,
        terminated=terminated,
    )


def _allow_swing_lift(
    state: ControllerState,
    snapshot: SensorSnapshot,
    args: argparse.Namespace,
) -> bool:
    """Return whether the swing-lift gate is currently open."""
    if state.phase == Phase.SHIFT_TO_LEFT:
        return (
            snapshot.left_contact
            and snapshot.right_contact
            and snapshot.left_force_ratio > args.shift_force_ratio_gate
            and snapshot.com_y >= snapshot.center_y + args.min_com_shift
        )
    if state.phase == Phase.LIFT_RIGHT_IF_UNLOADED:
        return snapshot.left_contact and snapshot.left_force_ratio >= args.lift_cancel_force_ratio
    if state.phase == Phase.SHIFT_TO_RIGHT:
        return (
            snapshot.left_contact
            and snapshot.right_contact
            and snapshot.right_force_ratio > args.shift_force_ratio_gate
            and snapshot.com_y <= snapshot.center_y - args.min_com_shift
        )
    if state.phase == Phase.LIFT_LEFT_IF_UNLOADED:
        return snapshot.right_contact and snapshot.right_force_ratio >= args.lift_cancel_force_ratio
    return False


def _needs_safe_recovery(
    env: SedonStandingEnv,
    snapshot: SensorSnapshot,
    args: argparse.Namespace,
) -> bool:
    """Return whether the controller should immediately fall back to safe stand."""
    min_safe_height = env._reward_config.min_base_height + args.recover_height_margin
    return bool(
        snapshot.base_proxy_contact
        or snapshot.upright < args.recover_upright
        or abs(snapshot.base_roll) > args.recover_roll
        or snapshot.base_z < min_safe_height
    )


def _maybe_transition_phase(
    state: ControllerState,
    snapshot: SensorSnapshot,
    args: argparse.Namespace,
) -> None:
    """Advance the controller FSM using contact/load/COM-based gates."""
    phase = state.phase
    phase_steps = state.phase_step + 1

    if phase == Phase.DOUBLE_SUPPORT_STAND:
        centered = abs(snapshot.com_y - snapshot.center_y) <= args.centering_com_tolerance
        if (
            phase_steps >= args.stand_hold_steps
            and snapshot.left_contact
            and snapshot.right_contact
            and centered
        ):
            if state.next_shift_side == "left":
                state.phase = Phase.SHIFT_TO_LEFT
            else:
                state.phase = Phase.SHIFT_TO_RIGHT
            state.phase_step = 0
            return
        state.phase_step = phase_steps
        return

    if phase == Phase.SHIFT_TO_LEFT:
        if _allow_swing_lift(state, snapshot, args):
            state.entered_left_support = True
            state.phase = Phase.LIFT_RIGHT_IF_UNLOADED
            state.phase_step = 0
            return
        if phase_steps >= args.max_shift_steps:
            state.phase = Phase.DOUBLE_SUPPORT_STAND
            state.next_shift_side = "left"
            state.phase_step = 0
            return
        state.phase_step = phase_steps
        return

    if phase == Phase.LIFT_RIGHT_IF_UNLOADED:
        if not _allow_swing_lift(state, snapshot, args):
            state.phase = Phase.DOUBLE_SUPPORT_STAND
            state.next_shift_side = "left"
            state.phase_step = 0
            return
        if (
            snapshot.right_foot_z_delta >= args.target_lift_z
            or phase_steps >= args.max_lift_steps
        ):
            state.phase = Phase.PLACE_RIGHT
            state.phase_step = 0
            return
        state.phase_step = phase_steps
        return

    if phase == Phase.PLACE_RIGHT:
        placed = (
            snapshot.right_contact
            and snapshot.right_foot_z_delta <= args.place_contact_z_tolerance
        )
        if placed and phase_steps >= args.min_place_steps:
            state.phase = Phase.SHIFT_TO_RIGHT
            state.next_shift_side = "right"
            state.phase_step = 0
            return
        if phase_steps >= args.max_place_steps:
            state.phase = (
                Phase.SHIFT_TO_RIGHT if snapshot.left_contact and snapshot.right_contact else Phase.DOUBLE_SUPPORT_STAND
            )
            state.next_shift_side = "right"
            state.phase_step = 0
            return
        state.phase_step = phase_steps
        return

    if phase == Phase.SHIFT_TO_RIGHT:
        if _allow_swing_lift(state, snapshot, args):
            state.entered_right_support = True
            state.phase = Phase.LIFT_LEFT_IF_UNLOADED
            state.phase_step = 0
            return
        if phase_steps >= args.max_shift_steps:
            state.phase = Phase.DOUBLE_SUPPORT_STAND
            state.next_shift_side = "right"
            state.phase_step = 0
            return
        state.phase_step = phase_steps
        return

    if phase == Phase.LIFT_LEFT_IF_UNLOADED:
        if not _allow_swing_lift(state, snapshot, args):
            state.phase = Phase.DOUBLE_SUPPORT_STAND
            state.next_shift_side = "right"
            state.phase_step = 0
            return
        if snapshot.left_foot_z_delta >= args.target_lift_z or phase_steps >= args.max_lift_steps:
            state.phase = Phase.PLACE_LEFT
            state.phase_step = 0
            return
        state.phase_step = phase_steps
        return

    if phase == Phase.PLACE_LEFT:
        placed = (
            snapshot.left_contact
            and snapshot.left_foot_z_delta <= args.place_contact_z_tolerance
        )
        if placed and phase_steps >= args.min_place_steps:
            state.phase = Phase.SHIFT_TO_LEFT
            state.next_shift_side = "left"
            state.phase_step = 0
            return
        if phase_steps >= args.max_place_steps:
            state.phase = (
                Phase.SHIFT_TO_LEFT if snapshot.left_contact and snapshot.right_contact else Phase.DOUBLE_SUPPORT_STAND
            )
            state.next_shift_side = "left"
            state.phase_step = 0
            return
        state.phase_step = phase_steps
        return

    raise ValueError(f"Unsupported phase: {phase}")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write preview rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--render-viewer", action="store_true")
    parser.add_argument("--viewer-sleep", type=float, default=0.0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--stand-hold-steps", type=int, default=24)
    parser.add_argument("--max-shift-steps", type=int, default=90)
    parser.add_argument("--shift-ramp-steps", type=int, default=30)
    parser.add_argument("--max-lift-steps", type=int, default=28)
    parser.add_argument("--min-place-steps", type=int, default=8)
    parser.add_argument("--max-place-steps", type=int, default=30)
    parser.add_argument("--shift-roll", type=float, default=0.10)
    parser.add_argument("--shift-support-left-knee-delta", type=float, default=0.0)
    parser.add_argument("--shift-support-left-ankle-delta", type=float, default=0.0)
    parser.add_argument("--shift-support-right-knee-delta", type=float, default=0.0)
    parser.add_argument("--shift-support-right-ankle-delta", type=float, default=0.0)
    parser.add_argument("--shift-unload-right-hip-pitch-delta", type=float, default=0.0)
    parser.add_argument("--shift-unload-right-knee-pitch-delta", type=float, default=0.0)
    parser.add_argument("--shift-unload-right-ankle-pitch-delta", type=float, default=0.0)
    parser.add_argument("--shift-unload-left-hip-pitch-delta", type=float, default=0.0)
    parser.add_argument("--shift-unload-left-knee-pitch-delta", type=float, default=0.0)
    parser.add_argument("--shift-unload-left-ankle-pitch-delta", type=float, default=0.0)
    parser.add_argument("--min-com-shift", type=float, default=0.008)
    parser.add_argument("--centering-com-tolerance", type=float, default=0.010)
    parser.add_argument("--shift-force-ratio-gate", type=float, default=0.65)
    parser.add_argument("--lift-cancel-force-ratio", type=float, default=0.55)
    parser.add_argument("--target-lift-z", type=float, default=0.010)
    parser.add_argument("--place-contact-z-tolerance", type=float, default=0.003)
    parser.add_argument("--recover-upright", type=float, default=0.82)
    parser.add_argument("--recover-roll", type=float, default=0.30)
    parser.add_argument("--recover-height-margin", type=float, default=0.02)
    parser.add_argument("--swing-hip-pitch-delta", type=float, default=-0.015)
    parser.add_argument("--swing-knee-pitch-delta", type=float, default=-0.030)
    parser.add_argument("--swing-ankle-pitch-delta", type=float, default=0.015)
    return parser


def _summarize_rollout(
    state: ControllerState,
    phase_counts: Counter[str],
    *,
    max_abs_com_y_delta: float,
    max_foot_z_delta: float,
    any_swing_foot_lift: bool,
    terminated_step: int | None,
) -> PreviewSummary:
    """Build a compact console summary."""
    return PreviewSummary(
        entered_left_support=state.entered_left_support,
        entered_right_support=state.entered_right_support,
        any_swing_foot_lift=any_swing_foot_lift,
        phase_steps={phase.value: int(phase_counts[phase.value]) for phase in Phase},
        max_abs_com_y_delta=max_abs_com_y_delta,
        max_foot_z_delta=max_foot_z_delta,
        terminated_step=terminated_step,
    )


def _phase_uses_right_swing(phase: Phase) -> bool:
    """Return whether the phase should count right-foot swing clearance."""
    return phase in (Phase.LIFT_RIGHT_IF_UNLOADED, Phase.PLACE_RIGHT)


def _phase_uses_left_swing(phase: Phase) -> bool:
    """Return whether the phase should count left-foot swing clearance."""
    return phase in (Phase.LIFT_LEFT_IF_UNLOADED, Phase.PLACE_LEFT)


def main(argv: list[str] | None = None) -> int:
    """Run the Blue-like contact-gated preview."""
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.viewer_sleep < 0.0:
        raise ValueError("--viewer-sleep must be non-negative.")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive.")
    if args.max_shift_steps <= 0:
        raise ValueError("--max-shift-steps must be positive.")
    if args.max_lift_steps <= 0:
        raise ValueError("--max-lift-steps must be positive.")
    if args.max_place_steps <= 0:
        raise ValueError("--max-place-steps must be positive.")
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative.")

    reward_config = load_sedon_config_from_env()
    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=reward_config)
    rows: list[dict[str, object]] = []
    phase_counts: Counter[str] = Counter()
    state = ControllerState()
    terminated_step: int | None = None
    max_abs_com_y_delta = 0.0
    max_foot_z_delta = 0.0
    any_swing_foot_lift = False

    try:
        env.reset(seed=args.seed)
        if args.settle_steps > 0:
            nominal_target = env._apply_safe_joint_target_clamps(env._nominal_joint_qpos.copy())
            for _ in range(args.settle_steps):
                env._do_pd_simulation(nominal_target)
        initial_com_y = float(_overall_com(env)[1])
        initial_foot_bottoms = env._foot_bottom_heights()
        initial_left_foot_z = float(initial_foot_bottoms[1])
        initial_right_foot_z = float(initial_foot_bottoms[0])

        viewer = None
        if args.render_viewer:
            try:
                import mujoco.viewer
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "mujoco.viewer is unavailable in this Python environment. "
                    "Run without --render-viewer for headless preview."
                ) from exc
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            viewer.__enter__()

        print(
            "step phase com_y base_roll L_ratio R_ratio "
            "L_z_delta R_z_delta allow_lift terminated"
        )
        try:
            for step in range(1, args.steps + 1):
                last_snapshot = _sample_snapshot(
                    env,
                    initial_left_foot_z=initial_left_foot_z,
                    initial_right_foot_z=initial_right_foot_z,
                )
                if _needs_safe_recovery(env, last_snapshot, args):
                    state.phase = Phase.DOUBLE_SUPPORT_STAND
                    state.phase_step = 0

                active_phase = state.phase
                target_positions = _build_target(env, state, args)
                env._do_pd_simulation(target_positions)

                snapshot = _sample_snapshot(
                    env,
                    initial_left_foot_z=initial_left_foot_z,
                    initial_right_foot_z=initial_right_foot_z,
                )
                allow_swing_lift = _allow_swing_lift(state, snapshot, args)

                max_abs_com_y_delta = max(max_abs_com_y_delta, abs(snapshot.com_y - initial_com_y))
                if _phase_uses_right_swing(active_phase):
                    max_foot_z_delta = max(max_foot_z_delta, snapshot.right_foot_z_delta)
                    any_swing_foot_lift = any_swing_foot_lift or snapshot.right_foot_z_delta > 0.005
                elif _phase_uses_left_swing(active_phase):
                    max_foot_z_delta = max(max_foot_z_delta, snapshot.left_foot_z_delta)
                    any_swing_foot_lift = any_swing_foot_lift or snapshot.left_foot_z_delta > 0.005
                phase_counts[active_phase.value] += 1

                row = {
                    "step": step,
                    "phase": active_phase.value,
                    "com_y": snapshot.com_y,
                    "base_roll": snapshot.base_roll,
                    "left_foot_y": snapshot.left_foot_y,
                    "right_foot_y": snapshot.right_foot_y,
                    "left_foot_z": snapshot.left_foot_z,
                    "right_foot_z": snapshot.right_foot_z,
                    "left_contact_count": snapshot.left_contact_count,
                    "right_contact_count": snapshot.right_contact_count,
                    "left_normal_force": snapshot.left_normal_force,
                    "right_normal_force": snapshot.right_normal_force,
                    "left_force_ratio": snapshot.left_force_ratio,
                    "right_force_ratio": snapshot.right_force_ratio,
                    "allow_swing_lift": allow_swing_lift,
                    "target_left_hip_roll": float(target_positions[HIP_ROLL_INDEX["left"]]),
                    "target_right_hip_roll": float(target_positions[HIP_ROLL_INDEX["right"]]),
                    "target_left_hip_pitch": float(target_positions[HIP_PITCH_INDEX["left"]]),
                    "target_right_hip_pitch": float(target_positions[HIP_PITCH_INDEX["right"]]),
                    "target_left_knee_pitch": float(target_positions[KNEE_PITCH_INDEX["left"]]),
                    "target_right_knee_pitch": float(target_positions[KNEE_PITCH_INDEX["right"]]),
                    "terminated": snapshot.terminated,
                }
                rows.append(row)

                if step == 1 or step % args.print_every == 0 or snapshot.terminated:
                    print(
                        f"{step:>4} {active_phase.value:>24} {snapshot.com_y:>7.4f} "
                        f"{snapshot.base_roll:>9.4f} {snapshot.left_force_ratio:>7.3f} "
                        f"{snapshot.right_force_ratio:>7.3f} {snapshot.left_foot_z_delta:>9.4f} "
                        f"{snapshot.right_foot_z_delta:>9.4f} {str(allow_swing_lift):>10} "
                        f"{str(snapshot.terminated):>10}"
                    )

                if viewer is not None:
                    viewer.sync()
                    if args.viewer_sleep > 0.0:
                        time.sleep(args.viewer_sleep)
                    if not viewer.is_running():
                        break

                if snapshot.terminated:
                    terminated_step = step
                    break

                if _needs_safe_recovery(env, snapshot, args):
                    state.phase = Phase.DOUBLE_SUPPORT_STAND
                    state.phase_step = 0
                else:
                    _maybe_transition_phase(state, snapshot, args)
        finally:
            if viewer is not None:
                viewer.__exit__(None, None, None)
    finally:
        env.close()

    _write_rows(args.out_csv, rows)
    summary = _summarize_rollout(
        state,
        phase_counts,
        max_abs_com_y_delta=max_abs_com_y_delta,
        max_foot_z_delta=max_foot_z_delta,
        any_swing_foot_lift=any_swing_foot_lift,
        terminated_step=terminated_step,
    )

    print(f"\ncsv: {args.out_csv}")
    print(f"steps: {len(rows)}")
    print(f"entered_left_support: {summary.entered_left_support}")
    print(f"entered_right_support: {summary.entered_right_support}")
    print(f"any_swing_foot_lift_gt_0.005m: {summary.any_swing_foot_lift}")
    for phase_name, count in summary.phase_steps.items():
        print(f"{phase_name}_steps: {count}")
    print(f"max_abs_com_y_delta: {summary.max_abs_com_y_delta:.5f}")
    print(f"max_foot_z_delta: {summary.max_foot_z_delta:.5f}")
    print(f"terminated_step: {summary.terminated_step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
