"""Preview a Blue-like closed-loop balance controller for Seedon.

This tool is a controller prototype, not a PPO/reward/gait seed change. It
adds a small state estimator plus a contact-aware balance controller on top of
Seedon's existing joint-space PD tracking.
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

from seedon_baseline.env import SeedonStandingEnv, load_seedon_config_from_env
from tools.seedon_debug_common import (
    BASE_PROXY_GEOM,
    DEBUG_OUT_DIR,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    contact_pairs,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "blue_balance_controller_preview.csv"
HIP_ROLL_INDEX = {"right": 1, "left": 6}
HIP_PITCH_INDEX = {"right": 2, "left": 7}
KNEE_PITCH_INDEX = {"right": 3, "left": 8}
ANKLE_PITCH_INDEX = {"right": 4, "left": 9}
SUPPORT_TO_SWING = {"left": "right", "right": "left"}
DEFAULT_ABLATION_MODES = (
    "full_controller",
    "no_base_roll_stabilizer",
    "no_com_feedback",
    "no_force_ratio_feedback",
    "support_roll_only",
)


class Phase(str, Enum):
    """High-level phases for the closed-loop balance controller."""

    DOUBLE_SUPPORT_STAND = "double_support_stand"
    ACQUIRE_LEFT_SUPPORT = "acquire_left_support"
    LIFT_RIGHT = "lift_right"
    PLACE_RIGHT = "place_right"
    ACQUIRE_RIGHT_SUPPORT = "acquire_right_support"
    LIFT_LEFT = "lift_left"
    PLACE_LEFT = "place_left"


@dataclass
class RuntimeState:
    """Mutable rollout/controller state."""

    phase: Phase = Phase.DOUBLE_SUPPORT_STAND
    phase_step: int = 0
    next_support_side: str = "left"
    previous_com_y: float = 0.0
    previous_support_com_shift: float = 0.0
    entered_left_support: bool = False
    entered_right_support: bool = False


@dataclass(frozen=True)
class StateEstimate:
    """State estimator output for the balance controller."""

    com_y: float
    com_y_velocity: float
    base_roll: float
    base_roll_velocity: float
    left_force_ratio: float
    right_force_ratio: float
    left_normal_force: float
    right_normal_force: float
    left_contact_count: int
    right_contact_count: int
    left_contact: bool
    right_contact: bool
    base_proxy_contact: bool
    left_foot_y: float
    right_foot_y: float
    left_foot_z: float
    right_foot_z: float
    left_foot_z_delta: float
    right_foot_z_delta: float
    support_side: str
    swing_side: str
    support_force_ratio: float
    swing_force_ratio: float
    support_com_shift: float
    support_com_shift_velocity: float
    base_z: float
    upright: float
    terminated: bool

    @property
    def center_y(self) -> float:
        """Return midpoint between the two feet."""
        return 0.5 * (self.left_foot_y + self.right_foot_y)


@dataclass(frozen=True)
class ControllerCommand:
    """Controller output mapped to joint-space targets."""

    support_side: str
    swing_side: str
    phase: str
    target_support_force_ratio: float
    target_support_com_shift: float
    support_roll_cmd: float
    unload_scale: float
    swing_lift_scale: float
    allow_lift: bool
    target_positions: np.ndarray


@dataclass(frozen=True)
class PreviewSummary:
    """Compact rollout summary."""

    entered_left_support: bool
    entered_right_support: bool
    phase_steps: dict[str, int]
    max_abs_com_y_delta: float
    max_left_force_ratio: float
    max_right_force_ratio: float
    max_left_support_com_shift: float
    max_right_support_com_shift: float
    any_swing_lift: bool
    terminated_step: int | None
    max_support_roll_cmd: float
    mean_support_roll_cmd_last_50: float
    max_unload_scale: float
    mean_unload_scale_last_50: float
    mean_hip_roll_tracking_error_last_50: float
    corr_support_roll_cmd_support_force_ratio: float
    corr_support_roll_cmd_support_com_shift: float
    corr_support_force_ratio_support_com_shift: float


@dataclass(frozen=True)
class AblationSummary:
    """One acquire-support ablation result."""

    mode: str
    side: str
    max_support_force_ratio: float
    mean_support_force_ratio_last_50: float
    max_support_com_shift: float
    mean_support_com_shift_last_50: float
    max_base_roll: float
    terminated_step: int | None
    both_contact_ratio: float
    none_contact_ratio: float
    max_support_roll_cmd: float
    mean_support_roll_cmd_last_50: float
    max_unload_scale: float
    mean_unload_scale_last_50: float
    mean_hip_roll_tracking_error_last_50: float
    corr_support_roll_cmd_support_force_ratio: float
    corr_support_roll_cmd_support_com_shift: float
    corr_support_force_ratio_support_com_shift: float


def _overall_com(env: SeedonStandingEnv) -> np.ndarray:
    """Return whole-body COM in world coordinates."""
    masses = env.model.body_mass
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Model has no positive body mass.")
    return np.sum(env.data.xipos * masses[:, None], axis=0) / total_mass


def _quat_to_roll(quat: np.ndarray) -> float:
    """Return base roll from a MuJoCo quaternion."""
    w, x, y, z = [float(value) for value in quat]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    return float(math.atan2(sinr_cosp, cosr_cosp))


def _foot_floor_load(env: SeedonStandingEnv, side: str) -> tuple[int, float]:
    """Return floor-contact count and normal-force sum for one foot."""
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


def _contact_flags(env: SeedonStandingEnv) -> tuple[bool, bool, bool]:
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


def _contact_state_label(left_contact: bool, right_contact: bool) -> str:
    """Return a compact contact-state label."""
    if left_contact and right_contact:
        return "both"
    if left_contact:
        return "left_only"
    if right_contact:
        return "right_only"
    return "none"


def _phase_support_side(state: RuntimeState) -> str:
    """Return the active support side for the current phase."""
    if state.phase in (Phase.ACQUIRE_LEFT_SUPPORT, Phase.LIFT_RIGHT, Phase.PLACE_RIGHT):
        return "left"
    if state.phase in (Phase.ACQUIRE_RIGHT_SUPPORT, Phase.LIFT_LEFT, Phase.PLACE_LEFT):
        return "right"
    return state.next_support_side


def _apply_support_roll(target: np.ndarray, support_side: str, magnitude: float) -> None:
    """Apply mirrored hip-roll offsets."""
    if support_side == "left":
        target[HIP_ROLL_INDEX["right"]] += magnitude
        target[HIP_ROLL_INDEX["left"]] -= magnitude
        return
    if support_side == "right":
        target[HIP_ROLL_INDEX["right"]] -= magnitude
        target[HIP_ROLL_INDEX["left"]] += magnitude
        return
    raise ValueError(f"Unsupported support side: {support_side}")


def _apply_swing_unload(
    target: np.ndarray,
    swing_side: str,
    *,
    hip_pitch_delta: float,
    knee_pitch_delta: float,
    ankle_pitch_delta: float,
    scale: float,
) -> None:
    """Apply swing-leg unload target."""
    target[HIP_PITCH_INDEX[swing_side]] += hip_pitch_delta * scale
    target[KNEE_PITCH_INDEX[swing_side]] += knee_pitch_delta * scale
    target[ANKLE_PITCH_INDEX[swing_side]] += ankle_pitch_delta * scale


def _apply_stance_brace(
    target: np.ndarray,
    support_side: str,
    *,
    knee_delta: float,
    ankle_delta: float,
) -> None:
    """Apply stance-leg stabilizer target."""
    target[KNEE_PITCH_INDEX[support_side]] += knee_delta
    target[ANKLE_PITCH_INDEX[support_side]] += ankle_delta


def _apply_swing_lift(
    target: np.ndarray,
    swing_side: str,
    *,
    hip_pitch_delta: float,
    knee_pitch_delta: float,
    ankle_pitch_delta: float,
    scale: float,
) -> None:
    """Apply swing-foot lift target."""
    target[HIP_PITCH_INDEX[swing_side]] += hip_pitch_delta * scale
    target[KNEE_PITCH_INDEX[swing_side]] += knee_pitch_delta * scale
    target[ANKLE_PITCH_INDEX[swing_side]] += ankle_pitch_delta * scale


def _safe_corrcoef(xs: list[float], ys: list[float]) -> float:
    """Return a stable Pearson correlation with zero fallback."""
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return 0.0
    xs_array = np.asarray(xs, dtype=np.float64)
    ys_array = np.asarray(ys, dtype=np.float64)
    if np.allclose(xs_array, xs_array[0]) or np.allclose(ys_array, ys_array[0]):
        return 0.0
    return float(np.corrcoef(xs_array, ys_array)[0, 1])


def _estimate_state(
    env: SeedonStandingEnv,
    runtime: RuntimeState,
    *,
    initial_left_foot_z: float,
    initial_right_foot_z: float,
) -> StateEstimate:
    """Estimate COM/contact/base state for one control step."""
    dt = float(env.dt)
    support_side = _phase_support_side(runtime)
    swing_side = SUPPORT_TO_SWING[support_side]
    com_y = float(_overall_com(env)[1])
    com_y_velocity = (com_y - runtime.previous_com_y) / max(dt, 1e-9)
    runtime.previous_com_y = com_y

    left_count, left_force = _foot_floor_load(env, "left")
    right_count, right_force = _foot_floor_load(env, "right")
    total_force = left_force + right_force
    if total_force > 1e-9:
        left_force_ratio = left_force / total_force
        right_force_ratio = right_force / total_force
    else:
        left_force_ratio = 0.0
        right_force_ratio = 0.0

    left_contact, right_contact, base_proxy_contact = _contact_flags(env)
    left_foot_geom_id = env._geom_id(LEFT_FOOT_GEOM)
    right_foot_geom_id = env._geom_id(RIGHT_FOOT_GEOM)
    foot_bottoms = env._foot_bottom_heights()
    left_foot_y = float(env.data.geom_xpos[left_foot_geom_id][1])
    right_foot_y = float(env.data.geom_xpos[right_foot_geom_id][1])
    center_y = 0.5 * (left_foot_y + right_foot_y)
    # Seedon's left foot is on negative world-Y, so support-side COM shift should
    # still be positive when COM moves toward the active support foot.
    signed_shift = center_y - com_y if support_side == "left" else com_y - center_y
    support_shift_velocity = (signed_shift - runtime.previous_support_com_shift) / max(dt, 1e-9)
    runtime.previous_support_com_shift = signed_shift

    base_z = env._base_height()
    upright = env._base_upright()
    observation = env._get_obs()
    terminated = env._is_terminated(base_z, upright, observation)

    return StateEstimate(
        com_y=com_y,
        com_y_velocity=com_y_velocity,
        base_roll=_quat_to_roll(env.data.xquat[env._base_body_id]),
        base_roll_velocity=float(env.data.qvel[3]),
        left_force_ratio=float(left_force_ratio),
        right_force_ratio=float(right_force_ratio),
        left_normal_force=float(left_force),
        right_normal_force=float(right_force),
        left_contact_count=left_count,
        right_contact_count=right_count,
        left_contact=left_contact,
        right_contact=right_contact,
        base_proxy_contact=base_proxy_contact,
        left_foot_y=left_foot_y,
        right_foot_y=right_foot_y,
        left_foot_z=float(foot_bottoms[1]),
        right_foot_z=float(foot_bottoms[0]),
        left_foot_z_delta=float(foot_bottoms[1] - initial_left_foot_z),
        right_foot_z_delta=float(foot_bottoms[0] - initial_right_foot_z),
        support_side=support_side,
        swing_side=swing_side,
        support_force_ratio=float(left_force_ratio if support_side == "left" else right_force_ratio),
        swing_force_ratio=float(right_force_ratio if support_side == "left" else left_force_ratio),
        support_com_shift=float(signed_shift),
        support_com_shift_velocity=float(support_shift_velocity),
        base_z=float(base_z),
        upright=float(upright),
        terminated=bool(terminated),
    )


def _needs_safe_recovery(estimate: StateEstimate, args: argparse.Namespace) -> bool:
    """Return whether the controller should fall back to safe stand."""
    return bool(
        estimate.base_proxy_contact
        or estimate.upright < args.recover_upright
        or abs(estimate.base_roll) > args.recover_roll
        or estimate.base_z < args.recover_height
    )


def _build_command(
    env: SeedonStandingEnv,
    runtime: RuntimeState,
    estimate: StateEstimate,
    args: argparse.Namespace,
    *,
    mode: str = "full_controller",
) -> ControllerCommand:
    """Build one closed-loop controller command."""
    support_side = estimate.support_side
    swing_side = estimate.swing_side
    target = env._nominal_joint_qpos.copy()

    if runtime.phase == Phase.DOUBLE_SUPPORT_STAND:
        return ControllerCommand(
            support_side=support_side,
            swing_side=swing_side,
            phase=runtime.phase.value,
            target_support_force_ratio=args.target_support_force_ratio,
            target_support_com_shift=args.target_support_com_shift,
            support_roll_cmd=0.0,
            unload_scale=0.0,
            swing_lift_scale=0.0,
            allow_lift=False,
            target_positions=env._apply_safe_joint_target_clamps(target),
        )

    support_force_error = args.target_support_force_ratio - estimate.support_force_ratio
    support_shift_error = args.target_support_com_shift - estimate.support_com_shift
    force_ratio_kp = args.force_ratio_kp
    com_shift_kp = args.com_shift_kp
    com_shift_kd = args.com_shift_kd
    roll_stabilizer_kp = args.roll_stabilizer_kp
    roll_stabilizer_kd = args.roll_stabilizer_kd
    unload_force_kp = args.unload_force_kp
    unload_com_kp = args.unload_com_kp

    if mode == "no_base_roll_stabilizer":
        roll_stabilizer_kp = 0.0
        roll_stabilizer_kd = 0.0
    elif mode == "no_com_feedback":
        com_shift_kp = 0.0
        com_shift_kd = 0.0
        unload_com_kp = 0.0
    elif mode == "no_force_ratio_feedback":
        force_ratio_kp = 0.0
        unload_force_kp = 0.0
    elif mode == "support_roll_only":
        unload_force_kp = 0.0
        unload_com_kp = 0.0

    support_roll_cmd = (
        force_ratio_kp * support_force_error
        + com_shift_kp * support_shift_error
        - com_shift_kd * estimate.support_com_shift_velocity
        - roll_stabilizer_kp * estimate.base_roll
        - roll_stabilizer_kd * estimate.base_roll_velocity
    )
    support_roll_cmd = float(np.clip(support_roll_cmd, 0.0, args.max_support_roll))

    unload_scale = (
        args.unload_bias
        + unload_force_kp * support_force_error
        + unload_com_kp * support_shift_error
    )
    unload_scale = float(np.clip(unload_scale, 0.0, args.max_unload_scale))
    allow_lift = (
        estimate.left_contact
        and estimate.right_contact
        and estimate.support_force_ratio >= args.lift_gate_force_ratio
        and estimate.support_com_shift >= args.min_lift_support_com_shift
    )

    swing_lift_scale = 0.0
    if runtime.phase in (Phase.LIFT_RIGHT, Phase.LIFT_LEFT):
        swing_lift_scale = float(
            np.clip(
                args.lift_kp * (args.target_swing_lift_z - (
                    estimate.right_foot_z_delta if swing_side == "right" else estimate.left_foot_z_delta
                )),
                0.0,
                1.0,
            )
        )
        if runtime.phase_step >= args.max_lift_steps - 1:
            swing_lift_scale = 1.0
    elif runtime.phase in (Phase.PLACE_RIGHT, Phase.PLACE_LEFT):
        remaining = max(args.max_place_steps - runtime.phase_step, 0)
        swing_lift_scale = remaining / max(args.max_place_steps, 1)

    _apply_support_roll(target, support_side, support_roll_cmd)
    if mode != "support_roll_only":
        _apply_stance_brace(
            target,
            support_side,
            knee_delta=args.stance_knee_brace,
            ankle_delta=args.stance_ankle_brace,
        )
        _apply_swing_unload(
            target,
            swing_side,
            hip_pitch_delta=args.swing_unload_hip_pitch,
            knee_pitch_delta=args.swing_unload_knee_pitch,
            ankle_pitch_delta=args.swing_unload_ankle_pitch,
            scale=unload_scale,
        )
        _apply_swing_lift(
            target,
            swing_side,
            hip_pitch_delta=args.swing_lift_hip_pitch,
            knee_pitch_delta=args.swing_lift_knee_pitch,
            ankle_pitch_delta=args.swing_lift_ankle_pitch,
            scale=swing_lift_scale,
        )
    else:
        unload_scale = 0.0
        swing_lift_scale = 0.0
        allow_lift = False

    return ControllerCommand(
        support_side=support_side,
        swing_side=swing_side,
        phase=runtime.phase.value,
        target_support_force_ratio=args.target_support_force_ratio,
        target_support_com_shift=args.target_support_com_shift,
        support_roll_cmd=support_roll_cmd,
        unload_scale=unload_scale,
        swing_lift_scale=swing_lift_scale,
        allow_lift=allow_lift,
        target_positions=env._apply_safe_joint_target_clamps(target),
    )


def _transition_phase(
    runtime: RuntimeState,
    estimate: StateEstimate,
    command: ControllerCommand,
    args: argparse.Namespace,
) -> None:
    """Advance the controller phase machine."""
    phase_steps = runtime.phase_step + 1

    if runtime.phase == Phase.DOUBLE_SUPPORT_STAND:
        if (
            phase_steps >= args.stand_hold_steps
            and estimate.left_contact
            and estimate.right_contact
            and not estimate.base_proxy_contact
        ):
            runtime.phase = (
                Phase.ACQUIRE_LEFT_SUPPORT
                if runtime.next_support_side == "left"
                else Phase.ACQUIRE_RIGHT_SUPPORT
            )
            runtime.phase_step = 0
            return
        runtime.phase_step = phase_steps
        return

    if runtime.phase == Phase.ACQUIRE_LEFT_SUPPORT:
        if command.allow_lift:
            runtime.entered_left_support = True
            runtime.phase = Phase.LIFT_RIGHT
            runtime.phase_step = 0
            return
        if phase_steps >= args.max_acquire_steps:
            runtime.phase = Phase.DOUBLE_SUPPORT_STAND
            runtime.next_support_side = "left"
            runtime.phase_step = 0
            return
        runtime.phase_step = phase_steps
        return

    if runtime.phase == Phase.ACQUIRE_RIGHT_SUPPORT:
        if command.allow_lift:
            runtime.entered_right_support = True
            runtime.phase = Phase.LIFT_LEFT
            runtime.phase_step = 0
            return
        if phase_steps >= args.max_acquire_steps:
            runtime.phase = Phase.DOUBLE_SUPPORT_STAND
            runtime.next_support_side = "right"
            runtime.phase_step = 0
            return
        runtime.phase_step = phase_steps
        return

    if runtime.phase == Phase.LIFT_RIGHT:
        if estimate.support_force_ratio < args.cancel_force_ratio or not estimate.left_contact:
            runtime.phase = Phase.DOUBLE_SUPPORT_STAND
            runtime.next_support_side = "left"
            runtime.phase_step = 0
            return
        if (
            estimate.right_foot_z_delta >= args.target_swing_lift_z
            or phase_steps >= args.max_lift_steps
        ):
            runtime.phase = Phase.PLACE_RIGHT
            runtime.phase_step = 0
            return
        runtime.phase_step = phase_steps
        return

    if runtime.phase == Phase.LIFT_LEFT:
        if estimate.support_force_ratio < args.cancel_force_ratio or not estimate.right_contact:
            runtime.phase = Phase.DOUBLE_SUPPORT_STAND
            runtime.next_support_side = "right"
            runtime.phase_step = 0
            return
        if (
            estimate.left_foot_z_delta >= args.target_swing_lift_z
            or phase_steps >= args.max_lift_steps
        ):
            runtime.phase = Phase.PLACE_LEFT
            runtime.phase_step = 0
            return
        runtime.phase_step = phase_steps
        return

    if runtime.phase == Phase.PLACE_RIGHT:
        if estimate.right_contact and phase_steps >= args.min_place_steps:
            runtime.phase = Phase.ACQUIRE_RIGHT_SUPPORT
            runtime.next_support_side = "right"
            runtime.phase_step = 0
            return
        if phase_steps >= args.max_place_steps:
            runtime.phase = Phase.DOUBLE_SUPPORT_STAND
            runtime.next_support_side = "right"
            runtime.phase_step = 0
            return
        runtime.phase_step = phase_steps
        return

    if runtime.phase == Phase.PLACE_LEFT:
        if estimate.left_contact and phase_steps >= args.min_place_steps:
            runtime.phase = Phase.ACQUIRE_LEFT_SUPPORT
            runtime.next_support_side = "left"
            runtime.phase_step = 0
            return
        if phase_steps >= args.max_place_steps:
            runtime.phase = Phase.DOUBLE_SUPPORT_STAND
            runtime.next_support_side = "left"
            runtime.phase_step = 0
            return
        runtime.phase_step = phase_steps
        return


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rollout rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_ablation_modes(raw_value: str) -> list[str]:
    """Parse and validate comma-separated ablation modes."""
    if not raw_value.strip():
        return []
    modes = [part.strip() for part in raw_value.split(",") if part.strip()]
    invalid = [mode for mode in modes if mode not in DEFAULT_ABLATION_MODES]
    if invalid:
        raise ValueError(
            f"Unsupported ablation mode(s): {', '.join(invalid)}. "
            f"Expected one of: {', '.join(DEFAULT_ABLATION_MODES)}"
        )
    return modes


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--render-viewer", action="store_true")
    parser.add_argument("--viewer-sleep", type=float, default=0.0)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--ablation-modes", default="")
    parser.add_argument("--ablation-side", choices=("left", "right"), default="left")
    parser.add_argument("--ablation-steps", type=int, default=160)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--stand-hold-steps", type=int, default=24)
    parser.add_argument("--max-acquire-steps", type=int, default=90)
    parser.add_argument("--max-lift-steps", type=int, default=30)
    parser.add_argument("--min-place-steps", type=int, default=8)
    parser.add_argument("--max-place-steps", type=int, default=30)
    parser.add_argument("--target-support-force-ratio", type=float, default=0.62)
    parser.add_argument("--lift-gate-force-ratio", type=float, default=0.65)
    parser.add_argument("--cancel-force-ratio", type=float, default=0.55)
    parser.add_argument("--target-support-com-shift", type=float, default=0.006)
    parser.add_argument("--min-lift-support-com-shift", type=float, default=0.004)
    parser.add_argument("--force-ratio-kp", type=float, default=0.18)
    parser.add_argument("--com-shift-kp", type=float, default=12.0)
    parser.add_argument("--com-shift-kd", type=float, default=2.0)
    parser.add_argument("--roll-stabilizer-kp", type=float, default=0.8)
    parser.add_argument("--roll-stabilizer-kd", type=float, default=0.12)
    parser.add_argument("--max-support-roll", type=float, default=0.14)
    parser.add_argument("--unload-bias", type=float, default=0.20)
    parser.add_argument("--unload-force-kp", type=float, default=2.0)
    parser.add_argument("--unload-com-kp", type=float, default=40.0)
    parser.add_argument("--max-unload-scale", type=float, default=1.0)
    parser.add_argument("--stance-knee-brace", type=float, default=0.04)
    parser.add_argument("--stance-ankle-brace", type=float, default=0.0)
    parser.add_argument("--swing-unload-hip-pitch", type=float, default=0.0)
    parser.add_argument("--swing-unload-knee-pitch", type=float, default=-0.06)
    parser.add_argument("--swing-unload-ankle-pitch", type=float, default=0.0)
    parser.add_argument("--swing-lift-hip-pitch", type=float, default=-0.015)
    parser.add_argument("--swing-lift-knee-pitch", type=float, default=-0.03)
    parser.add_argument("--swing-lift-ankle-pitch", type=float, default=0.015)
    parser.add_argument("--lift-kp", type=float, default=80.0)
    parser.add_argument("--target-swing-lift-z", type=float, default=0.010)
    parser.add_argument("--recover-upright", type=float, default=0.82)
    parser.add_argument("--recover-roll", type=float, default=0.30)
    parser.add_argument("--recover-height", type=float, default=0.36)
    return parser


def _summarize(
    runtime: RuntimeState,
    phase_counts: Counter[str],
    *,
    max_abs_com_y_delta: float,
    max_left_force_ratio: float,
    max_right_force_ratio: float,
    max_left_support_com_shift: float,
    max_right_support_com_shift: float,
    any_swing_lift: bool,
    terminated_step: int | None,
    max_support_roll_cmd: float,
    mean_support_roll_cmd_last_50: float,
    max_unload_scale: float,
    mean_unload_scale_last_50: float,
    mean_hip_roll_tracking_error_last_50: float,
    corr_support_roll_cmd_support_force_ratio: float,
    corr_support_roll_cmd_support_com_shift: float,
    corr_support_force_ratio_support_com_shift: float,
) -> PreviewSummary:
    """Build the rollout summary."""
    return PreviewSummary(
        entered_left_support=runtime.entered_left_support,
        entered_right_support=runtime.entered_right_support,
        phase_steps={phase.value: int(phase_counts[phase.value]) for phase in Phase},
        max_abs_com_y_delta=max_abs_com_y_delta,
        max_left_force_ratio=max_left_force_ratio,
        max_right_force_ratio=max_right_force_ratio,
        max_left_support_com_shift=max_left_support_com_shift,
        max_right_support_com_shift=max_right_support_com_shift,
        any_swing_lift=any_swing_lift,
        terminated_step=terminated_step,
        max_support_roll_cmd=max_support_roll_cmd,
        mean_support_roll_cmd_last_50=mean_support_roll_cmd_last_50,
        max_unload_scale=max_unload_scale,
        mean_unload_scale_last_50=mean_unload_scale_last_50,
        mean_hip_roll_tracking_error_last_50=mean_hip_roll_tracking_error_last_50,
        corr_support_roll_cmd_support_force_ratio=corr_support_roll_cmd_support_force_ratio,
        corr_support_roll_cmd_support_com_shift=corr_support_roll_cmd_support_com_shift,
        corr_support_force_ratio_support_com_shift=corr_support_force_ratio_support_com_shift,
    )


def _print_ablation_table(summaries: list[AblationSummary]) -> None:
    """Print a compact ablation comparison table."""
    print(
        "\nmode                         max_force mean_force50 max_shift mean_shift50 "
        "max_roll term both none"
    )
    for summary in summaries:
        terminated = "-" if summary.terminated_step is None else str(summary.terminated_step)
        print(
            f"{summary.mode:>26} "
            f"{summary.max_support_force_ratio:>9.3f} "
            f"{summary.mean_support_force_ratio_last_50:>12.3f} "
            f"{summary.max_support_com_shift:>9.4f} "
            f"{summary.mean_support_com_shift_last_50:>12.4f} "
            f"{summary.max_base_roll:>8.4f} "
            f"{terminated:>4} "
            f"{summary.both_contact_ratio:>4.2f} "
            f"{summary.none_contact_ratio:>4.2f}"
        )


def _print_timeseries_header() -> None:
    """Print the time-series diagnostic header."""
    print(
        "step mode phase support_roll_cmd unload_scale swing_lift "
        "target_l_hr target_r_hr qpos_l_hr qpos_r_hr err_l_hr err_r_hr "
        "support_ratio support_shift base_roll base_roll_vel contact_state"
    )


def _print_timeseries_row(row: dict[str, object]) -> None:
    """Print one compact time-series diagnostic row."""
    print(
        f"{int(row['step']):>4} "
        f"{str(row['mode']):>22} "
        f"{str(row['phase']):>24} "
        f"{float(row['support_roll_cmd']):>16.4f} "
        f"{float(row['unload_scale']):>12.4f} "
        f"{float(row['swing_lift_scale']):>10.4f} "
        f"{float(row['target_left_hip_roll']):>11.4f} "
        f"{float(row['target_right_hip_roll']):>11.4f} "
        f"{float(row['left_hip_roll_qpos']):>9.4f} "
        f"{float(row['right_hip_roll_qpos']):>9.4f} "
        f"{float(row['left_hip_roll_error']):>8.4f} "
        f"{float(row['right_hip_roll_error']):>8.4f} "
        f"{float(row['support_force_ratio']):>13.3f} "
        f"{float(row['support_com_shift']):>13.4f} "
        f"{float(row['base_roll']):>9.4f} "
        f"{float(row['base_roll_velocity']):>13.4f} "
        f"{str(row['contact_state']):>12}"
    )


def _run_ablation_mode(
    args: argparse.Namespace,
    *,
    mode: str,
    side: str,
) -> tuple[list[dict[str, object]], AblationSummary]:
    """Run one fixed acquire-support ablation rollout."""
    reward_config = load_seedon_config_from_env()
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=reward_config)
    rows: list[dict[str, object]] = []
    runtime = RuntimeState(phase=Phase.ACQUIRE_LEFT_SUPPORT if side == "left" else Phase.ACQUIRE_RIGHT_SUPPORT)
    max_support_force_ratio = 0.0
    max_support_com_shift = 0.0
    max_base_roll = 0.0
    max_support_roll_cmd = 0.0
    max_unload_scale = 0.0
    both_contact_steps = 0
    none_contact_steps = 0
    support_force_ratios: list[float] = []
    support_com_shifts: list[float] = []
    support_roll_cmds: list[float] = []
    unload_scales: list[float] = []
    hip_roll_tracking_errors: list[float] = []
    terminated_step: int | None = None

    try:
        env.reset(seed=args.seed)
        nominal_target = env._apply_safe_joint_target_clamps(env._nominal_joint_qpos.copy())
        for _ in range(args.settle_steps):
            env._do_pd_simulation(nominal_target)

        foot_bottoms = env._foot_bottom_heights()
        initial_left_foot_z = float(foot_bottoms[1])
        initial_right_foot_z = float(foot_bottoms[0])
        runtime.previous_com_y = float(_overall_com(env)[1])
        initial_center_y = 0.5 * (
            float(env.data.geom_xpos[env._geom_id(LEFT_FOOT_GEOM)][1])
            + float(env.data.geom_xpos[env._geom_id(RIGHT_FOOT_GEOM)][1])
        )
        runtime.previous_support_com_shift = (
            initial_center_y - runtime.previous_com_y
            if side == "left"
            else runtime.previous_com_y - initial_center_y
        )

        print(f"\n[ablation] mode={mode} side={side} steps={args.ablation_steps}")
        _print_timeseries_header()
        for step in range(1, args.ablation_steps + 1):
            estimate_before = _estimate_state(
                env,
                runtime,
                initial_left_foot_z=initial_left_foot_z,
                initial_right_foot_z=initial_right_foot_z,
            )
            command = _build_command(env, runtime, estimate_before, args, mode=mode)
            env._do_pd_simulation(command.target_positions)
            estimate_after = _estimate_state(
                env,
                runtime,
                initial_left_foot_z=initial_left_foot_z,
                initial_right_foot_z=initial_right_foot_z,
            )
            joint_positions = env._joint_positions()
            left_hip_roll_qpos = float(joint_positions[HIP_ROLL_INDEX["left"]])
            right_hip_roll_qpos = float(joint_positions[HIP_ROLL_INDEX["right"]])
            target_left_hip_roll = float(command.target_positions[HIP_ROLL_INDEX["left"]])
            target_right_hip_roll = float(command.target_positions[HIP_ROLL_INDEX["right"]])
            left_hip_roll_error = target_left_hip_roll - left_hip_roll_qpos
            right_hip_roll_error = target_right_hip_roll - right_hip_roll_qpos
            contact_state = _contact_state_label(estimate_after.left_contact, estimate_after.right_contact)

            support_force_ratios.append(estimate_after.support_force_ratio)
            support_com_shifts.append(estimate_after.support_com_shift)
            support_roll_cmds.append(command.support_roll_cmd)
            unload_scales.append(command.unload_scale)
            hip_roll_tracking_errors.append(
                0.5 * (abs(left_hip_roll_error) + abs(right_hip_roll_error))
            )
            max_support_force_ratio = max(max_support_force_ratio, estimate_after.support_force_ratio)
            max_support_com_shift = max(max_support_com_shift, estimate_after.support_com_shift)
            max_base_roll = max(max_base_roll, abs(estimate_after.base_roll))
            max_support_roll_cmd = max(max_support_roll_cmd, abs(command.support_roll_cmd))
            max_unload_scale = max(max_unload_scale, command.unload_scale)
            if estimate_after.left_contact and estimate_after.right_contact:
                both_contact_steps += 1
            elif not estimate_after.left_contact and not estimate_after.right_contact:
                none_contact_steps += 1

            rows.append(
                {
                    "mode": mode,
                    "side": side,
                    "step": step,
                    "phase": runtime.phase.value,
                    "support_side": command.support_side,
                    "com_y": estimate_after.com_y,
                    "com_y_velocity": estimate_after.com_y_velocity,
                    "base_roll": estimate_after.base_roll,
                    "base_roll_velocity": estimate_after.base_roll_velocity,
                    "left_normal_force": estimate_after.left_normal_force,
                    "right_normal_force": estimate_after.right_normal_force,
                    "left_force_ratio": estimate_after.left_force_ratio,
                    "right_force_ratio": estimate_after.right_force_ratio,
                    "support_force_ratio": estimate_after.support_force_ratio,
                    "swing_force_ratio": estimate_after.swing_force_ratio,
                    "support_com_shift": estimate_after.support_com_shift,
                    "support_com_shift_velocity": estimate_after.support_com_shift_velocity,
                    "left_contact": estimate_after.left_contact,
                    "right_contact": estimate_after.right_contact,
                    "left_contact_count": estimate_after.left_contact_count,
                    "right_contact_count": estimate_after.right_contact_count,
                    "base_proxy_contact": estimate_after.base_proxy_contact,
                    "support_roll_cmd": command.support_roll_cmd,
                    "unload_scale": command.unload_scale,
                    "swing_lift_scale": command.swing_lift_scale,
                    "target_left_hip_roll": target_left_hip_roll,
                    "target_right_hip_roll": target_right_hip_roll,
                    "left_hip_roll_qpos": left_hip_roll_qpos,
                    "right_hip_roll_qpos": right_hip_roll_qpos,
                    "left_hip_roll_error": left_hip_roll_error,
                    "right_hip_roll_error": right_hip_roll_error,
                    "contact_state": contact_state,
                    "terminated": estimate_after.terminated,
                }
            )

            if step % 10 == 0 or estimate_after.terminated:
                _print_timeseries_row(rows[-1])

            if estimate_after.terminated or _needs_safe_recovery(estimate_after, args):
                terminated_step = step
                break
            runtime.phase_step += 1
    finally:
        env.close()

    force_tail = support_force_ratios[-50:]
    shift_tail = support_com_shifts[-50:]
    roll_cmd_tail = support_roll_cmds[-50:]
    unload_tail = unload_scales[-50:]
    tracking_tail = hip_roll_tracking_errors[-50:]
    summary = AblationSummary(
        mode=mode,
        side=side,
        max_support_force_ratio=max_support_force_ratio,
        mean_support_force_ratio_last_50=float(np.mean(force_tail)) if force_tail else 0.0,
        max_support_com_shift=max_support_com_shift,
        mean_support_com_shift_last_50=float(np.mean(shift_tail)) if shift_tail else 0.0,
        max_base_roll=max_base_roll,
        terminated_step=terminated_step,
        both_contact_ratio=both_contact_steps / max(args.ablation_steps, 1),
        none_contact_ratio=none_contact_steps / max(args.ablation_steps, 1),
        max_support_roll_cmd=max_support_roll_cmd,
        mean_support_roll_cmd_last_50=float(np.mean(roll_cmd_tail)) if roll_cmd_tail else 0.0,
        max_unload_scale=max_unload_scale,
        mean_unload_scale_last_50=float(np.mean(unload_tail)) if unload_tail else 0.0,
        mean_hip_roll_tracking_error_last_50=float(np.mean(tracking_tail)) if tracking_tail else 0.0,
        corr_support_roll_cmd_support_force_ratio=_safe_corrcoef(support_roll_cmds, support_force_ratios),
        corr_support_roll_cmd_support_com_shift=_safe_corrcoef(support_roll_cmds, support_com_shifts),
        corr_support_force_ratio_support_com_shift=_safe_corrcoef(support_force_ratios, support_com_shifts),
    )
    return rows, summary


def _run_ablation_suite(args: argparse.Namespace, modes: list[str]) -> int:
    """Run acquire-support ablations and print a comparison summary."""
    all_rows: list[dict[str, object]] = []
    summaries: list[AblationSummary] = []
    for mode in modes:
        rows, summary = _run_ablation_mode(args, mode=mode, side=args.ablation_side)
        all_rows.extend(rows)
        summaries.append(summary)

    _write_rows(args.out_csv, all_rows)
    _print_ablation_table(summaries)
    print("\nmode correlations/actuation")
    for summary in summaries:
        terminated = "-" if summary.terminated_step is None else str(summary.terminated_step)
        print(
            f"{summary.mode:>26} "
            f"max_roll_cmd={summary.max_support_roll_cmd:.4f} "
            f"mean_roll_cmd50={summary.mean_support_roll_cmd_last_50:.4f} "
            f"max_unload={summary.max_unload_scale:.4f} "
            f"mean_unload50={summary.mean_unload_scale_last_50:.4f} "
            f"mean_track_err50={summary.mean_hip_roll_tracking_error_last_50:.4f} "
            f"corr(cmd,force)={summary.corr_support_roll_cmd_support_force_ratio:.3f} "
            f"corr(cmd,com)={summary.corr_support_roll_cmd_support_com_shift:.3f} "
            f"corr(force,com)={summary.corr_support_force_ratio_support_com_shift:.3f} "
            f"term={terminated}"
        )
    print(f"\ncsv: {args.out_csv}")
    print(f"ablation_side: {args.ablation_side}")
    print(f"ablation_steps: {args.ablation_steps}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the Blue-like closed-loop balance controller preview."""
    args = build_parser().parse_args(argv)
    ablation_modes = _parse_ablation_modes(args.ablation_modes)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.ablation_steps <= 0:
        raise ValueError("--ablation-steps must be positive.")
    if args.viewer_sleep < 0.0:
        raise ValueError("--viewer-sleep must be non-negative.")
    if args.print_every <= 0:
        raise ValueError("--print-every must be positive.")
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative.")
    if ablation_modes and args.render_viewer:
        raise ValueError("--render-viewer is not supported in ablation mode.")
    if ablation_modes:
        return _run_ablation_suite(args, ablation_modes)

    reward_config = load_seedon_config_from_env()
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=reward_config)
    rows: list[dict[str, object]] = []
    phase_counts: Counter[str] = Counter()
    runtime = RuntimeState()
    terminated_step: int | None = None
    max_abs_com_y_delta = 0.0
    max_left_force_ratio = 0.0
    max_right_force_ratio = 0.0
    max_left_support_com_shift = 0.0
    max_right_support_com_shift = 0.0
    any_swing_lift = False
    max_support_roll_cmd = 0.0
    max_unload_scale = 0.0
    support_roll_cmds: list[float] = []
    unload_scales: list[float] = []
    support_force_ratios: list[float] = []
    support_com_shifts: list[float] = []
    hip_roll_tracking_errors: list[float] = []

    try:
        env.reset(seed=args.seed)
        nominal_target = env._apply_safe_joint_target_clamps(env._nominal_joint_qpos.copy())
        for _ in range(args.settle_steps):
            env._do_pd_simulation(nominal_target)

        initial_com_y = float(_overall_com(env)[1])
        foot_bottoms = env._foot_bottom_heights()
        initial_left_foot_z = float(foot_bottoms[1])
        initial_right_foot_z = float(foot_bottoms[0])
        runtime.previous_com_y = initial_com_y

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

        _print_timeseries_header()
        try:
            for step in range(1, args.steps + 1):
                estimate_before = _estimate_state(
                    env,
                    runtime,
                    initial_left_foot_z=initial_left_foot_z,
                    initial_right_foot_z=initial_right_foot_z,
                )
                if _needs_safe_recovery(estimate_before, args):
                    runtime.phase = Phase.DOUBLE_SUPPORT_STAND
                    runtime.phase_step = 0
                    runtime.next_support_side = estimate_before.support_side

                active_phase = runtime.phase
                command = _build_command(env, runtime, estimate_before, args)
                env._do_pd_simulation(command.target_positions)
                estimate_after = _estimate_state(
                    env,
                    runtime,
                    initial_left_foot_z=initial_left_foot_z,
                    initial_right_foot_z=initial_right_foot_z,
                )
                joint_positions = env._joint_positions()
                left_hip_roll_qpos = float(joint_positions[HIP_ROLL_INDEX["left"]])
                right_hip_roll_qpos = float(joint_positions[HIP_ROLL_INDEX["right"]])
                target_left_hip_roll = float(command.target_positions[HIP_ROLL_INDEX["left"]])
                target_right_hip_roll = float(command.target_positions[HIP_ROLL_INDEX["right"]])
                left_hip_roll_error = target_left_hip_roll - left_hip_roll_qpos
                right_hip_roll_error = target_right_hip_roll - right_hip_roll_qpos
                contact_state = _contact_state_label(estimate_after.left_contact, estimate_after.right_contact)

                phase_counts[active_phase.value] += 1
                max_abs_com_y_delta = max(max_abs_com_y_delta, abs(estimate_after.com_y - initial_com_y))
                max_left_force_ratio = max(max_left_force_ratio, estimate_after.left_force_ratio)
                max_right_force_ratio = max(max_right_force_ratio, estimate_after.right_force_ratio)
                max_support_roll_cmd = max(max_support_roll_cmd, abs(command.support_roll_cmd))
                max_unload_scale = max(max_unload_scale, command.unload_scale)
                support_roll_cmds.append(command.support_roll_cmd)
                unload_scales.append(command.unload_scale)
                support_force_ratios.append(estimate_after.support_force_ratio)
                support_com_shifts.append(estimate_after.support_com_shift)
                hip_roll_tracking_errors.append(
                    0.5 * (abs(left_hip_roll_error) + abs(right_hip_roll_error))
                )
                if estimate_after.support_side == "left":
                    max_left_support_com_shift = max(max_left_support_com_shift, estimate_after.support_com_shift)
                else:
                    max_right_support_com_shift = max(max_right_support_com_shift, estimate_after.support_com_shift)
                any_swing_lift = any_swing_lift or max(
                    estimate_after.left_foot_z_delta,
                    estimate_after.right_foot_z_delta,
                ) > 0.005

                row = {
                    "mode": "full_controller",
                    "step": step,
                    "phase": active_phase.value,
                    "support_side": command.support_side,
                    "swing_side": command.swing_side,
                    "com_y": estimate_after.com_y,
                    "com_y_velocity": estimate_after.com_y_velocity,
                    "base_roll": estimate_after.base_roll,
                    "base_roll_velocity": estimate_after.base_roll_velocity,
                    "left_normal_force": estimate_after.left_normal_force,
                    "right_normal_force": estimate_after.right_normal_force,
                    "left_force_ratio": estimate_after.left_force_ratio,
                    "right_force_ratio": estimate_after.right_force_ratio,
                    "support_force_ratio": estimate_after.support_force_ratio,
                    "swing_force_ratio": estimate_after.swing_force_ratio,
                    "support_com_shift": estimate_after.support_com_shift,
                    "support_com_shift_velocity": estimate_after.support_com_shift_velocity,
                    "left_contact_count": estimate_after.left_contact_count,
                    "right_contact_count": estimate_after.right_contact_count,
                    "left_contact": estimate_after.left_contact,
                    "right_contact": estimate_after.right_contact,
                    "base_proxy_contact": estimate_after.base_proxy_contact,
                    "allow_lift": command.allow_lift,
                    "support_roll_cmd": command.support_roll_cmd,
                    "unload_scale": command.unload_scale,
                    "swing_lift_scale": command.swing_lift_scale,
                    "target_support_force_ratio": command.target_support_force_ratio,
                    "target_support_com_shift": command.target_support_com_shift,
                    "target_left_hip_roll": target_left_hip_roll,
                    "target_right_hip_roll": target_right_hip_roll,
                    "target_left_hip_pitch": float(command.target_positions[HIP_PITCH_INDEX["left"]]),
                    "target_right_hip_pitch": float(command.target_positions[HIP_PITCH_INDEX["right"]]),
                    "target_left_knee_pitch": float(command.target_positions[KNEE_PITCH_INDEX["left"]]),
                    "target_right_knee_pitch": float(command.target_positions[KNEE_PITCH_INDEX["right"]]),
                    "target_left_ankle_pitch": float(command.target_positions[ANKLE_PITCH_INDEX["left"]]),
                    "target_right_ankle_pitch": float(command.target_positions[ANKLE_PITCH_INDEX["right"]]),
                    "left_hip_roll_qpos": left_hip_roll_qpos,
                    "right_hip_roll_qpos": right_hip_roll_qpos,
                    "left_hip_roll_error": left_hip_roll_error,
                    "right_hip_roll_error": right_hip_roll_error,
                    "contact_state": contact_state,
                    "left_foot_z_delta": estimate_after.left_foot_z_delta,
                    "right_foot_z_delta": estimate_after.right_foot_z_delta,
                    "terminated": estimate_after.terminated,
                }
                rows.append(row)

                if step % 10 == 0 or estimate_after.terminated:
                    _print_timeseries_row(
                        {
                            "step": step,
                            "mode": "full_controller",
                            "phase": active_phase.value,
                            "support_roll_cmd": command.support_roll_cmd,
                            "unload_scale": command.unload_scale,
                            "swing_lift_scale": command.swing_lift_scale,
                            "target_left_hip_roll": target_left_hip_roll,
                            "target_right_hip_roll": target_right_hip_roll,
                            "left_hip_roll_qpos": left_hip_roll_qpos,
                            "right_hip_roll_qpos": right_hip_roll_qpos,
                            "left_hip_roll_error": left_hip_roll_error,
                            "right_hip_roll_error": right_hip_roll_error,
                            "support_force_ratio": estimate_after.support_force_ratio,
                            "support_com_shift": estimate_after.support_com_shift,
                            "base_roll": estimate_after.base_roll,
                            "base_roll_velocity": estimate_after.base_roll_velocity,
                            "contact_state": contact_state,
                        }
                    )

                if viewer is not None:
                    viewer.sync()
                    if args.viewer_sleep > 0.0:
                        time.sleep(args.viewer_sleep)
                    if not viewer.is_running():
                        break

                if estimate_after.terminated:
                    terminated_step = step
                    break

                if _needs_safe_recovery(estimate_after, args):
                    runtime.phase = Phase.DOUBLE_SUPPORT_STAND
                    runtime.phase_step = 0
                    runtime.next_support_side = estimate_after.support_side
                else:
                    _transition_phase(runtime, estimate_after, command, args)
        finally:
            if viewer is not None:
                viewer.__exit__(None, None, None)
    finally:
        env.close()

    _write_rows(args.out_csv, rows)
    roll_cmd_tail = support_roll_cmds[-50:]
    unload_tail = unload_scales[-50:]
    tracking_tail = hip_roll_tracking_errors[-50:]
    summary = _summarize(
        runtime,
        phase_counts,
        max_abs_com_y_delta=max_abs_com_y_delta,
        max_left_force_ratio=max_left_force_ratio,
        max_right_force_ratio=max_right_force_ratio,
        max_left_support_com_shift=max_left_support_com_shift,
        max_right_support_com_shift=max_right_support_com_shift,
        any_swing_lift=any_swing_lift,
        terminated_step=terminated_step,
        max_support_roll_cmd=max_support_roll_cmd,
        mean_support_roll_cmd_last_50=float(np.mean(roll_cmd_tail)) if roll_cmd_tail else 0.0,
        max_unload_scale=max_unload_scale,
        mean_unload_scale_last_50=float(np.mean(unload_tail)) if unload_tail else 0.0,
        mean_hip_roll_tracking_error_last_50=float(np.mean(tracking_tail)) if tracking_tail else 0.0,
        corr_support_roll_cmd_support_force_ratio=_safe_corrcoef(support_roll_cmds, support_force_ratios),
        corr_support_roll_cmd_support_com_shift=_safe_corrcoef(support_roll_cmds, support_com_shifts),
        corr_support_force_ratio_support_com_shift=_safe_corrcoef(support_force_ratios, support_com_shifts),
    )

    print(f"\ncsv: {args.out_csv}")
    print(f"steps: {len(rows)}")
    print(f"entered_left_support: {summary.entered_left_support}")
    print(f"entered_right_support: {summary.entered_right_support}")
    print(f"any_swing_lift_gt_0.005m: {summary.any_swing_lift}")
    print(f"max_abs_com_y_delta: {summary.max_abs_com_y_delta:.5f}")
    print(f"max_left_force_ratio: {summary.max_left_force_ratio:.3f}")
    print(f"max_right_force_ratio: {summary.max_right_force_ratio:.3f}")
    print(f"max_left_support_com_shift: {summary.max_left_support_com_shift:.5f}")
    print(f"max_right_support_com_shift: {summary.max_right_support_com_shift:.5f}")
    print(f"max_support_roll_cmd: {summary.max_support_roll_cmd:.5f}")
    print(f"mean_support_roll_cmd_last_50: {summary.mean_support_roll_cmd_last_50:.5f}")
    print(f"max_unload_scale: {summary.max_unload_scale:.5f}")
    print(f"mean_unload_scale_last_50: {summary.mean_unload_scale_last_50:.5f}")
    print(
        "mean_hip_roll_tracking_error_last_50: "
        f"{summary.mean_hip_roll_tracking_error_last_50:.5f}"
    )
    print(
        "corr(support_roll_cmd,support_force_ratio): "
        f"{summary.corr_support_roll_cmd_support_force_ratio:.3f}"
    )
    print(
        "corr(support_roll_cmd,support_com_shift): "
        f"{summary.corr_support_roll_cmd_support_com_shift:.3f}"
    )
    print(
        "corr(support_force_ratio,support_com_shift): "
        f"{summary.corr_support_force_ratio_support_com_shift:.3f}"
    )
    print(f"terminated_step: {summary.terminated_step}")
    for phase_name, count in summary.phase_steps.items():
        print(f"{phase_name}_steps: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
