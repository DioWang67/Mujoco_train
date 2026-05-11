"""
Preview Sedon hybrid torque balance controller.

Purpose:
    Prototype a Blue-like balance controller without changing MJCF / training scene /
    PPO / reward.

Concept:
    - Pose target keeps conservative standing posture.
    - Extra torque assist is injected into hip_roll actuators.
    - Main objective is support foot force ratio, not fixed hip_roll angle.

Usage:
    .\\.venv\\Scripts\\python.exe -m tools.preview_sedon_hybrid_torque_balance_controller --steps 320 --print-every 20

Conservative test:
    .\\.venv\\Scripts\\python.exe -m tools.preview_sedon_hybrid_torque_balance_controller --steps 320 --print-every 20 --target-force-ratio 0.58 --max-tau-assist 2.0 --kp-force-tau 4.0
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from sedon_baseline.env import (
    DEFAULT_SCENE_PATH,
    JOINT_NAMES,
    SedonStandingEnv,
)


ARTIFACT_DIR = Path("artifacts/sedon_debug")
DEFAULT_CSV = ARTIFACT_DIR / "hybrid_torque_balance_controller_preview.csv"


class Phase(str, Enum):
    DOUBLE_SUPPORT = "double_support"
    ACQUIRE_LEFT_SUPPORT = "acquire_left_support"
    LIFT_RIGHT = "lift_right"
    PLACE_RIGHT = "place_right"
    ACQUIRE_RIGHT_SUPPORT = "acquire_right_support"
    LIFT_LEFT = "lift_left"
    PLACE_LEFT = "place_left"
    SAFE_STAND = "safe_stand"


@dataclass(frozen=True)
class JointNames:
    left_hip_roll: str = "L_joint_hip_roll"
    right_hip_roll: str = "R_joint_hip_roll"
    left_hip_pitch: str = "L_joint_hip_pitch"
    right_hip_pitch: str = "R_joint_hip_pitch"
    left_knee_pitch: str = "L_joint_knee_pitch"
    right_knee_pitch: str = "R_joint_knee_pitch"
    left_ankle_pitch: str = "L_joint_ankle_pitch"
    right_ankle_pitch: str = "R_joint_ankle_pitch"


@dataclass
class ControllerConfig:
    target_support_force_ratio: float = 0.58
    release_support_force_ratio: float = 0.53
    support_hold_required_steps: int = 15

    # Pose target command.
    support_roll_target: float = 0.06
    swing_roll_unload_scale: float = 0.45
    support_knee_brace: float = 0.04
    support_ankle_brace: float = -0.02
    swing_unload_hip_pitch: float = 0.02
    swing_unload_knee: float = -0.05

    # Small lift only after stable support acquisition.
    swing_lift_hip_pitch: float = 0.06
    swing_lift_knee: float = -0.10
    swing_lift_ankle: float = 0.03

    # Torque assist controller.
    kp_force_tau: float = 4.0
    kd_force_tau: float = 0.6
    kp_com_tau: float = 0.6
    kd_com_tau: float = 0.1
    kp_base_roll_tau: float = 0.5
    kd_base_roll_tau: float = 0.08
    max_tau_assist: float = 2.0
    max_tau_delta_per_step: float = 0.15

    # Low-level PD used by this preview tool.
    pd_stiffness: float = 35.0
    pd_damping: float = 2.0
    torque_scale: float = 45.0

    # Timing.
    double_support_steps: int = 30
    acquire_max_steps: int = 180
    lift_steps: int = 40
    place_steps: int = 40
    physics_steps_per_control: int = 5

    # Safety.
    max_abs_base_roll: float = 0.35
    min_base_z: float = 0.18
    max_none_contact_steps: int = 3


@dataclass
class ForceState:
    left_force: float = 0.0
    right_force: float = 0.0
    left_contact_count: int = 0
    right_contact_count: int = 0

    @property
    def total_force(self) -> float:
        return self.left_force + self.right_force

    @property
    def left_ratio(self) -> float:
        if self.total_force <= 1e-9:
            return 0.0
        return self.left_force / self.total_force

    @property
    def right_ratio(self) -> float:
        if self.total_force <= 1e-9:
            return 0.0
        return self.right_force / self.total_force

    @property
    def contact_state(self) -> str:
        left = self.left_contact_count > 0
        right = self.right_contact_count > 0

        if left and right:
            return "both"
        if left:
            return "left_only"
        if right:
            return "right_only"
        return "none"


@dataclass
class RobotState:
    step: int
    phase: Phase
    com_y: float
    com_y_vel: float
    base_z: float
    base_roll: float
    base_roll_vel: float
    forces: ForceState


@dataclass
class ControllerMemory:
    phase: Phase = Phase.DOUBLE_SUPPORT
    phase_step: int = 0
    support_side: str = "left"

    previous_com_y: Optional[float] = None
    previous_base_roll: Optional[float] = None
    previous_force_error: float = 0.0

    left_tau_assist: float = 0.0
    right_tau_assist: float = 0.0
    support_hold_counter: int = 0
    none_contact_steps: int = 0

    entered_left_support: bool = False
    entered_right_support: bool = False
    entered_lift: bool = False
    terminated_step: Optional[int] = None


@dataclass
class StepCommand:
    target_positions: np.ndarray
    target_by_name: Dict[str, float]
    left_tau_assist: float
    right_tau_assist: float
    force_ratio_error: float
    allow_lift: bool


@dataclass
class StepLog:
    row: Dict[str, object]


def _safe_get_model_data(env: SedonStandingEnv):
    if hasattr(env, "model") and hasattr(env, "data"):
        return env.model, env.data

    if hasattr(env, "unwrapped"):
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "model") and hasattr(unwrapped, "data"):
            return unwrapped.model, unwrapped.data

    if hasattr(env, "env"):
        inner = env.env
        if hasattr(inner, "model") and hasattr(inner, "data"):
            return inner.model, inner.data

    raise RuntimeError("Cannot locate MuJoCo model/data from SedonStandingEnv.")


def _body_name(model, body_id: int) -> str:
    try:
        return model.body(body_id).name
    except Exception:
        return ""


def _geom_name(model, geom_id: int) -> str:
    try:
        return model.geom(geom_id).name
    except Exception:
        return ""


def _is_left_foot_contact(model, contact) -> bool:
    geom1 = _geom_name(model, contact.geom1).lower()
    geom2 = _geom_name(model, contact.geom2).lower()
    body1 = _body_name(model, int(model.geom_bodyid[contact.geom1])).lower()
    body2 = _body_name(model, int(model.geom_bodyid[contact.geom2])).lower()
    text = f"{geom1} {geom2} {body1} {body2}"
    return "left" in text or "l_" in text or "l-" in text


def _is_right_foot_contact(model, contact) -> bool:
    geom1 = _geom_name(model, contact.geom1).lower()
    geom2 = _geom_name(model, contact.geom2).lower()
    body1 = _body_name(model, int(model.geom_bodyid[contact.geom1])).lower()
    body2 = _body_name(model, int(model.geom_bodyid[contact.geom2])).lower()
    text = f"{geom1} {geom2} {body1} {body2}"
    return "right" in text or "r_" in text or "r-" in text


def _contact_normal_force(data, contact_id: int) -> float:
    contact = data.contact[contact_id]
    efc_addr = int(contact.efc_address)

    if efc_addr < 0:
        return 0.0

    try:
        return abs(float(data.efc_force[efc_addr]))
    except Exception:
        return 0.0


def read_force_state(model, data) -> ForceState:
    left_force = 0.0
    right_force = 0.0
    left_count = 0
    right_count = 0

    for contact_id in range(int(data.ncon)):
        contact = data.contact[contact_id]

        is_left = _is_left_foot_contact(model, contact)
        is_right = _is_right_foot_contact(model, contact)

        if not is_left and not is_right:
            continue

        normal_force = _contact_normal_force(data, contact_id)

        if is_left:
            left_force += normal_force
            left_count += 1

        if is_right:
            right_force += normal_force
            right_count += 1

    return ForceState(
        left_force=left_force,
        right_force=right_force,
        left_contact_count=left_count,
        right_contact_count=right_count,
    )


def get_base_roll_from_quat(qw: float, qx: float, qy: float, qz: float) -> float:
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.atan2(sinr_cosp, cosr_cosp)


def read_robot_state(env: SedonStandingEnv, memory: ControllerMemory, step: int) -> RobotState:
    model, data = _safe_get_model_data(env)

    base_z = float(data.qpos[2])
    qw, qx, qy, qz = [float(x) for x in data.qpos[3:7]]
    base_roll = get_base_roll_from_quat(qw, qx, qy, qz)

    com_y = float(data.subtree_com[0][1]) if hasattr(data, "subtree_com") else float(data.qpos[1])

    com_y_vel = 0.0 if memory.previous_com_y is None else com_y - memory.previous_com_y
    base_roll_vel = 0.0 if memory.previous_base_roll is None else base_roll - memory.previous_base_roll

    return RobotState(
        step=step,
        phase=memory.phase,
        com_y=com_y,
        com_y_vel=com_y_vel,
        base_z=base_z,
        base_roll=base_roll,
        base_roll_vel=base_roll_vel,
        forces=read_force_state(model, data),
    )


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rate_limit(value: float, previous: float, max_delta: float) -> float:
    delta = clamp(value - previous, -max_delta, max_delta)
    return previous + delta


def support_ratio(state: RobotState, side: str) -> float:
    return state.forces.left_ratio if side == "left" else state.forces.right_ratio


def swing_ratio(state: RobotState, side: str) -> float:
    return state.forces.right_ratio if side == "left" else state.forces.left_ratio


def support_sign(side: str) -> float:
    return 1.0 if side == "left" else -1.0


def build_base_target_positions(env: SedonStandingEnv) -> np.ndarray:
    if not hasattr(env, "_nominal_joint_qpos"):
        raise RuntimeError("SedonStandingEnv must expose _nominal_joint_qpos for this preview tool.")

    return env._nominal_joint_qpos.copy()


def set_named_target(target_positions: np.ndarray, joint_name: str, value: float) -> None:
    try:
        joint_index = JOINT_NAMES.index(joint_name)
    except ValueError:
        return

    target_positions[joint_index] = float(value)


def get_named_target(target_positions: np.ndarray, joint_name: str) -> float:
    try:
        joint_index = JOINT_NAMES.index(joint_name)
    except ValueError:
        return 0.0

    return float(target_positions[joint_index])


def apply_safe_clamps(env: SedonStandingEnv, target_positions: np.ndarray) -> np.ndarray:
    if hasattr(env, "_apply_safe_joint_target_clamps"):
        return env._apply_safe_joint_target_clamps(target_positions)
    return target_positions


def build_pose_target(
    env: SedonStandingEnv,
    side: str,
    cfg: ControllerConfig,
    phase: Phase,
    phase_step: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    j = JointNames()
    target = build_base_target_positions(env)

    # Conservative neutral brace.
    set_named_target(target, j.left_hip_roll, 0.0)
    set_named_target(target, j.right_hip_roll, 0.0)
    set_named_target(target, j.left_hip_pitch, 0.0)
    set_named_target(target, j.right_hip_pitch, 0.0)
    set_named_target(target, j.left_knee_pitch, 0.04)
    set_named_target(target, j.right_knee_pitch, 0.04)
    set_named_target(target, j.left_ankle_pitch, -0.02)
    set_named_target(target, j.right_ankle_pitch, -0.02)

    if phase in {Phase.DOUBLE_SUPPORT, Phase.SAFE_STAND}:
        target = apply_safe_clamps(env, target)
        return target, target_positions_to_dict(target)

    sign = support_sign(side)
    lift_scale = 0.0

    if phase in {Phase.LIFT_RIGHT, Phase.LIFT_LEFT}:
        lift_scale = clamp(phase_step / max(cfg.lift_steps, 1), 0.0, 1.0)
    elif phase in {Phase.PLACE_RIGHT, Phase.PLACE_LEFT}:
        lift_scale = clamp(1.0 - phase_step / max(cfg.place_steps, 1), 0.0, 1.0)

    if side == "left":
        support_hip_roll = j.left_hip_roll
        swing_hip_roll = j.right_hip_roll
        support_hip_pitch = j.left_hip_pitch
        support_knee = j.left_knee_pitch
        support_ankle = j.left_ankle_pitch
        swing_hip_pitch = j.right_hip_pitch
        swing_knee = j.right_knee_pitch
        swing_ankle = j.right_ankle_pitch
    else:
        support_hip_roll = j.right_hip_roll
        swing_hip_roll = j.left_hip_roll
        support_hip_pitch = j.right_hip_pitch
        support_knee = j.right_knee_pitch
        support_ankle = j.right_ankle_pitch
        swing_hip_pitch = j.left_hip_pitch
        swing_knee = j.left_knee_pitch
        swing_ankle = j.left_ankle_pitch

    set_named_target(target, support_hip_roll, sign * cfg.support_roll_target)
    set_named_target(target, swing_hip_roll, -sign * cfg.support_roll_target * cfg.swing_roll_unload_scale)

    set_named_target(target, support_hip_pitch, 0.0)
    set_named_target(target, support_knee, cfg.support_knee_brace)
    set_named_target(target, support_ankle, cfg.support_ankle_brace)

    set_named_target(target, swing_hip_pitch, cfg.swing_unload_hip_pitch)
    set_named_target(target, swing_knee, cfg.swing_unload_knee)
    set_named_target(target, swing_ankle, 0.0)

    if lift_scale > 0.0:
        set_named_target(
            target,
            swing_hip_pitch,
            get_named_target(target, swing_hip_pitch) + cfg.swing_lift_hip_pitch * lift_scale,
        )
        set_named_target(
            target,
            swing_knee,
            get_named_target(target, swing_knee) + cfg.swing_lift_knee * lift_scale,
        )
        set_named_target(
            target,
            swing_ankle,
            get_named_target(target, swing_ankle) + cfg.swing_lift_ankle * lift_scale,
        )

    target = apply_safe_clamps(env, target)
    return target, target_positions_to_dict(target)


def target_positions_to_dict(target_positions: np.ndarray) -> Dict[str, float]:
    return {name: float(target_positions[index]) for index, name in enumerate(JOINT_NAMES)}


def compute_torque_assist(
    state: RobotState,
    memory: ControllerMemory,
    cfg: ControllerConfig,
    side: str,
) -> Tuple[float, float, float]:
    current_ratio = support_ratio(state, side)
    force_error = cfg.target_support_force_ratio - current_ratio
    d_force_error = force_error - memory.previous_force_error

    sign = support_sign(side)
    support_com_shift = sign * state.com_y
    support_com_vel = sign * state.com_y_vel

    tau = (
        cfg.kp_force_tau * force_error
        + cfg.kd_force_tau * d_force_error
        - cfg.kp_com_tau * support_com_shift
        - cfg.kd_com_tau * support_com_vel
        - cfg.kp_base_roll_tau * state.base_roll
        - cfg.kd_base_roll_tau * state.base_roll_vel
    )

    tau = clamp(tau, -cfg.max_tau_assist, cfg.max_tau_assist)

    if side == "left":
        desired_left = sign * tau
        desired_right = -sign * tau * 0.6
    else:
        desired_right = sign * tau
        desired_left = -sign * tau * 0.6

    left_tau = rate_limit(desired_left, memory.left_tau_assist, cfg.max_tau_delta_per_step)
    right_tau = rate_limit(desired_right, memory.right_tau_assist, cfg.max_tau_delta_per_step)

    return left_tau, right_tau, force_error


def is_unsafe(state: RobotState, memory: ControllerMemory, cfg: ControllerConfig) -> bool:
    if abs(state.base_roll) > cfg.max_abs_base_roll:
        return True

    if state.base_z < cfg.min_base_z:
        return True

    if state.forces.contact_state == "none":
        memory.none_contact_steps += 1
    else:
        memory.none_contact_steps = 0

    return memory.none_contact_steps > cfg.max_none_contact_steps


def update_support_hold(state: RobotState, memory: ControllerMemory, cfg: ControllerConfig, side: str) -> None:
    if support_ratio(state, side) >= cfg.target_support_force_ratio:
        memory.support_hold_counter += 1
    else:
        memory.support_hold_counter = 0


def advance_phase(state: RobotState, memory: ControllerMemory, cfg: ControllerConfig) -> None:
    phase = memory.phase

    if phase == Phase.DOUBLE_SUPPORT:
        if memory.phase_step >= cfg.double_support_steps:
            memory.phase = Phase.ACQUIRE_LEFT_SUPPORT
            memory.support_side = "left"
            memory.phase_step = 0
            memory.support_hold_counter = 0
        return

    if phase == Phase.ACQUIRE_LEFT_SUPPORT:
        update_support_hold(state, memory, cfg, "left")
        if memory.support_hold_counter >= cfg.support_hold_required_steps:
            memory.entered_left_support = True
            memory.phase = Phase.LIFT_RIGHT
            memory.phase_step = 0
            memory.support_hold_counter = 0
            return

        if memory.phase_step >= cfg.acquire_max_steps:
            memory.phase = Phase.SAFE_STAND
            memory.phase_step = 0
            memory.support_hold_counter = 0
            return

    if phase == Phase.LIFT_RIGHT:
        if support_ratio(state, "left") < cfg.release_support_force_ratio:
            memory.phase = Phase.SAFE_STAND
            memory.phase_step = 0
            return

        if memory.phase_step >= cfg.lift_steps:
            memory.phase = Phase.PLACE_RIGHT
            memory.phase_step = 0
            return

    if phase == Phase.PLACE_RIGHT:
        if memory.phase_step >= cfg.place_steps:
            memory.phase = Phase.ACQUIRE_RIGHT_SUPPORT
            memory.support_side = "right"
            memory.phase_step = 0
            memory.support_hold_counter = 0
            return

    if phase == Phase.ACQUIRE_RIGHT_SUPPORT:
        update_support_hold(state, memory, cfg, "right")
        if memory.support_hold_counter >= cfg.support_hold_required_steps:
            memory.entered_right_support = True
            memory.phase = Phase.LIFT_LEFT
            memory.phase_step = 0
            memory.support_hold_counter = 0
            return

        if memory.phase_step >= cfg.acquire_max_steps:
            memory.phase = Phase.SAFE_STAND
            memory.phase_step = 0
            memory.support_hold_counter = 0
            return

    if phase == Phase.LIFT_LEFT:
        if support_ratio(state, "right") < cfg.release_support_force_ratio:
            memory.phase = Phase.SAFE_STAND
            memory.phase_step = 0
            return

        if memory.phase_step >= cfg.lift_steps:
            memory.phase = Phase.PLACE_LEFT
            memory.phase_step = 0
            return

    if phase == Phase.PLACE_LEFT:
        if memory.phase_step >= cfg.place_steps:
            memory.phase = Phase.DOUBLE_SUPPORT
            memory.phase_step = 0
            memory.support_hold_counter = 0
            return

    if phase == Phase.SAFE_STAND:
        if memory.phase_step >= cfg.double_support_steps:
            memory.phase = Phase.DOUBLE_SUPPORT
            memory.phase_step = 0
            memory.support_hold_counter = 0
            memory.left_tau_assist = 0.0
            memory.right_tau_assist = 0.0
            return


def build_command(env: SedonStandingEnv, state: RobotState, memory: ControllerMemory, cfg: ControllerConfig) -> StepCommand:
    phase = memory.phase

    if phase in {Phase.DOUBLE_SUPPORT, Phase.SAFE_STAND}:
        target_positions, target_by_name = build_pose_target(env, memory.support_side, cfg, phase, memory.phase_step)
        left_tau = rate_limit(0.0, memory.left_tau_assist, cfg.max_tau_delta_per_step)
        right_tau = rate_limit(0.0, memory.right_tau_assist, cfg.max_tau_delta_per_step)
        memory.left_tau_assist = left_tau
        memory.right_tau_assist = right_tau

        return StepCommand(
            target_positions=target_positions,
            target_by_name=target_by_name,
            left_tau_assist=left_tau,
            right_tau_assist=right_tau,
            force_ratio_error=0.0,
            allow_lift=False,
        )

    if phase in {Phase.ACQUIRE_LEFT_SUPPORT, Phase.LIFT_RIGHT, Phase.PLACE_RIGHT}:
        side = "left"
    else:
        side = "right"

    target_positions, target_by_name = build_pose_target(env, side, cfg, phase, memory.phase_step)

    left_tau, right_tau, force_error = compute_torque_assist(state, memory, cfg, side)

    memory.left_tau_assist = left_tau
    memory.right_tau_assist = right_tau

    allow_lift = False
    if phase in {Phase.LIFT_RIGHT, Phase.LIFT_LEFT}:
        allow_lift = support_ratio(state, side) >= cfg.release_support_force_ratio
        memory.entered_lift = memory.entered_lift or allow_lift

    return StepCommand(
        target_positions=target_positions,
        target_by_name=target_by_name,
        left_tau_assist=left_tau,
        right_tau_assist=right_tau,
        force_ratio_error=force_error,
        allow_lift=allow_lift,
    )


def resolve_actuator_id(model, actuator_or_joint_name: str, fallback_joint_index: Optional[int]) -> Optional[int]:
    for name in [
        actuator_or_joint_name,
        actuator_or_joint_name.replace("_joint_", "_actuator_"),
        actuator_or_joint_name.replace("joint", "motor"),
    ]:
        try:
            return int(model.actuator(name).id)
        except Exception:
            pass

    if fallback_joint_index is not None and fallback_joint_index < model.nu:
        return int(fallback_joint_index)

    return None


def get_joint_qpos_qvel(model, data, joint_name: str) -> Tuple[float, float]:
    joint_id = int(model.joint(joint_name).id)
    qpos_addr = int(model.jnt_qposadr[joint_id])
    dof_addr = int(model.jnt_dofadr[joint_id])
    return float(data.qpos[qpos_addr]), float(data.qvel[dof_addr])


def compute_pd_ctrl(env: SedonStandingEnv, target_positions: np.ndarray, cfg: ControllerConfig) -> np.ndarray:
    model, data = _safe_get_model_data(env)

    ctrl = np.zeros(model.nu, dtype=float)

    for joint_index, joint_name in enumerate(JOINT_NAMES):
        if joint_index >= model.nu:
            continue

        qpos, qvel = get_joint_qpos_qvel(model, data, joint_name)
        target = float(target_positions[joint_index])

        torque = cfg.pd_stiffness * (target - qpos) - cfg.pd_damping * qvel
        ctrl[joint_index] = cfg.torque_scale * torque

    return ctrl


def inject_hip_roll_torque(env: SedonStandingEnv, ctrl: np.ndarray, left_tau: float, right_tau: float) -> np.ndarray:
    model, _ = _safe_get_model_data(env)
    j = JointNames()

    left_joint_index = JOINT_NAMES.index(j.left_hip_roll) if j.left_hip_roll in JOINT_NAMES else None
    right_joint_index = JOINT_NAMES.index(j.right_hip_roll) if j.right_hip_roll in JOINT_NAMES else None

    left_actuator_id = resolve_actuator_id(model, j.left_hip_roll, left_joint_index)
    right_actuator_id = resolve_actuator_id(model, j.right_hip_roll, right_joint_index)

    if left_actuator_id is not None and left_actuator_id < len(ctrl):
        ctrl[left_actuator_id] += left_tau

    if right_actuator_id is not None and right_actuator_id < len(ctrl):
        ctrl[right_actuator_id] += right_tau

    return ctrl


def clamp_ctrl_to_range(model, ctrl: np.ndarray) -> np.ndarray:
    if not hasattr(model, "actuator_ctrllimited"):
        return ctrl

    for actuator_id in range(model.nu):
        limited = bool(model.actuator_ctrllimited[actuator_id])
        if not limited:
            continue

        low, high = model.actuator_ctrlrange[actuator_id]
        ctrl[actuator_id] = clamp(float(ctrl[actuator_id]), float(low), float(high))

    return ctrl


def hybrid_step_env(env: SedonStandingEnv, command: StepCommand, cfg: ControllerConfig) -> None:
    """
    Step using SedonStandingEnv's existing PD simulation path.

    Important:
        Do not call mujoco.mj_step() directly here.
        The existing env has its own simulation/reset/safety assumptions.
    """
    if not hasattr(env, "_do_pd_simulation"):
        raise RuntimeError("SedonStandingEnv must expose _do_pd_simulation for this preview tool.")

    target_positions = command.target_positions.copy()
    if hasattr(env, "_do_pd_simulation_with_torque_assist"):
        env._do_pd_simulation_with_torque_assist(
            target_positions,
            left_tau_assist=command.left_tau_assist,
            right_tau_assist=command.right_tau_assist,
        )
        return

    env._do_pd_simulation(target_positions)
    
def reset_env(env: SedonStandingEnv):
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        return reset_result[0]
    return reset_result


def render_env(env: SedonStandingEnv, sleep_s: float) -> None:
    if hasattr(env, "render"):
        env.render()

    if sleep_s > 0:
        time.sleep(sleep_s)


def build_log_row(state: RobotState, command: StepCommand, memory: ControllerMemory, env: SedonStandingEnv) -> Dict[str, object]:
    model, data = _safe_get_model_data(env)
    j = JointNames()

    left_qpos, _ = get_joint_qpos_qvel(model, data, j.left_hip_roll)
    right_qpos, _ = get_joint_qpos_qvel(model, data, j.right_hip_roll)

    left_target = float(command.target_by_name.get(j.left_hip_roll, 0.0))
    right_target = float(command.target_by_name.get(j.right_hip_roll, 0.0))
    if hasattr(env, "last_hip_roll_ctrl_assist_delta"):
        left_injected_ctrl_delta, right_injected_ctrl_delta = env.last_hip_roll_ctrl_assist_delta()
    else:
        left_injected_ctrl_delta, right_injected_ctrl_delta = command.left_tau_assist, command.right_tau_assist

    support_side = memory.support_side
    current_force_ratio = support_ratio(state, support_side)

    return {
        "step": state.step,
        "phase": state.phase.value,
        "phase_step": memory.phase_step,
        "support_side": support_side,
        "target_force_ratio": command.force_ratio_error + current_force_ratio,
        "current_force_ratio": current_force_ratio,
        "force_ratio_error": command.force_ratio_error,
        "hold_counter": memory.support_hold_counter,
        "allow_lift": int(command.allow_lift),
        "left_tau_assist": command.left_tau_assist,
        "right_tau_assist": command.right_tau_assist,
        "left_injected_ctrl_delta": left_injected_ctrl_delta,
        "right_injected_ctrl_delta": right_injected_ctrl_delta,
        "left_normal_force": state.forces.left_force,
        "right_normal_force": state.forces.right_force,
        "left_force_ratio": state.forces.left_ratio,
        "right_force_ratio": state.forces.right_ratio,
        "left_contact_count": state.forces.left_contact_count,
        "right_contact_count": state.forces.right_contact_count,
        "contact_state": state.forces.contact_state,
        "com_y": state.com_y,
        "com_y_vel": state.com_y_vel,
        "base_z": state.base_z,
        "base_roll": state.base_roll,
        "base_roll_vel": state.base_roll_vel,
        "target_left_hip_roll": left_target,
        "target_right_hip_roll": right_target,
        "left_hip_roll_qpos": left_qpos,
        "right_hip_roll_qpos": right_qpos,
        "left_hip_roll_error": left_target - left_qpos,
        "right_hip_roll_error": right_target - right_qpos,
        "terminated": int(memory.terminated_step is not None),
    }


def write_csv(path: Path, logs: List[StepLog]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not logs:
        return

    fieldnames = list(logs[0].row.keys())

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for log in logs:
            writer.writerow(log.row)


def summarize(logs: List[StepLog], memory: ControllerMemory) -> Dict[str, object]:
    if not logs:
        return {}

    rows = [log.row for log in logs]

    left_ratios = np.array([float(r["left_force_ratio"]) for r in rows], dtype=float)
    right_ratios = np.array([float(r["right_force_ratio"]) for r in rows], dtype=float)
    com_y = np.array([float(r["com_y"]) for r in rows], dtype=float)
    left_tau = np.array([float(r["left_tau_assist"]) for r in rows], dtype=float)
    right_tau = np.array([float(r["right_tau_assist"]) for r in rows], dtype=float)
    contact_states = [str(r["contact_state"]) for r in rows]

    baseline_com_y = float(com_y[0])
    max_abs_com_shift = float(np.max(np.abs(com_y - baseline_com_y)))

    both_ratio = contact_states.count("both") / max(len(contact_states), 1)
    none_ratio = contact_states.count("none") / max(len(contact_states), 1)

    last_n = min(50, len(rows))

    return {
        "steps": len(rows),
        "entered_left_support": memory.entered_left_support,
        "entered_right_support": memory.entered_right_support,
        "entered_lift": memory.entered_lift,
        "terminated_step": memory.terminated_step,
        "max_left_force_ratio": float(np.max(left_ratios)),
        "max_right_force_ratio": float(np.max(right_ratios)),
        "mean_left_force_ratio_last_50": float(np.mean(left_ratios[-last_n:])),
        "mean_right_force_ratio_last_50": float(np.mean(right_ratios[-last_n:])),
        "max_abs_com_shift": max_abs_com_shift,
        "max_abs_left_tau_assist": float(np.max(np.abs(left_tau))),
        "max_abs_right_tau_assist": float(np.max(np.abs(right_tau))),
        "mean_abs_left_tau_last_50": float(np.mean(np.abs(left_tau[-last_n:]))),
        "mean_abs_right_tau_last_50": float(np.mean(np.abs(right_tau[-last_n:]))),
        "both_contact_ratio": both_ratio,
        "none_contact_ratio": none_ratio,
        "authority_failure": (
            float(np.max(left_ratios)) < 0.60
            and float(np.max(right_ratios)) < 0.60
            and both_ratio > 0.90
        ),
    }


def print_summary(summary: Dict[str, object]) -> None:
    print("\n=== Hybrid Torque Balance Controller Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


def print_row(row: Dict[str, object]) -> None:
    print(
        f"step={row['step']:>4} "
        f"phase={row['phase']:<22} "
        f"side={row['support_side']:<5} "
        f"ratio={float(row['current_force_ratio']):.3f} "
        f"L/R={float(row['left_force_ratio']):.3f}/{float(row['right_force_ratio']):.3f} "
        f"tau={float(row['left_tau_assist']):+.2f}/{float(row['right_tau_assist']):+.2f} "
        f"hold={row['hold_counter']:>2} "
        f"com_y={float(row['com_y']):+.5f} "
        f"roll={float(row['base_roll']):+.4f} "
        f"contact={row['contact_state']}"
    )


def run_controller(args: argparse.Namespace) -> Dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    env = SedonStandingEnv(scene_path=DEFAULT_SCENE_PATH.resolve())
    reset_env(env)

    cfg = ControllerConfig(
        target_support_force_ratio=args.target_force_ratio,
        release_support_force_ratio=args.release_force_ratio,
        support_hold_required_steps=args.support_hold_required_steps,
        support_roll_target=args.support_roll_target,
        max_tau_assist=args.max_tau_assist,
        kp_force_tau=args.kp_force_tau,
        kd_force_tau=args.kd_force_tau,
        kp_com_tau=args.kp_com_tau,
        kd_com_tau=args.kd_com_tau,
        kp_base_roll_tau=args.kp_base_roll_tau,
        kd_base_roll_tau=args.kd_base_roll_tau,
        physics_steps_per_control=args.physics_steps_per_control,
        pd_stiffness=args.pd_stiffness,
        pd_damping=args.pd_damping,
        torque_scale=args.torque_scale,
    )

    memory = ControllerMemory()
    logs: List[StepLog] = []

    for step in range(args.steps):
        state = read_robot_state(env, memory, step)

        if is_unsafe(state, memory, cfg):
            if memory.terminated_step is None:
                memory.terminated_step = step
            memory.phase = Phase.SAFE_STAND

        command = build_command(env, state, memory, cfg)

        row = build_log_row(state, command, memory, env)
        logs.append(StepLog(row=row))

        if args.print_every > 0 and step % args.print_every == 0:
            print_row(row)

        if args.render_viewer:
            render_env(env, args.viewer_sleep)

        hybrid_step_env(env, command, cfg)

        memory.previous_com_y = state.com_y
        memory.previous_base_roll = state.base_roll
        memory.previous_force_error = command.force_ratio_error

        advance_phase(state, memory, cfg)
        memory.phase_step += 1

    output_path = Path(args.output_csv)
    write_csv(output_path, logs)

    summary = summarize(logs, memory)
    print_summary(summary)
    print(f"\nCSV written to: {output_path}")

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview Sedon hybrid torque balance controller.")

    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--render-viewer", action="store_true")
    parser.add_argument("--viewer-sleep", type=float, default=0.02)
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_CSV))

    parser.add_argument("--target-force-ratio", type=float, default=0.58)
    parser.add_argument("--release-force-ratio", type=float, default=0.53)
    parser.add_argument("--support-hold-required-steps", type=int, default=15)
    parser.add_argument("--support-roll-target", type=float, default=0.06)

    parser.add_argument("--max-tau-assist", type=float, default=2.0)
    parser.add_argument("--kp-force-tau", type=float, default=4.0)
    parser.add_argument("--kd-force-tau", type=float, default=0.6)
    parser.add_argument("--kp-com-tau", type=float, default=0.6)
    parser.add_argument("--kd-com-tau", type=float, default=0.1)
    parser.add_argument("--kp-base-roll-tau", type=float, default=0.5)
    parser.add_argument("--kd-base-roll-tau", type=float, default=0.08)

    parser.add_argument("--pd-stiffness", type=float, default=35.0)
    parser.add_argument("--pd-damping", type=float, default=2.0)
    parser.add_argument("--torque-scale", type=float, default=45.0)
    parser.add_argument("--physics-steps-per-control", type=int, default=5)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_controller(args)


if __name__ == "__main__":
    main()
