"""
Preview Seedon Blue-like force-ratio controller.

Purpose:
    Prototype a Blue-like controller that tries to acquire left/right support
    by tracking foot normal-force ratio, instead of blindly tracking fixed
    hip-roll targets.

Usage:
    .\\.venv\\Scripts\\python.exe -m tools.preview_seedon_force_ratio_controller --steps 320 --print-every 20
    .\\.venv\\Scripts\\python.exe -m tools.preview_seedon_force_ratio_controller --steps 320 --render-viewer

Notes:
    - Does not modify MJCF / training scene / PPO / reward.
    - Uses SeedonStandingEnv joint target pathway.
    - If force ratio cannot exceed gate, lift is not allowed.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from seedon_baseline.env import (
    DEFAULT_SCENE_PATH,
    JOINT_NAMES,
    SeedonStandingEnv,
    SeedonStandingConfig,
    load_seedon_config_from_env,
)


ARTIFACT_DIR = Path("artifacts/seedon_debug")
DEFAULT_CSV = ARTIFACT_DIR / "force_ratio_controller_preview.csv"


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
    target_support_force_ratio: float = 0.65
    release_support_force_ratio: float = 0.58

    # Force-ratio controller gains.
    kp_force: float = 0.22
    kd_force: float = 0.04

    # COM/base stabilizer terms.
    kp_com: float = 0.10
    kd_com: float = 0.02
    kp_base_roll: float = 0.08
    kd_base_roll: float = 0.01

    # Command limits.
    max_support_roll_cmd: float = 0.22
    min_support_roll_cmd: float = 0.00
    max_cmd_delta_per_step: float = 0.006

    # Brace / unload targets.
    support_knee_brace: float = 0.04
    support_ankle_brace: float = -0.02
    swing_unload_knee: float = -0.06
    swing_unload_hip_pitch: float = 0.02

    # Lift targets.
    swing_lift_hip_pitch: float = 0.10
    swing_lift_knee: float = -0.16
    swing_lift_ankle: float = 0.04

    # Phase timing.
    double_support_steps: int = 30
    acquire_max_steps: int = 180
    lift_steps: int = 50
    place_steps: int = 40

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

    support_roll_cmd: float = 0.0
    none_contact_steps: int = 0

    entered_left_support: bool = False
    entered_right_support: bool = False
    entered_lift: bool = False
    terminated_step: Optional[int] = None


@dataclass
class StepCommand:
    target: Dict[str, float]
    support_roll_cmd: float
    force_ratio_error: float
    allow_lift: bool


@dataclass
class StepLog:
    row: Dict[str, object]


def _safe_get_model_data(env: SeedonStandingEnv):
    """
    Return MuJoCo model/data pair from SeedonStandingEnv.

    The exact env wrapper may differ slightly, so this tries common layouts.
    """
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

    raise RuntimeError("Cannot locate MuJoCo model/data from SeedonStandingEnv.")


def _resolve_joint_qpos_addr(model, joint_name: str) -> int:
    joint_id = model.joint(joint_name).id
    return int(model.jnt_qposadr[joint_id])


def _resolve_joint_qvel_addr(model, joint_name: str) -> int:
    joint_id = model.joint(joint_name).id
    return int(model.jnt_dofadr[joint_id])


def _get_joint_qpos(model, data, joint_name: str) -> float:
    return float(data.qpos[_resolve_joint_qpos_addr(model, joint_name)])


def _get_joint_qvel(model, data, joint_name: str) -> float:
    return float(data.qvel[_resolve_joint_qvel_addr(model, joint_name)])


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
    """
    Approximate contact normal force from MuJoCo efc rows.

    MuJoCo contact dim usually maps first row as normal constraint.
    This function intentionally stays conservative.
    """
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
    """
    Convert root quaternion to roll angle.

    MuJoCo root quaternion is commonly [w, x, y, z].
    """
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.atan2(sinr_cosp, cosr_cosp)


def read_robot_state(env: SeedonStandingEnv, memory: ControllerMemory, step: int) -> RobotState:
    model, data = _safe_get_model_data(env)

    # Root: qpos[0:3] pos, qpos[3:7] quat.
    base_z = float(data.qpos[2])
    qw, qx, qy, qz = [float(x) for x in data.qpos[3:7]]
    base_roll = get_base_roll_from_quat(qw, qx, qy, qz)

    com_y = float(data.subtree_com[0][1]) if hasattr(data, "subtree_com") else float(data.qpos[1])

    if memory.previous_com_y is None:
        com_y_vel = 0.0
    else:
        com_y_vel = com_y - memory.previous_com_y

    if memory.previous_base_roll is None:
        base_roll_vel = 0.0
    else:
        base_roll_vel = base_roll - memory.previous_base_roll

    forces = read_force_state(model, data)

    return RobotState(
        step=step,
        phase=memory.phase,
        com_y=com_y,
        com_y_vel=com_y_vel,
        base_z=base_z,
        base_roll=base_roll,
        base_roll_vel=base_roll_vel,
        forces=forces,
    )


def build_safe_stand_target() -> Dict[str, float]:
    """
    Conservative neutral standing target.

    Adjust these defaults if your Seedon standing seed uses different signs.
    """
    return {
        JointNames.left_hip_roll: 0.0,
        JointNames.right_hip_roll: 0.0,
        JointNames.left_hip_pitch: 0.0,
        JointNames.right_hip_pitch: 0.0,
        JointNames.left_knee_pitch: 0.04,
        JointNames.right_knee_pitch: 0.04,
        JointNames.left_ankle_pitch: -0.02,
        JointNames.right_ankle_pitch: -0.02,
    }


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rate_limit(value: float, previous: float, max_delta: float) -> float:
    delta = clamp(value - previous, -max_delta, max_delta)
    return previous + delta


def support_ratio(state: RobotState, side: str) -> float:
    return state.forces.left_ratio if side == "left" else state.forces.right_ratio


def swing_ratio(state: RobotState, side: str) -> float:
    return state.forces.right_ratio if side == "left" else state.forces.left_ratio


def compute_support_roll_sign(side: str) -> float:
    """
    Sign convention:
        left support: push body toward left side.
        right support: push body toward right side.

    If your model sign is reversed, flip these two signs.
    """
    return 1.0 if side == "left" else -1.0


def apply_support_targets(
    target: Dict[str, float],
    side: str,
    support_roll_cmd: float,
    cfg: ControllerConfig,
    lift_scale: float = 0.0,
) -> Dict[str, float]:
    """
    Map force-ratio controller output to joint targets.

    This keeps the controller simple:
    - support side gets hip-roll and brace.
    - swing side gets unload / lift pose.
    """
    j = JointNames()
    sign = compute_support_roll_sign(side)

    target = dict(target)

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

    # Mirrored hip-roll: support pushes toward support side, swing unloads opposite.
    target[support_hip_roll] = sign * support_roll_cmd
    target[swing_hip_roll] = -sign * support_roll_cmd * 0.55

    # Brace support leg.
    target[support_hip_pitch] = 0.0
    target[support_knee] = cfg.support_knee_brace
    target[support_ankle] = cfg.support_ankle_brace

    # Swing unload / lift.
    unload_scale = clamp(support_roll_cmd / max(cfg.max_support_roll_cmd, 1e-6), 0.0, 1.0)
    target[swing_hip_pitch] = cfg.swing_unload_hip_pitch * unload_scale
    target[swing_knee] = cfg.swing_unload_knee * unload_scale
    target[swing_ankle] = 0.0

    if lift_scale > 0.0:
        target[swing_hip_pitch] += cfg.swing_lift_hip_pitch * lift_scale
        target[swing_knee] += cfg.swing_lift_knee * lift_scale
        target[swing_ankle] += cfg.swing_lift_ankle * lift_scale

    return target


def compute_force_ratio_command(
    state: RobotState,
    memory: ControllerMemory,
    cfg: ControllerConfig,
    side: str,
) -> Tuple[float, float]:
    """
    Compute support hip-roll command from force-ratio feedback.

    Main control objective:
        support_force_ratio -> cfg.target_support_force_ratio
    """
    current_ratio = support_ratio(state, side)
    error = cfg.target_support_force_ratio - current_ratio
    d_error = error - memory.previous_force_error

    # Encourage COM toward support side, but do not make it the main target.
    sign = compute_support_roll_sign(side)
    support_com_shift = sign * state.com_y

    force_term = cfg.kp_force * error + cfg.kd_force * d_error
    com_term = -cfg.kp_com * support_com_shift - cfg.kd_com * sign * state.com_y_vel

    # Base roll stabilizer prevents uncontrolled roll, not lateral acquisition.
    base_term = -cfg.kp_base_roll * state.base_roll - cfg.kd_base_roll * state.base_roll_vel

    raw_cmd = memory.support_roll_cmd + force_term + com_term + base_term

    cmd = clamp(raw_cmd, cfg.min_support_roll_cmd, cfg.max_support_roll_cmd)
    cmd = rate_limit(cmd, memory.support_roll_cmd, cfg.max_cmd_delta_per_step)

    return cmd, error


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


def advance_phase(state: RobotState, memory: ControllerMemory, cfg: ControllerConfig) -> None:
    phase = memory.phase

    if phase == Phase.DOUBLE_SUPPORT:
        if memory.phase_step >= cfg.double_support_steps:
            memory.phase = Phase.ACQUIRE_LEFT_SUPPORT
            memory.support_side = "left"
            memory.phase_step = 0
            memory.support_roll_cmd = 0.0
        return

    if phase == Phase.ACQUIRE_LEFT_SUPPORT:
        if support_ratio(state, "left") >= cfg.target_support_force_ratio:
            memory.entered_left_support = True
            memory.phase = Phase.LIFT_RIGHT
            memory.phase_step = 0
            return

        if memory.phase_step >= cfg.acquire_max_steps:
            memory.phase = Phase.SAFE_STAND
            memory.phase_step = 0
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
            memory.support_roll_cmd = 0.0
            return

    if phase == Phase.ACQUIRE_RIGHT_SUPPORT:
        if support_ratio(state, "right") >= cfg.target_support_force_ratio:
            memory.entered_right_support = True
            memory.phase = Phase.LIFT_LEFT
            memory.phase_step = 0
            return

        if memory.phase_step >= cfg.acquire_max_steps:
            memory.phase = Phase.SAFE_STAND
            memory.phase_step = 0
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
            memory.support_roll_cmd = 0.0
            return

    if phase == Phase.SAFE_STAND:
        if memory.phase_step >= cfg.double_support_steps:
            memory.phase = Phase.DOUBLE_SUPPORT
            memory.phase_step = 0
            memory.support_roll_cmd = 0.0
            return


def build_command(state: RobotState, memory: ControllerMemory, cfg: ControllerConfig) -> StepCommand:
    target = build_safe_stand_target()
    allow_lift = False
    force_error = 0.0

    phase = memory.phase

    if phase in {Phase.DOUBLE_SUPPORT, Phase.SAFE_STAND}:
        memory.support_roll_cmd = rate_limit(
            value=0.0,
            previous=memory.support_roll_cmd,
            max_delta=cfg.max_cmd_delta_per_step,
        )
        return StepCommand(
            target=target,
            support_roll_cmd=memory.support_roll_cmd,
            force_ratio_error=0.0,
            allow_lift=False,
        )

    if phase in {Phase.ACQUIRE_LEFT_SUPPORT, Phase.LIFT_RIGHT, Phase.PLACE_RIGHT}:
        side = "left"
    else:
        side = "right"

    cmd, force_error = compute_force_ratio_command(state, memory, cfg, side)
    memory.support_roll_cmd = cmd

    lift_scale = 0.0

    if phase in {Phase.LIFT_RIGHT, Phase.LIFT_LEFT}:
        allow_lift = support_ratio(state, side) >= cfg.release_support_force_ratio
        if allow_lift:
            memory.entered_lift = True
            lift_scale = clamp(memory.phase_step / max(cfg.lift_steps, 1), 0.0, 1.0)

    if phase in {Phase.PLACE_RIGHT, Phase.PLACE_LEFT}:
        lift_scale = clamp(1.0 - memory.phase_step / max(cfg.place_steps, 1), 0.0, 1.0)

    target = apply_support_targets(
        target=target,
        side=side,
        support_roll_cmd=cmd,
        cfg=cfg,
        lift_scale=lift_scale,
    )

    return StepCommand(
        target=target,
        support_roll_cmd=cmd,
        force_ratio_error=force_error,
        allow_lift=allow_lift,
    )


def apply_joint_targets(env: SeedonStandingEnv, target: Dict[str, float]) -> None:
    """
    Apply target to SeedonStandingEnv.

    This function supports several common target APIs.
    If your env exposes a different method, modify only this function.
    """
    if hasattr(env, "set_joint_targets"):
        env.set_joint_targets(target)
        return

    if hasattr(env, "set_target_pose"):
        env.set_target_pose(target)
        return

    if hasattr(env, "target_pose"):
        for key, value in target.items():
            env.target_pose[key] = value
        return

    if hasattr(env, "_target_pose"):
        for key, value in target.items():
            env._target_pose[key] = value
        return

    if hasattr(env, "joint_targets"):
        for key, value in target.items():
            env.joint_targets[key] = value
        return

    if hasattr(env, "_nominal_joint_qpos") and hasattr(env, "_apply_safe_joint_target_clamps"):
        target_positions = env._nominal_joint_qpos.copy()
        joint_index_by_name = {joint_name: index for index, joint_name in enumerate(JOINT_NAMES)}
        for joint_name, value in target.items():
            joint_index = joint_index_by_name.get(joint_name)
            if joint_index is None:
                continue
            target_positions[joint_index] = float(value)
        env._preview_pending_target_positions = env._apply_safe_joint_target_clamps(target_positions)
        return

    raise RuntimeError(
        "Cannot apply joint targets. Please adapt apply_joint_targets() "
        "to your SeedonStandingEnv target API."
    )


def step_env(env: SeedonStandingEnv):
    """
    Step env once.

    SeedonStandingEnv variants may either accept an action or use internal targets.
    """
    pending_target = getattr(env, "_preview_pending_target_positions", None)
    if pending_target is not None and hasattr(env, "_do_pd_simulation"):
        env._do_pd_simulation(pending_target)
        env._preview_pending_target_positions = None
        return None

    try:
        return env.step(None)
    except TypeError:
        return env.step()


def reset_env(env: SeedonStandingEnv):
    if hasattr(env, "_preview_pending_target_positions"):
        env._preview_pending_target_positions = None
    reset_result = env.reset()

    if isinstance(reset_result, tuple):
        return reset_result[0]

    return reset_result


def render_env(env: SeedonStandingEnv, sleep_s: float) -> None:
    if hasattr(env, "render"):
        env.render()

    if sleep_s > 0:
        time.sleep(sleep_s)


def get_target_value(target: Dict[str, float], key: str) -> float:
    return float(target.get(key, 0.0))


def build_log_row(
    state: RobotState,
    command: StepCommand,
    memory: ControllerMemory,
    env: SeedonStandingEnv,
) -> Dict[str, object]:
    model, data = _safe_get_model_data(env)
    j = JointNames()

    left_hip_roll_qpos = _get_joint_qpos(model, data, j.left_hip_roll)
    right_hip_roll_qpos = _get_joint_qpos(model, data, j.right_hip_roll)

    left_target = get_target_value(command.target, j.left_hip_roll)
    right_target = get_target_value(command.target, j.right_hip_roll)

    current_support_side = memory.support_side
    current_force_ratio = support_ratio(state, current_support_side)

    return {
        "step": state.step,
        "phase": state.phase.value,
        "phase_step": memory.phase_step,
        "support_side": current_support_side,
        "target_force_ratio": ControllerConfig().target_support_force_ratio,
        "current_force_ratio": current_force_ratio,
        "force_ratio_error": command.force_ratio_error,
        "support_roll_cmd": command.support_roll_cmd,
        "allow_lift": int(command.allow_lift),
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
        "left_hip_roll_qpos": left_hip_roll_qpos,
        "right_hip_roll_qpos": right_hip_roll_qpos,
        "left_hip_roll_error": left_target - left_hip_roll_qpos,
        "right_hip_roll_error": right_target - right_hip_roll_qpos,
        "target_left_hip_pitch": get_target_value(command.target, j.left_hip_pitch),
        "target_right_hip_pitch": get_target_value(command.target, j.right_hip_pitch),
        "target_left_knee_pitch": get_target_value(command.target, j.left_knee_pitch),
        "target_right_knee_pitch": get_target_value(command.target, j.right_knee_pitch),
        "target_left_ankle_pitch": get_target_value(command.target, j.left_ankle_pitch),
        "target_right_ankle_pitch": get_target_value(command.target, j.right_ankle_pitch),
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
    support_roll_cmd = np.array([float(r["support_roll_cmd"]) for r in rows], dtype=float)
    contact_states = [str(r["contact_state"]) for r in rows]

    baseline_com_y = float(com_y[0])
    max_abs_com_shift = float(np.max(np.abs(com_y - baseline_com_y)))

    both_ratio = contact_states.count("both") / max(len(contact_states), 1)
    none_ratio = contact_states.count("none") / max(len(contact_states), 1)

    last_n = min(50, len(rows))

    summary = {
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
        "max_support_roll_cmd": float(np.max(np.abs(support_roll_cmd))),
        "mean_support_roll_cmd_last_50": float(np.mean(np.abs(support_roll_cmd[-last_n:]))),
        "both_contact_ratio": both_ratio,
        "none_contact_ratio": none_ratio,
        "authority_failure": (
            float(np.max(left_ratios)) < 0.60
            and float(np.max(right_ratios)) < 0.60
            and both_ratio > 0.90
        ),
    }

    return summary


def print_summary(summary: Dict[str, object]) -> None:
    print("\n=== Force Ratio Controller Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


def print_row(row: Dict[str, object]) -> None:
    print(
        f"step={row['step']:>4} "
        f"phase={row['phase']:<22} "
        f"side={row['support_side']:<5} "
        f"ratio={float(row['current_force_ratio']):.3f} "
        f"L/R={float(row['left_force_ratio']):.3f}/{float(row['right_force_ratio']):.3f} "
        f"cmd={float(row['support_roll_cmd']):.3f} "
        f"com_y={float(row['com_y']):+.5f} "
        f"roll={float(row['base_roll']):+.4f} "
        f"contact={row['contact_state']}"
    )


def run_controller(args: argparse.Namespace) -> Dict[str, object]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    scene_path = DEFAULT_SCENE_PATH.resolve()

    env = SeedonStandingEnv(
        scene_path=scene_path,
    )
    reset_env(env)

    cfg = ControllerConfig(
        target_support_force_ratio=args.target_force_ratio,
        kp_force=args.kp_force,
        kd_force=args.kd_force,
        kp_com=args.kp_com,
        kd_com=args.kd_com,
        kp_base_roll=args.kp_base_roll,
        kd_base_roll=args.kd_base_roll,
        max_support_roll_cmd=args.max_support_roll_cmd,
    )

    memory = ControllerMemory()
    logs: List[StepLog] = []

    for step in range(args.steps):
        state = read_robot_state(env, memory, step)

        if is_unsafe(state, memory, cfg):
            if memory.terminated_step is None:
                memory.terminated_step = step
            memory.phase = Phase.SAFE_STAND

        command = build_command(state, memory, cfg)
        apply_joint_targets(env, command.target)

        row = build_log_row(state, command, memory, env)
        logs.append(StepLog(row=row))

        if args.print_every > 0 and step % args.print_every == 0:
            print_row(row)

        if args.render_viewer:
            render_env(env, args.viewer_sleep)

        step_env(env)

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
    parser = argparse.ArgumentParser(description="Preview Seedon force-ratio Blue-like controller.")

    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--render-viewer", action="store_true")
    parser.add_argument("--viewer-sleep", type=float, default=0.02)
    parser.add_argument("--output-csv", type=str, default=str(DEFAULT_CSV))

    parser.add_argument("--target-force-ratio", type=float, default=0.65)
    parser.add_argument("--kp-force", type=float, default=0.22)
    parser.add_argument("--kd-force", type=float, default=0.04)
    parser.add_argument("--kp-com", type=float, default=0.10)
    parser.add_argument("--kd-com", type=float, default=0.02)
    parser.add_argument("--kp-base-roll", type=float, default=0.08)
    parser.add_argument("--kd-base-roll", type=float, default=0.01)
    parser.add_argument("--max-support-roll-cmd", type=float, default=0.22)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_controller(args)


if __name__ == "__main__":
    main()
