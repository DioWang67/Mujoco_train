"""Simplified capture-step controller skeleton for Sedon.

This Class C tool is a deterministic controller/debug probe, not a walking
claim. It changes the control logic from "lift a foot" toward "place the next
foot to catch the forward-falling body" and writes a CSV timeline for review.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from sedon_baseline.env import SedonStandingEnv, load_sedon_config_from_env
from tools.sedon_blue_like_phase1_rollover_diagnostic import (
    _foot_contact_forces,
    _region_for_geom,
    _side_for_geom,
)
from tools.sedon_debug_common import DEBUG_OUT_DIR, DEFAULT_SCENE_PATH, geom_name, require_scene


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V5A_SCENE = DEBUG_OUT_DIR / "blue_like_sole_experiments_v5" / "training_scene_v5_a.xml"
DEFAULT_OUT_DIR = DEBUG_OUT_DIR / "capture_step_controller_v1"

R_HIP_ROLL, R_HIP_PITCH, R_KNEE, R_ANKLE = 1, 2, 3, 4
L_HIP_ROLL, L_HIP_PITCH, L_KNEE, L_ANKLE = 6, 7, 8, 9


class CapturePhase(str, Enum):
    """Blue-like capture-step controller phases."""

    DOUBLE_SUPPORT = "DOUBLE_SUPPORT"
    COM_FORWARD_COMMIT = "COM_FORWARD_COMMIT"
    LOAD_SUPPORT_FOOT = "LOAD_SUPPORT_FOOT"
    UNLOAD_SWING_FOOT = "UNLOAD_SWING_FOOT"
    SWING_FORWARD = "SWING_FORWARD"
    TOUCHDOWN_CAPTURE = "TOUCHDOWN_CAPTURE"
    TRANSFER_WEIGHT = "TRANSFER_WEIGHT"


@dataclass
class CaptureRuntime:
    """Mutable FSM state for the capture-step skeleton."""

    phase: CapturePhase = CapturePhase.DOUBLE_SUPPORT
    phase_step: int = 0
    support_side: str = "left"
    swing_side: str = "right"
    cycle_count: int = 0


@dataclass(frozen=True)
class CaptureConfig:
    """Tunable constants for capture-step v1."""

    double_support_steps: int = 35
    commit_steps: int = 40
    load_steps: int = 55
    unload_steps: int = 65
    swing_steps: int = 55
    touchdown_steps: int = 45
    transfer_steps: int = 45
    initial_forward_velocity: float = 0.08
    k_vx: float = 0.55
    step_offset_x: float = 0.035
    nominal_stance_y: float = 0.055
    lateral_capture_offset: float = 0.012
    forward_lean_pitch: float = 0.018
    support_hip_roll: float = 0.055
    swing_unload_hip_roll: float = 0.025
    swing_hip_pitch_gain: float = 0.9
    max_swing_hip_pitch: float = 0.18
    knee_soften: float = -0.035
    ankle_soften: float = -0.020


@dataclass(frozen=True)
class CaptureSummary:
    """Aggregate capture-step controller result."""

    steps: int
    forward_displacement: float
    mean_forward_velocity: float
    min_upright: float
    contact_none_ratio: float
    jump_count: int
    toe_handoff_detected: bool
    support_force_ratio_mean: float
    swing_force_ratio_min: float
    left_right_phase_switch_count: int
    phase_cycles: int
    still_grounded_shuffle: bool
    csv_path: str
    json_path: str


def _default_scene() -> Path:
    """Return the preferred v5_a scene when present, otherwise the standard scene."""

    return DEFAULT_V5A_SCENE if DEFAULT_V5A_SCENE.is_file() else DEFAULT_SCENE_PATH


def _other_side(side: str) -> str:
    """Return the opposite support/swing side."""

    if side == "right":
        return "left"
    if side == "left":
        return "right"
    raise ValueError(f"Unsupported side: {side}")


def _side_index(side: str) -> int:
    """Return Sedon foot geom index for a side."""

    if side == "right":
        return 0
    if side == "left":
        return 1
    raise ValueError(f"Unsupported side: {side}")


def _phase_duration(phase: CapturePhase, config: CaptureConfig) -> int:
    """Return configured duration for one phase."""

    durations = {
        CapturePhase.DOUBLE_SUPPORT: config.double_support_steps,
        CapturePhase.COM_FORWARD_COMMIT: config.commit_steps,
        CapturePhase.LOAD_SUPPORT_FOOT: config.load_steps,
        CapturePhase.UNLOAD_SWING_FOOT: config.unload_steps,
        CapturePhase.SWING_FORWARD: config.swing_steps,
        CapturePhase.TOUCHDOWN_CAPTURE: config.touchdown_steps,
        CapturePhase.TRANSFER_WEIGHT: config.transfer_steps,
    }
    return durations[phase]


def _advance(runtime: CaptureRuntime, config: CaptureConfig) -> str:
    """Advance the FSM when the current phase duration expires."""

    if runtime.phase_step < _phase_duration(runtime.phase, config):
        runtime.phase_step += 1
        return ""

    order = list(CapturePhase)
    index = order.index(runtime.phase)
    if runtime.phase == CapturePhase.TRANSFER_WEIGHT:
        runtime.support_side = runtime.swing_side
        runtime.swing_side = _other_side(runtime.support_side)
        runtime.phase = CapturePhase.DOUBLE_SUPPORT
        runtime.phase_step = 0
        runtime.cycle_count += 1
        return "switch_support"
    runtime.phase = order[index + 1]
    runtime.phase_step = 0
    return f"enter_{runtime.phase.value.lower()}"


def _signed_roll_targets(runtime: CaptureRuntime, config: CaptureConfig, load_alpha: float) -> tuple[float, float]:
    """Return right/left hip-roll target offsets for support loading."""

    support = config.support_hip_roll * load_alpha
    swing = config.swing_unload_hip_roll * load_alpha
    if runtime.support_side == "left":
        return swing, support
    return -support, -swing


def _capture_target_xy(env: SedonStandingEnv, runtime: CaptureRuntime, config: CaptureConfig) -> tuple[float, float]:
    """Compute simple capture foot target from base position and velocity."""

    base_x = float(env.data.qpos[0])
    base_vx = float(env.data.qvel[0])
    target_x = base_x + config.k_vx * base_vx + config.step_offset_x
    lateral_sign = -1.0 if runtime.swing_side == "right" else 1.0
    target_y = lateral_sign * (config.nominal_stance_y + config.lateral_capture_offset)
    return target_x, target_y


def _swing_foot_xy(env: SedonStandingEnv, runtime: CaptureRuntime) -> tuple[float, float, float]:
    """Return current swing foot center position."""

    geom_id = env._foot_geom_ids[_side_index(runtime.swing_side)]
    pos = env.data.geom_xpos[geom_id]
    return float(pos[0]), float(pos[1]), float(pos[2] - env.model.geom_size[geom_id][2])


def _toe_center_contact_for_side(env: SedonStandingEnv, side: str) -> tuple[bool, bool]:
    """Return toe and center contact booleans for one side from all foot geoms."""

    toe = False
    center = False
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        name_a = geom_name(env.model, int(contact.geom1))
        name_b = geom_name(env.model, int(contact.geom2))
        if "floor" not in {name_a, name_b}:
            continue
        foot_name = name_b if name_a == "floor" else name_a
        if _side_for_geom(foot_name) != side:
            continue
        foot_geom_id = int(contact.geom2 if name_a == "floor" else contact.geom1)
        foot_body_id = int(env.model.geom_bodyid[foot_geom_id])
        world_pos = np.asarray(contact.pos, dtype=np.float64)
        body_pos = env.data.xpos[foot_body_id]
        body_xmat = env.data.xmat[foot_body_id].reshape(3, 3)
        local_pos = body_xmat.T @ (world_pos - body_pos)
        region = _region_for_geom(foot_name, float(local_pos[0]))
        toe = toe or region == "toe"
        center = center or region == "center"
    return toe, center


def _minimum_jerk(alpha: float) -> float:
    """Smooth 0..1 interpolation."""

    t = float(np.clip(alpha, 0.0, 1.0))
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


def _controller_target(
    env: SedonStandingEnv,
    runtime: CaptureRuntime,
    config: CaptureConfig,
) -> tuple[np.ndarray, float, float]:
    """Return the current joint target and capture foot target.

    The mapping is intentionally simple: hip roll loads support, hip pitch
    creates forward swing tendency, and the debug target records where a future
    IK/WBC layer should place the foot.
    """

    target = env._nominal_joint_qpos.copy()
    phase_alpha = _minimum_jerk(runtime.phase_step / max(_phase_duration(runtime.phase, config), 1))
    target_foot_x, target_foot_y = _capture_target_xy(env, runtime, config)
    load_alpha = 0.0
    swing_alpha = 0.0

    if runtime.phase == CapturePhase.COM_FORWARD_COMMIT:
        target[R_HIP_PITCH] += config.forward_lean_pitch * phase_alpha
        target[L_HIP_PITCH] += config.forward_lean_pitch * phase_alpha
    elif runtime.phase == CapturePhase.LOAD_SUPPORT_FOOT:
        load_alpha = phase_alpha
    elif runtime.phase == CapturePhase.UNLOAD_SWING_FOOT:
        load_alpha = 1.0
    elif runtime.phase == CapturePhase.SWING_FORWARD:
        load_alpha = 1.0
        swing_alpha = phase_alpha
    elif runtime.phase == CapturePhase.TOUCHDOWN_CAPTURE:
        load_alpha = 1.0 - 0.35 * phase_alpha
        swing_alpha = 1.0 - 0.25 * phase_alpha
    elif runtime.phase == CapturePhase.TRANSFER_WEIGHT:
        load_alpha = 1.0 - phase_alpha
        swing_alpha = 0.0

    right_roll, left_roll = _signed_roll_targets(runtime, config, load_alpha)
    target[R_HIP_ROLL] += right_roll
    target[L_HIP_ROLL] += left_roll

    actual_x, _, _ = _swing_foot_xy(env, runtime)
    forward_error = max(0.0, target_foot_x - actual_x)
    swing_pitch = min(config.max_swing_hip_pitch, config.swing_hip_pitch_gain * forward_error) * swing_alpha
    if runtime.swing_side == "right":
        target[R_HIP_PITCH] += swing_pitch
        target[R_KNEE] += config.knee_soften * swing_alpha
        target[R_ANKLE] += config.ankle_soften * swing_alpha
    else:
        target[L_HIP_PITCH] += swing_pitch
        target[L_KNEE] += config.knee_soften * swing_alpha
        target[L_ANKLE] += config.ankle_soften * swing_alpha
    return env._apply_safe_joint_target_clamps(target), target_foot_x, target_foot_y


def _force_ratios(forces: dict[str, float | bool | int], runtime: CaptureRuntime) -> tuple[float, float, float, float]:
    """Return support/swing force and force ratios."""

    support_force = float(forces[f"{runtime.support_side}_force"])
    swing_force = float(forces[f"{runtime.swing_side}_force"])
    total = max(support_force + swing_force, 1e-9)
    return support_force, swing_force, support_force / total, swing_force / total


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write controller timeline rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _support_switches(rows: list[dict[str, Any]]) -> int:
    """Count support-side switches in timeline rows."""

    switches = 0
    previous = ""
    for row in rows:
        side = str(row["support_side"])
        if previous and side != previous:
            switches += 1
        previous = side
    return switches


def _count_jump_bursts(rows: list[dict[str, Any]]) -> int:
    """Count contiguous jump/contact-none bursts."""

    count = 0
    active = False
    for row in rows:
        jumped = bool(row["jump_indicator"])
        if jumped and not active:
            count += 1
            active = True
        elif not jumped:
            active = False
    return count


def _toe_handoff_detected(rows: list[dict[str, Any]]) -> bool:
    """Return whether center contact is followed by toe contact."""

    for side in ("left", "right"):
        saw_center = False
        for row in rows:
            if str(row["support_side"]) != side:
                continue
            saw_center = saw_center or bool(row["center_contact"])
            if saw_center and bool(row["toe_contact"]):
                return True
    return False


def _summarize(rows: list[dict[str, Any]], csv_path: Path, json_path: Path, cycles: int) -> CaptureSummary:
    """Aggregate controller debug rows."""

    if not rows:
        return CaptureSummary(
            steps=0,
            forward_displacement=0.0,
            mean_forward_velocity=0.0,
            min_upright=0.0,
            contact_none_ratio=1.0,
            jump_count=0,
            toe_handoff_detected=False,
            support_force_ratio_mean=0.0,
            swing_force_ratio_min=0.0,
            left_right_phase_switch_count=0,
            phase_cycles=cycles,
            still_grounded_shuffle=True,
            csv_path=str(csv_path),
            json_path=str(json_path),
        )
    contact_none_ratio = sum(1 for row in rows if bool(row["contact_none"])) / len(rows)
    single_contact_ratio = sum(1 for row in rows if str(row["contact_state"]) in {"left", "right"}) / len(rows)
    return CaptureSummary(
        steps=len(rows),
        forward_displacement=float(rows[-1]["base_x"]) - float(rows[0]["base_x"]),
        mean_forward_velocity=float(np.mean([float(row["base_vx"]) for row in rows])),
        min_upright=min(float(row["upright"]) for row in rows),
        contact_none_ratio=contact_none_ratio,
        jump_count=_count_jump_bursts(rows),
        toe_handoff_detected=_toe_handoff_detected(rows),
        support_force_ratio_mean=float(np.mean([float(row["support_force_ratio"]) for row in rows])),
        swing_force_ratio_min=min(float(row["swing_force_ratio"]) for row in rows),
        left_right_phase_switch_count=_support_switches(rows),
        phase_cycles=cycles,
        still_grounded_shuffle=single_contact_ratio < 0.05,
        csv_path=str(csv_path),
        json_path=str(json_path),
    )


def run_controller(
    *,
    scene_path: Path,
    csv_path: Path,
    json_path: Path,
    steps: int,
    seed: int,
    config: CaptureConfig,
) -> CaptureSummary:
    """Run the capture-step controller skeleton and persist debug outputs.

    Args:
        scene_path: MuJoCo XML scene.
        csv_path: Destination CSV timeline path.
        json_path: Destination JSON summary path.
        steps: Number of control steps.
        seed: Deterministic seed.
        config: Controller constants.

    Returns:
        Aggregate controller summary.

    Raises:
        FileNotFoundError: If scene_path does not exist.
        ValueError: If steps is not positive.
    """

    if steps <= 0:
        raise ValueError("steps must be positive.")
    scene = require_scene(scene_path)
    env = SedonStandingEnv(
        scene_path=scene,
        reset_noise_scale=0.0,
        reward_config=load_sedon_config_from_env(),
    )
    runtime = CaptureRuntime()
    rows: list[dict[str, Any]] = []
    try:
        env.reset(seed=seed)
        env.data.qvel[0] = float(config.initial_forward_velocity)
        mujoco.mj_forward(env.model, env.data)
        robot_weight = float(np.sum(env.model.body_mass) * 9.81)
        initial_x = float(env.data.qpos[0])
        previous_total_force = 0.0
        for step in range(1, steps + 1):
            event = _advance(runtime, config)
            target, target_foot_x, target_foot_y = _controller_target(env, runtime, config)
            env._do_pd_simulation(target)
            env._gait_step += 1
            forces = _foot_contact_forces(env)
            support_force, swing_force, support_ratio, swing_ratio = _force_ratios(forces, runtime)
            left_force = float(forces["left_force"])
            right_force = float(forces["right_force"])
            total_force = left_force + right_force
            if left_force > 0.1 and right_force > 0.1:
                contact_state = "both"
            elif left_force > 0.1:
                contact_state = "left"
            elif right_force > 0.1:
                contact_state = "right"
            else:
                contact_state = "none"
            actual_x, actual_y, actual_z = _swing_foot_xy(env, runtime)
            toe_contact, center_contact = _toe_center_contact_for_side(env, runtime.support_side)
            contact_none = contact_state == "none"
            jump_indicator = bool(contact_none or (float(env.data.qvel[2]) > 0.05 and total_force < 0.10 * robot_weight))
            rows.append(
                {
                    "time": float(env.data.time),
                    "step": step,
                    "phase": runtime.phase.value,
                    "phase_step": runtime.phase_step,
                    "event": event,
                    "support_side": runtime.support_side,
                    "swing_side": runtime.swing_side,
                    "base_x": float(env.data.qpos[0]),
                    "base_vx": float(env.data.qvel[0]),
                    "base_z": float(env._base_height()),
                    "upright": float(env._base_upright()),
                    "left_force": left_force,
                    "right_force": right_force,
                    "support_force": support_force,
                    "swing_force": swing_force,
                    "support_force_ratio": support_ratio,
                    "swing_force_ratio": swing_ratio,
                    "target_foot_x": target_foot_x,
                    "target_foot_y": target_foot_y,
                    "actual_foot_x": actual_x,
                    "actual_foot_y": actual_y,
                    "actual_foot_z": actual_z,
                    "toe_contact": toe_contact,
                    "center_contact": center_contact,
                    "contact_state": contact_state,
                    "contact_none": contact_none,
                    "jump_indicator": jump_indicator,
                    "forward_displacement": float(env.data.qpos[0]) - initial_x,
                    "touchdown_impact_proxy": max(0.0, total_force - previous_total_force) / max(robot_weight, 1e-9),
                    "target_joint_values": json.dumps([round(float(value), 6) for value in target]),
                }
            )
            previous_total_force = total_force
    finally:
        env.close()

    _write_csv(csv_path, rows)
    summary = _summarize(rows, csv_path, json_path, runtime.cycle_count)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-path", type=Path, default=_default_scene())
    parser.add_argument("--csv", type=Path, default=DEFAULT_OUT_DIR / "capture_step_controller_v1.csv")
    parser.add_argument("--json-summary", type=Path, default=DEFAULT_OUT_DIR / "capture_step_controller_v1_summary.json")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-forward-velocity", type=float, default=CaptureConfig.initial_forward_velocity)
    parser.add_argument("--k-vx", type=float, default=CaptureConfig.k_vx)
    parser.add_argument("--step-offset-x", type=float, default=CaptureConfig.step_offset_x)
    parser.add_argument("--support-hip-roll", type=float, default=CaptureConfig.support_hip_roll)
    parser.add_argument("--swing-hip-pitch-gain", type=float, default=CaptureConfig.swing_hip_pitch_gain)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the capture-step controller skeleton from CLI."""

    args = build_parser().parse_args(argv)
    config = CaptureConfig(
        initial_forward_velocity=args.initial_forward_velocity,
        k_vx=args.k_vx,
        step_offset_x=args.step_offset_x,
        support_hip_roll=args.support_hip_roll,
        swing_hip_pitch_gain=args.swing_hip_pitch_gain,
    )
    summary = run_controller(
        scene_path=args.scene_path,
        csv_path=args.csv,
        json_path=args.json_summary,
        steps=args.steps,
        seed=args.seed,
        config=config,
    )
    print(
        "capture_v1 steps={steps} forward={forward:.4f} mean_vx={vx:.4f} "
        "toe_handoff={toe} switches={switches} grounded_shuffle={grounded}".format(
            steps=summary.steps,
            forward=summary.forward_displacement,
            vx=summary.mean_forward_velocity,
            toe=summary.toe_handoff_detected,
            switches=summary.left_right_phase_switch_count,
            grounded=summary.still_grounded_shuffle,
        )
    )
    print(f"csv={summary.csv_path}")
    print(f"json={summary.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
