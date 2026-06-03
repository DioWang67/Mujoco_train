"""Explicit Seedon locomotion controller v2 skeleton and audit."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from seedon_baseline.env import SeedonStandingConfig, SeedonStandingEnv
from tools.audit_seedon_shuffle_v0 import _count_contact_none_bursts, _load_config, audit_shuffle
from tools.blue_forward_shuffle_v1 import DEFAULT_CONFIG, DEFAULT_MODEL, DEFAULT_VECNORM


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "seedon_debug" / "controller_v2"

R_HIP_ROLL, R_HIP_PITCH, R_KNEE, R_ANKLE = 1, 2, 3, 4
L_HIP_ROLL, L_HIP_PITCH, L_KNEE, L_ANKLE = 6, 7, 8, 9
SWING_JOINTS = {
    "right": (R_HIP_PITCH, R_KNEE, R_ANKLE),
    "left": (L_HIP_PITCH, L_KNEE, L_ANKLE),
}


class LocomotionPhase(str, Enum):
    """Finite-state-machine phases for explicit locomotion control."""

    DOUBLE_SUPPORT = "DOUBLE_SUPPORT"
    PRELOAD_LEFT = "PRELOAD_LEFT"
    PRELOAD_RIGHT = "PRELOAD_RIGHT"
    UNLOAD_LEFT = "UNLOAD_LEFT"
    UNLOAD_RIGHT = "UNLOAD_RIGHT"
    SWING_LEFT = "SWING_LEFT"
    SWING_RIGHT = "SWING_RIGHT"
    TOUCHDOWN_LEFT = "TOUCHDOWN_LEFT"
    TOUCHDOWN_RIGHT = "TOUCHDOWN_RIGHT"


@dataclass
class ControllerRuntime:
    """Mutable FSM runtime state."""

    phase: LocomotionPhase = LocomotionPhase.DOUBLE_SUPPORT
    phase_step: int = 0
    swing_side: str = "right"
    cycle_count: int = 0
    gate_reached: bool = False
    swing_start_z: float = 0.0
    touchdown_start_z: float = 0.0
    touchdown_start_target: float = 0.0
    active_target: np.ndarray | None = None
    fail_reason: str = ""


@dataclass(frozen=True)
class ControllerConfig:
    """Tunable controller constants for v2 audit."""

    target_clearance: float
    contact_force_threshold_n: float
    double_support_steps: int
    preload_steps: int
    unload_timeout_steps: int
    swing_steps: int
    touchdown_steps: int
    hold_steps: int
    support_hip_roll_limit: float
    swing_hip_roll_limit: float
    force_feedback_gain: float
    ik_eps: float
    ik_damping: float
    max_joint_delta_per_step: float
    touchdown_max_vertical_velocity: float
    impact_fail_multiplier: float


@dataclass(frozen=True)
class ControllerAudit:
    """Aggregate audit result for one controller target-clearance run."""

    target_clearance: float
    steps: int
    contact_none_ratio: float
    jump_count: int
    min_upright: float
    base_drop_post: float
    impact_post: float
    max_contact_force_post: float
    max_clearance: float
    min_swing_force: float
    mean_support_ratio: float
    actuator_ctrl_saturation_max: float
    actuator_force_saturation_max: float
    gate_reached_count: int
    gate_timeout_count: int
    touchdown_spike_fail_count: int
    phase1_passed: bool
    phase2_passed: bool
    fail_reasons: str
    timeline_path: str


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _minimum_jerk(t: float) -> float:
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def _side_index(side: str) -> int:
    return 0 if side == "right" else 1


def _support_side(swing_side: str) -> str:
    return "left" if swing_side == "right" else "right"


def _swing_force(env: SeedonStandingEnv, swing_side: str) -> float:
    forces = env._foot_force_state()
    return float(forces[f"{swing_side}_force"])


def _support_force(env: SeedonStandingEnv, swing_side: str) -> float:
    forces = env._foot_force_state()
    return float(forces[f"{_support_side(swing_side)}_force"])


def _support_ratio(env: SeedonStandingEnv, swing_side: str) -> float:
    support = _support_force(env, swing_side)
    swing = _swing_force(env, swing_side)
    return support / max(support + swing, 1e-9)


def _foot_z(env: SeedonStandingEnv, side: str) -> float:
    return float(env._foot_bottom_heights()[_side_index(side)])


def _clearance(env: SeedonStandingEnv, swing_side: str) -> float:
    swing = _foot_z(env, swing_side)
    support = _foot_z(env, _support_side(swing_side))
    return max(0.0, swing - support)


def _contact_state(env: SeedonStandingEnv) -> str:
    flags = env._floor_contact_flags()
    return env._contact_state(flags)


def _ctrl_saturation(env: SeedonStandingEnv) -> float:
    span = np.maximum(env._ctrl_range[:, 1] - env._ctrl_range[:, 0], 1e-9)
    lower = np.abs(env.data.ctrl - env._ctrl_range[:, 0])
    upper = np.abs(env.data.ctrl - env._ctrl_range[:, 1])
    return float(np.mean(np.minimum(lower, upper) <= 0.02 * span))


def _force_saturation(env: SeedonStandingEnv) -> float:
    if not hasattr(env.model, "actuator_forcerange") or not hasattr(env.data, "actuator_force"):
        return 0.0
    force_range = np.asarray(env.model.actuator_forcerange, dtype=np.float64)
    actuator_force = np.asarray(env.data.actuator_force, dtype=np.float64)
    if force_range.shape[0] != actuator_force.shape[0] or actuator_force.size == 0:
        return 0.0
    span = np.maximum(force_range[:, 1] - force_range[:, 0], 1e-9)
    finite = np.isfinite(force_range).all(axis=1) & (span > 1e-8)
    lower = np.abs(actuator_force - force_range[:, 0])
    upper = np.abs(actuator_force - force_range[:, 1])
    return float(np.mean(finite & (np.minimum(lower, upper) <= 0.02 * span)))


def _preload_target(env: SeedonStandingEnv, swing_side: str, config: ControllerConfig) -> np.ndarray:
    """Return force-feedback hip-roll preload target for the desired swing side."""

    target = env._nominal_joint_qpos.copy()
    force_error = max(0.0, _swing_force(env, swing_side) - config.contact_force_threshold_n)
    feedback = min(config.support_hip_roll_limit, config.force_feedback_gain * force_error)
    swing_feedback = min(config.swing_hip_roll_limit, 0.5 * config.force_feedback_gain * force_error)
    if swing_side == "right":
        target[L_HIP_ROLL] += feedback
        target[R_HIP_ROLL] += swing_feedback
    else:
        target[R_HIP_ROLL] -= feedback
        target[L_HIP_ROLL] -= swing_feedback
    return target


def _set_joint_positions_in_data(env: SeedonStandingEnv, data: mujoco.MjData, joint_positions: np.ndarray) -> None:
    for joint_index, joint_id in enumerate(env._joint_ids):
        data.qpos[env.model.jnt_qposadr[joint_id]] = float(joint_positions[joint_index])


def _kinematic_foot_z(env: SeedonStandingEnv, joint_positions: np.ndarray, side: str) -> float:
    data = mujoco.MjData(env.model)
    data.qpos[:] = env.data.qpos
    data.qvel[:] = env.data.qvel
    _set_joint_positions_in_data(env, data, joint_positions)
    mujoco.mj_forward(env.model, data)
    geom_id = env._foot_geom_ids[_side_index(side)]
    return float(data.geom_xpos[geom_id][2] - env.model.geom_size[geom_id][2])


def _ik_delta_for_foot_z(env: SeedonStandingEnv, swing_side: str, target_z: float, config: ControllerConfig) -> np.ndarray:
    """Return limited per-step joint delta from finite-difference foot-z Jacobian."""

    current = env._joint_positions()
    current_z = _foot_z(env, swing_side)
    dz_error = float(np.clip(target_z - current_z, -0.002, 0.002))
    joints = SWING_JOINTS[swing_side]
    jacobian = []
    for joint_index in joints:
        plus = current.copy()
        minus = current.copy()
        plus[joint_index] += config.ik_eps
        minus[joint_index] -= config.ik_eps
        dz = _kinematic_foot_z(env, plus, swing_side) - _kinematic_foot_z(env, minus, swing_side)
        jacobian.append(dz / max(2.0 * config.ik_eps, 1e-9))
    jac = np.asarray(jacobian, dtype=np.float64)
    denom = float(np.dot(jac, jac) + config.ik_damping)
    delta_local = jac * dz_error / max(denom, 1e-9)
    delta_local = np.clip(delta_local, -config.max_joint_delta_per_step, config.max_joint_delta_per_step)
    delta = np.zeros_like(current)
    for local_index, joint_index in enumerate(joints):
        delta[joint_index] = delta_local[local_index]
    return delta


def _next_swing_side(side: str) -> str:
    return "left" if side == "right" else "right"


def _enter_phase(runtime: ControllerRuntime, phase: LocomotionPhase, env: SeedonStandingEnv) -> None:
    runtime.phase = phase
    runtime.phase_step = 0
    if phase in {LocomotionPhase.SWING_LEFT, LocomotionPhase.SWING_RIGHT}:
        runtime.swing_start_z = _foot_z(env, runtime.swing_side)
    if phase in {LocomotionPhase.TOUCHDOWN_LEFT, LocomotionPhase.TOUCHDOWN_RIGHT}:
        runtime.touchdown_start_z = _foot_z(env, runtime.swing_side)
        runtime.touchdown_start_target = runtime.touchdown_start_z


def _phase_for(side: str, prefix: str) -> LocomotionPhase:
    return LocomotionPhase[f"{prefix}_{side.upper()}"]


def _controller_target(env: SeedonStandingEnv, runtime: ControllerRuntime, config: ControllerConfig) -> tuple[np.ndarray, float, str]:
    """Advance FSM logic and return current joint target and foot-z target."""

    target = env._nominal_joint_qpos.copy() if runtime.active_target is None else runtime.active_target.copy()
    foot_z_target = _foot_z(env, runtime.swing_side)
    event = ""

    if runtime.phase == LocomotionPhase.DOUBLE_SUPPORT:
        target = env._nominal_joint_qpos.copy()
        if runtime.phase_step >= config.double_support_steps:
            _enter_phase(runtime, _phase_for(runtime.swing_side, "PRELOAD"), env)
            event = "enter_preload"

    elif runtime.phase in {LocomotionPhase.PRELOAD_LEFT, LocomotionPhase.PRELOAD_RIGHT}:
        target = _preload_target(env, runtime.swing_side, config)
        if runtime.phase_step >= config.preload_steps:
            _enter_phase(runtime, _phase_for(runtime.swing_side, "UNLOAD"), env)
            event = "enter_unload"

    elif runtime.phase in {LocomotionPhase.UNLOAD_LEFT, LocomotionPhase.UNLOAD_RIGHT}:
        target = _preload_target(env, runtime.swing_side, config)
        if _swing_force(env, runtime.swing_side) <= config.contact_force_threshold_n:
            runtime.gate_reached = True
            _enter_phase(runtime, _phase_for(runtime.swing_side, "SWING"), env)
            event = "contact_gate_reached"
        elif runtime.phase_step >= config.unload_timeout_steps:
            runtime.gate_reached = False
            _enter_phase(runtime, _phase_for(runtime.swing_side, "TOUCHDOWN"), env)
            event = "contact_gate_timeout"

    elif runtime.phase in {LocomotionPhase.SWING_LEFT, LocomotionPhase.SWING_RIGHT}:
        alpha = _minimum_jerk(min(1.0, runtime.phase_step / max(config.swing_steps, 1)))
        foot_z_target = runtime.swing_start_z + config.target_clearance * alpha
        target = env._joint_positions() + _ik_delta_for_foot_z(env, runtime.swing_side, foot_z_target, config)
        if runtime.phase_step >= config.swing_steps:
            _enter_phase(runtime, _phase_for(runtime.swing_side, "TOUCHDOWN"), env)
            event = "enter_touchdown"

    elif runtime.phase in {LocomotionPhase.TOUCHDOWN_LEFT, LocomotionPhase.TOUCHDOWN_RIGHT}:
        if runtime.gate_reached:
            alpha = _minimum_jerk(min(1.0, runtime.phase_step / max(config.touchdown_steps, 1)))
            support_z = _foot_z(env, _support_side(runtime.swing_side))
            foot_z_target = runtime.touchdown_start_target + (support_z - runtime.touchdown_start_target) * alpha
            target = env._joint_positions() + _ik_delta_for_foot_z(env, runtime.swing_side, foot_z_target, config)
        else:
            target = _preload_target(env, runtime.swing_side, config)
        if runtime.phase_step >= config.touchdown_steps + config.hold_steps:
            runtime.swing_side = _next_swing_side(runtime.swing_side)
            runtime.cycle_count += 1
            runtime.gate_reached = False
            _enter_phase(runtime, LocomotionPhase.DOUBLE_SUPPORT, env)
            event = "cycle_complete"

    runtime.phase_step += 1
    runtime.active_target = env._apply_safe_joint_target_clamps(target)
    return runtime.active_target, foot_z_target, event


def _timeline_row(
    step: int,
    env: SeedonStandingEnv,
    runtime: ControllerRuntime,
    foot_z_target: float,
    event: str,
    baseline_weight: float,
) -> dict[str, Any]:
    swing_force = _swing_force(env, runtime.swing_side)
    support_force = _support_force(env, runtime.swing_side)
    total_force = swing_force + support_force
    return {
        "step": step,
        "phase": runtime.phase.value,
        "phase_step": runtime.phase_step,
        "swing_side": runtime.swing_side,
        "event": event,
        "swing_force": swing_force,
        "support_force": support_force,
        "support_ratio": support_force / max(total_force, 1e-9),
        "foot_z_target": foot_z_target,
        "foot_z_actual": _foot_z(env, runtime.swing_side),
        "clearance": _clearance(env, runtime.swing_side),
        "contact_state": _contact_state(env),
        "right_contact": bool(env._floor_contact_flags()["right"]),
        "left_contact": bool(env._floor_contact_flags()["left"]),
        "impact": total_force / max(baseline_weight, 1e-9),
        "base_height": env._base_height(),
        "base_drop": max(0.0, env._episode_base_height - env._base_height()),
        "upright": env._base_upright(),
        "base_roll": env._base_roll(),
        "base_pitch": env._base_pitch(),
        "ctrl_saturation": _ctrl_saturation(env),
        "force_saturation": _force_saturation(env),
    }


def _write_timeline(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_controller(
    config_path: Path,
    controller: ControllerConfig,
    *,
    out_dir: Path,
    steps: int,
    seed: int,
    warmup_steps: int,
    teacher_impact: float,
) -> ControllerAudit:
    """Run one controller rollout and return aggregate audit metrics."""

    env_config = _load_config(config_path)
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=env_config)
    runtime = ControllerRuntime()
    rows: list[dict[str, Any]] = []
    baseline_weight = 0.0
    gate_reached_count = 0
    gate_timeout_count = 0
    touchdown_spike_fail_count = 0
    try:
        env.reset(seed=seed)
        baseline_weight = float(np.sum(env.model.body_mass) * 9.81)
        runtime.active_target = env._nominal_joint_qpos.copy()
        for step in range(1, steps + 1):
            target, foot_z_target, event = _controller_target(env, runtime, controller)
            if event == "contact_gate_reached":
                gate_reached_count += 1
            elif event == "contact_gate_timeout":
                gate_timeout_count += 1
            previous_foot_z = _foot_z(env, runtime.swing_side)
            env._do_pd_simulation(target)
            env._gait_step += 1
            foot_velocity = abs(_foot_z(env, runtime.swing_side) - previous_foot_z) / max(env.dt, 1e-9)
            row = _timeline_row(step, env, runtime, foot_z_target, event, baseline_weight)
            if runtime.phase in {LocomotionPhase.TOUCHDOWN_LEFT, LocomotionPhase.TOUCHDOWN_RIGHT}:
                if foot_velocity > controller.touchdown_max_vertical_velocity:
                    row["event"] = f"{row['event']};touchdown_velocity_limit"
                if row["impact"] > teacher_impact * controller.impact_fail_multiplier:
                    touchdown_spike_fail_count += 1
                    row["event"] = f"{row['event']};touchdown_spike"
            rows.append(row)
    finally:
        env.close()

    timeline_path = out_dir / "timelines" / f"controller_v2_clearance_{controller.target_clearance:.4f}.csv"
    _write_timeline(timeline_path, rows)
    post = rows[warmup_steps:] or rows
    contact_none_steps = sum(1 for row in rows if row["contact_state"] == "none")
    jump_count = _count_contact_none_bursts(rows)
    min_upright = min((float(row["upright"]) for row in rows), default=0.0)
    base0 = float(post[0]["base_height"]) if post else 0.0
    base_drop = max(0.0, base0 - min((float(row["base_height"]) for row in post), default=base0))
    impact_post = max((float(row["impact"]) for row in post), default=0.0)
    max_force_post = impact_post * baseline_weight
    max_clearance = max((float(row["clearance"]) for row in rows), default=0.0)
    min_swing_force = min((float(row["swing_force"]) for row in rows), default=float("inf"))
    mean_support_ratio = float(np.mean([float(row["support_ratio"]) for row in rows])) if rows else 0.0
    ctrl_sat = max((float(row["ctrl_saturation"]) for row in rows), default=0.0)
    force_sat = max((float(row["force_saturation"]) for row in rows), default=0.0)
    phase1_fail: list[str] = []
    if contact_none_steps:
        phase1_fail.append("contact_none")
    if jump_count:
        phase1_fail.append("jump")
    if min_upright < 0.99:
        phase1_fail.append("upright")
    if base_drop > 0.015:
        phase1_fail.append("base_drop")
    if impact_post > teacher_impact * 1.2:
        phase1_fail.append("impact")
    phase1_passed = not phase1_fail
    phase2_passed = (
        phase1_passed
        and gate_reached_count > 0
        and max_clearance >= 0.0005
        and min_swing_force <= controller.contact_force_threshold_n
    )
    return ControllerAudit(
        target_clearance=controller.target_clearance,
        steps=len(rows),
        contact_none_ratio=contact_none_steps / max(1, len(rows)),
        jump_count=jump_count,
        min_upright=min_upright,
        base_drop_post=base_drop,
        impact_post=impact_post,
        max_contact_force_post=max_force_post,
        max_clearance=max_clearance,
        min_swing_force=min_swing_force,
        mean_support_ratio=mean_support_ratio,
        actuator_ctrl_saturation_max=ctrl_sat,
        actuator_force_saturation_max=force_sat,
        gate_reached_count=gate_reached_count,
        gate_timeout_count=gate_timeout_count,
        touchdown_spike_fail_count=touchdown_spike_fail_count,
        phase1_passed=phase1_passed,
        phase2_passed=phase2_passed,
        fail_reasons=",".join(phase1_fail),
        timeline_path=str(timeline_path),
    )


def write_results(path: Path, rows: list[ControllerAudit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(path: Path, rows: list[ControllerAudit], teacher_impact: float) -> None:
    lines = [
        "# Seedon Explicit Locomotion Controller V2",
        "",
        f"teacher_landing_impact_post_warmup: {teacher_impact:.6f}",
        f"runs: {len(rows)}",
        "",
        "| clearance | phase1 | phase2 | max_clearance | min_swing_force | impact | drop | upright | gates | timeouts | reasons |",
        "|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.target_clearance:.4f} | {row.phase1_passed} | {row.phase2_passed} | "
            f"{row.max_clearance:.6f} | {row.min_swing_force:.2f} | {row.impact_post:.3f} | "
            f"{row.base_drop_post:.5f} | {row.min_upright:.3f} | {row.gate_reached_count} | "
            f"{row.gate_timeout_count} | {row.fail_reasons} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vecnorm-path", type=Path, default=DEFAULT_VECNORM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-warmup-steps", type=int, default=20)
    parser.add_argument("--target-clearances", type=_parse_float_list, default="0.0005,0.001")
    parser.add_argument("--contact-force-threshold", type=float, default=20.0)
    parser.add_argument("--double-support-steps", type=int, default=40)
    parser.add_argument("--preload-steps", type=int, default=60)
    parser.add_argument("--unload-timeout-steps", type=int, default=120)
    parser.add_argument("--swing-steps", type=int, default=60)
    parser.add_argument("--touchdown-steps", type=int, default=80)
    parser.add_argument("--hold-steps", type=int, default=40)
    parser.add_argument("--support-hip-roll-limit", type=float, default=0.08)
    parser.add_argument("--swing-hip-roll-limit", type=float, default=0.035)
    parser.add_argument("--force-feedback-gain", type=float, default=0.001)
    parser.add_argument("--ik-eps", type=float, default=0.003)
    parser.add_argument("--ik-damping", type=float, default=1e-5)
    parser.add_argument("--max-joint-delta-per-step", type=float, default=0.004)
    parser.add_argument("--touchdown-max-vertical-velocity", type=float, default=0.03)
    parser.add_argument("--impact-fail-multiplier", type=float, default=1.2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    teacher = audit_shuffle(
        args.config,
        args.model_path,
        args.vecnorm_path,
        args.steps,
        args.seed,
        audit_warmup_steps=args.audit_warmup_steps,
    )
    rows: list[ControllerAudit] = []
    for clearance in args.target_clearances:
        controller = ControllerConfig(
            target_clearance=clearance,
            contact_force_threshold_n=args.contact_force_threshold,
            double_support_steps=args.double_support_steps,
            preload_steps=args.preload_steps,
            unload_timeout_steps=args.unload_timeout_steps,
            swing_steps=args.swing_steps,
            touchdown_steps=args.touchdown_steps,
            hold_steps=args.hold_steps,
            support_hip_roll_limit=args.support_hip_roll_limit,
            swing_hip_roll_limit=args.swing_hip_roll_limit,
            force_feedback_gain=args.force_feedback_gain,
            ik_eps=args.ik_eps,
            ik_damping=args.ik_damping,
            max_joint_delta_per_step=args.max_joint_delta_per_step,
            touchdown_max_vertical_velocity=args.touchdown_max_vertical_velocity,
            impact_fail_multiplier=args.impact_fail_multiplier,
        )
        row = run_controller(
            args.config,
            controller,
            out_dir=args.out_dir,
            steps=args.steps,
            seed=args.seed,
            warmup_steps=args.audit_warmup_steps,
            teacher_impact=teacher.landing_impact_post_warmup,
        )
        rows.append(row)
        print(
            f"clearance={clearance:.4f} phase1={row.phase1_passed} phase2={row.phase2_passed} "
            f"max_clear={row.max_clearance:.6f} min_swing_force={row.min_swing_force:.2f} "
            f"impact={row.impact_post:.3f} drop={row.base_drop_post:.5f} upright={row.min_upright:.3f} "
            f"gates={row.gate_reached_count} timeouts={row.gate_timeout_count} reasons={row.fail_reasons or '-'}"
        )
    write_results(args.out_dir / "seedon_explicit_locomotion_controller_v2.csv", rows)
    write_summary(args.out_dir / "summary.md", rows, teacher.landing_impact_post_warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
