"""Audit pose_1..4 smooth reference with dynamic PD and medium lateral assist."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from sedon_baseline.env import (
    SedonStandingConfig,
    SedonStandingEnv,
    load_reference_gait_seed,
)
from tools.sedon_debug_common import (
    DEFAULT_SCENE_PATH,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    geom_name,
    require_scene,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_PATH = REPO_ROOT / "configs" / "sedon" / "reference_march_pose_1_4_mirrored_seed.json"
MEDIUM_LATERAL_ASSIST_N = 10.0


@dataclass(frozen=True)
class ReferencePdAudit:
    """Aggregated deterministic reference tracking diagnostics."""

    steps: int
    peak_support_ratio: float
    contact_none: bool
    contact_none_ratio: float
    jump_count: int
    clearance: float
    landing_impact: float
    tracking_error: float
    max_tracking_error: float
    base_height: float
    min_base_height: float
    upright: float
    min_upright: float
    both_contact_ratio: float


def _smoothstep(phase: float) -> float:
    """Return smoothstep interpolation alpha in [0, 1]."""
    phase = float(np.clip(phase, 0.0, 1.0))
    return phase * phase * (3.0 - 2.0 * phase)


def _reference_target_at_step(seed_path: Path, step: int) -> tuple[np.ndarray, str]:
    """Return smooth interpolated absolute joint target and support hint."""
    seed = load_reference_gait_seed(seed_path)
    phase_step = step % seed.cycle_steps
    cursor = 0
    for index, keyframe in enumerate(seed.keyframes):
        duration = max(1, keyframe.duration_steps)
        if phase_step < cursor + duration:
            local_step = phase_step - cursor
            next_keyframe = seed.keyframes[(index + 1) % len(seed.keyframes)]
            alpha = _smoothstep(local_step / max(1, duration - 1))
            target = (
                (1.0 - alpha) * keyframe.joint_targets
                + alpha * next_keyframe.joint_targets
            )
            return target, keyframe.support_mode
        cursor += duration
    last = seed.keyframes[-1]
    return last.joint_targets.copy(), last.support_mode


def _foot_contact_forces(env: SedonStandingEnv) -> tuple[float, float]:
    """Return floor contact normal force for left and right feet."""
    left = 0.0
    right = 0.0
    wrench = np.zeros(6, dtype=np.float64)
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        name_a = geom_name(env.model, int(contact.geom1))
        name_b = geom_name(env.model, int(contact.geom2))
        pair = {name_a, name_b}
        if pair not in ({FLOOR_GEOM, LEFT_FOOT_GEOM}, {FLOOR_GEOM, RIGHT_FOOT_GEOM}):
            continue
        mujoco.mj_contactForce(env.model, env.data, contact_index, wrench)
        normal_force = abs(float(wrench[0]))
        if pair == {FLOOR_GEOM, LEFT_FOOT_GEOM}:
            left += normal_force
        else:
            right += normal_force
    return left, right


def _contact_state(left_force: float, right_force: float) -> str:
    """Classify foot-floor load state."""
    left_contact = left_force > 5.0
    right_contact = right_force > 5.0
    if left_contact and right_contact:
        return "both"
    if left_contact:
        return "left"
    if right_contact:
        return "right"
    return "none"


def _count_none_bursts(states: list[str]) -> int:
    """Count contiguous no-contact bursts."""
    bursts = 0
    in_burst = False
    for state in states:
        if state == "none" and not in_burst:
            bursts += 1
            in_burst = True
        elif state != "none":
            in_burst = False
    return bursts


def _assist_force_y(support_hint: str) -> float:
    """Return medium lateral assist direction for the designated support side."""
    if support_hint == "right":
        return -MEDIUM_LATERAL_ASSIST_N
    if support_hint == "left":
        return MEDIUM_LATERAL_ASSIST_N
    return 0.0


def _step_pd_with_lateral_assist(
    env: SedonStandingEnv,
    target: np.ndarray,
    support_hint: str,
) -> None:
    """Advance one control step with PD tracking and virtual lateral assist."""
    base_id = env._base_body_id
    force_y = _assist_force_y(support_hint)
    clamped_target = env._apply_safe_joint_target_clamps(target)
    for _ in range(env.frame_skip):
        env.data.ctrl[:] = env._pd_control(clamped_target)
        if force_y:
            env.data.xfrc_applied[base_id, 1] = force_y
        mujoco.mj_step(env.model, env.data)
        env.data.xfrc_applied[base_id, :] = 0.0


def audit_reference_pd(
    scene_path: Path,
    seed_path: Path,
    *,
    steps: int,
    reference_scale: float,
) -> ReferencePdAudit:
    """Run deterministic dynamic PD tracking for the grounded pose_1..4 reference."""
    if steps <= 0:
        raise ValueError("--steps must be positive.")
    if reference_scale < 0.0:
        raise ValueError("--reference-scale must be non-negative.")

    config = SedonStandingConfig(
        task_mode="reference_march",
        target_forward_velocity=0.0,
        march_forward_velocity=0.0,
        march_forward_progress_weight=0.0,
        march_forward_velocity_weight=0.0,
        march_swing_forward_weight=0.0,
        reference_gait_seed_path=str(seed_path),
        reference_gait_seed_scale=reference_scale,
        march_require_both_contact=False,
        march_contact_none_terminate_steps=0,
    )
    env = SedonStandingEnv(
        scene_path=require_scene(scene_path),
        reset_noise_scale=0.0,
        reward_config=config,
    )
    env.reset(seed=0)
    initial_bottoms = env._foot_bottom_heights()
    total_weight = float(np.sum(env.model.body_mass) * 9.81)

    support_ratios: list[float] = []
    contact_states: list[str] = []
    clearances: list[float] = []
    impacts: list[float] = []
    tracking_errors: list[float] = []
    base_heights: list[float] = []
    uprights: list[float] = []
    try:
        for step in range(steps):
            raw_target, support_hint = _reference_target_at_step(seed_path, step)
            target = env._nominal_joint_qpos + reference_scale * (
                raw_target - env._nominal_joint_qpos
            )
            _step_pd_with_lateral_assist(env, target, support_hint)
            env._gait_step += 1

            left_force, right_force = _foot_contact_forces(env)
            total_force = left_force + right_force
            if support_hint == "left":
                support_ratio = left_force / (total_force + 1e-6)
                clearance = max(0.0, float(env._foot_bottom_heights()[0] - initial_bottoms[0]))
            elif support_hint == "right":
                support_ratio = right_force / (total_force + 1e-6)
                clearance = max(0.0, float(env._foot_bottom_heights()[1] - initial_bottoms[1]))
            else:
                support_ratio = 0.5 if total_force > 0.0 else 0.0
                clearance = max(0.0, float(np.max(env._foot_bottom_heights() - initial_bottoms)))

            support_ratios.append(float(support_ratio))
            contact_states.append(_contact_state(left_force, right_force))
            clearances.append(float(clearance))
            impacts.append(float(total_force / max(total_weight, 1e-6)))
            tracking_errors.append(float(np.mean(np.abs(env._joint_positions() - target))))
            base_heights.append(float(env._base_height()))
            uprights.append(float(env._base_upright()))
    finally:
        env.close()

    none_steps = sum(1 for state in contact_states if state == "none")
    both_steps = sum(1 for state in contact_states if state == "both")
    return ReferencePdAudit(
        steps=steps,
        peak_support_ratio=float(max(support_ratios, default=0.0)),
        contact_none=bool(none_steps > 0),
        contact_none_ratio=float(none_steps / max(1, len(contact_states))),
        jump_count=_count_none_bursts(contact_states),
        clearance=float(max(clearances, default=0.0)),
        landing_impact=float(max(impacts, default=0.0)),
        tracking_error=float(np.mean(tracking_errors)) if tracking_errors else 0.0,
        max_tracking_error=float(max(tracking_errors, default=0.0)),
        base_height=float(base_heights[-1]) if base_heights else 0.0,
        min_base_height=float(min(base_heights, default=0.0)),
        upright=float(uprights[-1]) if uprights else 0.0,
        min_upright=float(min(uprights, default=0.0)),
        both_contact_ratio=float(both_steps / max(1, len(contact_states))),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--seed-path", type=Path, default=DEFAULT_SEED_PATH)
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument(
        "--reference-scale",
        type=float,
        default=0.5,
        help="Scale pose_1..4 absolute targets toward nominal stance.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run deterministic reference PD audit."""
    args = build_parser().parse_args(argv)
    summary = audit_reference_pd(
        args.scene,
        args.seed_path,
        steps=args.steps,
        reference_scale=args.reference_scale,
    )
    for key, value in asdict(summary).items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
