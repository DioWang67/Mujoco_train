"""Evaluate whether assisted deterministic shuffle can reduce lateral assist."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from seedon_baseline.env import (
    ReferenceGaitSeed,
    SeedonStandingConfig,
    SeedonStandingEnv,
    load_reference_gait_seed,
)
from tools.seedon_debug_common import (
    DEBUG_OUT_DIR,
    DEFAULT_SCENE_PATH,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    geom_name,
    require_scene,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_PATH = REPO_ROOT / "configs" / "seedon" / "reference_march_pose_1_4_mirrored_seed.json"
DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "assisted_shuffle_curriculum_eval.csv"
DEFAULT_ASSIST_FORCES = (10.0, 8.0, 6.0, 4.0, 2.0, 0.0)
CONTACT_FORCE_THRESHOLD_N = 5.0


@dataclass(frozen=True)
class AssistedShuffleRow:
    """One assisted deterministic shuffle evaluation row."""

    assist_force_n: float
    peak_support_ratio: float
    min_swing_ratio: float
    clearance: float
    contact_none_ratio: float
    jump_count: int
    landing_impact: float
    tracking_error: float
    base_height_drop: float
    upright: float
    both_contact_ratio: float
    passed: bool


def _smoothstep(phase: float) -> float:
    """Return smoothstep interpolation alpha in [0, 1]."""
    phase = float(np.clip(phase, 0.0, 1.0))
    return phase * phase * (3.0 - 2.0 * phase)


def _reference_target_at_step(seed: ReferenceGaitSeed, step: int) -> tuple[np.ndarray, str]:
    """Return smooth interpolated absolute joint target and support hint."""
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


def _foot_contact_forces(env: SeedonStandingEnv) -> tuple[float, float]:
    """Return left/right foot-floor contact normal forces."""
    left_force = 0.0
    right_force = 0.0
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
            left_force += normal_force
        else:
            right_force += normal_force
    return left_force, right_force


def _contact_state(left_force: float, right_force: float) -> str:
    """Classify foot-floor contact state from normal forces."""
    left_contact = left_force > CONTACT_FORCE_THRESHOLD_N
    right_contact = right_force > CONTACT_FORCE_THRESHOLD_N
    if left_contact and right_contact:
        return "both"
    if left_contact:
        return "left"
    if right_contact:
        return "right"
    return "none"


def _count_none_bursts(states: list[str]) -> int:
    """Count contiguous no-contact bursts."""
    count = 0
    in_burst = False
    for state in states:
        if state == "none" and not in_burst:
            count += 1
            in_burst = True
        elif state != "none":
            in_burst = False
    return count


def _assist_force_y(support_hint: str, assist_force_n: float) -> float:
    """Return lateral assist direction for the designated support side."""
    if support_hint == "right":
        return -assist_force_n
    if support_hint == "left":
        return assist_force_n
    return 0.0


def _step_pd_with_assist(
    env: SeedonStandingEnv,
    target: np.ndarray,
    support_hint: str,
    assist_force_n: float,
) -> None:
    """Advance one control step with dynamic PD tracking and lateral assist."""
    force_y = _assist_force_y(support_hint, assist_force_n)
    base_id = env._base_body_id
    clamped_target = env._apply_safe_joint_target_clamps(target)
    for _ in range(env.frame_skip):
        env.data.ctrl[:] = env._pd_control(clamped_target)
        if force_y:
            env.data.xfrc_applied[base_id, 1] = force_y
        mujoco.mj_step(env.model, env.data)
        env.data.xfrc_applied[base_id, :] = 0.0


def _evaluate_assist_force(
    scene_path: Path,
    seed: ReferenceGaitSeed,
    seed_path: Path,
    *,
    assist_force_n: float,
    steps: int,
    reference_scale: float,
) -> AssistedShuffleRow:
    """Evaluate one lateral assist magnitude."""
    config = SeedonStandingConfig(
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
    env = SeedonStandingEnv(
        scene_path=scene_path,
        reset_noise_scale=0.0,
        reward_config=config,
    )
    env.reset(seed=0)
    initial_bottoms = env._foot_bottom_heights()
    initial_base_height = float(env._base_height())
    total_weight = float(np.sum(env.model.body_mass) * 9.81)

    support_ratios: list[float] = []
    swing_ratios: list[float] = []
    clearances: list[float] = []
    contact_states: list[str] = []
    landing_impacts: list[float] = []
    tracking_errors: list[float] = []
    base_heights: list[float] = []
    uprights: list[float] = []
    try:
        for step in range(steps):
            raw_target, support_hint = _reference_target_at_step(seed, step)
            target = env._nominal_joint_qpos + reference_scale * (
                raw_target - env._nominal_joint_qpos
            )
            _step_pd_with_assist(env, target, support_hint, assist_force_n)
            env._gait_step += 1

            left_force, right_force = _foot_contact_forces(env)
            total_force = left_force + right_force
            left_ratio = left_force / (total_force + 1e-6)
            right_ratio = right_force / (total_force + 1e-6)
            foot_bottoms = env._foot_bottom_heights()
            if support_hint == "left":
                support_ratio = left_ratio
                swing_ratio = right_ratio
                clearance = max(0.0, float(foot_bottoms[0] - initial_bottoms[0]))
            elif support_hint == "right":
                support_ratio = right_ratio
                swing_ratio = left_ratio
                clearance = max(0.0, float(foot_bottoms[1] - initial_bottoms[1]))
            else:
                support_ratio = 0.5 if total_force > 0.0 else 0.0
                swing_ratio = support_ratio
                clearance = max(0.0, float(np.max(foot_bottoms - initial_bottoms)))

            support_ratios.append(float(support_ratio))
            swing_ratios.append(float(swing_ratio))
            clearances.append(float(clearance))
            contact_states.append(_contact_state(left_force, right_force))
            landing_impacts.append(float(total_force / max(total_weight, 1e-6)))
            tracking_errors.append(float(np.mean(np.abs(env._joint_positions() - target))))
            base_heights.append(float(env._base_height()))
            uprights.append(float(env._base_upright()))
    finally:
        env.close()

    none_steps = sum(1 for state in contact_states if state == "none")
    both_steps = sum(1 for state in contact_states if state == "both")
    jump_count = _count_none_bursts(contact_states)
    contact_none_ratio = none_steps / max(1, len(contact_states))
    clearance = float(max(clearances, default=0.0))
    peak_support_ratio = float(max(support_ratios, default=0.0))
    upright = float(min(uprights, default=0.0))
    passed = (
        contact_none_ratio == 0.0
        and jump_count == 0
        and clearance >= 0.0008
        and peak_support_ratio >= 0.55
        and upright >= 0.99
    )
    return AssistedShuffleRow(
        assist_force_n=float(assist_force_n),
        peak_support_ratio=peak_support_ratio,
        min_swing_ratio=float(min(swing_ratios, default=0.0)),
        clearance=clearance,
        contact_none_ratio=float(contact_none_ratio),
        jump_count=jump_count,
        landing_impact=float(max(landing_impacts, default=0.0)),
        tracking_error=float(np.mean(tracking_errors)) if tracking_errors else 0.0,
        base_height_drop=float(initial_base_height - min(base_heights, default=initial_base_height)),
        upright=upright,
        both_contact_ratio=float(both_steps / max(1, len(contact_states))),
        passed=passed,
    )


def _parse_assist_forces(raw_value: str) -> tuple[float, ...]:
    """Parse comma-separated assist force magnitudes."""
    forces: list[float] = []
    for raw_part in raw_value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        force = float(part)
        if force < 0.0:
            raise argparse.ArgumentTypeError("assist forces must be non-negative.")
        forces.append(force)
    if not forces:
        raise argparse.ArgumentTypeError("at least one assist force is required.")
    return tuple(forces)


def _write_csv(path: Path, rows: list[AssistedShuffleRow]) -> None:
    """Write evaluation rows to CSV."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def run_eval(
    scene_path: Path,
    seed_path: Path,
    out_csv: Path,
    *,
    assist_forces: tuple[float, ...],
    steps: int,
    reference_scale: float,
) -> list[AssistedShuffleRow]:
    """Run assisted shuffle curriculum evaluation."""
    if steps <= 0:
        raise ValueError("--steps must be positive.")
    if reference_scale < 0.0:
        raise ValueError("--reference-scale must be non-negative.")
    resolved_scene = require_scene(scene_path)
    if not seed_path.is_file():
        raise FileNotFoundError(f"Reference seed not found: {seed_path}")
    seed = load_reference_gait_seed(seed_path)

    rows = [
        _evaluate_assist_force(
            resolved_scene,
            seed,
            seed_path,
            assist_force_n=force,
            steps=steps,
            reference_scale=reference_scale,
        )
        for force in assist_forces
    ]
    _write_csv(out_csv, rows)
    print(f"wrote rows to {out_csv}")
    for row in rows:
        print(
            f"assist={row.assist_force_n:>4.1f}N "
            f"peak={row.peak_support_ratio:.3f} swing_min={row.min_swing_ratio:.3f} "
            f"clearance={row.clearance:.6f} none={row.contact_none_ratio:.3f} "
            f"jumps={row.jump_count} impact={row.landing_impact:.3f} "
            f"track={row.tracking_error:.5f} drop={row.base_height_drop:.5f} "
            f"upright={row.upright:.3f} both={row.both_contact_ratio:.3f} "
            f"pass={row.passed}"
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--seed-path", type=Path, default=DEFAULT_SEED_PATH)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--reference-scale", type=float, default=0.5)
    parser.add_argument(
        "--assist-forces",
        type=_parse_assist_forces,
        default=DEFAULT_ASSIST_FORCES,
        help="Comma-separated assist force magnitudes, e.g. 10,8,6,4,2,0.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run assisted shuffle curriculum evaluation."""
    args = build_parser().parse_args(argv)
    run_eval(
        args.scene,
        args.seed_path,
        args.out_csv,
        assist_forces=args.assist_forces,
        steps=args.steps,
        reference_scale=args.reference_scale,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
