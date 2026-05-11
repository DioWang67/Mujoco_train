"""Audit Sedon gait rollouts for contact-safe reverse-knee shaping.

This tool is intentionally diagnostic rather than a training entrypoint. It
checks whether a rollout is progressing through lateral load transfer, swing
foot unload, and micro-lift without relying on foot collision artifacts.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from sedon_baseline.env import DEFAULT_SCENE_PATH, SedonStandingConfig, SedonStandingEnv
from sedon_baseline.env import load_sedon_config_from_env
from tools.sedon_debug_common import (
    BASE_PROXY_GEOM,
    DEBUG_OUT_DIR,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    contact_pairs,
    geom_id,
    require_scene,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "gait_audit.csv"
FOOT_GEOM_BY_SIDE = {"left": LEFT_FOOT_GEOM, "right": RIGHT_FOOT_GEOM}
FOOT_INDEX_BY_SIDE = {"right": 0, "left": 1}
KNEE_INDEX_BY_SIDE = {"right": 3, "left": 8}
MICRO_LIFT_MIN_M = 0.001
MICRO_LIFT_MAX_M = 0.010
SUPPORT_FORCE_GATE = 0.65
SWING_FORCE_GATE = 0.25


@dataclass(frozen=True)
class AuditStep:
    """Per-step Sedon gait audit metrics written to CSV."""

    step: int
    reward: float
    terminated: bool
    truncated: bool
    phase_name: str
    support_side_hint: str
    left_foot_contact: bool
    right_foot_contact: bool
    base_proxy_contact: bool
    foot_foot_collision: bool
    left_normal_force: float
    right_normal_force: float
    left_force_fraction: float
    right_force_fraction: float
    contact_state: str
    inferred_support_side: str
    swing_unload_success: bool
    micro_lift_success: bool
    left_foot_clearance: float
    right_foot_clearance: float
    swing_foot_clearance: float
    left_foot_forward_delta: float
    right_foot_forward_delta: float
    swing_foot_forward_delta: float
    left_knee_pitch: float
    right_knee_pitch: float
    swing_knee_pitch: float
    stance_knee_pitch: float
    swing_knee_positive: bool
    stance_knee_collapsed: bool
    com_x: float
    com_y: float
    com_z: float
    base_height: float
    base_roll: float
    base_pitch: float
    upright: float
    lateral_com_shift: float
    vertical_com_delta: float
    action_rms: float
    torque_rms: float
    joint_velocity_rms: float
    excessive_action: bool
    excessive_torque: bool


@dataclass(frozen=True)
class AuditSummary:
    """Aggregate gait audit summary printed after rollout."""

    mode: str
    steps: int
    terminated: bool
    diagnosis: str
    left_support_duration: int
    right_support_duration: int
    double_support_ratio: float
    single_support_ratio: float
    no_contact_ratio: float
    swing_unload_success_count: int
    micro_lift_success_count: int
    max_clearance: float
    max_left_clearance: float
    max_right_clearance: float
    max_lateral_com_shift: float
    vertical_com_oscillation: float
    mean_base_height: float
    min_base_height: float
    mean_upright: float
    base_proxy_contact_steps: int
    foot_foot_collision_steps: int
    left_swing_knee_positive_steps: int
    right_swing_knee_positive_steps: int
    max_left_knee_pitch: float
    max_right_knee_pitch: float
    swing_knee_positive_micro_lift_count: int
    mean_action_rms: float
    mean_torque_rms: float
    mean_joint_velocity_rms: float


def _standing_config(config: SedonStandingConfig) -> SedonStandingConfig:
    """Return a config with gait seed disabled for standing-only audit."""
    return replace(
        config,
        target_forward_velocity=max(config.target_forward_velocity, 1e-6),
        gait_hip_roll_amp=0.0,
        gait_hip_pitch_amp=0.0,
        gait_knee_pitch_amp=0.0,
        gait_ankle_pitch_amp=0.0,
        fsm_right_swing_scale=0.0,
        fsm_left_swing_scale=0.0,
        fsm_right_support_roll_scale=0.0,
        fsm_left_support_roll_scale=0.0,
    )


def _overall_com(env: SedonStandingEnv) -> np.ndarray:
    """Return the whole-body COM in world coordinates."""
    masses = env.model.body_mass
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Sedon model has no positive body mass.")
    return np.sum(env.data.xipos * masses[:, None], axis=0) / total_mass


def _base_roll_pitch(env: SedonStandingEnv) -> tuple[float, float]:
    """Return approximate base roll and pitch from the base rotation matrix."""
    xmat = env.data.xmat[env._base_body_id].reshape(3, 3)
    roll = float(np.arctan2(xmat[2, 1], xmat[2, 2]))
    pitch = float(np.arctan2(-xmat[2, 0], np.sqrt(xmat[2, 1] ** 2 + xmat[2, 2] ** 2)))
    return roll, pitch


def _foot_floor_load(env: SedonStandingEnv, side: str) -> tuple[bool, float]:
    """Return whether a foot contacts the floor and its summed normal force."""
    foot_geom_name = FOOT_GEOM_BY_SIDE[side]
    normal_force_sum = 0.0
    contact_found = False
    wrench = np.zeros(6, dtype=np.float64)
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        name_a = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        name_b = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        if {name_a, name_b} != {FLOOR_GEOM, foot_geom_name}:
            continue
        mujoco.mj_contactForce(env.model, env.data, contact_index, wrench)
        contact_found = True
        normal_force_sum += abs(float(wrench[0]))
    return contact_found, normal_force_sum


def _contact_flags(env: SedonStandingEnv) -> tuple[bool, bool, bool, bool]:
    """Return left, right, base-proxy, and foot-foot contact flags."""
    pair_sets = [set((name_a, name_b)) for name_a, name_b, _ in contact_pairs(env.model, env.data)]
    left_contact = {FLOOR_GEOM, LEFT_FOOT_GEOM} in pair_sets
    right_contact = {FLOOR_GEOM, RIGHT_FOOT_GEOM} in pair_sets
    base_proxy_contact = {FLOOR_GEOM, BASE_PROXY_GEOM} in pair_sets
    foot_foot_collision = {LEFT_FOOT_GEOM, RIGHT_FOOT_GEOM} in pair_sets
    return left_contact, right_contact, base_proxy_contact, foot_foot_collision


def _contact_state(left_contact: bool, right_contact: bool) -> str:
    """Return a compact foot-floor contact state."""
    if left_contact and right_contact:
        return "both"
    if left_contact:
        return "left_only"
    if right_contact:
        return "right_only"
    return "none"


def _infer_support_side(left_fraction: float, right_fraction: float) -> str:
    """Infer the current support side from normal force fractions."""
    if left_fraction >= SUPPORT_FORCE_GATE and right_fraction <= SWING_FORCE_GATE:
        return "left"
    if right_fraction >= SUPPORT_FORCE_GATE and left_fraction <= SWING_FORCE_GATE:
        return "right"
    return "none"


def _policy_action_provider(model_path: Path, vecnorm_path: Path | None, env: SedonStandingEnv):
    """Return a callable that predicts actions with an optional SB3 VecNormalize."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Policy audit requires stable-baselines3. Use --mode zero or --mode scripted "
            "when SB3 is not installed."
        ) from exc

    if not model_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {model_path}")
    model = PPO.load(str(model_path))
    if vecnorm_path is None:
        return lambda obs: model.predict(obs, deterministic=True)[0]
    if not vecnorm_path.is_file():
        raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_path}")

    vec_env = DummyVecEnv(
        [
            lambda: SedonStandingEnv(
                scene_path=env._scene_path,
                reset_noise_scale=0.0,
                reward_config=env._reward_config,
            )
        ]
    )
    vecnorm = VecNormalize.load(str(vecnorm_path), vec_env)
    vecnorm.training = False
    vecnorm.norm_reward = False

    def _predict(obs: np.ndarray) -> np.ndarray:
        norm_obs = vecnorm.normalize_obs(obs[None, :])
        action, _ = model.predict(norm_obs, deterministic=True)
        return np.asarray(action[0], dtype=np.float64)

    return _predict


def _sample_step(
    env: SedonStandingEnv,
    *,
    step: int,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict,
    action: np.ndarray,
    initial_left_foot_z: float,
    initial_right_foot_z: float,
    initial_left_foot_x: float,
    initial_right_foot_x: float,
    initial_com: np.ndarray,
) -> AuditStep:
    """Collect one gait audit sample after an environment step."""
    left_contact, right_contact, base_proxy_contact, foot_foot_collision = _contact_flags(env)
    left_force_contact, left_normal_force = _foot_floor_load(env, "left")
    right_force_contact, right_normal_force = _foot_floor_load(env, "right")
    left_contact = left_contact or left_force_contact
    right_contact = right_contact or right_force_contact
    total_force = left_normal_force + right_normal_force
    left_fraction = left_normal_force / total_force if total_force > 1e-9 else 0.0
    right_fraction = right_normal_force / total_force if total_force > 1e-9 else 0.0

    foot_bottoms = env._foot_bottom_heights()
    left_foot_id = geom_id(env.model, LEFT_FOOT_GEOM)
    right_foot_id = geom_id(env.model, RIGHT_FOOT_GEOM)
    left_clearance = float(foot_bottoms[FOOT_INDEX_BY_SIDE["left"]] - initial_left_foot_z)
    right_clearance = float(foot_bottoms[FOOT_INDEX_BY_SIDE["right"]] - initial_right_foot_z)
    left_forward_delta = float(env.data.geom_xpos[left_foot_id][0] - initial_left_foot_x)
    right_forward_delta = float(env.data.geom_xpos[right_foot_id][0] - initial_right_foot_x)

    support_side = _infer_support_side(left_fraction, right_fraction)
    swing_side = "right" if support_side == "left" else "left" if support_side == "right" else "none"
    swing_clearance = 0.0
    swing_forward_delta = 0.0
    swing_knee_pitch = 0.0
    stance_knee_pitch = 0.0
    if swing_side == "left":
        swing_clearance = left_clearance
        swing_forward_delta = left_forward_delta
        swing_knee_pitch = float(info.get("left_knee_qpos", np.nan))
        stance_knee_pitch = float(info.get("right_knee_qpos", np.nan))
    elif swing_side == "right":
        swing_clearance = right_clearance
        swing_forward_delta = right_forward_delta
        swing_knee_pitch = float(info.get("right_knee_qpos", np.nan))
        stance_knee_pitch = float(info.get("left_knee_qpos", np.nan))

    com = _overall_com(env)
    base_roll, base_pitch = _base_roll_pitch(env)
    action_rms = float(np.sqrt(np.mean(np.square(action)))) if action.size else 0.0
    torque_rms = float(np.sqrt(np.mean(np.square(env.data.ctrl)))) if env.data.ctrl.size else 0.0
    joint_velocity_rms = float(np.sqrt(np.mean(np.square(env._joint_velocities()))))
    contact_state = _contact_state(left_contact, right_contact)
    swing_unload_success = (
        support_side != "none"
        and ((support_side == "left" and right_fraction < SWING_FORCE_GATE)
             or (support_side == "right" and left_fraction < SWING_FORCE_GATE))
    )
    swing_knee_positive = bool(swing_side != "none" and swing_knee_pitch > 0.0)
    micro_lift_success = (
        swing_unload_success
        and MICRO_LIFT_MIN_M <= swing_clearance <= MICRO_LIFT_MAX_M
        and swing_knee_positive
        and not base_proxy_contact
        and not foot_foot_collision
    )

    return AuditStep(
        step=step,
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
        phase_name=str(info.get("phase_name", "")),
        support_side_hint=str(info.get("support_side", "")),
        left_foot_contact=left_contact,
        right_foot_contact=right_contact,
        base_proxy_contact=base_proxy_contact,
        foot_foot_collision=foot_foot_collision,
        left_normal_force=left_normal_force,
        right_normal_force=right_normal_force,
        left_force_fraction=left_fraction,
        right_force_fraction=right_fraction,
        contact_state=contact_state,
        inferred_support_side=support_side,
        swing_unload_success=swing_unload_success,
        micro_lift_success=micro_lift_success,
        left_foot_clearance=left_clearance,
        right_foot_clearance=right_clearance,
        swing_foot_clearance=swing_clearance,
        left_foot_forward_delta=left_forward_delta,
        right_foot_forward_delta=right_forward_delta,
        swing_foot_forward_delta=swing_forward_delta,
        left_knee_pitch=float(info.get("left_knee_qpos", np.nan)),
        right_knee_pitch=float(info.get("right_knee_qpos", np.nan)),
        swing_knee_pitch=swing_knee_pitch,
        stance_knee_pitch=stance_knee_pitch,
        swing_knee_positive=swing_knee_positive,
        stance_knee_collapsed=bool(support_side != "none" and stance_knee_pitch < -0.25),
        com_x=float(com[0]),
        com_y=float(com[1]),
        com_z=float(com[2]),
        base_height=float(info.get("base_height", env._base_height())),
        base_roll=base_roll,
        base_pitch=base_pitch,
        upright=float(info.get("upright", env._base_upright())),
        lateral_com_shift=float(com[1] - initial_com[1]),
        vertical_com_delta=float(com[2] - initial_com[2]),
        action_rms=action_rms,
        torque_rms=torque_rms,
        joint_velocity_rms=joint_velocity_rms,
        excessive_action=bool(action_rms > 0.8),
        excessive_torque=bool(torque_rms > 85.0),
    )


def _diagnose(rows: list[AuditStep], terminated: bool) -> str:
    """Classify a Sedon rollout using conservative gait-shaping gates."""
    if not rows:
        return "unstable_or_terminated"
    if any(row.foot_foot_collision for row in rows):
        return "foot_collision_contaminated"
    if any(row.base_proxy_contact for row in rows):
        return "base_proxy_contact_failure"
    if terminated:
        return "unstable_or_terminated"
    max_clearance = max(0.0, max(max(row.left_foot_clearance, row.right_foot_clearance) for row in rows))
    min_base_height = min(row.base_height for row in rows)
    vertical_com_oscillation = max(row.com_z for row in rows) - min(row.com_z for row in rows)
    has_wrong_direction_lift = any(
        row.swing_foot_clearance > MICRO_LIFT_MAX_M and not row.swing_knee_positive
        for row in rows
    )
    if max_clearance > 0.04 and vertical_com_oscillation > 0.04:
        return "fake_lift_by_bounce_or_penetration"
    if has_wrong_direction_lift:
        return "fake_lift_by_bounce_or_penetration"
    if any(row.micro_lift_success for row in rows):
        return "micro_lift_success"
    if any(row.swing_unload_success for row in rows):
        return "swing_unload_success"
    if any(
        row.left_force_fraction >= SUPPORT_FORCE_GATE
        or row.right_force_fraction >= SUPPORT_FORCE_GATE
        for row in rows
    ):
        return "load_transfer_success"
    if min_base_height >= 0.40 and np.mean([row.upright for row in rows]) >= 0.95:
        return "stable_standing_only"
    return "unstable_or_terminated"


def _summarize(mode: str, rows: list[AuditStep], terminated: bool) -> AuditSummary:
    """Build aggregate gait metrics from per-step audit rows."""
    steps = len(rows)
    if steps == 0:
        return AuditSummary(
            mode=mode,
            steps=0,
            terminated=terminated,
            diagnosis="unstable_or_terminated",
            left_support_duration=0,
            right_support_duration=0,
            double_support_ratio=0.0,
            single_support_ratio=0.0,
            no_contact_ratio=0.0,
            swing_unload_success_count=0,
            micro_lift_success_count=0,
            max_clearance=0.0,
            max_left_clearance=0.0,
            max_right_clearance=0.0,
            max_lateral_com_shift=0.0,
            vertical_com_oscillation=0.0,
            mean_base_height=0.0,
            min_base_height=0.0,
            mean_upright=0.0,
            base_proxy_contact_steps=0,
            foot_foot_collision_steps=0,
            left_swing_knee_positive_steps=0,
            right_swing_knee_positive_steps=0,
            max_left_knee_pitch=0.0,
            max_right_knee_pitch=0.0,
            swing_knee_positive_micro_lift_count=0,
            mean_action_rms=0.0,
            mean_torque_rms=0.0,
            mean_joint_velocity_rms=0.0,
        )
    left_support_duration = sum(row.inferred_support_side == "left" for row in rows)
    right_support_duration = sum(row.inferred_support_side == "right" for row in rows)
    double_support = sum(row.contact_state == "both" for row in rows)
    single_support = sum(row.contact_state in ("left_only", "right_only") for row in rows)
    no_contact = sum(row.contact_state == "none" for row in rows)
    left_swing_knee_positive_steps = sum(
        row.inferred_support_side == "right" and row.swing_knee_positive for row in rows
    )
    right_swing_knee_positive_steps = sum(
        row.inferred_support_side == "left" and row.swing_knee_positive for row in rows
    )
    return AuditSummary(
        mode=mode,
        steps=steps,
        terminated=terminated,
        diagnosis=_diagnose(rows, terminated),
        left_support_duration=left_support_duration,
        right_support_duration=right_support_duration,
        double_support_ratio=double_support / steps,
        single_support_ratio=single_support / steps,
        no_contact_ratio=no_contact / steps,
        swing_unload_success_count=sum(row.swing_unload_success for row in rows),
        micro_lift_success_count=sum(row.micro_lift_success for row in rows),
        max_clearance=max(
            0.0,
            max(max(row.left_foot_clearance, row.right_foot_clearance) for row in rows),
        ),
        max_left_clearance=max(0.0, max(row.left_foot_clearance for row in rows)),
        max_right_clearance=max(0.0, max(row.right_foot_clearance for row in rows)),
        max_lateral_com_shift=max(abs(row.lateral_com_shift) for row in rows),
        vertical_com_oscillation=max(row.com_z for row in rows) - min(row.com_z for row in rows),
        mean_base_height=float(np.mean([row.base_height for row in rows])),
        min_base_height=min(row.base_height for row in rows),
        mean_upright=float(np.mean([row.upright for row in rows])),
        base_proxy_contact_steps=sum(row.base_proxy_contact for row in rows),
        foot_foot_collision_steps=sum(row.foot_foot_collision for row in rows),
        left_swing_knee_positive_steps=left_swing_knee_positive_steps,
        right_swing_knee_positive_steps=right_swing_knee_positive_steps,
        max_left_knee_pitch=max(row.left_knee_pitch for row in rows),
        max_right_knee_pitch=max(row.right_knee_pitch for row in rows),
        swing_knee_positive_micro_lift_count=sum(
            row.swing_knee_positive
            and MICRO_LIFT_MIN_M <= row.swing_foot_clearance <= MICRO_LIFT_MAX_M
            for row in rows
        ),
        mean_action_rms=float(np.mean([row.action_rms for row in rows])),
        mean_torque_rms=float(np.mean([row.torque_rms for row in rows])),
        mean_joint_velocity_rms=float(np.mean([row.joint_velocity_rms for row in rows])),
    )


def run_audit(
    *,
    scene_path: Path,
    mode: str,
    steps: int,
    seed: int,
    out_csv: Path,
    model_path: Path | None,
    vecnorm_path: Path | None,
) -> AuditSummary:
    """Run a Sedon gait audit and write per-step metrics to CSV."""
    if steps <= 0:
        raise ValueError("steps must be positive.")
    reward_config = load_sedon_config_from_env()
    if mode == "zero":
        reward_config = _standing_config(reward_config)
    env = SedonStandingEnv(
        scene_path=scene_path,
        reset_noise_scale=0.0,
        reward_config=reward_config,
    )
    rows: list[AuditStep] = []
    try:
        obs, _ = env.reset(seed=seed)
        mujoco.mj_forward(env.model, env.data)
        initial_foot_bottoms = env._foot_bottom_heights()
        left_foot_id = geom_id(env.model, LEFT_FOOT_GEOM)
        right_foot_id = geom_id(env.model, RIGHT_FOOT_GEOM)
        initial_left_foot_x = float(env.data.geom_xpos[left_foot_id][0])
        initial_right_foot_x = float(env.data.geom_xpos[right_foot_id][0])
        initial_com = _overall_com(env)
        if mode == "policy":
            if model_path is None:
                raise ValueError("--model-path is required when --mode policy.")
            predict_action = _policy_action_provider(model_path, vecnorm_path, env)
        else:
            predict_action = lambda observation: np.zeros(env.action_space.shape, dtype=np.float64)

        terminated = False
        truncated = False
        for step in range(1, steps + 1):
            action = np.asarray(predict_action(obs), dtype=np.float64)
            obs, reward, terminated, truncated, info = env.step(action)
            rows.append(
                _sample_step(
                    env,
                    step=step,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    action=action,
                    initial_left_foot_z=float(initial_foot_bottoms[FOOT_INDEX_BY_SIDE["left"]]),
                    initial_right_foot_z=float(initial_foot_bottoms[FOOT_INDEX_BY_SIDE["right"]]),
                    initial_left_foot_x=initial_left_foot_x,
                    initial_right_foot_x=initial_right_foot_x,
                    initial_com=initial_com,
                )
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AuditStep.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return _summarize(mode, rows, bool(rows and (rows[-1].terminated or rows[-1].truncated)))


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Audit Sedon rollout contact, unload, micro-lift, and knee phase metrics."
    )
    parser.add_argument(
        "--scene-path",
        type=Path,
        default=DEFAULT_SCENE_PATH,
        help="Sedon MJCF scene path.",
    )
    parser.add_argument(
        "--mode",
        choices=("zero", "scripted", "policy"),
        default="scripted",
        help="Rollout source: zero disables gait seed, scripted uses config gait with zero actions, policy loads PPO.",
    )
    parser.add_argument("--steps", type=int, default=400, help="Maximum rollout steps.")
    parser.add_argument("--seed", type=int, default=123, help="Environment seed.")
    parser.add_argument("--model-path", type=Path, default=None, help="PPO checkpoint for policy mode.")
    parser.add_argument("--vecnorm-path", type=Path, default=None, help="Optional VecNormalize path.")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV, help="Per-step CSV output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the gait audit CLI."""
    args = build_parser().parse_args(argv)
    scene_path = require_scene(args.scene_path)
    summary = run_audit(
        scene_path=scene_path,
        mode=args.mode,
        steps=args.steps,
        seed=args.seed,
        out_csv=args.out_csv,
        model_path=args.model_path,
        vecnorm_path=args.vecnorm_path,
    )
    print(f"CSV: {args.out_csv}")
    for key, value in asdict(summary).items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
