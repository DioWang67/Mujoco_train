"""Visual Seedon gait rollout player with MuJoCo viewer and step diagnostics.

This tool plays zero/scripted/policy Seedon rollouts while printing per-step
foot contact, force, clearance, knee pitch, base height, upright, and
support phase diagnostics. Optional CSV output is provided for audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from seedon_baseline.env import (
    DEFAULT_SCENE_PATH,
    JOINT_NAMES,
    SeedonStandingEnv,
    load_seedon_config_from_env,
)
from tools.seedon_debug_common import (
    BASE_PROXY_GEOM,
    DEBUG_OUT_DIR,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    contact_pairs,
    geom_name,
    require_scene,
)

DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "seedon_gait_viewer.csv"


@dataclass(frozen=True)
class GaitViewerStep:
    step: int
    mode: str
    phase_name: str
    support_side: str
    swing_side: str
    left_foot_contact: bool
    right_foot_contact: bool
    base_proxy_contact: bool
    foot_foot_collision: bool
    left_normal_force: float
    right_normal_force: float
    left_force_fraction: float
    right_force_fraction: float
    left_foot_clearance: float
    right_foot_clearance: float
    swing_foot_clearance: float
    left_knee_pitch: float
    right_knee_pitch: float
    base_height: float
    upright: float
    com_x: float
    com_y: float
    com_z: float
    contact_state: str
    contact_pairs: str


@dataclass(frozen=True)
class GaitSeedKeyframe:
    """One exported pose-editor keyframe for scripted reference playback."""

    name: str
    support_mode: str
    joint_targets: np.ndarray
    duration_steps: int


@dataclass(frozen=True)
class GaitSeed:
    """Validated gait seed exported by ``debug_seedon_pose_editor``."""

    keyframes: tuple[GaitSeedKeyframe, ...]
    target_type: str = "absolute"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("zero", "scripted", "policy"), default="scripted")
    parser.add_argument("--scene-path", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--vecnorm-path", type=Path, default=None)
    parser.add_argument(
        "--gait-seed-path",
        type=Path,
        default=None,
        help="Optional pose-editor gait seed JSON for --mode scripted playback.",
    )
    parser.add_argument(
        "--seed-playback",
        choices=("pd", "kinematic", "kinematic-grounded"),
        default="pd",
        help=(
            "How to play pose-editor seeds. 'pd' runs dynamic PD tracking; "
            "'kinematic' directly sets qpos like the pose editor paused view; "
            "'kinematic-grounded' also shifts base height so the support foot stays on the floor."
        ),
    )
    parser.add_argument(
        "--seed-interpolation",
        choices=("smooth", "hold"),
        default="smooth",
        help=(
            "Interpolate between seed keyframes or hold each keyframe for its duration. "
            "'hold' matches the pose editor sequence preview."
        ),
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--pause-start", type=float, default=0.0, help="Seconds to pause before starting rollout.")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--no-viewer", action="store_true", help="Disable MuJoCo viewer and run headless.")
    return parser


def _load_gait_seed(path: Path) -> GaitSeed:
    """Load and validate a pose-editor gait seed JSON file.

    Args:
        path: JSON file produced by ``debug_seedon_pose_editor``.

    Returns:
        Validated gait seed keyframes.

    Raises:
        FileNotFoundError: If the seed file does not exist.
        ValueError: If required fields are missing or incompatible.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Gait seed not found: {path}. Create it first with "
            "`python -m tools.debug_seedon_pose_editor`, add keyframes, then click Export sequence."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Gait seed JSON must be an object.")
    joint_names = payload.get("joint_names")
    if list(joint_names or []) != list(JOINT_NAMES):
        raise ValueError(
            "Gait seed joint_names do not match Seedon JOINT_NAMES. "
            "Re-export the seed with the current pose editor."
        )
    raw_keyframes = payload.get("keyframes")
    if not isinstance(raw_keyframes, list) or not raw_keyframes:
        raise ValueError("Gait seed must contain at least one keyframe.")
    target_type = str(payload.get("target_type", "absolute"))
    if target_type not in ("absolute", "offset"):
        raise ValueError("Gait seed target_type must be either 'absolute' or 'offset'.")

    keyframes: list[GaitSeedKeyframe] = []
    for index, raw in enumerate(raw_keyframes):
        if not isinstance(raw, dict):
            raise ValueError(f"Gait seed keyframe {index} must be an object.")
        targets = np.asarray(raw.get("joint_targets"), dtype=np.float64)
        if targets.shape != (len(JOINT_NAMES),):
            raise ValueError(
                f"Gait seed keyframe {index} joint_targets must contain {len(JOINT_NAMES)} values."
            )
        duration_steps = int(raw.get("duration_steps", 60))
        if duration_steps <= 0:
            raise ValueError(f"Gait seed keyframe {index} duration_steps must be positive.")
        keyframes.append(
            GaitSeedKeyframe(
                name=str(raw.get("name", f"keyframe_{index + 1}")),
                support_mode=str(raw.get("support_mode", "double")),
                joint_targets=targets,
                duration_steps=duration_steps,
            )
        )
    return GaitSeed(keyframes=tuple(keyframes), target_type=target_type)


def _seed_target_at_step(
    seed: GaitSeed,
    step: int,
    *,
    interpolation: str = "smooth",
) -> tuple[np.ndarray, str, str]:
    """Return interpolated joint targets, phase name, and support mode for a seed step."""
    if interpolation not in ("smooth", "hold"):
        raise ValueError("interpolation must be either 'smooth' or 'hold'.")
    total_steps = sum(keyframe.duration_steps for keyframe in seed.keyframes)
    phase_step = (step - 1) % max(1, total_steps)
    cursor = 0
    for index, keyframe in enumerate(seed.keyframes):
        next_cursor = cursor + keyframe.duration_steps
        if phase_step < next_cursor:
            if interpolation == "hold":
                return keyframe.joint_targets.copy(), keyframe.name, keyframe.support_mode
            local_step = phase_step - cursor
            next_keyframe = seed.keyframes[(index + 1) % len(seed.keyframes)]
            alpha = local_step / max(1, keyframe.duration_steps - 1)
            smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            target = (
                (1.0 - smooth_alpha) * keyframe.joint_targets
                + smooth_alpha * next_keyframe.joint_targets
            )
            return target, keyframe.name, keyframe.support_mode
        cursor = next_cursor
    last = seed.keyframes[-1]
    return last.joint_targets.copy(), last.name, last.support_mode


def _build_policy_provider(
    model_path: Path,
    vecnorm_path: Path | None,
    env: SeedonStandingEnv,
) -> Callable[[np.ndarray], np.ndarray]:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Policy mode requires stable-baselines3. Install project requirements."
        ) from exc

    if not model_path or not model_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {model_path}")
    model = PPO.load(str(model_path))
    if vecnorm_path is None:
        return lambda obs: model.predict(obs, deterministic=True)[0]

    if not vecnorm_path.is_file():
        raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_path}")

    def _make_vec_env():
        return DummyVecEnv(
            [
                lambda: SeedonStandingEnv(
                    scene_path=env._scene_path,
                    reset_noise_scale=0.0,
                    reward_config=env._reward_config,
                )
            ]
        )

    vec_env = _make_vec_env()
    vecnorm = VecNormalize.load(str(vecnorm_path), vec_env)
    vecnorm.training = False
    vecnorm.norm_reward = False

    def _predict(obs: np.ndarray) -> np.ndarray:
        norm_obs = vecnorm.normalize_obs(obs[None, :])
        action, _ = model.predict(norm_obs, deterministic=True)
        return np.asarray(action[0], dtype=np.float64)

    return _predict


def _contact_flags(env: SeedonStandingEnv) -> tuple[bool, bool, bool, bool, list[str]]:
    left = False
    right = False
    base = False
    foot_foot = False
    pairs: list[str] = []
    for name_a, name_b, _ in contact_pairs(env.model, env.data):
        pairs.append(f"{name_a}-{name_b}")
        if {name_a, name_b} == {FLOOR_GEOM, LEFT_FOOT_GEOM}:
            left = True
        elif {name_a, name_b} == {FLOOR_GEOM, RIGHT_FOOT_GEOM}:
            right = True
        elif {name_a, name_b} == {FLOOR_GEOM, BASE_PROXY_GEOM}:
            base = True
        elif {name_a, name_b} == {LEFT_FOOT_GEOM, RIGHT_FOOT_GEOM}:
            foot_foot = True
    return left, right, base, foot_foot, pairs


def _foot_normal_force(env: SeedonStandingEnv, foot_geom_name: str) -> float:
    total_force = 0.0
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        name_a = geom_name(env.model, int(contact.geom1))
        name_b = geom_name(env.model, int(contact.geom2))
        if {name_a, name_b} != {FLOOR_GEOM, foot_geom_name}:
            continue
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(env.model, env.data, contact_index, wrench)
        total_force += abs(float(wrench[0]))
    return total_force


def _compute_support_side(left_force: float, right_force: float) -> str:
    total = left_force + right_force
    if total <= 0.0:
        return "none"
    left_fraction = left_force / total
    right_fraction = right_force / total
    if left_fraction >= 0.65 and right_fraction <= 0.25:
        return "left"
    if right_fraction >= 0.65 and left_fraction <= 0.25:
        return "right"
    return "none"


def _overall_com(env: SeedonStandingEnv) -> np.ndarray:
    masses = env.model.body_mass
    total_mass = float(np.sum(masses))
    return np.sum(env.data.xipos * masses[:, None], axis=0) / total_mass


def _add_marker(
    viewer: object,
    *,
    position: np.ndarray,
    radius: float,
    rgba: tuple[float, float, float, float],
) -> None:
    """Add a simple debug sphere to the passive MuJoCo viewer."""
    scene = getattr(viewer, "user_scn", None)
    if scene is None or scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.eye(3, dtype=np.float64).ravel(),
        np.array(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def _update_viewer_markers(viewer: object, env: SeedonStandingEnv, row: GaitViewerStep) -> None:
    """Refresh COM and foot-contact markers in the MuJoCo viewer."""
    scene = getattr(viewer, "user_scn", None)
    if scene is None:
        return
    scene.ngeom = 0
    _add_marker(
        viewer,
        position=np.array([row.com_x, row.com_y, row.com_z], dtype=np.float64),
        radius=0.025,
        rgba=(1.0, 0.1, 0.1, 0.85),
    )
    right_position = env.data.geom_xpos[env._foot_geom_ids[0]].copy()
    left_position = env.data.geom_xpos[env._foot_geom_ids[1]].copy()
    right_position[2] += 0.035
    left_position[2] += 0.035
    _add_marker(
        viewer,
        position=right_position,
        radius=0.018,
        rgba=(0.1, 0.3, 1.0, 0.85 if row.right_foot_contact else 0.25),
    )
    _add_marker(
        viewer,
        position=left_position,
        radius=0.018,
        rgba=(0.1, 1.0, 0.3, 0.85 if row.left_foot_contact else 0.25),
    )


def _build_row(
    env: SeedonStandingEnv,
    step: int,
    mode: str,
    action: np.ndarray,
    initial_left_bottom: float,
    initial_right_bottom: float,
    phase_name: str | None = None,
    support_hint: str | None = None,
) -> GaitViewerStep:
    left_contact, right_contact, base_contact, foot_foot, pairs = _contact_flags(env)
    left_force = _foot_normal_force(env, LEFT_FOOT_GEOM)
    right_force = _foot_normal_force(env, RIGHT_FOOT_GEOM)
    total_force = left_force + right_force
    left_fraction = left_force / total_force if total_force > 0.0 else 0.0
    right_fraction = right_force / total_force if total_force > 0.0 else 0.0
    foot_bottoms = env._foot_bottom_heights()
    swing_clearance = 0.0
    support_side = _compute_support_side(left_force, right_force)
    if support_side == "none" and support_hint in ("left", "right"):
        support_side = support_hint
    swing_side = "none"
    if support_side == "left":
        swing_side = "right"
        swing_clearance = float(foot_bottoms[0] - initial_right_bottom)
    elif support_side == "right":
        swing_side = "left"
        swing_clearance = float(foot_bottoms[1] - initial_left_bottom)
    com = _overall_com(env)
    contact_state = (
        "both" if left_contact and right_contact else "left_only" if left_contact else "right_only" if right_contact else "none"
    )
    return GaitViewerStep(
        step=step,
        mode=mode,
        phase_name=phase_name or (str(env._task_phase_metadata().get("phase_name", "")) if mode != "zero" else "standing"),
        support_side=support_side,
        swing_side=swing_side,
        left_foot_contact=left_contact,
        right_foot_contact=right_contact,
        base_proxy_contact=base_contact,
        foot_foot_collision=foot_foot,
        left_normal_force=float(left_force),
        right_normal_force=float(right_force),
        left_force_fraction=float(left_fraction),
        right_force_fraction=float(right_fraction),
        left_foot_clearance=float(foot_bottoms[1] - initial_left_bottom),
        right_foot_clearance=float(foot_bottoms[0] - initial_right_bottom),
        swing_foot_clearance=float(swing_clearance),
        left_knee_pitch=float(env._joint_positions()[8]),
        right_knee_pitch=float(env._joint_positions()[3]),
        base_height=float(env._base_height()),
        upright=float(env._base_upright()),
        com_x=float(com[0]),
        com_y=float(com[1]),
        com_z=float(com[2]),
        contact_state=contact_state,
        contact_pairs=", ".join(pairs),
    )


def _step_seed_reference(
    env: SeedonStandingEnv,
    target_positions: np.ndarray,
) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
    """Advance Seedon by one env step while tracking exported reference targets."""
    target = env._apply_safe_joint_target_clamps(target_positions)
    env._do_pd_simulation(target)
    env._gait_step += 1
    obs = env._get_obs()
    base_height = env._base_height()
    upright = env._base_upright()
    terminated = env._is_terminated(base_height, upright, obs)
    joint_positions = env._joint_positions()
    info = {
        "base_height": base_height,
        "upright": upright,
        "right_knee_qpos": float(joint_positions[3]),
        "left_knee_qpos": float(joint_positions[8]),
    }
    return obs, 0.0, bool(terminated), False, info


def _step_seed_kinematic_reference(
    env: SeedonStandingEnv,
    target_positions: np.ndarray,
    support_hint: str | None = None,
    *,
    ground_support: bool = False,
) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
    """Set a seed target directly, matching the pose editor paused preview."""
    target_qpos = env._default_qpos.copy()
    env._set_base_pose(target_qpos)
    for joint_id, joint_target in zip(env._joint_ids, target_positions):
        target_qpos[env.model.jnt_qposadr[joint_id]] = float(joint_target)
    env.set_state(target_qpos, np.zeros_like(env.data.qvel))
    mujoco.mj_forward(env.model, env.data)
    if ground_support:
        foot_bottoms = env._foot_bottom_heights()
        if support_hint == "right":
            ground_offset = float(foot_bottoms[0])
        elif support_hint == "left":
            ground_offset = float(foot_bottoms[1])
        else:
            ground_offset = float(np.min(foot_bottoms))
        if np.isfinite(ground_offset):
            target_qpos[2] -= ground_offset
            env.set_state(target_qpos, np.zeros_like(env.data.qvel))
            mujoco.mj_forward(env.model, env.data)
    env._gait_step += 1
    obs = env._get_obs()
    joint_positions = env._joint_positions()
    info = {
        "base_height": env._base_height(),
        "upright": env._base_upright(),
        "right_knee_qpos": float(joint_positions[3]),
        "left_knee_qpos": float(joint_positions[8]),
    }
    return obs, 0.0, False, False, info


def _write_csv(path: Path, rows: list[GaitViewerStep]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.speed <= 0.0:
        raise ValueError("--speed must be positive.")
    scene_path = require_scene(args.scene_path)
    reward_config = load_seedon_config_from_env()
    if args.mode == "zero":
        reward_config = reward_config.__class__(**{**reward_config.__dict__, "gait_hip_roll_amp": 0.0, "gait_hip_pitch_amp": 0.0, "gait_knee_pitch_amp": 0.0, "gait_ankle_pitch_amp": 0.0, "fsm_right_swing_scale": 0.0, "fsm_left_swing_scale": 0.0, "fsm_right_support_roll_scale": 0.0, "fsm_left_support_roll_scale": 0.0})
    env = SeedonStandingEnv(scene_path=scene_path, reset_noise_scale=0.0, reward_config=reward_config)
    try:
        obs, _ = env.reset(seed=123)
    except TypeError:
        obs = env.reset(seed=123)
    action_provider = None
    if args.mode == "policy":
        if args.checkpoint_path is None:
            raise ValueError("--checkpoint-path is required when --mode policy.")
        action_provider = _build_policy_provider(args.checkpoint_path, args.vecnorm_path, env)
    elif args.mode == "scripted":
        action_provider = lambda _obs: np.zeros(env.action_space.shape, dtype=np.float64)
    else:
        action_provider = lambda _obs: np.zeros(env.action_space.shape, dtype=np.float64)

    gait_seed = _load_gait_seed(args.gait_seed_path) if args.gait_seed_path is not None else None
    if gait_seed is not None and args.mode != "scripted":
        raise ValueError("--gait-seed-path is only supported with --mode scripted.")

    if args.pause_start > 0.0:
        print(f"Pausing for {args.pause_start:.2f} seconds before rollout start...")
        time.sleep(args.pause_start)

    viewer = None
    if not args.no_viewer:
        try:
            import mujoco.viewer  # type: ignore[import]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "MuJoCo viewer unavailable. Install MuJoCo viewer extras or run with --no-viewer."
            ) from exc
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        viewer.__enter__()
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    rows: list[GaitViewerStep] = []
    initial_bottoms = env._foot_bottom_heights()
    try:
        for step in range(1, args.steps + 1):
            phase_name = None
            support_hint = None
            if gait_seed is not None:
                target_positions, phase_name, support_hint = _seed_target_at_step(
                    gait_seed,
                    step,
                    interpolation=args.seed_interpolation,
                )
                if gait_seed.target_type == "offset":
                    target_positions = env._nominal_joint_qpos + target_positions
                action = np.zeros(env.action_space.shape, dtype=np.float64)
                if args.seed_playback in ("kinematic", "kinematic-grounded"):
                    obs, reward, terminated, truncated, info = _step_seed_kinematic_reference(
                        env,
                        target_positions,
                        support_hint,
                        ground_support=args.seed_playback == "kinematic-grounded",
                    )
                else:
                    obs, reward, terminated, truncated, info = _step_seed_reference(env, target_positions)
            else:
                action = action_provider(obs)
                obs, reward, terminated, truncated, info = env.step(action)
            row = _build_row(
                env,
                step,
                args.mode,
                action,
                float(initial_bottoms[1]),
                float(initial_bottoms[0]),
                phase_name=phase_name,
                support_hint=support_hint,
            )
            rows.append(row)
            print(
                f"{step:>4} phase={row.phase_name:<16} support={row.support_side:<5} swing={row.swing_side:<5} "
                f"Lcnt={int(row.left_foot_contact)} Rcnt={int(row.right_foot_contact)} "
                f"Lfrc={row.left_normal_force:>6.2f} Rfrc={row.right_normal_force:>6.2f} "
                f"Lclr={row.left_foot_clearance:>6.4f} Rclr={row.right_foot_clearance:>6.4f} "
                f"Rknee={row.right_knee_pitch:>6.3f} Lknee={row.left_knee_pitch:>6.3f} "
                f"base_z={row.base_height:>5.3f} upr={row.upright:>5.3f} contact={row.contact_state}"
            )
            if viewer is not None:
                _update_viewer_markers(viewer, env, row)
                viewer.sync()
                if not viewer.is_running():
                    break
            time.sleep(max(0.0, float(env.dt) / args.speed))
            if terminated or truncated:
                print(f"Rollout ended at step {step} terminated={terminated} truncated={truncated}")
                break
    finally:
        if viewer is not None:
            viewer.__exit__(None, None, None)
        env.close()
        _write_csv(args.out_csv, rows)
        print(f"CSV written to: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
