"""Minimal MuJoCo locomotion environment for the private Sedon robot model."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import mujoco
    from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
    from gymnasium.spaces import Box

    _MUJOCO_IMPORT_ERROR: ModuleNotFoundError | None = None
except ModuleNotFoundError as exc:
    mujoco = None
    Box = None
    _MUJOCO_IMPORT_ERROR = exc

    class MujocoEnv:  # type: ignore[no-redef]
        """Placeholder so config/helper imports work without MuJoCo installed."""

        pass


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_PATH = REPO_ROOT / "private_assets" / "sedon" / "training_scene.xml"
JOINT_NAMES = (
    "R_joint_hip_yaw",
    "R_joint_hip_roll",
    "R_joint_hip_pitch",
    "R_joint_knee_pitch",
    "R_joint_ankle_pitch",
    "L_joint_hip_yaw",
    "L_joint_hip_roll",
    "L_joint_hip_pitch",
    "L_joint_knee_pitch",
    "L_joint_ankle_pitch",
)
RIGHT_HIP_ROLL_ACTUATOR_INDEX = 1
RIGHT_KNEE_JOINT_INDEX = 3
LEFT_HIP_ROLL_ACTUATOR_INDEX = 6
LEFT_KNEE_JOINT_INDEX = 8
FOOT_GEOM_NAMES = ("R_foot_collision", "L_foot_collision")
FLOOR_GEOM_NAME = "floor"
BASE_PROXY_GEOM_NAME = "base_proxy"
CONFIG_OVERRIDES_ENV = "SEDON_CONFIG_OVERRIDES"


@dataclass(frozen=True)
class SedonStandingConfig:
    """Reward and termination settings for the Sedon locomotion task.

    Args:
        target_base_height: Desired base height in meters.
        target_forward_velocity: Desired base x velocity in meters per second.
        min_base_height: Episode terminates below this height.
        max_base_height: Episode terminates above this height.
        min_upright: Episode terminates below this base upright alignment.
        termination_penalty: Penalty applied when an episode terminates early.
        torque_scale: Maximum absolute PD torque command before actuator clipping.
        action_joint_delta_scale: Maximum joint target offset represented by action 1.0.
        task_mode: High-level task objective, either ``"walk"`` or ``"com_shift"``.
        gait_mode: Built-in deterministic gait seed mode, either ``"fsm"`` or ``"sin"``.
        gait_cycle_steps: Number of RL steps in one built-in walking gait cycle.
        fsm_right_lift_steps: RL steps spent lifting the right swing leg.
        fsm_right_lower_steps: RL steps spent lowering the right swing leg.
        fsm_left_lift_steps: RL steps spent lifting the left swing leg.
        fsm_left_lower_steps: RL steps spent lowering the left swing leg.
        fsm_double_support_steps: RL steps spent in neutral double-support between swings.
        fsm_right_swing_scale: Extra scale on right-leg swing pitch offsets.
        fsm_left_swing_scale: Extra scale on left-leg swing pitch offsets.
        fsm_right_support_roll_scale: Roll scale while left leg supports and right leg swings.
        fsm_left_support_roll_scale: Roll scale while right leg supports and left leg swings.
        fsm_swing_cap: Upper cap applied to FSM swing interpolation.
        gait_hip_roll_amp: Hip roll amplitude for shifting weight between feet.
        gait_right_hip_roll_sign: Direction multiplier for the right hip roll gait offset.
        gait_left_hip_roll_sign: Direction multiplier for the left hip roll gait offset.
        gait_hip_pitch_amp: Signed hip pitch amplitude for the built-in gait target.
        gait_knee_pitch_amp: Signed knee pitch amplitude for the built-in gait target.
        gait_ankle_pitch_amp: Signed ankle pitch amplitude for the built-in gait target.
        nominal_hip_pitch_offset: Signed stance offset applied to both hip-pitch joints.
        nominal_knee_pitch_offset: Signed stance offset applied to both knee-pitch joints.
        nominal_ankle_pitch_offset: Signed stance offset applied to both ankle-pitch joints.
        com_shift_center_hold_steps: RL steps spent holding the base near center.
        com_shift_shift_steps: RL steps spent shifting slightly toward one lateral side.
        com_shift_support_roll_amp: Hip roll amplitude for deterministic COM shifting.
        com_shift_unload_hip_pitch_amp: Swing hip pitch amplitude used for foot unload.
        com_shift_unload_knee_pitch_amp: Swing knee bend amplitude used for foot unload.
        com_shift_unload_ankle_pitch_amp: Swing ankle bend amplitude used for foot unload.
        com_shift_lateral_target_magnitude: Fixed lateral target magnitude in meters for shift phases.
        com_shift_lateral_target_weight: Reward weight for matching the desired lateral COM shift.
        com_shift_lateral_target_sharpness: Exponential sharpness for lateral target error.
        com_shift_lateral_progress_weight: Reward weight for reducing lateral target error step to step.
        com_shift_both_contact_reward_weight: Reward weight for keeping both feet planted.
        com_shift_support_contact_reward_weight: Reward weight for keeping the designated support foot planted during shift phases.
        com_shift_swing_unload_reward_weight: Reward weight for slightly unloading the swing foot during shift phases.
        com_shift_shift_phase_error_penalty_weight: Penalty coefficient for missing the requested lateral shift during shift phases.
        com_shift_swing_height_target: Target swing-foot height delta over the support foot for unload credit.
        com_shift_single_contact_penalty_weight: Penalty coefficient when only one foot remains in contact.
        com_shift_no_contact_penalty_weight: Penalty coefficient when both feet lose floor contact.
        com_shift_forward_velocity_abs_penalty_weight: Penalty coefficient for absolute forward drift speed.
        com_shift_forward_displacement_penalty_weight: Penalty coefficient for accumulated forward drift.
        com_shift_forward_overspeed_penalty_weight: Extra penalty coefficient above ``forward_overspeed_limit``.
        com_shift_lateral_velocity_penalty_weight: Penalty coefficient for lateral speed in COM-shift mode.
        pd_stiffness: Joint-space proportional gain for stance tracking.
        pd_damping: Joint-space velocity damping gain for stance tracking.
        alive_reward: Reward granted each non-terminal step.
        height_weight: Weight for matching target height.
        height_sharpness: Exponential penalty sharpness for base-height error.
        upright_weight: Weight for keeping the base z-axis upright.
        forward_velocity_weight: Weight for tracking target forward speed.
        min_rewarded_forward_velocity: Forward speed below which stable policies are penalized.
        low_forward_velocity_penalty_weight: Penalty for standing still while stable.
        min_cumulative_progress_per_step: Expected x progress per RL step before static penalties apply.
        cumulative_progress_grace_steps: Initial steps exempt from cumulative progress penalties.
        progress_reward_weight: Weight for positive per-step x displacement.
        near_fall_upright_threshold: Upright value where soft near-fall penalty starts.
        near_fall_height_threshold: Base height where soft near-fall penalty starts.
        near_fall_upright_penalty_weight: Penalty weight for low upright margin.
        near_fall_height_penalty_weight: Penalty weight for low base-height margin.
        forward_overspeed_limit: Base x velocity threshold that starts an overspeed penalty.
        forward_overspeed_penalty_weight: Penalty coefficient for rushing forward too fast.
        backward_velocity_penalty_weight: Penalty coefficient for moving backward.
        lateral_velocity_penalty_weight: Penalty coefficient for lateral drift speed.
        pose_weight: Weight for keeping actuated joints near the nominal stance.
        pose_sharpness: Exponential penalty sharpness for joint pose error.
        action_penalty_weight: Penalty coefficient for squared normalized action.
        action_rate_penalty_weight: Penalty coefficient for changing actions too abruptly.
        velocity_penalty_weight: Penalty coefficient for joint velocity.
        base_xy_velocity_penalty_weight: Deprecated full horizontal speed penalty.
        base_roll_pitch_rate_penalty_weight: Penalty coefficient for roll/pitch angular speed.
        foot_flat_weight: Weight for keeping both foot collision boxes flat.
        foot_height_penalty_weight: Penalty coefficient for foot bottom height error.
        foot_air_penalty_weight: Penalty coefficient for feet not near the floor.
        max_base_xy_drift: Episode terminates if the base drifts farther than this radius.
        right_knee_safe_lower: Optional soft-safe lower qpos bound for the right knee.
        right_knee_safe_upper: Optional soft-safe upper qpos bound for the right knee.
        left_knee_safe_lower: Optional soft-safe lower qpos bound for the left knee.
        left_knee_safe_upper: Optional soft-safe upper qpos bound for the left knee.
    """

    target_base_height: float = 0.446
    target_forward_velocity: float = 0.06
    min_base_height: float = 0.34
    max_base_height: float = 0.65
    min_upright: float = 0.75
    termination_penalty: float = 50.0
    torque_scale: float = 45.0
    action_joint_delta_scale: float = 0.08
    task_mode: str = "walk"
    gait_mode: str = "fsm"
    gait_cycle_steps: int = 480
    fsm_right_lift_steps: int = 180
    fsm_right_lower_steps: int = 140
    fsm_left_lift_steps: int = 200
    fsm_left_lower_steps: int = 160
    fsm_double_support_steps: int = 0
    fsm_right_swing_scale: float = 1.0
    fsm_left_swing_scale: float = 0.9
    fsm_right_support_roll_scale: float = 1.2
    fsm_left_support_roll_scale: float = 0.7
    fsm_swing_cap: float = 0.76
    gait_hip_roll_amp: float = 0.04
    gait_right_hip_roll_sign: float = 1.0
    gait_left_hip_roll_sign: float = -1.0
    gait_hip_pitch_amp: float = 0.24
    gait_knee_pitch_amp: float = -0.24
    gait_ankle_pitch_amp: float = -0.16
    nominal_hip_pitch_offset: float = 0.0
    nominal_knee_pitch_offset: float = 0.0
    nominal_ankle_pitch_offset: float = 0.0
    com_shift_center_hold_steps: int = 80
    com_shift_shift_steps: int = 80
    com_shift_support_roll_amp: float = 0.03
    com_shift_unload_hip_pitch_amp: float = 0.0
    com_shift_unload_knee_pitch_amp: float = 0.0
    com_shift_unload_ankle_pitch_amp: float = 0.0
    com_shift_lateral_target_magnitude: float = 0.02
    com_shift_lateral_target_weight: float = 3.0
    com_shift_lateral_target_sharpness: float = 120.0
    com_shift_lateral_progress_weight: float = 18.0
    com_shift_both_contact_reward_weight: float = 3.0
    com_shift_support_contact_reward_weight: float = 3.0
    com_shift_swing_unload_reward_weight: float = 2.0
    com_shift_shift_phase_error_penalty_weight: float = 8.0
    com_shift_swing_height_target: float = 0.004
    com_shift_single_contact_penalty_weight: float = 2.0
    com_shift_no_contact_penalty_weight: float = 6.0
    com_shift_forward_velocity_abs_penalty_weight: float = 6.0
    com_shift_forward_displacement_penalty_weight: float = 5.0
    com_shift_forward_overspeed_penalty_weight: float = 35.0
    com_shift_lateral_velocity_penalty_weight: float = 1.5
    pd_stiffness: float = 35.0
    pd_damping: float = 2.0
    alive_reward: float = 0.2
    height_weight: float = 2.5
    height_sharpness: float = 40.0
    upright_weight: float = 3.5
    forward_velocity_weight: float = 5.0
    min_rewarded_forward_velocity: float = 0.035
    low_forward_velocity_penalty_weight: float = 4.0
    min_cumulative_progress_per_step: float = 0.00005
    cumulative_progress_grace_steps: int = 20
    progress_reward_weight: float = 5.6
    near_fall_upright_threshold: float = 0.78
    near_fall_height_threshold: float = 0.37
    near_fall_upright_penalty_weight: float = 20.0
    near_fall_height_penalty_weight: float = 30.0
    forward_overspeed_limit: float = 0.2
    forward_overspeed_penalty_weight: float = 5.0
    backward_velocity_penalty_weight: float = 3.0
    lateral_velocity_penalty_weight: float = 2.0
    pose_weight: float = 0.05
    pose_sharpness: float = 8.0
    action_penalty_weight: float = 0.015
    action_rate_penalty_weight: float = 0.008
    velocity_penalty_weight: float = 0.001
    base_xy_velocity_penalty_weight: float = 0.0
    base_roll_pitch_rate_penalty_weight: float = 0.4
    foot_flat_weight: float = 0.8
    foot_height_penalty_weight: float = 2.0
    foot_air_penalty_weight: float = 0.05
    max_base_xy_drift: float = 2.0
    right_knee_safe_lower: float | None = None
    right_knee_safe_upper: float | None = None
    left_knee_safe_lower: float | None = None
    left_knee_safe_upper: float | None = None


def load_sedon_config_from_env() -> SedonStandingConfig:
    """Return ``SedonStandingConfig`` with optional JSON environment overrides.

    The ``SEDON_CONFIG_OVERRIDES`` value may be either a JSON object string or
    a path to a JSON file. Unknown keys are rejected to catch misspelled sweep
    parameters early.
    """
    config = SedonStandingConfig()
    raw_overrides = os.environ.get(CONFIG_OVERRIDES_ENV)
    if not raw_overrides:
        return config

    override_source = Path(raw_overrides)
    try:
        if override_source.is_file():
            overrides = json.loads(override_source.read_text(encoding="utf-8"))
        else:
            overrides = json.loads(raw_overrides)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{CONFIG_OVERRIDES_ENV} must be a JSON object or a path to one."
        ) from exc

    if not isinstance(overrides, dict):
        raise ValueError(f"{CONFIG_OVERRIDES_ENV} must decode to a JSON object.")

    valid_keys = set(SedonStandingConfig.__dataclass_fields__)
    unknown_keys = sorted(set(overrides) - valid_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown Sedon config override(s): {', '.join(unknown_keys)}"
        )
    return SedonStandingConfig(**{**config.__dict__, **overrides})


def compute_standing_reward(
    base_height: float,
    upright: float,
    joint_velocity_l2: float,
    action_l2: float,
    action_rate_l2: float,
    joint_position_error_l2: float,
    forward_velocity: float,
    lateral_velocity_l2: float,
    base_xy_velocity_l2: float,
    base_roll_pitch_rate_l2: float,
    foot_flatness: float,
    foot_height_error_l2: float,
    feet_near_floor: int,
    config: SedonStandingConfig,
    progress_delta_x: float = 0.0,
    cumulative_progress_x: float = 0.0,
    elapsed_steps: int = 1,
) -> dict[str, float]:
    """Compute shaped reward terms for standing.

    Args:
        base_height: Current base body height.
        upright: Dot product between local base z-axis and world z-axis.
        joint_velocity_l2: Squared norm of actuated joint velocities.
        action_l2: Squared norm of normalized actions.
        action_rate_l2: Squared norm of the action delta from the previous step.
        joint_position_error_l2: Squared norm of actuated joint deviation from the seed pose.
        forward_velocity: Base x velocity in meters per second.
        lateral_velocity_l2: Squared base y velocity.
        base_xy_velocity_l2: Squared norm of horizontal base velocity.
        base_roll_pitch_rate_l2: Squared norm of base roll/pitch angular rate.
        foot_flatness: Mean foot z-axis alignment with world z-axis.
        foot_height_error_l2: Squared norm of foot bottom height error from the floor.
        feet_near_floor: Number of feet whose bottom is close to the floor.
        config: Reward coefficients.
        progress_delta_x: Per-step base x displacement since the previous step.
        cumulative_progress_x: Base x displacement since episode reset.
        elapsed_steps: Number of elapsed environment steps in the episode.

    Returns:
        Reward component mapping including ``total``.
    """
    height_error = base_height - config.target_base_height
    height = float(np.exp(-config.height_sharpness * height_error * height_error))
    upright_clipped = float(np.clip(upright, -1.0, 1.0))
    forward_progress = float(
        np.clip(forward_velocity / config.target_forward_velocity, 0.0, 1.0)
    )
    upright_gate = float(
        np.clip(
            (upright_clipped - config.min_upright) / (1.0 - config.min_upright),
            0.0,
            1.0,
        )
    )
    stability_gate = height * upright_gate
    backward_velocity = max(0.0, -forward_velocity)
    low_forward_shortfall = max(
        0.0,
        (config.min_rewarded_forward_velocity - forward_velocity)
        / config.min_rewarded_forward_velocity,
    )
    progress_steps = max(0, elapsed_steps - config.cumulative_progress_grace_steps)
    expected_progress = progress_steps * config.min_cumulative_progress_per_step
    cumulative_progress_shortfall = 0.0
    if expected_progress > 0.0:
        cumulative_progress_shortfall = max(
            0.0,
            (expected_progress - cumulative_progress_x) / expected_progress,
        )
    overspeed = max(0.0, forward_velocity - config.forward_overspeed_limit)
    progress = max(0.0, progress_delta_x)
    upright_margin = max(0.0, config.near_fall_upright_threshold - upright_clipped)
    height_margin = max(0.0, config.near_fall_height_threshold - base_height)
    pose = float(np.exp(-config.pose_sharpness * joint_position_error_l2))
    foot_flatness_clipped = float(np.clip(foot_flatness, 0.0, 1.0))
    missing_feet = max(0, len(FOOT_GEOM_NAMES) - feet_near_floor)
    components = {
        "alive": config.alive_reward,
        "height": height,
        "upright": max(0.0, upright_clipped),
        "forward_velocity": forward_progress * stability_gate,
        "progress": progress * stability_gate,
        "stability_gate": stability_gate,
        "near_fall_upright_penalty": upright_margin * upright_margin,
        "near_fall_height_penalty": height_margin * height_margin,
        "low_forward_velocity_penalty": low_forward_shortfall
        * low_forward_shortfall
        * stability_gate
        * cumulative_progress_shortfall,
        "cumulative_progress_shortfall": cumulative_progress_shortfall,
        "forward_overspeed_penalty": overspeed * overspeed,
        "backward_velocity_penalty": backward_velocity * backward_velocity,
        "lateral_velocity_penalty": lateral_velocity_l2,
        "pose": pose,
        "foot_flat": foot_flatness_clipped,
        "action_penalty": action_l2,
        "action_rate_penalty": action_rate_l2,
        "velocity_penalty": joint_velocity_l2,
        "base_xy_velocity_penalty": base_xy_velocity_l2,
        "base_roll_pitch_rate_penalty": base_roll_pitch_rate_l2,
        "foot_height_penalty": foot_height_error_l2,
        "foot_air_penalty": float(missing_feet),
    }
    total = components["alive"]
    total += config.height_weight * components["height"]
    total += config.upright_weight * components["upright"]
    total += config.forward_velocity_weight * components["forward_velocity"]
    total += config.progress_reward_weight * components["progress"]
    total -= (
        config.near_fall_upright_penalty_weight
        * components["near_fall_upright_penalty"]
    )
    total -= (
        config.near_fall_height_penalty_weight
        * components["near_fall_height_penalty"]
    )
    total -= (
        config.low_forward_velocity_penalty_weight
        * components["low_forward_velocity_penalty"]
    )
    total -= (
        config.forward_overspeed_penalty_weight
        * components["forward_overspeed_penalty"]
    )
    total -= (
        config.backward_velocity_penalty_weight
        * components["backward_velocity_penalty"]
    )
    total -= (
        config.lateral_velocity_penalty_weight
        * components["lateral_velocity_penalty"]
    )
    total += config.pose_weight * components["pose"]
    total += config.foot_flat_weight * components["foot_flat"]
    total -= config.action_penalty_weight * components["action_penalty"]
    total -= config.action_rate_penalty_weight * components["action_rate_penalty"]
    total -= config.velocity_penalty_weight * components["velocity_penalty"]
    total -= config.base_xy_velocity_penalty_weight * components["base_xy_velocity_penalty"]
    total -= (
        config.base_roll_pitch_rate_penalty_weight
        * components["base_roll_pitch_rate_penalty"]
    )
    total -= config.foot_height_penalty_weight * components["foot_height_penalty"]
    total -= config.foot_air_penalty_weight * components["foot_air_penalty"]
    components["total"] = float(total)
    return components


def compute_com_shift_reward(
    base_height: float,
    upright: float,
    joint_velocity_l2: float,
    action_l2: float,
    action_rate_l2: float,
    joint_position_error_l2: float,
    forward_velocity: float,
    lateral_velocity: float,
    base_roll_pitch_rate_l2: float,
    foot_flatness: float,
    base_y: float,
    target_base_y: float,
    previous_abs_y_error: float,
    current_base_x_displacement: float,
    in_double_support: bool,
    support_contact: bool,
    swing_contact: bool,
    support_foot_bottom_z: float,
    swing_foot_bottom_z: float,
    config: SedonStandingConfig,
) -> dict[str, float]:
    """Compute reward terms for COM shift and foot-unload curriculum training."""
    height_error = base_height - config.target_base_height
    height = float(np.exp(-config.height_sharpness * height_error * height_error))
    upright_clipped = float(np.clip(upright, -1.0, 1.0))
    pose = float(np.exp(-config.pose_sharpness * joint_position_error_l2))
    lateral_error = base_y - target_base_y
    current_abs_y_error = abs(lateral_error)
    lateral_progress = float(
        np.clip(previous_abs_y_error - current_abs_y_error, -0.02, 0.02)
    )
    target_magnitude = abs(target_base_y)
    if in_double_support or target_magnitude <= 1e-9:
        lateral_target = float(
            np.exp(-config.com_shift_lateral_target_sharpness * lateral_error * lateral_error)
        )
    else:
        lateral_target = float(
            np.clip(1.0 - current_abs_y_error / max(target_magnitude, 1e-9), 0.0, 1.0)
        )
    foot_flat = float(np.clip(foot_flatness, 0.0, 1.0))
    both_contact = bool(support_contact and swing_contact)
    single_contact = bool(support_contact ^ swing_contact)
    no_contact = bool((not support_contact) and (not swing_contact))
    shift_phase_gate = float(not in_double_support)
    support_contact_reward = float(support_contact and not in_double_support)
    double_support_contact = float(both_contact and in_double_support)
    swing_height_delta = max(0.0, swing_foot_bottom_z - support_foot_bottom_z)
    swing_unload = float(
        np.clip(
            max(swing_height_delta, 0.0) / max(config.com_shift_swing_height_target, 1e-9),
            0.0,
            1.0,
        )
    ) * shift_phase_gate
    shift_phase_error_penalty = (
        float(current_abs_y_error / max(target_magnitude, 1e-9)) * shift_phase_gate
        if target_magnitude > 1e-9
        else 0.0
    )
    stability_gate = height * max(0.0, upright_clipped)
    progress_gate = float(
        support_contact
        and upright_clipped > 0.95
        and base_height > 0.40
        and abs(forward_velocity) < config.forward_overspeed_limit
    )
    forward_velocity_abs = abs(forward_velocity)
    forward_overspeed = max(
        0.0,
        forward_velocity_abs - config.forward_overspeed_limit,
    )
    components = {
        "alive": config.alive_reward,
        "height": height,
        "upright": max(0.0, upright_clipped),
        "stability_gate": stability_gate,
        "progress_gate": progress_gate,
        "pose": pose,
        "foot_flat": foot_flat,
        "lateral_target": lateral_target,
        "lateral_progress": lateral_progress,
        "support_contact": support_contact_reward,
        "double_support_contact": double_support_contact,
        "swing_unload": swing_unload,
        "shift_phase_error_penalty": shift_phase_error_penalty,
        "both_contact_reward": float(both_contact),
        "single_contact_penalty": float(single_contact),
        "no_contact_penalty": float(no_contact),
        "forward_velocity_abs_penalty": forward_velocity_abs,
        "forward_displacement_penalty": abs(current_base_x_displacement),
        "forward_overspeed_penalty": forward_overspeed * forward_overspeed,
        "lateral_velocity_penalty": lateral_velocity * lateral_velocity,
        "action_penalty": action_l2,
        "action_rate_penalty": action_rate_l2,
        "velocity_penalty": joint_velocity_l2,
        "base_roll_pitch_rate_penalty": base_roll_pitch_rate_l2,
        "lateral_error": current_abs_y_error,
    }
    total = components["alive"]
    total += config.height_weight * components["height"]
    total += config.upright_weight * components["upright"]
    total += config.pose_weight * components["pose"]
    total += config.foot_flat_weight * components["foot_flat"]
    total += (
        config.com_shift_lateral_target_weight
        * components["lateral_target"]
        * components["stability_gate"]
    )
    total += (
        config.com_shift_lateral_progress_weight
        * components["lateral_progress"]
        * components["progress_gate"]
    )
    total += (
        config.com_shift_both_contact_reward_weight
        * components["double_support_contact"]
        * components["stability_gate"]
    )
    total += (
        config.com_shift_support_contact_reward_weight
        * components["support_contact"]
        * components["stability_gate"]
    )
    total += (
        config.com_shift_swing_unload_reward_weight
        * components["swing_unload"]
        * components["stability_gate"]
    )
    total -= (
        config.com_shift_shift_phase_error_penalty_weight
        * components["shift_phase_error_penalty"]
    )
    total -= (
        config.com_shift_single_contact_penalty_weight
        * components["single_contact_penalty"]
    )
    total -= (
        config.com_shift_no_contact_penalty_weight
        * components["no_contact_penalty"]
    )
    total -= (
        config.com_shift_forward_velocity_abs_penalty_weight
        * components["forward_velocity_abs_penalty"]
    )
    total -= (
        config.com_shift_forward_displacement_penalty_weight
        * components["forward_displacement_penalty"]
    )
    total -= (
        config.com_shift_forward_overspeed_penalty_weight
        * components["forward_overspeed_penalty"]
    )
    total -= (
        config.com_shift_lateral_velocity_penalty_weight
        * components["lateral_velocity_penalty"]
    )
    total -= config.action_penalty_weight * components["action_penalty"]
    total -= config.action_rate_penalty_weight * components["action_rate_penalty"]
    total -= config.velocity_penalty_weight * components["velocity_penalty"]
    total -= (
        config.base_roll_pitch_rate_penalty_weight
        * components["base_roll_pitch_rate_penalty"]
    )
    components["total"] = float(total)
    return components


class SedonStandingEnv(MujocoEnv):
    """MuJoCo environment for initial Sedon standing/balance experiments.

    Args:
        scene_path: Path to ``training_scene.xml`` generated by
            ``tools.build_sedon_training_scene``.
        frame_skip: Number of MuJoCo solver steps per environment step.
        reward_config: Optional standing reward configuration.
        reset_noise_scale: Uniform noise applied to actuated joint positions.
        **kwargs: Forwarded to ``MujocoEnv`` such as ``render_mode``.

    Raises:
        ModuleNotFoundError: If MuJoCo/Gymnasium MuJoCo dependencies are absent.
        FileNotFoundError: If the private Sedon training scene has not been built.
        ValueError: If the scene does not expose the expected 10 actuators.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        scene_path: Path | str = DEFAULT_SCENE_PATH,
        frame_skip: int = 10,
        reward_config: SedonStandingConfig | None = None,
        reset_noise_scale: float = 0.01,
        **kwargs,
    ) -> None:
        if _MUJOCO_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "SedonStandingEnv requires 'mujoco' and Gymnasium MuJoCo. "
                "Install project requirements before creating the environment."
            ) from _MUJOCO_IMPORT_ERROR

        self._scene_path = Path(scene_path).expanduser().resolve()
        if not self._scene_path.is_file():
            raise FileNotFoundError(
                f"Sedon training scene not found: {self._scene_path}. "
                "Run `python -m tools.convert_urdf_to_mjcf` and then "
                "`python -m tools.build_sedon_training_scene` first."
            )
        if reset_noise_scale < 0.0:
            raise ValueError("reset_noise_scale must be non-negative.")

        self._reward_config = reward_config or load_sedon_config_from_env()
        self._reset_noise_scale = reset_noise_scale
        self._prev_action = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self._prev_base_x = 0.0
        self._episode_base_x = 0.0
        self._gait_step = 0

        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(43,),
            dtype=np.float64,
        )
        super().__init__(
            model_path=str(self._scene_path),
            frame_skip=frame_skip,
            observation_space=observation_space,
            default_camera_config={
                "distance": 1.4,
                "lookat": np.array([0.0, 0.0, 0.35]),
                "elevation": -18.0,
                "azimuth": 135.0,
            },
            **kwargs,
        )

        if self.model.nu != len(JOINT_NAMES):
            raise ValueError(
                f"Expected {len(JOINT_NAMES)} Sedon actuators, got {self.model.nu}."
            )

        self._base_body_id = self._body_id("base_link")
        self._joint_ids = [self._joint_id(name) for name in JOINT_NAMES]
        self._foot_geom_ids = [self._geom_id(name) for name in FOOT_GEOM_NAMES]
        self._floor_geom_id = self._geom_id(FLOOR_GEOM_NAME)
        self._base_proxy_geom_id = self._geom_id(BASE_PROXY_GEOM_NAME)
        self._ctrl_range = self.model.actuator_ctrlrange.copy()
        self._last_ctrl_assist_delta = np.zeros(self.model.nu, dtype=np.float64)
        self._default_qpos = self.init_qpos.copy()
        self._default_qvel = self.init_qvel.copy()
        self._set_base_pose(self._default_qpos)
        self._nominal_joint_qpos = (
            self._extract_joint_positions(self._default_qpos)
            + self._nominal_joint_pose_offsets()
        )

        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(len(JOINT_NAMES),),
            dtype=np.float64,
        )

    def step(self, action: np.ndarray):
        """Advance the Sedon standing simulation by one RL step."""
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != self.action_space.shape:
            raise ValueError(
                f"Sedon action must have shape {self.action_space.shape}, "
                f"got {action_array.shape}."
            )
        clipped_action = np.clip(action_array, -1.0, 1.0).astype(np.float64)
        task_mode = self._reward_config.task_mode.lower()
        task_phase = self._task_phase_metadata()
        previous_base_x = float(self.data.qpos[0])
        previous_base_y = float(self.data.qpos[1])
        desired_base_y = 0.0
        previous_abs_y_error = 0.0
        if task_mode == "com_shift":
            desired_base_y = self._desired_base_y(task_phase)
            previous_abs_y_error = abs(previous_base_y - desired_base_y)
        gait_target = self._nominal_joint_qpos + self._gait_joint_offsets()
        target_positions = (
            gait_target
            + clipped_action * self._reward_config.action_joint_delta_scale
        )
        target_positions = self._apply_safe_joint_target_clamps(target_positions)
        self._do_pd_simulation(target_positions)
        self._gait_step += 1

        obs = self._get_obs()
        current_base_x = float(self.data.qpos[0])
        progress_delta_x = current_base_x - self._prev_base_x
        cumulative_progress_x = current_base_x - self._episode_base_x
        base_height = self._base_height()
        upright = self._base_upright()
        joint_positions = self._joint_positions()
        joint_velocities = self._joint_velocities()
        joint_velocity_l2 = float(np.dot(joint_velocities, joint_velocities))
        action_l2 = float(np.dot(clipped_action, clipped_action))
        action_delta = clipped_action - self._prev_action
        action_rate_l2 = float(np.dot(action_delta, action_delta))
        joint_position_error = joint_positions - gait_target
        joint_position_error_l2 = float(np.dot(joint_position_error, joint_position_error))
        base_xy_velocity = self.data.qvel[0:2]
        forward_velocity = float(base_xy_velocity[0])
        lateral_velocity_l2 = float(base_xy_velocity[1] * base_xy_velocity[1])
        base_xy_velocity_l2 = float(np.dot(base_xy_velocity, base_xy_velocity))
        base_roll_pitch_rate = self.data.qvel[3:5]
        base_roll_pitch_rate_l2 = float(np.dot(base_roll_pitch_rate, base_roll_pitch_rate))
        foot_flatness = self._foot_flatness()
        foot_height_error_l2 = self._foot_height_error_l2()
        feet_near_floor = self._feet_near_floor()
        contact_flags = self._floor_contact_flags()
        knee_violation = self._knee_safe_range_violation()
        base_y = float(self.data.qpos[1])
        support_contact = False
        swing_contact = False
        support_foot_bottom_z = float("nan")
        swing_foot_bottom_z = float("nan")
        if task_mode == "com_shift":
            support_contact, swing_contact = self._task_support_contacts(task_phase, contact_flags)
            support_foot_bottom_z, swing_foot_bottom_z = self._task_foot_bottom_heights(
                task_phase
            )
            rewards = compute_com_shift_reward(
                base_height=base_height,
                upright=upright,
                joint_velocity_l2=joint_velocity_l2,
                action_l2=action_l2,
                action_rate_l2=action_rate_l2,
                joint_position_error_l2=joint_position_error_l2,
                forward_velocity=forward_velocity,
                lateral_velocity=float(base_xy_velocity[1]),
                base_roll_pitch_rate_l2=base_roll_pitch_rate_l2,
                foot_flatness=foot_flatness,
                base_y=base_y,
                target_base_y=desired_base_y,
                previous_abs_y_error=previous_abs_y_error,
                current_base_x_displacement=current_base_x - self._episode_base_x,
                in_double_support=bool(task_phase["in_double_support"]),
                support_contact=support_contact,
                swing_contact=swing_contact,
                support_foot_bottom_z=support_foot_bottom_z,
                swing_foot_bottom_z=swing_foot_bottom_z,
                config=self._reward_config,
            )
        else:
            rewards = compute_standing_reward(
                base_height=base_height,
                upright=upright,
                joint_velocity_l2=joint_velocity_l2,
                action_l2=action_l2,
                action_rate_l2=action_rate_l2,
                joint_position_error_l2=joint_position_error_l2,
                forward_velocity=forward_velocity,
                lateral_velocity_l2=lateral_velocity_l2,
                base_xy_velocity_l2=base_xy_velocity_l2,
                base_roll_pitch_rate_l2=base_roll_pitch_rate_l2,
                foot_flatness=foot_flatness,
                foot_height_error_l2=foot_height_error_l2,
                feet_near_floor=feet_near_floor,
                config=self._reward_config,
                progress_delta_x=progress_delta_x,
                cumulative_progress_x=cumulative_progress_x,
                elapsed_steps=self._gait_step,
            )
        terminated = self._is_terminated(base_height, upright, obs)
        if terminated:
            rewards["total"] -= self._reward_config.termination_penalty

        info = {
            "base_height": base_height,
            "base_x_position": current_base_x,
            "progress_delta_x": progress_delta_x,
            "cumulative_progress_x": cumulative_progress_x,
            "upright": upright,
            "joint_velocity_l2": joint_velocity_l2,
            "action_l2": action_l2,
            "action_rate_l2": action_rate_l2,
            "joint_position_error_l2": joint_position_error_l2,
            "forward_velocity": forward_velocity,
            "lateral_velocity_l2": lateral_velocity_l2,
            "base_xy_velocity_l2": base_xy_velocity_l2,
            "base_y_position": base_y,
            "base_roll_pitch_rate_l2": base_roll_pitch_rate_l2,
            "gait_phase": self._gait_phase(),
            "foot_flatness": foot_flatness,
            "foot_height_error_l2": foot_height_error_l2,
            "feet_near_floor": feet_near_floor,
            "task_mode": task_mode,
            "phase_name": task_phase["phase_name"],
            "support_side": task_phase["support_side"],
            "in_double_support": task_phase["in_double_support"],
            "desired_base_y": desired_base_y,
            "right_contact": contact_flags["right"],
            "left_contact": contact_flags["left"],
            "base_proxy_floor_contact": contact_flags["base_proxy"],
            "support_contact": support_contact,
            "swing_contact": swing_contact,
            "support_foot_bottom_z": support_foot_bottom_z,
            "swing_foot_bottom_z": swing_foot_bottom_z,
            "right_knee_qpos": float(joint_positions[RIGHT_KNEE_JOINT_INDEX]),
            "left_knee_qpos": float(joint_positions[LEFT_KNEE_JOINT_INDEX]),
            "right_knee_safe_violation": float(knee_violation["right"]),
            "left_knee_safe_violation": float(knee_violation["left"]),
            "knee_safe_violation_sum": float(knee_violation["total"]),
        }
        for key, value in rewards.items():
            info[f"reward_{key}"] = value

        self._prev_action = clipped_action.copy()
        self._prev_base_x = current_base_x
        return obs, float(rewards["total"]), terminated, False, info

    def reset_model(self) -> np.ndarray:
        """Reset the floating base and actuated joints to a standing seed pose."""
        qpos = self._default_qpos.copy()
        qvel = self._default_qvel.copy()
        self._set_base_pose(qpos)

        for joint_index, joint_id in enumerate(self._joint_ids):
            qpos_adr = self.model.jnt_qposadr[joint_id]
            qpos[qpos_adr] = self._nominal_joint_qpos[joint_index]
            if self._reset_noise_scale > 0.0:
                qpos[qpos_adr] += self.np_random.uniform(
                    -self._reset_noise_scale,
                    self._reset_noise_scale,
                )

        qvel[:] = 0.0
        self.set_state(qpos, qvel)
        self._prev_action = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self._last_ctrl_assist_delta.fill(0.0)
        self._prev_base_x = float(self.data.qpos[0])
        self._episode_base_x = self._prev_base_x
        self._gait_step = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Return base state, joint state, and previous action."""
        base_quat = self.data.qpos[3:7].copy()
        base_velocity = self.data.qvel[0:6].copy()
        obs = np.concatenate(
            [
                np.array([self._base_height()], dtype=np.float64),
                base_quat,
                base_velocity,
                np.array(
                    [np.sin(self._gait_phase()), np.cos(self._gait_phase())],
                    dtype=np.float64,
                ),
                self._joint_positions(),
                self._joint_velocities(),
                self._prev_action,
            ]
        )
        return obs.astype(np.float64)

    def _set_base_pose(self, qpos: np.ndarray) -> None:
        """Set the free base to the configured starting height and identity rotation."""
        qpos[0:3] = np.array([0.0, 0.0, self._reward_config.target_base_height])
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

    def _is_terminated(
        self,
        base_height: float,
        upright: float,
        observation: np.ndarray,
    ) -> bool:
        """Return whether the current state is no longer a valid standing episode."""
        if not np.isfinite(observation).all():
            return True
        if base_height < self._reward_config.min_base_height:
            return True
        if base_height > self._reward_config.max_base_height:
            return True
        if float(np.linalg.norm(self.data.qpos[0:2])) > self._reward_config.max_base_xy_drift:
            return True
        return bool(upright < self._reward_config.min_upright)

    def _base_height(self) -> float:
        """Return the floating base body height."""
        return float(self.data.xpos[self._base_body_id][2])

    def _base_upright(self) -> float:
        """Return alignment between base local z-axis and world z-axis."""
        xmat = self.data.xmat[self._base_body_id].reshape(3, 3)
        return float(xmat[2, 2])

    def _joint_positions(self) -> np.ndarray:
        """Return actuated joint positions in stable order."""
        return np.array(
            [self.data.qpos[self.model.jnt_qposadr[joint_id]] for joint_id in self._joint_ids],
            dtype=np.float64,
        )

    def _joint_velocities(self) -> np.ndarray:
        """Return actuated joint velocities in stable order."""
        return np.array(
            [self.data.qvel[self.model.jnt_dofadr[joint_id]] for joint_id in self._joint_ids],
            dtype=np.float64,
        )

    def _extract_joint_positions(self, qpos: np.ndarray) -> np.ndarray:
        """Return actuated joint positions from a MuJoCo qpos vector."""
        return np.array(
            [qpos[self.model.jnt_qposadr[joint_id]] for joint_id in self._joint_ids],
            dtype=np.float64,
        )

    def _pd_control(self, target_positions: np.ndarray) -> np.ndarray:
        """Return a clipped torque command for the current joint state."""
        pd_ctrl = (
            self._reward_config.pd_stiffness * (target_positions - self._joint_positions())
            - self._reward_config.pd_damping * self._joint_velocities()
        )
        scaled_ctrl = np.clip(
            pd_ctrl,
            -self._reward_config.torque_scale,
            self._reward_config.torque_scale,
        )
        return np.clip(scaled_ctrl, self._ctrl_range[:, 0], self._ctrl_range[:, 1])

    def _gait_phase(self) -> float:
        """Return the current built-in gait phase in radians."""
        cycle_steps = self._task_cycle_steps()
        return float(2.0 * np.pi * (self._gait_step % cycle_steps) / cycle_steps)

    def _nominal_joint_pose_offsets(self) -> np.ndarray:
        """Return the symmetric standing offsets applied on top of scene defaults."""
        offsets = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        offsets[2] = self._reward_config.nominal_hip_pitch_offset
        offsets[3] = self._reward_config.nominal_knee_pitch_offset
        offsets[4] = self._reward_config.nominal_ankle_pitch_offset
        offsets[7] = self._reward_config.nominal_hip_pitch_offset
        offsets[8] = self._reward_config.nominal_knee_pitch_offset
        offsets[9] = self._reward_config.nominal_ankle_pitch_offset
        return offsets

    def _gait_joint_offsets(self) -> np.ndarray:
        """Return a small periodic joint target that seeds forward stepping."""
        if self._reward_config.task_mode.lower() == "com_shift":
            return self._com_shift_joint_offsets()
        mode = self._reward_config.gait_mode.lower()
        if mode == "fsm":
            return self._fsm_gait_joint_offsets()
        if mode == "sin":
            return self._sin_gait_joint_offsets()
        raise ValueError(f"Unsupported Sedon gait_mode: {self._reward_config.gait_mode}")

    def _task_cycle_steps(self) -> int:
        """Return the active cycle length for the configured task."""
        if self._reward_config.task_mode.lower() == "com_shift":
            return max(
                1,
                2 * self._reward_config.com_shift_center_hold_steps
                + 2 * self._reward_config.com_shift_shift_steps,
            )
        return max(1, self._reward_config.gait_cycle_steps)

    @staticmethod
    def _smoothstep(value: float) -> float:
        """Return a smooth 0..1 transition value."""
        clipped = float(np.clip(value, 0.0, 1.0))
        return clipped * clipped * (3.0 - 2.0 * clipped)

    def _sin_gait_joint_offsets(self) -> np.ndarray:
        """Return the legacy sinusoidal deterministic gait target."""
        phase = self._gait_phase()
        swing = np.sin(phase)
        cycle_steps = self._task_cycle_steps()
        warmup = min(1.0, self._gait_step / cycle_steps)
        right_swing = max(0.0, swing) ** 2 * warmup
        left_swing = max(0.0, -swing) ** 2 * warmup
        lateral_sway = swing * warmup
        offsets = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        offsets[1] = (
            self._reward_config.gait_hip_roll_amp
            * self._reward_config.gait_right_hip_roll_sign
            * lateral_sway
        )
        offsets[2] = self._reward_config.gait_hip_pitch_amp * right_swing
        offsets[3] = self._reward_config.gait_knee_pitch_amp * right_swing
        offsets[4] = self._reward_config.gait_ankle_pitch_amp * right_swing
        offsets[6] = (
            self._reward_config.gait_hip_roll_amp
            * self._reward_config.gait_left_hip_roll_sign
            * lateral_sway
        )
        offsets[7] = self._reward_config.gait_hip_pitch_amp * left_swing
        offsets[8] = self._reward_config.gait_knee_pitch_amp * left_swing
        offsets[9] = self._reward_config.gait_ankle_pitch_amp * left_swing
        return offsets

    def _com_shift_joint_offsets(self) -> np.ndarray:
        """Return a conservative deterministic seed for COM shift and foot unload."""
        task_phase = self._task_phase_metadata()
        phase_name = str(task_phase["phase_name"])
        if phase_name.startswith("center_hold"):
            return np.zeros(len(JOINT_NAMES), dtype=np.float64)
        shift_alpha = float(task_phase["shift_alpha"])
        if phase_name == "right_shift":
            return self._com_shift_support_pose("right", unload_scale=0.0) * shift_alpha
        if phase_name == "left_shift":
            return self._com_shift_support_pose("left", unload_scale=0.0) * shift_alpha
        return np.zeros(len(JOINT_NAMES), dtype=np.float64)

    def _com_shift_support_pose(
        self,
        support_side: str,
        *,
        unload_scale: float,
    ) -> np.ndarray:
        """Return a simple support-side target for COM shift practice."""
        offsets = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        support_roll = self._reward_config.com_shift_support_roll_amp
        hip_pitch = self._reward_config.com_shift_unload_hip_pitch_amp * unload_scale
        knee_pitch = self._reward_config.com_shift_unload_knee_pitch_amp * unload_scale
        ankle_pitch = self._reward_config.com_shift_unload_ankle_pitch_amp * unload_scale
        if support_side == "right":
            offsets[1] += support_roll
            offsets[6] += -support_roll
            offsets[7] += hip_pitch
            offsets[8] += knee_pitch
            offsets[9] += ankle_pitch
            return offsets
        if support_side == "left":
            offsets[1] += -support_roll
            offsets[6] += support_roll
            offsets[2] += hip_pitch
            offsets[3] += knee_pitch
            offsets[4] += ankle_pitch
            return offsets
        raise ValueError(f"Unsupported support_side: {support_side}")

    def _task_phase_metadata(self) -> dict[str, object]:
        """Return the current curriculum phase and designated support side."""
        if self._reward_config.task_mode.lower() != "com_shift":
            return {
                "phase_name": "walk",
                "support_side": "none",
                "in_double_support": False,
                "shift_alpha": 0.0,
            }
        hold_steps = max(1, self._reward_config.com_shift_center_hold_steps)
        shift_steps = max(1, self._reward_config.com_shift_shift_steps)
        cycle_steps = self._task_cycle_steps()
        phase_step = self._gait_step % cycle_steps
        if phase_step < hold_steps:
            return {
                "phase_name": "center_hold_right",
                "support_side": "center",
                "in_double_support": True,
                "shift_alpha": 0.0,
            }
        phase_step -= hold_steps
        if phase_step < shift_steps:
            return {
                "phase_name": "right_shift",
                "support_side": "right",
                "in_double_support": True,
                "shift_alpha": self._smoothstep((phase_step + 1) / shift_steps),
            }
        phase_step -= shift_steps
        if phase_step < hold_steps:
            return {
                "phase_name": "center_hold_left",
                "support_side": "center",
                "in_double_support": True,
                "shift_alpha": 0.0,
            }
        phase_step -= hold_steps
        return {
            "phase_name": "left_shift",
            "support_side": "left",
            "in_double_support": True,
            "shift_alpha": self._smoothstep((phase_step + 1) / shift_steps),
        }

    def _fsm_gait_joint_offsets(self) -> np.ndarray:
        """Return FSM gait with explicit lift/lower phases."""
        right_lift_steps = max(1, self._reward_config.fsm_right_lift_steps)
        right_lower_steps = max(1, self._reward_config.fsm_right_lower_steps)
        left_lift_steps = max(1, self._reward_config.fsm_left_lift_steps)
        left_lower_steps = max(1, self._reward_config.fsm_left_lower_steps)
        double_support_steps = max(0, self._reward_config.fsm_double_support_steps)

        right_swing_scale = self._reward_config.fsm_right_swing_scale
        left_swing_scale = self._reward_config.fsm_left_swing_scale
        right_support_roll_scale = self._reward_config.fsm_right_support_roll_scale
        left_support_roll_scale = self._reward_config.fsm_left_support_roll_scale

        total_steps = (
            right_lift_steps
            + right_lower_steps
            + double_support_steps
            + left_lift_steps
            + left_lower_steps
            + double_support_steps
        )

        phase_step = self._gait_step % total_steps
        offsets = np.zeros(len(JOINT_NAMES), dtype=np.float64)

        support_roll = self._reward_config.gait_hip_roll_amp
        hip_pitch = self._reward_config.gait_hip_pitch_amp
        knee_pitch = self._reward_config.gait_knee_pitch_amp
        ankle_pitch = self._reward_config.gait_ankle_pitch_amp
        swing_cap = self._reward_config.fsm_swing_cap

        def capped(value: float) -> float:
            return min(value, swing_cap)

        def apply_right_swing(swing: float) -> None:
            # left support, right swing
            scaled_swing = right_swing_scale * swing
            offsets[6] += support_roll * right_support_roll_scale * swing
            offsets[2] += hip_pitch * scaled_swing
            offsets[3] += knee_pitch * scaled_swing
            offsets[4] += ankle_pitch * scaled_swing

        def apply_left_swing(swing: float) -> None:
            # right support, left swing
            scaled_swing = left_swing_scale * swing
            offsets[1] += -(support_roll * left_support_roll_scale) * swing
            offsets[7] += hip_pitch * scaled_swing
            offsets[8] += knee_pitch * scaled_swing
            offsets[9] += ankle_pitch * scaled_swing

        # Phase 1: right lift
        if phase_step < right_lift_steps:
            s = capped(self._smoothstep((phase_step + 1) / right_lift_steps))
            apply_right_swing(s)
            return offsets

        phase_step -= right_lift_steps

        # Phase 2: right lower
        if phase_step < right_lower_steps:
            s = capped(
                1.0 - self._smoothstep((phase_step + 1) / right_lower_steps)
            )
            apply_right_swing(s)
            return offsets

        phase_step -= right_lower_steps

        # Phase 3: double support
        if phase_step < double_support_steps:
            return offsets

        phase_step -= double_support_steps

        # Phase 4: left lift
        if phase_step < left_lift_steps:
            s = capped(self._smoothstep((phase_step + 1) / left_lift_steps))
            apply_left_swing(s)
            return offsets

        phase_step -= left_lift_steps

        # Phase 5: left lower
        if phase_step < left_lower_steps:
            s = capped(1.0 - self._smoothstep((phase_step + 1) / left_lower_steps))
            apply_left_swing(s)
            return offsets

        return offsets

    def _do_pd_simulation(self, target_positions: np.ndarray) -> None:
        """Step MuJoCo while refreshing the stance PD torque every physics step."""
        self._run_pd_simulation(target_positions)

    def _do_pd_simulation_with_torque_assist(
        self,
        target_positions: np.ndarray,
        left_tau_assist: float,
        right_tau_assist: float,
    ) -> None:
        """Step MuJoCo using the normal PD path plus hip-roll torque assist."""
        self._run_pd_simulation(
            target_positions,
            left_tau_assist=float(left_tau_assist),
            right_tau_assist=float(right_tau_assist),
        )

    def _run_pd_simulation(
        self,
        target_positions: np.ndarray,
        *,
        left_tau_assist: float = 0.0,
        right_tau_assist: float = 0.0,
    ) -> None:
        """Run the per-physics-step PD update loop used by Sedon control."""
        self._last_ctrl_assist_delta.fill(0.0)
        for _ in range(self.frame_skip):
            ctrl, injected_delta = self._apply_hip_roll_torque_assist(
                self._pd_control(target_positions),
                left_tau_assist=left_tau_assist,
                right_tau_assist=right_tau_assist,
            )
            self.data.ctrl[:] = ctrl
            self._last_ctrl_assist_delta[:] = injected_delta
            mujoco.mj_step(self.model, self.data)

    def _apply_hip_roll_torque_assist(
        self,
        ctrl: np.ndarray,
        *,
        left_tau_assist: float,
        right_tau_assist: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply hip-roll torque assist and return clamped ctrl plus actual delta."""
        base_ctrl = np.asarray(ctrl, dtype=np.float64)
        assisted_ctrl = base_ctrl.copy()
        assisted_ctrl[LEFT_HIP_ROLL_ACTUATOR_INDEX] += float(left_tau_assist)
        assisted_ctrl[RIGHT_HIP_ROLL_ACTUATOR_INDEX] += float(right_tau_assist)
        assisted_ctrl = np.clip(
            assisted_ctrl,
            self._ctrl_range[:, 0],
            self._ctrl_range[:, 1],
        )
        return assisted_ctrl, assisted_ctrl - base_ctrl

    def last_hip_roll_ctrl_assist_delta(self) -> tuple[float, float]:
        """Return the last clamp-adjusted injected ctrl delta for left/right hip roll."""
        return (
            float(self._last_ctrl_assist_delta[LEFT_HIP_ROLL_ACTUATOR_INDEX]),
            float(self._last_ctrl_assist_delta[RIGHT_HIP_ROLL_ACTUATOR_INDEX]),
        )

    def _foot_bottom_heights(self) -> np.ndarray:
        """Return estimated bottom heights for the collision foot boxes."""
        heights = []
        for geom_id in self._foot_geom_ids:
            geom_z = self.data.geom_xpos[geom_id][2]
            half_height = self.model.geom_size[geom_id][2]
            heights.append(float(geom_z - half_height))
        return np.array(heights, dtype=np.float64)

    def _foot_flatness(self) -> float:
        """Return mean alignment between foot box z-axes and world z-axis."""
        alignments = []
        for geom_id in self._foot_geom_ids:
            xmat = self.data.geom_xmat[geom_id].reshape(3, 3)
            alignments.append(float(xmat[2, 2]))
        return float(np.mean(alignments))

    def _foot_height_error_l2(self) -> float:
        """Return squared foot bottom height error from the floor plane."""
        foot_bottom_heights = self._foot_bottom_heights()
        return float(np.dot(foot_bottom_heights, foot_bottom_heights))

    def _feet_near_floor(self) -> int:
        """Return how many feet are close enough to the floor for standing."""
        return int(np.count_nonzero(np.abs(self._foot_bottom_heights()) <= 0.015))

    def _joint_range(self, joint_index: int) -> tuple[float, float]:
        """Return the MuJoCo qpos range for one actuated joint index."""
        joint_id = self._joint_ids[joint_index]
        lower, upper = self.model.jnt_range[joint_id]
        return float(lower), float(upper)

    def knee_safe_ranges(self) -> dict[str, tuple[float, float] | None]:
        """Return configured soft-safe knee qpos bounds, or ``None`` when unset."""
        ranges = {
            "right": (
                self._reward_config.right_knee_safe_lower,
                self._reward_config.right_knee_safe_upper,
            ),
            "left": (
                self._reward_config.left_knee_safe_lower,
                self._reward_config.left_knee_safe_upper,
            ),
        }
        return {
            leg: None if lower is None or upper is None else (float(lower), float(upper))
            for leg, (lower, upper) in ranges.items()
        }

    def _range_violation(self, value: float, allowed_range: tuple[float, float] | None) -> float:
        """Return distance outside an optional inclusive safe range."""
        if allowed_range is None:
            return 0.0
        lower, upper = allowed_range
        if value < lower:
            return float(lower - value)
        if value > upper:
            return float(value - upper)
        return 0.0

    def _knee_safe_range_violation(self) -> dict[str, float]:
        """Return per-knee violation magnitudes against configured soft-safe bounds."""
        joint_positions = self._joint_positions()
        safe_ranges = self.knee_safe_ranges()
        right = self._range_violation(
            float(joint_positions[RIGHT_KNEE_JOINT_INDEX]),
            safe_ranges["right"],
        )
        left = self._range_violation(
            float(joint_positions[LEFT_KNEE_JOINT_INDEX]),
            safe_ranges["left"],
        )
        return {"right": right, "left": left, "total": right + left}

    def _apply_safe_joint_target_clamps(self, target_positions: np.ndarray) -> np.ndarray:
        """Clamp knee targets into configured soft-safe ranges when provided."""
        clamped = np.asarray(target_positions, dtype=np.float64).copy()
        safe_ranges = self.knee_safe_ranges()
        for leg_name, joint_index in (
            ("right", RIGHT_KNEE_JOINT_INDEX),
            ("left", LEFT_KNEE_JOINT_INDEX),
        ):
            allowed_range = safe_ranges[leg_name]
            if allowed_range is None:
                continue
            clamped[joint_index] = float(np.clip(clamped[joint_index], *allowed_range))
        return clamped

    def _floor_contact_flags(self) -> dict[str, bool]:
        """Return whether each key geom currently contacts the floor."""
        flags = {"right": False, "left": False, "base_proxy": False}
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair == {self._floor_geom_id, self._foot_geom_ids[0]}:
                flags["right"] = True
            elif pair == {self._floor_geom_id, self._foot_geom_ids[1]}:
                flags["left"] = True
            elif pair == {self._floor_geom_id, self._base_proxy_geom_id}:
                flags["base_proxy"] = True
        return flags

    def _desired_base_y(self, task_phase: dict[str, object]) -> float:
        """Return the desired base lateral position for the current task phase."""
        phase_name = str(task_phase["phase_name"])
        target = self._reward_config.com_shift_lateral_target_magnitude
        if phase_name == "right_shift":
            return target
        if phase_name == "left_shift":
            return -target
        return 0.0

    def _task_support_contacts(
        self,
        task_phase: dict[str, object],
        contact_flags: dict[str, bool],
    ) -> tuple[bool, bool]:
        """Return support/swing contact booleans for the current task phase."""
        support_side = str(task_phase["support_side"])
        if support_side == "right":
            return contact_flags["right"], contact_flags["left"]
        if support_side == "left":
            return contact_flags["left"], contact_flags["right"]
        both_contact = contact_flags["right"] and contact_flags["left"]
        return both_contact, both_contact

    def _task_foot_bottom_heights(
        self,
        task_phase: dict[str, object],
    ) -> tuple[float, float]:
        """Return support and swing foot bottom heights for the current task phase."""
        foot_bottom_heights = self._foot_bottom_heights()
        support_side = str(task_phase["support_side"])
        if support_side == "right":
            return float(foot_bottom_heights[0]), float(foot_bottom_heights[1])
        if support_side == "left":
            return float(foot_bottom_heights[1]), float(foot_bottom_heights[0])
        mean_height = float(np.mean(foot_bottom_heights))
        return mean_height, mean_height

    def _body_id(self, name: str) -> int:
        """Resolve a MuJoCo body id by name."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body '{name}' not found in Sedon model.")
        return body_id

    def _geom_id(self, name: str) -> int:
        """Resolve a MuJoCo geom id by name."""
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"Geom '{name}' not found in Sedon model.")
        return geom_id

    def _joint_id(self, name: str) -> int:
        """Resolve a MuJoCo joint id by name."""
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint '{name}' not found in Sedon model.")
        return joint_id
