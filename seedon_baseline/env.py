"""Minimal MuJoCo locomotion environment for the private Seedon robot model."""

from __future__ import annotations

import json
import math
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
DEFAULT_SCENE_PATH = REPO_ROOT / "private_assets" / "seedon_v5_22" / "training_scene.xml"
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
CONFIG_OVERRIDES_ENV = "SEEDON_CONFIG_OVERRIDES"


@dataclass(frozen=True)
class ReferenceGaitKeyframe:
    """One pose-editor keyframe used as a conservative gait reference."""

    name: str
    support_mode: str
    joint_targets: np.ndarray
    duration_steps: int


@dataclass(frozen=True)
class ReferenceGaitSeed:
    """Validated reference gait seed loaded from JSON."""

    keyframes: tuple[ReferenceGaitKeyframe, ...]
    target_type: str
    cycle_steps: int


@dataclass(frozen=True)
class SeedonStandingConfig:
    """Reward and termination settings for the Seedon locomotion task.

    Args:
        target_base_height: Desired base height in meters.
        target_forward_velocity: Desired base x velocity in meters per second.
        min_base_height: Episode terminates below this height.
        max_base_height: Episode terminates above this height.
        min_upright: Episode terminates below this base upright alignment.
        termination_penalty: Penalty applied when an episode terminates early.
        torque_scale: Maximum absolute PD torque command before actuator clipping.
        action_joint_delta_scale: Maximum joint target offset represented by action 1.0.
        task_mode: High-level task objective, such as ``"walk"``,
            ``"com_shift"``, ``"march_in_place"``, or ``"reference_march"``.
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
        blue_support_ratio_weight: Weight for Blue-style support-side load transfer in walk mode.
        blue_swing_unload_weight: Weight for unloading the swing foot during Blue-style walking.
        blue_clearance_weight: Weight for keeping swing clearance low but nonzero.
        blue_target_support_ratio: Desired support-foot normal-force fraction.
        blue_target_swing_ratio: Desired swing-foot normal-force fraction.
        blue_target_clearance: Desired low swing-foot clearance in meters.
        blue_max_clearance: Clearance above this value is treated as hopping.
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
        episode_action_mean_penalty_weight: Penalty coefficient for persistent
            residual bias accumulated over an episode.
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
        march_curriculum_phase: March-in-place curriculum phase from 1 to 5.
        march_phase_steps: RL steps for each scripted march phase-clock segment.
        march_load_transfer_ratio: Target support-foot force fraction for load transfer.
        march_swing_unload_ratio: Target swing-foot force fraction for unload.
        march_micro_lift_height: Target swing-foot clearance for phase 3+.
        march_forward_velocity: Small forward target enabled only in phase 5.
        march_forward_progress_weight: Reward weight for positive forward velocity
            in phase 5.
        march_swing_forward_distance: Desired swing-foot x lead over the support
            foot in phase 5.
        march_swing_forward_weight: Reward weight for small forward swing-foot
            placement in phase 5.
        march_no_contact_penalty_weight: Penalty when both feet lose floor load.
        march_single_contact_penalty_weight: Penalty when only one foot has floor
            load while the task expects a low shuffle.
        march_support_force_error_penalty_weight: Penalty coefficient for
            missing the phase-designated support foot load target.
        march_swing_force_penalty_weight: Penalty coefficient for keeping
            too much load on the swing foot in phase 3+.
        march_residual_mode: Residual action mask mode, ``"all"`` or ``"hip_roll_only"``.
        march_require_both_contact: Terminate march episodes when either foot leaves contact.
        autonomy_tracking_variance_penalty_weight: Penalty coefficient for
            episode tracking-error variance during autonomy curricula.
        autonomy_contact_transition_penalty_weight: Penalty coefficient for
            contact state changes during autonomy curricula.
        autonomy_landing_impact_penalty_weight: Extra landing impact penalty
            used when policy autonomy is enabled.
        autonomy_base_height_drop_penalty_weight: Extra base-height drop
            penalty used when policy autonomy is enabled.
        reference_gait_seed_path: Optional pose-editor gait seed used by
            ``reference_march``.
        reference_gait_seed_scale: Multiplier applied to seed targets before
            tracking. Keep below 1.0 for cautious physical training.
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
    blue_support_ratio_weight: float = 0.0
    blue_swing_unload_weight: float = 0.0
    blue_clearance_weight: float = 0.0
    blue_target_support_ratio: float = 0.62
    blue_target_swing_ratio: float = 0.35
    blue_target_clearance: float = 0.006
    blue_max_clearance: float = 0.02
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
    episode_action_mean_penalty_weight: float = 0.0
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
    march_curriculum_phase: int = 1
    march_phase_steps: int = 60
    march_load_transfer_ratio: float = 0.62
    march_swing_unload_ratio: float = 0.25
    march_micro_lift_height: float = 0.005
    march_forward_velocity: float = 0.025
    march_force_ratio_weight: float = 6.0
    march_com_y_weight: float = 4.0
    march_base_roll_penalty_weight: float = 3.0
    march_clearance_weight: float = 6.0
    march_forward_progress_weight: float = 0.0
    march_forward_velocity_weight: float = 3.0
    march_swing_forward_distance: float = 0.0
    march_swing_forward_weight: float = 0.0
    march_no_contact_penalty_weight: float = 0.0
    march_single_contact_penalty_weight: float = 0.0
    march_forward_velocity_abs_penalty_weight: float = 8.0
    march_base_drift_penalty_weight: float = 8.0
    march_torque_saturation_penalty_weight: float = 2.0
    march_base_pitch_penalty_weight: float = 8.0
    march_both_contact_reward_weight: float = 4.0
    march_support_ratio_baseline: float = 0.518
    march_support_ratio_progress_weight: float = 8.0
    march_target_side_force_ratio_weight: float = 2.0
    march_support_force_error_penalty_weight: float = 0.0
    march_swing_force_penalty_weight: float = 0.0
    march_clearance_support_ratio_gate: float = 0.0
    march_contact_none_terminate_steps: int = 0
    march_jump_penalty_weight: float = 0.0
    march_base_height_drop_penalty_weight: float = 0.0
    march_base_height_drop_limit: float = 0.015
    march_landing_impact_penalty_weight: float = 0.0
    march_landing_impact_limit: float = 1.3
    march_residual_mode: str = "all"
    march_require_both_contact: bool = False
    autonomy_tracking_variance_penalty_weight: float = 0.0
    autonomy_contact_transition_penalty_weight: float = 0.0
    autonomy_landing_impact_penalty_weight: float = 0.0
    autonomy_base_height_drop_penalty_weight: float = 0.0
    reference_gait_seed_path: str | None = None
    reference_gait_seed_scale: float = 0.6


def load_seedon_config_from_env() -> SeedonStandingConfig:
    """Return ``SeedonStandingConfig`` with optional JSON environment overrides.

    The ``SEEDON_CONFIG_OVERRIDES`` value may be either a JSON object string or
    a path to a JSON file. Unknown keys are rejected to catch misspelled sweep
    parameters early.
    """
    config = SeedonStandingConfig()
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

    valid_keys = set(SeedonStandingConfig.__dataclass_fields__)
    unknown_keys = sorted(set(overrides) - valid_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown Seedon config override(s): {', '.join(unknown_keys)}"
        )
    return SeedonStandingConfig(**{**config.__dict__, **overrides})


def load_reference_gait_seed(path: Path) -> ReferenceGaitSeed:
    """Load a pose-editor gait seed for reference-march training.

    Args:
        path: JSON seed path using the ``seedon_gait_seed.v1`` schema.

    Returns:
        Validated reference seed with NumPy joint target arrays.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the JSON schema or joint list is incompatible.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Reference gait seed not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "seedon_gait_seed.v1":
        raise ValueError(f"Unsupported reference gait seed schema: {path}")
    if tuple(payload.get("joint_names", [])) != JOINT_NAMES:
        raise ValueError(f"Reference gait seed joint_names do not match Seedon joints: {path}")
    target_type = str(payload.get("target_type", "absolute")).lower()
    if target_type not in {"absolute", "offset"}:
        raise ValueError(f"Unsupported reference gait seed target_type: {target_type}")
    keyframes: list[ReferenceGaitKeyframe] = []
    for raw_keyframe in payload.get("keyframes", []):
        targets = np.asarray(raw_keyframe.get("joint_targets", []), dtype=np.float64)
        if targets.shape != (len(JOINT_NAMES),):
            raise ValueError(f"Invalid reference keyframe joint_targets in {path}")
        keyframes.append(
            ReferenceGaitKeyframe(
                name=str(raw_keyframe.get("name", f"keyframe_{len(keyframes) + 1}")),
                support_mode=str(raw_keyframe.get("support_mode", "double")).lower(),
                joint_targets=targets,
                duration_steps=max(1, int(raw_keyframe.get("duration_steps", 60))),
            )
        )
    if not keyframes:
        raise ValueError(f"Reference gait seed has no keyframes: {path}")
    return ReferenceGaitSeed(
        keyframes=tuple(keyframes),
        target_type=target_type,
        cycle_steps=sum(keyframe.duration_steps for keyframe in keyframes),
    )


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
    config: SeedonStandingConfig,
    progress_delta_x: float = 0.0,
    cumulative_progress_x: float = 0.0,
    elapsed_steps: int = 1,
    in_double_support: bool = True,
    support_force_ratio: float = 0.0,
    swing_force_ratio: float = 0.0,
    swing_foot_clearance: float = 0.0,
    episode_action_mean_l2: float = 0.0,
) -> dict[str, float]:
    """Compute shaped reward terms for standing.

    Args:
        base_height: Current base body height.
        upright: Dot product between local base z-axis and world z-axis.
        joint_velocity_l2: Squared norm of actuated joint velocities.
        action_l2: Squared norm of normalized actions.
        action_rate_l2: Squared norm of the action delta from the previous step.
        joint_position_error_l2: Squared norm of actuated joint deviation from the seed pose.
        episode_action_mean_l2: Squared norm of the running episode mean action.
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
        in_double_support: Whether the active gait phase is a double-support recovery window.
        support_force_ratio: Normal-force fraction carried by the designated support foot.
        swing_force_ratio: Normal-force fraction carried by the designated swing foot.
        swing_foot_clearance: Swing foot bottom height above support foot bottom.

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
    single_support_gate = 0.0 if in_double_support else 1.0
    support_ratio = float(np.clip(support_force_ratio, 0.0, 1.0))
    swing_ratio = float(np.clip(swing_force_ratio, 0.0, 1.0))
    blue_support_ratio = float(
        np.clip(
            support_ratio / max(config.blue_target_support_ratio, 1e-9),
            0.0,
            1.0,
        )
    )
    blue_swing_unload = float(
        np.clip(
            1.0 - swing_ratio / max(config.blue_target_swing_ratio, 1e-9),
            0.0,
            1.0,
        )
    )
    clearance_error = swing_foot_clearance - config.blue_target_clearance
    blue_clearance = float(np.exp(-25000.0 * clearance_error * clearance_error))
    if swing_foot_clearance > config.blue_max_clearance:
        excess_clearance = swing_foot_clearance - config.blue_max_clearance
        blue_clearance *= float(np.exp(-12000.0 * excess_clearance * excess_clearance))
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
        "blue_support_ratio": blue_support_ratio * single_support_gate,
        "blue_swing_unload": blue_swing_unload * single_support_gate,
        "blue_clearance": blue_clearance * single_support_gate,
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
        "episode_action_mean_penalty": episode_action_mean_l2,
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
    total += (
        config.blue_support_ratio_weight
        * components["blue_support_ratio"]
        * components["stability_gate"]
    )
    total += (
        config.blue_swing_unload_weight
        * components["blue_swing_unload"]
        * components["stability_gate"]
    )
    total += (
        config.blue_clearance_weight
        * components["blue_clearance"]
        * components["stability_gate"]
    )
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
    total -= (
        config.episode_action_mean_penalty_weight
        * components["episode_action_mean_penalty"]
    )
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
    config: SeedonStandingConfig,
    episode_action_mean_l2: float = 0.0,
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
        "episode_action_mean_penalty": episode_action_mean_l2,
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
    total -= (
        config.episode_action_mean_penalty_weight
        * components["episode_action_mean_penalty"]
    )
    total -= config.velocity_penalty_weight * components["velocity_penalty"]
    total -= (
        config.base_roll_pitch_rate_penalty_weight
        * components["base_roll_pitch_rate_penalty"]
    )
    components["total"] = float(total)
    return components


def compute_march_in_place_reward(
    *,
    base_height: float,
    upright: float,
    joint_velocity_l2: float,
    action_l2: float,
    action_rate_l2: float,
    joint_position_error_l2: float,
    forward_velocity: float,
    base_roll_pitch_rate_l2: float,
    foot_flatness: float,
    com_y: float,
    target_com_y: float,
    base_roll: float,
    base_pitch: float,
    support_force_ratio: float,
    swing_force_ratio: float,
    support_ratio_baseline: float,
    target_support_force_ratio: float,
    target_swing_force_ratio: float,
    foot_clearance: float,
    target_foot_clearance: float,
    base_xy_drift: float,
    torque_saturation: float,
    curriculum_phase: int,
    config: SeedonStandingConfig,
    swing_foot_forward_delta: float = 0.0,
    target_swing_foot_forward_delta: float = 0.0,
    base_height_drop: float = 0.0,
    total_force_normalized: float = 1.0,
    episode_action_mean_l2: float = 0.0,
    tracking_error_variance: float = 0.0,
    contact_transition: float = 0.0,
) -> dict[str, float]:
    """Compute reward terms for the staged Seedon march-in-place curriculum.

    Args:
        base_height: Current base body height in meters.
        upright: Base local z-axis alignment with world z-axis.
        joint_velocity_l2: Squared norm of actuated joint velocities.
        action_l2: Squared norm of normalized action.
        action_rate_l2: Squared action delta norm.
        joint_position_error_l2: Squared deviation from scripted nominal gait.
        episode_action_mean_l2: Squared norm of the running episode mean action.
        tracking_error_variance: Running variance of joint tracking error.
        contact_transition: Whether the contact state changed on this step.
        forward_velocity: Base x velocity in meters per second.
        base_roll_pitch_rate_l2: Squared roll/pitch angular rate.
        foot_flatness: Mean foot z-axis alignment.
        com_y: Current whole-body COM y position.
        target_com_y: Desired lateral COM target for load transfer.
        base_roll: Base roll angle in radians.
        support_force_ratio: Support foot normal-force fraction.
        swing_force_ratio: Swing foot normal-force fraction.
        target_support_force_ratio: Desired support-foot force fraction.
        target_swing_force_ratio: Desired swing-foot force fraction.
        foot_clearance: Swing foot bottom height relative to support foot.
        target_foot_clearance: Desired relative clearance in meters.
        swing_foot_forward_delta: Swing foot x position minus support foot x position.
        target_swing_foot_forward_delta: Desired small swing-foot forward lead.
        torque_saturation: Fraction of actuators at their control limit.
        curriculum_phase: Active curriculum phase, clamped by the caller to 1..5.
        config: Reward coefficients.

    Returns:
        Reward component mapping including ``total``.
    """
    height_error = base_height - config.target_base_height
    height = float(np.exp(-config.height_sharpness * height_error * height_error))
    upright_clipped = float(np.clip(upright, -1.0, 1.0))
    stability_gate = height * max(0.0, upright_clipped)
    pose = float(np.exp(-config.pose_sharpness * joint_position_error_l2))
    foot_flat = float(np.clip(foot_flatness, 0.0, 1.0))
    force_ratio_error = max(0.0, target_support_force_ratio - support_force_ratio)
    support_force_error = support_force_ratio - target_support_force_ratio
    support_ratio_progress = max(0.0, support_force_ratio - support_ratio_baseline)
    swing_unload_error = max(0.0, swing_force_ratio - target_swing_force_ratio)
    has_support_load = support_force_ratio > 1e-6
    has_swing_load = swing_force_ratio > 1e-6
    no_contact_penalty = float(not has_support_load and not has_swing_load)
    single_contact_penalty = float(has_support_load != has_swing_load)
    com_y_error = abs(com_y - target_com_y)
    clearance_error = max(0.0, target_foot_clearance - foot_clearance)
    swing_force_excess = max(0.0, swing_force_ratio - target_swing_force_ratio)
    target_forward_velocity = config.march_forward_velocity if curriculum_phase >= 5 else 0.0
    forward_velocity_error = forward_velocity - target_forward_velocity
    forward_velocity_progress = (
        float(np.clip(forward_velocity / max(target_forward_velocity, 1e-9), 0.0, 1.0))
        if target_forward_velocity > 0.0
        else 0.0
    )
    swing_forward_progress = (
        float(
            np.clip(
                swing_foot_forward_delta / max(target_swing_foot_forward_delta, 1e-9),
                0.0,
                1.0,
            )
        )
        if target_swing_foot_forward_delta > 0.0
        else 0.0
    )
    swing_forward_error = max(
        0.0,
        target_swing_foot_forward_delta - swing_foot_forward_delta,
    )
    height_drop_excess = max(0.0, base_height_drop - config.march_base_height_drop_limit)
    landing_impact_excess = max(
        0.0,
        total_force_normalized - config.march_landing_impact_limit,
    )
    components = {
        "alive": config.alive_reward,
        "height": height,
        "upright": max(0.0, upright_clipped),
        "stability_gate": stability_gate,
        "pose": pose,
        "foot_flat": foot_flat,
        "force_ratio": float(np.clip(1.0 - force_ratio_error, 0.0, 1.0)),
        "support_force_error_penalty": support_force_error * support_force_error,
        "support_ratio_progress": support_ratio_progress,
        "target_side_force_ratio": support_force_ratio,
        "swing_unload": float(np.clip(1.0 - swing_unload_error, 0.0, 1.0)),
        "swing_force_penalty": swing_force_excess * swing_force_excess,
        "no_contact_penalty": no_contact_penalty,
        "jump_penalty": no_contact_penalty,
        "single_contact_penalty": single_contact_penalty,
        "com_y_penalty": com_y_error * com_y_error,
        "base_roll_penalty": base_roll * base_roll,
        "base_pitch_penalty": base_pitch * base_pitch,
        "both_contact": float(support_force_ratio > 0.0 and swing_force_ratio > 0.0),
        "foot_clearance": float(
            np.clip(
                foot_clearance / max(target_foot_clearance, 1e-9),
                0.0,
                1.0,
            )
            if target_foot_clearance > 0.0
            else 1.0
        ),
        "foot_clearance_penalty": clearance_error * clearance_error,
        "forward_velocity_progress": forward_velocity_progress,
        "forward_velocity_penalty": forward_velocity_error * forward_velocity_error,
        "swing_forward_progress": swing_forward_progress,
        "swing_forward_penalty": swing_forward_error * swing_forward_error,
        "forward_velocity_abs_penalty": abs(forward_velocity),
        "base_drift_penalty": base_xy_drift * base_xy_drift,
        "base_height_drop_penalty": height_drop_excess * height_drop_excess,
        "landing_impact_penalty": landing_impact_excess * landing_impact_excess,
        "autonomy_tracking_variance_penalty": tracking_error_variance,
        "autonomy_contact_transition_penalty": contact_transition,
        "autonomy_landing_impact_penalty": landing_impact_excess * landing_impact_excess,
        "autonomy_base_height_drop_penalty": height_drop_excess * height_drop_excess,
        "torque_saturation_penalty": torque_saturation,
        "action_penalty": action_l2,
        "action_rate_penalty": action_rate_l2,
        "episode_action_mean_penalty": episode_action_mean_l2,
        "velocity_penalty": joint_velocity_l2,
        "base_roll_pitch_rate_penalty": base_roll_pitch_rate_l2,
    }
    total = components["alive"]
    total += config.height_weight * components["height"]
    total += config.upright_weight * components["upright"]
    total += config.pose_weight * components["pose"]
    total += config.foot_flat_weight * components["foot_flat"]
    total += (
        config.march_force_ratio_weight
        * components["force_ratio"]
        * components["stability_gate"]
    )
    total += (
        config.march_support_ratio_progress_weight
        * components["support_ratio_progress"]
        * components["stability_gate"]
    )
    total += (
        config.march_target_side_force_ratio_weight
        * components["target_side_force_ratio"]
        * components["stability_gate"]
    )
    total -= (
        config.march_support_force_error_penalty_weight
        * components["support_force_error_penalty"]
    )
    if curriculum_phase >= 2:
        total += (
            config.march_force_ratio_weight
            * components["swing_unload"]
            * components["stability_gate"]
        )
    total -= config.march_com_y_weight * components["com_y_penalty"]
    total -= config.march_base_roll_penalty_weight * components["base_roll_penalty"]
    total -= config.march_base_pitch_penalty_weight * components["base_pitch_penalty"]
    total += (
        config.march_both_contact_reward_weight
        * components["both_contact"]
        * components["stability_gate"]
    )
    if curriculum_phase >= 3:
        clearance_gate = float(
            support_force_ratio >= config.march_clearance_support_ratio_gate
        )
        total += (
            config.march_clearance_weight
            * components["foot_clearance"]
            * components["stability_gate"]
            * clearance_gate
        )
        total -= config.march_clearance_weight * components["foot_clearance_penalty"]
        total -= (
            config.march_swing_force_penalty_weight
            * components["swing_force_penalty"]
        )
        total -= config.march_no_contact_penalty_weight * components["no_contact_penalty"]
        total -= (
            config.march_single_contact_penalty_weight
            * components["single_contact_penalty"]
        )
    if curriculum_phase >= 5:
        total += (
            config.march_forward_progress_weight
            * components["forward_velocity_progress"]
            * components["stability_gate"]
        )
        total += (
            config.march_swing_forward_weight
            * components["swing_forward_progress"]
            * components["stability_gate"]
        )
        total -= config.march_swing_forward_weight * components["swing_forward_penalty"]
        total -= (
            config.march_forward_velocity_weight
            * components["forward_velocity_penalty"]
        )
    else:
        total -= (
            config.march_forward_velocity_abs_penalty_weight
            * components["forward_velocity_abs_penalty"]
        )
    total -= config.march_base_drift_penalty_weight * components["base_drift_penalty"]
    total -= (
        config.march_base_height_drop_penalty_weight
        * components["base_height_drop_penalty"]
    )
    total -= (
        config.march_landing_impact_penalty_weight
        * components["landing_impact_penalty"]
    )
    total -= (
        config.autonomy_tracking_variance_penalty_weight
        * components["autonomy_tracking_variance_penalty"]
    )
    total -= (
        config.autonomy_contact_transition_penalty_weight
        * components["autonomy_contact_transition_penalty"]
    )
    total -= (
        config.autonomy_landing_impact_penalty_weight
        * components["autonomy_landing_impact_penalty"]
    )
    total -= (
        config.autonomy_base_height_drop_penalty_weight
        * components["autonomy_base_height_drop_penalty"]
    )
    total -= config.march_jump_penalty_weight * components["jump_penalty"]
    total -= (
        config.march_torque_saturation_penalty_weight
        * components["torque_saturation_penalty"]
    )
    total -= config.action_penalty_weight * components["action_penalty"]
    total -= config.action_rate_penalty_weight * components["action_rate_penalty"]
    total -= (
        config.episode_action_mean_penalty_weight
        * components["episode_action_mean_penalty"]
    )
    total -= config.velocity_penalty_weight * components["velocity_penalty"]
    total -= (
        config.base_roll_pitch_rate_penalty_weight
        * components["base_roll_pitch_rate_penalty"]
    )
    components["total"] = float(total)
    return components


class SeedonStandingEnv(MujocoEnv):
    """MuJoCo environment for initial Seedon standing/balance experiments.

    Args:
        scene_path: Path to ``training_scene.xml`` generated by
            ``tools.build_seedon_training_scene``.
        frame_skip: Number of MuJoCo solver steps per environment step.
        reward_config: Optional standing reward configuration.
        reset_noise_scale: Uniform noise applied to actuated joint positions.
        **kwargs: Forwarded to ``MujocoEnv`` such as ``render_mode``.

    Raises:
        ModuleNotFoundError: If MuJoCo/Gymnasium MuJoCo dependencies are absent.
        FileNotFoundError: If the private Seedon training scene has not been built.
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
        reward_config: SeedonStandingConfig | None = None,
        reset_noise_scale: float = 0.01,
        **kwargs,
    ) -> None:
        if _MUJOCO_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "SeedonStandingEnv requires 'mujoco' and Gymnasium MuJoCo. "
                "Install project requirements before creating the environment."
            ) from _MUJOCO_IMPORT_ERROR

        self._scene_path = Path(scene_path).expanduser().resolve()
        if not self._scene_path.is_file():
            raise FileNotFoundError(
                f"Seedon training scene not found: {self._scene_path}. "
                "Run `python -m tools.convert_urdf_to_mjcf` and then "
                "`python -m tools.build_seedon_training_scene` first."
            )
        if reset_noise_scale < 0.0:
            raise ValueError("reset_noise_scale must be non-negative.")

        self._reward_config = reward_config or load_seedon_config_from_env()
        if not 1 <= int(self._reward_config.march_curriculum_phase) <= 5:
            raise ValueError("march_curriculum_phase must be between 1 and 5.")
        self._reference_gait_seed: ReferenceGaitSeed | None = None
        if self._reward_config.reference_gait_seed_path:
            seed_path = Path(self._reward_config.reference_gait_seed_path).expanduser()
            if not seed_path.is_absolute():
                seed_path = REPO_ROOT / seed_path
            self._reference_gait_seed = load_reference_gait_seed(seed_path)
        if (
            self._reward_config.task_mode.lower() == "reference_march"
            and self._reference_gait_seed is None
        ):
            raise ValueError("reference_march requires reference_gait_seed_path.")
        self._reset_noise_scale = reset_noise_scale
        self._prev_action = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self._episode_action_sum = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self._episode_action_count = 0
        self._tracking_error_mean = 0.0
        self._tracking_error_m2 = 0.0
        self._tracking_error_count = 0
        self._prev_contact_state = "unknown"
        self._prev_base_x = 0.0
        self._episode_base_x = 0.0
        self._episode_base_height = 0.0
        self._gait_step = 0
        self._contact_none_steps = 0

        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(44,),
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
                f"Expected {len(JOINT_NAMES)} Seedon actuators, got {self.model.nu}."
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
        """Advance the Seedon standing simulation by one RL step."""
        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != self.action_space.shape:
            raise ValueError(
                f"Seedon action must have shape {self.action_space.shape}, "
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
        elif task_mode in {"march_in_place", "reference_march"}:
            desired_base_y = self._desired_base_y(task_phase)
        gait_target = self._nominal_joint_qpos + self._gait_joint_offsets()
        residual_action = clipped_action * self._residual_action_mask()
        scaled_residual = residual_action * self._reward_config.action_joint_delta_scale
        target_positions = (
            gait_target
            + scaled_residual
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
        action_l2 = float(np.dot(residual_action, residual_action))
        action_delta = clipped_action - self._prev_action
        action_rate_l2 = float(np.dot(action_delta, action_delta))
        self._episode_action_sum += residual_action
        self._episode_action_count += 1
        episode_mean_action = self._episode_action_sum / max(1, self._episode_action_count)
        episode_action_mean_l2 = float(np.dot(episode_mean_action, episode_mean_action))
        joint_position_error = joint_positions - gait_target
        joint_position_error_l2 = float(np.dot(joint_position_error, joint_position_error))
        tracking_error = float(
            np.sqrt(joint_position_error_l2 / max(len(JOINT_NAMES), 1))
        )
        self._tracking_error_count += 1
        tracking_delta = tracking_error - self._tracking_error_mean
        self._tracking_error_mean += tracking_delta / self._tracking_error_count
        tracking_delta2 = tracking_error - self._tracking_error_mean
        self._tracking_error_m2 += tracking_delta * tracking_delta2
        tracking_error_variance = (
            self._tracking_error_m2 / max(self._tracking_error_count - 1, 1)
            if self._tracking_error_count > 1
            else 0.0
        )
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
        contact_state = self._contact_state(contact_flags)
        contact_transition = float(
            self._prev_contact_state != "unknown"
            and contact_state != self._prev_contact_state
        )
        foot_forces = self._foot_force_state()
        total_robot_weight = float(np.sum(self.model.body_mass) * 9.81)
        total_contact_force_normalized = (
            (foot_forces["right_force"] + foot_forces["left_force"])
            / max(total_robot_weight, 1e-6)
        )
        base_height_drop = max(0.0, self._episode_base_height - base_height)
        knee_violation = self._knee_safe_range_violation()
        base_y = float(self.data.qpos[1])
        com_y = self._overall_com_y()
        base_roll = self._base_roll()
        base_pitch = self._base_pitch()
        torque_saturation = self._torque_saturation_fraction()
        support_contact = False
        swing_contact = False
        support_foot_bottom_z = float("nan")
        swing_foot_bottom_z = float("nan")
        support_force_ratio = 0.0
        swing_force_ratio = 0.0
        foot_clearance = 0.0
        support_foot_x = float("nan")
        swing_foot_x = float("nan")
        swing_foot_forward_delta = 0.0
        if str(task_phase["support_side"]) in {"right", "left"}:
            support_contact, swing_contact = self._task_support_contacts(task_phase, contact_flags)
            support_foot_bottom_z, swing_foot_bottom_z = self._task_foot_bottom_heights(
                task_phase
            )
            support_foot_x, swing_foot_x = self._task_foot_forward_positions(task_phase)
            swing_foot_forward_delta = swing_foot_x - support_foot_x
            support_force_ratio, swing_force_ratio = self._task_force_ratios(
                task_phase,
                foot_forces,
            )
            foot_clearance = max(0.0, swing_foot_bottom_z - support_foot_bottom_z)
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
                episode_action_mean_l2=episode_action_mean_l2,
            )
        elif task_mode in {"march_in_place", "reference_march"}:
            support_contact, swing_contact = self._task_support_contacts(task_phase, contact_flags)
            support_foot_bottom_z, swing_foot_bottom_z = self._task_foot_bottom_heights(
                task_phase
            )
            support_foot_x, swing_foot_x = self._task_foot_forward_positions(task_phase)
            swing_foot_forward_delta = swing_foot_x - support_foot_x
            support_force_ratio, swing_force_ratio = self._task_force_ratios(
                task_phase,
                foot_forces,
            )
            foot_clearance = max(0.0, swing_foot_bottom_z - support_foot_bottom_z)
            target_clearance = self._march_target_clearance(task_mode, task_phase)
            rewards = compute_march_in_place_reward(
                base_height=base_height,
                upright=upright,
                joint_velocity_l2=joint_velocity_l2,
                action_l2=action_l2,
                action_rate_l2=action_rate_l2,
                joint_position_error_l2=joint_position_error_l2,
                forward_velocity=forward_velocity,
                base_roll_pitch_rate_l2=base_roll_pitch_rate_l2,
                foot_flatness=foot_flatness,
                com_y=com_y,
                target_com_y=desired_base_y,
                base_roll=base_roll,
                base_pitch=base_pitch,
                support_force_ratio=support_force_ratio,
                swing_force_ratio=swing_force_ratio,
                support_ratio_baseline=self._reward_config.march_support_ratio_baseline,
                target_support_force_ratio=self._reward_config.march_load_transfer_ratio,
                target_swing_force_ratio=self._reward_config.march_swing_unload_ratio,
                foot_clearance=foot_clearance,
                target_foot_clearance=target_clearance,
                swing_foot_forward_delta=swing_foot_forward_delta,
                target_swing_foot_forward_delta=(
                    self._reward_config.march_swing_forward_distance
                ),
                base_xy_drift=float(np.linalg.norm(self.data.qpos[0:2])),
                torque_saturation=torque_saturation,
                curriculum_phase=int(self._reward_config.march_curriculum_phase),
                config=self._reward_config,
                base_height_drop=base_height_drop,
                total_force_normalized=total_contact_force_normalized,
                episode_action_mean_l2=episode_action_mean_l2,
                tracking_error_variance=tracking_error_variance,
                contact_transition=contact_transition,
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
                in_double_support=bool(task_phase["in_double_support"]),
                support_force_ratio=support_force_ratio,
                swing_force_ratio=swing_force_ratio,
                swing_foot_clearance=foot_clearance,
                episode_action_mean_l2=episode_action_mean_l2,
            )
        termination_reason = self._termination_reason(
            base_height,
            upright,
            obs,
            contact_flags=contact_flags,
        )
        terminated = termination_reason != "none"
        if terminated:
            rewards["total"] -= self._reward_config.termination_penalty

        info = {
            "base_height": base_height,
            "base_height_drop": base_height_drop,
            "base_x_position": current_base_x,
            "progress_delta_x": progress_delta_x,
            "cumulative_progress_x": cumulative_progress_x,
            "upright": upright,
            "joint_velocity_l2": joint_velocity_l2,
            "action_l2": action_l2,
            "action_rate_l2": action_rate_l2,
            "episode_action_mean_l2": episode_action_mean_l2,
            "raw_action_max": float(np.max(clipped_action)),
            "raw_action_abs_max": float(np.max(np.abs(clipped_action))),
            "raw_action_mean": float(np.mean(clipped_action)),
            "raw_action_std": float(np.std(clipped_action)),
            "scaled_residual_max": float(np.max(scaled_residual)),
            "scaled_residual_abs_max": float(np.max(np.abs(scaled_residual))),
            "scaled_residual_mean": float(np.mean(scaled_residual)),
            "scaled_residual_std": float(np.std(scaled_residual)),
            "joint_position_error_l2": joint_position_error_l2,
            "tracking_error": tracking_error,
            "tracking_error_variance": tracking_error_variance,
            "contact_transition": contact_transition,
            "forward_velocity": forward_velocity,
            "lateral_velocity_l2": lateral_velocity_l2,
            "base_xy_velocity_l2": base_xy_velocity_l2,
            "base_y_position": base_y,
            "COM_y": com_y,
            "base_roll": base_roll,
            "base_pitch": base_pitch,
            "base_roll_pitch_rate_l2": base_roll_pitch_rate_l2,
            "gait_phase": self._gait_phase(),
            "foot_flatness": foot_flatness,
            "foot_height_error_l2": foot_height_error_l2,
            "feet_near_floor": feet_near_floor,
            "task_mode": task_mode,
            "phase_name": task_phase["phase_name"],
            "curriculum_phase": task_phase["curriculum_phase"],
            "support_side": task_phase["support_side"],
            "swing_side": task_phase["swing_side"],
            "in_double_support": task_phase["in_double_support"],
            "desired_base_y": desired_base_y,
            "right_contact": contact_flags["right"],
            "left_contact": contact_flags["left"],
            "base_proxy_floor_contact": contact_flags["base_proxy"],
            "right_normal_force": foot_forces["right_force"],
            "left_normal_force": foot_forces["left_force"],
            "total_contact_force_normalized": total_contact_force_normalized,
            "right_force_ratio": foot_forces["right_ratio"],
            "left_force_ratio": foot_forces["left_ratio"],
            "force_ratio": support_force_ratio,
            "swing_force_ratio": swing_force_ratio,
            "target_side_force_ratio": support_force_ratio,
            "support_ratio_progress": max(
                0.0,
                support_force_ratio - self._reward_config.march_support_ratio_baseline,
            ),
            "reward_gate_active": bool(
                support_force_ratio
                >= self._reward_config.march_clearance_support_ratio_gate
            ),
            "contact_none_steps": self._contact_none_steps,
            "both_contact": bool(contact_flags["right"] and contact_flags["left"]),
            "support_contact": support_contact,
            "swing_contact": swing_contact,
            "support_foot_bottom_z": support_foot_bottom_z,
            "swing_foot_bottom_z": swing_foot_bottom_z,
            "foot_clearance": foot_clearance,
            "support_foot_x": support_foot_x,
            "swing_foot_x": swing_foot_x,
            "swing_foot_forward_delta": swing_foot_forward_delta,
            "torque_saturation": torque_saturation,
            "termination_reason": termination_reason,
            "right_knee_qpos": float(joint_positions[RIGHT_KNEE_JOINT_INDEX]),
            "left_knee_qpos": float(joint_positions[LEFT_KNEE_JOINT_INDEX]),
            "right_knee_safe_violation": float(knee_violation["right"]),
            "left_knee_safe_violation": float(knee_violation["left"]),
            "knee_safe_violation_sum": float(knee_violation["total"]),
        }
        for key, value in rewards.items():
            info[f"reward_{key}"] = value

        self._prev_action = clipped_action.copy()
        self._prev_contact_state = contact_state
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
        self._episode_action_sum = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        self._episode_action_count = 0
        self._tracking_error_mean = 0.0
        self._tracking_error_m2 = 0.0
        self._tracking_error_count = 0
        self._prev_contact_state = "unknown"
        self._last_ctrl_assist_delta.fill(0.0)
        self._prev_base_x = float(self.data.qpos[0])
        self._episode_base_x = self._prev_base_x
        self._episode_base_height = self._base_height()
        self._gait_step = 0
        self._contact_none_steps = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Return base orientation, velocity, contact load, joint state, and action history."""
        base_linear_velocity = self.data.qvel[0:3].copy()
        base_angular_velocity = self.data.qvel[3:6].copy()
        foot_forces = self._foot_force_state()
        obs = np.concatenate(
            [
                np.array([self._base_height()], dtype=np.float64),
                self._projected_gravity(),
                base_angular_velocity,
                base_linear_velocity,
                np.array(
                    [np.sin(self._gait_phase()), np.cos(self._gait_phase())],
                    dtype=np.float64,
                ),
                self._joint_positions(),
                self._joint_velocities(),
                np.array(
                    [foot_forces["right_ratio"], foot_forces["left_ratio"]],
                    dtype=np.float64,
                ),
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
        return self._termination_reason(base_height, upright, observation) != "none"

    def _termination_reason(
        self,
        base_height: float,
        upright: float,
        observation: np.ndarray,
        contact_flags: dict[str, bool] | None = None,
    ) -> str:
        """Return a stable termination reason string for diagnostics."""
        if not np.isfinite(observation).all():
            return "non_finite_observation"
        if base_height < self._reward_config.min_base_height:
            return "base_height_low"
        if base_height > self._reward_config.max_base_height:
            return "base_height_high"
        if float(np.linalg.norm(self.data.qpos[0:2])) > self._reward_config.max_base_xy_drift:
            return "base_xy_drift"
        if upright < self._reward_config.min_upright:
            return "base_upright_low"
        if (
            self._reward_config.task_mode.lower()
            in {"march_in_place", "reference_march"}
            and self._reward_config.march_require_both_contact
        ):
            flags = contact_flags or self._floor_contact_flags()
            if not (flags["right"] and flags["left"]):
                return "both_contact_lost"
        if (
            self._reward_config.task_mode.lower()
            in {"march_in_place", "reference_march"}
            and self._reward_config.march_contact_none_terminate_steps > 0
        ):
            flags = contact_flags or self._floor_contact_flags()
            if flags["right"] or flags["left"]:
                self._contact_none_steps = 0
            else:
                self._contact_none_steps += 1
            if self._contact_none_steps >= self._reward_config.march_contact_none_terminate_steps:
                return "contact_none"
        return "none"

    def set_reward_config(self, reward_config: SeedonStandingConfig) -> None:
        """Replace the active reward config without changing the gait seed.

        Args:
            reward_config: New reward configuration. The reference gait seed
                path must match the existing seed path because this method is
                intended for coefficient curricula, not gait replacement.

        Raises:
            ValueError: If the update would replace the reference gait seed.
        """
        if reward_config.reference_gait_seed_path != self._reward_config.reference_gait_seed_path:
            raise ValueError("set_reward_config cannot change reference_gait_seed_path.")
        self._reward_config = reward_config

    def _contact_state(self, contact_flags: dict[str, bool]) -> str:
        """Return compact foot contact state for transition penalties."""
        right = bool(contact_flags["right"])
        left = bool(contact_flags["left"])
        if right and left:
            return "both"
        if right:
            return "right"
        if left:
            return "left"
        return "none"

    def _base_height(self) -> float:
        """Return the floating base body height."""
        return float(self.data.xpos[self._base_body_id][2])

    def _base_upright(self) -> float:
        """Return alignment between base local z-axis and world z-axis."""
        xmat = self.data.xmat[self._base_body_id].reshape(3, 3)
        return float(xmat[2, 2])

    def _projected_gravity(self) -> np.ndarray:
        """Return world gravity direction projected into the base frame."""
        xmat = self.data.xmat[self._base_body_id].reshape(3, 3)
        return xmat.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)

    def _base_roll(self) -> float:
        """Return floating-base roll angle from the MuJoCo root quaternion."""
        qw, qx, qy, qz = [float(value) for value in self.data.qpos[3:7]]
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        return float(math.atan2(sinr_cosp, cosr_cosp))

    def _base_pitch(self) -> float:
        """Return floating-base pitch angle from the MuJoCo root quaternion."""
        qw, qx, qy, qz = [float(value) for value in self.data.qpos[3:7]]
        sinp = 2.0 * (qw * qy - qz * qx)
        return float(math.asin(float(np.clip(sinp, -1.0, 1.0))))

    def _overall_com_y(self) -> float:
        """Return whole-model COM y when MuJoCo exposes subtree COM data."""
        if hasattr(self.data, "subtree_com"):
            return float(self.data.subtree_com[0][1])
        return float(self.data.qpos[1])

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
        if self._reward_config.task_mode.lower() == "march_in_place":
            return self._march_in_place_joint_offsets()
        if self._reward_config.task_mode.lower() == "reference_march":
            return self._reference_march_joint_offsets()
        mode = self._reward_config.gait_mode.lower()
        if mode == "fsm":
            return self._fsm_gait_joint_offsets()
        if mode == "sin":
            return self._sin_gait_joint_offsets()
        raise ValueError(f"Unsupported Seedon gait_mode: {self._reward_config.gait_mode}")

    def _residual_action_mask(self) -> np.ndarray:
        """Return the active residual-action mask for the configured task."""
        mode = self._reward_config.march_residual_mode.lower()
        if (
            self._reward_config.task_mode.lower()
            not in {"march_in_place", "reference_march"}
            or mode == "all"
        ):
            return np.ones(len(JOINT_NAMES), dtype=np.float64)
        if mode == "hip_roll_only":
            mask = np.zeros(len(JOINT_NAMES), dtype=np.float64)
            mask[RIGHT_HIP_ROLL_ACTUATOR_INDEX] = 1.0
            mask[LEFT_HIP_ROLL_ACTUATOR_INDEX] = 1.0
            return mask
        raise ValueError(
            "Unsupported march_residual_mode "
            f"'{self._reward_config.march_residual_mode}'."
        )

    def _task_cycle_steps(self) -> int:
        """Return the active cycle length for the configured task."""
        if self._reward_config.task_mode.lower() == "com_shift":
            return max(
                1,
                2 * self._reward_config.com_shift_center_hold_steps
                + 2 * self._reward_config.com_shift_shift_steps,
            )
        if self._reward_config.task_mode.lower() == "march_in_place":
            return self._march_cycle_steps()
        if self._reward_config.task_mode.lower() == "reference_march":
            return self._require_reference_gait_seed().cycle_steps
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

    def _march_in_place_joint_offsets(self) -> np.ndarray:
        """Return phase-clock scripted march targets used before RL residual action."""
        task_phase = self._task_phase_metadata()
        support_side = str(task_phase["support_side"])
        phase_name = str(task_phase["phase_name"])
        phase_alpha = float(task_phase["phase_alpha"])
        if support_side not in {"right", "left"}:
            return np.zeros(len(JOINT_NAMES), dtype=np.float64)
        unload_scale = 0.0
        if phase_name.endswith("swing_unload"):
            unload_scale = phase_alpha
        elif phase_name.endswith("micro_lift"):
            unload_scale = 1.0
        elif phase_name.endswith("lower"):
            unload_scale = 1.0 - phase_alpha
        return self._com_shift_support_pose(
            support_side,
            unload_scale=unload_scale,
        ) * max(phase_alpha, 0.25)

    def _march_target_clearance(
        self,
        task_mode: str,
        task_phase: dict[str, object],
    ) -> float:
        """Return the swing-foot clearance target for march-style tasks."""
        if int(self._reward_config.march_curriculum_phase) < 3:
            return 0.0
        if task_mode == "march_in_place":
            phase_name = str(task_phase["phase_name"])
            if phase_name.endswith(("micro_lift", "lower")):
                return self._reward_config.march_micro_lift_height
            return 0.0
        if task_mode == "reference_march" and str(task_phase["swing_side"]) in {"right", "left"}:
            return self._reward_config.march_micro_lift_height
        return 0.0

    def _require_reference_gait_seed(self) -> ReferenceGaitSeed:
        """Return the loaded reference gait seed or raise a clear config error."""
        if self._reference_gait_seed is None:
            raise ValueError("reference_march requires reference_gait_seed_path.")
        return self._reference_gait_seed

    def _reference_seed_target_at_step(self) -> tuple[np.ndarray, ReferenceGaitKeyframe, float]:
        """Return the interpolated reference joint target for the active gait step."""
        seed = self._require_reference_gait_seed()
        phase_step = self._gait_step % seed.cycle_steps
        cursor = 0
        for index, current in enumerate(seed.keyframes):
            duration = max(1, current.duration_steps)
            if phase_step < cursor + duration:
                local_step = phase_step - cursor
                next_keyframe = seed.keyframes[(index + 1) % len(seed.keyframes)]
                alpha = self._smoothstep((local_step + 1) / duration)
                target = (
                    (1.0 - alpha) * current.joint_targets
                    + alpha * next_keyframe.joint_targets
                )
                return target, current, alpha
            cursor += duration
        last_keyframe = seed.keyframes[-1]
        return last_keyframe.joint_targets.copy(), last_keyframe, 1.0

    def _reference_march_joint_offsets(self) -> np.ndarray:
        """Return reference seed offsets relative to the current nominal stance."""
        seed = self._require_reference_gait_seed()
        target, _, _ = self._reference_seed_target_at_step()
        scaled_target = target * self._reward_config.reference_gait_seed_scale
        if seed.target_type == "offset":
            return scaled_target
        return scaled_target - self._nominal_joint_qpos

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
            if self._reward_config.task_mode.lower() == "march_in_place":
                return self._march_phase_metadata()
            if self._reward_config.task_mode.lower() == "reference_march":
                return self._reference_march_phase_metadata()
            return self._walk_phase_metadata()
        hold_steps = max(1, self._reward_config.com_shift_center_hold_steps)
        shift_steps = max(1, self._reward_config.com_shift_shift_steps)
        cycle_steps = self._task_cycle_steps()
        phase_step = self._gait_step % cycle_steps
        if phase_step < hold_steps:
            return {
                "phase_name": "center_hold_right",
                "curriculum_phase": 0,
                "support_side": "center",
                "swing_side": "none",
                "in_double_support": True,
                "shift_alpha": 0.0,
                "phase_alpha": 0.0,
            }
        phase_step -= hold_steps
        if phase_step < shift_steps:
            return {
                "phase_name": "right_shift",
                "curriculum_phase": 0,
                "support_side": "right",
                "swing_side": "left",
                "in_double_support": True,
                "shift_alpha": self._smoothstep((phase_step + 1) / shift_steps),
                "phase_alpha": self._smoothstep((phase_step + 1) / shift_steps),
            }
        phase_step -= shift_steps
        if phase_step < hold_steps:
            return {
                "phase_name": "center_hold_left",
                "curriculum_phase": 0,
                "support_side": "center",
                "swing_side": "none",
                "in_double_support": True,
                "shift_alpha": 0.0,
                "phase_alpha": 0.0,
            }
        phase_step -= hold_steps
        return {
            "phase_name": "left_shift",
            "curriculum_phase": 0,
            "support_side": "left",
            "swing_side": "right",
            "in_double_support": True,
            "shift_alpha": self._smoothstep((phase_step + 1) / shift_steps),
            "phase_alpha": self._smoothstep((phase_step + 1) / shift_steps),
        }

    def _walk_phase_metadata(self) -> dict[str, object]:
        """Return support/swing metadata for the active walk gait phase."""
        if self._reward_config.gait_mode.lower() != "fsm":
            return {
                "phase_name": "walk",
                "curriculum_phase": 0,
                "support_side": "none",
                "swing_side": "none",
                "in_double_support": False,
                "shift_alpha": 0.0,
                "phase_alpha": 0.0,
            }

        right_lift_steps = max(1, self._reward_config.fsm_right_lift_steps)
        right_lower_steps = max(1, self._reward_config.fsm_right_lower_steps)
        left_lift_steps = max(1, self._reward_config.fsm_left_lift_steps)
        left_lower_steps = max(1, self._reward_config.fsm_left_lower_steps)
        double_support_steps = max(0, self._reward_config.fsm_double_support_steps)
        total_steps = (
            right_lift_steps
            + right_lower_steps
            + double_support_steps
            + left_lift_steps
            + left_lower_steps
            + double_support_steps
        )
        phase_step = self._gait_step % max(1, total_steps)

        def phase(
            name: str,
            support_side: str,
            swing_side: str,
            in_double_support: bool,
            step_in_phase: int,
            phase_steps: int,
        ) -> dict[str, object]:
            alpha = self._smoothstep((step_in_phase + 1) / max(1, phase_steps))
            return {
                "phase_name": name,
                "curriculum_phase": 0,
                "support_side": support_side,
                "swing_side": swing_side,
                "in_double_support": in_double_support,
                "shift_alpha": alpha,
                "phase_alpha": alpha,
            }

        if phase_step < right_lift_steps:
            return phase("right_swing_lift", "left", "right", False, phase_step, right_lift_steps)
        phase_step -= right_lift_steps
        if phase_step < right_lower_steps:
            return phase("right_swing_lower", "left", "right", False, phase_step, right_lower_steps)
        phase_step -= right_lower_steps
        if phase_step < double_support_steps:
            return phase(
                "double_support_after_right",
                "center",
                "none",
                True,
                phase_step,
                max(1, double_support_steps),
            )
        phase_step -= double_support_steps
        if phase_step < left_lift_steps:
            return phase("left_swing_lift", "right", "left", False, phase_step, left_lift_steps)
        phase_step -= left_lift_steps
        if phase_step < left_lower_steps:
            return phase("left_swing_lower", "right", "left", False, phase_step, left_lower_steps)
        phase_step -= left_lower_steps
        return phase(
            "double_support_after_left",
            "center",
            "none",
            True,
            phase_step,
            max(1, double_support_steps),
        )

    def _reference_march_phase_metadata(self) -> dict[str, object]:
        """Return support/swing metadata from the loaded reference seed."""
        _, keyframe, phase_alpha = self._reference_seed_target_at_step()
        support_mode = keyframe.support_mode
        if support_mode == "left":
            support_side = "left"
            swing_side = "right"
            in_double_support = False
        elif support_mode == "right":
            support_side = "right"
            swing_side = "left"
            in_double_support = False
        else:
            support_side = "center"
            swing_side = "none"
            in_double_support = True
        return {
            "phase_name": keyframe.name,
            "curriculum_phase": int(self._reward_config.march_curriculum_phase),
            "support_side": support_side,
            "swing_side": swing_side,
            "in_double_support": in_double_support,
            "shift_alpha": phase_alpha,
            "phase_alpha": phase_alpha,
        }

    def _march_segments(self) -> list[tuple[str, str, str, bool]]:
        """Return active phase-clock segments for the configured march curriculum."""
        curriculum_phase = int(self._reward_config.march_curriculum_phase)
        per_side = [("load_transfer", True)]
        if curriculum_phase >= 2:
            per_side.append(("swing_unload", True))
        if curriculum_phase >= 3:
            per_side.append(("micro_lift", False))
        if curriculum_phase >= 4:
            per_side.append(("lower", False))
        segments: list[tuple[str, str, str, bool]] = []
        for support_side, swing_side in (("right", "left"), ("left", "right")):
            for name, in_double_support in per_side:
                segments.append(
                    (
                        f"{support_side}_{name}",
                        support_side,
                        swing_side,
                        in_double_support,
                    )
                )
        return segments

    def _march_cycle_steps(self) -> int:
        """Return the active phase-clock length for march-in-place training."""
        phase_steps = max(1, self._reward_config.march_phase_steps)
        return max(1, len(self._march_segments()) * phase_steps)

    def _march_phase_metadata(self) -> dict[str, object]:
        """Return current phase-clock metadata for march-in-place curriculum."""
        phase_steps = max(1, self._reward_config.march_phase_steps)
        segments = self._march_segments()
        segment_index = (self._gait_step // phase_steps) % len(segments)
        phase_step = self._gait_step % phase_steps
        phase_name, support_side, swing_side, in_double_support = segments[segment_index]
        phase_alpha = self._smoothstep((phase_step + 1) / phase_steps)
        return {
            "phase_name": phase_name,
            "curriculum_phase": int(self._reward_config.march_curriculum_phase),
            "support_side": support_side,
            "swing_side": swing_side,
            "in_double_support": in_double_support,
            "shift_alpha": phase_alpha,
            "phase_alpha": phase_alpha,
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
        """Run the per-physics-step PD update loop used by Seedon control."""
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

    def _foot_force_state(self) -> dict[str, float]:
        """Return foot normal forces and normalized load fractions."""
        right_force = self._foot_floor_normal_force(self._foot_geom_ids[0])
        left_force = self._foot_floor_normal_force(self._foot_geom_ids[1])
        total_force = right_force + left_force
        if total_force > 1e-9:
            right_ratio = right_force / total_force
            left_ratio = left_force / total_force
        else:
            right_ratio = 0.0
            left_ratio = 0.0
        return {
            "right_force": float(right_force),
            "left_force": float(left_force),
            "right_ratio": float(right_ratio),
            "left_ratio": float(left_ratio),
        }

    def _foot_floor_normal_force(self, foot_geom_id: int) -> float:
        """Return summed MuJoCo normal force between one foot geom and the floor."""
        normal_force_sum = 0.0
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair != {self._floor_geom_id, int(foot_geom_id)}:
                continue
            wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_index, wrench)
            normal_force_sum += abs(float(wrench[0]))
        return normal_force_sum

    def _torque_saturation_fraction(self) -> float:
        """Return fraction of actuators currently close to their control limits."""
        if not hasattr(self.data, "ctrl") or self.data.ctrl.size == 0:
            return 0.0
        lower_margin = np.abs(self.data.ctrl - self._ctrl_range[:, 0])
        upper_margin = np.abs(self.data.ctrl - self._ctrl_range[:, 1])
        ctrl_span = np.maximum(self._ctrl_range[:, 1] - self._ctrl_range[:, 0], 1e-9)
        saturated = np.minimum(lower_margin, upper_margin) <= 0.02 * ctrl_span
        return float(np.count_nonzero(saturated) / self.data.ctrl.size)

    def _desired_base_y(self, task_phase: dict[str, object]) -> float:
        """Return the desired base lateral position for the current task phase."""
        phase_name = str(task_phase["phase_name"])
        target = self._reward_config.com_shift_lateral_target_magnitude
        if phase_name == "right_shift" or str(task_phase.get("support_side")) == "right":
            return target
        if phase_name == "left_shift" or str(task_phase.get("support_side")) == "left":
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

    def _task_force_ratios(
        self,
        task_phase: dict[str, object],
        foot_forces: dict[str, float],
    ) -> tuple[float, float]:
        """Return support and swing force ratios for the current task phase."""
        support_side = str(task_phase["support_side"])
        if support_side == "right":
            return foot_forces["right_ratio"], foot_forces["left_ratio"]
        if support_side == "left":
            return foot_forces["left_ratio"], foot_forces["right_ratio"]
        return 0.5 * (foot_forces["right_ratio"] + foot_forces["left_ratio"]), 0.0

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

    def _task_foot_forward_positions(
        self,
        task_phase: dict[str, object],
    ) -> tuple[float, float]:
        """Return support and swing foot x positions for the current task phase."""
        right_x = float(self.data.geom_xpos[self._foot_geom_ids[0]][0])
        left_x = float(self.data.geom_xpos[self._foot_geom_ids[1]][0])
        support_side = str(task_phase["support_side"])
        if support_side == "right":
            return right_x, left_x
        if support_side == "left":
            return left_x, right_x
        mean_x = 0.5 * (right_x + left_x)
        return mean_x, mean_x

    def _body_id(self, name: str) -> int:
        """Resolve a MuJoCo body id by name."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body '{name}' not found in Seedon model.")
        return body_id

    def _geom_id(self, name: str) -> int:
        """Resolve a MuJoCo geom id by name."""
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"Geom '{name}' not found in Seedon model.")
        return geom_id

    def _joint_id(self, name: str) -> int:
        """Resolve a MuJoCo joint id by name."""
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint '{name}' not found in Seedon model.")
        return joint_id
