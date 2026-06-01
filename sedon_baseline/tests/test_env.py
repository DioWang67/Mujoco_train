"""Regression tests for the Sedon standing environment."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.mujoco

from sedon_baseline.env import (
    compute_com_shift_reward,
    compute_march_in_place_reward,
    DEFAULT_SCENE_PATH,
    JOINT_NAMES,
    LEFT_HIP_ROLL_ACTUATOR_INDEX,
    LEFT_KNEE_JOINT_INDEX,
    RIGHT_HIP_ROLL_ACTUATOR_INDEX,
    RIGHT_KNEE_JOINT_INDEX,
    SedonStandingConfig,
    SedonStandingEnv,
    compute_standing_reward,
    load_reference_gait_seed,
)

sedon_env_module = sys.modules[SedonStandingEnv.__module__]


def test_compute_standing_reward_prefers_target_height_and_upright_pose() -> None:
    config = SedonStandingConfig()

    good = compute_standing_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=config.target_forward_velocity,
        lateral_velocity_l2=0.0,
        base_xy_velocity_l2=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        foot_height_error_l2=0.0,
        feet_near_floor=2,
        config=config,
        elapsed_steps=config.cumulative_progress_grace_steps + 1,
    )
    bad = compute_standing_reward(
        base_height=config.min_base_height,
        upright=0.0,
        joint_velocity_l2=10.0,
        action_l2=10.0,
        action_rate_l2=10.0,
        joint_position_error_l2=2.0,
        forward_velocity=-0.2,
        lateral_velocity_l2=1.0,
        base_xy_velocity_l2=4.0,
        base_roll_pitch_rate_l2=4.0,
        foot_flatness=0.4,
        foot_height_error_l2=0.05,
        feet_near_floor=0,
        config=config,
    )

    assert good["total"] > bad["total"]


def test_compute_standing_reward_penalizes_low_crouch() -> None:
    config = SedonStandingConfig()

    target_pose = compute_standing_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=config.target_forward_velocity,
        lateral_velocity_l2=0.0,
        base_xy_velocity_l2=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        foot_height_error_l2=0.0,
        feet_near_floor=2,
        config=config,
    )
    low_crouch = compute_standing_reward(
        base_height=0.27,
        upright=0.95,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=config.target_forward_velocity,
        lateral_velocity_l2=0.0,
        base_xy_velocity_l2=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        foot_height_error_l2=0.0,
        feet_near_floor=2,
        config=config,
    )

    assert low_crouch["total"] < target_pose["total"] * 0.5


def test_compute_standing_reward_penalizes_joint_pose_deviation() -> None:
    config = SedonStandingConfig()

    nominal = compute_standing_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=config.target_forward_velocity,
        lateral_velocity_l2=0.0,
        base_xy_velocity_l2=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        foot_height_error_l2=0.0,
        feet_near_floor=2,
        config=config,
    )
    toe_stance = compute_standing_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.45,
        forward_velocity=0.0,
        lateral_velocity_l2=0.2,
        base_xy_velocity_l2=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=0.5,
        foot_height_error_l2=0.03,
        feet_near_floor=1,
        config=config,
    )

    assert toe_stance["total"] < nominal["total"]


def test_compute_standing_reward_penalizes_horizontal_drift_and_shaking() -> None:
    config = SedonStandingConfig()

    stable = compute_standing_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=config.target_forward_velocity,
        lateral_velocity_l2=0.0,
        base_xy_velocity_l2=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        foot_height_error_l2=0.0,
        feet_near_floor=2,
        config=config,
    )
    unstable = compute_standing_reward(
        base_height=config.target_base_height,
        upright=0.92,
        joint_velocity_l2=4.0,
        action_l2=1.0,
        action_rate_l2=2.0,
        joint_position_error_l2=0.15,
        forward_velocity=-0.1,
        lateral_velocity_l2=0.5,
        base_xy_velocity_l2=1.5,
        base_roll_pitch_rate_l2=2.0,
        foot_flatness=0.6,
        foot_height_error_l2=0.02,
        feet_near_floor=1,
        config=config,
    )

    assert unstable["total"] < stable["total"]


def test_compute_standing_reward_prefers_target_forward_velocity() -> None:
    config = SedonStandingConfig()

    standing_still = compute_standing_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=0.0,
        lateral_velocity_l2=0.0,
        base_xy_velocity_l2=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        foot_height_error_l2=0.0,
        feet_near_floor=2,
        config=config,
        elapsed_steps=config.cumulative_progress_grace_steps + 1,
    )
    walking_forward = compute_standing_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=config.target_forward_velocity,
        lateral_velocity_l2=0.0,
        base_xy_velocity_l2=config.target_forward_velocity
        * config.target_forward_velocity,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        foot_height_error_l2=0.0,
        feet_near_floor=2,
        config=config,
    )
    moving_sideways = compute_standing_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=0.0,
        lateral_velocity_l2=0.25,
        base_xy_velocity_l2=0.25,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        foot_height_error_l2=0.0,
        feet_near_floor=2,
        config=config,
    )

    assert walking_forward["total"] > standing_still["total"]
    assert walking_forward["total"] > moving_sideways["total"]
    assert standing_still["low_forward_velocity_penalty"] > 0.0
    assert walking_forward["low_forward_velocity_penalty"] == 0.0


def test_compute_standing_reward_prefers_blue_style_support_phase() -> None:
    config = SedonStandingConfig(
        blue_support_ratio_weight=4.0,
        blue_swing_unload_weight=4.0,
        blue_clearance_weight=3.0,
        blue_target_support_ratio=0.62,
        blue_target_swing_ratio=0.35,
        blue_target_clearance=0.006,
        blue_max_clearance=0.018,
    )

    poor = compute_standing_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=config.target_forward_velocity,
        lateral_velocity_l2=0.0,
        base_xy_velocity_l2=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        foot_height_error_l2=0.0,
        feet_near_floor=2,
        config=config,
        in_double_support=False,
        support_force_ratio=0.50,
        swing_force_ratio=0.50,
        swing_foot_clearance=0.030,
    )
    good = compute_standing_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=config.target_forward_velocity,
        lateral_velocity_l2=0.0,
        base_xy_velocity_l2=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        foot_height_error_l2=0.0,
        feet_near_floor=2,
        config=config,
        in_double_support=False,
        support_force_ratio=0.70,
        swing_force_ratio=0.20,
        swing_foot_clearance=0.006,
    )

    assert good["blue_support_ratio"] > poor["blue_support_ratio"]
    assert good["blue_swing_unload"] > poor["blue_swing_unload"]
    assert good["blue_clearance"] > poor["blue_clearance"]
    assert good["total"] > poor["total"]


def test_sedon_gait_offsets_keep_configured_joint_signs() -> None:
    env = SedonStandingEnv.__new__(SedonStandingEnv)
    env._reward_config = SedonStandingConfig(
        gait_mode="sin",
        gait_cycle_steps=100,
        gait_hip_pitch_amp=-0.12,
        gait_knee_pitch_amp=0.21,
        gait_ankle_pitch_amp=-0.08,
    )
    env._gait_step = 25

    offsets = env._sin_gait_joint_offsets()

    assert offsets[2] < 0.0
    assert offsets[3] > 0.0
    assert offsets[4] < 0.0
    assert offsets[7] == pytest.approx(0.0)
    assert offsets[8] == pytest.approx(0.0)
    assert offsets[9] == pytest.approx(0.0)


def test_sedon_nominal_joint_pose_offsets_apply_symmetrically() -> None:
    env = SedonStandingEnv.__new__(SedonStandingEnv)
    env._reward_config = SedonStandingConfig(
        nominal_hip_pitch_offset=-0.18,
        nominal_knee_pitch_offset=0.32,
        nominal_ankle_pitch_offset=-0.14,
    )

    offsets = env._nominal_joint_pose_offsets()

    expected = np.zeros(10, dtype=np.float64)
    expected[2] = expected[7] = -0.18
    expected[3] = expected[8] = 0.32
    expected[4] = expected[9] = -0.14
    np.testing.assert_allclose(offsets, expected)


def test_sedon_knee_safe_ranges_return_none_when_unset() -> None:
    env = SedonStandingEnv.__new__(SedonStandingEnv)
    env._reward_config = SedonStandingConfig()

    assert env.knee_safe_ranges() == {"right": None, "left": None}


def test_sedon_knee_safe_range_violation_reports_excess_distance() -> None:
    env = SedonStandingEnv.__new__(SedonStandingEnv)
    env._reward_config = SedonStandingConfig(
        right_knee_safe_lower=0.0,
        right_knee_safe_upper=0.4,
        left_knee_safe_lower=0.0,
        left_knee_safe_upper=0.3,
    )
    env._joint_positions = lambda: np.array(  # type: ignore[method-assign]
        [0.0, 0.0, 0.0, 0.55, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0],
        dtype=np.float64,
    )

    violation = env._knee_safe_range_violation()

    assert violation["right"] == pytest.approx(0.15)
    assert violation["left"] == pytest.approx(0.1)
    assert violation["total"] == pytest.approx(0.25)


def test_sedon_safe_joint_target_clamps_limit_knees() -> None:
    env = SedonStandingEnv.__new__(SedonStandingEnv)
    env._reward_config = SedonStandingConfig(
        right_knee_safe_lower=-1.0,
        right_knee_safe_upper=0.0,
        left_knee_safe_lower=-0.8,
        left_knee_safe_upper=0.0,
    )
    unclamped = np.array([0.0, 0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0])

    clamped = env._apply_safe_joint_target_clamps(unclamped)

    assert clamped[RIGHT_KNEE_JOINT_INDEX] == pytest.approx(0.0)
    assert clamped[LEFT_KNEE_JOINT_INDEX] == pytest.approx(0.0)
    assert clamped[2] == pytest.approx(unclamped[2])


def test_compute_com_shift_reward_prefers_matching_shift_target() -> None:
    config = SedonStandingConfig(
        task_mode="com_shift",
        target_base_height=0.434,
        com_shift_lateral_target_magnitude=0.025,
        com_shift_lateral_target_weight=8.0,
        com_shift_shift_phase_error_penalty_weight=12.0,
        com_shift_support_contact_reward_weight=3.0,
    )

    centered = compute_com_shift_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=0.0,
        lateral_velocity=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        base_y=0.0,
        target_base_y=0.025,
        previous_abs_y_error=0.025,
        current_base_x_displacement=0.0,
        in_double_support=False,
        support_contact=True,
        swing_contact=True,
        support_foot_bottom_z=0.0,
        swing_foot_bottom_z=0.0,
        config=config,
    )
    shifted = compute_com_shift_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=0.0,
        lateral_velocity=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        base_y=0.022,
        target_base_y=0.025,
        previous_abs_y_error=0.025,
        current_base_x_displacement=0.0,
        in_double_support=False,
        support_contact=True,
        swing_contact=True,
        support_foot_bottom_z=0.0,
        swing_foot_bottom_z=0.001,
        config=config,
    )

    assert shifted["lateral_target"] > centered["lateral_target"]
    assert shifted["shift_phase_error_penalty"] < centered["shift_phase_error_penalty"]
    assert shifted["total"] > centered["total"]


def test_compute_com_shift_reward_prefers_double_support_during_center_hold() -> None:
    config = SedonStandingConfig(
        task_mode="com_shift",
        target_base_height=0.434,
        com_shift_both_contact_reward_weight=5.0,
        com_shift_support_contact_reward_weight=3.0,
    )

    both_contact = compute_com_shift_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=0.0,
        lateral_velocity=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        base_y=0.0,
        target_base_y=0.0,
        previous_abs_y_error=0.0,
        current_base_x_displacement=0.0,
        in_double_support=True,
        support_contact=True,
        swing_contact=True,
        support_foot_bottom_z=0.0,
        swing_foot_bottom_z=0.0,
        config=config,
    )
    missing_swing_contact = compute_com_shift_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=0.0,
        lateral_velocity=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        base_y=0.0,
        target_base_y=0.0,
        previous_abs_y_error=0.0,
        current_base_x_displacement=0.0,
        in_double_support=True,
        support_contact=True,
        swing_contact=False,
        support_foot_bottom_z=0.0,
        swing_foot_bottom_z=0.0,
        config=config,
    )

    assert both_contact["double_support_contact"] == pytest.approx(1.0)
    assert missing_swing_contact["double_support_contact"] == pytest.approx(0.0)
    assert both_contact["total"] > missing_swing_contact["total"]


def test_compute_march_reward_prefers_load_transfer_and_clearance() -> None:
    config = SedonStandingConfig(
        task_mode="march_in_place",
        march_curriculum_phase=3,
        march_load_transfer_ratio=0.62,
        march_swing_unload_ratio=0.25,
        march_micro_lift_height=0.005,
    )

    poor = compute_march_in_place_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        com_y=0.0,
        target_com_y=0.02,
        base_roll=0.0,
        base_pitch=0.0,
        support_force_ratio=0.50,
        swing_force_ratio=0.50,
        support_ratio_baseline=0.518,
        target_support_force_ratio=config.march_load_transfer_ratio,
        target_swing_force_ratio=config.march_swing_unload_ratio,
        foot_clearance=0.0,
        target_foot_clearance=config.march_micro_lift_height,
        base_xy_drift=0.0,
        torque_saturation=0.0,
        curriculum_phase=3,
        config=config,
    )
    good = compute_march_in_place_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        com_y=0.019,
        target_com_y=0.02,
        base_roll=0.0,
        base_pitch=0.0,
        support_force_ratio=0.70,
        swing_force_ratio=0.20,
        support_ratio_baseline=0.518,
        target_support_force_ratio=config.march_load_transfer_ratio,
        target_swing_force_ratio=config.march_swing_unload_ratio,
        foot_clearance=0.006,
        target_foot_clearance=config.march_micro_lift_height,
        base_xy_drift=0.0,
        torque_saturation=0.0,
        curriculum_phase=3,
        config=config,
    )

    assert good["force_ratio"] > poor["force_ratio"]
    assert good["support_ratio_progress"] > poor["support_ratio_progress"]
    assert good["target_side_force_ratio"] > poor["target_side_force_ratio"]
    assert good["swing_unload"] > poor["swing_unload"]
    assert good["foot_clearance"] > poor["foot_clearance"]
    assert good["total"] > poor["total"]


def test_march_phase_one_only_cycles_load_transfer() -> None:
    env = SedonStandingEnv.__new__(SedonStandingEnv)
    env._reward_config = SedonStandingConfig(
        task_mode="march_in_place",
        march_curriculum_phase=1,
        march_phase_steps=4,
    )

    names = []
    for step in range(8):
        env._gait_step = step
        names.append(env._task_phase_metadata()["phase_name"])

    assert set(names) == {"right_load_transfer", "left_load_transfer"}


def test_march_hip_roll_only_residual_mask_freezes_pitch_joints() -> None:
    env = SedonStandingEnv.__new__(SedonStandingEnv)
    env._reward_config = SedonStandingConfig(
        task_mode="march_in_place",
        march_residual_mode="hip_roll_only",
    )

    mask = env._residual_action_mask()

    assert mask[RIGHT_HIP_ROLL_ACTUATOR_INDEX] == pytest.approx(1.0)
    assert mask[LEFT_HIP_ROLL_ACTUATOR_INDEX] == pytest.approx(1.0)
    assert mask[2] == pytest.approx(0.0)
    assert mask[RIGHT_KNEE_JOINT_INDEX] == pytest.approx(0.0)
    assert mask[4] == pytest.approx(0.0)
    assert mask[7] == pytest.approx(0.0)
    assert mask[LEFT_KNEE_JOINT_INDEX] == pytest.approx(0.0)
    assert mask[9] == pytest.approx(0.0)


def test_reference_gait_seed_loader_validates_pose_editor_schema(tmp_path: Path) -> None:
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(
        """{
          "schema": "sedon_gait_seed.v1",
          "target_type": "absolute",
          "joint_names": [
            "R_joint_hip_yaw", "R_joint_hip_roll", "R_joint_hip_pitch",
            "R_joint_knee_pitch", "R_joint_ankle_pitch", "L_joint_hip_yaw",
            "L_joint_hip_roll", "L_joint_hip_pitch", "L_joint_knee_pitch",
            "L_joint_ankle_pitch"
          ],
          "keyframes": [
            {
              "name": "right_swing",
              "support_mode": "left",
              "joint_targets": [0, 0, 0.2, -0.4, 0.1, 0, 0, 0, 0, 0],
              "duration_steps": 3
            },
            {
              "name": "left_swing",
              "support_mode": "right",
              "joint_targets": [0, 0, 0, 0, 0, 0, 0, 0.2, -0.4, 0.1],
              "duration_steps": 5
            }
          ]
        }""",
        encoding="utf-8",
    )

    seed = load_reference_gait_seed(seed_path)

    assert seed.target_type == "absolute"
    assert seed.cycle_steps == 8
    assert [keyframe.name for keyframe in seed.keyframes] == [
        "right_swing",
        "left_swing",
    ]


def test_reference_march_offsets_track_scaled_absolute_seed() -> None:
    env = SedonStandingEnv.__new__(SedonStandingEnv)
    env._reward_config = SedonStandingConfig(
        task_mode="reference_march",
        reference_gait_seed_scale=0.5,
        march_curriculum_phase=2,
    )
    seed = load_reference_gait_seed(
        Path("configs/sedon/reference_march_pose_1_4_mirrored_seed.json")
    )
    env._reference_gait_seed = seed
    env._nominal_joint_qpos = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    env._gait_step = 60

    metadata = env._task_phase_metadata()
    offsets = env._gait_joint_offsets()
    mask = env._residual_action_mask()

    assert metadata["support_side"] == "left"
    assert metadata["swing_side"] == "right"
    assert metadata["curriculum_phase"] == 2
    assert offsets[RIGHT_KNEE_JOINT_INDEX] < 0.0
    assert offsets[RIGHT_KNEE_JOINT_INDEX] > -0.5
    assert mask.shape == (len(JOINT_NAMES),)


def test_reference_march_phase_three_targets_clearance_during_swing() -> None:
    env = SedonStandingEnv.__new__(SedonStandingEnv)
    env._reward_config = SedonStandingConfig(
        task_mode="reference_march",
        reference_gait_seed_scale=0.5,
        march_curriculum_phase=3,
        march_micro_lift_height=0.006,
    )
    env._reference_gait_seed = load_reference_gait_seed(
        Path("configs/sedon/reference_march_pose_1_4_mirrored_seed.json")
    )
    env._nominal_joint_qpos = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    env._gait_step = 60

    metadata = env._task_phase_metadata()
    target_clearance = env._march_target_clearance("reference_march", metadata)

    assert metadata["support_side"] == "left"
    assert metadata["swing_side"] == "right"
    assert target_clearance == pytest.approx(0.006)


def test_march_phase_three_penalizes_excess_swing_force() -> None:
    config = SedonStandingConfig(
        task_mode="reference_march",
        march_curriculum_phase=3,
        march_swing_unload_ratio=0.03,
        march_swing_force_penalty_weight=90.0,
    )
    low_swing_force = compute_march_in_place_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        com_y=0.0,
        target_com_y=0.0,
        base_roll=0.0,
        base_pitch=0.0,
        support_force_ratio=0.9,
        swing_force_ratio=0.02,
        support_ratio_baseline=config.march_support_ratio_baseline,
        target_support_force_ratio=config.march_load_transfer_ratio,
        target_swing_force_ratio=config.march_swing_unload_ratio,
        foot_clearance=config.march_micro_lift_height,
        target_foot_clearance=config.march_micro_lift_height,
        base_xy_drift=0.0,
        torque_saturation=0.0,
        curriculum_phase=3,
        config=config,
    )
    high_swing_force = compute_march_in_place_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        com_y=0.0,
        target_com_y=0.0,
        base_roll=0.0,
        base_pitch=0.0,
        support_force_ratio=0.5,
        swing_force_ratio=0.5,
        support_ratio_baseline=config.march_support_ratio_baseline,
        target_support_force_ratio=config.march_load_transfer_ratio,
        target_swing_force_ratio=config.march_swing_unload_ratio,
        foot_clearance=config.march_micro_lift_height,
        target_foot_clearance=config.march_micro_lift_height,
        base_xy_drift=0.0,
        torque_saturation=0.0,
        curriculum_phase=3,
        config=config,
    )

    assert high_swing_force["swing_force_penalty"] > low_swing_force["swing_force_penalty"]
    assert low_swing_force["total"] > high_swing_force["total"]


def test_march_penalizes_missing_support_force_target() -> None:
    config = SedonStandingConfig(
        task_mode="reference_march",
        march_curriculum_phase=5,
        march_load_transfer_ratio=0.68,
        march_support_force_error_penalty_weight=36.0,
    )

    matched = compute_march_in_place_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=config.march_forward_velocity,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        com_y=0.0,
        target_com_y=0.0,
        base_roll=0.0,
        base_pitch=0.0,
        support_force_ratio=0.68,
        swing_force_ratio=0.32,
        support_ratio_baseline=config.march_support_ratio_baseline,
        target_support_force_ratio=config.march_load_transfer_ratio,
        target_swing_force_ratio=config.march_swing_unload_ratio,
        foot_clearance=config.march_micro_lift_height,
        target_foot_clearance=config.march_micro_lift_height,
        base_xy_drift=0.0,
        torque_saturation=0.0,
        curriculum_phase=5,
        config=config,
    )
    symmetric = compute_march_in_place_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=config.march_forward_velocity,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        com_y=0.0,
        target_com_y=0.0,
        base_roll=0.0,
        base_pitch=0.0,
        support_force_ratio=0.5,
        swing_force_ratio=0.5,
        support_ratio_baseline=config.march_support_ratio_baseline,
        target_support_force_ratio=config.march_load_transfer_ratio,
        target_swing_force_ratio=config.march_swing_unload_ratio,
        foot_clearance=config.march_micro_lift_height,
        target_foot_clearance=config.march_micro_lift_height,
        base_xy_drift=0.0,
        torque_saturation=0.0,
        curriculum_phase=5,
        config=config,
    )

    assert symmetric["support_force_error_penalty"] > matched["support_force_error_penalty"]
    assert matched["total"] > symmetric["total"]


def test_march_phase_five_rewards_swing_foot_forward_lead() -> None:
    config = SedonStandingConfig(
        task_mode="reference_march",
        march_curriculum_phase=5,
        march_swing_forward_distance=0.018,
        march_swing_forward_weight=30.0,
    )

    stuck = compute_march_in_place_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=config.march_forward_velocity,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        com_y=0.0,
        target_com_y=0.0,
        base_roll=0.0,
        base_pitch=0.0,
        support_force_ratio=0.7,
        swing_force_ratio=0.2,
        support_ratio_baseline=config.march_support_ratio_baseline,
        target_support_force_ratio=config.march_load_transfer_ratio,
        target_swing_force_ratio=config.march_swing_unload_ratio,
        foot_clearance=config.march_micro_lift_height,
        target_foot_clearance=config.march_micro_lift_height,
        base_xy_drift=0.0,
        torque_saturation=0.0,
        curriculum_phase=5,
        config=config,
        swing_foot_forward_delta=0.0,
        target_swing_foot_forward_delta=config.march_swing_forward_distance,
    )
    shuffling = compute_march_in_place_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=config.march_forward_velocity,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        com_y=0.0,
        target_com_y=0.0,
        base_roll=0.0,
        base_pitch=0.0,
        support_force_ratio=0.7,
        swing_force_ratio=0.2,
        support_ratio_baseline=config.march_support_ratio_baseline,
        target_support_force_ratio=config.march_load_transfer_ratio,
        target_swing_force_ratio=config.march_swing_unload_ratio,
        foot_clearance=config.march_micro_lift_height,
        target_foot_clearance=config.march_micro_lift_height,
        base_xy_drift=0.0,
        torque_saturation=0.0,
        curriculum_phase=5,
        config=config,
        swing_foot_forward_delta=0.018,
        target_swing_foot_forward_delta=config.march_swing_forward_distance,
    )

    assert stuck["swing_forward_progress"] == pytest.approx(0.0)
    assert shuffling["swing_forward_progress"] == pytest.approx(1.0)
    assert shuffling["total"] > stuck["total"]


def test_march_phase_three_penalizes_losing_all_foot_load() -> None:
    config = SedonStandingConfig(
        task_mode="reference_march",
        march_curriculum_phase=3,
        march_no_contact_penalty_weight=25.0,
        march_single_contact_penalty_weight=10.0,
    )

    both_loaded = compute_march_in_place_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        com_y=0.0,
        target_com_y=0.0,
        base_roll=0.0,
        base_pitch=0.0,
        support_force_ratio=0.6,
        swing_force_ratio=0.4,
        support_ratio_baseline=config.march_support_ratio_baseline,
        target_support_force_ratio=config.march_load_transfer_ratio,
        target_swing_force_ratio=config.march_swing_unload_ratio,
        foot_clearance=config.march_micro_lift_height,
        target_foot_clearance=config.march_micro_lift_height,
        base_xy_drift=0.0,
        torque_saturation=0.0,
        curriculum_phase=3,
        config=config,
    )
    airborne = compute_march_in_place_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        com_y=0.0,
        target_com_y=0.0,
        base_roll=0.0,
        base_pitch=0.0,
        support_force_ratio=0.0,
        swing_force_ratio=0.0,
        support_ratio_baseline=config.march_support_ratio_baseline,
        target_support_force_ratio=config.march_load_transfer_ratio,
        target_swing_force_ratio=config.march_swing_unload_ratio,
        foot_clearance=config.march_micro_lift_height,
        target_foot_clearance=config.march_micro_lift_height,
        base_xy_drift=0.0,
        torque_saturation=0.0,
        curriculum_phase=3,
        config=config,
    )

    assert airborne["no_contact_penalty"] == pytest.approx(1.0)
    assert both_loaded["no_contact_penalty"] == pytest.approx(0.0)
    assert both_loaded["total"] > airborne["total"]


def test_reference_teacher_reward_penalizes_drop_impact_and_jump() -> None:
    config = SedonStandingConfig(
        task_mode="reference_march",
        march_curriculum_phase=3,
        march_jump_penalty_weight=120.0,
        march_base_height_drop_penalty_weight=80.0,
        march_landing_impact_penalty_weight=8.0,
    )

    stable = compute_march_in_place_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        com_y=0.0,
        target_com_y=0.0,
        base_roll=0.0,
        base_pitch=0.0,
        support_force_ratio=0.6,
        swing_force_ratio=0.4,
        support_ratio_baseline=config.march_support_ratio_baseline,
        target_support_force_ratio=0.6,
        target_swing_force_ratio=0.4,
        foot_clearance=0.0012,
        target_foot_clearance=0.0012,
        base_xy_drift=0.0,
        torque_saturation=0.0,
        curriculum_phase=3,
        config=config,
        base_height_drop=0.0,
        total_force_normalized=1.0,
    )
    harsh = compute_march_in_place_reward(
        base_height=config.target_base_height,
        upright=1.0,
        joint_velocity_l2=0.0,
        action_l2=0.0,
        action_rate_l2=0.0,
        joint_position_error_l2=0.0,
        forward_velocity=0.0,
        base_roll_pitch_rate_l2=0.0,
        foot_flatness=1.0,
        com_y=0.0,
        target_com_y=0.0,
        base_roll=0.0,
        base_pitch=0.0,
        support_force_ratio=0.0,
        swing_force_ratio=0.0,
        support_ratio_baseline=config.march_support_ratio_baseline,
        target_support_force_ratio=0.6,
        target_swing_force_ratio=0.4,
        foot_clearance=0.0,
        target_foot_clearance=0.0012,
        base_xy_drift=0.0,
        torque_saturation=0.0,
        curriculum_phase=3,
        config=config,
        base_height_drop=0.03,
        total_force_normalized=1.8,
    )

    assert harsh["jump_penalty"] > stable["jump_penalty"]
    assert harsh["base_height_drop_penalty"] > stable["base_height_drop_penalty"]
    assert harsh["landing_impact_penalty"] > stable["landing_impact_penalty"]
    assert stable["total"] > harsh["total"]


def test_march_phase_five_includes_full_alternating_sequence() -> None:
    env = SedonStandingEnv.__new__(SedonStandingEnv)
    env._reward_config = SedonStandingConfig(
        task_mode="march_in_place",
        march_curriculum_phase=5,
        march_phase_steps=2,
    )

    names = []
    for step in range(16):
        env._gait_step = step
        names.append(env._task_phase_metadata()["phase_name"])

    assert names[0] == "right_load_transfer"
    assert "right_swing_unload" in names
    assert "right_micro_lift" in names
    assert "left_lower" in names


def test_sedon_termination_reason_reports_specific_failure() -> None:
    env = SedonStandingEnv.__new__(SedonStandingEnv)
    env._reward_config = SedonStandingConfig()
    env.data = SimpleNamespace(qpos=np.zeros(7, dtype=np.float64))
    observation = np.zeros(44, dtype=np.float64)

    reason = env._termination_reason(
        base_height=env._reward_config.min_base_height - 0.01,
        upright=1.0,
        observation=observation,
    )

    assert reason == "base_height_low"


def test_sedon_fsm_offsets_honor_configured_double_support_window() -> None:
    env = SedonStandingEnv.__new__(SedonStandingEnv)
    env._reward_config = SedonStandingConfig(
        gait_mode="fsm",
        fsm_right_lift_steps=2,
        fsm_right_lower_steps=2,
        fsm_left_lift_steps=2,
        fsm_left_lower_steps=2,
        fsm_double_support_steps=3,
        gait_hip_roll_amp=0.1,
        gait_hip_pitch_amp=-0.2,
        gait_knee_pitch_amp=-0.2,
        gait_ankle_pitch_amp=0.2,
    )
    env._gait_step = 5

    offsets = env._fsm_gait_joint_offsets()

    np.testing.assert_allclose(offsets, np.zeros(10, dtype=np.float64))


def test_sedon_fsm_walk_phase_metadata_tracks_support_side() -> None:
    env = SedonStandingEnv.__new__(SedonStandingEnv)
    env._reward_config = SedonStandingConfig(
        task_mode="walk",
        gait_mode="fsm",
        fsm_right_lift_steps=2,
        fsm_right_lower_steps=2,
        fsm_left_lift_steps=2,
        fsm_left_lower_steps=2,
        fsm_double_support_steps=2,
    )

    env._gait_step = 0
    right_swing = env._task_phase_metadata()
    env._gait_step = 4
    double_support = env._task_phase_metadata()
    env._gait_step = 6
    left_swing = env._task_phase_metadata()

    assert right_swing["phase_name"] == "right_swing_lift"
    assert right_swing["support_side"] == "left"
    assert right_swing["swing_side"] == "right"
    assert right_swing["in_double_support"] is False
    assert double_support["support_side"] == "center"
    assert double_support["in_double_support"] is True
    assert left_swing["phase_name"] == "left_swing_lift"
    assert left_swing["support_side"] == "right"
    assert left_swing["swing_side"] == "left"


def test_sedon_fsm_offsets_use_configured_support_roll_scale() -> None:
    env = SedonStandingEnv.__new__(SedonStandingEnv)
    env._reward_config = SedonStandingConfig(
        gait_mode="fsm",
        fsm_right_lift_steps=4,
        fsm_right_lower_steps=1,
        fsm_left_lift_steps=1,
        fsm_left_lower_steps=1,
        fsm_double_support_steps=0,
        fsm_right_support_roll_scale=2.0,
        gait_hip_roll_amp=0.05,
        gait_hip_pitch_amp=-0.1,
        gait_knee_pitch_amp=-0.1,
        gait_ankle_pitch_amp=0.1,
    )
    env._gait_step = 1

    offsets = env._fsm_gait_joint_offsets()

    assert offsets[6] > 0.0
    assert offsets[6] == pytest.approx(0.05)


def test_do_pd_simulation_with_torque_assist_clamps_and_records_actual_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = SedonStandingEnv.__new__(SedonStandingEnv)
    env.frame_skip = 2
    env.model = object()
    env.data = SimpleNamespace(ctrl=np.zeros(10, dtype=np.float64))
    env._ctrl_range = np.tile(np.array([[-1.0, 1.0]], dtype=np.float64), (10, 1))
    env._ctrl_range[RIGHT_HIP_ROLL_ACTUATOR_INDEX] = np.array([-0.45, 0.45], dtype=np.float64)
    env._ctrl_range[LEFT_HIP_ROLL_ACTUATOR_INDEX] = np.array([-0.5, 0.5], dtype=np.float64)
    env._last_ctrl_assist_delta = np.zeros(10, dtype=np.float64)

    base_ctrl = np.zeros(10, dtype=np.float64)
    base_ctrl[RIGHT_HIP_ROLL_ACTUATOR_INDEX] = 0.4
    base_ctrl[LEFT_HIP_ROLL_ACTUATOR_INDEX] = -0.45
    env._pd_control = lambda target_positions: base_ctrl.copy()  # type: ignore[method-assign]

    expected_ctrl = base_ctrl.copy()
    expected_ctrl[RIGHT_HIP_ROLL_ACTUATOR_INDEX] = 0.45
    expected_ctrl[LEFT_HIP_ROLL_ACTUATOR_INDEX] = -0.5

    observed_ctrl: list[np.ndarray] = []

    def fake_mj_step(model: object, data: SimpleNamespace) -> None:
        observed_ctrl.append(np.array(data.ctrl, copy=True))

    monkeypatch.setattr(sedon_env_module, "mujoco", SimpleNamespace(mj_step=fake_mj_step))

    env._do_pd_simulation_with_torque_assist(
        np.zeros(10, dtype=np.float64),
        left_tau_assist=-0.2,
        right_tau_assist=0.2,
    )

    assert len(observed_ctrl) == env.frame_skip
    for ctrl in observed_ctrl:
        np.testing.assert_allclose(ctrl, expected_ctrl)
    np.testing.assert_allclose(env.data.ctrl, expected_ctrl)
    assert env.last_hip_roll_ctrl_assist_delta() == pytest.approx((-0.05, 0.05))
    assert env.data.ctrl[0] == pytest.approx(base_ctrl[0])


def test_do_pd_simulation_without_assist_keeps_zero_injected_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = SedonStandingEnv.__new__(SedonStandingEnv)
    env.frame_skip = 1
    env.model = object()
    env.data = SimpleNamespace(ctrl=np.zeros(10, dtype=np.float64))
    env._ctrl_range = np.tile(np.array([[-1.0, 1.0]], dtype=np.float64), (10, 1))
    env._last_ctrl_assist_delta = np.ones(10, dtype=np.float64)

    base_ctrl = np.linspace(-0.4, 0.4, num=10, dtype=np.float64)
    env._pd_control = lambda target_positions: base_ctrl.copy()  # type: ignore[method-assign]

    monkeypatch.setattr(
        sedon_env_module,
        "mujoco",
        SimpleNamespace(mj_step=lambda model, data: None),
    )

    env._do_pd_simulation(np.zeros(10, dtype=np.float64))

    np.testing.assert_allclose(env.data.ctrl, base_ctrl)
    assert env.last_hip_roll_ctrl_assist_delta() == pytest.approx((0.0, 0.0))


@pytest.fixture
def sedon_env_class() -> type[SedonStandingEnv]:
    """Return the Sedon env class, skipping when private assets are unavailable."""
    pytest.importorskip("mujoco")
    if not Path(DEFAULT_SCENE_PATH).is_file():
        pytest.skip("Sedon private training_scene.xml has not been generated.")
    return SedonStandingEnv


def test_sedon_env_reset_returns_expected_observation_shape(
    sedon_env_class: type[SedonStandingEnv],
) -> None:
    env = sedon_env_class(reset_noise_scale=0.0)
    try:
        obs, _ = env.reset(seed=42)
        assert obs.shape == (44,)
        assert np.isfinite(obs).all()
        assert env.action_space.shape == (10,)
    finally:
        env.close()


def test_sedon_env_zero_action_step_stays_numeric(
    sedon_env_class: type[SedonStandingEnv],
) -> None:
    env = sedon_env_class(reset_noise_scale=0.0)
    try:
        env.reset(seed=42)
        action = np.zeros(env.action_space.shape, dtype=np.float64)
        obs, reward, _, _, info = env.step(action)

        assert obs.shape == (44,)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert info["base_height"] > 0.0
    finally:
        env.close()
