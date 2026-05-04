"""Regression tests for the Sedon standing environment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.mujoco

from sedon_baseline.env import (
    DEFAULT_SCENE_PATH,
    SedonStandingConfig,
    SedonStandingEnv,
    compute_standing_reward,
)


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
        assert obs.shape == (43,)
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

        assert obs.shape == (43,)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert info["base_height"] > 0.0
    finally:
        env.close()
