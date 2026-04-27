"""Regression tests for the fixed-base grasp reset/controller setup."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.mujoco

from grasp_baseline.env import FixedBaseGraspEnv


@pytest.fixture
def fixed_base_grasp_env_class() -> type[FixedBaseGraspEnv]:
    """Return the MuJoCo env class, skipping cleanly when MuJoCo is absent."""
    pytest.importorskip("mujoco")
    return FixedBaseGraspEnv


def test_reset_pose_starts_near_cube_for_learning(
    fixed_base_grasp_env_class: type[FixedBaseGraspEnv],
) -> None:
    env = fixed_base_grasp_env_class(task_phase="full", randomize_cube_pose=False)
    try:
        env.reset(seed=42)
        distance = float(np.linalg.norm(env._cube_pos() - env._gripper_pos()))
        assert distance < 0.10
    finally:
        env.close()


def test_zero_action_does_not_leave_reach_zone_immediately(
    fixed_base_grasp_env_class: type[FixedBaseGraspEnv],
) -> None:
    env = fixed_base_grasp_env_class(task_phase="full", randomize_cube_pose=False)
    try:
        env.reset(seed=42)
        action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float64)
        last_distance = 0.0
        for _ in range(10):
            _, _, terminated, truncated, info = env.step(action)
            last_distance = float(info["distance_to_cube"])
            assert not terminated
            assert not truncated
        assert last_distance < 0.11
    finally:
        env.close()
