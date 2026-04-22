"""Tests for the standalone grasp reward and success logic."""

from grasp_baseline.env import (
    GraspRewardConfig,
    GraspRewardInputs,
    compute_reward_components,
    is_successful_grasp,
)


def _make_inputs(**overrides) -> GraspRewardInputs:
    base = dict(
        distance_to_cube=0.08,
        left_contact=False,
        right_contact=False,
        cube_height_above_table=0.0,
        cube_to_gripper_distance=0.08,
        gripper_closing_command=0.0,
        gripper_opening=0.02,
        joint_limit_fraction=0.0,
        action_l2=0.0,
        cube_speed=0.0,
        cube_dropped=False,
        hold_ratio=0.0,
    )
    base.update(overrides)
    return GraspRewardInputs(**base)


def test_reach_reward_increases_when_gripper_moves_closer() -> None:
    config = GraspRewardConfig()
    far_inputs = _make_inputs(distance_to_cube=0.12)
    near_inputs = _make_inputs(distance_to_cube=0.03)

    far_reward = compute_reward_components(far_inputs, config, task_phase="reach")
    near_reward = compute_reward_components(near_inputs, config, task_phase="reach")

    assert near_reward["reach"] > far_reward["reach"]
    assert near_reward["total"] > far_reward["total"]


def test_lift_phase_reward_requires_cube_height() -> None:
    config = GraspRewardConfig()
    grounded = _make_inputs(
        left_contact=True,
        right_contact=True,
        cube_to_gripper_distance=0.03,
        cube_height_above_table=0.0,
        hold_ratio=0.2,
    )
    lifted = _make_inputs(
        left_contact=True,
        right_contact=True,
        cube_to_gripper_distance=0.03,
        cube_height_above_table=config.success_lift_height,
        hold_ratio=0.8,
    )

    grounded_reward = compute_reward_components(grounded, config, task_phase="lift")
    lifted_reward = compute_reward_components(lifted, config, task_phase="lift")

    assert lifted_reward["lift"] > grounded_reward["lift"]
    assert lifted_reward["hold"] > grounded_reward["hold"]
    assert lifted_reward["total"] > grounded_reward["total"]


def test_success_requires_contacts_alignment_height_and_hold_steps() -> None:
    config = GraspRewardConfig(success_hold_steps=10)
    ready = _make_inputs(
        left_contact=True,
        right_contact=True,
        cube_height_above_table=0.07,
        cube_to_gripper_distance=0.03,
    )
    missing_contact = _make_inputs(
        left_contact=True,
        right_contact=False,
        cube_height_above_table=0.07,
        cube_to_gripper_distance=0.03,
    )

    assert not is_successful_grasp(ready, config, hold_steps=9)
    assert is_successful_grasp(ready, config, hold_steps=10)
    assert not is_successful_grasp(missing_contact, config, hold_steps=10)
