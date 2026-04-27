"""Fixed-base grasping baseline built separately from the H1 walking env.

This module intentionally does not modify or depend on the existing H1
locomotion reward structure. It provides a minimal manipulation baseline:

- fixed robot base
- 3-DoF arm
- 1 parallel gripper
- 1 cube on a table

The goal is to learn ``reach -> grasp -> lift`` before adding locomotion or
vision. Reward shaping and success logic are exposed through pure helpers so
the core rules can be tested without running long RL jobs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

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
        """Placeholder so pure reward helpers remain importable without MuJoCo."""

        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_XML_PATH = os.path.join(_HERE, "assets", "scene.xml")

_TASK_PHASES = {"reach", "grasp", "lift", "full"}


@dataclass(frozen=True)
class GraspRewardConfig:
    """Reward and success thresholds for the fixed-base grasp task."""

    reach_scale: float = 4.5
    grasp_contact_weight: float = 0.35
    close_weight: float = 0.25
    lift_weight: float = 3.0
    hold_weight: float = 2.5
    action_penalty_weight: float = 0.01
    joint_limit_penalty_weight: float = 0.15
    speed_penalty_weight: float = 0.02
    drop_penalty_weight: float = 1.5
    success_bonus: float = 5.0
    near_object_distance: float = 0.08
    alignment_tolerance: float = 0.07
    success_lift_height: float = 0.06
    success_hold_steps: int = 20
    joint_limit_soft_fraction: float = 0.90


@dataclass(frozen=True)
class GraspRewardInputs:
    """Inputs required to compute reward components and success checks."""

    distance_to_cube: float
    left_contact: bool
    right_contact: bool
    cube_height_above_table: float
    cube_to_gripper_distance: float
    gripper_closing_command: float
    gripper_opening: float
    joint_limit_fraction: float
    action_l2: float
    cube_speed: float
    cube_dropped: bool
    hold_ratio: float


def compute_reward_components(
    inputs: GraspRewardInputs,
    config: GraspRewardConfig,
    task_phase: str = "full",
) -> dict[str, float]:
    """Return shaped reward components for the grasp task.

    Args:
        inputs: Current geometric and control statistics.
        config: Reward weights and thresholds.
        task_phase: One of ``reach``, ``grasp``, ``lift``, or ``full``.

    Returns:
        A dict containing positive reward terms, penalties, and ``total``.

    Raises:
        ValueError: If ``task_phase`` is not supported.
    """
    if task_phase not in _TASK_PHASES:
        raise ValueError(
            f"Unsupported task_phase '{task_phase}'. Expected one of "
            f"{sorted(_TASK_PHASES)}."
        )

    near_object = max(
        0.0,
        1.0 - (inputs.distance_to_cube / config.near_object_distance),
    )
    alignment = max(
        0.0,
        1.0 - (inputs.cube_to_gripper_distance / config.alignment_tolerance),
    )
    contact_count = float(inputs.left_contact) + float(inputs.right_contact)
    lift_progress = max(
        0.0,
        inputs.cube_height_above_table / config.success_lift_height,
    )

    rewards = {
        "reach": float(np.exp(-config.reach_scale * inputs.distance_to_cube)),
        "grasp_contact": contact_count * alignment,
        "close": inputs.gripper_closing_command * near_object,
        "lift": min(lift_progress, 1.5),
        "hold": inputs.hold_ratio * alignment,
        "action_penalty": inputs.action_l2,
        "joint_limit_penalty": inputs.joint_limit_fraction,
        "speed_penalty": inputs.cube_speed,
        "drop_penalty": 1.0 if inputs.cube_dropped else 0.0,
    }

    total = rewards["reach"]
    total -= config.action_penalty_weight * rewards["action_penalty"]
    total -= config.joint_limit_penalty_weight * rewards["joint_limit_penalty"]

    if task_phase in {"grasp", "lift", "full"}:
        total += config.grasp_contact_weight * rewards["grasp_contact"]
        total += config.close_weight * rewards["close"]

    if task_phase in {"lift", "full"}:
        total += config.lift_weight * rewards["lift"]
        total += config.hold_weight * rewards["hold"]
        total -= config.speed_penalty_weight * rewards["speed_penalty"]
        total -= config.drop_penalty_weight * rewards["drop_penalty"]

    rewards["total"] = float(total)
    return rewards


def is_successful_grasp(
    inputs: GraspRewardInputs,
    config: GraspRewardConfig,
    hold_steps: int,
) -> bool:
    """Check whether the episode meets the grasp success condition.

    Args:
        inputs: Current geometric and control statistics.
        config: Reward and success thresholds.
        hold_steps: Consecutive timesteps that already satisfied the hold rule.

    Returns:
        ``True`` when the cube is lifted, aligned with the gripper, and held
        long enough to avoid transient false positives.
    """
    both_fingers_touching = inputs.left_contact and inputs.right_contact
    lifted_enough = inputs.cube_height_above_table >= config.success_lift_height
    aligned = inputs.cube_to_gripper_distance <= config.alignment_tolerance
    return bool(
        both_fingers_touching
        and lifted_enough
        and aligned
        and hold_steps >= config.success_hold_steps
    )


class FixedBaseGraspEnv(MujocoEnv):
    """MuJoCo environment for fixed-base cube grasping.

    Args:
        task_phase: Reward curriculum phase. ``full`` is the default end-to-end
            task. Earlier phases can be used to debug or stage training.
        randomize_cube_pose: Whether to randomize the cube position at reset.
        cube_xy_range: Maximum absolute XY offset applied to the cube reset
            position when randomization is enabled.
        frame_skip: Number of MuJoCo steps per environment step.
        reward_config: Optional reward override.
        **kwargs: Forwarded to ``MujocoEnv`` such as ``render_mode``.
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        task_phase: str = "full",
        randomize_cube_pose: bool = True,
        cube_xy_range: float = 0.025,
        frame_skip: int = 10,
        reward_config: GraspRewardConfig | None = None,
        **kwargs,
    ) -> None:
        if _MUJOCO_IMPORT_ERROR is not None:
            raise ModuleNotFoundError(
                "FixedBaseGraspEnv requires the 'mujoco' and Gymnasium MuJoCo "
                "dependencies. Install project requirements before creating "
                "the simulator environment."
            ) from _MUJOCO_IMPORT_ERROR
        if task_phase not in _TASK_PHASES:
            raise ValueError(
                f"Unsupported task_phase '{task_phase}'. Expected one of "
                f"{sorted(_TASK_PHASES)}."
            )
        if cube_xy_range < 0.0:
            raise ValueError("cube_xy_range must be non-negative.")

        self._task_phase = task_phase
        self._randomize_cube_pose = randomize_cube_pose
        self._cube_xy_range = cube_xy_range
        self._reward_config = reward_config or GraspRewardConfig()
        self._arm_action_scale = np.array([0.45, 0.45, 0.55], dtype=np.float64)
        # Start with the gripper hovering near the cube instead of hanging
        # beside the table. This keeps early exploration inside a learnable
        # reach region instead of spending most episodes recovering from a bad
        # default posture.
        self._default_arm_targets = np.array([0.0, 0.15, 0.85], dtype=np.float64)
        self._max_arm_target_step = np.array([0.12, 0.12, 0.14], dtype=np.float64)
        self._max_finger_target_step = 0.004
        self._min_finger_opening = 0.001
        self._max_finger_opening = 0.020
        self._prev_action = np.zeros(4, dtype=np.float64)
        self._episode_steps = 0
        self._lift_hold_steps = 0
        self._had_lift = False
        self._drop_penalized = False
        self._last_cube_pos = np.zeros(3, dtype=np.float64)

        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(28,),
            dtype=np.float64,
        )
        super().__init__(
            model_path=_XML_PATH,
            frame_skip=frame_skip,
            observation_space=observation_space,
            default_camera_config={
                "distance": 1.5,
                "lookat": np.array([0.32, 0.0, 0.36]),
                "elevation": -20.0,
                "azimuth": 135.0,
            },
            **kwargs,
        )

        self._arm_joint_names = [
            "shoulder_yaw",
            "shoulder_pitch",
            "elbow_pitch",
        ]
        self._finger_joint_names = [
            "left_finger_slide",
            "right_finger_slide",
        ]
        self._controlled_joint_names = (
            self._arm_joint_names + self._finger_joint_names
        )
        self._arm_joint_ids = [self._joint_id(name) for name in self._arm_joint_names]
        self._finger_joint_ids = [
            self._joint_id(name) for name in self._finger_joint_names
        ]
        self._controlled_joint_ids = [
            self._joint_id(name) for name in self._controlled_joint_names
        ]

        self._cube_body_id = self._body_id("cube")
        self._cube_joint_id = self._joint_id("cube_free")
        self._cube_geom_id = self._geom_id("cube_geom")
        self._left_finger_geom_id = self._geom_id("left_finger_geom")
        self._right_finger_geom_id = self._geom_id("right_finger_geom")
        self._table_geom_id = self._geom_id("table_geom")
        self._gripper_site_id = self._site_id("gripper_center")
        self._cube_qpos_adr = self.model.jnt_qposadr[self._cube_joint_id]
        self._cube_qvel_adr = self.model.jnt_dofadr[self._cube_joint_id]

        self._table_top_height = self._compute_table_top_height()
        cube_half_extents = self.model.geom_size[self._cube_geom_id]
        self._cube_half_extent = float(np.max(cube_half_extents))
        self._cube_spawn_center = np.array(
            [
                0.39,
                0.0,
                self._table_top_height + self._cube_half_extent,
            ],
            dtype=np.float64,
        )

        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float64,
        )

    def step(self, action: np.ndarray):
        """Advance the simulation and compute shaped grasp reward."""
        clipped_action = np.clip(action, -1.0, 1.0).astype(np.float64)
        arm_action = clipped_action[:3]
        closing_command = 0.5 * (clipped_action[3] + 1.0)
        finger_target = self._max_finger_opening - closing_command * (
            self._max_finger_opening - self._min_finger_opening
        )
        current_arm_targets = np.array(
            [self._joint_qpos(joint_id) for joint_id in self._arm_joint_ids],
            dtype=np.float64,
        )
        desired_arm_targets = self._default_arm_targets + arm_action * self._arm_action_scale
        arm_target_delta = np.clip(
            desired_arm_targets - current_arm_targets,
            -self._max_arm_target_step,
            self._max_arm_target_step,
        )
        safe_arm_targets = current_arm_targets + arm_target_delta
        current_finger_opening = self._finger_opening()
        safe_finger_target = np.clip(
            finger_target,
            current_finger_opening - self._max_finger_target_step,
            current_finger_opening + self._max_finger_target_step,
        )
        safe_finger_target = float(
            np.clip(safe_finger_target, self._min_finger_opening, self._max_finger_opening)
        )

        ctrl = np.empty(5, dtype=np.float64)
        ctrl[:3] = safe_arm_targets
        ctrl[3] = safe_finger_target
        ctrl[4] = safe_finger_target

        self.do_simulation(ctrl, self.frame_skip)
        self._episode_steps += 1

        reward_inputs = self._collect_reward_inputs(clipped_action)
        success_ready = self._eligible_for_hold_count(reward_inputs)
        if success_ready:
            self._lift_hold_steps += 1
        else:
            self._lift_hold_steps = 0

        reward_inputs = self._collect_reward_inputs(
            clipped_action,
            hold_steps_override=self._lift_hold_steps,
        )
        rewards = compute_reward_components(
            reward_inputs,
            self._reward_config,
            task_phase=self._task_phase,
        )
        success = is_successful_grasp(
            reward_inputs,
            self._reward_config,
            self._lift_hold_steps,
        )
        if success:
            rewards["total"] += self._reward_config.success_bonus

        cube_fell = self._cube_has_fallen()
        terminated = bool(success or cube_fell)
        obs = self._get_obs()

        info = {
            "task_phase": self._task_phase,
            "cube_height_above_table": reward_inputs.cube_height_above_table,
            "cube_to_gripper_distance": reward_inputs.cube_to_gripper_distance,
            "distance_to_cube": reward_inputs.distance_to_cube,
            "left_contact": reward_inputs.left_contact,
            "right_contact": reward_inputs.right_contact,
            "lift_hold_steps": self._lift_hold_steps,
            "is_success": success,
            "cube_fell": cube_fell,
        }
        for key, value in rewards.items():
            info[f"reward_{key}"] = value

        self._prev_action = clipped_action.copy()
        self._last_cube_pos = self._cube_pos().copy()
        return obs, float(rewards["total"]), terminated, False, info

    def reset_model(self) -> np.ndarray:
        """Reset arm pose and respawn the cube on the table."""
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        for joint_id, joint_target in zip(
            self._arm_joint_ids,
            self._default_arm_targets,
            strict=True,
        ):
            qpos[self.model.jnt_qposadr[joint_id]] = joint_target

        for joint_id in self._finger_joint_ids:
            qpos[self.model.jnt_qposadr[joint_id]] = self._max_finger_opening

        cube_pos = self._cube_spawn_center.copy()
        if self._randomize_cube_pose:
            cube_pos[0] += self.np_random.uniform(
                -self._cube_xy_range,
                self._cube_xy_range,
            )
            cube_pos[1] += self.np_random.uniform(
                -self._cube_xy_range,
                self._cube_xy_range,
            )

        yaw = self.np_random.uniform(-0.35, 0.35)
        cube_quat = np.array(
            [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
            dtype=np.float64,
        )
        qpos[self._cube_qpos_adr : self._cube_qpos_adr + 3] = cube_pos
        qpos[self._cube_qpos_adr + 3 : self._cube_qpos_adr + 7] = cube_quat
        qvel[self._cube_qvel_adr : self._cube_qvel_adr + 6] = 0.0

        self.set_state(qpos, qvel)
        self._episode_steps = 0
        self._lift_hold_steps = 0
        self._had_lift = False
        self._drop_penalized = False
        self._prev_action = np.zeros(4, dtype=np.float64)
        self._last_cube_pos = self._cube_pos().copy()
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Assemble the observation vector."""
        joint_pos = np.array(
            [self._joint_qpos(joint_id) for joint_id in self._controlled_joint_ids],
            dtype=np.float64,
        )
        joint_vel = np.array(
            [self._joint_qvel(joint_id) for joint_id in self._controlled_joint_ids],
            dtype=np.float64,
        )
        gripper_pos = self._gripper_pos()
        cube_pos = self._cube_pos()
        cube_vel = (cube_pos - self._last_cube_pos) / self.dt
        relative_pos = cube_pos - gripper_pos
        contact_flags = np.array(
            [float(self._left_finger_touches_cube()), float(self._right_finger_touches_cube())],
            dtype=np.float64,
        )
        obs = np.concatenate(
            [
                joint_pos,
                joint_vel,
                gripper_pos,
                cube_pos,
                cube_vel,
                relative_pos,
                self._prev_action,
                contact_flags,
            ]
        )
        return obs.astype(np.float64)

    def _collect_reward_inputs(
        self,
        action: np.ndarray,
        hold_steps_override: int | None = None,
    ) -> GraspRewardInputs:
        """Collect the current reward features from simulation state."""
        gripper_pos = self._gripper_pos()
        cube_pos = self._cube_pos()
        cube_vel = (cube_pos - self._last_cube_pos) / self.dt
        left_contact = self._left_finger_touches_cube()
        right_contact = self._right_finger_touches_cube()
        cube_height_above_table = max(
            0.0,
            cube_pos[2] - self._cube_half_extent - self._table_top_height,
        )
        cube_to_gripper_distance = float(np.linalg.norm(cube_pos - gripper_pos))
        if cube_height_above_table >= 0.5 * self._reward_config.success_lift_height:
            self._had_lift = True

        cube_dropped = (
            self._had_lift
            and cube_height_above_table <= 0.01
            and not left_contact
            and not right_contact
            and not self._drop_penalized
        )
        if cube_dropped:
            self._drop_penalized = True

        hold_steps = (
            self._lift_hold_steps
            if hold_steps_override is None
            else hold_steps_override
        )
        hold_ratio = min(
            1.0,
            hold_steps / max(self._reward_config.success_hold_steps, 1),
        )
        return GraspRewardInputs(
            distance_to_cube=float(np.linalg.norm(cube_pos - gripper_pos)),
            left_contact=left_contact,
            right_contact=right_contact,
            cube_height_above_table=float(cube_height_above_table),
            cube_to_gripper_distance=cube_to_gripper_distance,
            gripper_closing_command=float(0.5 * (action[3] + 1.0)),
            gripper_opening=float(self._finger_opening()),
            joint_limit_fraction=float(self._joint_limit_fraction()),
            action_l2=float(np.dot(action, action)),
            cube_speed=float(np.linalg.norm(cube_vel)),
            cube_dropped=cube_dropped,
            hold_ratio=float(hold_ratio),
        )

    def _eligible_for_hold_count(self, inputs: GraspRewardInputs) -> bool:
        """Check whether the current state should accumulate hold progress."""
        return bool(
            inputs.left_contact
            and inputs.right_contact
            and inputs.cube_height_above_table >= self._reward_config.success_lift_height
            and inputs.cube_to_gripper_distance
            <= self._reward_config.alignment_tolerance
        )

    def _cube_has_fallen(self) -> bool:
        """Terminate early when the cube falls off the table."""
        cube_pos = self._cube_pos()
        return bool(
            cube_pos[2] < self._table_top_height - 0.08
            or abs(cube_pos[1]) > 0.35
            or cube_pos[0] < 0.12
            or cube_pos[0] > 0.75
        )

    def _joint_limit_fraction(self) -> float:
        """Return how strongly the arm is pushing into soft joint limits."""
        fractions: list[float] = []
        for joint_id in self._arm_joint_ids:
            qpos_adr = self.model.jnt_qposadr[joint_id]
            joint_pos = self.data.qpos[qpos_adr]
            lo, hi = self.model.jnt_range[joint_id]
            if hi <= lo:
                continue
            center = 0.5 * (lo + hi)
            half_range = 0.5 * (hi - lo)
            normalized = abs(joint_pos - center) / half_range
            soft = self._reward_config.joint_limit_soft_fraction
            fractions.append(max(0.0, (normalized - soft) / max(1e-6, 1.0 - soft)))
        return float(np.mean(fractions)) if fractions else 0.0

    def _finger_opening(self) -> float:
        """Return the average finger slide distance."""
        left = self._joint_qpos(self._finger_joint_ids[0])
        right = self._joint_qpos(self._finger_joint_ids[1])
        return 0.5 * (left + right)

    def _left_finger_touches_cube(self) -> bool:
        """Return whether the left finger currently contacts the cube."""
        return self._geom_pair_in_contact(self._left_finger_geom_id, self._cube_geom_id)

    def _right_finger_touches_cube(self) -> bool:
        """Return whether the right finger currently contacts the cube."""
        return self._geom_pair_in_contact(self._right_finger_geom_id, self._cube_geom_id)

    def _geom_pair_in_contact(self, geom_a: int, geom_b: int) -> bool:
        """Check if two geoms are in active contact."""
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            if (
                contact.geom1 == geom_a
                and contact.geom2 == geom_b
            ) or (
                contact.geom1 == geom_b
                and contact.geom2 == geom_a
            ):
                return True
        return False

    def _joint_qpos(self, joint_id: int) -> float:
        """Return scalar qpos for a named hinge/slide joint."""
        return float(self.data.qpos[self.model.jnt_qposadr[joint_id]])

    def _joint_qvel(self, joint_id: int) -> float:
        """Return scalar qvel for a named hinge/slide joint."""
        return float(self.data.qvel[self.model.jnt_dofadr[joint_id]])

    def _cube_pos(self) -> np.ndarray:
        """Return the cube world position."""
        return self.data.xpos[self._cube_body_id].copy()

    def _gripper_pos(self) -> np.ndarray:
        """Return the gripper center world position."""
        return self.data.site_xpos[self._gripper_site_id].copy()

    def _compute_table_top_height(self) -> float:
        """Return the table top height from model geometry."""
        table_body_id = self._body_id("table")
        geom_pos = self.model.geom_pos[self._table_geom_id]
        geom_half_size = self.model.geom_size[self._table_geom_id]
        body_pos = self.model.body_pos[table_body_id]
        return float(body_pos[2] + geom_pos[2] + geom_half_size[2])

    def _body_id(self, name: str) -> int:
        """Resolve a MuJoCo body id by name."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body '{name}' not found in grasp model.")
        return body_id

    def _geom_id(self, name: str) -> int:
        """Resolve a MuJoCo geom id by name."""
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id < 0:
            raise ValueError(f"Geom '{name}' not found in grasp model.")
        return geom_id

    def _joint_id(self, name: str) -> int:
        """Resolve a MuJoCo joint id by name."""
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint '{name}' not found in grasp model.")
        return joint_id

    def _site_id(self, name: str) -> int:
        """Resolve a MuJoCo site id by name."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"Site '{name}' not found in grasp model.")
        return site_id
