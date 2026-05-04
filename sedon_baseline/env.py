"""Minimal MuJoCo locomotion environment for the private Sedon robot model."""

from __future__ import annotations

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
FOOT_GEOM_NAMES = ("R_foot_collision", "L_foot_collision")


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
        gait_cycle_steps: Number of RL steps in one built-in walking gait cycle.
        gait_hip_pitch_amp: Hip pitch amplitude for the built-in gait target.
        gait_knee_pitch_amp: Knee pitch amplitude for the built-in gait target.
        gait_ankle_pitch_amp: Ankle pitch amplitude for the built-in gait target.
        pd_stiffness: Joint-space proportional gain for stance tracking.
        pd_damping: Joint-space velocity damping gain for stance tracking.
        alive_reward: Reward granted each non-terminal step.
        height_weight: Weight for matching target height.
        height_sharpness: Exponential penalty sharpness for base-height error.
        upright_weight: Weight for keeping the base z-axis upright.
        forward_velocity_weight: Weight for tracking target forward speed.
        min_rewarded_forward_velocity: Forward speed below which stable policies are penalized.
        low_forward_velocity_penalty_weight: Penalty for standing still while stable.
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
    """

    target_base_height: float = 0.446
    target_forward_velocity: float = 0.12
    min_base_height: float = 0.34
    max_base_height: float = 0.65
    min_upright: float = 0.75
    termination_penalty: float = 3000.0
    torque_scale: float = 45.0
    action_joint_delta_scale: float = 0.25
    gait_cycle_steps: int = 60
    gait_hip_pitch_amp: float = -0.06
    gait_knee_pitch_amp: float = 0.09
    gait_ankle_pitch_amp: float = 0.045
    pd_stiffness: float = 35.0
    pd_damping: float = 2.0
    alive_reward: float = 0.0
    height_weight: float = 1.0
    height_sharpness: float = 40.0
    upright_weight: float = 1.0
    forward_velocity_weight: float = 5.0
    min_rewarded_forward_velocity: float = 0.05
    low_forward_velocity_penalty_weight: float = 4.0
    forward_overspeed_limit: float = 0.15
    forward_overspeed_penalty_weight: float = 120.0
    backward_velocity_penalty_weight: float = 5.0
    lateral_velocity_penalty_weight: float = 3.0
    pose_weight: float = 0.6
    pose_sharpness: float = 8.0
    action_penalty_weight: float = 0.01
    action_rate_penalty_weight: float = 0.08
    velocity_penalty_weight: float = 0.003
    base_xy_velocity_penalty_weight: float = 0.0
    base_roll_pitch_rate_penalty_weight: float = 0.1
    foot_flat_weight: float = 0.4
    foot_height_penalty_weight: float = 8.0
    foot_air_penalty_weight: float = 0.05
    max_base_xy_drift: float = 2.0


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
    overspeed = max(0.0, forward_velocity - config.forward_overspeed_limit)
    pose = float(np.exp(-config.pose_sharpness * joint_position_error_l2))
    foot_flatness_clipped = float(np.clip(foot_flatness, 0.0, 1.0))
    missing_feet = max(0, len(FOOT_GEOM_NAMES) - feet_near_floor)
    components = {
        "alive": config.alive_reward,
        "height": height,
        "upright": max(0.0, upright_clipped),
        "forward_velocity": forward_progress * stability_gate,
        "stability_gate": stability_gate,
        "low_forward_velocity_penalty": low_forward_shortfall
        * low_forward_shortfall
        * stability_gate,
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

        self._scene_path = Path(scene_path)
        if not self._scene_path.is_file():
            raise FileNotFoundError(
                f"Sedon training scene not found: {self._scene_path}. "
                "Run `python -m tools.convert_urdf_to_mjcf` and then "
                "`python -m tools.build_sedon_training_scene` first."
            )
        if reset_noise_scale < 0.0:
            raise ValueError("reset_noise_scale must be non-negative.")

        self._reward_config = reward_config or SedonStandingConfig()
        self._reset_noise_scale = reset_noise_scale
        self._prev_action = np.zeros(len(JOINT_NAMES), dtype=np.float64)
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
        self._ctrl_range = self.model.actuator_ctrlrange.copy()
        self._default_qpos = self.init_qpos.copy()
        self._default_qvel = self.init_qvel.copy()
        self._nominal_joint_qpos = self._extract_joint_positions(self._default_qpos)
        self._set_base_pose(self._default_qpos)

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
        gait_target = self._nominal_joint_qpos + self._gait_joint_offsets()
        target_positions = (
            gait_target
            + clipped_action * self._reward_config.action_joint_delta_scale
        )
        self._do_pd_simulation(target_positions)
        self._gait_step += 1

        obs = self._get_obs()
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
        )
        terminated = self._is_terminated(base_height, upright, obs)
        if terminated:
            rewards["total"] -= self._reward_config.termination_penalty

        info = {
            "base_height": base_height,
            "base_x_position": float(self.data.qpos[0]),
            "upright": upright,
            "joint_velocity_l2": joint_velocity_l2,
            "action_l2": action_l2,
            "action_rate_l2": action_rate_l2,
            "joint_position_error_l2": joint_position_error_l2,
            "forward_velocity": forward_velocity,
            "lateral_velocity_l2": lateral_velocity_l2,
            "base_xy_velocity_l2": base_xy_velocity_l2,
            "base_roll_pitch_rate_l2": base_roll_pitch_rate_l2,
            "gait_phase": self._gait_phase(),
            "foot_flatness": foot_flatness,
            "foot_height_error_l2": foot_height_error_l2,
            "feet_near_floor": feet_near_floor,
        }
        for key, value in rewards.items():
            info[f"reward_{key}"] = value

        self._prev_action = clipped_action.copy()
        return obs, float(rewards["total"]), terminated, False, info

    def reset_model(self) -> np.ndarray:
        """Reset the floating base and actuated joints to a standing seed pose."""
        qpos = self._default_qpos.copy()
        qvel = self._default_qvel.copy()
        self._set_base_pose(qpos)

        for joint_id in self._joint_ids:
            qpos_adr = self.model.jnt_qposadr[joint_id]
            if self._reset_noise_scale > 0.0:
                qpos[qpos_adr] += self.np_random.uniform(
                    -self._reset_noise_scale,
                    self._reset_noise_scale,
                )

        qvel[:] = 0.0
        self.set_state(qpos, qvel)
        self._prev_action = np.zeros(len(JOINT_NAMES), dtype=np.float64)
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
        cycle_steps = max(1, self._reward_config.gait_cycle_steps)
        return float(2.0 * np.pi * (self._gait_step % cycle_steps) / cycle_steps)

    def _gait_joint_offsets(self) -> np.ndarray:
        """Return a small periodic joint target that seeds forward stepping."""
        phase = self._gait_phase()
        swing = np.sin(phase)
        right_swing = max(0.0, swing)
        left_swing = max(0.0, -swing)
        offsets = np.zeros(len(JOINT_NAMES), dtype=np.float64)
        offsets[2] = self._reward_config.gait_hip_pitch_amp * swing
        offsets[3] = self._reward_config.gait_knee_pitch_amp * right_swing
        offsets[4] = self._reward_config.gait_ankle_pitch_amp * right_swing
        offsets[7] = -self._reward_config.gait_hip_pitch_amp * swing
        offsets[8] = self._reward_config.gait_knee_pitch_amp * left_swing
        offsets[9] = self._reward_config.gait_ankle_pitch_amp * left_swing
        return offsets

    def _do_pd_simulation(self, target_positions: np.ndarray) -> None:
        """Step MuJoCo while refreshing the stance PD torque every physics step."""
        for _ in range(self.frame_skip):
            self.data.ctrl[:] = self._pd_control(target_positions)
            mujoco.mj_step(self.model, self.data)

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
