"""Minimal MuJoCo standing environment for the private Sedon robot model."""

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


@dataclass(frozen=True)
class SedonStandingConfig:
    """Reward and termination settings for the Sedon standing task.

    Args:
        target_base_height: Desired base height in meters.
        min_base_height: Episode terminates below this height.
        max_base_height: Episode terminates above this height.
        min_upright: Episode terminates below this base upright alignment.
        torque_scale: Normalized action multiplier before clipping to motor limits.
        alive_reward: Reward granted each non-terminal step.
        height_weight: Weight for matching target height.
        height_sharpness: Exponential penalty sharpness for base-height error.
        upright_weight: Weight for keeping the base z-axis upright.
        pose_weight: Weight for keeping actuated joints near the nominal stance.
        pose_sharpness: Exponential penalty sharpness for joint pose error.
        action_penalty_weight: Penalty coefficient for squared normalized action.
        action_rate_penalty_weight: Penalty coefficient for changing actions too abruptly.
        velocity_penalty_weight: Penalty coefficient for joint velocity.
        base_xy_velocity_penalty_weight: Penalty coefficient for horizontal drift speed.
        base_roll_pitch_rate_penalty_weight: Penalty coefficient for roll/pitch angular speed.
        max_base_xy_drift: Episode terminates if the base drifts farther than this radius.
    """

    target_base_height: float = 0.46
    min_base_height: float = 0.34
    max_base_height: float = 0.65
    min_upright: float = 0.75
    torque_scale: float = 45.0
    alive_reward: float = 0.2
    height_weight: float = 3.0
    height_sharpness: float = 40.0
    upright_weight: float = 2.0
    pose_weight: float = 2.5
    pose_sharpness: float = 8.0
    action_penalty_weight: float = 0.005
    action_rate_penalty_weight: float = 0.02
    velocity_penalty_weight: float = 0.003
    base_xy_velocity_penalty_weight: float = 1.0
    base_roll_pitch_rate_penalty_weight: float = 0.1
    max_base_xy_drift: float = 0.08


def compute_standing_reward(
    base_height: float,
    upright: float,
    joint_velocity_l2: float,
    action_l2: float,
    action_rate_l2: float,
    joint_position_error_l2: float,
    base_xy_velocity_l2: float,
    base_roll_pitch_rate_l2: float,
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
        base_xy_velocity_l2: Squared norm of horizontal base velocity.
        base_roll_pitch_rate_l2: Squared norm of base roll/pitch angular rate.
        config: Reward coefficients.

    Returns:
        Reward component mapping including ``total``.
    """
    height_error = base_height - config.target_base_height
    height = float(np.exp(-config.height_sharpness * height_error * height_error))
    upright_clipped = float(np.clip(upright, -1.0, 1.0))
    pose = float(np.exp(-config.pose_sharpness * joint_position_error_l2))
    components = {
        "alive": config.alive_reward,
        "height": height,
        "upright": max(0.0, upright_clipped),
        "pose": pose,
        "action_penalty": action_l2,
        "action_rate_penalty": action_rate_l2,
        "velocity_penalty": joint_velocity_l2,
        "base_xy_velocity_penalty": base_xy_velocity_l2,
        "base_roll_pitch_rate_penalty": base_roll_pitch_rate_l2,
    }
    total = components["alive"]
    total += config.height_weight * components["height"]
    total += config.upright_weight * components["upright"]
    total += config.pose_weight * components["pose"]
    total -= config.action_penalty_weight * components["action_penalty"]
    total -= config.action_rate_penalty_weight * components["action_rate_penalty"]
    total -= config.velocity_penalty_weight * components["velocity_penalty"]
    total -= config.base_xy_velocity_penalty_weight * components["base_xy_velocity_penalty"]
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

        observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(41,),
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
        scaled_ctrl = clipped_action * self._reward_config.torque_scale
        ctrl_low = self._ctrl_range[:, 0]
        ctrl_high = self._ctrl_range[:, 1]
        ctrl = np.clip(scaled_ctrl, ctrl_low, ctrl_high)

        self.do_simulation(ctrl, self.frame_skip)

        obs = self._get_obs()
        base_height = self._base_height()
        upright = self._base_upright()
        joint_positions = self._joint_positions()
        joint_velocities = self._joint_velocities()
        joint_velocity_l2 = float(np.dot(joint_velocities, joint_velocities))
        action_l2 = float(np.dot(clipped_action, clipped_action))
        action_delta = clipped_action - self._prev_action
        action_rate_l2 = float(np.dot(action_delta, action_delta))
        joint_position_error = joint_positions - self._nominal_joint_qpos
        joint_position_error_l2 = float(np.dot(joint_position_error, joint_position_error))
        base_xy_velocity = self.data.qvel[0:2]
        base_xy_velocity_l2 = float(np.dot(base_xy_velocity, base_xy_velocity))
        base_roll_pitch_rate = self.data.qvel[3:5]
        base_roll_pitch_rate_l2 = float(np.dot(base_roll_pitch_rate, base_roll_pitch_rate))
        rewards = compute_standing_reward(
            base_height=base_height,
            upright=upright,
            joint_velocity_l2=joint_velocity_l2,
            action_l2=action_l2,
            action_rate_l2=action_rate_l2,
            joint_position_error_l2=joint_position_error_l2,
            base_xy_velocity_l2=base_xy_velocity_l2,
            base_roll_pitch_rate_l2=base_roll_pitch_rate_l2,
            config=self._reward_config,
        )
        terminated = self._is_terminated(base_height, upright, obs)
        if terminated:
            rewards["total"] -= 2.0

        info = {
            "base_height": base_height,
            "upright": upright,
            "joint_velocity_l2": joint_velocity_l2,
            "action_l2": action_l2,
            "action_rate_l2": action_rate_l2,
            "joint_position_error_l2": joint_position_error_l2,
            "base_xy_velocity_l2": base_xy_velocity_l2,
            "base_roll_pitch_rate_l2": base_roll_pitch_rate_l2,
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

    def _body_id(self, name: str) -> int:
        """Resolve a MuJoCo body id by name."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body '{name}' not found in Sedon model.")
        return body_id

    def _joint_id(self, name: str) -> int:
        """Resolve a MuJoCo joint id by name."""
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint '{name}' not found in Sedon model.")
        return joint_id
