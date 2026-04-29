"""Unitree H1 walking environment — MuJoCo + Gymnasium.

Adapted from the official ``unitree_rl_gym`` (legged_gym) reward structure.
Uses PD position control (action = joint angle offsets from default pose)
instead of raw torque control, matching the proven approach for humanoid
locomotion.

Control: PD position control at 50 Hz (timestep=0.002, frame_skip=10).
  action ∈ [-1, 1]^19 → target_pos = default_pos + action × action_scale
  torque = Kp × (target - current) − Kd × velocity

Observation (dim = 73):
  projected gravity       (3)   — gravity vector in pelvis frame
  base linear velocity    (3)   — pelvis lin vel in body frame
  base angular velocity   (3)   — pelvis ang vel in body frame
  velocity command        (3)   — [vx_cmd, vy_cmd, vyaw_cmd]
  joint pos − default     (19)  — joint angle offsets
  joint velocities        (19)  — joint angular velocities
  previous action         (19)  — last policy output
  gait phase              (4)   — [sin, cos] × 2 legs

Reward: 15-term structure from unitree_rl_gym H1 config.

Features:
  - Domain Randomization: friction, mass, observation noise, action delay
  - Command Randomization: random vx/vy/vyaw each episode
  - Curriculum Learning: adjustable target velocity via set_target_velocity()
  - Geom ID auto-detection from MuJoCo model body names
"""

import mujoco
import os
from collections import deque

import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_XML_PATH = os.path.join(
    _REPO_ROOT, "mujoco_menagerie", "unitree_h1", "scene.xml",
)

# ── PD gains (from unitree_rl_gym H1 config) ────────────────────────────
# Order: L_hip_yaw, L_hip_roll, L_hip_pitch, L_knee, L_ankle,
#        R_hip_yaw, R_hip_roll, R_hip_pitch, R_knee, R_ankle,
#        torso,
#        L_sh_pitch, L_sh_roll, L_sh_yaw, L_elbow,
#        R_sh_pitch, R_sh_roll, R_sh_yaw, R_elbow
_KP = np.array([
    200, 200, 200, 300, 40,     # left leg
    200, 200, 200, 300, 40,     # right leg
    300,                         # torso
    150, 150, 100, 100,          # left arm
    150, 150, 100, 100,          # right arm
], dtype=np.float64)

_KD = np.array([
    5, 5, 5, 8, 2,              # left leg
    5, 5, 5, 8, 2,              # right leg
    6,                           # torso
    2, 2, 2, 2,                  # left arm
    2, 2, 2, 2,                  # right arm
], dtype=np.float64)

_ACTION_SCALE = 0.25  # radians; policy output [-1,1] → ±0.25 rad offset

# ── Gait parameters ─────────────────────────────────────────────────────
_GAIT_FREQ = 1.5         # Hz — walking cadence
_STANCE_THRESHOLD = 0.55  # phase < this → stance; >= this → swing

# Hip yaw/roll joint indices in qpos[7:] for hip_pos penalty.
_HIP_YAW_ROLL_IDX = np.array([0, 1, 5, 6])

# ── Domain Randomization defaults ───────────────────────────────────────
_DR_FRICTION_RANGE = (0.5, 1.5)   # multiplier on baseline friction
_DR_MASS_RANGE = (0.9, 1.1)       # multiplier on baseline mass
_DR_OBS_NOISE_STD = 0.02          # Gaussian noise on observations
_DR_ACTION_DELAY_STEPS = 0        # max delay steps (0 = no delay)

# ── Command Randomization ranges ────────────────────────────────────────
_CMD_VX_RANGE = (0.3, 1.5)
_CMD_VY_RANGE = (-0.3, 0.3)
_CMD_VYAW_RANGE = (-0.3, 0.3)


def _auto_detect_geom_ids(model: mujoco.MjModel) -> dict:
    """Read geom IDs from the MuJoCo model by body name pattern.

    Returns dict with keys: floor_geom, left_foot_geoms, right_foot_geoms,
    penalised_geoms, left_ankle_body, right_ankle_body.
    """
    def _body_id(name: str) -> int:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in model")
        return bid

    def _geom_id(name: str) -> int:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid < 0:
            raise ValueError(f"Geom '{name}' not found in model")
        return gid

    def _geoms_for_body(body_name: str) -> set[int]:
        bid = _body_id(body_name)
        return {i for i in range(model.ngeom) if model.geom_bodyid[i] == bid}

    # Floor geom.
    floor_geom = _geom_id("floor")

    # Foot geoms = all geoms on ankle_link bodies.
    left_foot_geoms = frozenset(_geoms_for_body("left_ankle_link"))
    right_foot_geoms = frozenset(_geoms_for_body("right_ankle_link"))

    # Penalised geoms = hip + knee bodies (both legs).
    penalised_bodies = [
        "left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link",
        "left_knee_link",
        "right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link",
        "right_knee_link",
    ]
    penalised_geoms = frozenset()
    for name in penalised_bodies:
        penalised_geoms = penalised_geoms | frozenset(_geoms_for_body(name))

    # Ankle body IDs for swing height reward.
    left_ankle_body = _body_id("left_ankle_link")
    right_ankle_body = _body_id("right_ankle_link")

    return {
        "floor_geom": floor_geom,
        "left_foot_geoms": left_foot_geoms,
        "right_foot_geoms": right_foot_geoms,
        "penalised_geoms": penalised_geoms,
        "left_ankle_body": left_ankle_body,
        "right_ankle_body": right_ankle_body,
    }


# ── Termination thresholds ───────────────────────────────────────────────
_HEALTHY_Z_RANGE = (0.65, 1.30)
_HEALTHY_PITCH = 0.7
_HEALTHY_ROLL = 0.7

_DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array([0.0, 0.0, 1.0]),
    "elevation": -20.0,
}

# ── Reward scales (from unitree_rl_gym H1RoughCfg) ──────────────────────
_DEFAULT_REWARD_SCALES = {
    "tracking_lin_vel": 2.0,
    "tracking_ang_vel": 0.5,
    "alive": 0.5,
    "contact": 0.18,
    # Reduced: z-vel spikes during fall were overwhelming the signal.
    "lin_vel_z": -0.5,
    # Keep small: spikes during early learning overwhelm the signal.
    "ang_vel_xy": -0.01,
    "orientation": -1.0,
    # Reduced: was -10, too harsh before robot learns to stand.
    "base_height": -2.0,
    "dof_acc": -2.5e-7,
    "action_rate": -0.01,
    "collision": -1.0,
    "dof_pos_limits": -5.0,
    "hip_pos": -0.5,
    "contact_no_vel": -0.2,
    # Reduced: -20 too harsh before robot can even stand.
    "feet_swing_height": -5.0,
}
_TRACKING_SIGMA = 0.25
_BASE_HEIGHT_TARGET = 1.05
_SWING_HEIGHT_TARGET = 0.08
_SOFT_DOF_POS_LIMIT = 0.9  # fraction of joint range


def _quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion (w,x,y,z) → 3×3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-w*z),     2*(x*z+w*y)],
        [2*(x*y+w*z),     1 - 2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),     2*(y*z+w*x),     1 - 2*(x*x+y*y)],
    ])


def _quat_to_roll_pitch(quat: np.ndarray) -> tuple[float, float]:
    """MuJoCo quaternion (w,x,y,z) → (roll, pitch) in radians."""
    w, x, y, z = quat
    roll = np.arctan2(
        2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y),
    )
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return float(roll), float(pitch)


class H1Env(MujocoEnv):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        target_velocity: float = 1.0,
        reward_scales: dict | None = None,
        frame_skip: int = 10,
        domain_randomization: bool = False,
        randomize_commands: bool = False,
        **kwargs,
    ):
        self._target_velocity = target_velocity
        self._reward_scales = dict(_DEFAULT_REWARD_SCALES)
        if reward_scales:
            self._reward_scales.update(reward_scales)

        self._domain_randomization = domain_randomization
        self._randomize_commands = randomize_commands
        # Progressive DR intensity in [0, 1]. 0 = almost no randomization,
        # 1 = full configured ranges.
        self._dr_level = 1.0 if domain_randomization else 0.0

        obs_dim = 73  # see docstring
        observation_space = Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float64,
        )

        MujocoEnv.__init__(
            self,
            _XML_PATH,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=_DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # ── Auto-detect geometry IDs from model ─────────────────────
        geom_ids = _auto_detect_geom_ids(self.model)
        self._floor_geom = geom_ids["floor_geom"]
        self._left_foot_geoms = geom_ids["left_foot_geoms"]
        self._right_foot_geoms = geom_ids["right_foot_geoms"]
        self._penalised_geoms = geom_ids["penalised_geoms"]
        self._left_ankle_body = geom_ids["left_ankle_body"]
        self._right_ankle_body = geom_ids["right_ankle_body"]

        # Control limits.
        self._ctrl_range = self.model.actuator_ctrlrange.copy()

        # Default joint positions from home keyframe.
        self._default_dof_pos = self.model.key_qpos[0][7:].copy()

        # State for reward computation.
        self._prev_action = np.zeros(self.model.nu, dtype=np.float64)
        self._prev_dof_vel = np.zeros(self.model.nv - 6, dtype=np.float64)
        self._gait_phase = 0.0  # left leg phase [0, 1)
        self._episode_time = 0.0

        # Velocity command: [vx, vy, vyaw].
        self._command = np.array(
            [self._target_velocity, 0.0, 0.0], dtype=np.float64,
        )

        # Precompute soft joint limits for dof_pos_limits penalty.
        jnt_range = np.zeros((self.model.nq - 7, 2), dtype=np.float64)
        for i in range(1, self.model.njnt):  # skip free joint
            idx = i - 1
            jnt_range[idx] = self.model.jnt_range[i]
        range_span = jnt_range[:, 1] - jnt_range[:, 0]
        midpoint = (jnt_range[:, 1] + jnt_range[:, 0]) / 2.0
        half_soft = range_span * _SOFT_DOF_POS_LIMIT / 2.0
        self._soft_lower = midpoint - half_soft
        self._soft_upper = midpoint + half_soft

        # Override action space to [-1, 1].
        self.action_space = Box(
            low=-1.0, high=1.0,
            shape=(self.model.nu,), dtype=np.float32,
        )

        # ── Domain Randomization state ──────────────────────────────
        if self._domain_randomization:
            # Store baseline values for randomization.
            self._base_friction = self.model.geom_friction.copy()
            self._base_mass = self.model.body_mass.copy()
            # Action delay buffer (simulates motor latency).
            self._action_buffer: deque = deque(
                maxlen=_DR_ACTION_DELAY_STEPS + 1,
            )
            self._action_delay = 0  # actual delay, sampled each reset

    # ── Core API ──────────────────────────────────────────────────────────

    def set_target_velocity(self, velocity: float) -> None:
        """Set forward velocity command (for curriculum learning).

        Deliberately NOT clipped to _CMD_VX_RANGE: curriculum stage 1 (0.2)
        sits below the randomize-commands floor (0.3) and needs to pass
        through untouched so early-stage training focuses on balance.
        For randomize_commands, reset_model() re-clips as needed.
        """
        self._target_velocity = float(velocity)
        if not self._randomize_commands:
            self._command[0] = self._target_velocity

    def set_dr_level(self, level: float) -> None:
        """Set domain-randomization intensity in [0, 1]."""
        self._dr_level = float(np.clip(level, 0.0, 1.0))

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # Domain Randomization: action delay (simulates motor latency).
        if self._domain_randomization and self._action_delay > 0:
            self._action_buffer.append(action.copy())
            if len(self._action_buffer) > self._action_delay:
                action = self._action_buffer[0]
            else:
                action = np.zeros_like(action)

        # PD position control with per-substep gravity compensation.
        # Recompute torque each substep so gravity_ff stays accurate.
        target_pos = self._default_dof_pos + action * _ACTION_SCALE
        for _ in range(self.frame_skip):
            torque = (
                _KP * (target_pos - self.data.qpos[7:])
                - _KD * self.data.qvel[6:]
                + self.data.qfrc_bias[6:]
            )
            self.data.ctrl[:] = np.clip(
                torque, self._ctrl_range[:, 0], self._ctrl_range[:, 1],
            )
            mujoco.mj_step(self.model, self.data)
        if self.render_mode == "human":
            self.render()

        dt = self.model.opt.timestep * self.frame_skip
        self._episode_time += dt

        # Update gait phase.
        self._gait_phase = (self._gait_phase + dt * _GAIT_FREQ) % 1.0

        # ── State extraction ─────────────────────────────────────────
        qpos = self.data.qpos
        qvel = self.data.qvel
        pelvis_z = float(qpos[2])
        roll, pitch = _quat_to_roll_pitch(qpos[3:7])

        dof_pos = qpos[7:].copy()
        dof_vel = qvel[6:].copy()

        # Rotate base velocities into body frame (matches legged_gym).
        rot = _quat_to_rotation_matrix(qpos[3:7])
        base_lin_vel = rot.T @ qvel[0:3]
        base_ang_vel = rot.T @ qvel[3:6]
        gravity_world = np.array([0.0, 0.0, -1.0])
        projected_gravity = rot.T @ gravity_world

        # Foot contacts.
        left_contact, right_contact = self._detect_foot_contacts()

        # Foot velocities and positions.
        left_foot_vel = self.data.cvel[self._left_ankle_body][3:]
        right_foot_vel = self.data.cvel[self._right_ankle_body][3:]
        left_foot_z = float(self.data.xpos[self._left_ankle_body][2])
        right_foot_z = float(self.data.xpos[self._right_ankle_body][2])

        # Gait phases for each leg.
        phase_left = self._gait_phase
        phase_right = (self._gait_phase + 0.5) % 1.0

        # ── Reward computation ───────────────────────────────────────
        rewards = {}
        s = self._reward_scales

        # Positive rewards.
        rewards["tracking_lin_vel"] = self._rw_tracking_lin_vel(
            base_lin_vel,
        )
        rewards["tracking_ang_vel"] = self._rw_tracking_ang_vel(
            base_ang_vel,
        )
        # Unconditional survival reward — foundation for learning to stand.
        rewards["alive"] = 1.0
        rewards["contact"] = self._rw_contact(
            left_contact, right_contact, phase_left, phase_right,
        )

        # Penalties (raw values are positive; scales are negative).
        rewards["lin_vel_z"] = float(base_lin_vel[2] ** 2)
        rewards["ang_vel_xy"] = float(np.sum(base_ang_vel[:2] ** 2))
        rewards["orientation"] = float(
            np.sum(projected_gravity[:2] ** 2),
        )
        rewards["base_height"] = float(
            (pelvis_z - _BASE_HEIGHT_TARGET) ** 2,
        )
        rewards["dof_acc"] = self._rw_dof_acc(dof_vel, dt)
        rewards["action_rate"] = float(
            np.sum((action - self._prev_action) ** 2),
        )
        rewards["collision"] = self._rw_collision()
        rewards["dof_pos_limits"] = self._rw_dof_pos_limits(dof_pos)
        rewards["hip_pos"] = float(
            np.sum(dof_pos[_HIP_YAW_ROLL_IDX] ** 2),
        )
        rewards["contact_no_vel"] = self._rw_contact_no_vel(
            left_contact, right_contact,
            left_foot_vel, right_foot_vel,
        )
        rewards["feet_swing_height"] = self._rw_feet_swing_height(
            left_contact, right_contact,
            left_foot_z, right_foot_z,
        )

        total_reward = sum(s[k] * rewards[k] for k in rewards)

        # Termination.
        terminated = not self._is_healthy(pelvis_z, roll, pitch)

        # Update state.
        self._prev_action = action.copy()
        self._prev_dof_vel = dof_vel.copy()

        obs = self._get_obs(
            projected_gravity, base_lin_vel, base_ang_vel,
            dof_pos, dof_vel, phase_left, phase_right,
        )

        info = {
            "x_velocity": float(base_lin_vel[0]),
            "pelvis_z": pelvis_z,
            "roll": roll,
            "pitch": pitch,
            "left_contact": left_contact,
            "right_contact": right_contact,
        }
        for k, v in rewards.items():
            info[f"reward_{k}"] = s[k] * v

        return obs, float(total_reward), terminated, False, info

    def reset_model(self):
        """Reset to home keyframe with small noise."""
        # ── Domain Randomization: physics parameters ────────────────
        if self._domain_randomization:
            # Friction: interpolate around 1.0 as curriculum progresses.
            fric_lo = 1.0 - (1.0 - _DR_FRICTION_RANGE[0]) * self._dr_level
            fric_hi = 1.0 + (_DR_FRICTION_RANGE[1] - 1.0) * self._dr_level
            fric_mult = self.np_random.uniform(fric_lo, fric_hi)
            self.model.geom_friction[:] = self._base_friction * fric_mult

            # Mass: per-body uniform multiplier with progressive range.
            mass_lo = 1.0 - (1.0 - _DR_MASS_RANGE[0]) * self._dr_level
            mass_hi = 1.0 + (_DR_MASS_RANGE[1] - 1.0) * self._dr_level
            mass_mult = self.np_random.uniform(
                mass_lo, mass_hi, size=self.model.nbody,
            )
            self.model.body_mass[:] = self._base_mass * mass_mult

            # Action delay: random integer in [0, max].
            self._action_delay = self.np_random.integers(
                0, _DR_ACTION_DELAY_STEPS + 1,
            )
            self._action_buffer.clear()

        # ── Command Randomization ───────────────────────────────────
        if self._randomize_commands:
            # Respect curriculum: allow vx_hi below the usual _CMD_VX_RANGE[0]
            # floor (e.g. 0.2 m/s in stage 1 for balance-focused training).
            # Only the upper bound is enforced; a tiny lower bound (0.05)
            # avoids sampling exactly zero.
            vx_hi = float(
                min(max(self._target_velocity, 0.05), _CMD_VX_RANGE[1]),
            )
            vx_lo = 0.4 * vx_hi
            self._command[0] = float(
                self.np_random.uniform(vx_lo, vx_hi),
            )
            self._command[1] = float(
                self.np_random.uniform(*_CMD_VY_RANGE),
            )
            self._command[2] = float(
                self.np_random.uniform(*_CMD_VYAW_RANGE),
            )
        else:
            self._command[:] = [self._target_velocity, 0.0, 0.0]

        # ── Standard reset ──────────────────────────────────────────
        self.data.qpos[:] = self.model.key_qpos[0]
        self.data.qvel[:] = 0.0

        noise = 0.01
        nq_joints = self.model.nq - 7
        nv_joints = self.model.nv - 6
        self.data.qpos[7:] += self.np_random.uniform(
            -noise, noise, size=nq_joints,
        )
        self.data.qvel[6:] += self.np_random.uniform(
            -noise, noise, size=nv_joints,
        )

        self._prev_action = np.zeros(self.model.nu, dtype=np.float64)
        self._prev_dof_vel = np.zeros(nv_joints, dtype=np.float64)
        self._gait_phase = 0.0
        self._episode_time = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Rotate base velocities + gravity into body frame so reset-step obs
        # matches the convention used in step() (see lines 350-355). Without
        # this, the first obs of each episode is in world frame and creates
        # a coordinate mismatch that destabilises training.
        rot = _quat_to_rotation_matrix(self.data.qpos[3:7])
        projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0])
        base_lin_vel = rot.T @ self.data.qvel[0:3]
        base_ang_vel = rot.T @ self.data.qvel[3:6]
        phase_left = self._gait_phase
        phase_right = (self._gait_phase + 0.5) % 1.0
        return self._get_obs(
            projected_gravity,
            base_lin_vel,
            base_ang_vel,
            self.data.qpos[7:].copy(),
            self.data.qvel[6:].copy(),
            phase_left, phase_right,
        )

    # ── Observation ──────────────────────────────────────────────────────

    def _get_obs(
        self,
        projected_gravity: np.ndarray,
        base_lin_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        phase_left: float,
        phase_right: float,
    ) -> np.ndarray:
        dof_pos_offset = dof_pos - self._default_dof_pos
        gait_obs = np.array([
            np.sin(2 * np.pi * phase_left),
            np.cos(2 * np.pi * phase_left),
            np.sin(2 * np.pi * phase_right),
            np.cos(2 * np.pi * phase_right),
        ])
        obs = np.concatenate([
            projected_gravity,           # 3
            base_lin_vel,                # 3
            base_ang_vel,                # 3
            self._command,               # 3
            dof_pos_offset,              # 19
            dof_vel,                     # 19
            self._prev_action,           # 19
            gait_obs,                    # 4
        ]).astype(np.float64)            # total: 73

        # Domain Randomization: observation noise.
        if self._domain_randomization:
            obs_noise = _DR_OBS_NOISE_STD * self._dr_level
            obs += self.np_random.normal(0.0, obs_noise, size=obs.shape)

        return obs

    # ── Reward functions ─────────────────────────────────────────────────
    # Each returns a raw (unsigned) value. The sign comes from the scale.

    def _rw_tracking_lin_vel(
        self, base_lin_vel: np.ndarray,
    ) -> float:
        error_sq = np.sum(
            (self._command[:2] - base_lin_vel[:2]) ** 2,
        )
        return float(np.exp(-error_sq / _TRACKING_SIGMA))

    def _rw_tracking_ang_vel(
        self, base_ang_vel: np.ndarray,
    ) -> float:
        error_sq = (self._command[2] - base_ang_vel[2]) ** 2
        return float(np.exp(-error_sq / _TRACKING_SIGMA))

    def _rw_contact(
        self,
        left_contact: bool,
        right_contact: bool,
        phase_left: float,
        phase_right: float,
    ) -> float:
        """Foot contact should match gait phase."""
        score = 0.0
        left_should_stance = phase_left < _STANCE_THRESHOLD
        right_should_stance = phase_right < _STANCE_THRESHOLD
        if left_contact == left_should_stance:
            score += 1.0
        if right_contact == right_should_stance:
            score += 1.0
        return score

    def _rw_dof_acc(
        self, dof_vel: np.ndarray, dt: float,
    ) -> float:
        acc = (dof_vel - self._prev_dof_vel) / dt
        return float(np.sum(acc ** 2))

    def _rw_collision(self) -> float:
        """Count collisions on hip/knee bodies."""
        count = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 in self._penalised_geoms:
                count += 1.0
            if c.geom2 in self._penalised_geoms:
                count += 1.0
        return count

    def _rw_dof_pos_limits(self, dof_pos: np.ndarray) -> float:
        """Penalise joints approaching their limits."""
        below = np.clip(self._soft_lower - dof_pos, 0.0, None)
        above = np.clip(dof_pos - self._soft_upper, 0.0, None)
        return float(np.sum(below + above))

    def _rw_contact_no_vel(
        self,
        left_contact: bool,
        right_contact: bool,
        left_foot_vel: np.ndarray,
        right_foot_vel: np.ndarray,
    ) -> float:
        """Feet should not move when in contact."""
        cost = 0.0
        if left_contact:
            cost += float(np.sum(left_foot_vel[:3] ** 2))
        if right_contact:
            cost += float(np.sum(right_foot_vel[:3] ** 2))
        return cost

    def _rw_feet_swing_height(
        self,
        left_contact: bool,
        right_contact: bool,
        left_foot_z: float,
        right_foot_z: float,
    ) -> float:
        """Swing feet should reach target clearance height."""
        cost = 0.0
        if not left_contact:
            cost += (left_foot_z - _SWING_HEIGHT_TARGET) ** 2
        if not right_contact:
            cost += (right_foot_z - _SWING_HEIGHT_TARGET) ** 2
        return cost

    # ── Helpers ──────────────────────────────────────────────────────────

    def _is_healthy(
        self, z: float, roll: float, pitch: float,
    ) -> bool:
        z_min, z_max = _HEALTHY_Z_RANGE
        return (
            z_min <= z <= z_max
            and abs(roll) <= _HEALTHY_ROLL
            and abs(pitch) <= _HEALTHY_PITCH
        )

    def _detect_foot_contacts(self) -> tuple[bool, bool]:
        """Check if left/right foot geoms touch the floor."""
        left = False
        right = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 != self._floor_geom and g2 != self._floor_geom:
                continue
            other = g1 if g2 == self._floor_geom else g2
            if other in self._left_foot_geoms:
                left = True
            elif other in self._right_foot_geoms:
                right = True
            if left and right:
                break
        return left, right
