# Sedon URDF / Controller Source Scan

Task class: Class C read-only source inventory. This scan does not rerun extraction/readiness/foot prototype pipelines, does not run simulation, and does not modify training or evaluation code.

## Summary

- Overall: `PARTIAL`
- URDF/Xacro: `FOUND`
- Controller YAML/config: `NOT_FOUND`
- Motor spec: `NOT_FOUND`
- Python/MuJoCo control semantics: `FOUND`
- Actuator envelope: `PARTIAL`
- Note: Joint effort/velocity limits are present, but this is only a partial actuator envelope; it is not a complete motor model without gear/reduction/control mode/PID or motor spec.

## URDF/Xacro Findings

### `private_assets/sedon/mjcf_source/sedon.urdf`

- robot name: `urdf`
- joints: `10`
- inertial links: `11`
- collisions: `11`
- transmissions: `0`

| joint | line | axis | lower | upper | effort | velocity |
|---|---:|---|---:|---:|---:|---:|
| `R_joint_hip_yaw` | 88 | `0 0 1` | `-0.175` | `0.175` | `100` | `1` |
| `R_joint_hip_roll` | 146 | `1 0 0` | `-0.2625` | `0.2625` | `100` | `1` |
| `R_joint_hip_pitch` | 204 | `0 1 0` | `-1.05` | `1.575` | `100` | `1` |
| `R_joint_knee_pitch` | 262 | `0 1 0` | `-1.575` | `1.575` | `100` | `1` |
| `R_joint_ankle_pitch` | 320 | `0 1 0` | `-1.05` | `1.575` | `100` | `1` |
| `L_joint_hip_yaw` | 378 | `0 0 1` | `-0.175` | `0.175` | `100` | `1` |
| `L_joint_hip_roll` | 436 | `1 0 0` | `-0.2625` | `0.2625` | `100` | `1` |
| `L_joint_hip_pitch` | 494 | `0 1 0` | `-1.05` | `1.575` | `100` | `1` |
| `L_joint_knee_pitch` | 552 | `0 1 0` | `-1.575` | `1.575` | `100` | `1` |
| `L_joint_ankle_pitch` | 610 | `0 1 0` | `-1.05` | `1.575` | `100` | `1` |

- transmission / mechanicalReduction / hardwareInterface: `NOT_FOUND`

### `private_assets/sedon/original_urdf_package/urdf/urdf/urdf.urdf`

- robot name: `urdf`
- joints: `10`
- inertial links: `11`
- collisions: `11`
- transmissions: `0`

| joint | line | axis | lower | upper | effort | velocity |
|---|---:|---|---:|---:|---:|---:|
| `R_joint_hip_yaw` | 88 | `0 0 1` | `-0.175` | `0.175` | `100` | `1` |
| `R_joint_hip_roll` | 146 | `1 0 0` | `-0.2625` | `0.2625` | `100` | `1` |
| `R_joint_hip_pitch` | 204 | `0 1 0` | `-1.05` | `1.575` | `100` | `1` |
| `R_joint_knee_pitch` | 262 | `0 1 0` | `-1.575` | `1.575` | `100` | `1` |
| `R_joint_ankle_pitch` | 320 | `0 1 0` | `-1.05` | `1.575` | `100` | `1` |
| `L_joint_hip_yaw` | 378 | `0 0 1` | `-0.175` | `0.175` | `100` | `1` |
| `L_joint_hip_roll` | 436 | `1 0 0` | `-0.2625` | `0.2625` | `100` | `1` |
| `L_joint_hip_pitch` | 494 | `0 1 0` | `-1.05` | `1.575` | `100` | `1` |
| `L_joint_knee_pitch` | 552 | `0 1 0` | `-1.575` | `1.575` | `100` | `1` |
| `L_joint_ankle_pitch` | 610 | `0 1 0` | `-1.05` | `1.575` | `100` | `1` |

- transmission / mechanicalReduction / hardwareInterface: `NOT_FOUND`

### `private_assets/SEEDON_URDF_5_22_转动惯量改平/urdf/SEEDON_URDF_5_21.urdf`

- robot name: `SEEDON_URDF_5_21`
- joints: `10`
- inertial links: `11`
- collisions: `11`
- transmissions: `0`

| joint | line | axis | lower | upper | effort | velocity |
|---|---:|---|---:|---:|---:|---:|
| `L_joint_hip_yaw` | 88 | `0 0 1` | `-0.175` | `0.175` | `300` | `1` |
| `L_joint_hip_roll` | 146 | `1 0 0` | `-0.2625` | `0.2625` | `300` | `1` |
| `L_joint_hip_pitch` | 204 | `0 1 0` | `-1.05` | `1.575` | `300` | `1` |
| `L_joint_knee_pitch` | 262 | `0 1 0` | `-1.575` | `1.575` | `300` | `1` |
| `L_joint_ankle_pitch` | 320 | `0 1 0` | `-1.05` | `1.575` | `300` | `1` |
| `R_joint_hip_yaw` | 378 | `0 0 1` | `-0.175` | `0.175` | `0` | `0` |
| `R_joint_hip_roll` | 436 | `1 0 0` | `-0.2625` | `0.2625` | `300` | `1` |
| `R_joint_hip_pitch` | 494 | `0 1 0` | `-1.05` | `1.575` | `300` | `1` |
| `R_joint_knee` | 552 | `0 1 0` | `-1.575` | `1.575` | `300` | `1` |
| `R_joint_knee_pitch` | 610 | `0 1 0` | `-1.05` | `1.575` | `300` | `1` |

- transmission / mechanicalReduction / hardwareInterface: `NOT_FOUND`

## Controller Config Findings

NOT FOUND: no ROS controller YAML/config source was found in the scan scope.

## Python/MuJoCo Control Findings

| file | line | field | confidence | value |
|---|---:|---|---|---|
| `sedon_baseline/env.py` | 2264 | `data.ctrl` | `medium` | `self.data.ctrl[:] = ctrl` |
| `sedon_baseline/env.py` | 2429 | `data.ctrl` | `medium` | `if not hasattr(self.data, "ctrl") or self.data.ctrl.size == 0:` |
| `sedon_baseline/env.py` | 2431 | `data.ctrl` | `medium` | `lower_margin = np.abs(self.data.ctrl - self._ctrl_range[:, 0])` |
| `sedon_baseline/env.py` | 2432 | `data.ctrl` | `medium` | `upper_margin = np.abs(self.data.ctrl - self._ctrl_range[:, 1])` |
| `sedon_baseline/env.py` | 2435 | `data.ctrl` | `medium` | `return float(np.count_nonzero(saturated) / self.data.ctrl.size)` |
| `sedon_baseline/env.py` | 1238 | `_do_pd_simulation` | `medium` | `self._do_pd_simulation(target_positions)` |
| `sedon_baseline/env.py` | 2232 | `_do_pd_simulation` | `medium` | `def _do_pd_simulation(self, target_positions: np.ndarray) -> None:` |
| `sedon_baseline/env.py` | 2236 | `_do_pd_simulation` | `medium` | `def _do_pd_simulation_with_torque_assist(` |
| `sedon_baseline/env.py` | 1193 | `_ctrl_range` | `medium` | `self._ctrl_range = self.model.actuator_ctrlrange.copy()` |
| `sedon_baseline/env.py` | 1724 | `_ctrl_range` | `medium` | `return np.clip(scaled_ctrl, self._ctrl_range[:, 0], self._ctrl_range[:, 1])` |
| `sedon_baseline/env.py` | 2282 | `_ctrl_range` | `medium` | `self._ctrl_range[:, 0],` |
| `sedon_baseline/env.py` | 2283 | `_ctrl_range` | `medium` | `self._ctrl_range[:, 1],` |
| `sedon_baseline/env.py` | 2431 | `_ctrl_range` | `medium` | `lower_margin = np.abs(self.data.ctrl - self._ctrl_range[:, 0])` |
| `sedon_baseline/env.py` | 2432 | `_ctrl_range` | `medium` | `upper_margin = np.abs(self.data.ctrl - self._ctrl_range[:, 1])` |
| `sedon_baseline/env.py` | 2433 | `_ctrl_range` | `medium` | `ctrl_span = np.maximum(self._ctrl_range[:, 1] - self._ctrl_range[:, 0], 1e-9)` |
| `sedon_baseline/env.py` | 1193 | `ctrlrange` | `medium` | `self._ctrl_range = self.model.actuator_ctrlrange.copy()` |
| `sedon_baseline/env.py` | 483 | `np.clip` | `medium` | `np.clip(forward_velocity / config.target_forward_velocity, 0.0, 1.0)` |
| `sedon_baseline/env.py` | 671 | `np.clip` | `medium` | `np.clip(1.0 - current_abs_y_error / max(target_magnitude, 1e-9), 0.0, 1.0)` |
| `sedon_baseline/env.py` | 893 | `np.clip` | `medium` | `float(np.clip(forward_velocity / max(target_forward_velocity, 1e-9), 0.0, 1.0))` |
| `sedon_baseline/env.py` | 1218 | `np.clip` | `medium` | `clipped_action = np.clip(action_array, -1.0, 1.0).astype(np.float64)` |
| `sedon_baseline/env.py` | 1719 | `np.clip` | `medium` | `scaled_ctrl = np.clip(` |
| `sedon_baseline/env.py` | 1724 | `np.clip` | `medium` | `return np.clip(scaled_ctrl, self._ctrl_range[:, 0], self._ctrl_range[:, 1])` |
| `sedon_baseline/env.py` | 2280 | `np.clip` | `medium` | `assisted_ctrl = np.clip(` |
| `sedon_baseline/env.py` | 2379 | `np.clip` | `medium` | `clamped[joint_index] = float(np.clip(clamped[joint_index], *allowed_range))` |
| `sedon_baseline/env.py` | 85 | `action_joint_delta_scale` | `medium` | `action_joint_delta_scale: Maximum joint target offset represented by action 1.0.` |
| `sedon_baseline/env.py` | 216 | `action_joint_delta_scale` | `medium` | `action_joint_delta_scale: float = 0.08` |
| `sedon_baseline/env.py` | 1232 | `action_joint_delta_scale` | `medium` | `scaled_residual = residual_action * self._reward_config.action_joint_delta_scale` |
| `sedon_baseline/env.py` | 1198 | `nominal_joint_qpos` | `medium` | `self._nominal_joint_qpos = (` |
| `sedon_baseline/env.py` | 1230 | `nominal_joint_qpos` | `medium` | `gait_target = self._nominal_joint_qpos + self._gait_joint_offsets()` |
| `sedon_baseline/env.py` | 1528 | `nominal_joint_qpos` | `medium` | `qpos[qpos_adr] = self._nominal_joint_qpos[joint_index]` |
| `sedon_baseline/env.py` | 1907 | `nominal_joint_qpos` | `medium` | `return scaled_target - self._nominal_joint_qpos` |
| `sedon_baseline/env.py` | 321 | `torque_saturation` | `medium` | `march_torque_saturation_penalty_weight: float = 2.0` |
| `sedon_baseline/env.py` | 829 | `torque_saturation` | `medium` | `torque_saturation: float,` |
| `sedon_baseline/env.py` | 866 | `torque_saturation` | `medium` | `torque_saturation: Fraction of actuators at their control limit.` |
| `sedon_baseline/env.py` | 959 | `torque_saturation` | `medium` | `"torque_saturation_penalty": torque_saturation,` |
| `sedon_baseline/env.py` | 1072 | `torque_saturation` | `medium` | `config.march_torque_saturation_penalty_weight` |
| `sedon_baseline/env.py` | 1073 | `torque_saturation` | `medium` | `* components["torque_saturation_penalty"]` |
| `sedon_baseline/env.py` | 1299 | `torque_saturation` | `medium` | `torque_saturation = self._torque_saturation_fraction()` |
| `sedon_baseline/env.py` | 1389 | `torque_saturation` | `medium` | `torque_saturation=torque_saturation,` |
| `sedon_baseline/env.py` | 1504 | `torque_saturation` | `medium` | `"torque_saturation": torque_saturation,` |
| `sedon_baseline/env.py` | 2427 | `torque_saturation` | `medium` | `def _torque_saturation_fraction(self) -> float:` |
| `sedon_baseline/env.py` | 321 | `saturation` | `medium` | `march_torque_saturation_penalty_weight: float = 2.0` |
| `sedon_baseline/env.py` | 829 | `saturation` | `medium` | `torque_saturation: float,` |
| `sedon_baseline/env.py` | 866 | `saturation` | `medium` | `torque_saturation: Fraction of actuators at their control limit.` |
| `sedon_baseline/env.py` | 959 | `saturation` | `medium` | `"torque_saturation_penalty": torque_saturation,` |
| `sedon_baseline/env.py` | 1072 | `saturation` | `medium` | `config.march_torque_saturation_penalty_weight` |
| `sedon_baseline/env.py` | 1073 | `saturation` | `medium` | `* components["torque_saturation_penalty"]` |
| `sedon_baseline/env.py` | 1299 | `saturation` | `medium` | `torque_saturation = self._torque_saturation_fraction()` |
| `sedon_baseline/env.py` | 1389 | `saturation` | `medium` | `torque_saturation=torque_saturation,` |
| `sedon_baseline/env.py` | 1504 | `saturation` | `medium` | `"torque_saturation": torque_saturation,` |
| `sedon_baseline/env.py` | 2427 | `saturation` | `medium` | `def _torque_saturation_fraction(self) -> float:` |
| `sedon_baseline/env.py` | 130 | `stiffness` | `medium` | `pd_stiffness: Joint-space proportional gain for stance tracking.` |
| `sedon_baseline/env.py` | 131 | `damping` | `medium` | `pd_damping: Joint-space velocity damping gain for stance tracking.` |
| `sedon_baseline/env.py` | 260 | `stiffness` | `medium` | `pd_stiffness: float = 35.0` |
| `sedon_baseline/env.py` | 261 | `damping` | `medium` | `pd_damping: float = 2.0` |
| `sedon_baseline/env.py` | 1237 | `_apply_safe_joint_target_clamps` | `medium` | `target_positions = self._apply_safe_joint_target_clamps(target_positions)` |
| `sedon_baseline/env.py` | 1716 | `stiffness` | `medium` | `self._reward_config.pd_stiffness * (target_positions - self._joint_positions())` |
| `sedon_baseline/env.py` | 1717 | `damping` | `medium` | `- self._reward_config.pd_damping * self._joint_velocities()` |
| `sedon_baseline/env.py` | 2368 | `_apply_safe_joint_target_clamps` | `medium` | `def _apply_safe_joint_target_clamps(self, target_positions: np.ndarray) -> np.ndarray:` |
| `private_assets/sedon/training_scene.xml` | 80 | `ctrlrange` | `medium` | `<motor name="R_joint_hip_yaw_motor" joint="R_joint_hip_yaw" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene.xml` | 81 | `ctrlrange` | `medium` | `<motor name="R_joint_hip_roll_motor" joint="R_joint_hip_roll" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene.xml` | 82 | `ctrlrange` | `medium` | `<motor name="R_joint_hip_pitch_motor" joint="R_joint_hip_pitch" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene.xml` | 83 | `ctrlrange` | `medium` | `<motor name="R_joint_knee_pitch_motor" joint="R_joint_knee_pitch" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene.xml` | 84 | `ctrlrange` | `medium` | `<motor name="R_joint_ankle_pitch_motor" joint="R_joint_ankle_pitch" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene.xml` | 85 | `ctrlrange` | `medium` | `<motor name="L_joint_hip_yaw_motor" joint="L_joint_hip_yaw" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene.xml` | 86 | `ctrlrange` | `medium` | `<motor name="L_joint_hip_roll_motor" joint="L_joint_hip_roll" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene.xml` | 87 | `ctrlrange` | `medium` | `<motor name="L_joint_hip_pitch_motor" joint="L_joint_hip_pitch" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene.xml` | 88 | `ctrlrange` | `medium` | `<motor name="L_joint_knee_pitch_motor" joint="L_joint_knee_pitch" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene.xml` | 89 | `ctrlrange` | `medium` | `<motor name="L_joint_ankle_pitch_motor" joint="L_joint_ankle_pitch" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene_v5_a.xml` | 80 | `ctrlrange` | `medium` | `<motor name="R_joint_hip_yaw_motor" joint="R_joint_hip_yaw" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene_v5_a.xml` | 81 | `ctrlrange` | `medium` | `<motor name="R_joint_hip_roll_motor" joint="R_joint_hip_roll" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene_v5_a.xml` | 82 | `ctrlrange` | `medium` | `<motor name="R_joint_hip_pitch_motor" joint="R_joint_hip_pitch" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene_v5_a.xml` | 83 | `ctrlrange` | `medium` | `<motor name="R_joint_knee_pitch_motor" joint="R_joint_knee_pitch" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene_v5_a.xml` | 84 | `ctrlrange` | `medium` | `<motor name="R_joint_ankle_pitch_motor" joint="R_joint_ankle_pitch" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene_v5_a.xml` | 85 | `ctrlrange` | `medium` | `<motor name="L_joint_hip_yaw_motor" joint="L_joint_hip_yaw" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene_v5_a.xml` | 86 | `ctrlrange` | `medium` | `<motor name="L_joint_hip_roll_motor" joint="L_joint_hip_roll" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene_v5_a.xml` | 87 | `ctrlrange` | `medium` | `<motor name="L_joint_hip_pitch_motor" joint="L_joint_hip_pitch" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene_v5_a.xml` | 88 | `ctrlrange` | `medium` | `<motor name="L_joint_knee_pitch_motor" joint="L_joint_knee_pitch" ctrlrange="-100 100" ctrllimited="true" />` |
| `private_assets/sedon/training_scene_v5_a.xml` | 89 | `ctrlrange` | `medium` | `<motor name="L_joint_ankle_pitch_motor" joint="L_joint_ankle_pitch" ctrlrange="-100 100" ctrllimited="true" />` |
| `sedon_baseline/tests/test_env.py` | 1169 | `data.ctrl` | `medium` | `observed_ctrl.append(np.array(data.ctrl, copy=True))` |
| `sedon_baseline/tests/test_env.py` | 1182 | `data.ctrl` | `medium` | `np.testing.assert_allclose(env.data.ctrl, expected_ctrl)` |
| `sedon_baseline/tests/test_env.py` | 1184 | `data.ctrl` | `medium` | `assert env.data.ctrl[0] == pytest.approx(base_ctrl[0])` |
| `sedon_baseline/tests/test_env.py` | 1208 | `data.ctrl` | `medium` | `np.testing.assert_allclose(env.data.ctrl, base_ctrl)` |
| `sedon_baseline/tests/test_env.py` | 1145 | `_do_pd_simulation` | `medium` | `def test_do_pd_simulation_with_torque_assist_clamps_and_records_actual_delta(` |
| `sedon_baseline/tests/test_env.py` | 1173 | `_do_pd_simulation` | `medium` | `env._do_pd_simulation_with_torque_assist(` |
| `sedon_baseline/tests/test_env.py` | 1187 | `_do_pd_simulation` | `medium` | `def test_do_pd_simulation_without_assist_keeps_zero_injected_delta(` |
| `sedon_baseline/tests/test_env.py` | 1206 | `_do_pd_simulation` | `medium` | `env._do_pd_simulation(np.zeros(10, dtype=np.float64))` |
| `sedon_baseline/tests/test_env.py` | 1152 | `_ctrl_range` | `medium` | `env._ctrl_range = np.tile(np.array([[-1.0, 1.0]], dtype=np.float64), (10, 1))` |
| `sedon_baseline/tests/test_env.py` | 1153 | `_ctrl_range` | `medium` | `env._ctrl_range[RIGHT_HIP_ROLL_ACTUATOR_INDEX] = np.array([-0.45, 0.45], dtype=np.float64)` |
| `sedon_baseline/tests/test_env.py` | 1154 | `_ctrl_range` | `medium` | `env._ctrl_range[LEFT_HIP_ROLL_ACTUATOR_INDEX] = np.array([-0.5, 0.5], dtype=np.float64)` |
| `sedon_baseline/tests/test_env.py` | 1194 | `_ctrl_range` | `medium` | `env._ctrl_range = np.tile(np.array([[-1.0, 1.0]], dtype=np.float64), (10, 1))` |
| `sedon_baseline/tests/test_env.py` | 663 | `nominal_joint_qpos` | `medium` | `env._nominal_joint_qpos = np.zeros(len(JOINT_NAMES), dtype=np.float64)` |
| `sedon_baseline/tests/test_env.py` | 689 | `nominal_joint_qpos` | `medium` | `env._nominal_joint_qpos = np.zeros(len(JOINT_NAMES), dtype=np.float64)` |
| `sedon_baseline/tests/test_env.py` | 538 | `torque_saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 564 | `torque_saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 729 | `torque_saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 755 | `torque_saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 794 | `torque_saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 820 | `torque_saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 859 | `torque_saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 887 | `torque_saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 929 | `torque_saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 955 | `torque_saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 996 | `torque_saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 1024 | `torque_saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 538 | `saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 564 | `saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 729 | `saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 755 | `saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 794 | `saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 820 | `saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 859 | `saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 887 | `saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 929 | `saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 955 | `saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 996 | `saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 1024 | `saturation` | `medium` | `torque_saturation=0.0,` |
| `sedon_baseline/tests/test_env.py` | 382 | `_apply_safe_joint_target_clamps` | `medium` | `clamped = env._apply_safe_joint_target_clamps(unclamped)` |
| `configs/sedon/sedon_actuator_envelope.yaml` | 16 | `ctrlrange` | `medium` | `"explicit_ctrlrange_count": 10,` |
| `configs/sedon/sedon_actuator_envelope.yaml` | 21 | `ctrlrange` | `medium` | `"reason": "Sedon actuator ctrlrange is explicit, but kp/forcerange/gear and control semantics require clarification."` |

## What This Resolves

- `joint_axis`: `FOUND`
- `joint_range`: `FOUND`
- `joint_effort`: `PARTIAL`
- `joint_velocity`: `PARTIAL`
- `inertial_mass_com_inertia`: `FOUND`
- `collision_geometry`: `FOUND`
- `python_control_path`: `FOUND`

## Still Missing

- `motor_max_torque`: `manual_required`
- `motor_max_velocity`: `manual_required`
- `gear_ratio`: `manual_required`
- `control_mode`: `manual_required`
- `pid_gains`: `manual_required`
- `encoder_imu_sensor_data`: `manual_required`
- `external_sedon_mechanical_or_motor_spec_path`: `not_found_user_path_required`

## Recommendation

- Foot x actuator sensitivity: `PARTIAL_READY_FOR_BOUNDED_DIAGNOSTIC_ONLY`
- Blocked for full actuator model: `True`
- Needs user external paths: `True`
- Reason: URDF joint limits and Python/MuJoCo control path are available, but motor model, gear ratio, PID gains, control mode, and sensor specs remain manual_required.
