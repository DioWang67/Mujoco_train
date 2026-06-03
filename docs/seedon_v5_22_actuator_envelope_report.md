# Seedon v5_22 Actuator Envelope Report

Task class: Class C actuator-envelope diagnostic. This report does not claim walking success and does not convert team-provided torque into verified joint forcerange.

## Summary

- Status: `PARTIAL_ACTUATOR_ENVELOPE`
- Valid for: `bounded_diagnostic_only`
- Invalid for: `['sim2real_claim', 'walking_success_claim', 'verified_joint_forcerange_claim']`
- Torque side: `unknown_motor_side_or_joint_output`
- Can proceed to foot x actuator/controller sensitivity: `True`

## Source Inputs

- `model_path`: `private_assets/seedon_v5_22/training_scene.xml`
- `source_urdf`: `private_assets/SEEDON_URDF_5_22/urdf/SEEDON_URDF_5_21.urdf`
- `controller_source`: `seedon_baseline/env.py`
- `torque_data`: `{'source': 'provided_by_team_message', 'confidence': 'medium', 'torque_side': 'unknown_motor_side_or_joint_output', 'groups': {'hip_pitch': {'rated_nm': 20.0, 'peak_nm': 60.0}, 'ankle_pitch': {'rated_nm': 5.0, 'peak_nm': 14.0}, 'other_leg_joints': {'rated_nm': 6.0, 'peak_nm': 17.0}}}`

## Actuator Mapping Table

| idx | actuator | joint | group | ctrlrange | forcerange | gear status |
|---:|---|---|---|---|---|---|
| 0 | `R_joint_hip_yaw_motor` | `R_joint_hip_yaw` | `other_leg_joints` | `[-100.0, 100.0]` | `None` | `manual_required` |
| 1 | `R_joint_hip_roll_motor` | `R_joint_hip_roll` | `other_leg_joints` | `[-100.0, 100.0]` | `None` | `manual_required` |
| 2 | `R_joint_hip_pitch_motor` | `R_joint_hip_pitch` | `hip_pitch` | `[-100.0, 100.0]` | `None` | `manual_required` |
| 3 | `R_joint_knee_pitch_motor` | `R_joint_knee_pitch` | `other_leg_joints` | `[-100.0, 100.0]` | `None` | `manual_required` |
| 4 | `R_joint_ankle_pitch_motor` | `R_joint_ankle_pitch` | `ankle_pitch` | `[-100.0, 100.0]` | `None` | `manual_required` |
| 5 | `L_joint_hip_yaw_motor` | `L_joint_hip_yaw` | `other_leg_joints` | `[-100.0, 100.0]` | `None` | `manual_required` |
| 6 | `L_joint_hip_roll_motor` | `L_joint_hip_roll` | `other_leg_joints` | `[-100.0, 100.0]` | `None` | `manual_required` |
| 7 | `L_joint_hip_pitch_motor` | `L_joint_hip_pitch` | `hip_pitch` | `[-100.0, 100.0]` | `None` | `manual_required` |
| 8 | `L_joint_knee_pitch_motor` | `L_joint_knee_pitch` | `other_leg_joints` | `[-100.0, 100.0]` | `None` | `manual_required` |
| 9 | `L_joint_ankle_pitch_motor` | `L_joint_ankle_pitch` | `ankle_pitch` | `[-100.0, 100.0]` | `None` | `manual_required` |

## Torque Envelope Table

| joint | group | rated Nm | peak Nm | torque side | source | confidence | verified forcerange? |
|---|---|---:|---:|---|---|---|---|
| `R_joint_hip_yaw` | `other_leg_joints` | 6.0 | 17.0 | `unknown_motor_side_or_joint_output` | `provided_by_team_message` | `medium` | `False` |
| `R_joint_hip_roll` | `other_leg_joints` | 6.0 | 17.0 | `unknown_motor_side_or_joint_output` | `provided_by_team_message` | `medium` | `False` |
| `R_joint_hip_pitch` | `hip_pitch` | 20.0 | 60.0 | `unknown_motor_side_or_joint_output` | `provided_by_team_message` | `medium` | `False` |
| `R_joint_knee_pitch` | `other_leg_joints` | 6.0 | 17.0 | `unknown_motor_side_or_joint_output` | `provided_by_team_message` | `medium` | `False` |
| `R_joint_ankle_pitch` | `ankle_pitch` | 5.0 | 14.0 | `unknown_motor_side_or_joint_output` | `provided_by_team_message` | `medium` | `False` |
| `L_joint_hip_yaw` | `other_leg_joints` | 6.0 | 17.0 | `unknown_motor_side_or_joint_output` | `provided_by_team_message` | `medium` | `False` |
| `L_joint_hip_roll` | `other_leg_joints` | 6.0 | 17.0 | `unknown_motor_side_or_joint_output` | `provided_by_team_message` | `medium` | `False` |
| `L_joint_hip_pitch` | `hip_pitch` | 20.0 | 60.0 | `unknown_motor_side_or_joint_output` | `provided_by_team_message` | `medium` | `False` |
| `L_joint_knee_pitch` | `other_leg_joints` | 6.0 | 17.0 | `unknown_motor_side_or_joint_output` | `provided_by_team_message` | `medium` | `False` |
| `L_joint_ankle_pitch` | `ankle_pitch` | 5.0 | 14.0 | `unknown_motor_side_or_joint_output` | `provided_by_team_message` | `medium` | `False` |

## Controller Semantics

- `source_file`: `seedon_baseline/env.py`
- `_do_pd_simulation`: `{'line': 2232, 'text': 'def _do_pd_simulation(self, target_positions: np.ndarray) -> None:'}`
- `_ctrl_range`: `{'line': 1193, 'text': 'self._ctrl_range = self.model.actuator_ctrlrange.copy()'}`
- `np_clip_saturation`: `{'line': 1724, 'text': 'return np.clip(scaled_ctrl, self._ctrl_range[:, 0], self._ctrl_range[:, 1])'}`
- `action_joint_delta_scale`: `{'line': 216, 'text': 'action_joint_delta_scale: float = 0.08'}`
- `pd_stiffness`: `{'line': 260, 'text': 'pd_stiffness: float = 35.0'}`
- `pd_damping`: `{'line': 261, 'text': 'pd_damping: float = 2.0'}`
- `safe_target_clamp`: `{'line': 2368, 'text': 'def _apply_safe_joint_target_clamps(self, target_positions: np.ndarray) -> np.ndarray:'}`
- `status`: `FOUND`

## Known / Partial / Manual Required Fields

- Known: `['actuator_count', 'actuator_order', 'actuator_name', 'joint_mapping', 'ctrlrange', 'joint_axis', 'joint_range']`
- Partial: `['urdf_effort_velocity', 'rated_torque', 'peak_torque', 'python_pd_controller_semantics']`
- `forcerange`: `manual_required`
- `verified_gear_ratio`: `manual_required`
- `confirmed_kp`: `manual_required`
- `confirmed_kd`: `manual_required`
- `max_velocity`: `manual_required`
- `control_mode`: `manual_required`
- `current_limit`: `manual_required`
- `encoder_resolution`: `manual_required`
- `backlash`: `manual_required`
- `rated_torque`: `PARTIAL_TEAM_PROVIDED_TORQUE_SIDE_UNKNOWN`
- `peak_torque`: `PARTIAL_TEAM_PROVIDED_TORQUE_SIDE_UNKNOWN`

## Why Ankle Pitch Is High Risk

- Ankle pitch has the lowest provided rated torque: `5 Nm`.
- Ankle pitch is directly involved in toe handoff / rollover authority.
- Current foot contact geoms cannot classify center/toe/heel, so ankle authority sensitivity must stay bounded.
- Torque side is unknown, so values cannot be treated as verified joint-output torque.

## What Can Be Used For Bounded Diagnostic

- `rated_safe` profile for conservative bounded diagnostic.
- `peak_upper_bound` profile for short-burst upper-bound diagnostic only.
- `ankle_risk_sweep` values: `5, 10, 14, 20 Nm`.
- MJCF actuator mapping and ctrlrange can be used as simulation metadata, not verified motor spec.

## What Must Not Be Claimed

- Do not claim walking success.
- Do not claim sim2real validity.
- Do not claim provided torque is verified joint forcerange.
- Do not claim continuous gait is safe under `peak_upper_bound`.

## Next Step Recommendation

- Proceed: `True`
- Mode: `bounded_diagnostic_only`
- Reason: v5_22 mechanical sanity gate is PARTIAL_PASS and torque data is available, but torque side and controller/motor specs remain unresolved.
