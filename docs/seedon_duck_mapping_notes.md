# Seedon / Open Duck Mini Semantic Mapping Notes

This file documents the first semantic joint mapping layer between Seedon and Open Duck Mini v2. It is reference data for Duck-like dynamic gait analysis only; it is not a direct controller, action, reward, or training configuration.

## Inputs

| File | Purpose |
|---|---|
| `configs/seedon/seedon_robot_parameters.yaml` | Extracted Seedon MJCF parameter snapshot. |
| `references/open_duck_mini/duck_robot_parameters.yaml` | Extracted Open Duck Mini v2 MJCF parameter snapshot. |
| `references/open_duck_mini/seedon_duck_joint_mapping.yaml` | Semantic leg-joint alignment. |

## Mapping Summary

| Semantic joint | Seedon joint | Duck joint | Status |
|---|---|---|---|
| right hip yaw | `R_joint_hip_yaw` | `right_hip_yaw` | mapped, sign validation required |
| right hip roll | `R_joint_hip_roll` | `right_hip_roll` | mapped, sign validation required |
| right hip pitch | `R_joint_hip_pitch` | `right_hip_pitch` | mapped, sign validation required |
| right knee pitch | `R_joint_knee_pitch` | `right_knee` | mapped, sign validation required |
| right ankle pitch | `R_joint_ankle_pitch` | `right_ankle` | mapped, sign validation required |
| left hip yaw | `L_joint_hip_yaw` | `left_hip_yaw` | mapped, sign validation required |
| left hip roll | `L_joint_hip_roll` | `left_hip_roll` | mapped, sign validation required |
| left hip pitch | `L_joint_hip_pitch` | `left_hip_pitch` | mapped, sign validation required |
| left knee pitch | `L_joint_knee_pitch` | `left_knee` | mapped, sign validation required |
| left ankle pitch | `L_joint_ankle_pitch` | `left_ankle` | mapped, sign validation required |

## Important Differences

- Seedon source MJCF exposes `motor` actuators with `ctrlrange=[-100, 100]`; explicit `kp` and actuator `forcerange` are not present.
- Duck source MJCF exposes `position` actuators using default class `sts3215`, with `kp=13.37` and `forcerange=[-3.23, 3.23]`.
- Seedon joint axes are explicit. Duck joint axes are not explicit in the selected XML, so MuJoCo defaults or compiled model behavior must be validated before using sign-sensitive references.
- Seedon foot collision is explicit box geometry; Duck foot contact is mesh-based and selected by name heuristic.
- Seedon explicit mass is about `10.0454`; Duck explicit mass is about `2.1071`. Mass and scale differences mean Duck gait amplitude cannot be copied directly.

## Transfer Rules

- Use Duck gait period, support timing, and qualitative low-clearance style as references.
- Do not copy Duck joint positions directly into Seedon.
- Do not copy Duck actuator gains or force ranges into Seedon.
- Validate joint sign and approximate amplitude on a scripted, non-training smoke test before any reward/config change.
- Keep generated comparisons and mappings separate from training configs until a specific experiment is approved.

## Next Validation Step

Build a read-only sign/amplitude validation tool that loads the mapping and compares small positive/negative perturbations for each mapped joint in both models. The output should be a mapping confidence report, not a training change.
