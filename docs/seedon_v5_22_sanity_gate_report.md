# Seedon v5_22 Mechanical Sanity Gate Report

Task class: Class C mechanical sanity diagnostic. This report does not claim walking success and does not run PPO.

## Summary

- Gate decision: `PARTIAL_PASS`
- Source path: `private_assets/seedon_v5_22/training_scene.xml`
- Source type: `converted_mjcf`
- Appears v5_22: `True`
- Bodies / joints / geoms / actuators: `12` / `11` / `15` / `10`

## Model Source Resolution

- `source_path`: `private_assets/seedon_v5_22/training_scene.xml`
- `source_type`: `converted_mjcf`
- `appears_v5_22`: `True`
- `warnings`: `[]`

## Model Load

- `success`: `True`
- `counts`: `{'bodies': 12, 'joints': 11, 'actuators': 10, 'geoms': 15, 'sensors': 0, 'keyframes': 0}`

## Expected Structure

- Expected joints present: `True`
- Expected 10 actuated leg joints present: `True`
- Missing joints: `[]`
- Missing geoms: `[]`
- Right-leg mapping warnings: `[]`

## Reset Pose

- `base_height`: `0.62`
- `base_orientation_rpy`: `{'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}`
- `joint_qpos_within_limits`: `True`
- `joint_limit_violations`: `[]`
- `contact_count`: `0`
- `obvious_contact_explosion`: `False`
- `foot_geoms_detected`: `['R_foot_collision', 'L_foot_collision']`

## Zero-Action Settle

- `phase`: `zero_action`
- `steps_run`: `250`
- `initial_base_height`: `0.62`
- `final_base_height`: `0.14535268549582558`
- `base_height_drift`: `-0.4746473145041744`
- `roll_drift`: `3.113669842591438`
- `pitch_drift`: `-0.30111089834107274`
- `yaw_drift`: `3.1146445733753927`
- `final_roll`: `3.113669842591438`
- `final_pitch`: `-0.30111089834107274`
- `final_yaw`: `3.1146445733753927`
- `contact_count_total`: `725`
- `fall_or_large_tilt`: `True`
- `unstable_or_exploding`: `True`
- `exploded`: `False`
- `contact_none_rate`: `0.492`

## Nominal PD Hold

- `phase`: `nominal_pd_hold`
- `steps_run`: `250`
- `initial_base_height`: `0.62`
- `final_base_height`: `0.4339800472243704`
- `base_height_drift`: `-0.1860199527756296`
- `roll_drift`: `0.00012464447770891448`
- `pitch_drift`: `-0.4451052142637733`
- `yaw_drift`: `-0.00017042560380311792`
- `final_roll`: `0.00012464447770891448`
- `final_pitch`: `-0.4451052142637733`
- `final_yaw`: `-0.00017042560380311792`
- `contact_count_total`: `946`
- `fall_or_large_tilt`: `False`
- `unstable_or_exploding`: `False`
- `exploded`: `False`
- `pd_hold`: `{'kp': 25.0, 'kd': 1.5, 'source': 'assumption', 'confidence': 'low', 'valid_for': 'sanity_gate_only'}`
- `joint_target_clamp_rate`: `0.0`
- `actuator_saturation_rate`: `0.0`

## Foot Contact Observability

- `foot_contact_pair_count`: `1671`
- `foot_contact_pairs_observed`: `['floor::L_foot_collision', 'floor::R_foot_collision']`
- `center_toe_heel_classification_possible`: `False`
- `contact_model_observable`: `True`
- `classification_note`: `Geom names are insufficient for center/toe/heel classification.`

## Actuator / Motor Missing Fields

- `ctrlrange`: `FOUND`
- `forcerange`: `manual_required`
- `gear`: `manual_required`
- `kp`: `manual_required`
- `kd`: `manual_required`
- `rated_torque`: `manual_required`
- `peak_torque`: `manual_required`
- `max_velocity`: `manual_required`
- `control_mode`: `manual_required`
- `current_limit`: `manual_required`
- `encoder_resolution`: `manual_required`
- `backlash`: `manual_required`

## Recommendation

- Can proceed to controller / motor gap closure: `True`
- Keep all missing motor/controller values as `manual_required` until external specs are provided.
