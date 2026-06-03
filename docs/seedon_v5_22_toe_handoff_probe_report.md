# Seedon v5_22 Toe Handoff Targeted Probe Report

Task class: Class C bounded diagnostic. This report does not run PPO, does not claim walking success, and does not create verified motor specs.

## Summary

- Status: `TOE_HANDOFF_PROBE_COMPLETE`
- Probe rows: `36`
- Toe handoff candidate found: `False`
- Candidate count: `0`
- Best row: `duck_like_multi_patch / ankle_toe_down_bias / ankle_boost_hypothesis`
- Next recommendation: foot geometry tuning first; targeted posture probes did not expose center-to-toe handoff.

## Probe Setup

- Foot variants: `['duck_like_multi_patch', 'simple_toe_center_heel']`
- Actuator profiles: `['ankle_boost_hypothesis', 'peak_upper_bound', 'rated_safe']`
- Posture cases: `['ankle_toe_down_bias', 'forward_com_shift', 'medium_forward_base_pitch', 'mild_knee_hip_flexion_forward_lean', 'neutral_stance', 'small_forward_base_pitch']`
- Prototype thresholds: `{'source': 'prototype_threshold', 'confidence': 'low', 'toe_force_ratio_min': 0.45, 'center_force_ratio_max': 0.35, 'heel_force_ratio_max': 0.3, 'contact_none_rate_max': 0.25, 'large_tilt_degrees': 35.0, 'max_base_height_drift': 0.25}`
- Method limitation: MuJoCo contact force is read from raw contact normal force via `mj_contactForce`; patch attribution depends on prototype geom names.

## Best Toe Handoff Candidates

- No toe handoff candidate met the prototype thresholds.

## Force Ratio Table

| foot variant | profile | posture | toe ratio | center ratio | heel ratio | transfer score | observable score | result |
|---|---|---|---:|---:|---:|---:|---:|---|
| `simple_toe_center_heel` | `rated_safe` | `neutral_stance` | 0.143627 | 0.322892 | 0.53348 | -0.179265 | 0.0670049 | `insufficient_contact_persistence` |
| `simple_toe_center_heel` | `rated_safe` | `small_forward_base_pitch` | 0.134186 | 0.401317 | 0.464497 | -0.267131 | 0.0718569 | `insufficient_contact_persistence` |
| `simple_toe_center_heel` | `rated_safe` | `medium_forward_base_pitch` | 0.203018 | 0.332743 | 0.464239 | -0.129725 | 0.108769 | `insufficient_contact_persistence` |
| `simple_toe_center_heel` | `rated_safe` | `ankle_toe_down_bias` | 0.52407 | 0.273196 | 0.202734 | 0.250874 | 0.417823 | `insufficient_contact_persistence` |
| `simple_toe_center_heel` | `rated_safe` | `forward_com_shift` | 0.151289 | 0.368139 | 0.480572 | -0.216849 | 0.078584 | `insufficient_contact_persistence` |
| `simple_toe_center_heel` | `rated_safe` | `mild_knee_hip_flexion_forward_lean` | 0.255665 | 0.347188 | 0.397147 | -0.0915228 | 0.154128 | `insufficient_contact_persistence` |
| `simple_toe_center_heel` | `peak_upper_bound` | `neutral_stance` | 0.143627 | 0.322892 | 0.53348 | -0.179265 | 0.0670049 | `insufficient_contact_persistence` |
| `simple_toe_center_heel` | `peak_upper_bound` | `small_forward_base_pitch` | 0.266566 | 0.387356 | 0.346079 | -0.12079 | 0.174313 | `insufficient_contact_persistence` |
| `simple_toe_center_heel` | `peak_upper_bound` | `medium_forward_base_pitch` | 0.41192 | 0.317047 | 0.271033 | 0.0948732 | 0.300276 | `insufficient_contact_persistence` |
| `simple_toe_center_heel` | `peak_upper_bound` | `ankle_toe_down_bias` | 0.748188 | 0.25175 | 6.13461e-05 | 0.496438 | 0.748143 | `insufficient_contact_persistence` |
| `simple_toe_center_heel` | `peak_upper_bound` | `forward_com_shift` | 0.364738 | 0.351526 | 0.283737 | 0.0132124 | 0.261248 | `insufficient_contact_persistence` |
| `simple_toe_center_heel` | `peak_upper_bound` | `mild_knee_hip_flexion_forward_lean` | 0.495598 | 0.387513 | 0.11689 | 0.108085 | 0.437667 | `insufficient_contact_persistence` |
| `simple_toe_center_heel` | `ankle_boost_hypothesis` | `neutral_stance` | 0.143627 | 0.322892 | 0.53348 | -0.179265 | 0.0670049 | `insufficient_contact_persistence` |
| `simple_toe_center_heel` | `ankle_boost_hypothesis` | `small_forward_base_pitch` | 0.266566 | 0.387356 | 0.346079 | -0.12079 | 0.174313 | `insufficient_contact_persistence` |
| `simple_toe_center_heel` | `ankle_boost_hypothesis` | `medium_forward_base_pitch` | 0.470014 | 0.322629 | 0.207357 | 0.147386 | 0.372554 | `insufficient_contact_persistence` |
| `simple_toe_center_heel` | `ankle_boost_hypothesis` | `ankle_toe_down_bias` | 0.883346 | 0.116654 | 0 | 0.766691 | 0.883346 | `insufficient_contact_persistence` |
| `simple_toe_center_heel` | `ankle_boost_hypothesis` | `forward_com_shift` | 0.383377 | 0.349977 | 0.266646 | 0.0333995 | 0.281151 | `insufficient_contact_persistence` |
| `simple_toe_center_heel` | `ankle_boost_hypothesis` | `mild_knee_hip_flexion_forward_lean` | 0.584607 | 0.373931 | 0.0414622 | 0.210675 | 0.560368 | `insufficient_contact_persistence` |
| `duck_like_multi_patch` | `rated_safe` | `neutral_stance` | 0.0795299 | 0.301214 | 0.619256 | -0.221684 | 0.0302806 | `insufficient_contact_persistence` |
| `duck_like_multi_patch` | `rated_safe` | `small_forward_base_pitch` | 0.0747524 | 0.374557 | 0.550691 | -0.299805 | 0.033587 | `insufficient_contact_persistence` |
| `duck_like_multi_patch` | `rated_safe` | `medium_forward_base_pitch` | 0.291112 | 0.247552 | 0.461336 | 0.0435604 | 0.156812 | `insufficient_contact_persistence` |
| `duck_like_multi_patch` | `rated_safe` | `ankle_toe_down_bias` | 0.621987 | 0.241973 | 0.13604 | 0.380014 | 0.537372 | `insufficient_contact_persistence` |
| `duck_like_multi_patch` | `rated_safe` | `forward_com_shift` | 0.244961 | 0.313853 | 0.441187 | -0.0688917 | 0.136887 | `insufficient_contact_persistence` |
| `duck_like_multi_patch` | `rated_safe` | `mild_knee_hip_flexion_forward_lean` | 0.499153 | 0.244998 | 0.255849 | 0.254155 | 0.371445 | `insufficient_contact_persistence` |
| `duck_like_multi_patch` | `peak_upper_bound` | `neutral_stance` | 0.0795299 | 0.301214 | 0.619256 | -0.221684 | 0.0302806 | `insufficient_contact_persistence` |
| `duck_like_multi_patch` | `peak_upper_bound` | `small_forward_base_pitch` | 0.254915 | 0.371128 | 0.373957 | -0.116213 | 0.159588 | `insufficient_contact_persistence` |
| `duck_like_multi_patch` | `peak_upper_bound` | `medium_forward_base_pitch` | 0.51096 | 0.241135 | 0.247905 | 0.269825 | 0.38429 | `insufficient_contact_persistence` |
| `duck_like_multi_patch` | `peak_upper_bound` | `ankle_toe_down_bias` | 0.840853 | 0.159147 | 0 | 0.681707 | 0.840853 | `insufficient_contact_persistence` |
| `duck_like_multi_patch` | `peak_upper_bound` | `forward_com_shift` | 0.462808 | 0.314347 | 0.222845 | 0.148461 | 0.359674 | `insufficient_contact_persistence` |
| `duck_like_multi_patch` | `peak_upper_bound` | `mild_knee_hip_flexion_forward_lean` | 0.722909 | 0.212679 | 0.064412 | 0.51023 | 0.676345 | `insufficient_contact_persistence` |
| `duck_like_multi_patch` | `ankle_boost_hypothesis` | `neutral_stance` | 0.0795299 | 0.301214 | 0.619256 | -0.221684 | 0.0302806 | `insufficient_contact_persistence` |
| `duck_like_multi_patch` | `ankle_boost_hypothesis` | `small_forward_base_pitch` | 0.257405 | 0.370965 | 0.37163 | -0.11356 | 0.161746 | `insufficient_contact_persistence` |
| `duck_like_multi_patch` | `ankle_boost_hypothesis` | `medium_forward_base_pitch` | 0.594347 | 0.241554 | 0.164099 | 0.352794 | 0.496816 | `insufficient_contact_persistence` |
| `duck_like_multi_patch` | `ankle_boost_hypothesis` | `ankle_toe_down_bias` | 0.928829 | 0.0711706 | 0 | 0.857659 | 0.928829 | `insufficient_contact_persistence` |
| `duck_like_multi_patch` | `ankle_boost_hypothesis` | `forward_com_shift` | 0.495575 | 0.311366 | 0.193059 | 0.18421 | 0.3999 | `insufficient_contact_persistence` |
| `duck_like_multi_patch` | `ankle_boost_hypothesis` | `mild_knee_hip_flexion_forward_lean` | 0.769953 | 0.210779 | 0.0192682 | 0.559174 | 0.755117 | `insufficient_contact_persistence` |

## Whether Ankle Boost Changes Force Transfer

- Changes force transfer versus rated_safe: `True`

## Whether Foot Variant Changes Force Transfer

- Changes force transfer between variants: `True`

## Whether Toe Handoff Is Physically Observable

- Toe handoff is not observable under the current controlled posture sweep and prototype thresholds.

## If Not Observable, Recommended Foot Geometry Tuning

- Increase toe patch ability to carry load without simultaneously loading center/heel.
- Sweep toe patch height and x-offset before changing gait rewards.
- Keep center/toe/heel patch names explicit so diagnostics remain observable.

## What Must Not Be Claimed

- Do not claim walking success.
- Do not claim sim2real validity.
- Do not treat provided torque as verified joint-output forcerange.
- Do not treat `peak_upper_bound` as a continuous gait claim.

## Next Recommendation

- foot geometry tuning first; targeted posture probes did not expose center-to-toe handoff.
