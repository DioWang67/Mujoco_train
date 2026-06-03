# Seedon v5_22 Foot x Actuator Sensitivity Report

Task class: Class C bounded diagnostic. This report does not run PPO, does not claim walking success, and does not create verified motor specs.

## Summary

- Status: `BOUNDED_SENSITIVITY_COMPLETE`
- Matrix rows: `12`
- Valid for: `bounded_diagnostic_only`
- Invalid for: `['walking_success_claim', 'sim2real_claim', 'verified_motor_spec_claim', 'continuous_gait_claim']`
- Rated safe enough for bounded diagnostics: `True`
- Ankle boost changes toe handoff or bridge: `False`
- Ankle boost reduces ankle saturation: `True`
- Next blocker: verified actuator/controller specs and foot geometry that avoids toe/heel bridge while preserving toe handoff observability

## Matrix Setup

- Foot variants: `['current_box_baseline', 'duck_like_multi_patch', 'simple_toe_center_heel']`
- Actuator profiles: `['ankle_boost_hypothesis', 'mid_burst', 'peak_upper_bound', 'rated_safe']`
- All profiles remain diagnostic only; torque side is `unknown_motor_side_or_joint_output`.
- `ankle_boost_hypothesis` is `source=assumption`, `confidence=low`.

## Result Table

| foot variant | profile | classify C/T/H | bridge | toe handoff L/R | fall/tilt | tilt max | ankle sat | result |
|---|---|---:|---:|---|---:|---:|---:|---|
| `current_box_baseline` | `rated_safe` | False | unavailable | unavailable/unavailable | False | 0.228776 | 0.258333 | `insufficient_contact_observability` |
| `current_box_baseline` | `mid_burst` | False | unavailable | unavailable/unavailable | False | 0.228776 | 0.141667 | `insufficient_contact_observability` |
| `current_box_baseline` | `peak_upper_bound` | False | unavailable | unavailable/unavailable | False | 0.228776 | 0.0916667 | `insufficient_contact_observability` |
| `current_box_baseline` | `ankle_boost_hypothesis` | False | unavailable | unavailable/unavailable | False | 0.228776 | 0.0416667 | `insufficient_contact_observability` |
| `simple_toe_center_heel` | `rated_safe` | True | False | False/False | False | 0.216953 | 0.25 | `inconclusive_bounded_diagnostic` |
| `simple_toe_center_heel` | `mid_burst` | True | False | False/False | False | 0.216953 | 0.125 | `inconclusive_bounded_diagnostic` |
| `simple_toe_center_heel` | `peak_upper_bound` | True | False | False/False | False | 0.216953 | 0.0583333 | `inconclusive_bounded_diagnostic` |
| `simple_toe_center_heel` | `ankle_boost_hypothesis` | True | False | False/False | False | 0.216953 | 0 | `inconclusive_bounded_diagnostic` |
| `duck_like_multi_patch` | `rated_safe` | True | False | False/False | False | 0.216815 | 0.25 | `inconclusive_bounded_diagnostic` |
| `duck_like_multi_patch` | `mid_burst` | True | False | False/False | False | 0.216815 | 0.125 | `inconclusive_bounded_diagnostic` |
| `duck_like_multi_patch` | `peak_upper_bound` | True | False | False/False | False | 0.216815 | 0.0666667 | `inconclusive_bounded_diagnostic` |
| `duck_like_multi_patch` | `ankle_boost_hypothesis` | True | False | False/False | False | 0.216815 | 0 | `inconclusive_bounded_diagnostic` |

## Bridge Contact Sensitivity

- `current_box_baseline`: bridge contact = `unavailable`
- `simple_toe_center_heel`: bridge contact = `False`
- `duck_like_multi_patch`: bridge contact = `False`

## Toe Handoff Sensitivity

- `current_box_baseline::rated_safe`: left=`unavailable`, right=`unavailable`
- `current_box_baseline::mid_burst`: left=`unavailable`, right=`unavailable`
- `current_box_baseline::peak_upper_bound`: left=`unavailable`, right=`unavailable`
- `current_box_baseline::ankle_boost_hypothesis`: left=`unavailable`, right=`unavailable`
- `simple_toe_center_heel::rated_safe`: left=`False`, right=`False`
- `simple_toe_center_heel::mid_burst`: left=`False`, right=`False`
- `simple_toe_center_heel::peak_upper_bound`: left=`False`, right=`False`
- `simple_toe_center_heel::ankle_boost_hypothesis`: left=`False`, right=`False`
- `duck_like_multi_patch::rated_safe`: left=`False`, right=`False`
- `duck_like_multi_patch::mid_burst`: left=`False`, right=`False`
- `duck_like_multi_patch::peak_upper_bound`: left=`False`, right=`False`
- `duck_like_multi_patch::ankle_boost_hypothesis`: left=`False`, right=`False`

## Ankle Saturation Analysis

- `current_box_baseline / rated_safe`: ankle saturation `0.258333`, actuator saturation `0.0516667`
- `current_box_baseline / mid_burst`: ankle saturation `0.141667`, actuator saturation `0.0283333`
- `current_box_baseline / peak_upper_bound`: ankle saturation `0.0916667`, actuator saturation `0.0183333`
- `current_box_baseline / ankle_boost_hypothesis`: ankle saturation `0.0416667`, actuator saturation `0.00833333`
- `simple_toe_center_heel / rated_safe`: ankle saturation `0.25`, actuator saturation `0.05`
- `simple_toe_center_heel / mid_burst`: ankle saturation `0.125`, actuator saturation `0.025`
- `simple_toe_center_heel / peak_upper_bound`: ankle saturation `0.0583333`, actuator saturation `0.0116667`
- `simple_toe_center_heel / ankle_boost_hypothesis`: ankle saturation `0`, actuator saturation `0`
- `duck_like_multi_patch / rated_safe`: ankle saturation `0.25`, actuator saturation `0.05`
- `duck_like_multi_patch / mid_burst`: ankle saturation `0.125`, actuator saturation `0.025`
- `duck_like_multi_patch / peak_upper_bound`: ankle saturation `0.0666667`, actuator saturation `0.0133333`
- `duck_like_multi_patch / ankle_boost_hypothesis`: ankle saturation `0`, actuator saturation `0`

## Rated vs Peak Interpretation

- `rated_safe` uses provided rated torque only and is valid for bounded diagnostics.
- `mid_burst` is an intermediate diagnostic profile, not a verified controller mode.
- `peak_upper_bound` is an upper-bound diagnostic only and is invalid for continuous gait claims.
- None of these values are verified MuJoCo joint forceranges because torque side is unknown.

## What Can Be Concluded

- v5_22 foot/contact observability and bounded profile saturation can be compared without modifying the source model.
- Rows with unavailable contact metrics cannot support center-first rollover analysis.
- Bridge contact remains a geometry issue, not proof of controller success or failure.

## What Must Not Be Claimed

- Do not claim walking success.
- Do not claim sim2real validity.
- Do not treat provided torque as verified joint-output forcerange.
- Do not use `peak_upper_bound` as a continuous gait claim.

## Next Recommendation

- Foot geometry tuning: tune center/toe/heel spacing and height to remove toe/heel bridge.
- Controller authority tuning: keep tests bounded and compare ankle saturation before changing rewards.
- Actuator spec request: request torque side, gear ratio, max velocity, current limit, encoder, backlash, and control mode.
- Do not PPO until foot observability and actuator/controller specs are less ambiguous.
