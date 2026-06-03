# Sedon v5_22 STL-Derived Foot Collision Probe Report

Task class: Class C contact diagnostic. This workflow reads ankle-pitch STL files, creates artifact-only simplified collision variants, does not modify source XML/URDF/STL/train/eval/env, does not run PPO, and does not claim walking success.

## Summary

- Status: `STL_DERIVED_FOOT_COLLISION_PROBE_COMPLETE`
- Best variant: `stl_fitted_lowered_toe_box`
- Best posture/profile: `ankle_toe_down_bias / ankle_boost_hypothesis`
- Best contact none rate: `0.78`
- Previous toe handoff best contact none rate: `0.85`
- Previous continuous bottom best contact none rate: `0.82`
- Improvement vs previous best 0.82: `0.039999999999999925`
- Prototype success found: `False`
- Useful signal found: `False`
- Replace current box prototype: `no; contact persistence remains insufficient`

## STL Bottom Profile

| side | shape candidate | bbox length | bbox width | bbox height | bottom length | bottom width |
|---|---|---:|---:|---:|---:|---:|
| `left` | `curved_candidate` | 0.199268 | 0.0739498 | 0.14486 | 0.125257 | 0.0289119 |
| `right` | `curved_candidate` | 0.199268 | 0.0739498 | 0.14486 | 0.125257 | 0.0289119 |

## Collision Variants

- `stl_fitted_box`: one fitted box per foot from lowest-z STL bottom bounds.
- `stl_fitted_lowered_toe_box`: fitted base box plus a lower toe-biased box.
- `stl_fitted_rocker_capsules`: three cross-foot capsules approximating heel/center/toe rocker support.
- `stl_fitted_continuous_bottom`: one continuous ellipsoid bottom proxy per foot.

All variants are tagged `source=stl_derived_prototype, valid_for=contact_diagnostic_only` and are not final collisions.

## Probe Results

| variant | profile | posture | none rate | persistence | x progression | heel | center | toe | rollover | bridge | tilt/fall | result |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| `stl_fitted_box` | `peak_upper_bound` | `neutral_stance` | 0.86 | 0.14 | 0.0199118 | 0.58429 | 0 | 0.41571 | 0.172815 | `True` | `False` | `bridge_like_pattern` |
| `stl_fitted_box` | `peak_upper_bound` | `ankle_toe_down_bias` | 0.79 | 0.21 | 0.081573 | 0 | 0 | 1 | 1 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_box` | `peak_upper_bound` | `mild_forward_lean` | 0.83 | 0.17 | 0.0388949 | 0.301985 | 0 | 0.698015 | 0.487225 | `True` | `False` | `bridge_like_pattern` |
| `stl_fitted_box` | `peak_upper_bound` | `medium_forward_base_pitch` | 0.84 | 0.16 | 0.0341717 | 0.383538 | 0 | 0.616462 | 0.380025 | `True` | `False` | `bridge_like_pattern` |
| `stl_fitted_box` | `ankle_boost_hypothesis` | `neutral_stance` | 0.86 | 0.14 | 0.0199118 | 0.58429 | 0 | 0.41571 | 0.172815 | `True` | `False` | `bridge_like_pattern` |
| `stl_fitted_box` | `ankle_boost_hypothesis` | `ankle_toe_down_bias` | 0.79 | 0.21 | 0.0813903 | 0 | 0 | 1 | 1 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_box` | `ankle_boost_hypothesis` | `mild_forward_lean` | 0.83 | 0.17 | 0.0388616 | 0.223342 | 0 | 0.776658 | 0.603198 | `True` | `False` | `bridge_like_pattern` |
| `stl_fitted_box` | `ankle_boost_hypothesis` | `medium_forward_base_pitch` | 0.84 | 0.16 | 0.0341527 | 0.313171 | 0 | 0.686829 | 0.471735 | `True` | `False` | `bridge_like_pattern` |
| `stl_fitted_lowered_toe_box` | `peak_upper_bound` | `neutral_stance` | 0.85 | 0.15 | 0.0401323 | 0.28365 | 0.632927 | 0.0834229 | 0.05976 | `True` | `False` | `bridge_like_pattern` |
| `stl_fitted_lowered_toe_box` | `peak_upper_bound` | `ankle_toe_down_bias` | 0.78 | 0.22 | 0.0726892 | 0 | 0.0678335 | 0.932166 | 0.932166 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_lowered_toe_box` | `peak_upper_bound` | `mild_forward_lean` | 0.81 | 0.19 | 0.0620594 | 0 | 0.669369 | 0.330631 | 0.330631 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_lowered_toe_box` | `peak_upper_bound` | `medium_forward_base_pitch` | 0.82 | 0.18 | 0.0522794 | 0.0417768 | 0.77961 | 0.178613 | 0.171152 | `True` | `False` | `bridge_like_pattern` |
| `stl_fitted_lowered_toe_box` | `ankle_boost_hypothesis` | `neutral_stance` | 0.85 | 0.15 | 0.0401323 | 0.28365 | 0.632927 | 0.0834229 | 0.05976 | `True` | `False` | `bridge_like_pattern` |
| `stl_fitted_lowered_toe_box` | `ankle_boost_hypothesis` | `ankle_toe_down_bias` | 0.78 | 0.22 | 0.0815318 | 0 | 0 | 1 | 1 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_lowered_toe_box` | `ankle_boost_hypothesis` | `mild_forward_lean` | 0.81 | 0.19 | 0.0681754 | 0 | 0.503938 | 0.496062 | 0.496062 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_lowered_toe_box` | `ankle_boost_hypothesis` | `medium_forward_base_pitch` | 0.82 | 0.18 | 0.0556536 | 0 | 0.697793 | 0.302207 | 0.302207 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_rocker_capsules` | `peak_upper_bound` | `neutral_stance` | 0.85 | 0.15 | 0.0206893 | 0.0186242 | 0.981376 | 0 | 0 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_rocker_capsules` | `peak_upper_bound` | `ankle_toe_down_bias` | 0.83 | 0.17 | 0.041654 | 0 | 0.0245328 | 0.975467 | 0.975467 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_rocker_capsules` | `peak_upper_bound` | `mild_forward_lean` | 0.85 | 0.15 | 0.0391821 | 0 | 0.564256 | 0.435744 | 0.435744 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_rocker_capsules` | `peak_upper_bound` | `medium_forward_base_pitch` | 0.85 | 0.15 | 0.0297209 | 0 | 0.693935 | 0.306065 | 0.306065 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_rocker_capsules` | `ankle_boost_hypothesis` | `neutral_stance` | 0.85 | 0.15 | 0.0206893 | 0.0186242 | 0.981376 | 0 | 0 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_rocker_capsules` | `ankle_boost_hypothesis` | `ankle_toe_down_bias` | 0.83 | 0.17 | 0.041685 | 0 | 0.00611294 | 0.993887 | 0.993887 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_rocker_capsules` | `ankle_boost_hypothesis` | `mild_forward_lean` | 0.85 | 0.15 | 0.0391821 | 0 | 0.564256 | 0.435744 | 0.435744 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_rocker_capsules` | `ankle_boost_hypothesis` | `medium_forward_base_pitch` | 0.85 | 0.15 | 0.0297209 | 0 | 0.693935 | 0.306065 | 0.306065 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_continuous_bottom` | `peak_upper_bound` | `neutral_stance` | 0.85 | 0.15 | 0.0109552 | 0 | 1 | 0 | 0 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_continuous_bottom` | `peak_upper_bound` | `ankle_toe_down_bias` | 0.8 | 0.2 | 0.0730478 | 0 | 0 | 1 | 1 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_continuous_bottom` | `peak_upper_bound` | `mild_forward_lean` | 0.83 | 0.17 | 0.0451674 | 0.121014 | 0.264144 | 0.614843 | 0.540438 | `True` | `False` | `bridge_like_pattern` |
| `stl_fitted_continuous_bottom` | `peak_upper_bound` | `medium_forward_base_pitch` | 0.84 | 0.16 | 0.0342147 | 0.260888 | 0.284341 | 0.454772 | 0.336127 | `True` | `False` | `bridge_like_pattern` |
| `stl_fitted_continuous_bottom` | `ankle_boost_hypothesis` | `neutral_stance` | 0.85 | 0.15 | 0.0109552 | 0 | 1 | 0 | 0 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_continuous_bottom` | `ankle_boost_hypothesis` | `ankle_toe_down_bias` | 0.8 | 0.2 | 0.0755557 | 0 | 0 | 1 | 1 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_continuous_bottom` | `ankle_boost_hypothesis` | `mild_forward_lean` | 0.83 | 0.17 | 0.05438 | 0 | 0.282695 | 0.717305 | 0.717305 | `False` | `False` | `insufficient_contact_persistence` |
| `stl_fitted_continuous_bottom` | `ankle_boost_hypothesis` | `medium_forward_base_pitch` | 0.84 | 0.16 | 0.0429833 | 0 | 0.443732 | 0.556268 | 0.556268 | `False` | `False` | `insufficient_contact_persistence` |

## Limitations

- STL orientation is assumed to use local z as vertical; this is `source=assumption, confidence=low`.
- The ankle-pitch STL is not treated as a final collision mesh.
- Contact metrics are diagnostic-only and do not establish walking success.
