# Seedon v5_22 Continuous Foot Bottom Probe Report

Task class: Class C contact persistence diagnostic. This report does not run PPO, does not claim walking success, and does not define final mechanical foot design.

## Summary

- Status: `CONTINUOUS_FOOT_BOTTOM_PROBE_COMPLETE`
- Result rows: `60`
- Best variant: `current_box_baseline`
- Best posture/profile: `ankle_toe_down_bias / peak_upper_bound`
- Best contact none rate: `0.82`
- Improvement vs previous 0.85: `0.030000000000000027`
- Continuous better than discrete: `True`
- Candidate found: `False`
- Next step: mechanical foot bottom redesign / mesh review

## Why Duck Suggests Continuous Foot Bottom

- Duck local XML exposes active foot contact candidates as `left_foot_bottom_tpu` and `right_foot_bottom_tpu` mesh collisions.
- Duck does not expose explicit toe/center/heel primitive patches, so this workflow tests continuous bottom contact plus contact-point x-region classification.

## Variant Design

- `current_box_baseline`: current v5_22 foot collision box.
- `discrete_toe_center_heel`: artifact-only discrete prototype patches.
- `continuous_rocker_bottom`: artifact-only single ellipsoid bottom per foot.
- `hybrid_continuous_with_region_classifier`: same continuous collision concept, classified by local contact x thirds.

## Contact Persistence Comparison

- Best discrete persistence: `0.15000000000000002`
- Best continuous persistence: `0.17000000000000004`

## Contact X Progression / Region Classification

| variant | profile | posture | none rate | x mean | heel | center | toe | progression | result |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `current_box_baseline` | `rated_safe` | `neutral_stance` | 0.9 | 0.0250069 | 0.628303 | 0 | 0.371697 | 0.0250069 | `insufficient_contact_persistence` |
| `current_box_baseline` | `rated_safe` | `small_forward_base_pitch` | 0.88 | 0.035007 | 0.614779 | 0 | 0.385221 | 0.035007 | `insufficient_contact_persistence` |
| `current_box_baseline` | `rated_safe` | `medium_forward_base_pitch` | 0.87 | 0.0458505 | 0.614132 | 0 | 0.385868 | 0.0458505 | `insufficient_contact_persistence` |
| `current_box_baseline` | `rated_safe` | `ankle_toe_down_bias` | 0.82 | 0.0689306 | 0.438638 | 0 | 0.561362 | 0.0689306 | `insufficient_contact_persistence` |
| `current_box_baseline` | `rated_safe` | `mild_knee_hip_flexion_forward_lean` | 0.85 | 0.0595559 | 0.493955 | 0 | 0.506045 | 0.0595559 | `insufficient_contact_persistence` |
| `current_box_baseline` | `peak_upper_bound` | `neutral_stance` | 0.9 | 0.0250069 | 0.628303 | 0 | 0.371697 | 0.0250069 | `insufficient_contact_persistence` |
| `current_box_baseline` | `peak_upper_bound` | `small_forward_base_pitch` | 0.88 | 0.034978 | 0.50756 | 0 | 0.49244 | 0.034978 | `insufficient_contact_persistence` |
| `current_box_baseline` | `peak_upper_bound` | `medium_forward_base_pitch` | 0.87 | 0.0457963 | 0.466291 | 0 | 0.533709 | 0.0457963 | `insufficient_contact_persistence` |
| `current_box_baseline` | `peak_upper_bound` | `ankle_toe_down_bias` | 0.82 | 0.0940638 | 0 | 0 | 1 | 0.0940638 | `insufficient_contact_persistence` |
| `current_box_baseline` | `peak_upper_bound` | `mild_knee_hip_flexion_forward_lean` | 0.85 | 0.0649807 | 0.28975 | 0 | 0.71025 | 0.0649807 | `insufficient_contact_persistence` |
| `current_box_baseline` | `ankle_boost_hypothesis` | `neutral_stance` | 0.9 | 0.0250069 | 0.628303 | 0 | 0.371697 | 0.0250069 | `insufficient_contact_persistence` |
| `current_box_baseline` | `ankle_boost_hypothesis` | `small_forward_base_pitch` | 0.88 | 0.0349716 | 0.484638 | 0 | 0.515362 | 0.0349716 | `insufficient_contact_persistence` |
| `current_box_baseline` | `ankle_boost_hypothesis` | `medium_forward_base_pitch` | 0.87 | 0.0505637 | 0.350269 | 0 | 0.649731 | 0.0505637 | `insufficient_contact_persistence` |
| `current_box_baseline` | `ankle_boost_hypothesis` | `ankle_toe_down_bias` | 0.82 | 0.0939533 | 0 | 0 | 1 | 0.0939533 | `insufficient_contact_persistence` |
| `current_box_baseline` | `ankle_boost_hypothesis` | `mild_knee_hip_flexion_forward_lean` | 0.85 | 0.064946 | 0.208717 | 0 | 0.791283 | 0.064946 | `insufficient_contact_persistence` |
| `discrete_toe_center_heel` | `rated_safe` | `neutral_stance` | 0.9 | 0.0254835 | 0.341144 | 0.639362 | 0.0194943 | 0.0254835 | `insufficient_contact_persistence` |
| `discrete_toe_center_heel` | `rated_safe` | `small_forward_base_pitch` | 0.89 | 0.0270309 | 0.248868 | 0.729093 | 0.0220383 | 0.0270309 | `insufficient_contact_persistence` |
| `discrete_toe_center_heel` | `rated_safe` | `medium_forward_base_pitch` | 0.88 | 0.0300937 | 0.211583 | 0.736638 | 0.0517797 | 0.0300937 | `insufficient_contact_persistence` |
| `discrete_toe_center_heel` | `rated_safe` | `ankle_toe_down_bias` | 0.85 | 0.0431314 | 0 | 0.75758 | 0.24242 | 0.0431314 | `insufficient_contact_persistence` |
| `discrete_toe_center_heel` | `rated_safe` | `mild_knee_hip_flexion_forward_lean` | 0.87 | 0.0336562 | 0.133828 | 0.738982 | 0.12719 | 0.0336562 | `insufficient_contact_persistence` |
| `discrete_toe_center_heel` | `peak_upper_bound` | `neutral_stance` | 0.9 | 0.0254835 | 0.341144 | 0.639362 | 0.0194943 | 0.0254835 | `insufficient_contact_persistence` |
| `discrete_toe_center_heel` | `peak_upper_bound` | `small_forward_base_pitch` | 0.89 | 0.0269994 | 0.153171 | 0.738507 | 0.108322 | 0.0269994 | `insufficient_contact_persistence` |
| `discrete_toe_center_heel` | `peak_upper_bound` | `medium_forward_base_pitch` | 0.88 | 0.0300149 | 0.0668404 | 0.772603 | 0.160557 | 0.0300149 | `insufficient_contact_persistence` |
| `discrete_toe_center_heel` | `peak_upper_bound` | `ankle_toe_down_bias` | 0.85 | 0.0463516 | 0 | 0.31572 | 0.68428 | 0.0463516 | `insufficient_contact_persistence` |
| `discrete_toe_center_heel` | `peak_upper_bound` | `mild_knee_hip_flexion_forward_lean` | 0.87 | 0.0366393 | 0.000467565 | 0.718823 | 0.280709 | 0.0366393 | `insufficient_contact_persistence` |
| `discrete_toe_center_heel` | `ankle_boost_hypothesis` | `neutral_stance` | 0.9 | 0.0254835 | 0.341144 | 0.639362 | 0.0194943 | 0.0254835 | `insufficient_contact_persistence` |
| `discrete_toe_center_heel` | `ankle_boost_hypothesis` | `small_forward_base_pitch` | 0.89 | 0.0269994 | 0.153171 | 0.738507 | 0.108322 | 0.0269994 | `insufficient_contact_persistence` |
| `discrete_toe_center_heel` | `ankle_boost_hypothesis` | `medium_forward_base_pitch` | 0.88 | 0.0300018 | 0.0351742 | 0.756797 | 0.208028 | 0.0300018 | `insufficient_contact_persistence` |
| `discrete_toe_center_heel` | `ankle_boost_hypothesis` | `ankle_toe_down_bias` | 0.85 | 0.0477242 | 0 | 0.118902 | 0.881098 | 0.0477242 | `insufficient_contact_persistence` |
| `discrete_toe_center_heel` | `ankle_boost_hypothesis` | `mild_knee_hip_flexion_forward_lean` | 0.87 | 0.0387707 | 0 | 0.589231 | 0.410769 | 0.0387707 | `insufficient_contact_persistence` |
| `continuous_rocker_bottom` | `rated_safe` | `neutral_stance` | 0.88 | 0.00698525 | 0.35057 | 0.64943 | 0 | 0.00698525 | `insufficient_contact_persistence` |
| `continuous_rocker_bottom` | `rated_safe` | `small_forward_base_pitch` | 0.88 | 0.0133268 | 0.450097 | 0.422915 | 0.126988 | 0.0133268 | `insufficient_contact_persistence` |
| `continuous_rocker_bottom` | `rated_safe` | `medium_forward_base_pitch` | 0.87 | 0.0366676 | 0.356448 | 0.31386 | 0.329691 | 0.0366676 | `insufficient_contact_persistence` |
| `continuous_rocker_bottom` | `rated_safe` | `ankle_toe_down_bias` | 0.83 | 0.0658158 | 0 | 0.298552 | 0.701448 | 0.0658158 | `insufficient_contact_persistence` |
| `continuous_rocker_bottom` | `rated_safe` | `mild_knee_hip_flexion_forward_lean` | 0.86 | 0.055429 | 0.137429 | 0.309307 | 0.553265 | 0.055429 | `insufficient_contact_persistence` |
| `continuous_rocker_bottom` | `peak_upper_bound` | `neutral_stance` | 0.88 | 0.0145862 | 0 | 1 | 0 | 0.0145862 | `insufficient_contact_persistence` |
| `continuous_rocker_bottom` | `peak_upper_bound` | `small_forward_base_pitch` | 0.88 | 0.024245 | 0.031776 | 0.831607 | 0.136617 | 0.024245 | `insufficient_contact_persistence` |
| `continuous_rocker_bottom` | `peak_upper_bound` | `medium_forward_base_pitch` | 0.87 | 0.0444876 | 0.0980157 | 0.443099 | 0.458885 | 0.0444876 | `insufficient_contact_persistence` |
| `continuous_rocker_bottom` | `peak_upper_bound` | `ankle_toe_down_bias` | 0.83 | 0.0756033 | 0 | 0 | 1 | 0.0756033 | `insufficient_contact_persistence` |
| `continuous_rocker_bottom` | `peak_upper_bound` | `mild_knee_hip_flexion_forward_lean` | 0.86 | 0.0638031 | 0 | 0.243821 | 0.756179 | 0.0638031 | `insufficient_contact_persistence` |
| `continuous_rocker_bottom` | `ankle_boost_hypothesis` | `neutral_stance` | 0.88 | 0.0145862 | 0 | 1 | 0 | 0.0145862 | `insufficient_contact_persistence` |
| `continuous_rocker_bottom` | `ankle_boost_hypothesis` | `small_forward_base_pitch` | 0.88 | 0.029649 | 0 | 0.865644 | 0.134356 | 0.029649 | `insufficient_contact_persistence` |
| `continuous_rocker_bottom` | `ankle_boost_hypothesis` | `medium_forward_base_pitch` | 0.87 | 0.0487142 | 0 | 0.527425 | 0.472575 | 0.0487142 | `insufficient_contact_persistence` |
| `continuous_rocker_bottom` | `ankle_boost_hypothesis` | `ankle_toe_down_bias` | 0.83 | 0.0790004 | 0 | 0 | 1 | 0.0790004 | `insufficient_contact_persistence` |
| `continuous_rocker_bottom` | `ankle_boost_hypothesis` | `mild_knee_hip_flexion_forward_lean` | 0.86 | 0.0674861 | 0 | 0.153835 | 0.846165 | 0.0674861 | `insufficient_contact_persistence` |
| `hybrid_continuous_with_region_classifier` | `rated_safe` | `neutral_stance` | 0.88 | 0.00698525 | 0.35057 | 0.64943 | 0 | 0.00698525 | `insufficient_contact_persistence` |
| `hybrid_continuous_with_region_classifier` | `rated_safe` | `small_forward_base_pitch` | 0.88 | 0.0133268 | 0.450097 | 0.422915 | 0.126988 | 0.0133268 | `insufficient_contact_persistence` |
| `hybrid_continuous_with_region_classifier` | `rated_safe` | `medium_forward_base_pitch` | 0.87 | 0.0366676 | 0.356448 | 0.31386 | 0.329691 | 0.0366676 | `insufficient_contact_persistence` |
| `hybrid_continuous_with_region_classifier` | `rated_safe` | `ankle_toe_down_bias` | 0.83 | 0.0658158 | 0 | 0.298552 | 0.701448 | 0.0658158 | `insufficient_contact_persistence` |
| `hybrid_continuous_with_region_classifier` | `rated_safe` | `mild_knee_hip_flexion_forward_lean` | 0.86 | 0.055429 | 0.137429 | 0.309307 | 0.553265 | 0.055429 | `insufficient_contact_persistence` |
| `hybrid_continuous_with_region_classifier` | `peak_upper_bound` | `neutral_stance` | 0.88 | 0.0145862 | 0 | 1 | 0 | 0.0145862 | `insufficient_contact_persistence` |
| `hybrid_continuous_with_region_classifier` | `peak_upper_bound` | `small_forward_base_pitch` | 0.88 | 0.024245 | 0.031776 | 0.831607 | 0.136617 | 0.024245 | `insufficient_contact_persistence` |
| `hybrid_continuous_with_region_classifier` | `peak_upper_bound` | `medium_forward_base_pitch` | 0.87 | 0.0444876 | 0.0980157 | 0.443099 | 0.458885 | 0.0444876 | `insufficient_contact_persistence` |
| `hybrid_continuous_with_region_classifier` | `peak_upper_bound` | `ankle_toe_down_bias` | 0.83 | 0.0756033 | 0 | 0 | 1 | 0.0756033 | `insufficient_contact_persistence` |
| `hybrid_continuous_with_region_classifier` | `peak_upper_bound` | `mild_knee_hip_flexion_forward_lean` | 0.86 | 0.0638031 | 0 | 0.243821 | 0.756179 | 0.0638031 | `insufficient_contact_persistence` |
| `hybrid_continuous_with_region_classifier` | `ankle_boost_hypothesis` | `neutral_stance` | 0.88 | 0.0145862 | 0 | 1 | 0 | 0.0145862 | `insufficient_contact_persistence` |
| `hybrid_continuous_with_region_classifier` | `ankle_boost_hypothesis` | `small_forward_base_pitch` | 0.88 | 0.029649 | 0 | 0.865644 | 0.134356 | 0.029649 | `insufficient_contact_persistence` |
| `hybrid_continuous_with_region_classifier` | `ankle_boost_hypothesis` | `medium_forward_base_pitch` | 0.87 | 0.0487142 | 0 | 0.527425 | 0.472575 | 0.0487142 | `insufficient_contact_persistence` |
| `hybrid_continuous_with_region_classifier` | `ankle_boost_hypothesis` | `ankle_toe_down_bias` | 0.83 | 0.0790004 | 0 | 0 | 1 | 0.0790004 | `insufficient_contact_persistence` |
| `hybrid_continuous_with_region_classifier` | `ankle_boost_hypothesis` | `mild_knee_hip_flexion_forward_lean` | 0.86 | 0.0674861 | 0 | 0.153835 | 0.846165 | 0.0674861 | `insufficient_contact_persistence` |

## Whether Continuous Bottom Improves Over Discrete Patches

- Continuous better than discrete by best persistence: `True`

## Whether Toe Handoff / Rollover Is Physically Observable

- Continuous rollover candidate found: `False`
- Classification uses prototype thresholds: rear/middle/front thirds in foot local x.

## Recommended Next Step

- mechanical foot bottom redesign / mesh review

## What Must Not Be Claimed

- Do not claim walking success.
- Do not claim sim2real validity.
- Do not treat continuous primitive as final mechanical design.
- Do not treat `peak_upper_bound` as a continuous gait claim.
