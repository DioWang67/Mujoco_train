# Sedon Blue-Like Dynamic Gait Progress Log

This append-only log tracks Sedon Blue / BDX-like dynamic gait experiments, diagnostics, controller results, and engineering decisions.

Core decisions:

- Grounded shuffle is only the Phase 0 baseline, not final walking.
- The target remains Blue / BDX-like dynamic gait.
- Visible stepping cannot be solved by lift scale or landing interpolation alone.
- The current bridge is controlled forward commit, support/swing force split, and toe rollover.
- Force split must be established before any full SWING_FORWARD phase is allowed.
- Toe handoff must be observed from contact / force, not hard-coded.
- Base x displacement alone is never sufficient to claim gait success.

## 2026-05-29 - Phase 0 v5_a Geometry And Teacher Baseline

### Goal

Establish the current grounded / slow forward shuffle baseline and best known sole geometry.

### Files

- `artifacts/sedon_debug/blue_like_sole_experiments_v5/training_scene_v5_a.xml`
- `models/sedon/teacher_safe_baseline/`

### Key Metrics

| Metric | Value |
|---|---:|
| v5_a center-first standing | observed |
| dynamic push center-to-toe handoff | observed previously |
| teacher pipeline | grounded shuffle |

### Result

Baseline only. Not walking.

### Engineering Interpretation

v5_a remains the best candidate geometry, but grounded shuffle must not be promoted to dynamic gait.

### Next Decision

Use v5_a for controlled forward commit and force split diagnostics.

## 2026-05-29 - Phase 1 Rollover Diagnostic

### Goal

Check whether forward momentum can produce controlled falling, contact continuity, rollover, and support alternation without requiring visible clearance.

### Files

- `tools/sedon_blue_like_phase1_rollover_diagnostic.py`

### Command

```powershell
.venv\Scripts\python.exe -m tools.sedon_blue_like_phase1_rollover_diagnostic --steps 600
```

### Outputs

- `artifacts/sedon_debug/blue_like_phase1_rollover/phase1_rollover_timeline.csv`
- `artifacts/sedon_debug/blue_like_phase1_rollover/phase1_rollover_summary.json`

### Key Metrics

| Metric | Value |
|---|---:|
| steps | 105 |
| mean_forward_velocity | 0.1126 |
| forward_displacement | 0.1556 |
| min_upright | 0.7160 |
| contact_none_ratio | 0.0952 |
| jump_count | 2 |
| support_force_ratio_peak | 1.0 |
| toe_handoff_detected | false |
| left_right_phase_switch_count | 4 |

### Result

FAIL: `contact_none,jump,upright,toe_handoff`.

### Engineering Interpretation

The probe creates forward momentum, but it is too uncontrolled and collapses into fall/no-contact. This is not a gait success.

### Next Decision

Replace raw dynamic push with bounded forward commit and force-gated recovery.

## 2026-05-29 - Capture V1 Controller Skeleton

### Goal

Create a first capture-step debug skeleton without claiming walking success.

### Files

- `tools/sedon_capture_step_controller_v1.py`

### Command

```powershell
.venv\Scripts\python.exe -m tools.sedon_capture_step_controller_v1 --steps 600
```

### Outputs

- `artifacts/sedon_debug/capture_step_controller_v1/capture_step_controller_v1.csv`
- `artifacts/sedon_debug/capture_step_controller_v1/capture_step_controller_v1_summary.json`

### Key Metrics

| Metric | Value |
|---|---:|
| forward_displacement | 0.0355 |
| mean_forward_velocity | 0.0031 |
| min_upright | 0.9927 |
| contact_none_ratio | 0.0 |
| jump_count | 0 |
| support_force_ratio_mean | 0.5019 |
| swing_force_ratio_min | 0.4647 |
| toe_handoff_detected | false |
| still_grounded_shuffle | true |

### Result

FAIL as dynamic gait. Safe but too conservative.

### Engineering Interpretation

The FSM is stable but does not open support/swing force split. It remains grounded shuffle.

### Next Decision

Do not enter full capture step. Build a Phase 1.5 force-split + rollover controller first.

## 2026-05-29 - Phase 1.5 Force Split + Rollover Controller

### Goal

Test bounded forward commit, stable support/swing force split, contact continuity, and natural rollover before any visible stepping.

### Files

- `tools/sedon_phase1_5_force_split_rollover_controller.py`
- `docs/sedon_blue_like_dynamic_gait_progress_log.md`

### Command

```powershell
.venv\Scripts\python.exe -m tools.sedon_phase1_5_force_split_rollover_controller --steps 600 --max-forward-bias 2.0 --forward-ramp-rate 0.04 --support-ratio-enter 0.58 --support-ratio-exit 0.54 --max-roll 0.055 --max-pitch 0.012
```

### Outputs

- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase1_5_force_split_rollover\phase1_5_timeline.csv`
- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase1_5_force_split_rollover\phase1_5_summary.json`

### Key Metrics

| Metric | Value |
|---|---:|
| contact_none_ratio | 0.000000 |
| jump_count | 0 |
| min_upright | 0.993311 |
| mean_forward_velocity | 0.002982 |
| forward_displacement | 0.035501 |
| support_force_ratio_peak | 0.557138 |
| support_force_ratio_hold_steps | 0 |
| swing_force_ratio_min | 0.442862 |
| force_gate_reached_count | 0 |
| left_right_phase_switch_count | 3 |
| toe_handoff_detected | false |
| toe_handoff_left_count | 0 |
| toe_handoff_right_count | 0 |
| still_grounded_shuffle | true |

### Result

FAIL: `support_force_ratio_peak,support_force_ratio_hold_steps`

### Engineering Interpretation

Phase 1.5 did not satisfy the force-split bridge. Do not proceed to capture stepping until support ratio can hold above the gate safely.

### Next Decision

Debug hip-roll direction, pelvis lean authority, support hip-roll contact-force effect, contact patch rollover, stance width, COM height, and v5_a controlled-commit rollover.

## 2026-05-29 - Phase 1.6 Load Transfer Attribution Sweep

### Goal

Identify which control channels can increase support/swing force split safely before attempting capture-step or visible stepping.

### Files

- `tools/sedon_phase1_6_load_transfer_attribution.py`
- `docs/sedon_blue_like_dynamic_gait_progress_log.md`

### Command

```powershell
.venv\Scripts\python.exe -m tools.sedon_phase1_6_load_transfer_attribution --steps 200
```

### Outputs

- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase1_6_load_transfer_attribution\phase1_6_trials.csv`
- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase1_6_load_transfer_attribution\phase1_6_summary.json`
- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase1_6_load_transfer_attribution\phase1_6_top_candidates.csv`

### Key Metrics

| Metric | Value |
|---|---:|
| total_trials | 224 |
| any_stable_split | false |
| max_stable_support_force_ratio_peak | 0.609395 |
| max_support_force_ratio_hold_steps_058 | 13 |
| any_toe_handoff | false |
| best_channel_set | pelvis_lean |
| best_sign_pattern | +1.0x |

### Result

INCONCLUSIVE

### Engineering Interpretation

Force split is not fully open, but it is close to threshold under stable conditions. Most effective channel_set by ranking: pelvis_lean+support_hip_roll+swing_hip_roll. Most effective sign_pattern: +1.0x. Gain increase reaches unstable split before stable split. Max support_force_ratio_peak overall=1.0000, stable=0.6094. Phase 2 recommendation: no. Next focus: Do not enter Phase 2. Tune controller load-transfer shaping around the best near-split channel before changing morphology.

### Next Decision

Do not enter Phase 2. Tune controller load-transfer shaping around the best near-split channel before changing morphology.

## 2026-05-29 - Phase 1.6 Load Transfer Attribution Sweep

### Goal

Identify which control channels can increase support/swing force split safely before attempting capture-step or visible stepping.

### Files

- `tools/sedon_phase1_6_load_transfer_attribution.py`
- `docs/sedon_blue_like_dynamic_gait_progress_log.md`

### Command

```powershell
.venv\Scripts\python.exe -m tools.sedon_phase1_6_load_transfer_attribution --steps 200
```

### Outputs

- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase1_6_load_transfer_attribution\phase1_6_trials.csv`
- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase1_6_load_transfer_attribution\phase1_6_summary.json`
- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase1_6_load_transfer_attribution\phase1_6_top_candidates.csv`

### Key Metrics

| Metric | Value |
|---|---:|
| total_trials | 224 |
| any_stable_split | false |
| max_stable_support_force_ratio_peak | 0.609395 |
| max_support_force_ratio_hold_steps_058 | 13 |
| any_toe_handoff | false |
| best_channel_set | pelvis_lean |
| best_sign_pattern | +1.0x |

### Result

INCONCLUSIVE

### Engineering Interpretation

Force split is not fully open, but it is close to threshold under stable conditions. Safest effective channel_set: pelvis_lean. Safest effective sign_pattern: +1.0x. Top sorted candidate=pelvis_lean+support_hip_roll+swing_hip_roll, sign=+1.0x, classification=unstable_split. Raw strongest channel_set=pelvis_lean+support_hip_roll+swing_hip_roll, sign=-1.0x, classification=unstable_split. Gain increase reaches unstable split before stable split. Max support_force_ratio_peak overall=1.0000, stable=0.6094. Phase 2 recommendation: no. Next focus: Do not enter Phase 2. Tune controller load-transfer shaping around the best near-split channel before changing morphology.

### Next Decision

Do not enter Phase 2. Tune controller load-transfer shaping around the best near-split channel before changing morphology.

## 2026-05-29 - Phase 1.6 Load Transfer Attribution Sweep

### Goal

Identify which control channels can increase support/swing force split safely before attempting capture-step or visible stepping.

### Files

- `tools/sedon_phase1_6_load_transfer_attribution.py`
- `docs/sedon_blue_like_dynamic_gait_progress_log.md`

### Command

```powershell
.venv\Scripts\python.exe -m tools.sedon_phase1_6_load_transfer_attribution --steps 200
```

### Outputs

- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase1_6_load_transfer_attribution\phase1_6_trials.csv`
- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase1_6_load_transfer_attribution\phase1_6_summary.json`
- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase1_6_load_transfer_attribution\phase1_6_top_candidates.csv`

### Key Metrics

| Metric | Value |
|---|---:|
| total_trials | 224 |
| any_stable_split | false |
| max_stable_support_force_ratio_peak | 0.609395 |
| max_support_force_ratio_hold_steps_058 | 13 |
| any_toe_handoff | false |
| best_channel_set | pelvis_lean+swing_hip_roll |
| best_sign_pattern | +1.0x |

### Result

INCONCLUSIVE

### Engineering Interpretation

Force split is not fully open, but it is close to threshold under stable conditions. Safest effective channel_set: pelvis_lean+swing_hip_roll. Safest effective sign_pattern: +1.0x. Top sorted candidate=pelvis_lean+support_hip_roll+swing_hip_roll, sign=+1.0x, classification=unstable_split. Raw strongest channel_set=pelvis_lean+support_hip_roll+swing_hip_roll, sign=-1.0x, classification=unstable_split. Gain increase reaches unstable split before stable split. Max support_force_ratio_peak overall=1.0000, stable=0.6094. Phase 2 recommendation: no. Next focus: Do not enter Phase 2. Tune controller load-transfer shaping around the best near-split channel before changing morphology.

### Next Decision

Do not enter Phase 2. Tune controller load-transfer shaping around the best near-split channel before changing morphology.

## 2026-05-29 - Phase 1.7 Load Transfer Profile Shaping

### Goal

Shape the pelvis_lean+swing_hip_roll load-transfer profile so that support_force_ratio >= 0.58 can be held continuously without contact-none, jump, or upright failure.

### Files

- `tools/sedon_phase1_7_load_transfer_profile_shaping.py`
- `docs/sedon_blue_like_dynamic_gait_progress_log.md`

### Command

```powershell
.venv\Scripts\python.exe -m tools.sedon_phase1_7_load_transfer_profile_shaping --steps 240
```

### Outputs

- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase1_7_load_transfer_profile_shaping\phase1_7_trials.csv`
- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase1_7_load_transfer_profile_shaping\phase1_7_summary.json`
- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase1_7_load_transfer_profile_shaping\phase1_7_top_candidates.csv`

### Key Metrics

| Metric | Value |
|---|---:|
| total_trials | 48 |
| any_stable_profile_split | false |
| max_stable_support_force_ratio_peak | 0.576995 |
| max_continuous_hold_058 | 0 |
| max_cumulative_hold_058 | 0 |
| best_profile_shape | linear |
| best_magnitude | 0.0225 |
| any_toe_handoff | false |

### Result

FAIL

### Engineering Interpretation

Profile shaping did not materially improve hold: max_continuous_hold_058=0, max_cumulative_hold_058=0. Best profile shape=linear, magnitude=0.0225. Ramp/hold/recover trend from best candidate: ramp=20, hold=40, recover=40. Magnitude is bounded by stability and hold continuity, not simply larger-is-better. Support symmetry is roughly symmetric. Phase 2 recommendation: no. Next focus: Do not enter Phase 2. Continue controller shaping and inspect toe rocker/contact patch if rollover remains absent.

### Next Decision

Do not enter Phase 2. Continue controller shaping and inspect toe rocker/contact patch if rollover remains absent.

## 2026-05-29 - Phase 1.7 Load Transfer Profile Shaping

### Goal

Shape the pelvis_lean+swing_hip_roll load-transfer profile so that support_force_ratio >= 0.58 can be held continuously without contact-none, jump, or upright failure.

### Files

- `tools/sedon_phase1_7_load_transfer_profile_shaping.py`
- `docs/sedon_blue_like_dynamic_gait_progress_log.md`

### Command

```powershell
.venv\Scripts\python.exe -m tools.sedon_phase1_7_load_transfer_profile_shaping --steps 240
```

### Outputs

- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase1_7_load_transfer_profile_shaping\phase1_7_trials.csv`
- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase1_7_load_transfer_profile_shaping\phase1_7_summary.json`
- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase1_7_load_transfer_profile_shaping\phase1_7_top_candidates.csv`

### Key Metrics

| Metric | Value |
|---|---:|
| total_trials | 48 |
| any_stable_profile_split | true |
| max_stable_support_force_ratio_peak | 0.594501 |
| max_continuous_hold_058 | 36 |
| max_cumulative_hold_058 | 36 |
| best_profile_shape | linear |
| best_magnitude | 0.03 |
| any_toe_handoff | false |

### Result

PASS

### Engineering Interpretation

Profile shaping improved hold: max_continuous_hold_058=36, max_cumulative_hold_058=36. Best profile shape=linear, magnitude=0.03. Ramp/hold/recover trend from best candidate: ramp=10, hold=80, recover=40. Magnitude is bounded by stability and hold continuity, not simply larger-is-better. Support symmetry is asymmetric left=0, right=36. Phase 2 recommendation: yes. Next focus: Consider Phase 2 force-gated micro capture only with this profile and strict safety gates.

### Next Decision

Consider Phase 2 force-gated micro capture only with this profile and strict safety gates.

## 2026-05-29 - Phase 2A Right-Support Force-Gated Micro Capture

### Goal

Test whether a very small left-leg micro capture intent can be safely added after the right-support force gate is established.

### Files

- `tools/sedon_phase2a_right_support_micro_capture.py`
- `docs/sedon_blue_like_dynamic_gait_progress_log.md`

### Command

```powershell
.venv\Scripts\python.exe -m tools.sedon_phase2a_right_support_micro_capture --steps 260
```

### Outputs

- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase2a_right_support_micro_capture\phase2a_trials.csv`
- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase2a_right_support_micro_capture\phase2a_summary.json`
- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase2a_right_support_micro_capture\phase2a_top_candidates.csv`

### Key Metrics

| Metric | Value |
|---|---:|
| total_trials | 180 |
| any_micro_capture_safe | true |
| max_left_foot_forward_delta_safe | 0.000080 |
| max_support_force_ratio_min_during_capture_safe | 0.592269 |
| max_continuous_hold_058_safe | 32 |
| any_toe_handoff | false |
| best_capture_forward_bias | 0.001 |
| best_capture_lateral_bias | 0.0 |
| best_capture_duration_steps | 20 |

### Result

PASS

### Engineering Interpretation

Phase 2A can add left-leg micro capture after right-support gate. Best capture_forward_bias=0.001, capture_lateral_bias=0.0, duration=20, classification=micro_capture_safe. Micro capture preserved the force split. Left foot forward delta best=0.000080. Toe handoff observed=False. Phase 2B recommendation: yes. Next focus: Consider Phase 2B only as another force-gated micro probe, not full walking.

### Next Decision

Consider Phase 2B only as another force-gated micro probe, not full walking.

## 2026-05-29 - Phase 2B Force-Gated Micro Capture Refinement

### Goal

Refine the left-leg micro capture intent under the right-support force gate, aiming to increase measurable left-foot forward reposition without breaking support force ratio, contact safety, or upright stability.

### Files

- `tools/sedon_phase2b_micro_capture_refinement.py`
- `docs/sedon_blue_like_dynamic_gait_progress_log.md`

### Command

```powershell
.venv\Scripts\python.exe -m tools.sedon_phase2b_micro_capture_refinement --steps 280
```

### Outputs

- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase2b_micro_capture_refinement\phase2b_trials.csv`
- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase2b_micro_capture_refinement\phase2b_summary.json`
- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase2b_micro_capture_refinement\phase2b_top_candidates.csv`

### Key Metrics

| Metric | Value |
|---|---:|
| total_trials | 315 |
| any_micro_capture_refined_safe | false |
| any_micro_capture_strong_safe | false |
| max_left_foot_forward_delta_safe | 0.000000 |
| max_support_force_ratio_min_during_capture_safe | 0.000000 |
| max_continuous_hold_058_safe | 0 |
| any_toe_handoff | false |
| best_capture_forward_bias | 0.0075 |
| best_capture_duration_steps | 30 |
| best_capture_profile_shape | linear |

### Result

FAIL

### Engineering Interpretation

Phase 2B did not reach >=0.3 mm safe forward delta. Strong >=1.0 mm safe capture=False. Best capture_forward_bias=0.0075, duration=30, profile=linear, classification=micro_capture_tiny_safe. Force gate limited the refinement or no refined candidate emerged. Toe handoff observed=False. Phase 2C recommendation: no. Next focus: Do not progress. Inspect left leg joint mapping, toe rocker, and contact patch before increasing capture bias.

### Next Decision

Do not progress. Inspect left leg joint mapping, toe rocker, and contact patch before increasing capture bias.

## 2026-05-29 - Phase 2C Contact-Constrained Foot Mapping + Rollover Diagnostic

### Goal

Diagnose why left-foot micro capture remains around 0.08 mm under the right-support force gate, and why toe handoff has not appeared despite stable load transfer.

### Files

- `tools/sedon_phase2c_contact_constrained_foot_mapping.py`
- `docs/sedon_blue_like_dynamic_gait_progress_log.md`

### Command

```powershell
.venv\Scripts\python.exe -m tools.sedon_phase2c_contact_constrained_foot_mapping --steps 240
```

### Outputs

- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase2c_contact_constrained_foot_mapping\phase2c_joint_mapping.csv`
- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase2c_contact_constrained_foot_mapping\phase2c_contact_constrained_mapping.csv`
- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase2c_contact_constrained_foot_mapping\phase2c_rollover_timeline.csv`
- `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase2c_contact_constrained_foot_mapping\phase2c_summary.json`

### Key Metrics

| Metric | Value |
|---|---:|
| any_forward_effective_joint_channel | true |
| joint_mapping_best_forward_channel | left_hip_pitch+left_knee_pitch -1x 0.0200 |
| contact_constraint_blocks_forward_motion | true |
| constraint_loss_ratio_x_mean | -8.312738 |
| right_support_rollover_detected | false |
| toe_handoff_detected | false |
| toe_handoff_detection_reliable | true |
| max_right_toe_force | 44.706362 |
| max_right_center_force | 0.000000 |

### Result

INCONCLUSIVE

### Engineering Interpretation

Left-foot kinematic forward mapping exists=True. Contact-constrained forward blocking rows=147. Best forward channel=left_hip_pitch+left_knee_pitch -1x 0.0200. Toe handoff detected=False; detection reliable=True. Right center steps=0, toe steps=240, max toe force=44.706, max center force=0.000. Controlled right-support profile rollover=False. Next focus: Fix toe/contact geometry or contact patch before further capture-controller work.

### Next Decision

Fix toe/contact geometry or contact patch before further capture-controller work.

## 2026-06-01 - Phase G1 Contact Geometry + Patch Classification Diagnostic

### Goal

Diagnose why the Phase 2C right-support profile shows zero center contact and persistent toe+heel contact, preventing center-first rollover and toe handoff.

This Phase G1 entry is specifically created to close the diagnostic gap from the Phase 2C right-support profile: the prior Phase 2C log reported no right center contact while toe contact remained active. That geometry/contact-patch failure mode can block center-first rollover and prevent the controller from ever learning a clean toe handoff.

### Files

- `docs/sedon_blue_like_dynamic_gait_progress_log.md`
- TBD - Phase G1 diagnostic script path not created yet.
- TBD - generated Phase G1 scene XML path not created yet.
- TBD - Phase G1 diagnostic CSV path not produced yet.
- TBD - Phase G1 diagnostic JSON path not produced yet.
- TBD - Phase G1 diagnostic report path not produced yet.

### Command

```text
TBD - diagnostic command not executed yet
```

### Outputs

- TBD - contact geometry diagnostic report not produced yet.
- TBD - patch classification CSV / JSON not produced yet.
- TBD - raw contact pair dump not produced yet.
- TBD - scan result summary not produced yet.

### Key Metrics

Current values are TBD because Phase G1 raw contact geometry and patch-classification diagnostics have not been executed yet. The diagnostic trigger comes from the previous Phase 2C progress-log data, which showed `max_right_center_force = 0.000000` and no toe handoff in the right-support profile.

| Metric                                  | Value |
| --------------------------------------- | ----: |
| center_geom_detected_right              |   TBD |
| neutral_center_first_right              |   TBD |
| neutral_toe_false_touch_right           |   TBD |
| neutral_heel_false_touch_right          |   TBD |
| right_toe_heel_bridge_contact_detected  |   TBD |
| any_center_first_candidate_in_scan      |   TBD |
| any_toe_handoff_candidate_in_scan       |   TBD |
| raw_contact_pair_confirms_center_absent |   TBD |

### Result

INCONCLUSIVE

### Engineering Interpretation

Phase 2C showed a right-support profile with zero measured center contact and no toe handoff. That is enough to justify a geometry-level diagnostic, but it is not enough by itself to prove whether the center patch is missing, misclassified, too high, filtered out by collision settings, or simply never selected by the contact solver under the tested pose.

The persistent toe contact, and the suspected toe+heel contact pattern, may indicate one of several geometry issues: rocker patches could be bridging through overlap, toe and heel patches could be too low relative to the intended center patch, collision margins could create false contacts, or the diagnostic classifier could be assigning raw MuJoCo contact pairs to the wrong patch labels. Phase G1 must verify this through raw contact geom pairs, not only through derived patch labels.

At this point, `center_geom_detected_right` and `raw_contact_pair_confirms_center_absent` are still TBD. Therefore, the current engineering conclusion is intentionally limited: the existing Phase 2C data indicates a contact-patch failure that can break center-first rollover, but Phase G1 has not yet identified the exact root cause.

If the right center patch never contacts the floor, the center-first rollover objective is not trainable because the reward/diagnostic signal cannot distinguish a valid center support phase. If toe+heel bridge contact is present in neutral stance, toe handoff is also compromised because the model may already be in a false toe/heel contact state before the controller reaches the intended transition. In either case, going straight to PPO would likely optimize around broken contact semantics instead of learning the intended gait.

### Next Decision

1. If the right center geom is missing or never contacts the floor, fix the right foot center patch geometry, pose, size, or collision group before any PPO training.
2. If toe+heel bridge contact exists, adjust toe / heel rocker geom spacing, height, or collision margin so neutral stance does not falsely touch both patches.
3. If patch classification is wrong, fix the diagnostic classifier and validate directly against raw MuJoCo contact geom pairs.
4. If Phase G1 passes, proceed to Phase G2: regenerate the corrected scene, then run neutral stance, pitch sweep, and PPO smoke tests.
5. If Phase G1 fails, do not enter PPO training; repair contact geometry first.

## 2026-06-01 - Phase M0 Sedon vs Open Duck Morphology Audit

### Goal

Audit whether Open Duck Mini v2 gait-level references can guide Sedon without directly transferring Duck joint angles, raw actions, ONNX weights, or controls.

### Files

* `tools/sedon_phase_m0_duck_morphology_audit.py`
* `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase_m0_duck_morphology_audit\sedon_joint_topology.csv`
* `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase_m0_duck_morphology_audit\duck_joint_topology.csv`
* `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase_m0_duck_morphology_audit\joint_topology_comparison.csv`
* `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase_m0_duck_morphology_audit\sedon_actuator_inventory.csv`
* `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase_m0_duck_morphology_audit\duck_actuator_inventory.csv`
* `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase_m0_duck_morphology_audit\actuator_comparison.csv`
* `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase_m0_duck_morphology_audit\sedon_morphology_metrics.json`
* `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase_m0_duck_morphology_audit\duck_morphology_metrics.json`
* `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase_m0_duck_morphology_audit\morphology_comparison.json`
* `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase_m0_duck_morphology_audit\morphology_comparison.csv`
* `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase_m0_duck_morphology_audit\sedon_foot_geom_inventory.csv`
* `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase_m0_duck_morphology_audit\duck_foot_geom_inventory.csv`
* `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase_m0_duck_morphology_audit\foot_contact_style_comparison.md`
* `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase_m0_duck_morphology_audit\sedon_joint_effect_probe.csv`
* `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase_m0_duck_morphology_audit\sedon_joint_sign_mapping.json`
* `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase_m0_duck_morphology_audit\sedon_duck_scaled_gait_reference.json`
* `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\phase_m0_duck_morphology_audit\phase_m0_duck_morphology_audit_report.md`
* `docs/sedon_blue_like_dynamic_gait_progress_log.md`

### Commands

```text
python -m py_compile tools/sedon_phase_m0_duck_morphology_audit.py
.venv\Scripts\python.exe -m tools.sedon_phase_m0_duck_morphology_audit
```

### Key Metrics

| Metric | Value |
| --- | ---: |
| sedon_target_vx | 0.141672 |
| sedon_gait_period | 0.570959 |
| sedon_foot_clearance_target | 0.042626 |
| sedon_flight_ratio_target | 0.000000 |
| sedon_action_scale_initial | 0.250000 |

### Result

INCONCLUSIVE

### Engineering Interpretation

Open Duck Mini v2 provides a useful gait-level reference, especially forward velocity, gait period, clearance, no-flight support timing, and roll/pitch envelope. It should not be used as a raw joint/action transfer source because Sedon differs in morphology, actuator semantics, joint sign mapping, and contact geometry. Sedon's split contact geometry remains a risk relative to Duck's single foot-bottom style.

### Next Decision

1. Keep Phase W0-DuckRef at gait-metric level only.
2. Keep or generate a simplified foot_bottom_collision comparison variant before PPO tuning on split patches.
3. Use Sedon joint sign mapping for controller/reference construction.
4. Do not train until Phase G1/G2 contact semantics are consistent.

## 2026-06-01 - Phase G1 Actual Result Update

### Goal

Replace the earlier Phase G1 TBD state with actual raw MuJoCo contact-pair evidence, without changing foot geometry or claiming walking success.

### Files

- `tools/sedon_phase_g1_raw_contact_pair_diagnostic.py`
- `artifacts/sedon_debug/phase_g1_raw_contact_pair_diagnostic/raw_contact_pairs.csv`
- `artifacts/sedon_debug/phase_g1_raw_contact_pair_diagnostic/contact_region_summary.csv`
- `artifacts/sedon_debug/phase_g1_raw_contact_pair_diagnostic/contact_geom_inventory.csv`
- `artifacts/sedon_debug/phase_g1_raw_contact_pair_diagnostic/phase_g1_summary.json`
- `artifacts/sedon_debug/phase_g1_raw_contact_pair_diagnostic/phase_g1_report.md`

### Commands

```text
python -m py_compile tools/sedon_phase_g1_raw_contact_pair_diagnostic.py
.venv\Scripts\python.exe -m tools.sedon_phase_g1_raw_contact_pair_diagnostic --neutral-only
.venv\Scripts\python.exe -m tools.sedon_phase_g1_raw_contact_pair_diagnostic --pitch-sweep
```

### Key Metrics

| Metric | Value |
| --- | ---: |
| center_geom_detected_right | true |
| any_right_center_raw_contact | true |
| max_right_center_force | 994.880460 |
| max_right_toe_force | 417.336193 |
| max_right_heel_force | 683.616468 |
| right_center_force_zero_all_steps | false |
| right_toe_force_dominates_neutral | false |
| contact_classifier_unknown_force_ratio | 0.243728 |

### Result

INCONCLUSIVE

### Engineering Interpretation

G1 disproves the strongest version of "right center geom does not exist / cannot carry force". Raw right-center contact exists and can carry substantial force. Therefore, the earlier Phase 2C `right_center_force=0` result is now likely classifier, reporting, posture/profile, or scenario dependent until a follow-up phase proves otherwise.

The current likely root cause remains `inconclusive_contact_geometry_requires_pitch_profile_review`. Before changing foot geometry, reconcile Phase 2C contact classification/profile behavior with the G1 raw contact evidence.

### Next Decision

Do not claim walking success. Do not start PPO from this alone. First run Pre-W0 actuator semantics, free-space joint sign validation, and a consolidated DuckRef readiness check.

## 2026-06-01 - Phase Pre-W0 DuckRef Readiness Audit

### Goal

Decide whether Sedon is ready for a conservative Phase W0-DuckRef scripted walking smoke test, using M0, G1, actuator semantics, and free-space joint sign evidence.

### Files

- `tools/sedon_phase_pre_w0_actuator_semantics_audit.py`
- `tools/sedon_phase_pre_w0_free_space_joint_sign_validation.py`
- `tools/sedon_phase_pre_w0_duckref_readiness_check.py`
- `artifacts/sedon_debug/phase_pre_w0_actuator_semantics_audit/sedon_actuator_semantics_inventory.csv`
- `artifacts/sedon_debug/phase_pre_w0_actuator_semantics_audit/sedon_actuator_semantics_summary.json`
- `artifacts/sedon_debug/phase_pre_w0_actuator_semantics_audit/sedon_actuator_semantics_report.md`
- `artifacts/sedon_debug/phase_pre_w0_free_space_joint_sign_validation/sedon_free_space_joint_effect_probe.csv`
- `artifacts/sedon_debug/phase_pre_w0_free_space_joint_sign_validation/sedon_free_space_joint_sign_mapping.json`
- `artifacts/sedon_debug/phase_pre_w0_free_space_joint_sign_validation/sedon_free_space_joint_sign_validation_report.md`
- `artifacts/sedon_debug/phase_pre_w0_duckref_readiness_check/phase_pre_w0_duckref_readiness_summary.json`
- `artifacts/sedon_debug/phase_pre_w0_duckref_readiness_check/phase_pre_w0_duckref_readiness_report.md`
- `docs/sedon_blue_like_dynamic_gait_progress_log.md`

### Commands

```text
python -m py_compile tools/sedon_phase_pre_w0_actuator_semantics_audit.py
python -m py_compile tools/sedon_phase_pre_w0_free_space_joint_sign_validation.py
python -m py_compile tools/sedon_phase_pre_w0_duckref_readiness_check.py
.venv\Scripts\python.exe -m tools.sedon_phase_pre_w0_actuator_semantics_audit
.venv\Scripts\python.exe -m tools.sedon_phase_pre_w0_free_space_joint_sign_validation
.venv\Scripts\python.exe -m tools.sedon_phase_pre_w0_duckref_readiness_check
```

### Key Metrics

| Metric | Value |
| --- | ---: |
| actuator total_actuators | 10 |
| actuator motor_like_count | 10 |
| actuator unknown_like_count | 0 |
| duck_action_scale_025_transfer_safe | inconclusive |
| free_space_joint_sign_validated | true |
| m0_vs_free_space_sign_disagreements_count | 0 |
| center_contact_physically_possible | true |
| contact_classifier_unknown_force_ratio | 0.243728 |
| split_contact_style_risk | true |
| scaled_duckref_too_aggressive_risk | true |

### Result

READY_FOR_W0_DUCKREF_CONSERVATIVE

### Engineering Interpretation

Sedon actuator metadata is clear enough for W0: all 10 actuators classify as motor-like and no unknown actuator semantics remain. Free-space joint sign validation agrees with the M0 mapping for all 10 validated leg joints. G1 confirms right-center contact is physically possible.

This is not a general Duck action transfer approval. `duck_action_scale_025_transfer_safe` remains inconclusive, split contact style remains a risk, and the scaled DuckRef target is too aggressive for a first smoke test. W0 should therefore use only a conservative scripted reference grid:

- target_vx: `0.05, 0.08, 0.10`
- gait_period: `0.70, 0.85`
- clearance: `0.005, 0.015, 0.025`
- action_scale: `0.10, 0.15, 0.20`

### Next Decision

Run Phase W0-DuckRef conservative scripted walking smoke test only. Do not recommend PPO yet unless W0 produces periodic foot advancement and partial unload.

## 2026-06-01 - Phase W0-DuckRef Conservative Scripted Walking Smoke Test

### Goal

Run a conservative scripted W0 smoke test after Pre-W0 readiness, without training PPO or claiming Blue-like dynamic gait success.

### Files

- `tools/sedon_phase_w0_duckref_conservative_smoke_test.py`
- `artifacts/sedon_debug/phase_w0_duckref_conservative_smoke_test/phase_w0_duckref_trials.csv`
- `artifacts/sedon_debug/phase_w0_duckref_conservative_smoke_test/phase_w0_duckref_best_timeline.csv`
- `artifacts/sedon_debug/phase_w0_duckref_conservative_smoke_test/phase_w0_duckref_summary.json`
- `artifacts/sedon_debug/phase_w0_duckref_conservative_smoke_test/phase_w0_duckref_report.md`

### Commands

```text
python -m py_compile tools/sedon_phase_w0_duckref_conservative_smoke_test.py
.venv\Scripts\python.exe -m tools.sedon_phase_w0_duckref_conservative_smoke_test
```

### Key Metrics

| Metric | Value |
| --- | ---: |
| total_trials | 21 |
| best_trial_id | 9 |
| delta_base_x | 0.133374 |
| average_forward_velocity | 0.066754 |
| contact_switch_count | 124 |
| single_support_ratio | 0.080000 |
| support_force_ratio_peak | 1.000000 |
| action_saturation_ratio | 0.000000 |
| gait_candidate_label | POSTURE_INSTABILITY |

### Result

FAIL

### Engineering Interpretation

Best trial classified as POSTURE_INSTABILITY with failure mode fall_or_large_tilt. The result is a scripted diagnostic only and must not be treated as a learned gait.

This is not Blue-like dynamic gait success.

### Next Decision

Tune posture stability and conservative amplitude before any PPO.

## 2026-06-02 - Sedon v5_22 Toe Handoff Targeted Probe

### Goal

Run a bounded, controlled posture sweep on Sedon v5_22 foot prototype variants to check whether center-to-toe load transfer is physically observable before controller gait sequencing or PPO.

### Files

- `tools/sedon/diagnostics/v5_22/run_sedon_v5_22_toe_handoff_probe.py`
- `configs/sedon/sedon_v5_22_toe_handoff_probe.yaml`
- `docs/sedon_v5_22_toe_handoff_probe_report.md`
- `artifacts/sedon_debug/v5_22_toe_handoff_probe/probe_results.csv`
- `artifacts/sedon_debug/v5_22_toe_handoff_probe/metrics.json`
- `artifacts/sedon_debug/v5_22_toe_handoff_probe/raw_contacts.csv`

### Commands

```text
.venv\Scripts\python.exe -B -m py_compile tools\sedon\diagnostics\v5_22\run_sedon_v5_22_toe_handoff_probe.py
.venv\Scripts\python.exe -B tools\sedon\diagnostics\v5_22\run_sedon_v5_22_toe_handoff_probe.py
```

### Key Metrics

| Metric | Value |
| --- | ---: |
| probe_rows | 36 |
| toe_handoff_candidate_found | false |
| candidate_count | 0 |
| best_foot_variant | duck_like_multi_patch |
| best_posture_case | ankle_toe_down_bias |
| best_actuator_profile | ankle_boost_hypothesis |
| best_toe_force_ratio | 0.928829 |
| best_center_force_ratio | 0.071171 |
| best_heel_force_ratio | 0.000000 |
| best_contact_none_rate | 0.850000 |
| best_result_label | insufficient_contact_persistence |

### Result

NO_TOE_HANDOFF_CANDIDATE

### Engineering Interpretation

The targeted probe can create high toe force ratios under ankle toe-down bias and ankle boost, but those rows do not pass the prototype toe handoff rule because contact persistence is poor. The best row has `contact_none_rate=0.85`, so the observed toe loading is intermittent and should not be treated as a stable toe handoff.

MuJoCo contact force is read from raw contact normal force via `mj_contactForce`; patch attribution depends on prototype geom names and is not a verified physical sensor. This remains bounded diagnostic only, not walking success and not sim2real evidence.

### Next Decision

Prioritize foot geometry tuning and contact persistence before controller gait sequencing. Do not PPO.
