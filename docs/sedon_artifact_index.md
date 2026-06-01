# Sedon Artifact Index

Inventory/index only. No artifacts were moved, deleted, renamed, or regenerated.

Source scan:

- `artifacts/sedon_debug/*`
- `docs/sedon_blue_like_dynamic_gait_progress_log.md`
- `tools/registry.py`
- `tools/README.md`

## Classification

- Task class: Class C, experiment artifact traceability.
- Engineering depth: folder-level index with known blocker relevance.
- Trade-off: loose CSV/JSON/XML files under `artifacts/sedon_debug/` are grouped by family after the folder table, because the requested primary unit is artifact folder.
- State safety: static documentation only.

## Artifact Folder Index

| Folder path | Source tool | Related phase | Key outputs | Known result | Keep reason | Archive candidate or not |
|---|---|---|---|---|---|---|
| `artifacts/sedon_debug/blue_forward_phase_timing_refine_v1/` | `tools/blue_forward_phase_timing_refine_v1.py` | grounded forward shuffle refinement | CSV, `summary.md` | historical refinement; not current blocker | traces forward-shaping timing decisions | Archive candidate |
| `artifacts/sedon_debug/blue_forward_shuffle_authority_sweep_v1/` | `tools/blue_forward_shuffle_authority_sweep_v1.py` | grounded forward shuffle authority | CSV, `summary.md` | historical sweep | documents forward authority limits | Archive candidate |
| `artifacts/sedon_debug/blue_forward_shuffle_v1/` | `tools/blue_forward_shuffle_v1.py` | grounded forward shuffle baseline | CSV, `summary.md` | baseline only | prevents mistaking shuffle for walking | Archive candidate |
| `artifacts/sedon_debug/blue_like_phase1_rollover/` | `tools/sedon_blue_like_phase1_rollover_diagnostic.py` | Phase 1 rollover | `phase1_rollover_timeline.csv`, summary JSON | FAIL: uncontrolled fall/no-contact/upright/toe_handoff | key negative result before capture-step work | Keep |
| `artifacts/sedon_debug/blue_like_sole_experiments/` | `tools/generate_sedon_blue_like_sole_variant.py`, `tools/validate_sedon_blue_like_sole.py` | sole geometry experiments | generated scenes/reports | historical geometry set | trace geometry evolution | Archive candidate |
| `artifacts/sedon_debug/blue_like_sole_experiments_v3/` | same as above | sole geometry v3 | generated scenes/reports | historical | trace geometry evolution | Archive candidate |
| `artifacts/sedon_debug/blue_like_sole_experiments_v4/` | same as above | sole geometry v4 | generated scenes/reports | historical | trace geometry evolution | Archive candidate |
| `artifacts/sedon_debug/blue_like_sole_experiments_v5/` | same as above | Phase 0 v5_a geometry | `training_scene_v5_a.xml` and validation outputs | best known grounded geometry baseline; not walking | important geometry baseline | Keep |
| `artifacts/sedon_debug/blue_step_reference_v1/` | `tools/build_blue_step_reference_v1.py` | Blue step reference v1 | candidate configs/CSV | historical reference generation | records failed/early reference attempt | Archive candidate |
| `artifacts/sedon_debug/blue_step_reference_v2/` | `tools/build_blue_step_reference_v2.py` | Blue step reference v2 | candidate configs/CSV | historical reference generation | records unload-before-lift attempt | Archive candidate |
| `artifacts/sedon_debug/blue_step_reference_v3_closed_loop/` | `tools/build_blue_step_reference_v3_closed_loop.py` | Blue step reference v3 | candidate configs/CSV/summary | historical trigger-search reference | records closed-loop trigger attempt | Archive candidate |
| `artifacts/sedon_debug/blue_unload_mechanism_search/` | `tools/blue_unload_mechanism_search.py` | unload-only search | CSV/summary/candidates | historical | documents unload-only mechanism search | Archive candidate |
| `artifacts/sedon_debug/blue_unload_refine_v2/` | `tools/blue_unload_refine_v2.py` | unload refine v2 | CSV/summary | historical | documents refined unload attempts | Archive candidate |
| `artifacts/sedon_debug/capture_step_controller_v1/` | `tools/sedon_capture_step_controller_v1.py` | capture controller v1 | `capture_step_controller_v1.csv`, summary JSON | FAIL as dynamic gait; safe but too conservative | key negative result | Keep |
| `artifacts/sedon_debug/controller_v2/` | `tools/sedon_explicit_locomotion_controller_v2.py` | explicit locomotion v2 | timeline/summary | experimental | controller traceability | Keep |
| `artifacts/sedon_debug/controller_v2a_unload/` | `tools/sedon_unload_controller_v2a.py` | unload controller v2a | CSV/summary | experimental | unload controller traceability | Keep |
| `artifacts/sedon_debug/explicit_contact_force_lift_controller_v1/` | `tools/explicit_contact_force_lift_controller_v1.py` | force-gated lift | CSV/summary | historical | evidence for force-gated lift limits | Archive candidate |
| `artifacts/sedon_debug/hip_roll_model_validation/` | hip-roll debug tools | hip-roll validation | validation CSVs/reports | historical/unknown | keeps actuator/contact model evidence | Review |
| `artifacts/sedon_debug/ik_lift_dynamic_validation_v1/` | `tools/ik_lift_dynamic_validation_v1.py` | IK lift validation | CSV/summary | historical | evidence for IK mapping under dynamics | Archive candidate |
| `artifacts/sedon_debug/kinematic_foot_jacobian_diagnostic_v1/` | `tools/kinematic_foot_jacobian_diagnostic_v1.py` | kinematic foot authority | CSV/summary | historical | proves/limits kinematic authority | Archive candidate |
| `artifacts/sedon_debug/lift_after_unload_v1/` | `tools/lift_after_unload_v1.py` | lift after unload | CSV/summary | historical | lift-after-unload trace | Archive candidate |
| `artifacts/sedon_debug/lift_execution_audit_v1/` | `tools/lift_execution_audit_v1.py` | lift execution audit | CSV/summary | historical; contact constraint was a concern | blocker-adjacent evidence | Keep |
| `artifacts/sedon_debug/micro_lift_gain_calibration_v1/` | `tools/micro_lift_gain_calibration_v1.py` | micro-lift calibration | CSV/summary | historical | documents gain cliffs/limits | Archive candidate |
| `artifacts/sedon_debug/micro_lift_mapping_search_v1/` | `tools/micro_lift_mapping_search_v1.py` | micro-lift mapping | CSV/summary | historical | documents mapping search | Archive candidate |
| `artifacts/sedon_debug/mjcf_source/` | `tools/convert_urdf_to_mjcf.py` | scene conversion | MJCF source/debug outputs | setup artifact | needed to trace generated training scene | Keep |
| `artifacts/sedon_debug/phase1_5_force_split_rollover/` | `tools/sedon_phase1_5_force_split_rollover_controller.py` | Phase 1.5 | timeline CSV, summary JSON | FAIL: support ratio gate not reached | key negative force-split evidence | Keep |
| `artifacts/sedon_debug/phase1_5_force_split_rollover_probe_roll060/` | same | Phase 1.5 probe | probe CSV/summary | historical probe | parameter trace | Archive candidate |
| `artifacts/sedon_debug/phase1_5_force_split_rollover_probe_roll065/` | same | Phase 1.5 probe | probe CSV/summary | historical probe | parameter trace | Archive candidate |
| `artifacts/sedon_debug/phase1_5_force_split_rollover_probe_roll070/` | same | Phase 1.5 probe | probe CSV/summary | historical probe | parameter trace | Archive candidate |
| `artifacts/sedon_debug/phase1_5_force_split_rollover_probe_roll075/` | same | Phase 1.5 probe | probe CSV/summary | historical probe | parameter trace | Archive candidate |
| `artifacts/sedon_debug/phase1_5_force_split_rollover_probe_roll075_fwd3/` | same | Phase 1.5 probe | probe CSV/summary | historical probe | parameter trace | Archive candidate |
| `artifacts/sedon_debug/phase1_5_force_split_rollover_probe_roll095/` | same | Phase 1.5 probe | probe CSV/summary | historical probe | parameter trace | Archive candidate |
| `artifacts/sedon_debug/phase1_6_load_transfer_attribution/` | `tools/sedon_phase1_6_load_transfer_attribution.py` | Phase 1.6 | trials/top candidates/summary JSON | PASS/INCONCLUSIVE history; load-transfer channels explored | important attribution evidence | Keep |
| `artifacts/sedon_debug/phase1_7_load_transfer_profile_shaping/` | `tools/sedon_phase1_7_load_transfer_profile_shaping.py` | Phase 1.7 | profile CSV/summary | PASS in progress log before later blockers | important bridge evidence | Keep |
| `artifacts/sedon_debug/phase1_7_load_transfer_profile_shaping_refine/` | same | Phase 1.7 refine | profile CSV/summary | historical refine | parameter trace | Archive candidate |
| `artifacts/sedon_debug/phase2a_right_support_micro_capture/` | `tools/sedon_phase2a_right_support_micro_capture.py` | Phase 2A | capture CSV/summary | FAIL | important negative micro-capture evidence | Keep |
| `artifacts/sedon_debug/phase2b_micro_capture_refinement/` | `tools/sedon_phase2b_micro_capture_refinement.py` | Phase 2B | refinement CSV/summary | FAIL | important negative micro-capture evidence | Keep |
| `artifacts/sedon_debug/phase2c_contact_constrained_foot_mapping/` | `tools/sedon_phase2c_contact_constrained_foot_mapping.py` | Phase 2C | foot mapping/contact CSV/summary | INCONCLUSIVE, contact constraints block forward motion in rows; right center force zero in tested profile | current blocker root evidence | Keep |
| `artifacts/sedon_debug/phase_g1_raw_contact_pair_diagnostic/` | `tools/sedon_phase_g1_raw_contact_pair_diagnostic.py` | Phase G1 | raw contact pairs, region summary, geom inventory, report, summary JSON | INCONCLUSIVE; right-center raw contact exists; classifier/profile issue remains | current blocker evidence | Keep |
| `artifacts/sedon_debug/phase_m0_duck_morphology_audit/` | `tools/sedon_phase_m0_duck_morphology_audit.py` | Phase M0 | topology/actuator/morphology/contact inventories, report, scaled reference | INCONCLUSIVE; Duck metrics useful, raw transfer blocked | current blocker/readiness evidence | Keep |
| `artifacts/sedon_debug/phase_pre_w0_actuator_semantics_audit/` | `tools/sedon_phase_pre_w0_actuator_semantics_audit.py` | Pre-W0 | actuator inventory, summary, report | actuator semantics clear enough | current readiness evidence | Keep |
| `artifacts/sedon_debug/phase_pre_w0_duckref_readiness_check/` | `tools/sedon_phase_pre_w0_duckref_readiness_check.py` | Pre-W0 | readiness summary/report | READY_FOR_W0_DUCKREF_CONSERVATIVE | key gate before W0 | Keep |
| `artifacts/sedon_debug/phase_pre_w0_free_space_joint_sign_validation/` | `tools/sedon_phase_pre_w0_free_space_joint_sign_validation.py` | Pre-W0 | joint effect probe, sign mapping, report | free-space sign validated | current readiness evidence | Keep |
| `artifacts/sedon_debug/phase_w0_duckref_conservative_smoke_test/` | `tools/sedon_phase_w0_duckref_conservative_smoke_test.py` | W0 DuckRef smoke | trials CSV, best timeline, summary, report | FAIL: POSTURE_INSTABILITY / fall_or_large_tilt | current blocker artifact | Keep |
| `artifacts/sedon_debug/reference_lift_sweep/` | `tools/reference_lift_sweep.py` | reference lift sweep | CSV/rendered top-k | historical | reference traceability | Archive candidate |
| `artifacts/sedon_debug/render/` | `tools/render_sedon_policy_comparison.py` | rendering/reporting | videos/images/CSV | reporting artifact | visual traceability | Keep |
| `artifacts/sedon_debug/rounded_sole_experiments/` | `tools/generate_sedon_foot_geometry_variants.py` | geometry variants | XML/validation outputs | historical | geometry traceability | Archive candidate |
| `artifacts/sedon_debug/soft_landing_refine_v1/` | `tools/soft_landing_refine_v1.py` | soft landing refine | CSV/summary | historical | landing refinement evidence | Archive candidate |
| `artifacts/sedon_debug/unload_authority_attribution_v1/` | `tools/unload_authority_attribution_v1.py` | unload authority | CSV/summary | experimental | unload channel traceability | Keep |
| `artifacts/sedon_debug/unload_controller_v2b_final_check/` | `tools/unload_controller_v2b_final_check.py` | unload v2b final check | CSV/summary | experimental; current authority may be insufficient per source text | controller authority evidence | Keep |

## Loose Artifact Families Under `artifacts/sedon_debug/`

| Artifact family | Likely source tool | Related phase | Known result / reason to keep |
|---|---|---|---|
| `autonomy_*`, `teacher_*` CSV/JSON/log files | `tools/audit_sedon_teacher_imitation.py`, `tools/autonomy_stage1_probe_report.py`, `tools/compare_sedon_teacher_checkpoints.py` | teacher/probe training audits | keep for policy audit traceability and checkpoint comparisons |
| `blue_*_preview.csv`, `preview_*`, `force_ratio_controller_preview.csv`, `hybrid_torque_balance_controller_preview.csv` | `preview_sedon_*`, `sweep_sedon_blue_contact_gated_targets.py` | controller preview | keep while controller design remains active |
| `forced_support_lift_check*`, `force_unload_controller*`, `single_support_*`, `target_tracking_*` | force/load/lift debug tools | load transfer and lift blockers | keep because they explain lift/unload authority limits |
| `mechanical_variant_*`, `geometry_sensitivity_sweep.csv`, `foot_*_sweep.csv`, `training_scene_*candidate*.xml` | geometry/mechanical sweeps | geometry sensitivity | keep until contact geometry decision is closed |
| `reference_march_*_gait_view.csv`, `sedon_reference_gait_seed.json`, `reference_teacher_*.json` | gait viewer/reference tools | reference motion | keep as reference seeds and visual audit inputs |
| `training_scene_debug_visible_geoms.xml`, `training_scene_long_narrow_foot.xml`, `training_scene_mech_best.xml`, `training_scene_physical_candidate*.xml` | geometry/build/debug tools | scene variants | do not use as canonical training scene without explicit decision; keep for traceability |

## Current Blocker Artifacts

Current blocker-related folders/files:

- `artifacts/sedon_debug/phase2c_contact_constrained_foot_mapping/`
- `artifacts/sedon_debug/phase_g1_raw_contact_pair_diagnostic/`
- `artifacts/sedon_debug/phase_m0_duck_morphology_audit/`
- `artifacts/sedon_debug/phase_pre_w0_actuator_semantics_audit/`
- `artifacts/sedon_debug/phase_pre_w0_free_space_joint_sign_validation/`
- `artifacts/sedon_debug/phase_pre_w0_duckref_readiness_check/`
- `artifacts/sedon_debug/phase_w0_duckref_conservative_smoke_test/`
- Supporting loose files from force/load/geometry diagnostics: `forced_support_lift_check*`, `force_unload_controller*`, `mechanical_variant_*`, `geometry_sensitivity_sweep.csv`, `foot_contact_geometry_sweep.csv`, `foot_proxy_redesign_sweep.csv`.

## Index Rules For Future Cleanup

- Do not delete historical phase folders until their key metrics are copied into a durable experiment registry.
- Do not promote any `training_scene_*.xml` in artifacts to canonical use without a separate scene decision.
- Keep all current blocker artifacts until W0 posture instability is resolved and a follow-up smoke test passes stability gates.
