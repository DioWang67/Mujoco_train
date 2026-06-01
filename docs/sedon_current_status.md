# Sedon Current Status

Snapshot date: 2026-06-01.

This is an inventory/status snapshot only. It does not change training logic, scenes, tools, or artifacts.

## Classification

- Task class: Class C, project experiment/status documentation.
- Engineering depth: concise current-state summary for traceability.
- Trade-off: this file points to source logs and artifact folders instead of duplicating every metric.
- State safety: no shared state or files outside the four new docs are modified.

## Current Status Summary

Sedon is still in a locomotion research/debug phase. The latest scanned progress log says the conservative W0 DuckRef scripted smoke test failed with `POSTURE_INSTABILITY` / `fall_or_large_tilt`. That means the project is not ready to treat the DuckRef W0 path as a PPO training target yet.

The strongest current conclusion is:

- actuator semantics are clear enough for conservative W0;
- free-space joint sign mapping is validated;
- raw right-center contact can exist and carry force;
- split contact semantics and profile/classifier behavior still require care;
- W0 conservative scripted gait did not pass posture stability.

## Active Tools

| Area | Active tools |
|---|---|
| Setup/check | `tools/preflight_check.py`, `tools/project_inventory.py`, `tools/build_sedon_training_scene.py`, `tools/smoke_sedon_env.py`, `tools/verify_sedon_static_seed.py` |
| Remote operation | `tools/remote_auto_deploy.py`, `tools/remote_training.py` |
| Evaluation/view/audit | `tools/sedon_eval.py`, `tools/debug_sedon_gait_viewer.py`, `tools/debug_sedon_gait_audit.py`, `tools/debug_sedon_pose_editor.py`, `tools/preview_sedon_gait.py` |
| Current blocker chain | `tools/sedon_phase2c_contact_constrained_foot_mapping.py`, `tools/sedon_phase_g1_raw_contact_pair_diagnostic.py`, `tools/sedon_phase_m0_duck_morphology_audit.py`, `tools/sedon_phase_pre_w0_actuator_semantics_audit.py`, `tools/sedon_phase_pre_w0_free_space_joint_sign_validation.py`, `tools/sedon_phase_pre_w0_duckref_readiness_check.py`, `tools/sedon_phase_w0_duckref_conservative_smoke_test.py` |
| March gates | `tools/check_sedon_march_phase1a_gates.py`, `tools/check_sedon_march_phase12_gates.py`, `tools/sweep_sedon_phase1a_hip_roll_assist.py` |

## Deprecated / Unknown Tools

No explicitly deprecated or broken tool was found in the scanned docs/registry.

Tools that need owner/status review:

- `tools/audit_sedon_grounded_reference_pd.py`
- `tools/debug_sedon_single_support_sweep.py`
- `tools/debug_sedon_static_tilt_handoff.py`
- `tools/inspect_foot_geom_z.py`
- `tools/open_duck_viewer_from_h1.py`
- `tools/sweep_sedon_dynamic_fsm.py`
- `tools/sweep_sedon_fsm_1600.py`

Historical phase tools should be treated as archived traceability assets, not deleted:

- `tools/sedon_blue_like_phase1_rollover_diagnostic.py`
- `tools/sedon_capture_step_controller_v1.py`
- `tools/sedon_phase1_5_force_split_rollover_controller.py`
- `tools/sedon_phase1_6_load_transfer_attribution.py`
- `tools/sedon_phase1_7_load_transfer_profile_shaping.py`
- `tools/sedon_phase2a_right_support_micro_capture.py`
- `tools/sedon_phase2b_micro_capture_refinement.py`
- Blue/reference/micro-lift generation and refinement tools listed in `docs/sedon_tool_inventory.md`.

## Current Blocker Artifacts

| Artifact | Why it matters |
|---|---|
| `artifacts/sedon_debug/phase_w0_duckref_conservative_smoke_test/` | latest W0 result; failed due posture instability |
| `artifacts/sedon_debug/phase_pre_w0_duckref_readiness_check/` | documents why W0 conservative smoke was allowed |
| `artifacts/sedon_debug/phase_pre_w0_actuator_semantics_audit/` | actuator semantics evidence |
| `artifacts/sedon_debug/phase_pre_w0_free_space_joint_sign_validation/` | joint sign evidence |
| `artifacts/sedon_debug/phase_g1_raw_contact_pair_diagnostic/` | raw contact-pair evidence; right-center contact exists |
| `artifacts/sedon_debug/phase2c_contact_constrained_foot_mapping/` | earlier contact-constrained mapping issue that triggered G1 |
| `artifacts/sedon_debug/phase_m0_duck_morphology_audit/` | explains why DuckRef is gait-metric-level only |

## Next Inventory / Cleanup Recommendations

1. Keep the four new docs as the current traceability entry points:
   - `docs/sedon_tool_inventory.md`
   - `docs/sedon_artifact_index.md`
   - `docs/sedon_experiment_registry.md`
   - `docs/sedon_current_status.md`
2. Do not archive or delete historical phase artifacts yet. First confirm that every historical phase has one registry row with source tool, artifact folder, result, and decision.
3. Add a lightweight "owner/status" pass for tools currently marked `unknown`.
4. If cleanup is desired later, do it in a separate commit that only moves files after updating references. This inventory pass intentionally does not move anything.
5. Before PPO training, resolve W0 posture stability or run a narrower scripted smoke that proves stable posture under conservative amplitude.

## References

- `docs/sedon_blue_like_dynamic_gait_progress_log.md`
- `docs/sedon_experiment_registry.md`
- `docs/sedon_artifact_index.md`
- `docs/sedon_tool_inventory.md`
- `docs/SEDON_WORKFLOW.md`
