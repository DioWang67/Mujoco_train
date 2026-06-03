# Seedon Experiment Registry

This registry links Seedon phases, source tools, artifacts, and decisions. It references `docs/seedon_blue_like_dynamic_gait_progress_log.md`; it does not replace or modify that append-only log.

## Classification

- Task class: Class C, experiment registry for research tools/artifacts.
- Engineering depth: concise phase-level registry, not a refactor plan.
- Trade-off: registry favors decision traceability over complete raw metric duplication.
- State safety: documentation only; no runtime or training state is modified.

## Current Decision Chain

| Phase | Source tool(s) | Artifact folder(s) | Result | Decision |
|---|---|---|---|---|
| Phase 0 v5_a | `generate_seedon_blue_like_sole_variant.py`, `validate_seedon_blue_like_sole.py` | `blue_like_sole_experiments_v5/` | Baseline only; not walking | Use v5_a as geometry baseline, do not call grounded shuffle walking. |
| Phase 1 rollover | `seedon_blue_like_phase1_rollover_diagnostic.py` | `blue_like_phase1_rollover/` | FAIL: contact-none/jump/upright/toe_handoff | Replace raw push with bounded commit and force-gated recovery. |
| Capture V1 | `seedon_capture_step_controller_v1.py` | `capture_step_controller_v1/` | FAIL as dynamic gait; stable but conservative | Build force-split/rollover bridge before full capture step. |
| Phase 1.5 | `seedon_phase1_5_force_split_rollover_controller.py` | `phase1_5_force_split_rollover*` | FAIL: support-force gate not held | Debug load-transfer authority and contact behavior. |
| Phase 1.6 | `seedon_phase1_6_load_transfer_attribution.py` | `phase1_6_load_transfer_attribution/` | load-transfer channels explored | Continue profile shaping and channel attribution. |
| Phase 1.7 | `seedon_phase1_7_load_transfer_profile_shaping.py` | `phase1_7_load_transfer_profile_shaping*` | partial bridge evidence | Proceed to micro-capture only cautiously. |
| Phase 2A | `seedon_phase2a_right_support_micro_capture.py` | `phase2a_right_support_micro_capture/` | FAIL | Refine micro-capture and contact constraints. |
| Phase 2B | `seedon_phase2b_micro_capture_refinement.py` | `phase2b_micro_capture_refinement/` | FAIL | Diagnose contact-constrained foot mapping. |
| Phase 2C | `seedon/diagnostics/contact/phase2c_contact_constrained_foot_mapping.py` with wrapper `seedon_phase2c_contact_constrained_foot_mapping.py` | `phase2c_contact_constrained_foot_mapping/` | INCONCLUSIVE; contact constraints block forward rows; right-center force was zero in tested profile | Run raw contact-pair geometry diagnostic before PPO. |
| Phase G1 | `seedon/diagnostics/contact/phase_g1_raw_contact_pair_diagnostic.py` with wrapper `seedon_phase_g1_raw_contact_pair_diagnostic.py` | `phase_g1_raw_contact_pair_diagnostic/` | INCONCLUSIVE; right-center raw contact exists | Reconcile classifier/profile behavior instead of immediately changing geometry. |
| Phase M0 | `seedon_phase_m0_duck_morphology_audit.py` | `phase_m0_duck_morphology_audit/` | INCONCLUSIVE; Duck gait metrics useful, raw transfer blocked | Use Duck only at gait-metric level. |
| Phase Pre-W0 actuator | `seedon_phase_pre_w0_actuator_semantics_audit.py` | `phase_pre_w0_actuator_semantics_audit/` | actuator semantics clear enough | Continue readiness chain. |
| Phase Pre-W0 sign | `seedon_phase_pre_w0_free_space_joint_sign_validation.py` | `phase_pre_w0_free_space_joint_sign_validation/` | joint sign validated | Continue readiness chain. |
| Phase Pre-W0 readiness | `seedon_phase_pre_w0_duckref_readiness_check.py` | `phase_pre_w0_duckref_readiness_check/` | READY_FOR_W0_DUCKREF_CONSERVATIVE | Run only conservative scripted W0; no PPO yet. |
| Phase W0 DuckRef smoke | `seedon_phase_w0_duckref_conservative_smoke_test.py` | `phase_w0_duckref_conservative_smoke_test/` | FAIL: POSTURE_INSTABILITY / fall_or_large_tilt | Tune posture stability and conservative amplitude before PPO. |

## Active Experiment Threads

| Thread | Active tools | Active artifacts | Current status |
|---|---|---|---|
| Contact semantics and patch classification | `seedon/diagnostics/contact/phase2c_contact_constrained_foot_mapping.py`, `seedon/diagnostics/contact/phase_g1_raw_contact_pair_diagnostic.py` | `phase2c_contact_constrained_foot_mapping/`, `phase_g1_raw_contact_pair_diagnostic/` | right-center contact exists in raw G1; classifier/profile mismatch remains unresolved |
| DuckRef conservative transfer | `seedon_phase_m0_duck_morphology_audit.py`, Pre-W0 tools, `seedon_phase_w0_duckref_conservative_smoke_test.py` | `phase_m0_*`, `phase_pre_w0_*`, `phase_w0_*` | W0 readiness passed, but scripted smoke failed due posture instability |
| Load transfer / unload authority | `debug_seedon_force_unload_controller.py`, `debug_seedon_forced_support_lift_check.py`, `unload_authority_attribution_v1.py`, `unload_controller_v2b_final_check.py` | `force_unload_controller*`, `forced_support_lift_check*`, `unload_*` folders | still relevant as blocker-adjacent evidence |
| Geometry sensitivity | `generate_seedon_foot_geometry_variants.py`, `geometry_sensitivity_sweep.py`, `debug_seedon_foot_contact_geometry_sweep.py`, `debug_seedon_foot_proxy_redesign_sweep.py` | `blue_like_sole_experiments_v5/`, `rounded_sole_experiments/`, geometry CSV/XML files | do not change canonical scene without explicit follow-up |
| Training/evaluation workflow | `build_seedon_training_scene.py`, `smoke_seedon_env.py`, `debug_seedon_gait_viewer.py`, `debug_seedon_gait_audit.py`, `seedon_eval.py` | viewer/audit CSVs, reports | available but PPO should wait for W0 stability decision |

## Known Blockers

| Blocker | Evidence | Current implication |
|---|---|---|
| W0 posture instability | `phase_w0_duckref_conservative_smoke_test/phase_w0_duckref_summary.json`, progress log | next work should tune posture stability/amplitude; do not start PPO based on W0 |
| Contact classifier/profile mismatch | Phase 2C vs G1 evidence | avoid geometry changes until raw-vs-derived contact interpretation is reconciled |
| Split contact style risk | M0 and Pre-W0 readiness reports | Duck gait metrics are useful, raw Duck action/joint transfer remains blocked |
| Load/unload authority limits | force unload, forced support lift, unload controller artifacts | informs controller design and reward gates |

## Registry Rules

- Use `docs/seedon_blue_like_dynamic_gait_progress_log.md` as the append-only detailed log.
- Use this file as the phase-level map for tools and artifacts.
- Do not mark a phase as successful based on forward displacement alone.
- Require contact, posture, support ratio, swing unload, and no-flight/no-jump evidence before promoting a gait candidate.
