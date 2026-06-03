# Seedon Tool Inventory

Generated for inventory/index only. This file does not move, archive, delete, or refactor any tool.

Source scan:

- `tools/*.py`
- `tools/registry.py`
- `tools/README.md`
- `docs/SEEDON_WORKFLOW.md`
- `docs/seedon_blue_like_dynamic_gait_progress_log.md`
- `artifacts/seedon_debug/*`

## Classification

- Task class: Class C, script/tool/experiment inventory.
- Engineering depth: documentation-first traceability; no training logic changes.
- Trade-off: status labels are conservative and evidence-based. When a tool is not in `tools/registry.py` and has no clear current workflow reference, it is marked `unknown` or `archived` instead of guessing.
- State safety: no shared runtime state is modified. This is a static index.

## Status Legend

| Status | Meaning |
|---|---|
| `active` | Current Seedon workflow or current blocker investigation depends on it. |
| `stable` | General project/operator tool with documented normal use. |
| `experimental` | Research probe, sweep, controller, or one-off diagnostic. |
| `archived` | Historical phase output/tool retained for traceability; not current next action. |
| `deprecated` | Explicitly superseded or should not be used for new work. |
| `broken` | Known unusable from scanned docs. |
| `unknown` | Not enough evidence from scanned docs/registry/artifacts. |

## Tool Inventory

| Tool path | Status | Category | Purpose | Inputs | Outputs | Related artifacts | Related docs | Keep / Review / Archive |
|---|---|---|---|---|---|---|---|---|
| `tools/agent_workspace.py` | stable | checks | Build sanitized source-only debug workspace. | source tree, name, output root | manifest and copied source subset | `artifacts/agent_workspace/` | `tools/README.md` | Keep |
| `tools/aggregate_compare.py` | stable | evaluation | Multi-seed H1 comparison. | reports/checkpoints, eval args | JSON/CSV reports | `reports/` | `tools/README.md` | Keep |
| `tools/assisted_shuffle_curriculum_eval.py` | experimental | Seedon diagnostic | Evaluate deterministic shuffle with reduced lateral assist. | Seedon env/config/checkpoint options | CSV summary | `artifacts/seedon_debug/assisted_shuffle_curriculum_eval.csv` | `tools/registry.py` | Keep |
| `tools/audit_seedon_grounded_reference_pd.py` | unknown | Seedon audit | Audit grounded reference PD behavior. | Seedon reference/config | console/summary metrics | none found by folder name | none explicit | Review |
| `tools/audit_seedon_shuffle_v0.py` | experimental | Seedon diagnostic | Audit low-threshold shuffle v0 curriculum. | `configs/seedon/shuffle_v0.json`, optional checkpoint | printed audit summary | shuffle/autonomy CSVs | `tools/registry.py` | Keep |
| `tools/audit_seedon_teacher_imitation.py` | experimental | Seedon diagnostic | Audit teacher-imitation policy without lateral assist. | model, vecnorm, teacher config | printed audit/gate summary | `teacher_*`, `autonomy_*` files | `tools/registry.py` | Keep |
| `tools/autonomy_stage1_probe_report.py` | experimental | Seedon reporting | Build autonomy stage1 probe report rows. | audit summaries/CSV inputs | CSV report | `autonomy_stage1_*`, `autonomy_probe_*` | artifact names | Keep |
| `tools/benchmark_matrix.py` | stable | evaluation | Run H1 benchmark matrix. | benchmark matrix config | JSON/CSV report | `reports/` | `tools/README.md` | Keep |
| `tools/blue_forward_phase_timing_refine_v1.py` | experimental | Seedon phase | Refine grounded shuffle forward timing. | config/reference, sweep args | CSV/summary.md | `artifacts/seedon_debug/blue_forward_phase_timing_refine_v1/` | progress log | Keep |
| `tools/blue_forward_shuffle_authority_sweep_v1.py` | experimental | Seedon phase | Sweep forward-shaping authority for grounded shuffle. | config/reference, sweep args | CSV/summary.md | `artifacts/seedon_debug/blue_forward_shuffle_authority_sweep_v1/` | progress log | Keep |
| `tools/blue_forward_shuffle_v1.py` | archived | Seedon phase | Audit tiny forward-drift teacher shaping baseline. | config/reference | CSV/summary.md | `artifacts/seedon_debug/blue_forward_shuffle_v1/` | progress log | Archive candidate |
| `tools/blue_unload_mechanism_search.py` | archived | Seedon phase | Search unload-only Blue-like reference mechanisms. | candidate grid | CSV/summary.md/configs | `artifacts/seedon_debug/blue_unload_mechanism_search/` | progress log | Archive candidate |
| `tools/blue_unload_refine_v2.py` | archived | Seedon phase | Refine top unload-only candidates. | prior candidates | CSV/summary.md/configs | `artifacts/seedon_debug/blue_unload_refine_v2/` | progress log | Archive candidate |
| `tools/build_blue_step_reference_v1.py` | archived | Seedon reference | Build initial Blue-like step reference family. | sweep args | candidate configs/CSV | `artifacts/seedon_debug/blue_step_reference_v1/` | progress log | Archive candidate |
| `tools/build_blue_step_reference_v2.py` | archived | Seedon reference | Build unload-before-lift Blue-like references. | sweep args | candidate configs/CSV | `artifacts/seedon_debug/blue_step_reference_v2/` | progress log | Archive candidate |
| `tools/build_blue_step_reference_v3_closed_loop.py` | archived | Seedon reference | Build references using dynamic unload trigger search. | sweep args | candidate configs/CSV/summary | `artifacts/seedon_debug/blue_step_reference_v3_closed_loop/` | progress log | Archive candidate |
| `tools/build_seedon_training_scene.py` | active | Seedon setup | Build floating-base Seedon training scene from converted MJCF. | private Seedon MJCF/URDF | `private_assets/seedon/training_scene.xml` | `artifacts/seedon_debug/mjcf_source/` when debugged | `tools/README.md`, `docs/SEEDON_WORKFLOW.md` | Keep |
| `tools/check_seedon_march_phase12_gates.py` | active | Seedon gate | Check Phase 1/2 residual/support gates. | model, vecnorm, thresholds | printed PASS/FAIL | phase/gate CSVs | `tools/registry.py` | Keep |
| `tools/check_seedon_march_phase1a_gates.py` | active | Seedon gate | Check Phase 1A hip-roll residual gates. | model, vecnorm, thresholds | printed PASS/FAIL | `phase1a_hip_roll_assist_sweep.csv` | `tools/registry.py` | Keep |
| `tools/compare_eval.py` | stable | evaluation | Compare base and DR H1 policies. | checkpoints/eval args | JSON/CSV report | `reports/` | `tools/README.md` | Keep |
| `tools/compare_seedon_teacher_checkpoints.py` | experimental | Seedon audit | Compare Seedon teacher-imitation checkpoints and impact windows. | checkpoint specs | summary/timeseries CSVs | `teacher_imitation_25k_vs_50k_*` | artifact names | Keep |
| `tools/convert_urdf_to_mjcf.py` | stable | Seedon setup | Convert private Seedon URDF/STL package to MuJoCo MJCF. | private URDF/meshes | MJCF source scene | `private_assets/seedon/mjcf_source/` | `tools/README.md` | Keep |
| `tools/debug_seedon_com_feedback_balance_sweep.py` | experimental | Seedon diagnostic | Sweep COM-feedback hip-roll balance controller. | scene/controller grid | CSV | `com_feedback_balance_sweep.csv` | `tools/README.md` | Review |
| `tools/debug_seedon_com_mass_distribution_sweep.py` | experimental | Seedon diagnostic | Sweep mass distribution and stance width. | runtime mass/stance grid | CSV | `com_mass_distribution_sweep.csv` | `tools/README.md` | Review |
| `tools/seedon/diagnostics/contact/debug_seedon_contacts.py` | stable | Seedon contact diagnostic | Inspect foot/base proxy geom placement and reset contacts. Legacy wrapper remains at `tools/debug_seedon_contacts.py`. | Seedon scene | console diagnostics | contact-related CSVs | `tools/README.md`, `docs/seedon_tools_migration_note.md` | Keep |
| `tools/debug_seedon_contact_initialization_sweep.py` | experimental | Seedon diagnostic | Sweep reset base-height contact initialization. | base-height offsets | CSV | `contact_initialization_sweep.csv` | `tools/README.md` | Review |
| `tools/debug_seedon_dynamic_push.py` | archived | Seedon diagnostic | Push/rollover diagnostic for dynamic commit. | scene, push speeds | CSV/summary table | dynamic push artifacts if present | progress log context | Archive candidate |
| `tools/debug_seedon_foot_contact_geometry_sweep.py` | experimental | Seedon diagnostic | Sweep foot box/friction/reset height. | scene, geometry/friction grid | CSV | `foot_contact_geometry_sweep.csv` | `tools/README.md` | Review |
| `tools/debug_seedon_foot_proxy_redesign_sweep.py` | experimental | Seedon diagnostic | Test temporary foot proxy variants. | proxy geometry grid | CSV | `foot_proxy_redesign_sweep.csv`, `rounded_sole_experiments/` | `tools/README.md` | Review |
| `tools/debug_seedon_forced_support_lift_check.py` | experimental | Seedon blocker diagnostic | Apply lateral force and test swing-foot lift. | lateral force grid | CSV | `forced_support_lift_check*` | `tools/README.md` | Keep |
| `tools/debug_seedon_force_unload_controller.py` | experimental | Seedon blocker diagnostic | Closed-loop force-unload proof. | support side, force gains | CSV | `force_unload_controller*.csv` | `tools/README.md` | Keep |
| `tools/debug_seedon_gait_audit.py` | active | Seedon audit | Audit contacts, support phases, micro-lift, COM, effort. | mode, checkpoint, scene, steps | CSV audit | `gait_audit_*`, policy audit CSVs | `docs/SEEDON_WORKFLOW.md` | Keep |
| `tools/debug_seedon_gait_viewer.py` | active | Seedon viewer | Play zero/scripted/policy rollouts with viewer and gait metrics. | mode, checkpoint, seed, scene | viewer, CSV | `gait_viewer*.csv`, `reference_*_gait_view.csv` | `docs/SEEDON_WORKFLOW.md` | Keep |
| `tools/debug_seedon_hip_roll_control_isolation.py` | experimental | Seedon diagnostic | Compare hip-roll tracking across contact modes. | hip-roll offsets | CSV | `hip_roll_control_isolation.csv` | `tools/README.md` | Review |
| `tools/debug_seedon_hip_roll_force_breakdown.py` | experimental | Seedon diagnostic | Decompose hip-roll generalized forces. | support side, offset | CSV | `hip_roll_force_breakdown.csv`, `hip_roll_contact_row_breakdown.csv` | `tools/README.md` | Review |
| `tools/debug_seedon_hip_roll_lateral_response.py` | experimental | Seedon diagnostic | Measure base/COM response to hip-roll targets. | offsets | CSV/report | `hip_roll_lateral_response.csv` | `tools/README.md` | Review |
| `tools/debug_seedon_joint_ranges.py` | stable | Seedon diagnostic | Compare URDF/prepared/MuJoCo joint ranges. | URDF/MJCF paths | CSV/console | `joint_range_compare.csv` | `tools/README.md` | Keep |
| `tools/debug_seedon_kinematic_foot_map.py` | stable | Seedon diagnostic | Kinematic foot clearance map without contact dynamics. | joint sweeps | CSV | `kinematic_foot_height_map.csv` | `tools/README.md` | Keep |
| `tools/debug_seedon_knee_direction.py` | stable | Seedon diagnostic | Infer knee bend direction and safe ranges. | offsets | console diagnostics | `knee_direction_diagnostic.csv` | `tools/README.md` | Keep |
| `tools/debug_seedon_lateral_controllability.py` | experimental | Seedon diagnostic | Measure lateral motion and contact ratio under support-roll/unload targets. | unload scales | CSV | `lateral_controllability.csv` | `tools/README.md` | Review |
| `tools/debug_seedon_lift_directions.py` | stable | Seedon diagnostic | Sweep swing-leg direction signs and clearance. | scene, relaxed-foot option | CSV | `lift_direction_sweep*.csv` | `tools/README.md` | Keep |
| `tools/debug_seedon_mechanical_variant_sweep.py` | experimental | Seedon diagnostic | Sweep temporary foot/COM/stance variants. | geometry/stance grid | CSV | `mechanical_variant_*` | `tools/README.md` | Review |
| `tools/debug_seedon_pd_load_transfer_sweep.py` | experimental | Seedon diagnostic | Sweep PD stiffness/damping under floor contact. | kp/kd/roll grid | CSV | `pd_load_transfer_sweep.csv` | `tools/README.md` | Review |
| `tools/debug_seedon_pose_editor.py` | active | Seedon reference | Interactive pose editor, save/load poses, support diagnostics. | scene, pose files | viewer, pose JSON | `seedon_pose_editor_poses.json`, pose seed JSONs | `docs/SEEDON_WORKFLOW.md` | Keep |
| `tools/debug_seedon_single_support_load_transfer.py` | experimental | Seedon blocker diagnostic | Gate COM/load transfer before tiny swing lift. | support side, load/lift steps | CSV | `single_support_load_transfer*.csv` | `tools/README.md` | Keep |
| `tools/debug_seedon_single_support_sweep.py` | unknown | Seedon diagnostic | Sweep single-support conditions. | sweep args | CSV | `single_support_hold_sweep.csv` | artifact name only | Review |
| `tools/debug_seedon_static_stability.py` | stable | Seedon diagnostic | Check COM placement/support boxes/static support attempts. | plan, relaxed-foot option | CSV/console | `static_stability_summary.csv` | `tools/README.md` | Keep |
| `tools/debug_seedon_static_tilt_handoff.py` | unknown | Seedon diagnostic | Static tilt handoff diagnostic. | scene/controller args | unknown | none found by folder name | none explicit | Review |
| `tools/debug_seedon_target_tracking.py` | stable | Seedon diagnostic | Compare target angles vs actual qpos. | mode, relaxed-foot option | CSV | `target_tracking_*.csv` | `tools/README.md` | Keep |
| `tools/debug_seedon_viewer.py` | stable | Seedon viewer | Open MuJoCo viewer with foot/base proxy coloring. | scene path | viewer | none | `tools/README.md` | Keep |
| `tools/deploy_release.py` | stable | release | Build/upload clean or worktree source archive. | git ref/worktree, remote config | tar.gz release archive | `artifacts/releases/` | `docs/REMOTE_LAYOUT.md`, `tools/README.md` | Keep |
| `tools/download_cuda_deps.py` | stable | maintenance | Download CUDA runtime packages for repair. | package settings | downloaded wheels/assets | local dependency cache | `tools/README.md` | Keep |
| `tools/download_missing.py` | stable | maintenance | Download missing packages for offline install. | package names | downloaded packages | local dependency cache | `tools/README.md` | Keep |
| `tools/dynamic_preload_sweep.py` | experimental | Seedon diagnostic | Sweep dynamic lateral preload motions. | preload grid | CSV | `dynamic_preload_sweep.csv` | `tools/registry.py` | Keep |
| `tools/eval_grasp.py` | stable | grasp | Evaluate trained fixed-base grasp checkpoint. | checkpoint/eval args | metrics/viewer | `reports/` | `tools/README.md` | Keep |
| `tools/explicit_contact_force_lift_controller_v1.py` | archived | Seedon phase | Gate IK micro-lift on swing-foot force reduction. | controller args | CSV/summary | `explicit_contact_force_lift_controller_v1/` | progress log | Archive candidate |
| `tools/fix_cusparselt.py` | stable | maintenance | Prepare cuSPARSELt repair instructions/assets. | local CUDA/PyTorch env | repair assets/instructions | local dependency cache | `tools/README.md` | Keep |
| `tools/gate_check.py` | stable | evaluation | Validate reports against gate profiles. | report JSON, gate profile | pass/fail exit/report | `reports/` | `tools/README.md` | Keep |
| `tools/generate_seedon_blue_like_sole_variant.py` | experimental | Seedon geometry | Generate Blue-like sole variant scene. | source scene, variant params | XML scene variants | `blue_like_sole_experiments*` | progress log | Review |
| `tools/generate_seedon_foot_geometry_variants.py` | experimental | Seedon geometry | Generate Seedon foot geometry variants. | scene, geometry params | XML variants | `rounded_sole_experiments/`, geometry XMLs | artifact names | Review |
| `tools/geometry_sensitivity_sweep.py` | experimental | Seedon diagnostic | Sweep temporary geometry variants for preload limits. | geometry grid | CSV | `geometry_sensitivity_sweep.csv` | `tools/registry.py` | Keep |
| `tools/grasp_sanity_check.py` | stable | grasp | Scripted grasp rollout sanity check. | grasp env settings | console metrics | reports if configured | `tools/README.md` | Keep |
| `tools/ik_lift_dynamic_validation_v1.py` | archived | Seedon phase | Validate IK micro-lift mapping under dynamic PD. | IK candidates | CSV/summary | `ik_lift_dynamic_validation_v1/` | progress log | Archive candidate |
| `tools/inspect_foot_geom_z.py` | unknown | Seedon diagnostic | Inspect foot geom z placement. | scene path | console output | none found by folder name | none explicit | Review |
| `tools/kinematic_foot_jacobian_diagnostic_v1.py` | archived | Seedon phase | Finite-difference swing-foot vertical authority diagnostic. | unload poses | CSV/summary | `kinematic_foot_jacobian_diagnostic_v1/` | progress log | Archive candidate |
| `tools/lateral_authority_audit.py` | experimental | Seedon diagnostic | Compare lateral assist against actuator authority. | audit settings | CSV | `lateral_authority_audit.csv` | `tools/registry.py` | Keep |
| `tools/lift_after_unload_v1.py` | archived | Seedon phase | Insert lift phases into refined unload windows. | prior unload candidates | CSV/summary | `lift_after_unload_v1/` | progress log | Archive candidate |
| `tools/lift_execution_audit_v1.py` | archived | Seedon phase | Audit commanded targets/qpos/limits/contacts in IK lift window. | lift candidate | CSV/summary | `lift_execution_audit_v1/` | progress log | Archive candidate |
| `tools/micro_lift_gain_calibration_v1.py` | archived | Seedon phase | Calibrate micro-lift mapping gain. | mapping candidates | CSV/summary | `micro_lift_gain_calibration_v1/` | progress log | Archive candidate |
| `tools/micro_lift_mapping_search_v1.py` | archived | Seedon phase | Search tiny joint mappings for visible micro-lift. | mapping grid | CSV/summary | `micro_lift_mapping_search_v1/` | progress log | Archive candidate |
| `tools/open_duck_viewer_from_h1.py` | unknown | viewer | Open Duck viewer from H1/Seedon context. | scene/model path | viewer | none found | none explicit | Review |
| `tools/plot_eval.py` | stable | evaluation | Plot evaluation CSV files. | eval CSV | plot image/window | `reports/` | `tools/README.md` | Keep |
| `tools/preflight_check.py` | active | checks | Check local prerequisites before training. | local environment | pass/fail console output | none | `tools/README.md` | Keep |
| `tools/prepare_package.py` | stable | release | Build offline dependency/source bundle. | project/dependency settings | bundle archive | package artifact dir | `tools/README.md` | Keep |
| `tools/preview_seedon_blue_balance_controller.py` | experimental | Seedon preview | Preview Blue-like closed-loop balance controller. | controller args | viewer/CSV | `blue_balance_controller_preview.csv` | `tools/README.md` | Keep |
| `tools/preview_seedon_blue_contact_gated.py` | experimental | Seedon preview | Preview contact-gated stepping controller. | controller args | viewer/CSV | `blue_contact_gated_preview.csv` | `tools/README.md` | Keep |
| `tools/preview_seedon_force_ratio_controller.py` | experimental | Seedon preview | Preview force-ratio controller. | controller args | viewer/CSV | `force_ratio_controller_preview.csv` | `tools/registry.py` | Keep |
| `tools/preview_seedon_gait.py` | active | Seedon preview | Preview deterministic gait targets before training. | gait mode, steps, scene | viewer/CSV | `preview_seedon_gait.csv`, `preview_*` | `tools/README.md` | Keep |
| `tools/preview_seedon_hybrid_torque_balance_controller.py` | experimental | Seedon preview | Preview hybrid pose plus hip-roll torque controller. | controller args | viewer/CSV | `hybrid_torque_balance_controller_preview.csv` | `tools/registry.py` | Keep |
| `tools/project_inventory.py` | active | checks | Print canonical directory/tool/script inventory. | source tree | console inventory | none | `tools/README.md`, `docs/PROJECT_GUIDE.md` | Keep |
| `tools/seedon/extractors/extract_seedon_parameters.py` | active | Seedon extractor | Extract explicit Seedon MJCF robot parameters for Duck-like gait reference work. | `private_assets/seedon/training_scene.xml` | `configs/seedon/seedon_robot_parameters.yaml`, `docs/seedon_parameter_index.md` | parameter snapshot docs/config | `docs/seedon_parameter_workflow.md` | Keep |
| `tools/seedon/extractors/extract_duck_parameters.py` | active | Duck reference extractor | Extract Open Duck Mini MJCF robot parameters from a caller-provided XML path. | `--duck-xml <path>` | `references/open_duck_mini/duck_robot_parameters.yaml`, `references/open_duck_mini/duck_extraction_report.md`, `references/open_duck_mini/source_manifest.yaml` | Duck reference directory | `docs/open_duck_reference_index.md`, `docs/seedon_parameter_workflow.md` | Keep |
| `tools/seedon/extractors/compare_seedon_duck_parameters.py` | active | Seedon/Duck comparison | Compare extracted Seedon and Duck parameter snapshots without inferring unavailable fields. | Seedon YAML, Duck YAML | `docs/seedon_duck_comparison.md` | parameter snapshots | `docs/seedon_parameter_workflow.md` | Keep |
| `tools/reference_lift_sweep.py` | archived | Seedon reference | Sweep pose-reference scale/lift/cadence. | reference grid | CSV/rendered top-k | `reference_lift_sweep/` | progress log | Archive candidate |
| `tools/registry.py` | active | tools metadata | Static registry for documented runnable tools. | static ToolEntry list | `python -m tools` output | none | `tools/README.md` | Keep |
| `tools/remote_auto_deploy.py` | active | release | Build, upload, activate, and smoke-check remote release. | `.env.remote`, release args | remote release/current link | release archives | `docs/SEEDON_WORKFLOW.md`, `docs/REMOTE_LAYOUT.md` | Keep |
| `tools/remote_training.py` | active | release | Start/inspect remote project training. | remote/project args | process/log/status output | remote run logs | `docs/SEEDON_WORKFLOW.md` | Keep |
| `tools/render_seedon_policy_comparison.py` | experimental | Seedon reporting | Render teacher/probe policy comparison video. | checkpoints, render args | video/CSV | `artifacts/seedon_debug/render/` | `tools/registry.py` | Keep |
| `tools/residual_safety_sweep.py` | experimental | Seedon diagnostic | Sweep residual action modes/scales against teacher reference. | residual grid | CSV | `residual_safety_sweep.csv` | `tools/registry.py` | Keep |
| `tools/seedon_blue_like_phase1_rollover_diagnostic.py` | archived | Seedon phase | Phase 1 rollover diagnostic. | steps/controller args | CSV/summary JSON | `blue_like_phase1_rollover/` | progress log | Archive candidate |
| `tools/seedon_capture_step_controller_v1.py` | archived | Seedon phase | Capture-step controller skeleton. | steps/controller args | CSV/summary JSON | `capture_step_controller_v1/` | progress log | Archive candidate |
| `tools/seedon_debug_common.py` | active | Seedon support | Shared helpers for Seedon debug tools. | imported by tools | helper functions | many Seedon artifacts | tool source | Keep |
| `tools/seedon_eval.py` | active | Seedon eval | Evaluate/render trained Seedon standing checkpoint. | model, vecnorm, episodes | viewer/GIF/report | `reports/seedon_eval.gif` | `tools/README.md` | Keep |
| `tools/seedon_explicit_locomotion_controller_v2.py` | experimental | Seedon controller | Run explicit locomotion FSM with contact-gated foot-z IK. | controller args | timeline output | `controller_v2/` | `tools/registry.py` | Keep |
| `tools/seedon_gait_sweep.py` | experimental | Seedon sweep | Sweep Seedon gait parameters. | sweep grid | CSV | gait/sweep CSVs | `tools/README.md` | Review |
| `tools/seedon_phase1_5_force_split_rollover_controller.py` | archived | Seedon phase | Phase 1.5 force-split plus rollover controller. | controller args | CSV/summary/log append | `phase1_5_force_split_rollover*` | progress log | Archive candidate |
| `tools/seedon_phase1_6_load_transfer_attribution.py` | archived | Seedon phase | Attribute load-transfer channels. | sweep grid | trials/top candidates/summary | `phase1_6_load_transfer_attribution/` | progress log | Archive candidate |
| `tools/seedon_phase1_7_load_transfer_profile_shaping.py` | archived | Seedon phase | Shape load-transfer profiles. | profile grid | CSV/summary | `phase1_7_load_transfer_profile_shaping*` | progress log | Archive candidate |
| `tools/seedon_phase2a_right_support_micro_capture.py` | archived | Seedon phase | Right-support micro-capture probe. | controller grid | CSV/summary | `phase2a_right_support_micro_capture/` | progress log | Archive candidate |
| `tools/seedon_phase2b_micro_capture_refinement.py` | archived | Seedon phase | Refine micro-capture candidates. | candidate grid | CSV/summary | `phase2b_micro_capture_refinement/` | progress log | Archive candidate |
| `tools/seedon/diagnostics/contact/phase2c_contact_constrained_foot_mapping.py` | active | Seedon blocker contact diagnostic | Diagnose contact-constrained foot forward mapping and patch behavior. Legacy wrapper remains at `tools/seedon_phase2c_contact_constrained_foot_mapping.py`. | scene/profile settings | CSV/summary | `phase2c_contact_constrained_foot_mapping/` | progress log, `docs/seedon_tools_migration_note.md` | Keep |
| `tools/seedon/diagnostics/contact/phase_g1_raw_contact_pair_diagnostic.py` | active | Seedon blocker contact diagnostic | Verify raw MuJoCo contact pairs and center/toe/heel evidence. Legacy wrapper remains at `tools/seedon_phase_g1_raw_contact_pair_diagnostic.py`. | neutral/pitch sweep args | raw contact CSV, report, summary | `phase_g1_raw_contact_pair_diagnostic/` | progress log, `docs/seedon_tools_migration_note.md` | Keep |
| `tools/seedon_phase_m0_duck_morphology_audit.py` | active | Seedon blocker diagnostic | Compare Seedon and Open Duck morphology at gait-metric level. | Seedon/Duck scene references | inventories, comparison report, scaled reference | `phase_m0_duck_morphology_audit/` | progress log | Keep |
| `tools/seedon_phase_pre_w0_actuator_semantics_audit.py` | active | Seedon blocker diagnostic | Audit actuator semantics before W0. | Seedon scene | inventory, summary, report | `phase_pre_w0_actuator_semantics_audit/` | progress log | Keep |
| `tools/seedon_phase_pre_w0_duckref_readiness_check.py` | active | Seedon blocker diagnostic | Consolidate M0/G1/actuator/sign evidence for W0 readiness. | prior summary artifacts | readiness summary/report | `phase_pre_w0_duckref_readiness_check/` | progress log | Keep |
| `tools/seedon_phase_pre_w0_free_space_joint_sign_validation.py` | active | Seedon blocker diagnostic | Validate joint sign mapping in free space. | Seedon scene | probe CSV, mapping JSON, report | `phase_pre_w0_free_space_joint_sign_validation/` | progress log | Keep |
| `tools/seedon_phase_w0_duckref_conservative_smoke_test.py` | active | Seedon blocker diagnostic | Conservative scripted DuckRef W0 smoke test. | Pre-W0 readiness summary, conservative grid | trials CSV, best timeline, summary/report | `phase_w0_duckref_conservative_smoke_test/` | progress log | Keep |
| `tools/seedon_unload_controller_v2a.py` | experimental | Seedon controller | Unload-only closed-loop hip-roll and lean correction. | controller args | CSV/summary | `controller_v2a_unload/` | `tools/registry.py` | Keep |
| `tools/smoke_seedon_env.py` | active | Seedon setup | Short Seedon standing env smoke test. | steps/env args | console pass/fail | none | `docs/SEEDON_WORKFLOW.md` | Keep |
| `tools/soft_landing_refine_v1.py` | archived | Seedon phase | Refine landing trajectories after lift. | candidate grid | CSV/summary | `soft_landing_refine_v1/` | progress log | Archive candidate |
| `tools/sweep.py` | stable | experiments | Optuna sweeps for H1 training parameters. | trial/step args | sweep DB/results | sweep outputs | `tools/README.md` | Keep |
| `tools/sweep_seedon_blue_contact_gated_targets.py` | experimental | Seedon sweep | Sweep contact-gated target candidates. | target grid | CSV/ranked candidates | `blue_contact_gated_target_sweep.csv` | `tools/README.md` | Keep |
| `tools/sweep_seedon_dynamic_fsm.py` | unknown | Seedon sweep | Sweep dynamic FSM parameters. | FSM sweep args | CSV | `dynamic_fsm_sweep.csv` | artifact name only | Review |
| `tools/sweep_seedon_fsm_1600.py` | unknown | Seedon sweep | Sweep 1600-step FSM candidates. | FSM sweep args | CSV | `fsm_1600_sweep.csv` | artifact name only | Review |
| `tools/sweep_seedon_phase1a_hip_roll_assist.py` | active | Seedon gate/sweep | Sweep scripted hip-roll assist for Phase 1A reachability. | sweep args | CSV | `phase1a_hip_roll_assist_sweep.csv` | `tools/registry.py` | Keep |
| `tools/sweep_seedon_preload.py` | experimental | Seedon sweep | Sweep small hip-roll preload targets. | preload grid | CSV | `seedon_preload_sweep.csv` | `tools/registry.py` | Keep |
| `tools/sweep_seedon_preload_v2.py` | experimental | Seedon sweep | Two-stage preload sweep. | stance/swing preload grid | CSV | `seedon_preload_sweep_v2*.csv` | `tools/registry.py` | Keep |
| `tools/trace_seedon_com_shift.py` | stable | Seedon trace | Trace COM shift diagnostics. | steps/out csv | CSV | `com_shift_trace.csv` | artifact name | Keep |
| `tools/trace_zero_action_gait.py` | stable | Seedon trace | Trace zero-action gait, foot heights, contacts. | steps, relaxed-foot | CSV | `zero_action_trace.csv` | `tools/README.md` | Keep |
| `tools/unload_authority_attribution_v1.py` | experimental | Seedon controller | Attribute unload force reduction to control channels. | controller args | CSV/summary | `unload_authority_attribution_v1/` | `tools/registry.py` | Keep |
| `tools/unload_controller_v2b_final_check.py` | experimental | Seedon controller | Final six-case unload v2b authority check. | controller args | CSV/summary | `unload_controller_v2b_final_check/` | `tools/registry.py` | Keep |
| `tools/validate_seedon_blue_like_sole.py` | experimental | Seedon geometry | Validate Blue-like sole variants. | generated scene manifest | summaries/reports | `blue_like_sole_experiments*` | progress log | Review |
| `tools/verify_seedon_static_seed.py` | active | Seedon gate | Verify zero-action standing seed against safety gates. | config, steps, thresholds | console pass/fail | `static_stability_summary.csv` adjacent context | `tools/README.md` | Keep |
| `tools/__init__.py` | stable | package | Mark `tools` as importable package. | none | package import support | none | source tree | Keep |
| `tools/__main__.py` | stable | package | Print registry index through `python -m tools`. | registry entries | console index | none | `tools/README.md` | Keep |

## Active Tool Set

Current active tools are the ones directly supporting setup, smoke/eval, gait inspection, remote operation, and the current blocker chain:

- Setup/checks: `preflight_check.py`, `project_inventory.py`, `convert_urdf_to_mjcf.py`, `build_seedon_training_scene.py`, `smoke_seedon_env.py`, `verify_seedon_static_seed.py`.
- Operation: `remote_auto_deploy.py`, `remote_training.py`, `seedon_eval.py`.
- Inspection: `debug_seedon_gait_viewer.py`, `debug_seedon_gait_audit.py`, `debug_seedon_pose_editor.py`, `preview_seedon_gait.py`.
- Current blocker diagnostics: `seedon/diagnostics/contact/phase2c_contact_constrained_foot_mapping.py`, `seedon/diagnostics/contact/phase_g1_raw_contact_pair_diagnostic.py`, `seedon_phase_m0_duck_morphology_audit.py`, `seedon_phase_pre_w0_actuator_semantics_audit.py`, `seedon_phase_pre_w0_free_space_joint_sign_validation.py`, `seedon_phase_pre_w0_duckref_readiness_check.py`, `seedon_phase_w0_duckref_conservative_smoke_test.py`.
- March/residual gates: `check_seedon_march_phase1a_gates.py`, `check_seedon_march_phase12_gates.py`, `sweep_seedon_phase1a_hip_roll_assist.py`.

## Deprecated / Broken / Unknown

- Explicit `deprecated`: none found in scanned docs or registry.
- Explicit `broken`: none found in scanned docs or registry.
- `unknown` because current registry/docs do not clearly state their role: `audit_seedon_grounded_reference_pd.py`, `debug_seedon_single_support_sweep.py`, `debug_seedon_static_tilt_handoff.py`, `inspect_foot_geom_z.py`, `open_duck_viewer_from_h1.py`, `sweep_seedon_dynamic_fsm.py`, `sweep_seedon_fsm_1600.py`.

## Review Notes

- Historical phase tools are marked `archived` rather than `deprecated` because their artifacts are still useful for traceability.
- No tool should be moved or deleted based on this file alone. Use this inventory to plan a later cleanup commit after confirming owners and reproducibility needs.
