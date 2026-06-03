# Tools Index

Run this to print the same index from the command line:

```bash
python -m tools
```

## How To Read This Directory

`tools/` contains both standard operator commands and narrow diagnostic
experiments. Use this order before reaching for one-off scripts:

| Tier | Use for | First tools to try |
|---|---|---|
| Standard | Normal smoke/eval/view/audit workflows. | `smoke_seedon_env`, `seedon_eval`, `debug_seedon_gait_viewer`, `debug_seedon_gait_audit` |
| Diagnostic | Answer one mechanical or reward question. | `debug_seedon_*`, `preview_seedon_*`, `trace_*` |
| Sweep | Compare many parameters and write artifacts. | `sweep_seedon_*`, `seedon_gait_sweep` |
| Packaging | Build and deploy releases. | `deploy_release`, `remote_auto_deploy`, `prepare_package` |

For the current Seedon workflow, start with `docs/SEEDON_WORKFLOW.md`.

## Checks

- `python -m tools.preflight_check`
  Check local runtime prerequisites before training.
- `python -m tools.project_inventory`
  Print the canonical directory map, tool/script counts, and disposable local
  output directories without deleting anything.
- `python -m tools.agent_workspace --name seedon_debug --force`
  Create an ignored, source-only debug workspace under
  `artifacts/agent_workspace/` for AI-assisted inspection. It copies code,
  tests, docs, and lightweight configs while excluding `private_assets/`,
  complete XML/MJCF/URDF/mesh files, models, logs, local env files, and
  generated artifacts. The output includes `SANITIZED_MANIFEST.json`.

## Evaluation

- `python -m tools.compare_eval`
  Compare base and DR H1 policies.
- `python -m tools.aggregate_compare`
  Run multi-seed H1 comparison and 95% confidence intervals.
- `python -m tools.benchmark_matrix`
  Run configured H1 benchmark scenarios.
- `python -m tools.gate_check`
  Validate reports against gate profiles.
- `python -m tools.plot_eval`
  Plot evaluation CSV files.

## Grasp

- `python -m tools.eval_grasp`
  Evaluate a trained fixed-base grasp checkpoint.
- `python -m tools.grasp_sanity_check`
  Run a scripted grasp rollout to verify reset/controller setup.

## Seedon

- `python -m tools.convert_urdf_to_mjcf`
  Convert the private Seedon URDF/STL package into a MuJoCo MJCF scene.
- `python -m tools.build_seedon_training_scene`
  Build the floating-base Seedon training scene from converted MJCF.
- `python -m tools.smoke_seedon_env --steps 20`
  Run a short Seedon standing environment smoke test.
- `python -m tools.debug_seedon_knee_direction --offsets=-0.3,-0.2,-0.1,0.1,0.2,0.3`
  Sweep isolated Seedon leg joints, infer knee bend direction, and suggest safe
  knee ranges while printing foot collision placement diagnostics.
- `python -m tools.debug_seedon_lateral_controllability --unload-scales 0,0.5,1.0`
  Hold deterministic support-roll and unload targets, then measure actual
  `base_y`, COM lateral motion, swing-foot unload, and contact ratios.
- `python -m tools.debug_seedon_hip_roll_lateral_response --offsets=-0.3,-0.2,-0.1,0.1,0.2,0.3`
  Track isolated `R_hip_roll`, `L_hip_roll`, and symmetric hip-roll targets for
  100 steps while measuring `base_y`, COM lateral response, foot contact ratios,
  and left/right mirror symmetry.
- `python -m tools.debug_seedon_hip_roll_control_isolation --offsets=-0.2,-0.1,0.1,0.2`
  Compare isolated hip-roll tracking across kinematic-only, fixed-base,
  free-base no-floor, and normal floor-contact dynamics to separate actuator/PD
  limits from contact/load suppression.
- `python -m tools.debug_seedon_hip_roll_force_breakdown --support-side left --hip-roll-offset 0.06 --steps 120`
  Hold an isolated with-floor hip-roll support target, then decompose the
  hip-roll DOF generalized forces into actuator, total constraint, non-limit
  constraint, joint-limit constraint, passive, and bias terms to check whether
  floor-contact constraints are eating the commanded roll.
- `python -m tools.debug_seedon_single_support_load_transfer --support-side left --load-steps 120 --lift-steps 80`
  Hold a deterministic support-roll target, measure COM-to-support-foot margin,
  left/right floor normal forces, and only attempt a tiny swing-foot lift if
  load transfer gates pass.
- `python -m tools.debug_seedon_pd_load_transfer_sweep --kp-values 35,70,140,280 --kd-values 2,4,8,16 --hip-roll-offsets 0.02,0.05,0.08`
  Sweep `pd_stiffness/pd_damping` under normal floor contact and report whether
  harder tracking actually increases COM lateral shift, support-foot load bias,
  or single-support time before instability.
- `python -m tools.debug_seedon_foot_contact_geometry_sweep --support-roll 0.10 --steps 120`
  Sweep in-memory foot box size, floor/foot friction, and reset base-height
  overrides while holding the same with-floor load-transfer target, to isolate
  whether contact geometry is locking Seedon into double support.
- `python -m tools.debug_seedon_contact_initialization_sweep --base-height-offsets -0.005,0,0.003,0.005,0.007,0.010,0.015`
  Sweep reset base-height offsets, record immediate and settled foot contact
  states, then apply the same load-transfer target to see whether a clean
  contact window changes COM lateral response.
- `python -m tools.debug_seedon_com_mass_distribution_sweep --top-k 10`
  Sweep runtime torso/pelvis/leg mass scales plus stance-width scaling, then
  rank which cases produce the largest stable lateral COM shift under the same
  standing seed and same hip-roll shift target.
- `python -m tools.debug_seedon_com_feedback_balance_sweep --top-k 10`
  Sweep a simple closed-loop COM-feedback hip-roll controller to test whether
  Seedon can generate larger stable lateral COM transfer with feedback, without
  changing PPO, reward, gait timing, or the committed training scene.
- `python -m tools.debug_seedon_foot_proxy_redesign_sweep --top-k 10`
  Create temporary foot contact proxy variants such as four-corner spheres,
  toe/heel boxes, rounded soles, and lateral edge boxes, then test whether any
  proxy layout unlocks stable lateral support transfer.
- `python -m tools.debug_seedon_forced_support_lift_check --support-side both --lateral-forces 0,2,5,10,15`
  Apply an external lateral force toward the requested support foot, then
  attempt a swing-foot lift. Use this to separate missing load transfer from
  impossible swing-leg clearance.
- `python -m tools.debug_seedon_force_unload_controller --support-sides left,right --force-kps 0.04,0.08,0.12`
  Run a focused closed-loop force-unload controller proof. It adjusts support
  hip-roll from foot normal-force feedback and reports whether Seedon can hold a
  stable support-force bias before any PPO reward is added.
- `python -m tools.debug_seedon_mechanical_variant_sweep --top-k 20`
  Sweep temporary foot-size, base-COM, stance-width, and support-roll variants
  against the same force-unload gate to find whether any "cheat" model can
  create stable single-side loading.
- `python -m tools.debug_seedon_gait_audit --scene-path artifacts/seedon_debug/training_scene_long_narrow_foot.xml --mode scripted --steps 400`
  Audit a Seedon zero-action/scripted/policy rollout with per-step contact
  forces, foot-foot/base-proxy contamination flags, support phase ratios,
  swing micro-lift, positive knee phase, COM stability, and effort metrics.
- `python -m tools.preview_seedon_blue_balance_controller --steps 320 --render-viewer`
  Preview a Blue-like closed-loop balance controller with state estimation for
  `COM_y`, `COM_y` velocity, `base_roll`, `base_roll` velocity, and foot normal
  force, then drive stance stabilization, swing unload, roll regulation, and
  force-ratio feedback into joint-space targets.
- `python -m tools.preview_seedon_blue_balance_controller --ablation-modes full_controller,no_base_roll_stabilizer,no_com_feedback,no_force_ratio_feedback,support_roll_only --ablation-side left --ablation-steps 160`
  Run a left-support acquire ablation table that compares which feedback terms
  actually increase support-side force ratio or COM shift, and which terms may
  be damping lateral transfer back out.
- `python -m tools.preview_seedon_gait --gait-mode blue_step --steps 240 --render-viewer`
  Preview deterministic `fsm`, `blue_step`, or `com_shift` target trajectories
  before training while printing knee qpos, foot bottom heights, contact state,
  base height, and uprightness at each step.
- `python -m tools.preview_seedon_blue_contact_gated --steps 320 --render-viewer`
  Preview a Blue-like contact-gated Seedon stepping controller that only enters
  swing lift after support-side force ratio and COM lateral shift gates pass,
  then logs per-step phase/load/foot-height diagnostics to CSV.
- `python -m tools.sweep_seedon_blue_contact_gated_targets --support-sides left --top-k 8`
  Sweep preview-side shift/unload target candidates, rank which ones get
  closest to a stable support-side force bias while both feet are still down,
  and print reusable CLI flags for replaying the best candidates.
- `python -m tools.verify_seedon_static_seed --config configs/seedon/zero_action_safe_stand.json`
  Verify that a Seedon standing seed survives a 400-step zero-action rollout
  without knee violations, base-proxy floor contact, or excessive forward drift.
- `python -m tools.debug_seedon_contacts`
  Inspect Seedon foot/base proxy geom placement and reset contacts.
- `python -m tools.trace_zero_action_gait --steps 400 --print-every 25 --relaxed-foot`
  Trace zero-action gait, foot heights, and contact pairs to CSV.
- `python -m tools.debug_seedon_static_stability --plan both --relaxed-foot`
  Check Seedon COM placement, foot support boxes, relaxed foot contact, and
  load-phase static single-support attempts.
- `python -m tools.debug_seedon_lift_directions --relaxed-foot`
  Sweep swing-leg hip/knee/ankle direction signs and measure actual foot
  clearance.
- `python -m tools.debug_seedon_joint_ranges`
  Compare original URDF, prepared URDF, and MuJoCo joint ranges.
- `python -m tools.debug_seedon_target_tracking --mode unload-lift --relaxed-foot`
  Trace target angles versus actual qpos for unload/lift diagnostics.
- `python -m tools.debug_seedon_kinematic_foot_map`
  Sweep fixed-base hip/knee/ankle qpos with `mj_forward` and rank foot
  clearance without contact dynamics.
- `python tools/debug_seedon_viewer.py`
  Open a MuJoCo viewer scene with foot collision boxes and base proxy colored.
  In the viewer visualization panel, enable contacts/contact forces to inspect
  active contact points.
- `python -m tools.seedon_eval --episodes 1 --render`
  Watch a trained Seedon standing checkpoint in the MuJoCo viewer.
- `python -m tools.seedon_eval --episodes 1 --record`
  Record a trained Seedon standing checkpoint to `reports/seedon_eval.gif`.

## Release

- `python -m tools.deploy_release`
  Create and optionally upload a clean source release archive.
- `python -m tools.remote_auto_deploy`
  Build from `.env.remote` settings, upload, activate `code/current`, and run a
  remote smoke check. Defaults to working-tree packaging for fast experiment
  iteration.
  Daily Seedon wrappers: `scripts\seedon_remote_deploy_and_check.bat` and
  `scripts\seedon_remote_check.bat`.
- `python -m tools.remote_training --project seedon --status`
  Start or inspect remote project training from the shared remote layout.
  Project-specific wrappers should stay thin and call this module.
- `python -m tools.prepare_package`
  Build an offline dependency/source bundle for a remote host.

## Maintenance

- `python -m tools.download_cuda_deps`
  Download CUDA runtime Python packages for remote repair.
- `python -m tools.download_missing`
  Download missing Python packages for offline install.
- `python -m tools.fix_cusparselt`
  Prepare cuSPARSELt repair instructions/assets.

## Experiments

- `python -m tools.sweep`
  Run Optuna sweeps for H1 training parameters.

## Rule of Thumb

- Put reusable Python CLIs here.
- Keep wrapper-only commands in `scripts/`.
- Avoid importing simulator/training dependencies at module import time unless
  the command truly needs them immediately.
