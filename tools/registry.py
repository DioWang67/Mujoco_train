"""Static index for command-line tools in this repository.

The registry intentionally avoids importing the tool modules themselves. Many
tools import MuJoCo, Stable-Baselines3, or plotting dependencies, so importing
them just to show help would make the lightweight index fragile.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolEntry:
    """Description of one runnable module under ``tools``.

    Args:
        module: Module name relative to ``tools``.
        category: Human-facing group for documentation and CLI listing.
        summary: Short description of when to use the tool.
        example: Representative command line.
    """

    module: str
    category: str
    summary: str
    example: str

    @property
    def command(self) -> str:
        """Return the canonical ``python -m`` command prefix."""
        return f"python -m tools.{self.module}"


TOOLS: tuple[ToolEntry, ...] = (
    ToolEntry(
        module="preflight_check",
        category="checks",
        summary="Check local runtime prerequisites before training.",
        example="python -m tools.preflight_check",
    ),
    ToolEntry(
        module="project_inventory",
        category="checks",
        summary="Print repository layout, tool counts, and disposable local output directories.",
        example="python -m tools.project_inventory",
    ),
    ToolEntry(
        module="agent_workspace",
        category="checks",
        summary="Create a sanitized source-only debug workspace for AI-assisted inspection.",
        example="python -m tools.agent_workspace --name seedon_debug --force",
    ),
    ToolEntry(
        module="compare_eval",
        category="evaluation",
        summary="Compare base and DR H1 policies on the same settings.",
        example=(
            "python -m tools.compare_eval --episodes 8 --vel 1.0 "
            "--out-json reports/compare_report.json"
        ),
    ),
    ToolEntry(
        module="aggregate_compare",
        category="evaluation",
        summary="Run multi-seed H1 comparison and report confidence intervals.",
        example=(
            "python -m tools.aggregate_compare --seeds 3 --episodes 5 "
            "--out-json reports/aggregate_compare.json"
        ),
    ),
    ToolEntry(
        module="benchmark_matrix",
        category="evaluation",
        summary="Run configured H1 benchmark scenarios from a matrix file.",
        example=(
            "python -m tools.benchmark_matrix --matrix configs/benchmark_matrix.json "
            "--out-json reports/benchmark_report.json"
        ),
    ),
    ToolEntry(
        module="gate_check",
        category="evaluation",
        summary="Validate compare or aggregate reports against release gates.",
        example=(
            "python -m tools.gate_check --report reports/compare_report.json "
            "--gates configs/gate_profiles.json --profile preprod"
        ),
    ),
    ToolEntry(
        module="plot_eval",
        category="evaluation",
        summary="Plot H1 evaluation CSV output.",
        example="python -m tools.plot_eval --file eval_ep1.csv --save",
    ),
    ToolEntry(
        module="eval_grasp",
        category="grasp",
        summary="Evaluate a trained fixed-base grasp checkpoint.",
        example="python -m tools.eval_grasp --episodes 10 --no-render",
    ),
    ToolEntry(
        module="grasp_sanity_check",
        category="grasp",
        summary="Run a scripted grasp rollout to verify reset/controller setup.",
        example="python -m tools.grasp_sanity_check",
    ),
    ToolEntry(
        module="convert_urdf_to_mjcf",
        category="seedon",
        summary="Convert the private Seedon URDF/STL package into a MuJoCo MJCF scene.",
        example="python -m tools.convert_urdf_to_mjcf",
    ),
    ToolEntry(
        module="build_seedon_training_scene",
        category="seedon",
        summary="Build the floating-base Seedon training scene from converted MJCF.",
        example="python -m tools.build_seedon_training_scene",
    ),
    ToolEntry(
        module="smoke_seedon_env",
        category="seedon",
        summary="Run a short Seedon standing environment smoke test.",
        example="python -m tools.smoke_seedon_env --steps 20",
    ),
    ToolEntry(
        module="debug_seedon_knee_direction",
        category="seedon",
        summary="Diagnose Seedon knee bend direction, foot boxes, and safe knee ranges.",
        example="python -m tools.debug_seedon_knee_direction --offsets=-0.3,-0.2,-0.1,0.1,0.2,0.3",
    ),
    ToolEntry(
        module="debug_seedon_lateral_controllability",
        category="seedon",
        summary="Measure whether support-roll and unload targets actually move Seedon laterally.",
        example="python -m tools.debug_seedon_lateral_controllability --unload-scales 0,0.5,1.0",
    ),
    ToolEntry(
        module="debug_seedon_hip_roll_lateral_response",
        category="seedon",
        summary="Measure base_y and COM_y response from isolated Seedon hip-roll targets.",
        example="python -m tools.debug_seedon_hip_roll_lateral_response --offsets=-0.3,-0.2,-0.1,0.1,0.2,0.3",
    ),
    ToolEntry(
        module="debug_seedon_hip_roll_control_isolation",
        category="seedon",
        summary="Compare isolated Seedon hip-roll tracking across kinematic, fixed-base, no-floor, and normal floor modes.",
        example="python -m tools.debug_seedon_hip_roll_control_isolation --offsets=-0.2,-0.1,0.1,0.2",
    ),
    ToolEntry(
        module="debug_seedon_hip_roll_force_breakdown",
        category="seedon",
        summary="Break down hip-roll actuator, bias, passive, and constraint forces under normal floor contact to see whether tracking loss comes from actuator limits or contact suppression.",
        example="python -m tools.debug_seedon_hip_roll_force_breakdown --support-side left --hip-roll-offset 0.06 --steps 120",
    ),
    ToolEntry(
        module="debug_seedon_single_support_load_transfer",
        category="seedon",
        summary="Measure whether Seedon can shift COM and floor load onto one support foot before a tiny swing-foot lift.",
        example="python -m tools.debug_seedon_single_support_load_transfer --support-side left --load-steps 120 --lift-steps 80",
    ),
    ToolEntry(
        module="debug_seedon_pd_load_transfer_sweep",
        category="seedon",
        summary="Sweep Seedon PD stiffness/damping under floor contact to separate soft tracking from contact-locked load transfer.",
        example="python -m tools.debug_seedon_pd_load_transfer_sweep --kp-values 35,70,140,280 --kd-values 2,4,8,16 --hip-roll-offsets 0.02,0.05,0.08",
    ),
    ToolEntry(
        module="debug_seedon_foot_contact_geometry_sweep",
        category="seedon",
        summary="Sweep Seedon foot box size, friction, and reset height overrides under with-floor load transfer without changing the training scene.",
        example="python -m tools.debug_seedon_foot_contact_geometry_sweep --support-roll 0.10 --steps 120",
    ),
    ToolEntry(
        module="debug_seedon_contact_initialization_sweep",
        category="seedon",
        summary="Sweep Seedon reset base-height offsets and compare initial versus settled floor contacts before the same load-transfer target.",
        example="python -m tools.debug_seedon_contact_initialization_sweep --base-height-offsets -0.005,0,0.003,0.005,0.007,0.010,0.015",
    ),
    ToolEntry(
        module="debug_seedon_com_mass_distribution_sweep",
        category="seedon",
        summary="Sweep Seedon runtime mass distribution and stance width overrides to test whether lateral COM transfer is geometry- or inertia-limited.",
        example="python -m tools.debug_seedon_com_mass_distribution_sweep --top-k 10",
    ),
    ToolEntry(
        module="debug_seedon_com_feedback_balance_sweep",
        category="seedon",
        summary="Sweep a simple COM-feedback hip-roll balance controller to test whether closed-loop lateral feedback can create usable Seedon support transfer.",
        example="python -m tools.debug_seedon_com_feedback_balance_sweep --top-k 10",
    ),
    ToolEntry(
        module="debug_seedon_foot_proxy_redesign_sweep",
        category="seedon",
        summary="Create temporary Seedon foot proxy variants and test whether alternative contact layouts unlock stable lateral load transfer.",
        example="python -m tools.debug_seedon_foot_proxy_redesign_sweep --top-k 10",
    ),
    ToolEntry(
        module="debug_seedon_forced_support_lift_check",
        category="seedon",
        summary="Apply external lateral support loading, then test whether Seedon can unload and lift the swing foot.",
        example="python -m tools.debug_seedon_forced_support_lift_check --support-side both --lateral-forces 0,2,5,10,15",
    ),
    ToolEntry(
        module="debug_seedon_force_unload_controller",
        category="seedon",
        summary="Run a focused closed-loop force-unload controller proof before adding PPO rewards.",
        example="python -m tools.debug_seedon_force_unload_controller --support-sides left,right --force-kps 0.04,0.08,0.12",
    ),
    ToolEntry(
        module="debug_seedon_mechanical_variant_sweep",
        category="seedon",
        summary="Sweep temporary foot, COM, and stance-width variants against the Seedon force-unload gate.",
        example="python -m tools.debug_seedon_mechanical_variant_sweep --top-k 20",
    ),
    ToolEntry(
        module="debug_seedon_gait_audit",
        category="seedon",
        summary="Audit Seedon rollout contact forces, support phases, swing micro-lift, knee phase, COM stability, and artifact failure modes.",
        example="python -m tools.debug_seedon_gait_audit --scene-path artifacts/seedon_debug/training_scene_long_narrow_foot.xml --mode scripted --steps 400",
    ),
    ToolEntry(
        module="debug_seedon_pose_editor",
        category="seedon",
        summary="Open an interactive Seedon pose editor with MuJoCo viewer, joint sliders, pose save/load, sequence export, and support diagnostics.",
        example="python -m tools.debug_seedon_pose_editor --scene private_assets/seedon/training_scene.xml",
    ),
    ToolEntry(
        module="debug_seedon_gait_viewer",
        category="seedon",
        summary="Play zero/scripted/policy Seedon rollouts with MuJoCo viewer and per-step gait metrics.",
        example="python -m tools.debug_seedon_gait_viewer --mode scripted --steps 400 --out-csv artifacts/seedon_debug/gait_viewer.csv",
    ),
    ToolEntry(
        module="render_seedon_policy_comparison",
        category="seedon",
        summary="Render teacher and probe Seedon policies plus a side-by-side comparison video from fixed side-view camera.",
        example="python -m tools.render_seedon_policy_comparison --steps 600 --fps 30",
    ),
    ToolEntry(
        module="reference_lift_sweep",
        category="seedon",
        summary="Sweep Seedon pose-reference scale, swing lift amplification, and cadence under deterministic teacher PD tracking.",
        example="python -m tools.reference_lift_sweep --steps 600 --render-top-k 3",
    ),
    ToolEntry(
        module="build_blue_step_reference_v1",
        category="seedon",
        summary="Build and sweep a new Blue-like visible stepping reference family with preload, micro-lift, and soft-land keyframes.",
        example="python -m tools.build_blue_step_reference_v1 --steps 600 --render-top-k 3",
    ),
    ToolEntry(
        module="build_blue_step_reference_v2",
        category="seedon",
        summary="Build and sweep Blue-like stepping references that unload the swing foot before micro-lift.",
        example="python -m tools.build_blue_step_reference_v2 --steps 600 --render-top-k 3",
    ),
    ToolEntry(
        module="build_blue_step_reference_v3_closed_loop",
        category="seedon",
        summary="Build Blue-like stepping references by dynamically finding unload trigger steps before micro-lift.",
        example="python -m tools.build_blue_step_reference_v3_closed_loop --steps 600 --render-top-k 3",
    ),
    ToolEntry(
        module="blue_unload_mechanism_search",
        category="seedon",
        summary="Search unload-only Blue-like reference mechanisms before adding visible lift.",
        example="python -m tools.blue_unload_mechanism_search --steps 600 --render-top-k 3",
    ),
    ToolEntry(
        module="blue_unload_refine_v2",
        category="seedon",
        summary="Locally refine top Blue unload-only candidates before attempting micro-lift.",
        example="python -m tools.blue_unload_refine_v2 --steps 600 --render-top-k 3",
    ),
    ToolEntry(
        module="lift_after_unload_v1",
        category="seedon",
        summary="Insert small lift phases into refined unload windows and audit visible stepping.",
        example="python -m tools.lift_after_unload_v1 --steps 600 --render-top-k 3",
    ),
    ToolEntry(
        module="soft_landing_refine_v1",
        category="seedon",
        summary="Refine landing trajectories for lift-after-unload references without training.",
        example="python -m tools.soft_landing_refine_v1 --steps 600 --render-top-k 3",
    ),
    ToolEntry(
        module="micro_lift_mapping_search_v1",
        category="seedon",
        summary="Search tiny joint mappings for stable visible micro-lift after unload.",
        example="python -m tools.micro_lift_mapping_search_v1 --steps 600 --render-top-k 3",
    ),
    ToolEntry(
        module="micro_lift_gain_calibration_v1",
        category="seedon",
        summary="Calibrate micro-lift mapping gain and classify usable, cliff, or ineffective behavior.",
        example="python -m tools.micro_lift_gain_calibration_v1 --steps 600 --render-top-k 3",
    ),
    ToolEntry(
        module="kinematic_foot_jacobian_diagnostic_v1",
        category="seedon",
        summary="Kinematic finite-difference diagnostic for swing-foot vertical authority at unload poses.",
        example="python -m tools.kinematic_foot_jacobian_diagnostic_v1",
    ),
    ToolEntry(
        module="ik_lift_dynamic_validation_v1",
        category="seedon",
        summary="Validate recommended kinematic IK micro-lift mapping under deterministic dynamic PD.",
        example="python -m tools.ik_lift_dynamic_validation_v1 --steps 600 --render-top-k 3",
    ),
    ToolEntry(
        module="lift_execution_audit_v1",
        category="seedon",
        summary="Audit commanded targets, actual qpos, actuator limits, foot z, and contacts inside one IK lift window.",
        example="python -m tools.lift_execution_audit_v1 --steps 600",
    ),
    ToolEntry(
        module="explicit_contact_force_lift_controller_v1",
        category="seedon",
        summary="Gate IK micro-lift on explicit swing-foot normal-force reduction without PPO or geometry changes.",
        example="python -m tools.explicit_contact_force_lift_controller_v1 --steps 600",
    ),
    ToolEntry(
        module="blue_forward_shuffle_v1",
        category="seedon",
        summary="Audit tiny forward-drift teacher shaping on the grounded Blue-like shuffle baseline.",
        example="python -m tools.blue_forward_shuffle_v1 --steps 600",
    ),
    ToolEntry(
        module="blue_forward_shuffle_authority_sweep_v1",
        category="seedon",
        summary="Sweep forward-shaping force, application point, phase gate, and cadence for grounded shuffle authority.",
        example="python -m tools.blue_forward_shuffle_authority_sweep_v1 --steps 600",
    ),
    ToolEntry(
        module="blue_forward_phase_timing_refine_v1",
        category="seedon",
        summary="Locally refine forward-shaping phase timing, duty cycle, and right/left force balance for grounded shuffle.",
        example="python -m tools.blue_forward_phase_timing_refine_v1 --steps 600",
    ),
    ToolEntry(
        module="seedon_explicit_locomotion_controller_v2",
        category="seedon",
        summary="Run the Seedon explicit locomotion controller v2 FSM with contact-gated foot-z IK and audit timeline output.",
        example="python -m tools.seedon_explicit_locomotion_controller_v2 --steps 600",
    ),
    ToolEntry(
        module="seedon_unload_controller_v2a",
        category="seedon",
        summary="Run unload-only closed-loop hip-roll and lean correction for Seedon controller v2a contact-force gates.",
        example="python -m tools.seedon_unload_controller_v2a --steps 600",
    ),
    ToolEntry(
        module="unload_authority_attribution_v1",
        category="seedon",
        summary="Attribute Seedon unload force reduction to individual control channels with single-channel sensitivity tests.",
        example="python -m tools.unload_authority_attribution_v1 --steps 600",
    ),
    ToolEntry(
        module="unload_controller_v2b_final_check",
        category="seedon",
        summary="Run the six-case final Seedon unload controller v2b authority check without lift, PPO, or knee/ankle channels.",
        example="python -m tools.unload_controller_v2b_final_check --steps 600",
    ),
    ToolEntry(
        module="preview_seedon_blue_balance_controller",
        category="seedon",
        summary="Preview a Blue-like closed-loop Seedon balance controller with COM, roll, and contact-force feedback instead of pure open-loop timing.",
        example="python -m tools.preview_seedon_blue_balance_controller --steps 320 --render-viewer",
    ),
    ToolEntry(
        module="preview_seedon_gait",
        category="seedon",
        summary="Preview deterministic Seedon gait targets in the simulator without PPO training or loading.",
        example="python -m tools.preview_seedon_gait --gait-mode blue_step --steps 240 --render-viewer",
    ),
    ToolEntry(
        module="preview_seedon_blue_contact_gated",
        category="seedon",
        summary="Preview a Blue-like Seedon gait controller that only lifts after contact-force and COM-based support gating succeeds.",
        example="python -m tools.preview_seedon_blue_contact_gated --steps 320 --render-viewer",
    ),
    ToolEntry(
        module="sweep_seedon_blue_contact_gated_targets",
        category="seedon",
        summary="Sweep preview-side Seedon shift targets and rank which candidates come closest to opening the Blue-like contact gate.",
        example="python -m tools.sweep_seedon_blue_contact_gated_targets --support-sides left --top-k 8",
    ),
    ToolEntry(
        module="verify_seedon_static_seed",
        category="seedon",
        summary="Verify a Seedon zero-action standing seed against static safety gates.",
        example="python -m tools.verify_seedon_static_seed --config configs/seedon/zero_action_safe_stand.json",
    ),
    ToolEntry(
        module="seedon_eval",
        category="seedon",
        summary="Evaluate or render a trained Seedon standing checkpoint.",
        example="python -m tools.seedon_eval --episodes 1 --record",
    ),
    ToolEntry(
        module="deploy_release",
        category="release",
        summary="Create and optionally upload a clean source release archive.",
        example=(
            "python -m tools.deploy_release --project-slug h1 "
            "--remote-host root@10.6.243.55 --upload"
        ),
    ),
    ToolEntry(
        module="remote_auto_deploy",
        category="release",
        summary="Build, upload, activate, and smoke-check a remote release using env credentials.",
        example="python -m tools.remote_auto_deploy --project-slug seedon --include-private-assets",
    ),
    ToolEntry(
        module="remote_training",
        category="release",
        summary="Start or inspect remote project training from the shared remote layout.",
        example="python -m tools.remote_training --project seedon --status",
    ),
    ToolEntry(
        module="prepare_package",
        category="release",
        summary="Build an offline dependency/source bundle for the remote host.",
        example="python -m tools.prepare_package",
    ),
    ToolEntry(
        module="download_cuda_deps",
        category="maintenance",
        summary="Download CUDA runtime Python packages for remote repair.",
        example="python -m tools.download_cuda_deps",
    ),
    ToolEntry(
        module="download_missing",
        category="maintenance",
        summary="Download missing Python packages for offline remote install.",
        example="python -m tools.download_missing",
    ),
    ToolEntry(
        module="fix_cusparselt",
        category="maintenance",
        summary="Prepare cuSPARSELt repair instructions/assets.",
        example="python -m tools.fix_cusparselt",
    ),
    ToolEntry(
        module="sweep",
        category="experiments",
        summary="Run Optuna sweeps for H1 training parameters.",
        example="python -m tools.sweep --n-trials 20 --steps 1000000",
    ),
    ToolEntry(
        module="preview_seedon_force_ratio_controller",
        category="diagnostic",
        summary="Preview Blue-like force-ratio controller for Seedon.",
        example="python -m tools.preview_seedon_force_ratio_controller --steps 320 --render-viewer",
    ),
    ToolEntry(
        module="preview_seedon_hybrid_torque_balance_controller",
        category="diagnostic",
        summary="Preview Seedon Blue-like hybrid pose + hip-roll torque balance controller.",
        example="python -m tools.preview_seedon_hybrid_torque_balance_controller --steps 320 --render-viewer",
    ),
    ToolEntry(
        module="check_seedon_march_phase12_gates",
        category="diagnostic",
        summary="Check Seedon march Phase 1/2 gates before enabling micro-lift.",
        example="python -m tools.check_seedon_march_phase12_gates --model-path models/seedon/latest_model.zip",
    ),
    ToolEntry(
        module="check_seedon_march_phase1a_gates",
        category="diagnostic",
        summary="Check Seedon march Phase 1A hip-roll residual gates.",
        example="python -m tools.check_seedon_march_phase1a_gates --model-path models/seedon/latest_model.zip --vecnorm-path models/seedon/vecnorm.pkl",
    ),
    ToolEntry(
        module="sweep_seedon_phase1a_hip_roll_assist",
        category="diagnostic",
        summary="Sweep scripted Seedon hip-roll assist for Phase 1A load-transfer reachability.",
        example="python -m tools.sweep_seedon_phase1a_hip_roll_assist",
    ),
    ToolEntry(
        module="sweep_seedon_preload",
        category="diagnostic",
        summary="Sweep small Seedon hip-roll preload targets for Blue-like standing load ratios.",
        example="python -m tools.sweep_seedon_preload --settle-steps 240 --top-k 10",
    ),
    ToolEntry(
        module="sweep_seedon_preload_v2",
        category="diagnostic",
        summary="Two-stage Seedon preload sweep with stance knee/ankle and swing hip-roll deltas.",
        example="python -m tools.sweep_seedon_preload_v2 --settle-steps 240 --top-k 20",
    ),
    ToolEntry(
        module="dynamic_preload_sweep",
        category="diagnostic",
        summary="Sweep sinusoidal/smoothstep Seedon lateral preload motions for in-place dynamic load transfer.",
        example="python -m tools.dynamic_preload_sweep --top-k 20",
    ),
    ToolEntry(
        module="audit_seedon_shuffle_v0",
        category="diagnostic",
        summary="Audit the low-threshold Seedon shuffle v0 curriculum.",
        example="python -m tools.audit_seedon_shuffle_v0 --config configs/seedon/shuffle_v0.json",
    ),
    ToolEntry(
        module="geometry_sensitivity_sweep",
        category="diagnostic",
        summary="Sweep temporary Seedon geometry variants to diagnose lateral preload limits.",
        example="python -m tools.geometry_sensitivity_sweep --top-k 20",
    ),
    ToolEntry(
        module="lateral_authority_audit",
        category="diagnostic",
        summary="Compare medium lateral assist against actuator authority replacements.",
        example="python -m tools.lateral_authority_audit",
    ),
    ToolEntry(
        module="assisted_shuffle_curriculum_eval",
        category="diagnostic",
        summary="Evaluate deterministic pose_1..4 shuffle while reducing lateral assist force.",
        example="python -m tools.assisted_shuffle_curriculum_eval",
    ),
    ToolEntry(
        module="audit_seedon_teacher_imitation",
        category="diagnostic",
        summary="Audit pose_1..4 teacher-imitation policy without lateral assist.",
        example="python -m tools.audit_seedon_teacher_imitation --model-path models/seedon/latest_model.zip --vecnorm-path models/seedon/vecnorm.pkl",
    ),
    ToolEntry(
        module="residual_safety_sweep",
        category="diagnostic",
        summary="Sweep residual action modes and scales against the Seedon teacher reference.",
        example="python -m tools.residual_safety_sweep --steps 480",
    ),
)


def tools_by_category() -> dict[str, list[ToolEntry]]:
    """Return tools grouped by category in registry order."""
    grouped: dict[str, list[ToolEntry]] = {}
    for tool in TOOLS:
        grouped.setdefault(tool.category, []).append(tool)
    return grouped
