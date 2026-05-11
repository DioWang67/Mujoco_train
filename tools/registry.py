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
        category="sedon",
        summary="Convert the private Sedon URDF/STL package into a MuJoCo MJCF scene.",
        example="python -m tools.convert_urdf_to_mjcf",
    ),
    ToolEntry(
        module="build_sedon_training_scene",
        category="sedon",
        summary="Build the floating-base Sedon training scene from converted MJCF.",
        example="python -m tools.build_sedon_training_scene",
    ),
    ToolEntry(
        module="smoke_sedon_env",
        category="sedon",
        summary="Run a short Sedon standing environment smoke test.",
        example="python -m tools.smoke_sedon_env --steps 20",
    ),
    ToolEntry(
        module="debug_sedon_knee_direction",
        category="sedon",
        summary="Diagnose Sedon knee bend direction, foot boxes, and safe knee ranges.",
        example="python -m tools.debug_sedon_knee_direction --offsets=-0.3,-0.2,-0.1,0.1,0.2,0.3",
    ),
    ToolEntry(
        module="debug_sedon_lateral_controllability",
        category="sedon",
        summary="Measure whether support-roll and unload targets actually move Sedon laterally.",
        example="python -m tools.debug_sedon_lateral_controllability --unload-scales 0,0.5,1.0",
    ),
    ToolEntry(
        module="debug_sedon_hip_roll_lateral_response",
        category="sedon",
        summary="Measure base_y and COM_y response from isolated Sedon hip-roll targets.",
        example="python -m tools.debug_sedon_hip_roll_lateral_response --offsets=-0.3,-0.2,-0.1,0.1,0.2,0.3",
    ),
    ToolEntry(
        module="debug_sedon_hip_roll_control_isolation",
        category="sedon",
        summary="Compare isolated Sedon hip-roll tracking across kinematic, fixed-base, no-floor, and normal floor modes.",
        example="python -m tools.debug_sedon_hip_roll_control_isolation --offsets=-0.2,-0.1,0.1,0.2",
    ),
    ToolEntry(
        module="debug_sedon_hip_roll_force_breakdown",
        category="sedon",
        summary="Break down hip-roll actuator, bias, passive, and constraint forces under normal floor contact to see whether tracking loss comes from actuator limits or contact suppression.",
        example="python -m tools.debug_sedon_hip_roll_force_breakdown --support-side left --hip-roll-offset 0.06 --steps 120",
    ),
    ToolEntry(
        module="debug_sedon_single_support_load_transfer",
        category="sedon",
        summary="Measure whether Sedon can shift COM and floor load onto one support foot before a tiny swing-foot lift.",
        example="python -m tools.debug_sedon_single_support_load_transfer --support-side left --load-steps 120 --lift-steps 80",
    ),
    ToolEntry(
        module="debug_sedon_pd_load_transfer_sweep",
        category="sedon",
        summary="Sweep Sedon PD stiffness/damping under floor contact to separate soft tracking from contact-locked load transfer.",
        example="python -m tools.debug_sedon_pd_load_transfer_sweep --kp-values 35,70,140,280 --kd-values 2,4,8,16 --hip-roll-offsets 0.02,0.05,0.08",
    ),
    ToolEntry(
        module="debug_sedon_foot_contact_geometry_sweep",
        category="sedon",
        summary="Sweep Sedon foot box size, friction, and reset height overrides under with-floor load transfer without changing the training scene.",
        example="python -m tools.debug_sedon_foot_contact_geometry_sweep --support-roll 0.10 --steps 120",
    ),
    ToolEntry(
        module="debug_sedon_contact_initialization_sweep",
        category="sedon",
        summary="Sweep Sedon reset base-height offsets and compare initial versus settled floor contacts before the same load-transfer target.",
        example="python -m tools.debug_sedon_contact_initialization_sweep --base-height-offsets -0.005,0,0.003,0.005,0.007,0.010,0.015",
    ),
    ToolEntry(
        module="debug_sedon_com_mass_distribution_sweep",
        category="sedon",
        summary="Sweep Sedon runtime mass distribution and stance width overrides to test whether lateral COM transfer is geometry- or inertia-limited.",
        example="python -m tools.debug_sedon_com_mass_distribution_sweep --top-k 10",
    ),
    ToolEntry(
        module="debug_sedon_com_feedback_balance_sweep",
        category="sedon",
        summary="Sweep a simple COM-feedback hip-roll balance controller to test whether closed-loop lateral feedback can create usable Sedon support transfer.",
        example="python -m tools.debug_sedon_com_feedback_balance_sweep --top-k 10",
    ),
    ToolEntry(
        module="debug_sedon_foot_proxy_redesign_sweep",
        category="sedon",
        summary="Create temporary Sedon foot proxy variants and test whether alternative contact layouts unlock stable lateral load transfer.",
        example="python -m tools.debug_sedon_foot_proxy_redesign_sweep --top-k 10",
    ),
    ToolEntry(
        module="debug_sedon_forced_support_lift_check",
        category="sedon",
        summary="Apply external lateral support loading, then test whether Sedon can unload and lift the swing foot.",
        example="python -m tools.debug_sedon_forced_support_lift_check --support-side both --lateral-forces 0,2,5,10,15",
    ),
    ToolEntry(
        module="debug_sedon_force_unload_controller",
        category="sedon",
        summary="Run a focused closed-loop force-unload controller proof before adding PPO rewards.",
        example="python -m tools.debug_sedon_force_unload_controller --support-sides left,right --force-kps 0.04,0.08,0.12",
    ),
    ToolEntry(
        module="debug_sedon_mechanical_variant_sweep",
        category="sedon",
        summary="Sweep temporary foot, COM, and stance-width variants against the Sedon force-unload gate.",
        example="python -m tools.debug_sedon_mechanical_variant_sweep --top-k 20",
    ),
    ToolEntry(
        module="debug_sedon_gait_audit",
        category="sedon",
        summary="Audit Sedon rollout contact forces, support phases, swing micro-lift, knee phase, COM stability, and artifact failure modes.",
        example="python -m tools.debug_sedon_gait_audit --scene-path artifacts/sedon_debug/training_scene_long_narrow_foot.xml --mode scripted --steps 400",
    ),
    ToolEntry(
        module="debug_sedon_pose_editor",
        category="sedon",
        summary="Open an interactive Sedon pose editor with MuJoCo viewer, joint sliders, pose save/load, sequence export, and support diagnostics.",
        example="python -m tools.debug_sedon_pose_editor --scene private_assets/sedon/training_scene.xml",
    ),
    ToolEntry(
        module="debug_sedon_gait_viewer",
        category="sedon",
        summary="Play zero/scripted/policy Sedon rollouts with MuJoCo viewer and per-step gait metrics.",
        example="python -m tools.debug_sedon_gait_viewer --mode scripted --steps 400 --out-csv artifacts/sedon_debug/gait_viewer.csv",
    ),
    ToolEntry(
        module="preview_sedon_blue_balance_controller",
        category="sedon",
        summary="Preview a Blue-like closed-loop Sedon balance controller with COM, roll, and contact-force feedback instead of pure open-loop timing.",
        example="python -m tools.preview_sedon_blue_balance_controller --steps 320 --render-viewer",
    ),
    ToolEntry(
        module="preview_sedon_gait",
        category="sedon",
        summary="Preview deterministic Sedon gait targets in the simulator without PPO training or loading.",
        example="python -m tools.preview_sedon_gait --gait-mode blue_step --steps 240 --render-viewer",
    ),
    ToolEntry(
        module="preview_sedon_blue_contact_gated",
        category="sedon",
        summary="Preview a Blue-like Sedon gait controller that only lifts after contact-force and COM-based support gating succeeds.",
        example="python -m tools.preview_sedon_blue_contact_gated --steps 320 --render-viewer",
    ),
    ToolEntry(
        module="sweep_sedon_blue_contact_gated_targets",
        category="sedon",
        summary="Sweep preview-side Sedon shift targets and rank which candidates come closest to opening the Blue-like contact gate.",
        example="python -m tools.sweep_sedon_blue_contact_gated_targets --support-sides left --top-k 8",
    ),
    ToolEntry(
        module="verify_sedon_static_seed",
        category="sedon",
        summary="Verify a Sedon zero-action standing seed against static safety gates.",
        example="python -m tools.verify_sedon_static_seed --config configs/sedon/zero_action_safe_stand.json",
    ),
    ToolEntry(
        module="sedon_eval",
        category="sedon",
        summary="Evaluate or render a trained Sedon standing checkpoint.",
        example="python -m tools.sedon_eval --episodes 1 --record",
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
        module="preview_sedon_force_ratio_controller",
        category="diagnostic",
        summary="Preview Blue-like force-ratio controller for Sedon.",
        example="python -m tools.preview_sedon_force_ratio_controller --steps 320 --render-viewer",
    ),
    ToolEntry(
        module="preview_sedon_hybrid_torque_balance_controller",
        category="diagnostic",
        summary="Preview Sedon Blue-like hybrid pose + hip-roll torque balance controller.",
        example="python -m tools.preview_sedon_hybrid_torque_balance_controller --steps 320 --render-viewer",
    ),
)


def tools_by_category() -> dict[str, list[ToolEntry]]:
    """Return tools grouped by category in registry order."""
    grouped: dict[str, list[ToolEntry]] = {}
    for tool in TOOLS:
        grouped.setdefault(tool.category, []).append(tool)
    return grouped
