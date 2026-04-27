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
)


def tools_by_category() -> dict[str, list[ToolEntry]]:
    """Return tools grouped by category in registry order."""
    grouped: dict[str, list[ToolEntry]] = {}
    for tool in TOOLS:
        grouped.setdefault(tool.category, []).append(tool)
    return grouped
