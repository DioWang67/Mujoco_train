"""Summarize the Sedon Duck-like gait readiness pipeline.

By default this runner is read-only: it does not re-run phase tools, so existing
phase outputs and artifact timestamps remain stable. It writes the pipeline
documentation from committed phase outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_VERSION = "sedon-duck-like-readiness-pipeline-v1"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT = REPO_ROOT / "docs" / "sedon_duck_like_pipeline.md"

PHASE_OUTPUTS = [
    {
        "phase": 1,
        "name": "normalized_scale_table",
        "path": REPO_ROOT / "configs" / "sedon" / "sedon_duck_scale_mapping.yaml",
        "status_path": ["summary", "readiness"],
        "pass_statuses": {"scale_reference_only"},
    },
    {
        "phase": 2,
        "name": "joint_axis_sign_validation",
        "path": REPO_ROOT / "configs" / "sedon" / "sedon_duck_joint_sign_mapping.yaml",
        "status_path": ["summary", "status"],
        "pass_statuses": {"READY"},
    },
    {
        "phase": 3,
        "name": "sedon_actuator_envelope",
        "path": REPO_ROOT / "configs" / "sedon" / "sedon_actuator_envelope.yaml",
        "status_path": ["summary", "status"],
        "pass_statuses": {"READY", "INCONCLUSIVE"},
    },
    {
        "phase": 4,
        "name": "sedon_contact_patch_readiness",
        "path": REPO_ROOT / "configs" / "sedon" / "sedon_contact_patch_status.yaml",
        "status_path": ["summary", "status"],
        "pass_statuses": {"READY"},
    },
    {
        "phase": 5,
        "name": "duck_gait_reference_metadata",
        "path": REPO_ROOT / "references" / "open_duck_mini" / "duck_gait_reference_metadata.yaml",
        "status_path": ["summary", "status"],
        "pass_statuses": {"FOUND_CANDIDATES", "READY"},
    },
    {
        "phase": 6,
        "name": "duck_like_task_space_reference",
        "path": REPO_ROOT / "configs" / "sedon" / "sedon_duck_like_gait_reference.yaml",
        "status_path": ["summary", "status"],
        "pass_statuses": {"REFERENCE_DRAFT", "READY"},
    },
    {
        "phase": 7,
        "name": "ik_feasibility",
        "path": REPO_ROOT
        / "artifacts"
        / "sedon_debug"
        / "duck_like_ik_feasibility"
        / "ik_feasibility_summary.yaml",
        "status_path": ["summary", "status"],
        "pass_statuses": {"READY", "FEASIBLE"},
    },
    {
        "phase": 8,
        "name": "scripted_smoke_test",
        "path": REPO_ROOT
        / "artifacts"
        / "sedon_debug"
        / "duck_like_scripted_smoke_test"
        / "scripted_smoke_test_summary.yaml",
        "status_path": ["summary", "status"],
        "pass_statuses": {"PASS", "READY"},
    },
]


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json_yaml(path: Path) -> dict[str, Any]:
    """Load a JSON-compatible YAML file."""

    if not path.is_file():
        raise ValueError(f"Missing phase output: {path}")
    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def nested_get(payload: dict[str, Any], path: list[str]) -> Any:
    """Read nested dictionary value."""

    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def phase_record(definition: dict[str, Any]) -> dict[str, Any]:
    """Build a phase summary record."""

    payload = load_json_yaml(definition["path"])
    status = nested_get(payload, definition["status_path"])
    passed = status in definition["pass_statuses"]
    return {
        "phase": definition["phase"],
        "name": definition["name"],
        "path": str(definition["path"].relative_to(REPO_ROOT)),
        "status": status,
        "passed": passed,
        "pass_statuses": sorted(definition["pass_statuses"]),
        "reason": nested_get(payload, ["summary", "reason"]),
        "blockers": nested_get(payload, ["summary", "blockers"]) or [],
    }


def build_pipeline() -> dict[str, Any]:
    """Build full pipeline summary."""

    phases = [phase_record(definition) for definition in PHASE_OUTPUTS]
    failed = [phase for phase in phases if not phase["passed"]]
    return {
        "tool_version": TOOL_VERSION,
        "generated_at": utc_now_iso(),
        "overall_status": "BLOCKED" if failed else "READY_FOR_PPO_PLAN_REVIEW",
        "ready_for_ppo_plan": not failed,
        "phases": phases,
        "blocking_phases": failed,
        "safety_notes": [
            "This pipeline does not modify train.py or eval.py.",
            "This pipeline does not claim walking success.",
            "PPO/imitation reward planning is blocked until all readiness gates pass.",
        ],
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Write pipeline Markdown report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sedon Duck-Like Readiness Pipeline",
        "",
        "Generated by `tools/sedon/experiments/run_duck_like_readiness_pipeline.py`.",
        "",
        "## Summary",
        "",
        f"- Overall status: `{payload['overall_status']}`",
        f"- Ready for PPO plan review: `{str(payload['ready_for_ppo_plan']).lower()}`",
        f"- Generated at: `{payload['generated_at']}`",
        "",
        "## Phase Status",
        "",
        "| Phase | Name | Status | Passed | Output | Reason / Blockers |",
        "|---:|---|---|---|---|---|",
    ]
    for phase in payload["phases"]:
        blockers = phase["blockers"]
        reason = phase["reason"] or ""
        detail = f"blockers={blockers}" if blockers else reason
        lines.append(
            f"| {phase['phase']} | `{phase['name']}` | `{phase['status']}` | `{phase['passed']}` | `{phase['path']}` | {detail} |"
        )
    lines.extend(["", "## Blocking Phases", ""])
    if payload["blocking_phases"]:
        for phase in payload["blocking_phases"]:
            lines.append(f"- Phase {phase['phase']} `{phase['name']}`: status `{phase['status']}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Safety Notes", ""])
    for note in payload["safety_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    """Generate the readiness pipeline report."""

    args = parse_args()
    try:
        payload = build_pipeline()
        write_report(args.report, payload)
    except ValueError as exc:
        print(f"Readiness pipeline failed: {exc}")
        return 1
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
