"""Search local Open Duck Mini reference files for gait/reference motion sources.

This extractor is intentionally conservative: it scans files already present in
the repository and records candidates. It does not download data, infer gait
periods from robot XML, or generate Sedon trajectories.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_VERSION = "duck-gait-reference-metadata-v1"
UNKNOWN = "unknown"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEARCH_ROOT = REPO_ROOT / "references" / "open_duck_mini"
DEFAULT_OUTPUT = REPO_ROOT / "references" / "open_duck_mini" / "duck_gait_reference_metadata.yaml"
DEFAULT_REPORT = REPO_ROOT / "docs" / "open_duck_gait_reference_index.md"

GAIT_KEYWORDS = (
    "gait",
    "walk",
    "walking",
    "trajectory",
    "traj",
    "motion",
    "reference",
    "step",
    "phase",
    "controller",
    "policy",
    "demo",
    "imitation",
)
REFERENCE_EXTENSIONS = {
    ".csv",
    ".json",
    ".yaml",
    ".yml",
    ".npz",
    ".npy",
    ".pkl",
    ".txt",
    ".xml",
}
ROBOT_STRUCTURE_EXTENSIONS = {".xml", ".stl", ".part", ".png"}


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON-compatible YAML output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=False)
        file.write("\n")


def candidate_reason(path: Path) -> tuple[bool, str, str]:
    """Classify whether a file could be a gait/reference source."""

    lower_name = path.name.lower()
    suffix = path.suffix.lower()
    keyword_hits = [keyword for keyword in GAIT_KEYWORDS if keyword in lower_name]
    if keyword_hits and suffix in REFERENCE_EXTENSIONS:
        if suffix == ".xml" and any(token in lower_name for token in ("scene", "sensor", "joint", "properties")):
            return False, "robot_or_scene_metadata", "XML metadata, not a gait/reference motion source."
        return True, "candidate_by_filename", f"Filename contains gait/reference keyword(s): {keyword_hits}."
    if suffix in ROBOT_STRUCTURE_EXTENSIONS:
        return False, "robot_structure_or_asset", "Robot XML/mesh/asset file; no motion timing or trajectory evidence."
    return False, "not_candidate", "No gait/reference keyword or supported motion data extension."


def scan_files(search_root: Path) -> dict[str, Any]:
    """Scan a local Duck reference tree for gait/reference motion candidates."""

    if not search_root.exists():
        raise ValueError(f"Search root not found: {search_root}")

    files = []
    candidates = []
    for path in sorted(search_root.rglob("*")):
        if not path.is_file():
            continue
        is_candidate, classification, reason = candidate_reason(path)
        relative = path.relative_to(REPO_ROOT).as_posix()
        record = {
            "path": relative,
            "extension": path.suffix.lower() or UNKNOWN,
            "size_bytes": path.stat().st_size,
            "classification": classification,
            "candidate": is_candidate,
            "source": "local repository file scan",
            "confidence": "low" if is_candidate else "high",
            "reason": reason,
        }
        files.append(record)
        if is_candidate:
            candidates.append(record)
    status = "FOUND_CANDIDATES" if candidates else "NOT_FOUND"
    return {
        "schema_version": 1,
        "tool_version": TOOL_VERSION,
        "generated_at": utc_now_iso(),
        "purpose": "Identify local Duck gait/reference motion source metadata for Sedon Duck-like gait preparation.",
        "search_root": str(search_root),
        "summary": {
            "status": status,
            "files_scanned": len(files),
            "candidate_count": len(candidates),
            "gait_period_available": False,
            "clearance_available": False,
            "joint_trajectory_available": False,
            "task_space_reference_available": False,
            "reason": "No validated gait/reference motion source found in local Duck reference files."
            if not candidates
            else "Filename candidates exist, but each candidate requires manual validation before use.",
        },
        "candidates": candidates,
        "scanned_files": files,
        "manual_required": [
            {
                "field": "duck_gait_period",
                "reason": "Not present in robot XML/source files scanned.",
            },
            {
                "field": "duck_step_clearance",
                "reason": "Not present in robot XML/source files scanned.",
            },
            {
                "field": "duck_reference_motion",
                "reason": "No validated local trajectory/controller/reference file identified.",
            },
        ],
        "safety_notes": [
            "Do not infer gait period or clearance from robot morphology XML.",
            "Do not treat scene XML files as gait references.",
            "Do not transfer any Duck joint trajectory without a validated source.",
        ],
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Write Duck gait reference index report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    lines = [
        "# Open Duck Gait Reference Index",
        "",
        "Generated by `tools/sedon/extractors/extract_duck_gait_reference_metadata.py`.",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Files scanned: `{summary['files_scanned']}`",
        f"- Candidate count: `{summary['candidate_count']}`",
        "- Gait period available: `false`",
        "- Step clearance available: `false`",
        "- Joint trajectory available: `false`",
        "- Task-space reference available: `false`",
        f"- Reason: {summary['reason']}",
        "",
        "## Candidates",
        "",
    ]
    if payload["candidates"]:
        lines.extend(["| Path | Classification | Confidence | Reason |", "|---|---|---|---|"])
        for candidate in payload["candidates"]:
            lines.append(
                f"| `{candidate['path']}` | `{candidate['classification']}` | `{candidate['confidence']}` | {candidate['reason']} |"
            )
    else:
        lines.append("No validated Duck gait/reference motion source was found in the local reference tree.")

    lines.extend(["", "## Manual Required / Unknown", ""])
    for item in payload["manual_required"]:
        lines.append(f"- `{item['field']}`: {item['reason']}")

    lines.extend(["", "## Safety Notes", ""])
    for note in payload["safety_notes"]:
        lines.append(f"- {note}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, default=DEFAULT_SEARCH_ROOT)
    parser.add_argument("--output-yaml", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    """Run Duck gait/reference metadata extraction."""

    args = parse_args()
    try:
        payload = scan_files(args.search_root)
        write_json_yaml(args.output_yaml, payload)
        write_report(args.report, payload)
    except ValueError as exc:
        print(f"Duck gait reference metadata extraction failed: {exc}")
        return 1
    print(f"Wrote {args.output_yaml}")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
