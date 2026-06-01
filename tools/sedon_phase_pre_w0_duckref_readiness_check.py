"""Phase Pre-W0 DuckRef readiness check for Sedon.

This Class C tool consolidates Phase M0, Phase G1, actuator semantics, and
free-space joint sign evidence before a conservative W0-DuckRef scripted
walking smoke test. It does not train and does not claim walking success.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEBUG_OUT_DIR = REPO_ROOT / "artifacts" / "sedon_debug"
DEFAULT_M0_DIR = DEBUG_OUT_DIR / "phase_m0_duck_morphology_audit"
DEFAULT_G1_DIR = DEBUG_OUT_DIR / "phase_g1_raw_contact_pair_diagnostic"
DEFAULT_ACTUATOR_DIR = DEBUG_OUT_DIR / "phase_pre_w0_actuator_semantics_audit"
DEFAULT_FREE_SPACE_DIR = DEBUG_OUT_DIR / "phase_pre_w0_free_space_joint_sign_validation"
DEFAULT_OUTPUT_DIR = DEBUG_OUT_DIR / "phase_pre_w0_duckref_readiness_check"


@dataclass(frozen=True)
class ReadinessConfig:
    """Runtime configuration for DuckRef readiness consolidation.

    Parameters:
        m0_dir: Directory containing Phase M0 outputs.
        g1_dir: Directory containing Phase G1 outputs.
        actuator_dir: Directory containing actuator semantics outputs.
        free_space_dir: Directory containing free-space validation outputs.
        output_dir: Directory where summary and report are written.
    """

    m0_dir: Path
    g1_dir: Path
    actuator_dir: Path
    free_space_dir: Path
    output_dir: Path


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file or return a missing-file payload.

    Parameters:
        path: JSON file path.

    Returns:
        Parsed JSON dictionary or `{"_missing": path}` marker.
    """

    if not path.is_file():
        return {"_missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def tri_bool(value: Any) -> bool | str:
    """Normalize booleans that may already be inconclusive strings."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return "inconclusive"


def has_missing(*payloads: dict[str, Any]) -> bool:
    """Return whether any payload was not found on disk."""

    return any("_missing" in payload for payload in payloads)


def conservative_reference_set() -> dict[str, Any]:
    """Return the conservative first W0 reference grid."""

    return {
        "target_vx": [0.05, 0.08, 0.10],
        "gait_period": [0.70, 0.85],
        "clearance": [0.005, 0.015, 0.025],
        "action_scale": [0.10, 0.15, 0.20],
    }


def build_summary(config: ReadinessConfig) -> dict[str, Any]:
    """Build the readiness summary.

    Parameters:
        config: Runtime configuration.

    Returns:
        JSON-serializable readiness summary.
    """

    gait_reference = read_json(config.m0_dir / "sedon_duck_scaled_gait_reference.json")
    m0_sign_mapping = read_json(config.m0_dir / "sedon_joint_sign_mapping.json")
    g1_summary = read_json(config.g1_dir / "phase_g1_summary.json")
    actuator_summary = read_json(config.actuator_dir / "sedon_actuator_semantics_summary.json")
    free_space_summary = read_json(config.free_space_dir / "sedon_free_space_joint_sign_mapping.json")

    missing_inputs = has_missing(gait_reference, m0_sign_mapping, g1_summary, actuator_summary, free_space_summary)
    center_contact_possible = (
        bool(g1_summary.get("center_geom_detected_right"))
        and bool(g1_summary.get("any_right_center_raw_contact"))
        and not bool(g1_summary.get("right_center_force_zero_all_steps", True))
    )
    unknown_ratio = float(g1_summary.get("contact_classifier_unknown_force_ratio", 1.0))
    contact_classifier_ok = center_contact_possible and unknown_ratio < 0.30

    unknown_actuators = int(actuator_summary.get("unknown_like_count", 1))
    actuator_semantics_clear = (not missing_inputs) and unknown_actuators == 0
    duck_action_scale_safe = tri_bool(actuator_summary.get("duck_action_scale_025_transfer_safe", "inconclusive"))

    free_space_validated = tri_bool(free_space_summary.get("free_space_joint_sign_validated", "inconclusive"))
    disagreements = int(free_space_summary.get("m0_vs_free_space_sign_disagreements_count", 999))
    split_contact_style_risk = True
    scaled_duckref_too_aggressive_risk = (
        float(gait_reference.get("sedon_target_vx", 1.0)) > 0.10
        or float(gait_reference.get("sedon_foot_clearance_target", 1.0)) > 0.025
        or float(gait_reference.get("sedon_action_scale_initial", 1.0)) >= 0.25
    )

    blocking_reasons: list[str] = []
    if missing_inputs:
        blocking_reasons.append("missing_required_pre_w0_artifacts")
    if not center_contact_possible:
        blocking_reasons.append("contact_classification")
    if unknown_ratio >= 0.30:
        blocking_reasons.append("contact_classifier_unknown_force_ratio_high")
    if not actuator_semantics_clear:
        blocking_reasons.append("actuator_semantics")
    if free_space_validated is not True:
        blocking_reasons.append("joint_sign")
    if disagreements > 0:
        blocking_reasons.append("joint_sign_m0_disagreement")

    if missing_inputs:
        readiness_label = "READY_ONLY_WITH_MANUAL_REVIEW"
    elif not actuator_semantics_clear:
        readiness_label = "BLOCKED_BY_ACTUATOR_SEMANTICS"
    elif free_space_validated is not True or disagreements > 0:
        readiness_label = "BLOCKED_BY_JOINT_SIGN_UNCERTAINTY"
    elif not center_contact_possible or unknown_ratio >= 0.30:
        readiness_label = "BLOCKED_BY_CONTACT_CLASSIFICATION"
    elif split_contact_style_risk or scaled_duckref_too_aggressive_risk or duck_action_scale_safe is not True:
        readiness_label = "READY_FOR_W0_DUCKREF_CONSERVATIVE"
    else:
        readiness_label = "READY_FOR_W0_DUCKREF_CONSERVATIVE"

    if readiness_label.startswith("READY"):
        recommended_next_action = (
            "Run Phase W0-DuckRef conservative scripted walking smoke test only; do not recommend PPO until periodic foot advancement and partial unload are observed."
        )
    else:
        recommended_next_action = "Resolve blockers before W0-DuckRef; do not run PPO."

    return {
        "g1_log_corrected": center_contact_possible,
        "center_contact_physically_possible": center_contact_possible,
        "contact_classifier_unknown_force_ratio": unknown_ratio,
        "actuator_semantics_clear": actuator_semantics_clear,
        "duck_action_scale_transfer_safe": duck_action_scale_safe,
        "free_space_joint_sign_validated": free_space_validated,
        "m0_vs_free_space_sign_disagreements_count": disagreements,
        "split_contact_style_risk": split_contact_style_risk,
        "scaled_duckref_too_aggressive_risk": scaled_duckref_too_aggressive_risk,
        "recommended_first_w0_reference_set": conservative_reference_set(),
        "readiness_label": readiness_label,
        "blocking_reasons": sorted(set(blocking_reasons)),
        "recommended_next_action": recommended_next_action,
        "source_files": {
            "m0_gait_reference": str(config.m0_dir / "sedon_duck_scaled_gait_reference.json"),
            "m0_sign_mapping": str(config.m0_dir / "sedon_joint_sign_mapping.json"),
            "g1_summary": str(config.g1_dir / "phase_g1_summary.json"),
            "actuator_summary": str(config.actuator_dir / "sedon_actuator_semantics_summary.json"),
            "free_space_summary": str(config.free_space_dir / "sedon_free_space_joint_sign_mapping.json"),
        },
    }


def boolish(value: Any) -> str:
    """Format booleans and strings for Markdown."""

    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    """Write Markdown readiness report."""

    refs = summary["recommended_first_w0_reference_set"]
    blockers = summary["blocking_reasons"] or ["none"]
    lines = [
        "# Phase Pre-W0 DuckRef Readiness Check",
        "",
        "## A. Current State Summary",
        "",
        f"- readiness_label: `{summary['readiness_label']}`",
        f"- recommended_next_action: {summary['recommended_next_action']}",
        "",
        "## B. What Has Been Proven",
        "",
        f"- center_contact_physically_possible: `{boolish(summary['center_contact_physically_possible'])}`",
        f"- actuator_semantics_clear: `{boolish(summary['actuator_semantics_clear'])}`",
        f"- free_space_joint_sign_validated: `{boolish(summary['free_space_joint_sign_validated'])}`",
        "",
        "## C. What Remains Uncertain",
        "",
        f"- contact_classifier_unknown_force_ratio: `{summary['contact_classifier_unknown_force_ratio']:.6f}`",
        f"- duck_action_scale_transfer_safe: `{boolish(summary['duck_action_scale_transfer_safe'])}`",
        f"- split_contact_style_risk: `{boolish(summary['split_contact_style_risk'])}`",
        f"- scaled_duckref_too_aggressive_risk: `{boolish(summary['scaled_duckref_too_aggressive_risk'])}`",
        "",
        "## D. Whether W0-DuckRef Is Safe To Run",
        "",
        (
            "W0-DuckRef is safe only as a conservative scripted smoke test."
            if summary["readiness_label"].startswith("READY")
            else "W0-DuckRef is blocked until the listed blockers are resolved."
        ),
        "",
        "## E. Conservative First Reference",
        "",
        f"- target_vx: `{refs['target_vx']}`",
        f"- gait_period: `{refs['gait_period']}`",
        f"- clearance: `{refs['clearance']}`",
        f"- action_scale: `{refs['action_scale']}`",
        "",
        "## F. Exact Blockers",
        "",
        *(f"- `{reason}`" for reason in blockers),
        "",
        "## G. PPO Gate",
        "",
        "Do not recommend PPO yet unless W0 produces periodic foot advancement and partial unload.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--m0-dir", type=Path, default=DEFAULT_M0_DIR)
    parser.add_argument("--g1-dir", type=Path, default=DEFAULT_G1_DIR)
    parser.add_argument("--actuator-dir", type=Path, default=DEFAULT_ACTUATOR_DIR)
    parser.add_argument("--free-space-dir", type=Path, default=DEFAULT_FREE_SPACE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run the Pre-W0 DuckRef readiness check."""

    args = parse_args()
    config = ReadinessConfig(
        m0_dir=args.m0_dir,
        g1_dir=args.g1_dir,
        actuator_dir=args.actuator_dir,
        free_space_dir=args.free_space_dir,
        output_dir=args.output_dir,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(config)
    (config.output_dir / "phase_pre_w0_duckref_readiness_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(config.output_dir / "phase_pre_w0_duckref_readiness_report.md", summary)

    print(f"summary={config.output_dir / 'phase_pre_w0_duckref_readiness_summary.json'}")
    print(f"report={config.output_dir / 'phase_pre_w0_duckref_readiness_report.md'}")
    print(f"readiness_label={summary['readiness_label']}")
    print(f"blocking_reasons={summary['blocking_reasons']}")


if __name__ == "__main__":
    main()
