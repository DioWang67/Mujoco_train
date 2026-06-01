"""Tests for the Pre-W0 DuckRef readiness consolidation logic."""

from __future__ import annotations

import json
from pathlib import Path

from tools.sedon_phase_pre_w0_duckref_readiness_check import ReadinessConfig, build_summary


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON fixture.

    Parameters:
        path: Destination path.
        payload: JSON-serializable fixture payload.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_config(tmp_path: Path) -> ReadinessConfig:
    """Create an isolated readiness config rooted at tmp_path."""

    return ReadinessConfig(
        m0_dir=tmp_path / "m0",
        g1_dir=tmp_path / "g1",
        actuator_dir=tmp_path / "actuator",
        free_space_dir=tmp_path / "free_space",
        output_dir=tmp_path / "out",
    )


def write_required_inputs(config: ReadinessConfig, *, unknown_actuators: int = 0) -> None:
    """Write the minimum required readiness input artifacts."""

    write_json(
        config.m0_dir / "sedon_duck_scaled_gait_reference.json",
        {
            "sedon_target_vx": 0.141,
            "sedon_foot_clearance_target": 0.043,
            "sedon_action_scale_initial": 0.25,
        },
    )
    write_json(config.m0_dir / "sedon_joint_sign_mapping.json", {"R_joint_hip_pitch": {}})
    write_json(
        config.g1_dir / "phase_g1_summary.json",
        {
            "center_geom_detected_right": True,
            "any_right_center_raw_contact": True,
            "right_center_force_zero_all_steps": False,
            "contact_classifier_unknown_force_ratio": 0.24,
        },
    )
    write_json(
        config.actuator_dir / "sedon_actuator_semantics_summary.json",
        {
            "unknown_like_count": unknown_actuators,
            "duck_action_scale_025_transfer_safe": "inconclusive",
        },
    )
    write_json(
        config.free_space_dir / "sedon_free_space_joint_sign_mapping.json",
        {
            "free_space_joint_sign_validated": True,
            "m0_vs_free_space_sign_disagreements_count": 0,
        },
    )


def test_readiness_allows_conservative_w0_when_evidence_is_clear(tmp_path: Path) -> None:
    """Clear evidence should allow only the conservative W0 smoke test."""

    config = make_config(tmp_path)
    write_required_inputs(config)

    summary = build_summary(config)

    assert summary["readiness_label"] == "READY_FOR_W0_DUCKREF_CONSERVATIVE"
    assert summary["blocking_reasons"] == []
    assert summary["scaled_duckref_too_aggressive_risk"] is True


def test_readiness_blocks_unknown_actuator_semantics(tmp_path: Path) -> None:
    """Unknown actuator semantics should block W0 instead of hiding risk."""

    config = make_config(tmp_path)
    write_required_inputs(config, unknown_actuators=1)

    summary = build_summary(config)

    assert summary["readiness_label"] == "BLOCKED_BY_ACTUATOR_SEMANTICS"
    assert summary["blocking_reasons"] == ["actuator_semantics"]
