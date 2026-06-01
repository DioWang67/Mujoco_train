import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("mujoco", types.ModuleType("mujoco"))

from tools.audit_sedon_shuffle_v0 import ShuffleAudit, audit_shuffle, teacher_relative_gate
from tools.audit_sedon_teacher_imitation import build_parser


def test_teacher_audit_parser_accepts_random_residual_and_checkpoint_alias() -> None:
    args = build_parser().parse_args(
        [
            "--steps",
            "480",
            "--random-residual",
            "--checkpoint",
            "models/sedon/latest_model.zip",
            "--audit-warmup-steps",
            "20",
        ]
    )

    assert args.steps == 480
    assert args.random_residual is True
    assert args.model_path == Path("models/sedon/latest_model.zip")
    assert args.audit_warmup_steps == 20


def test_random_residual_rejects_model_path() -> None:
    with pytest.raises(ValueError, match="random_residual"):
        audit_shuffle(
            Path("configs/sedon/reference_teacher_pose_1_4_imitation.json"),
            Path("models/sedon/latest_model.zip"),
            None,
            steps=1,
            seed=0,
            random_residual=True,
        )


def _audit_with_impacts(raw_impact: float, post_impact: float) -> ShuffleAudit:
    return ShuffleAudit(
        steps=480,
        audit_warmup_steps=20,
        terminated=False,
        termination_reason="none",
        both_contact_ratio=1.0,
        single_contact_ratio=0.0,
        contact_none_ratio=0.0,
        peak_support_ratio=0.602,
        min_swing_ratio=0.0,
        max_clearance=0.00102,
        base_height_drop_raw=0.018,
        base_height_drop_post_warmup=0.0082,
        base_height_drop=0.0082,
        jump_count=0,
        landing_impact_raw=raw_impact,
        landing_impact_post_warmup=post_impact,
        landing_impact=post_impact,
        max_contact_force_raw=raw_impact * 100.0,
        max_contact_force_post_warmup=post_impact * 100.0,
        foot_velocity_near_contact_raw=0.0014,
        foot_velocity_near_contact_post_warmup=0.000104,
        mean_tracking_error=0.0364,
        max_tracking_error=0.042,
        tracking_error_variance=0.000017,
        contact_transition_ratio=0.0,
        reward_gate_active_ratio=1.0,
        max_abs_forward_drift=0.036,
        passed=True,
    )


def test_teacher_relative_gate_ignores_raw_reset_spike() -> None:
    teacher = _audit_with_impacts(raw_impact=1.04, post_impact=1.013)
    probe = _audit_with_impacts(raw_impact=1.51, post_impact=1.013)

    gate = teacher_relative_gate(teacher, probe)

    assert gate.passed is True
    assert gate.reasons == ()
