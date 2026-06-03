"""Compare Seedon teacher baseline against a 25k autonomy-stage1 probe."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from tools.audit_seedon_shuffle_v0 import ShuffleAudit, audit_shuffle, teacher_relative_gate


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER_CONFIG = REPO_ROOT / "configs" / "seedon" / "reference_teacher_pose_1_4_imitation.json"
DEFAULT_PROBE_CONFIG = REPO_ROOT / "configs" / "seedon" / "autonomy_stage1_teacher_curriculum.json"
DEFAULT_CHECKPOINT = REPO_ROOT / "models" / "seedon" / "latest_model.zip"
DEFAULT_VECNORM = REPO_ROOT / "models" / "seedon" / "vecnorm.pkl"
DEFAULT_OUT_CSV = REPO_ROOT / "artifacts" / "seedon_debug" / "autonomy_stage1_probe_25k_report.csv"


@dataclass(frozen=True)
class ProbeComparisonRow:
    """One teacher/probe comparison row."""

    label: str
    contact_none_ratio: float
    jump_count: int
    peak_support_ratio: float
    clearance: float
    base_height_drop: float
    base_height_drop_post_warmup: float
    landing_impact: float
    landing_impact_post_warmup: float
    max_contact_force_post_warmup: float
    foot_velocity_near_contact_post_warmup: float
    mean_tracking_error: float
    tracking_error_variance: float
    contact_transition_ratio: float


def _row(label: str, audit: ShuffleAudit) -> ProbeComparisonRow:
    """Convert an audit summary into a report row."""
    return ProbeComparisonRow(
        label=label,
        contact_none_ratio=audit.contact_none_ratio,
        jump_count=audit.jump_count,
        peak_support_ratio=audit.peak_support_ratio,
        clearance=audit.max_clearance,
        base_height_drop=audit.base_height_drop,
        base_height_drop_post_warmup=audit.base_height_drop_post_warmup,
        landing_impact=audit.landing_impact,
        landing_impact_post_warmup=audit.landing_impact_post_warmup,
        max_contact_force_post_warmup=audit.max_contact_force_post_warmup,
        foot_velocity_near_contact_post_warmup=(
            audit.foot_velocity_near_contact_post_warmup
        ),
        mean_tracking_error=audit.mean_tracking_error,
        tracking_error_variance=audit.tracking_error_variance,
        contact_transition_ratio=audit.contact_transition_ratio,
    )


def probe_passed(teacher: ShuffleAudit, probe: ShuffleAudit) -> bool:
    """Return whether the probe satisfies strict teacher-relative gates."""
    return teacher_relative_gate(teacher, probe).passed


def write_report(path: Path, rows: list[ProbeComparisonRow]) -> None:
    """Write comparison rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-config", type=Path, default=DEFAULT_TEACHER_CONFIG)
    parser.add_argument("--probe-config", type=Path, default=DEFAULT_PROBE_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vecnorm-path", type=Path, default=DEFAULT_VECNORM)
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-warmup-steps", type=int, default=20)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run teacher-vs-probe comparison."""
    args = build_parser().parse_args(argv)
    teacher = audit_shuffle(
        args.teacher_config,
        None,
        None,
        args.steps,
        args.seed,
        audit_warmup_steps=args.audit_warmup_steps,
    )
    probe = audit_shuffle(
        args.probe_config,
        args.checkpoint,
        args.vecnorm_path,
        args.steps,
        args.seed,
        audit_warmup_steps=args.audit_warmup_steps,
    )
    rows = [_row("teacher_baseline", teacher), _row("autonomy_stage1_probe", probe)]
    write_report(args.out_csv, rows)
    gate = teacher_relative_gate(teacher, probe)
    print(f"wrote: {args.out_csv}")
    for row in rows:
        print(
            f"{row.label}: none={row.contact_none_ratio:.3f} jump={row.jump_count} "
            f"support={row.peak_support_ratio:.3f} clearance={row.clearance:.6f} "
            f"drop={row.base_height_drop_post_warmup:.5f} "
            f"impact={row.landing_impact_post_warmup:.3f} "
            f"force={row.max_contact_force_post_warmup:.2f} "
            f"foot_v={row.foot_velocity_near_contact_post_warmup:.6f} "
            f"track={row.mean_tracking_error:.5f} "
            f"track_var={row.tracking_error_variance:.8f} "
            f"contact_transition={row.contact_transition_ratio:.5f}"
        )
    print(f"gate_failed_reasons: {','.join(gate.reasons) or 'none'}")
    print(f"probe_passed: {gate.passed}")
    return 0 if gate.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
