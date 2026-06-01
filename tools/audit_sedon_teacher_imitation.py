"""Audit Sedon pose_1..4 teacher-imitation policy against baseline criteria."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.audit_sedon_shuffle_v0 import audit_shuffle, teacher_relative_gate


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "sedon" / "reference_teacher_pose_1_4_imitation.json"


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", "--checkpoint", type=Path, default=None)
    parser.add_argument("--vecnorm-path", type=Path, default=None)
    parser.add_argument("--teacher-baseline-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--teacher-baseline-model-path", type=Path, default=None)
    parser.add_argument("--teacher-baseline-vecnorm-path", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--audit-warmup-steps",
        type=int,
        default=20,
        help="Initial rollout steps excluded from impact/drop audit metrics.",
    )
    parser.add_argument(
        "--random-residual",
        action="store_true",
        help="Use seeded random normalized residual actions instead of zero residuals.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run teacher-imitation audit."""
    args = build_parser().parse_args(argv)
    summary = audit_shuffle(
        args.config,
        args.model_path,
        args.vecnorm_path,
        args.steps,
        args.seed,
        random_residual=args.random_residual,
        audit_warmup_steps=args.audit_warmup_steps,
    )
    teacher = audit_shuffle(
        args.teacher_baseline_config,
        args.teacher_baseline_model_path,
        args.teacher_baseline_vecnorm_path,
        args.steps,
        args.seed,
        audit_warmup_steps=args.audit_warmup_steps,
    )
    gate = teacher_relative_gate(teacher, summary)
    for key, value in summary.__dict__.items():
        if key != "passed":
            print(f"{key}: {value}")
    print(f"gate_failed_reasons: {','.join(gate.reasons) or 'none'}")
    print(f"teacher_passed: {gate.passed}")
    return 0 if gate.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
