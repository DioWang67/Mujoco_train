"""Check Sedon march Phase 1A gates across hip-roll residual scales."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.check_sedon_march_phase12_gates import (
    GateThresholds,
    _evaluate_phase,
    _load_config,
    _load_model,
    _print_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    REPO_ROOT / "configs" / "sedon" / "march_phase1a_load_transfer_roll_0005.json",
    REPO_ROOT / "configs" / "sedon" / "march_phase1a_load_transfer_roll_0010.json",
    REPO_ROOT / "configs" / "sedon" / "march_phase1a_load_transfer_roll_0015.json",
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the Phase 1A gate checker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--vecnorm-path", type=Path, default=None)
    parser.add_argument("--config", type=Path, action="append", default=None)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pass-v1-support-ratio", type=float, default=0.54)
    parser.add_argument("--phase1b-support-ratio", type=float, default=0.58)
    parser.add_argument("--min-both-contact-fraction", type=float, default=0.95)
    parser.add_argument("--max-base-drift", type=float, default=0.05)
    parser.add_argument("--max-abs-base-pitch", type=float, default=0.35)
    parser.add_argument("--max-abs-base-roll", type=float, default=0.08)
    parser.add_argument("--max-torque-saturation", type=float, default=0.10)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Check all configured Phase 1A gates and return nonzero on any failure."""
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")

    config_paths = tuple(args.config) if args.config else DEFAULT_CONFIGS
    model = _load_model(args.model_path)
    all_passed = True

    for offset, config_path in enumerate(config_paths):
        config = _load_config(config_path)
        baseline_thresholds = GateThresholds(
            phase1_support_ratio=0.0,
            stable_seconds=0.0,
            max_abs_base_roll=float("inf"),
            max_abs_base_pitch=float("inf"),
            max_torque_saturation=float("inf"),
            min_both_contact_fraction=0.0,
            max_base_drift=float("inf"),
        )
        baseline = _evaluate_phase(
            phase=1,
            config=config,
            model=None,
            vecnorm_path=None,
            seed=args.seed + offset,
            steps=args.steps,
            thresholds=baseline_thresholds,
        )
        thresholds = GateThresholds(
            phase1_support_ratio=args.pass_v1_support_ratio,
            stable_seconds=baseline.duration_s,
            max_abs_base_roll=args.max_abs_base_roll,
            max_abs_base_pitch=args.max_abs_base_pitch,
            max_torque_saturation=args.max_torque_saturation,
            min_both_contact_fraction=args.min_both_contact_fraction,
            max_base_drift=args.max_base_drift,
        )
        candidate = _evaluate_phase(
            phase=1,
            config=config,
            model=model,
            vecnorm_path=args.vecnorm_path,
            seed=args.seed + offset,
            steps=args.steps,
            thresholds=thresholds,
        )
        both_contact_failure = (
            candidate.both_contact_fraction < args.min_both_contact_fraction
            and "both_contact_fraction" not in candidate.failed_checks
        )
        failed_checks = candidate.failed_checks
        if both_contact_failure:
            failed_checks = (*failed_checks, "both_contact_fraction")
        passed = candidate.passed and not both_contact_failure
        print(f"\nconfig={config_path}")
        print(f"scripted_baseline_duration={baseline.duration_s:.2f}s")
        print(
            "phase1a_v2_gate="
            f"pass_v1_ratio>={args.pass_v1_support_ratio:.2f}, "
            f"phase1b_ratio>={args.phase1b_support_ratio:.2f}"
        )
        _print_summary(
            type(candidate)(
                **{
                    **candidate.__dict__,
                    "passed": passed,
                    "failed_checks": failed_checks,
                }
            )
        )
        print(
            "  "
            f"pass_v1={passed} "
            f"ready_for_1b={passed and candidate.max_support_ratio >= args.phase1b_support_ratio}"
        )
        all_passed = all_passed and passed

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
