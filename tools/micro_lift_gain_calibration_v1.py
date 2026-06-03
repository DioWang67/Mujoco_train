"""Calibrate micro-lift gain and classify whether a stable usable band exists."""

from __future__ import annotations

import argparse
from pathlib import Path

from tools.micro_lift_mapping_search_v1 import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_SOURCE_TOP,
    MicroLiftAudit,
    MicroLiftCandidate,
    _mapping_delta,
    build_candidates as build_base_candidates,
    build_seed,
    render_candidate,
    write_results,
    write_summary,
)


DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "seedon_debug" / "micro_lift_gain_calibration_v1"


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _classify(rows: list[MicroLiftAudit]) -> tuple[str, str]:
    usable = [
        row
        for row in rows
        if 0.0010 <= row.clearance <= 0.0020
        and row.min_upright >= 0.99
        and row.impact_post <= 1.25
        and row.contact_none_ratio == 0.0
        and row.jump_count == 0
    ]
    if usable:
        return "usable_band_found", f"{len(usable)} candidate(s) met the usable 1.0-2.0mm stable band."
    lifted = [row for row in rows if row.clearance >= 0.0010]
    if lifted and all(row.min_upright < 0.99 or row.impact_post > 1.25 for row in lifted):
        return "cliff_behavior", "clearance appeared only with posture or impact failure."
    if max((row.clearance for row in rows), default=0.0) < 0.0010:
        return "ineffective_mapping", "gain reached the requested maximum without dynamic clearance."
    return "inconclusive", "some clearance appeared, but it did not cleanly match usable/cliff/ineffective categories."


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-top", type=Path, default=DEFAULT_SOURCE_TOP)
    parser.add_argument("--source-top-k", type=int, default=3)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-warmup-steps", type=int, default=20)
    parser.add_argument("--neutral-duration", type=int, default=30)
    parser.add_argument("--base-target-clearance", type=float, default=0.0015)
    parser.add_argument("--gains", type=_parse_float_list, default="1.0,1.5,2.0,2.5,3.0,3.5,4.0")
    parser.add_argument(
        "--mapping-families",
        nargs="+",
        default=["knee_only", "hip_knee", "knee_ankle", "hip_knee_ankle_small_counter"],
    )
    parser.add_argument("--lift-durations", type=lambda raw: [int(x.strip()) for x in raw.split(",") if x.strip()], default="45,60")
    parser.add_argument("--landing-durations", type=lambda raw: [int(x.strip()) for x in raw.split(",") if x.strip()], default="90,120")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--render-top-k", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=368)
    return parser


def _gain_candidate(base: MicroLiftCandidate, gain: float, base_target_clearance: float) -> MicroLiftCandidate:
    return MicroLiftCandidate(
        candidate_id=f"{base.candidate_id}_gain{gain:.2f}".replace(".", "p"),
        source_candidate_id=base.source_candidate_id,
        source_timeline_path=base.source_timeline_path,
        unload=base.unload,
        right_window=base.right_window,
        left_window=base.left_window,
        target_clearance=base_target_clearance * gain,
        mapping_family=base.mapping_family,
        lift_duration=base.lift_duration,
        landing_duration=base.landing_duration,
    )


def main(argv: list[str] | None = None) -> int:
    """Run gain calibration and classify the lift behavior."""
    from tools.audit_seedon_shuffle_v0 import audit_shuffle
    from tools.micro_lift_mapping_search_v1 import audit_candidate, write_candidate_files

    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_args = argparse.Namespace(
        source_top=args.source_top,
        source_top_k=args.source_top_k,
        target_clearances=[args.base_target_clearance],
        mapping_families=args.mapping_families,
        lift_durations=args.lift_durations,
        landing_durations=args.landing_durations,
    )
    base_candidates = build_base_candidates(base_args)
    candidates = [_gain_candidate(candidate, gain, args.base_target_clearance) for candidate in base_candidates for gain in args.gains]
    baseline = audit_shuffle(
        args.base_config,
        None,
        None,
        args.steps,
        args.seed,
        audit_warmup_steps=args.audit_warmup_steps,
    )
    rows: list[MicroLiftAudit] = []
    print(f"candidates={len(candidates)}")
    for index, candidate in enumerate(candidates, start=1):
        config_path, seed_path = write_candidate_files(
            candidate,
            base_config=args.base_config,
            out_dir=args.out_dir,
            neutral_duration=args.neutral_duration,
        )
        row = audit_candidate(
            candidate,
            config_path=config_path,
            seed_path=seed_path,
            baseline_impact=baseline.landing_impact_post_warmup,
            out_dir=args.out_dir,
            steps=args.steps,
            seed=args.seed,
            warmup_steps=args.audit_warmup_steps,
        )
        rows.append(row)
        if index == 1 or index == len(candidates) or index % 50 == 0:
            print(
                f"[{index}/{len(candidates)}] {row.candidate_id}: "
                f"clear={row.clearance:.4f} upright={row.min_upright:.3f} "
                f"impact={row.impact_post:.3f} A={row.a_passed} B={row.b_passed} C={row.c_passed}"
            )
    ranked = sorted(rows, key=lambda item: (item.c_passed, item.b_passed, item.a_passed, item.score), reverse=True)
    write_results(args.out_dir / "micro_lift_gain_calibration_v1.csv", rows)
    write_results(args.out_dir / "micro_lift_gain_calibration_v1_top10.csv", ranked[: args.top_k])
    write_summary(args.out_dir / "summary.md", rows)
    classification, reason = _classify(rows)
    with (args.out_dir / "classification.txt").open("w", encoding="utf-8") as file:
        file.write(f"{classification}\n{reason}\n")
    print(f"classification={classification}: {reason}")
    for row in ranked[: args.top_k]:
        print(
            "TOP "
            f"{row.candidate_id}: clear={row.clearance:.4f} upright={row.min_upright:.3f} "
            f"impact={row.impact_post:.3f} drop={row.base_drop_post:.4f} "
            f"A={row.a_passed} B={row.b_passed} C={row.c_passed}"
        )
    for row in ranked[: args.render_top_k]:
        print(
            "rendered: "
            f"{render_candidate(row, steps=args.steps, seed=args.seed, fps=args.fps, width=args.width, height=args.height, out_dir=args.out_dir)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
