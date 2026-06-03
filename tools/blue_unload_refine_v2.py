"""Refine top Blue unload mechanisms with a local unload-only sweep."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from tools.blue_unload_mechanism_search import (
    DEFAULT_BASE_CONFIG,
    REPO_ROOT,
    UnloadCandidate,
    _fmt,
    _parse_float_list,
    audit_candidate,
    render_candidate,
    write_candidate_files,
)


DEFAULT_SOURCE_TOP10 = (
    REPO_ROOT
    / "artifacts"
    / "seedon_debug"
    / "blue_unload_mechanism_search"
    / "blue_unload_mechanism_search_top10.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "seedon_debug" / "blue_unload_refine_v2"


@dataclass(frozen=True)
class RefineAudit:
    """Unload refine audit row with stable and A/B/C pass gates."""

    candidate_id: str
    preload_duration: int
    support_hip_roll: float
    swing_hip_roll: float
    stance_knee: float
    stance_ankle: float
    pelvis_lean_proxy: float
    lateral_offset_proxy: float
    tiny_lift_amplitude: float
    min_swing_force_ratio: float
    max_support_ratio: float
    duration_below_0_45: int
    duration_below_0_40: int
    contact_none_ratio: float
    jump_count: int
    base_drop_post_warmup: float
    impact_post_warmup: float
    upright: float
    stable_gate: bool
    a_passed: bool
    b_passed: bool
    c_passed: bool
    b_margin: float
    score: float
    config_path: str
    seed_path: str
    timeline_path: str


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive integer durations")
    return values


def _unique(values: list[float]) -> list[float]:
    return sorted({round(value, 6) for value in values})


def _near(value: float, deltas: list[float], *, lower: float | None = None, upper: float | None = None) -> list[float]:
    values: list[float] = []
    for delta in deltas:
        candidate = value + delta
        if lower is not None and candidate < lower:
            continue
        if upper is not None and candidate > upper:
            continue
        values.append(candidate)
    return _unique(values)


def _read_top_rows(path: Path, limit: int) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source top10 CSV: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows in source top10 CSV: {path}")
    return rows[:limit]


def build_candidates(args: argparse.Namespace) -> list[UnloadCandidate]:
    """Build a de-duplicated local sweep around source top candidates."""
    source_rows = _read_top_rows(args.source_top10, args.source_top_k)
    candidates: dict[str, UnloadCandidate] = {}
    for row in source_rows:
        support_center = float(row["support_hip_roll"])
        swing_center = float(row["swing_hip_roll"])
        knee_center = float(row["stance_knee"])
        ankle_center = float(row["stance_ankle"])
        support_values = _near(support_center, args.support_deltas, lower=0.0, upper=0.10)
        swing_values = _near(swing_center, args.swing_deltas, lower=-0.06, upper=0.0)
        knee_values = _near(knee_center, args.knee_deltas, lower=-0.07, upper=0.0)
        ankle_values = _near(ankle_center, args.ankle_deltas, lower=0.0, upper=0.06)
        for duration in args.durations:
            for lean in args.leans:
                for lateral in args.laterals:
                    for support_roll in support_values:
                        for swing_roll in swing_values:
                            for knee in knee_values:
                                for ankle in ankle_values:
                                    candidate_id = (
                                        f"dur{duration}_sr{_fmt(support_roll)}_wr{_fmt(swing_roll)}"
                                        f"_sk{_fmt(knee)}_sa{_fmt(ankle)}_lean{_fmt(lean)}"
                                        f"_lat{_fmt(lateral)}_tl0"
                                    )
                                    candidates[candidate_id] = UnloadCandidate(
                                        candidate_id=candidate_id,
                                        preload_duration=duration,
                                        support_hip_roll=support_roll,
                                        swing_hip_roll=swing_roll,
                                        stance_knee=knee,
                                        stance_ankle=ankle,
                                        pelvis_lean_proxy=lean,
                                        lateral_offset_proxy=lateral,
                                        tiny_lift_amplitude=0.0,
                                    )
    return list(candidates.values())


def _score(row: RefineAudit) -> float:
    return (
        (100.0 if row.c_passed else 0.0)
        + (40.0 if row.b_passed else 0.0)
        + (10.0 if row.a_passed else 0.0)
        + (5.0 if row.stable_gate else -100.0)
        + max(0.0, 0.48 - row.min_swing_force_ratio) * 30.0
        + row.duration_below_0_45 * 0.08
        + row.duration_below_0_40 * 0.5
        + row.max_support_ratio * 2.0
        - max(0.0, row.impact_post_warmup - 1.02) * 10.0
        - max(0.0, row.base_drop_post_warmup - 0.002) * 80.0
    )


def evaluate_refine_row(raw: object) -> RefineAudit:
    """Apply refine v2 stable and A/B/C gates to a base unload audit row."""
    stable_gate = (
        raw.contact_none_ratio == 0
        and raw.jump_count == 0
        and raw.upright >= 0.985
        and raw.base_drop_post_warmup <= 0.005
        and raw.impact_post_warmup <= 1.05
    )
    a_passed = (
        stable_gate
        and raw.min_swing_force_ratio <= 0.45
        and raw.duration_below_0_45 >= 20
    )
    b_passed = (
        stable_gate
        and raw.min_swing_force_ratio <= 0.40
        and raw.duration_below_0_40 >= 8
    )
    c_passed = (
        stable_gate
        and raw.min_swing_force_ratio <= 0.38
        and raw.duration_below_0_40 >= 15
    )
    b_margin = max(0.0, raw.min_swing_force_ratio - 0.40) + max(0, 8 - raw.duration_below_0_40) * 0.005
    row = RefineAudit(
        candidate_id=raw.candidate_id,
        preload_duration=raw.preload_duration,
        support_hip_roll=raw.support_hip_roll,
        swing_hip_roll=raw.swing_hip_roll,
        stance_knee=raw.stance_knee,
        stance_ankle=raw.stance_ankle,
        pelvis_lean_proxy=raw.pelvis_lean_proxy,
        lateral_offset_proxy=raw.lateral_offset_proxy,
        tiny_lift_amplitude=raw.tiny_lift_amplitude,
        min_swing_force_ratio=raw.min_swing_force_ratio,
        max_support_ratio=raw.max_support_ratio,
        duration_below_0_45=raw.duration_below_0_45,
        duration_below_0_40=raw.duration_below_0_40,
        contact_none_ratio=raw.contact_none_ratio,
        jump_count=raw.jump_count,
        base_drop_post_warmup=raw.base_drop_post_warmup,
        impact_post_warmup=raw.impact_post_warmup,
        upright=raw.upright,
        stable_gate=stable_gate,
        a_passed=a_passed,
        b_passed=b_passed,
        c_passed=c_passed,
        b_margin=b_margin,
        score=0.0,
        config_path=raw.config_path,
        seed_path=raw.seed_path,
        timeline_path=raw.timeline_path,
    )
    return RefineAudit(**{**asdict(row), "score": _score(row)})


def write_results(path: Path, rows: list[RefineAudit]) -> None:
    """Write aggregate CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(path: Path, rows: list[RefineAudit]) -> None:
    """Write compact summary markdown."""
    ranked = sorted(rows, key=lambda item: (item.c_passed, item.b_passed, item.a_passed, item.score), reverse=True)
    lines = [
        "# Blue Unload Refine V2 Summary",
        "",
        f"Candidates: {len(rows)}",
        f"Stable gate: {sum(1 for row in rows if row.stable_gate)}",
        f"A pass: {sum(1 for row in rows if row.a_passed)}",
        f"B pass: {sum(1 for row in rows if row.b_passed)}",
        f"C pass: {sum(1 for row in rows if row.c_passed)}",
        "",
        "## Top Candidates",
        "",
        "| candidate | stable | A | B | C | min_swing | max_support | below_045 | below_040 | b_margin | impact | drop | upright |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked[:20]:
        lines.append(
            "| "
            f"{row.candidate_id} | {row.stable_gate} | {row.a_passed} | {row.b_passed} | {row.c_passed} | "
            f"{row.min_swing_force_ratio:.3f} | {row.max_support_ratio:.3f} | "
            f"{row.duration_below_0_45} | {row.duration_below_0_40} | {row.b_margin:.3f} | "
            f"{row.impact_post_warmup:.3f} | {row.base_drop_post_warmup:.4f} | {row.upright:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-top10", type=Path, default=DEFAULT_SOURCE_TOP10)
    parser.add_argument("--source-top-k", type=int, default=3)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-warmup-steps", type=int, default=20)
    parser.add_argument("--neutral-duration", type=int, default=30)
    parser.add_argument("--durations", type=_parse_int_list, default="120,140,160,180,200,220")
    parser.add_argument("--leans", type=_parse_float_list, default="0.015,0.020,0.025,0.030,0.035")
    parser.add_argument("--laterals", type=_parse_float_list, default="-0.010,-0.005,0,0.005,0.010")
    parser.add_argument("--support-deltas", type=_parse_float_list, default="0,0.005")
    parser.add_argument("--swing-deltas", type=_parse_float_list, default="0,-0.005")
    parser.add_argument("--knee-deltas", type=_parse_float_list, default="0")
    parser.add_argument("--ankle-deltas", type=_parse_float_list, default="0")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=1200)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--render-top-k", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=368)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run local unload refine v2 sweep."""
    args = build_parser().parse_args(argv)
    candidates = build_candidates(args)
    if len(candidates) > args.max_candidates:
        raise ValueError(
            f"Refine sweep would create {len(candidates)} candidates, "
            f"above --max-candidates={args.max_candidates}. "
            "Narrow --source-top-k or local delta options."
        )
    print(
        f"candidates={len(candidates)} source_top_k={args.source_top_k} "
        f"max_candidates={args.max_candidates}"
    )
    if args.dry_run:
        for candidate in candidates[: min(10, len(candidates))]:
            print(f"candidate: {candidate.candidate_id}")
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[RefineAudit] = []
    for index, candidate in enumerate(candidates, start=1):
        config_path, seed_path = write_candidate_files(
            candidate,
            base_config=args.base_config,
            out_dir=args.out_dir,
            neutral_duration=args.neutral_duration,
        )
        raw = audit_candidate(
            candidate,
            config_path=config_path,
            seed_path=seed_path,
            out_dir=args.out_dir,
            steps=args.steps,
            seed=args.seed,
            warmup_steps=args.audit_warmup_steps,
        )
        row = evaluate_refine_row(raw)
        rows.append(row)
        if args.progress_every > 0 and (
            index == 1 or index == len(candidates) or index % args.progress_every == 0
        ):
            print(
                f"[{index}/{len(candidates)}] {row.candidate_id}: "
                f"stable={row.stable_gate} A={row.a_passed} B={row.b_passed} C={row.c_passed} "
                f"min_swing={row.min_swing_force_ratio:.3f} below045={row.duration_below_0_45} "
                f"below040={row.duration_below_0_40} margin={row.b_margin:.3f}"
            )
    ranked = sorted(rows, key=lambda item: (item.c_passed, item.b_passed, item.a_passed, item.score), reverse=True)
    write_results(args.out_dir / "blue_unload_refine_v2.csv", rows)
    write_results(args.out_dir / "blue_unload_refine_v2_top20.csv", ranked[: args.top_k])
    write_summary(args.out_dir / "summary.md", rows)
    for row in ranked[: args.top_k]:
        print(
            "TOP "
            f"{row.candidate_id}: stable={row.stable_gate} A={row.a_passed} B={row.b_passed} C={row.c_passed} "
            f"score={row.score:.3f} min_swing={row.min_swing_force_ratio:.3f} "
            f"below045={row.duration_below_0_45} below040={row.duration_below_0_40} "
            f"margin={row.b_margin:.3f}"
        )
    for row in ranked[: args.render_top_k]:
        print(
            "rendered: "
            f"{render_candidate(row, steps=args.steps, seed=args.seed, fps=args.fps, width=args.width, height=args.height, out_dir=args.out_dir)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
