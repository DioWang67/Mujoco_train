"""Refine landing phases for lift-after-unload candidates."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sedon_baseline.env import SedonStandingEnv
from tools.audit_sedon_shuffle_v0 import _count_contact_none_bursts, _load_config, audit_shuffle
from tools.blue_unload_mechanism_search import DEFAULT_BASE_CONFIG, JOINT_NAMES, REPO_ROOT, _contact_state
from tools.render_sedon_policy_comparison import _make_side_camera, _save_mp4


DEFAULT_SOURCE_TOP = (
    REPO_ROOT
    / "artifacts"
    / "sedon_debug"
    / "lift_after_unload_v1"
    / "lift_after_unload_v1_top10.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "sedon_debug" / "soft_landing_refine_v1"


@dataclass(frozen=True)
class SoftLandingCandidate:
    """One landing-only refinement candidate."""

    candidate_id: str
    source_candidate_id: str
    source_seed_path: str
    landing_duration: int
    interpolation: str
    lift_hold_duration: int


@dataclass(frozen=True)
class SoftLandingAudit:
    """Dynamic audit row for one soft landing candidate."""

    candidate_id: str
    source_candidate_id: str
    landing_duration: int
    interpolation: str
    lift_hold_duration: int
    clearance: float
    min_swing_force_ratio: float
    single_contact_ratio: float
    impact_post: float
    max_contact_force_post: float
    base_drop_post: float
    upright: float
    foot_velocity_near_contact: float
    contact_none_ratio: float
    jump_count: int
    a_passed: bool
    b_passed: bool
    c_passed: bool
    score: float
    config_path: str
    seed_path: str
    timeline_path: str


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("expected non-negative integer values")
    return values


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_source_rows(path: Path, limit: int) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source top candidates CSV: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows in source top candidates CSV: {path}")
    filtered = [
        row
        for row in rows
        if float(row["clearance"]) >= 0.0015
        and (float(row["single_contact_ratio"]) > 0.0 or float(row.get("near_zero_force_ratio", 0.0)) > 0.0)
    ]
    return filtered[:limit]


def _curve(name: str) -> Callable[[float], float]:
    if name == "smoothstep":
        return lambda t: t * t * (3.0 - 2.0 * t)
    if name == "cubic_ease_out":
        return lambda t: 1.0 - (1.0 - t) ** 3
    if name == "minimum_jerk":
        return lambda t: 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
    raise ValueError(f"Unsupported interpolation: {name}")


def _kf(name: str, support_mode: str, target: np.ndarray, duration: int) -> dict[str, Any]:
    return {
        "name": name,
        "support_mode": support_mode,
        "joint_targets": target.tolist(),
        "duration_steps": max(1, int(duration)),
    }


def _keyframe(seed: dict[str, Any], name: str) -> dict[str, Any]:
    for keyframe in seed["keyframes"]:
        if keyframe["name"] == name:
            return keyframe
    raise ValueError(f"Missing keyframe {name} in source seed")


def _target(keyframe: dict[str, Any]) -> np.ndarray:
    return np.asarray(keyframe["joint_targets"], dtype=np.float64)


def _landing_segments(
    *,
    prefix: str,
    support_mode: str,
    lift_target: np.ndarray,
    unload_target: np.ndarray,
    soft_land_target: np.ndarray,
    landing_duration: int,
    interpolation: str,
) -> list[dict[str, Any]]:
    curve = _curve(interpolation)
    low_clearance_target = unload_target + (lift_target - unload_target) * 0.18
    touchdown_target = soft_land_target * 0.80 + low_clearance_target * 0.20
    segment_count = 5
    base_duration = max(1, landing_duration // segment_count)
    durations = [base_duration] * segment_count
    durations[-1] += max(0, landing_duration - sum(durations))
    segments: list[dict[str, Any]] = []
    for index in range(segment_count):
        progress = curve((index + 1) / segment_count)
        if index < segment_count - 1:
            target = (1.0 - progress) * lift_target + progress * low_clearance_target
            mode = support_mode
        else:
            target = touchdown_target
            mode = "double"
        segments.append(_kf(f"{prefix}_soft_land_{index + 1}_{interpolation}", mode, target, durations[index]))
    return segments


def build_seed(candidate: SoftLandingCandidate) -> dict[str, Any]:
    """Build a new seed by replacing only the landing portions of a source lift seed."""
    source_seed = _load_json(Path(candidate.source_seed_path))
    neutral = _keyframe(source_seed, "neutral")
    right_unload = _keyframe(source_seed, "right_unload_pre")
    right_lift = _keyframe(source_seed, "right_micro_lift")
    right_soft_land = _keyframe(source_seed, "right_soft_land")
    neutral_after_right = _keyframe(source_seed, "neutral_after_right")
    left_unload = _keyframe(source_seed, "left_unload_pre")
    left_lift = _keyframe(source_seed, "left_micro_lift")
    left_soft_land = _keyframe(source_seed, "left_soft_land")

    keyframes: list[dict[str, Any]] = [
        neutral,
        right_unload,
        right_lift,
    ]
    if candidate.lift_hold_duration > 0:
        keyframes.append(_kf("right_lift_hold", "left", _target(right_lift), candidate.lift_hold_duration))
    keyframes.extend(
        _landing_segments(
            prefix="right",
            support_mode="left",
            lift_target=_target(right_lift),
            unload_target=_target(right_unload),
            soft_land_target=_target(right_soft_land),
            landing_duration=candidate.landing_duration,
            interpolation=candidate.interpolation,
        )
    )
    keyframes.extend([neutral_after_right, left_unload, left_lift])
    if candidate.lift_hold_duration > 0:
        keyframes.append(_kf("left_lift_hold", "right", _target(left_lift), candidate.lift_hold_duration))
    keyframes.extend(
        _landing_segments(
            prefix="left",
            support_mode="right",
            lift_target=_target(left_lift),
            unload_target=_target(left_unload),
            soft_land_target=_target(left_soft_land),
            landing_duration=candidate.landing_duration,
            interpolation=candidate.interpolation,
        )
    )
    return {
        "schema": "sedon_gait_seed.v1",
        "target_type": "absolute",
        "description": "Generated soft landing refine v1 reference.",
        "joint_names": JOINT_NAMES,
        "keyframes": keyframes,
    }


def write_candidate_files(candidate: SoftLandingCandidate, *, base_config: Path, out_dir: Path) -> tuple[Path, Path]:
    """Write seed/config files."""
    seed_path = out_dir / "seeds" / f"{candidate.candidate_id}.json"
    config_path = out_dir / "configs" / f"{candidate.candidate_id}.json"
    config = _load_json(base_config)
    config["reference_gait_seed_path"] = str(seed_path.relative_to(REPO_ROOT)).replace("\\", "/")
    config["reference_gait_seed_scale"] = 1.0
    _write_json(seed_path, build_seed(candidate))
    _write_json(config_path, config)
    return config_path, seed_path


def build_candidates(args: argparse.Namespace) -> list[SoftLandingCandidate]:
    """Build landing refinement candidates from source lift rows."""
    rows = _read_source_rows(args.source_top, args.source_top_k)
    candidates: list[SoftLandingCandidate] = []
    for row in rows:
        for landing_duration in args.landing_durations:
            for interpolation in args.interpolations:
                for hold_duration in args.lift_hold_durations:
                    candidate_id = (
                        f"{row['candidate_id']}_land{landing_duration}"
                        f"_{interpolation}_hold{hold_duration}"
                    )
                    candidates.append(
                        SoftLandingCandidate(
                            candidate_id=candidate_id,
                            source_candidate_id=row["candidate_id"],
                            source_seed_path=row["seed_path"],
                            landing_duration=landing_duration,
                            interpolation=interpolation,
                            lift_hold_duration=hold_duration,
                        )
                    )
    return candidates


def _timeline_row(step: int, info: dict[str, Any], signed_clearance: float) -> dict[str, Any]:
    return {
        "step": step,
        "phase_name": info["phase_name"],
        "contact_state": _contact_state(info),
        "support_side": info["support_side"],
        "swing_side": info["swing_side"],
        "force_ratio": float(info["force_ratio"]),
        "swing_force_ratio": float(info["swing_force_ratio"]),
        "signed_clearance": signed_clearance,
        "clearance": max(0.0, signed_clearance),
        "base_height": float(info["base_height"]),
        "upright": float(info["upright"]),
        "left_force": float(info["left_normal_force"]),
        "right_force": float(info["right_normal_force"]),
    }


def _foot_velocity_near_contact(timeline: list[dict[str, Any]]) -> float:
    values: list[float] = []
    previous: dict[str, Any] | None = None
    for row in timeline:
        if previous is None:
            previous = row
            continue
        phase = str(row["phase_name"])
        if "soft_land" in phase or "lift_hold" in phase:
            force_crossing = float(row["swing_force_ratio"]) >= 0.05 or row["contact_state"] == "both"
            if force_crossing:
                values.append(max(0.0, float(previous["signed_clearance"]) - float(row["signed_clearance"])))
        previous = row
    return max(values, default=0.0)


def audit_candidate(
    candidate: SoftLandingCandidate,
    *,
    config_path: Path,
    seed_path: Path,
    baseline_impact: float,
    out_dir: Path,
    steps: int,
    seed: int,
    warmup_steps: int,
) -> SoftLandingAudit:
    """Run dynamic PD audit for one soft landing candidate."""
    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=_load_config(config_path))
    total_weight = float(np.sum(env.model.body_mass) * 9.81)
    infos: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    signed_clearances: list[float] = []
    try:
        env.reset(seed=seed)
        for step in range(1, steps + 1):
            _, _, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float64))
            info = dict(info)
            phase = env._task_phase_metadata()
            support_z, swing_z = env._task_foot_bottom_heights(phase)
            signed_clearance = float(swing_z - support_z)
            infos.append(info)
            signed_clearances.append(signed_clearance)
            timeline.append(_timeline_row(step, info, signed_clearance))
            if terminated or truncated:
                break
    finally:
        env.close()

    timeline_path = out_dir / "timelines" / f"{candidate.candidate_id}.csv"
    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    if timeline:
        with timeline_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(timeline[0].keys()))
            writer.writeheader()
            writer.writerows(timeline)

    post_infos = infos[warmup_steps:] or infos
    total = max(1, len(infos))
    none = sum(1 for info in infos if _contact_state(info) == "none")
    single = sum(1 for info in infos if _contact_state(info) in {"left", "right"})
    jump_count = _count_contact_none_bursts(infos)
    max_contact_force = max(
        (float(info["left_normal_force"]) + float(info["right_normal_force"]) for info in post_infos),
        default=0.0,
    )
    impact = max_contact_force / max(total_weight, 1e-6)
    post_initial_base = float(post_infos[0]["base_height"]) if post_infos else 0.0
    base_drop = max(0.0, post_initial_base - min(float(info["base_height"]) for info in post_infos)) if post_infos else 0.0
    clearance = max((max(0.0, value) for value in signed_clearances), default=0.0)
    min_upright = min((float(info["upright"]) for info in infos), default=0.0)
    a_passed = none == 0 and jump_count == 0 and min_upright >= 0.99 and impact <= baseline_impact * 1.2
    b_passed = a_passed and clearance >= 0.0015
    c_passed = b_passed and single > 0
    score = (
        (100.0 if c_passed else 0.0)
        + (40.0 if b_passed else 0.0)
        + (10.0 if a_passed else 0.0)
        + clearance * 1000.0
        + single / total * 5.0
        - max(0.0, impact - baseline_impact) * 4.0
        - max(0.0, base_drop - 0.005) * 40.0
        - max(0.0, 0.99 - min_upright) * 100.0
        - none * 10.0
    )
    lift_rows = [row for row in timeline if "micro_lift" in str(row["phase_name"]) or "lift_hold" in str(row["phase_name"])]
    min_swing = min((float(row["swing_force_ratio"]) for row in lift_rows), default=float("inf"))
    return SoftLandingAudit(
        candidate_id=candidate.candidate_id,
        source_candidate_id=candidate.source_candidate_id,
        landing_duration=candidate.landing_duration,
        interpolation=candidate.interpolation,
        lift_hold_duration=candidate.lift_hold_duration,
        clearance=clearance,
        min_swing_force_ratio=min_swing,
        single_contact_ratio=single / total,
        impact_post=impact,
        max_contact_force_post=max_contact_force,
        base_drop_post=base_drop,
        upright=min_upright,
        foot_velocity_near_contact=_foot_velocity_near_contact(timeline),
        contact_none_ratio=none / total,
        jump_count=jump_count,
        a_passed=a_passed,
        b_passed=b_passed,
        c_passed=c_passed,
        score=score,
        config_path=str(config_path),
        seed_path=str(seed_path),
        timeline_path=str(timeline_path),
    )


def write_results(path: Path, rows: list[SoftLandingAudit]) -> None:
    """Write aggregate CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def render_candidate(
    row: SoftLandingAudit,
    *,
    steps: int,
    seed: int,
    fps: int,
    width: int,
    height: int,
    out_dir: Path,
) -> Path:
    """Render one soft landing candidate to MP4."""
    import mujoco

    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=_load_config(Path(row.config_path)))
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    camera = _make_side_camera()
    frames: list[np.ndarray] = []
    try:
        env.reset(seed=seed)
        for _ in range(steps):
            env.step(np.zeros(env.action_space.shape, dtype=np.float64))
            renderer.update_scene(env.data, camera=camera)
            frames.append(np.asarray(renderer.render(), dtype=np.uint8))
    finally:
        renderer.close()
        env.close()
    path = out_dir / "render" / f"{row.candidate_id}.mp4"
    _save_mp4(path, frames, fps)
    return path


def write_summary(path: Path, rows: list[SoftLandingAudit]) -> None:
    """Write compact summary markdown."""
    ranked = sorted(rows, key=lambda item: (item.c_passed, item.b_passed, item.a_passed, item.score), reverse=True)
    lines = [
        "# Soft Landing Refine V1 Summary",
        "",
        f"Candidates: {len(rows)}",
        f"A pass: {sum(1 for row in rows if row.a_passed)}",
        f"B pass: {sum(1 for row in rows if row.b_passed)}",
        f"C pass: {sum(1 for row in rows if row.c_passed)}",
        "",
        "## Top Candidates",
        "",
        "| candidate | A | B | C | clearance | single | impact | max_force | drop | upright | foot_vel |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked[:10]:
        lines.append(
            "| "
            f"{row.candidate_id} | {row.a_passed} | {row.b_passed} | {row.c_passed} | "
            f"{row.clearance:.4f} | {row.single_contact_ratio:.3f} | "
            f"{row.impact_post:.3f} | {row.max_contact_force_post:.1f} | "
            f"{row.base_drop_post:.4f} | {row.upright:.3f} | "
            f"{row.foot_velocity_near_contact:.5f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-top", type=Path, default=DEFAULT_SOURCE_TOP)
    parser.add_argument("--source-top-k", type=int, default=10)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-warmup-steps", type=int, default=20)
    parser.add_argument("--landing-durations", type=_parse_int_list, default="60,90,120,160")
    parser.add_argument("--interpolations", nargs="+", default=["smoothstep", "cubic_ease_out", "minimum_jerk"])
    parser.add_argument("--lift-hold-durations", type=_parse_int_list, default="0,10,20")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--render-top-k", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=368)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run soft landing refinement."""
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline = audit_shuffle(
        args.base_config,
        None,
        None,
        args.steps,
        args.seed,
        audit_warmup_steps=args.audit_warmup_steps,
    )
    candidates = build_candidates(args)
    if not candidates:
        raise ValueError("No source candidates with clearance/single-contact evidence were found.")
    rows: list[SoftLandingAudit] = []
    print(f"candidates={len(candidates)}")
    for index, candidate in enumerate(candidates, start=1):
        config_path, seed_path = write_candidate_files(candidate, base_config=args.base_config, out_dir=args.out_dir)
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
        if index == 1 or index == len(candidates) or index % 25 == 0:
            print(
                f"[{index}/{len(candidates)}] {row.candidate_id}: "
                f"A={row.a_passed} B={row.b_passed} C={row.c_passed} "
                f"clear={row.clearance:.4f} single={row.single_contact_ratio:.3f} "
                f"impact={row.impact_post:.3f} drop={row.base_drop_post:.4f} upright={row.upright:.3f}"
            )
    ranked = sorted(rows, key=lambda item: (item.c_passed, item.b_passed, item.a_passed, item.score), reverse=True)
    write_results(args.out_dir / "soft_landing_refine_v1.csv", rows)
    write_results(args.out_dir / "soft_landing_refine_v1_top10.csv", ranked[: args.top_k])
    write_summary(args.out_dir / "summary.md", rows)
    for row in ranked[: args.top_k]:
        print(
            "TOP "
            f"{row.candidate_id}: A={row.a_passed} B={row.b_passed} C={row.c_passed} "
            f"score={row.score:.3f} clear={row.clearance:.4f} single={row.single_contact_ratio:.3f} "
            f"impact={row.impact_post:.3f} max_force={row.max_contact_force_post:.1f} "
            f"drop={row.base_drop_post:.4f} upright={row.upright:.3f} foot_vel={row.foot_velocity_near_contact:.5f}"
        )
    for row in ranked[: args.render_top_k]:
        print(
            "rendered: "
            f"{render_candidate(row, steps=args.steps, seed=args.seed, fps=args.fps, width=args.width, height=args.height, out_dir=args.out_dir)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
