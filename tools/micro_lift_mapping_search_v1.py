"""Search small stable micro-lift joint mappings after successful unload."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from seedon_baseline.env import SeedonStandingEnv
from tools.audit_seedon_shuffle_v0 import _count_contact_none_bursts, _load_config, audit_shuffle
from tools.blue_unload_mechanism_search import (
    DEFAULT_BASE_CONFIG,
    JOINT_NAMES,
    L_ANKLE,
    L_HIP_PITCH,
    L_KNEE,
    R_ANKLE,
    R_HIP_PITCH,
    R_KNEE,
    REPO_ROOT,
    UnloadCandidate,
    _contact_state,
    _fmt,
    _left_unload_target,
    _right_unload_target,
    _zero,
)
from tools.lift_after_unload_v1 import (
    UnloadWindow,
    _best_window,
    _load_timeline,
    _source_to_unload,
)
from tools.render_seedon_policy_comparison import _make_side_camera, _save_mp4


DEFAULT_SOURCE_TOP = (
    REPO_ROOT
    / "artifacts"
    / "seedon_debug"
    / "blue_unload_refine_v2"
    / "blue_unload_refine_v2_top20.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "seedon_debug" / "micro_lift_mapping_search_v1"


@dataclass(frozen=True)
class MicroLiftCandidate:
    """One small micro-lift mapping candidate."""

    candidate_id: str
    source_candidate_id: str
    source_timeline_path: str
    unload: UnloadCandidate
    right_window: UnloadWindow
    left_window: UnloadWindow
    target_clearance: float
    mapping_family: str
    lift_duration: int
    landing_duration: int


@dataclass(frozen=True)
class MicroLiftAudit:
    """Audit metrics for one micro-lift mapping candidate."""

    candidate_id: str
    source_candidate_id: str
    target_clearance: float
    mapping_family: str
    lift_duration: int
    landing_duration: int
    window_threshold: float
    clearance: float
    min_swing_force_ratio: float
    min_upright: float
    max_base_roll: float
    max_base_pitch: float
    impact_post: float
    base_drop_post: float
    single_contact_ratio: float
    near_zero_force_ratio: float
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


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive integer durations")
    return values


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _kf(name: str, support_mode: str, target: np.ndarray, duration: int) -> dict[str, Any]:
    return {
        "name": name,
        "support_mode": support_mode,
        "joint_targets": target.tolist(),
        "duration_steps": max(1, int(duration)),
    }


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
        if row.get("stable_gate", "True") == "True"
        and row.get("a_passed", "True") == "True"
    ]
    return filtered[:limit]


def _select_start_window(rows: list[dict[str, str]], phase_name: str) -> UnloadWindow | None:
    below040 = _best_window(rows, phase_name, 0.40)
    if below040 is not None:
        return below040
    return _best_window(rows, phase_name, 0.45)


def _mapping_delta(family: str, target_clearance: float, *, right: bool) -> np.ndarray:
    """Return tiny swing-leg joint deltas for a target clearance label."""
    scale = target_clearance / 0.0015
    target = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    if family == "knee_only":
        hip, knee, ankle = 0.0, -0.030, 0.0
    elif family == "hip_knee":
        hip, knee, ankle = 0.010, -0.032, 0.0
    elif family == "knee_ankle":
        hip, knee, ankle = 0.0, -0.032, 0.014
    elif family == "hip_knee_ankle_small_counter":
        hip, knee, ankle = 0.010, -0.036, 0.008
    else:
        raise ValueError(f"Unsupported mapping family: {family}")
    values = np.array([hip, knee, ankle], dtype=np.float64) * scale
    if right:
        target[[R_HIP_PITCH, R_KNEE, R_ANKLE]] = values
    else:
        target[[L_HIP_PITCH, L_KNEE, L_ANKLE]] = values
    return target


def _right_lift_target(c: MicroLiftCandidate) -> np.ndarray:
    return _right_unload_target(c.unload) + _mapping_delta(c.mapping_family, c.target_clearance, right=True)


def _left_lift_target(c: MicroLiftCandidate) -> np.ndarray:
    return _left_unload_target(c.unload) + _mapping_delta(c.mapping_family, c.target_clearance, right=False)


def build_seed(c: MicroLiftCandidate, neutral_duration: int) -> dict[str, Any]:
    """Build a reference seed with a very small mapping-specific micro-lift."""
    right_unload = _right_unload_target(c.unload)
    left_unload = _left_unload_target(c.unload)
    right_pre = max(1, c.right_window.start_local_step)
    left_pre = max(1, c.left_window.start_local_step)
    return {
        "schema": "seedon_gait_seed.v1",
        "target_type": "absolute",
        "description": "Generated micro-lift mapping search v1 reference.",
        "joint_names": JOINT_NAMES,
        "keyframes": [
            _kf("neutral", "double", _zero(), neutral_duration),
            _kf("right_unload_pre", "left", right_unload, right_pre),
            _kf("right_micro_lift", "left", _right_lift_target(c), c.lift_duration),
            _kf("right_soft_land", "double", right_unload * 0.45, c.landing_duration),
            _kf("neutral_after_right", "double", _zero(), neutral_duration),
            _kf("left_unload_pre", "right", left_unload, left_pre),
            _kf("left_micro_lift", "right", _left_lift_target(c), c.lift_duration),
            _kf("left_soft_land", "double", left_unload * 0.45, c.landing_duration),
        ],
    }


def write_candidate_files(c: MicroLiftCandidate, *, base_config: Path, out_dir: Path, neutral_duration: int) -> tuple[Path, Path]:
    """Write seed/config files."""
    seed_path = out_dir / "seeds" / f"{c.candidate_id}.json"
    config_path = out_dir / "configs" / f"{c.candidate_id}.json"
    config = _load_json(base_config)
    config["reference_gait_seed_path"] = str(seed_path.relative_to(REPO_ROOT)).replace("\\", "/")
    config["reference_gait_seed_scale"] = 1.0
    _write_json(seed_path, build_seed(c, neutral_duration))
    _write_json(config_path, config)
    return config_path, seed_path


def build_candidates(args: argparse.Namespace) -> list[MicroLiftCandidate]:
    """Build candidates from unload refine top rows."""
    rows = _read_source_rows(args.source_top, args.source_top_k)
    candidates: list[MicroLiftCandidate] = []
    for row in rows:
        unload = _source_to_unload(row)
        timeline = _load_timeline(Path(row["timeline_path"]))
        for lift_duration in args.lift_durations:
            right_window = _select_start_window(timeline, "right_unload")
            left_window = _select_start_window(timeline, "left_unload")
            if right_window is None or left_window is None:
                continue
            threshold = max(right_window.threshold, left_window.threshold)
            for target_clearance in args.target_clearances:
                for family in args.mapping_families:
                    for landing_duration in args.landing_durations:
                        candidate_id = (
                            f"{unload.candidate_id}_tc{_fmt(target_clearance)}"
                            f"_{family}_ld{lift_duration}_land{landing_duration}_thr{_fmt(threshold)}"
                        )
                        candidates.append(
                            MicroLiftCandidate(
                                candidate_id=candidate_id,
                                source_candidate_id=unload.candidate_id,
                                source_timeline_path=row["timeline_path"],
                                unload=unload,
                                right_window=right_window,
                                left_window=left_window,
                                target_clearance=target_clearance,
                                mapping_family=family,
                                lift_duration=lift_duration,
                                landing_duration=landing_duration,
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
        "base_roll": float(info["base_roll"]),
        "base_pitch": float(info["base_pitch"]),
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
        if "soft_land" in str(row["phase_name"]) and (
            float(row["swing_force_ratio"]) >= 0.05 or row["contact_state"] == "both"
        ):
            values.append(max(0.0, float(previous["signed_clearance"]) - float(row["signed_clearance"])))
        previous = row
    return max(values, default=0.0)


def audit_candidate(
    c: MicroLiftCandidate,
    *,
    config_path: Path,
    seed_path: Path,
    baseline_impact: float,
    out_dir: Path,
    steps: int,
    seed: int,
    warmup_steps: int,
) -> MicroLiftAudit:
    """Run dynamic PD audit for one micro-lift mapping candidate."""
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=_load_config(config_path))
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

    timeline_path = out_dir / "timelines" / f"{c.candidate_id}.csv"
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
    contact_forces_post = [
        float(info["left_normal_force"]) + float(info["right_normal_force"])
        for info in post_infos
    ]
    impact = max(contact_forces_post, default=0.0) / max(total_weight, 1e-6)
    post_initial_base = float(post_infos[0]["base_height"]) if post_infos else 0.0
    base_drop = max(0.0, post_initial_base - min(float(info["base_height"]) for info in post_infos)) if post_infos else 0.0
    lift_rows = [row for row in timeline if "micro_lift" in str(row["phase_name"])]
    min_swing = min((float(row["swing_force_ratio"]) for row in lift_rows), default=float("inf"))
    near_zero = sum(1 for row in lift_rows if float(row["swing_force_ratio"]) <= 0.05) / max(1, len(lift_rows))
    clearance = max((max(0.0, value) for value in signed_clearances), default=0.0)
    min_upright = min((float(info["upright"]) for info in infos), default=0.0)
    max_base_roll = max((abs(float(info["base_roll"])) for info in infos), default=0.0)
    max_base_pitch = max((abs(float(info["base_pitch"])) for info in infos), default=0.0)
    a_passed = (
        none == 0
        and jump_count == 0
        and min_upright >= 0.99
        and impact <= baseline_impact * 1.2
        and base_drop <= 0.015
    )
    b_passed = a_passed and clearance >= 0.0015
    c_passed = b_passed and (single > 0 or near_zero > 0.0)
    score = (
        (100.0 if c_passed else 0.0)
        + (40.0 if b_passed else 0.0)
        + (10.0 if a_passed else 0.0)
        - abs(clearance - 0.0017) * 500.0
        + max(0.0, 0.45 - min_swing) * 2.0
        + single / total * 2.0
        + near_zero * 3.0
        - max(0.0, impact - baseline_impact) * 4.0
        - max(0.0, base_drop - 0.005) * 40.0
        - max(0.0, 0.99 - min_upright) * 100.0
    )
    return MicroLiftAudit(
        candidate_id=c.candidate_id,
        source_candidate_id=c.source_candidate_id,
        target_clearance=c.target_clearance,
        mapping_family=c.mapping_family,
        lift_duration=c.lift_duration,
        landing_duration=c.landing_duration,
        window_threshold=max(c.right_window.threshold, c.left_window.threshold),
        clearance=clearance,
        min_swing_force_ratio=min_swing,
        min_upright=min_upright,
        max_base_roll=max_base_roll,
        max_base_pitch=max_base_pitch,
        impact_post=impact,
        base_drop_post=base_drop,
        single_contact_ratio=single / total,
        near_zero_force_ratio=near_zero,
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


def write_results(path: Path, rows: list[MicroLiftAudit]) -> None:
    """Write aggregate CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def render_candidate(
    row: MicroLiftAudit,
    *,
    steps: int,
    seed: int,
    fps: int,
    width: int,
    height: int,
    out_dir: Path,
) -> Path:
    """Render one micro-lift candidate to MP4."""
    import mujoco

    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=_load_config(Path(row.config_path)))
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


def write_summary(path: Path, rows: list[MicroLiftAudit]) -> None:
    """Write compact summary markdown."""
    ranked = sorted(rows, key=lambda item: (item.c_passed, item.b_passed, item.a_passed, item.score), reverse=True)
    lines = [
        "# Micro Lift Mapping Search V1 Summary",
        "",
        f"Candidates: {len(rows)}",
        f"A pass: {sum(1 for row in rows if row.a_passed)}",
        f"B pass: {sum(1 for row in rows if row.b_passed)}",
        f"C pass: {sum(1 for row in rows if row.c_passed)}",
        "",
        "## Top Candidates",
        "",
        "| candidate | A | B | C | clearance | upright | roll | pitch | impact | drop | single | foot_vel |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked[:10]:
        lines.append(
            "| "
            f"{row.candidate_id} | {row.a_passed} | {row.b_passed} | {row.c_passed} | "
            f"{row.clearance:.4f} | {row.min_upright:.3f} | {row.max_base_roll:.3f} | "
            f"{row.max_base_pitch:.3f} | {row.impact_post:.3f} | {row.base_drop_post:.4f} | "
            f"{row.single_contact_ratio:.3f} | {row.foot_velocity_near_contact:.5f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-top", type=Path, default=DEFAULT_SOURCE_TOP)
    parser.add_argument("--source-top-k", type=int, default=5)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-warmup-steps", type=int, default=20)
    parser.add_argument("--neutral-duration", type=int, default=30)
    parser.add_argument("--target-clearances", type=_parse_float_list, default="0.0012,0.0015,0.0020")
    parser.add_argument(
        "--mapping-families",
        nargs="+",
        default=["knee_only", "hip_knee", "knee_ankle", "hip_knee_ankle_small_counter"],
    )
    parser.add_argument("--lift-durations", type=_parse_int_list, default="30,45,60")
    parser.add_argument("--landing-durations", type=_parse_int_list, default="60,90,120")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--render-top-k", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=368)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run micro-lift mapping search."""
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
        raise ValueError("No micro-lift candidates were built from unload windows.")
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
                f"A={row.a_passed} B={row.b_passed} C={row.c_passed} "
                f"clear={row.clearance:.4f} upright={row.min_upright:.3f} "
                f"impact={row.impact_post:.3f} drop={row.base_drop_post:.4f}"
            )
    ranked = sorted(rows, key=lambda item: (item.c_passed, item.b_passed, item.a_passed, item.score), reverse=True)
    write_results(args.out_dir / "micro_lift_mapping_search_v1.csv", rows)
    write_results(args.out_dir / "micro_lift_mapping_search_v1_top10.csv", ranked[: args.top_k])
    write_summary(args.out_dir / "summary.md", rows)
    for row in ranked[: args.top_k]:
        print(
            "TOP "
            f"{row.candidate_id}: A={row.a_passed} B={row.b_passed} C={row.c_passed} "
            f"score={row.score:.3f} clear={row.clearance:.4f} upright={row.min_upright:.3f} "
            f"roll={row.max_base_roll:.3f} pitch={row.max_base_pitch:.3f} "
            f"impact={row.impact_post:.3f} drop={row.base_drop_post:.4f} "
            f"single={row.single_contact_ratio:.3f} foot_vel={row.foot_velocity_near_contact:.5f}"
        )
    for row in ranked[: args.render_top_k]:
        print(
            "rendered: "
            f"{render_candidate(row, steps=args.steps, seed=args.seed, fps=args.fps, width=args.width, height=args.height, out_dir=args.out_dir)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
