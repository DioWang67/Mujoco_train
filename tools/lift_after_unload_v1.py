"""Insert small lift phases into the best unload windows from refine v2."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sedon_baseline.env import SedonStandingEnv
from tools.audit_sedon_shuffle_v0 import _count_contact_none_bursts, _load_config, audit_shuffle
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
from tools.render_sedon_policy_comparison import _make_side_camera, _save_mp4


DEFAULT_SOURCE_TOP = (
    REPO_ROOT
    / "artifacts"
    / "sedon_debug"
    / "blue_unload_refine_v2"
    / "blue_unload_refine_v2_top20.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "sedon_debug" / "lift_after_unload_v1"
LIFT_VECTOR = np.array([0.18, -0.36, 0.18], dtype=np.float64)


@dataclass(frozen=True)
class UnloadWindow:
    """Continuous unload window for one side."""

    threshold: float
    start_local_step: int
    length: int


@dataclass(frozen=True)
class LiftCandidate:
    """One lift-after-unload candidate."""

    candidate_id: str
    source_candidate_id: str
    source_timeline_path: str
    unload: UnloadCandidate
    right_window: UnloadWindow
    left_window: UnloadWindow
    lift_height_target: float
    lift_duration: int
    landing_duration: int


@dataclass(frozen=True)
class LiftAudit:
    """Dynamic audit row for lift-after-unload."""

    candidate_id: str
    source_candidate_id: str
    lift_height_target: float
    lift_duration: int
    landing_duration: int
    window_threshold: float
    right_window_start: int
    left_window_start: int
    right_window_length: int
    left_window_length: int
    clearance: float
    min_swing_force_ratio: float
    single_contact_ratio: float
    near_zero_force_ratio: float
    duration_below_0_45: int
    duration_below_0_40: int
    contact_none_ratio: float
    jump_count: int
    impact_post: float
    base_drop_post: float
    upright: float
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


def _source_to_unload(row: dict[str, str]) -> UnloadCandidate:
    return UnloadCandidate(
        candidate_id=row["candidate_id"],
        preload_duration=int(row["preload_duration"]),
        support_hip_roll=float(row["support_hip_roll"]),
        swing_hip_roll=float(row["swing_hip_roll"]),
        stance_knee=float(row["stance_knee"]),
        stance_ankle=float(row["stance_ankle"]),
        pelvis_lean_proxy=float(row["pelvis_lean_proxy"]),
        lateral_offset_proxy=float(row["lateral_offset_proxy"]),
        tiny_lift_amplitude=float(row.get("tiny_lift_amplitude", 0.0)),
    )


def _read_source_rows(path: Path, limit: int) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source top candidates CSV: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows in source top candidates CSV: {path}")
    return rows[:limit]


def _load_timeline(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing unload timeline: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"No rows in unload timeline: {path}")
    return rows


def _best_window(rows: list[dict[str, str]], phase_name: str, threshold: float) -> UnloadWindow | None:
    best_start = 0
    best_length = 0
    current_start = 0
    current_length = 0
    local_step = 0
    for row in rows:
        if row["phase_name"] != phase_name:
            if current_length > best_length:
                best_start = current_start
                best_length = current_length
            current_length = 0
            continue
        local_step += 1
        if float(row["swing_force_ratio"]) <= threshold:
            if current_length == 0:
                current_start = local_step
            current_length += 1
        else:
            if current_length > best_length:
                best_start = current_start
                best_length = current_length
            current_length = 0
    if current_length > best_length:
        best_start = current_start
        best_length = current_length
    if best_length <= 0:
        return None
    return UnloadWindow(threshold=threshold, start_local_step=best_start, length=best_length)


def _select_window(rows: list[dict[str, str]], phase_name: str, lift_duration: int) -> UnloadWindow | None:
    below040 = _best_window(rows, phase_name, 0.40)
    if below040 and below040.length >= lift_duration:
        return below040
    below045 = _best_window(rows, phase_name, 0.45)
    if below045 and below045.length >= lift_duration:
        return below045
    return None


def _right_lift_target(unload: UnloadCandidate, lift_height_target: float) -> np.ndarray:
    target = _right_unload_target(unload)
    target[[R_HIP_PITCH, R_KNEE, R_ANKLE]] += LIFT_VECTOR * (lift_height_target / 0.001)
    return target


def _left_lift_target(unload: UnloadCandidate, lift_height_target: float) -> np.ndarray:
    target = _left_unload_target(unload)
    target[[L_HIP_PITCH, L_KNEE, L_ANKLE]] += LIFT_VECTOR * (lift_height_target / 0.001)
    return target


def build_seed(c: LiftCandidate, neutral_duration: int) -> dict[str, Any]:
    """Build a final seed with lift inserted at the selected unload window."""
    unload = c.unload
    right_unload = _right_unload_target(unload)
    left_unload = _left_unload_target(unload)
    right_pre = max(1, c.right_window.start_local_step)
    left_pre = max(1, c.left_window.start_local_step)
    return {
        "schema": "sedon_gait_seed.v1",
        "target_type": "absolute",
        "description": "Generated lift-after-unload v1 reference.",
        "joint_names": JOINT_NAMES,
        "keyframes": [
            _kf("neutral", "double", _zero(), neutral_duration),
            _kf("right_unload_pre", "left", right_unload, right_pre),
            _kf("right_micro_lift", "left", _right_lift_target(unload, c.lift_height_target), c.lift_duration),
            _kf("right_soft_land", "double", right_unload * 0.35, c.landing_duration),
            _kf("neutral_after_right", "double", _zero(), neutral_duration),
            _kf("left_unload_pre", "right", left_unload, left_pre),
            _kf("left_micro_lift", "right", _left_lift_target(unload, c.lift_height_target), c.lift_duration),
            _kf("left_soft_land", "double", left_unload * 0.35, c.landing_duration),
        ],
    }


def write_candidate_files(c: LiftCandidate, *, base_config: Path, out_dir: Path, neutral_duration: int) -> tuple[Path, Path]:
    """Write seed/config files for a lift candidate."""
    seed_path = out_dir / "seeds" / f"{c.candidate_id}.json"
    config_path = out_dir / "configs" / f"{c.candidate_id}.json"
    config = _load_json(base_config)
    config["reference_gait_seed_path"] = str(seed_path.relative_to(REPO_ROOT)).replace("\\", "/")
    config["reference_gait_seed_scale"] = 1.0
    _write_json(seed_path, build_seed(c, neutral_duration))
    _write_json(config_path, config)
    return config_path, seed_path


def build_candidates(args: argparse.Namespace) -> list[LiftCandidate]:
    """Build candidates from source top rows and their best unload windows."""
    rows = _read_source_rows(args.source_top, args.source_top_k)
    candidates: list[LiftCandidate] = []
    for row in rows:
        unload = _source_to_unload(row)
        timeline = _load_timeline(Path(row["timeline_path"]))
        for lift_duration in args.lift_durations:
            right_window = _select_window(timeline, "right_unload", lift_duration)
            left_window = _select_window(timeline, "left_unload", lift_duration)
            if right_window is None or left_window is None:
                continue
            threshold = max(right_window.threshold, left_window.threshold)
            for lift_height in args.lift_heights:
                for landing_duration in args.landing_durations:
                    candidate_id = (
                        f"{unload.candidate_id}_wh{_fmt(lift_height)}_ld{lift_duration}"
                        f"_land{landing_duration}_thr{_fmt(threshold)}"
                    )
                    candidates.append(
                        LiftCandidate(
                            candidate_id=candidate_id,
                            source_candidate_id=unload.candidate_id,
                            source_timeline_path=row["timeline_path"],
                            unload=unload,
                            right_window=right_window,
                            left_window=left_window,
                            lift_height_target=lift_height,
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
        "left_force": float(info["left_normal_force"]),
        "right_force": float(info["right_normal_force"]),
    }


def audit_candidate(
    c: LiftCandidate,
    *,
    config_path: Path,
    seed_path: Path,
    baseline_impact: float,
    out_dir: Path,
    steps: int,
    seed: int,
    warmup_steps: int,
) -> LiftAudit:
    """Run dynamic PD audit for one lift-after-unload candidate."""
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
    duration_below_045 = sum(1 for row in lift_rows if float(row["swing_force_ratio"]) <= 0.45)
    duration_below_040 = sum(1 for row in lift_rows if float(row["swing_force_ratio"]) <= 0.40)
    clearance = max((max(0.0, value) for value in signed_clearances), default=0.0)
    min_upright = min((float(info["upright"]) for info in infos), default=0.0)
    near_zero = sum(1 for row in lift_rows if float(row["swing_force_ratio"]) <= 0.05) / max(1, len(lift_rows))
    a_passed = none == 0 and jump_count == 0 and min_upright >= 0.99 and impact <= baseline_impact * 1.2
    b_passed = a_passed and clearance >= 0.0015
    c_passed = b_passed and (single > 0 or near_zero > 0)
    score = (
        (100.0 if c_passed else 0.0)
        + (40.0 if b_passed else 0.0)
        + (10.0 if a_passed else 0.0)
        + clearance * 1000.0
        + max(0.0, 0.45 - min_swing) * 2.0
        + single / total * 2.0
        + near_zero * 4.0
        - max(0.0, impact - baseline_impact) * 2.0
        - max(0.0, base_drop - 0.010) * 30.0
        - none * 10.0
    )
    return LiftAudit(
        candidate_id=c.candidate_id,
        source_candidate_id=c.source_candidate_id,
        lift_height_target=c.lift_height_target,
        lift_duration=c.lift_duration,
        landing_duration=c.landing_duration,
        window_threshold=max(c.right_window.threshold, c.left_window.threshold),
        right_window_start=c.right_window.start_local_step,
        left_window_start=c.left_window.start_local_step,
        right_window_length=c.right_window.length,
        left_window_length=c.left_window.length,
        clearance=clearance,
        min_swing_force_ratio=min_swing,
        single_contact_ratio=single / total,
        near_zero_force_ratio=near_zero,
        duration_below_0_45=duration_below_045,
        duration_below_0_40=duration_below_040,
        contact_none_ratio=none / total,
        jump_count=jump_count,
        impact_post=impact,
        base_drop_post=base_drop,
        upright=min_upright,
        a_passed=a_passed,
        b_passed=b_passed,
        c_passed=c_passed,
        score=score,
        config_path=str(config_path),
        seed_path=str(seed_path),
        timeline_path=str(timeline_path),
    )


def write_results(path: Path, rows: list[LiftAudit]) -> None:
    """Write aggregate CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def render_candidate(
    row: LiftAudit,
    *,
    steps: int,
    seed: int,
    fps: int,
    width: int,
    height: int,
    out_dir: Path,
) -> Path:
    """Render one lift candidate to MP4."""
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


def write_summary(path: Path, rows: list[LiftAudit]) -> None:
    """Write compact markdown summary."""
    ranked = sorted(rows, key=lambda item: (item.c_passed, item.b_passed, item.a_passed, item.score), reverse=True)
    lines = [
        "# Lift After Unload V1 Summary",
        "",
        f"Candidates: {len(rows)}",
        f"A pass: {sum(1 for row in rows if row.a_passed)}",
        f"B pass: {sum(1 for row in rows if row.b_passed)}",
        f"C pass: {sum(1 for row in rows if row.c_passed)}",
        "",
        "## Top Candidates",
        "",
        "| candidate | A | B | C | clearance | min_swing | single | near_zero | below045 | below040 | impact | drop | upright |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked[:10]:
        lines.append(
            "| "
            f"{row.candidate_id} | {row.a_passed} | {row.b_passed} | {row.c_passed} | "
            f"{row.clearance:.4f} | {row.min_swing_force_ratio:.3f} | "
            f"{row.single_contact_ratio:.3f} | {row.near_zero_force_ratio:.3f} | "
            f"{row.duration_below_0_45} | {row.duration_below_0_40} | "
            f"{row.impact_post:.3f} | {row.base_drop_post:.4f} | {row.upright:.3f} |"
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
    parser.add_argument("--neutral-duration", type=int, default=30)
    parser.add_argument("--lift-heights", type=_parse_float_list, default="0.001,0.0015,0.002,0.003")
    parser.add_argument("--lift-durations", type=_parse_int_list, default="20,30,45")
    parser.add_argument("--landing-durations", type=_parse_int_list, default="45,60,90")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--render-top-k", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=368)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build, audit, and render lift-after-unload candidates."""
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
        raise ValueError("No lift candidates were built; unload windows may be shorter than lift durations.")
    rows: list[LiftAudit] = []
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
        if index == 1 or index == len(candidates) or index % 25 == 0:
            print(
                f"[{index}/{len(candidates)}] {row.candidate_id}: "
                f"A={row.a_passed} B={row.b_passed} C={row.c_passed} "
                f"clear={row.clearance:.4f} min_swing={row.min_swing_force_ratio:.3f} "
                f"single={row.single_contact_ratio:.3f} impact={row.impact_post:.3f}"
            )
    ranked = sorted(rows, key=lambda item: (item.c_passed, item.b_passed, item.a_passed, item.score), reverse=True)
    write_results(args.out_dir / "lift_after_unload_v1.csv", rows)
    write_results(args.out_dir / "lift_after_unload_v1_top10.csv", ranked[: args.top_k])
    write_summary(args.out_dir / "summary.md", rows)
    for row in ranked[: args.top_k]:
        print(
            "TOP "
            f"{row.candidate_id}: A={row.a_passed} B={row.b_passed} C={row.c_passed} "
            f"score={row.score:.3f} clear={row.clearance:.4f} min_swing={row.min_swing_force_ratio:.3f} "
            f"single={row.single_contact_ratio:.3f} near_zero={row.near_zero_force_ratio:.3f} "
            f"impact={row.impact_post:.3f} drop={row.base_drop_post:.4f}"
        )
    for row in ranked[: args.render_top_k]:
        print(
            "rendered: "
            f"{render_candidate(row, steps=args.steps, seed=args.seed, fps=args.fps, width=args.width, height=args.height, out_dir=args.out_dir)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
