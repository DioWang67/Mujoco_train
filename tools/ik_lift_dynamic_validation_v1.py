"""Validate kinematic IK lift mapping under deterministic dynamic PD."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
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
    _left_unload_target,
    _right_unload_target,
    _zero,
)
from tools.kinematic_foot_jacobian_diagnostic_v1 import _set_joint_positions, _state
from tools.lift_after_unload_v1 import UnloadWindow, _best_window, _load_timeline
from tools.render_seedon_policy_comparison import _make_side_camera, _save_mp4


SOURCE_CANDIDATE_ID = "dur180_sr0p08_wrm0p02_skm0p03_sa0p04_lean0p035_latm0p01_tl0"
DEFAULT_SOURCE_TOP = (
    REPO_ROOT
    / "artifacts"
    / "seedon_debug"
    / "blue_unload_refine_v2"
    / "blue_unload_refine_v2_top20.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "seedon_debug" / "ik_lift_dynamic_validation_v1"


@dataclass(frozen=True)
class IkLiftCandidate:
    """One IK mapping dynamic validation candidate."""

    candidate_id: str
    unload: UnloadCandidate
    right_window: UnloadWindow
    left_window: UnloadWindow
    gain: int
    lift_duration: int
    landing_duration: int
    expected_kinematic_clearance: float


@dataclass(frozen=True)
class IkLiftAudit:
    """Audit row for one IK lift validation candidate."""

    candidate_id: str
    gain: int
    lift_duration: int
    landing_duration: int
    dynamic_clearance: float
    expected_kinematic_clearance: float
    clearance_efficiency: float
    min_upright: float
    max_base_roll: float
    max_base_pitch: float
    impact_post: float
    base_drop_post: float
    single_contact_ratio: float
    min_swing_force_ratio: float
    contact_none_ratio: float
    jump_count: int
    classification: str
    score: float
    config_path: str
    seed_path: str
    timeline_path: str


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive integer values")
    return values


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _source_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        if row["candidate_id"] == SOURCE_CANDIDATE_ID:
            return row
    raise ValueError(f"Missing source candidate {SOURCE_CANDIDATE_ID} in {path}")


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
        tiny_lift_amplitude=0.0,
    )


def _minimum_jerk(t: float) -> float:
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


def _kf(name: str, support_mode: str, target: np.ndarray, duration: int) -> dict[str, Any]:
    return {
        "name": name,
        "support_mode": support_mode,
        "joint_targets": target.tolist(),
        "duration_steps": max(1, int(duration)),
    }


def _ik_delta(gain: int, *, right: bool) -> np.ndarray:
    eps = 0.005 * gain
    delta = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    if right:
        delta[R_HIP_PITCH] = 0.5 * eps
        delta[R_KNEE] = -eps
        delta[R_ANKLE] = -0.25 * eps
    else:
        delta[L_HIP_PITCH] = 0.5 * eps
        delta[L_KNEE] = -eps
        delta[L_ANKLE] = -0.25 * eps
    return delta


def _segments(prefix: str, support: str, start: np.ndarray, target: np.ndarray, duration: int) -> list[dict[str, Any]]:
    count = 5
    base = max(1, duration // count)
    durations = [base] * count
    durations[-1] += max(0, duration - sum(durations))
    return [
        _kf(f"{prefix}_{index + 1}", support, start + (target - start) * _minimum_jerk((index + 1) / count), durations[index])
        for index in range(count)
    ]


def _expected_clearance(unload: UnloadCandidate, gain: int, *, side: str, base_config: Path, seed: int) -> float:
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=_load_config(base_config))
    try:
        env.reset(seed=seed)
        base = _right_unload_target(unload) if side == "right" else _left_unload_target(unload)
        _set_joint_positions(env, base)
        before = _state(env, side)
        _set_joint_positions(env, base + _ik_delta(gain, right=(side == "right")))
        after = _state(env, side)
        return max(0.0, float(np.asarray(after["foot_pos"])[2] - np.asarray(before["foot_pos"])[2]))
    finally:
        env.close()


def build_seed(c: IkLiftCandidate, neutral_duration: int) -> dict[str, Any]:
    """Build minimum-jerk lift and landing seed."""
    right_unload = _right_unload_target(c.unload)
    left_unload = _left_unload_target(c.unload)
    right_lift = right_unload + _ik_delta(c.gain, right=True)
    left_lift = left_unload + _ik_delta(c.gain, right=False)
    keyframes: list[dict[str, Any]] = [
        _kf("neutral", "double", _zero(), neutral_duration),
        _kf("right_unload_pre", "left", right_unload, max(1, c.right_window.start_local_step)),
    ]
    keyframes.extend(_segments("right_lift_mj", "left", right_unload, right_lift, c.lift_duration))
    keyframes.extend(_segments("right_land_mj", "double", right_lift, right_unload * 0.45, c.landing_duration))
    keyframes.extend([
        _kf("neutral_after_right", "double", _zero(), neutral_duration),
        _kf("left_unload_pre", "right", left_unload, max(1, c.left_window.start_local_step)),
    ])
    keyframes.extend(_segments("left_lift_mj", "right", left_unload, left_lift, c.lift_duration))
    keyframes.extend(_segments("left_land_mj", "double", left_lift, left_unload * 0.45, c.landing_duration))
    return {
        "schema": "seedon_gait_seed.v1",
        "target_type": "absolute",
        "description": "Generated IK lift dynamic validation v1 reference.",
        "joint_names": JOINT_NAMES,
        "keyframes": keyframes,
    }


def write_candidate_files(c: IkLiftCandidate, *, base_config: Path, out_dir: Path, neutral_duration: int) -> tuple[Path, Path]:
    seed_path = out_dir / "seeds" / f"{c.candidate_id}.json"
    config_path = out_dir / "configs" / f"{c.candidate_id}.json"
    config = _load_json(base_config)
    config["reference_gait_seed_path"] = str(seed_path.relative_to(REPO_ROOT)).replace("\\", "/")
    config["reference_gait_seed_scale"] = 1.0
    _write_json(seed_path, build_seed(c, neutral_duration))
    _write_json(config_path, config)
    return config_path, seed_path


def build_candidates(args: argparse.Namespace) -> list[IkLiftCandidate]:
    row = _source_row(args.source_top)
    unload = _source_to_unload(row)
    timeline = _load_timeline(Path(row["timeline_path"]))
    right_window = _best_window(timeline, "right_unload", 0.40) or _best_window(timeline, "right_unload", 0.45)
    left_window = _best_window(timeline, "left_unload", 0.40) or _best_window(timeline, "left_unload", 0.45)
    if right_window is None or left_window is None:
        raise ValueError("Source candidate has no usable unload window.")
    expected = {
        gain: max(
            _expected_clearance(unload, gain, side="right", base_config=args.base_config, seed=args.seed),
            _expected_clearance(unload, gain, side="left", base_config=args.base_config, seed=args.seed),
        )
        for gain in args.gains
    }
    candidates: list[IkLiftCandidate] = []
    for gain in args.gains:
        for lift_duration in args.lift_durations:
            for landing_duration in args.landing_durations:
                candidates.append(
                    IkLiftCandidate(
                        candidate_id=f"{SOURCE_CANDIDATE_ID}_ikgain{gain}_ld{lift_duration}_land{landing_duration}",
                        unload=unload,
                        right_window=right_window,
                        left_window=left_window,
                        gain=gain,
                        lift_duration=lift_duration,
                        landing_duration=landing_duration,
                        expected_kinematic_clearance=expected[gain],
                    )
                )
    return candidates


def _timeline_row(step: int, info: dict[str, Any], signed_clearance: float) -> dict[str, Any]:
    return {
        "step": step,
        "phase_name": info["phase_name"],
        "contact_state": _contact_state(info),
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


def _classify(clearance: float, upright: float, impact: float, drop: float, none: int, jumps: int, baseline: float) -> str:
    stable = upright >= 0.99 and impact <= baseline * 1.2 and drop <= 0.015 and none == 0 and jumps == 0
    if stable and clearance < 0.001:
        return "stable_no_lift"
    if stable and 0.001 <= clearance <= 0.002:
        return "usable_micro_lift"
    if clearance >= 0.001 and not stable:
        return "cliff"
    return "other"


def audit_candidate(c: IkLiftCandidate, *, config_path: Path, seed_path: Path, baseline_impact: float, out_dir: Path, steps: int, seed: int, warmup_steps: int) -> IkLiftAudit:
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=_load_config(config_path))
    total_weight = float(np.sum(env.model.body_mass) * 9.81)
    infos: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    clearances: list[float] = []
    try:
        env.reset(seed=seed)
        for step in range(1, steps + 1):
            _, _, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float64))
            info = dict(info)
            phase = env._task_phase_metadata()
            support_z, swing_z = env._task_foot_bottom_heights(phase)
            signed = float(swing_z - support_z)
            infos.append(info)
            clearances.append(max(0.0, signed))
            timeline.append(_timeline_row(step, info, signed))
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
    post = infos[warmup_steps:] or infos
    none = sum(1 for info in infos if _contact_state(info) == "none")
    jumps = _count_contact_none_bursts(infos)
    force = max((float(info["left_normal_force"]) + float(info["right_normal_force"]) for info in post), default=0.0)
    impact = force / max(total_weight, 1e-6)
    base0 = float(post[0]["base_height"]) if post else 0.0
    drop = max(0.0, base0 - min(float(info["base_height"]) for info in post)) if post else 0.0
    clearance = max(clearances, default=0.0)
    upright = min((float(info["upright"]) for info in infos), default=0.0)
    roll = max((abs(float(info["base_roll"])) for info in infos), default=0.0)
    pitch = max((abs(float(info["base_pitch"])) for info in infos), default=0.0)
    single = sum(1 for info in infos if _contact_state(info) in {"left", "right"}) / max(1, len(infos))
    lift_rows = [row for row in timeline if "lift_mj" in str(row["phase_name"])]
    min_swing = min((float(row["swing_force_ratio"]) for row in lift_rows), default=float("inf"))
    classification = _classify(clearance, upright, impact, drop, none, jumps, baseline_impact)
    efficiency = clearance / max(c.expected_kinematic_clearance, 1e-9)
    score = (
        (100 if classification == "usable_micro_lift" else 0)
        + (10 if classification == "stable_no_lift" else 0)
        - abs(clearance - 0.0015) * 500
        - max(0.0, impact - baseline_impact) * 4
        - max(0.0, 0.99 - upright) * 100
        - max(0.0, drop - 0.005) * 40
    )
    return IkLiftAudit(
        candidate_id=c.candidate_id,
        gain=c.gain,
        lift_duration=c.lift_duration,
        landing_duration=c.landing_duration,
        dynamic_clearance=clearance,
        expected_kinematic_clearance=c.expected_kinematic_clearance,
        clearance_efficiency=efficiency,
        min_upright=upright,
        max_base_roll=roll,
        max_base_pitch=pitch,
        impact_post=impact,
        base_drop_post=drop,
        single_contact_ratio=single,
        min_swing_force_ratio=min_swing,
        contact_none_ratio=none / max(1, len(infos)),
        jump_count=jumps,
        classification=classification,
        score=score,
        config_path=str(config_path),
        seed_path=str(seed_path),
        timeline_path=str(timeline_path),
    )


def write_results(path: Path, rows: list[IkLiftAudit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def render_candidate(row: IkLiftAudit, *, steps: int, seed: int, fps: int, width: int, height: int, out_dir: Path) -> Path:
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


def write_classification(path: Path, rows: list[IkLiftAudit]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.classification] = counts.get(row.classification, 0) + 1
    if counts.get("usable_micro_lift", 0) > 0:
        verdict = "usable_micro_lift"
    elif counts.get("cliff", 0) > 0:
        verdict = "cliff"
    elif counts.get("stable_no_lift", 0) == len(rows):
        verdict = "stable_no_lift"
    else:
        verdict = "mixed_no_usable_band"
    lines = [verdict, *(f"{key}: {value}" for key, value in sorted(counts.items()))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-top", type=Path, default=DEFAULT_SOURCE_TOP)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-warmup-steps", type=int, default=20)
    parser.add_argument("--neutral-duration", type=int, default=30)
    parser.add_argument("--gains", type=_parse_int_list, default="1,2,3,4,5,6,7,8")
    parser.add_argument("--lift-durations", type=_parse_int_list, default="60,90,120")
    parser.add_argument("--landing-durations", type=_parse_int_list, default="90,120,160")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--render-top-k", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=368)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline = audit_shuffle(args.base_config, None, None, args.steps, args.seed, audit_warmup_steps=args.audit_warmup_steps)
    candidates = build_candidates(args)
    rows: list[IkLiftAudit] = []
    print(f"candidates={len(candidates)}")
    for index, candidate in enumerate(candidates, start=1):
        config_path, seed_path = write_candidate_files(candidate, base_config=args.base_config, out_dir=args.out_dir, neutral_duration=args.neutral_duration)
        row = audit_candidate(candidate, config_path=config_path, seed_path=seed_path, baseline_impact=baseline.landing_impact_post_warmup, out_dir=args.out_dir, steps=args.steps, seed=args.seed, warmup_steps=args.audit_warmup_steps)
        rows.append(row)
        if index == 1 or index == len(candidates) or index % 20 == 0:
            print(f"[{index}/{len(candidates)}] {row.candidate_id}: {row.classification} clear={row.dynamic_clearance:.4f} expected={row.expected_kinematic_clearance:.4f} eff={row.clearance_efficiency:.2f} upright={row.min_upright:.3f} impact={row.impact_post:.3f}")
    ranked = sorted(rows, key=lambda row: (row.classification == "usable_micro_lift", row.classification == "stable_no_lift", row.score), reverse=True)
    write_results(args.out_dir / "ik_lift_dynamic_validation_v1.csv", rows)
    write_results(args.out_dir / "ik_lift_dynamic_validation_v1_top10.csv", ranked[: args.top_k])
    write_classification(args.out_dir / "classification.txt", rows)
    for row in ranked[: args.top_k]:
        print(f"TOP {row.candidate_id}: {row.classification} clear={row.dynamic_clearance:.4f} expected={row.expected_kinematic_clearance:.4f} eff={row.clearance_efficiency:.2f} upright={row.min_upright:.3f} impact={row.impact_post:.3f} drop={row.base_drop_post:.4f}")
    for row in ranked[: args.render_top_k]:
        print("rendered: " f"{render_candidate(row, steps=args.steps, seed=args.seed, fps=args.fps, width=args.width, height=args.height, out_dir=args.out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
