"""Search Blue-like unload mechanisms without requiring foot clearance."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from seedon_baseline.env import SeedonStandingEnv
from tools.audit_seedon_shuffle_v0 import _count_contact_none_bursts, _load_config
from tools.render_seedon_policy_comparison import _make_side_camera, _save_mp4


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "seedon" / "reference_teacher_pose_1_4_imitation.json"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "seedon_debug" / "blue_unload_mechanism_search"
JOINT_NAMES = [
    "R_joint_hip_yaw",
    "R_joint_hip_roll",
    "R_joint_hip_pitch",
    "R_joint_knee_pitch",
    "R_joint_ankle_pitch",
    "L_joint_hip_yaw",
    "L_joint_hip_roll",
    "L_joint_hip_pitch",
    "L_joint_knee_pitch",
    "L_joint_ankle_pitch",
]
R_HIP_ROLL, R_HIP_PITCH, R_KNEE, R_ANKLE = 1, 2, 3, 4
L_HIP_ROLL, L_HIP_PITCH, L_KNEE, L_ANKLE = 6, 7, 8, 9
TINY_LIFT_VECTOR = np.array([0.08, -0.16, 0.08], dtype=np.float64)


@dataclass(frozen=True)
class UnloadCandidate:
    """One unload-only reference candidate."""

    candidate_id: str
    preload_duration: int
    support_hip_roll: float
    swing_hip_roll: float
    stance_knee: float
    stance_ankle: float
    pelvis_lean_proxy: float
    lateral_offset_proxy: float
    tiny_lift_amplitude: float


@dataclass(frozen=True)
class UnloadAudit:
    """Dynamic audit metrics for one unload candidate."""

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
    a_passed: bool
    b_passed: bool
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


def _fmt(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.3g}".replace("-", "m").replace(".", "p")


def _zero() -> np.ndarray:
    return np.zeros(len(JOINT_NAMES), dtype=np.float64)


def _right_unload_target(c: UnloadCandidate) -> np.ndarray:
    """Return target for unloading the right swing foot onto left support."""
    target = _zero()
    target[L_HIP_ROLL] = c.support_hip_roll + c.pelvis_lean_proxy + c.lateral_offset_proxy
    target[R_HIP_ROLL] = c.swing_hip_roll + c.pelvis_lean_proxy - c.lateral_offset_proxy
    target[L_KNEE] = c.stance_knee
    target[L_ANKLE] = c.stance_ankle
    if c.tiny_lift_amplitude:
        target[[R_HIP_PITCH, R_KNEE, R_ANKLE]] += TINY_LIFT_VECTOR * c.tiny_lift_amplitude
    return target


def _left_unload_target(c: UnloadCandidate) -> np.ndarray:
    """Return target for unloading the left swing foot onto right support."""
    target = _zero()
    target[R_HIP_ROLL] = -c.support_hip_roll - c.pelvis_lean_proxy - c.lateral_offset_proxy
    target[L_HIP_ROLL] = -c.swing_hip_roll - c.pelvis_lean_proxy + c.lateral_offset_proxy
    target[R_KNEE] = c.stance_knee
    target[R_ANKLE] = c.stance_ankle
    if c.tiny_lift_amplitude:
        target[[L_HIP_PITCH, L_KNEE, L_ANKLE]] += TINY_LIFT_VECTOR * c.tiny_lift_amplitude
    return target


def _kf(name: str, support_mode: str, target: np.ndarray, duration: int) -> dict[str, Any]:
    return {
        "name": name,
        "support_mode": support_mode,
        "joint_targets": target.tolist(),
        "duration_steps": duration,
    }


def build_seed(c: UnloadCandidate, neutral_duration: int) -> dict[str, Any]:
    """Build an unload-only reference seed."""
    return {
        "schema": "seedon_gait_seed.v1",
        "target_type": "absolute",
        "description": "Generated Blue unload mechanism search reference.",
        "joint_names": JOINT_NAMES,
        "keyframes": [
            _kf("neutral", "double", _zero(), neutral_duration),
            _kf("right_unload", "left", _right_unload_target(c), c.preload_duration),
            _kf("neutral_after_right", "double", _zero(), neutral_duration),
            _kf("left_unload", "right", _left_unload_target(c), c.preload_duration),
        ],
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_candidate_files(c: UnloadCandidate, *, base_config: Path, out_dir: Path, neutral_duration: int) -> tuple[Path, Path]:
    """Write candidate seed/config files."""
    seed_path = out_dir / "seeds" / f"{c.candidate_id}.json"
    config_path = out_dir / "configs" / f"{c.candidate_id}.json"
    config = _load_json(base_config)
    config["reference_gait_seed_path"] = str(seed_path.relative_to(REPO_ROOT)).replace("\\", "/")
    config["reference_gait_seed_scale"] = 1.0
    _write_json(seed_path, build_seed(c, neutral_duration))
    _write_json(config_path, config)
    return config_path, seed_path


def _contact_state(info: dict[str, Any]) -> str:
    left = bool(info["left_contact"])
    right = bool(info["right_contact"])
    if left and right:
        return "both"
    if left:
        return "left"
    if right:
        return "right"
    return "none"


def _timeline_row(step: int, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": step,
        "phase_name": info["phase_name"],
        "contact_state": _contact_state(info),
        "support_side": info["support_side"],
        "swing_side": info["swing_side"],
        "force_ratio": float(info["force_ratio"]),
        "swing_force_ratio": float(info["swing_force_ratio"]),
        "base_height": float(info["base_height"]),
        "upright": float(info["upright"]),
        "left_force": float(info["left_normal_force"]),
        "right_force": float(info["right_normal_force"]),
    }


def audit_candidate(
    c: UnloadCandidate,
    *,
    config_path: Path,
    seed_path: Path,
    out_dir: Path,
    steps: int,
    seed: int,
    warmup_steps: int,
) -> UnloadAudit:
    """Run dynamic PD unload audit and write a timeline CSV."""
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=_load_config(config_path))
    total_weight = float(np.sum(env.model.body_mass) * 9.81)
    infos: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    try:
        env.reset(seed=seed)
        for step in range(1, steps + 1):
            _, _, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float64))
            info = dict(info)
            infos.append(info)
            timeline.append(_timeline_row(step, info))
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

    if not infos:
        return UnloadAudit(
            candidate_id=c.candidate_id,
            preload_duration=c.preload_duration,
            support_hip_roll=c.support_hip_roll,
            swing_hip_roll=c.swing_hip_roll,
            stance_knee=c.stance_knee,
            stance_ankle=c.stance_ankle,
            pelvis_lean_proxy=c.pelvis_lean_proxy,
            lateral_offset_proxy=c.lateral_offset_proxy,
            tiny_lift_amplitude=c.tiny_lift_amplitude,
            min_swing_force_ratio=float("inf"),
            max_support_ratio=0.0,
            duration_below_0_45=0,
            duration_below_0_40=0,
            contact_none_ratio=1.0,
            jump_count=1,
            base_drop_post_warmup=float("inf"),
            impact_post_warmup=float("inf"),
            upright=0.0,
            a_passed=False,
            b_passed=False,
            score=-float("inf"),
            config_path=str(config_path),
            seed_path=str(seed_path),
            timeline_path=str(timeline_path),
        )

    unload_rows = [row for row in timeline if str(row["phase_name"]).endswith("_unload")]
    post_infos = infos[warmup_steps:] or infos
    none = sum(1 for info in infos if _contact_state(info) == "none")
    jump_count = _count_contact_none_bursts(infos)
    post_initial_base = float(post_infos[0]["base_height"])
    base_drop = max(0.0, post_initial_base - min(float(info["base_height"]) for info in post_infos))
    impact = max(
        (float(info["left_normal_force"]) + float(info["right_normal_force"])) / max(total_weight, 1e-6)
        for info in post_infos
    )
    min_swing = min((float(row["swing_force_ratio"]) for row in unload_rows), default=float("inf"))
    max_support = max((float(row["force_ratio"]) for row in unload_rows), default=0.0)
    duration_below_045 = sum(1 for row in unload_rows if float(row["swing_force_ratio"]) <= 0.45)
    duration_below_040 = sum(1 for row in unload_rows if float(row["swing_force_ratio"]) <= 0.40)
    min_upright = min(float(info["upright"]) for info in infos)
    contact_none_ratio = none / max(1, len(infos))
    a_passed = (
        min_swing <= 0.45
        and max_support >= 0.55
        and none == 0
        and jump_count == 0
        and min_upright >= 0.99
    )
    b_passed = a_passed and min_swing <= 0.40 and max_support >= 0.60
    score = (
        (10.0 if a_passed else 0.0)
        + (20.0 if b_passed else 0.0)
        + max(0.0, 0.50 - min_swing) * 8.0
        + max_support * 2.0
        + duration_below_045 / max(1, len(unload_rows))
        + duration_below_040 / max(1, len(unload_rows)) * 2.0
        - max(0.0, base_drop - 0.015) * 30.0
        - max(0.0, impact - 1.2) * 2.0
        - none * 10.0
    )
    return UnloadAudit(
        candidate_id=c.candidate_id,
        preload_duration=c.preload_duration,
        support_hip_roll=c.support_hip_roll,
        swing_hip_roll=c.swing_hip_roll,
        stance_knee=c.stance_knee,
        stance_ankle=c.stance_ankle,
        pelvis_lean_proxy=c.pelvis_lean_proxy,
        lateral_offset_proxy=c.lateral_offset_proxy,
        tiny_lift_amplitude=c.tiny_lift_amplitude,
        min_swing_force_ratio=min_swing,
        max_support_ratio=max_support,
        duration_below_0_45=duration_below_045,
        duration_below_0_40=duration_below_040,
        contact_none_ratio=contact_none_ratio,
        jump_count=jump_count,
        base_drop_post_warmup=base_drop,
        impact_post_warmup=impact,
        upright=min_upright,
        a_passed=a_passed,
        b_passed=b_passed,
        score=score,
        config_path=str(config_path),
        seed_path=str(seed_path),
        timeline_path=str(timeline_path),
    )


def build_candidates(args: argparse.Namespace) -> list[UnloadCandidate]:
    """Build candidate grid."""
    candidates: list[UnloadCandidate] = []
    for duration, support_roll, swing_roll, knee, ankle, lean, lateral, tiny_lift in product(
        args.preload_durations,
        args.support_hip_rolls,
        args.swing_hip_rolls,
        args.stance_knees,
        args.stance_ankles,
        args.pelvis_lean_proxies,
        args.lateral_offset_proxies,
        args.tiny_lift_amplitudes,
    ):
        candidate_id = (
            f"dur{duration}_sr{_fmt(support_roll)}_wr{_fmt(swing_roll)}"
            f"_sk{_fmt(knee)}_sa{_fmt(ankle)}_lean{_fmt(lean)}"
            f"_lat{_fmt(lateral)}_tl{_fmt(tiny_lift)}"
        )
        candidates.append(
            UnloadCandidate(
                candidate_id,
                duration,
                support_roll,
                swing_roll,
                knee,
                ankle,
                lean,
                lateral,
                tiny_lift,
            )
        )
    return candidates


def write_results(path: Path, rows: list[UnloadAudit]) -> None:
    """Write aggregate CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def render_candidate(
    row: UnloadAudit,
    *,
    steps: int,
    seed: int,
    fps: int,
    width: int,
    height: int,
    out_dir: Path,
) -> Path:
    """Render one unload candidate to MP4."""
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


def write_summary(path: Path, rows: list[UnloadAudit]) -> None:
    """Write a compact markdown summary."""
    ranked = sorted(rows, key=lambda item: (item.b_passed, item.a_passed, item.score), reverse=True)
    lines = [
        "# Blue Unload Mechanism Search Summary",
        "",
        f"Candidates: {len(rows)}",
        f"A pass: {sum(1 for row in rows if row.a_passed)}",
        f"B pass: {sum(1 for row in rows if row.b_passed)}",
        "",
        "## Top Candidates",
        "",
        "| candidate | A | B | min_swing | max_support | below_045 | below_040 | none | jump | impact | drop | upright |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked[:10]:
        lines.append(
            "| "
            f"{row.candidate_id} | {row.a_passed} | {row.b_passed} | "
            f"{row.min_swing_force_ratio:.3f} | {row.max_support_ratio:.3f} | "
            f"{row.duration_below_0_45} | {row.duration_below_0_40} | "
            f"{row.contact_none_ratio:.3f} | {row.jump_count} | "
            f"{row.impact_post_warmup:.3f} | {row.base_drop_post_warmup:.4f} | "
            f"{row.upright:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-warmup-steps", type=int, default=20)
    parser.add_argument("--neutral-duration", type=int, default=30)
    parser.add_argument("--preload-durations", type=_parse_int_list, default="120,160,200")
    parser.add_argument("--support-hip-rolls", type=_parse_float_list, default="0.025,0.05,0.075")
    parser.add_argument("--swing-hip-rolls", type=_parse_float_list, default="-0.02,-0.04")
    parser.add_argument("--stance-knees", type=_parse_float_list, default="-0.03,-0.05")
    parser.add_argument("--stance-ankles", type=_parse_float_list, default="0.02,0.04")
    parser.add_argument("--pelvis-lean-proxies", type=_parse_float_list, default="-0.015,0,0.015")
    parser.add_argument("--lateral-offset-proxies", type=_parse_float_list, default="0,0.015")
    parser.add_argument("--tiny-lift-amplitudes", type=_parse_float_list, default="0")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--render-top-k", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=368)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run unload mechanism search."""
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[UnloadAudit] = []
    for candidate in build_candidates(args):
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
            out_dir=args.out_dir,
            steps=args.steps,
            seed=args.seed,
            warmup_steps=args.audit_warmup_steps,
        )
        rows.append(row)
        print(
            f"{row.candidate_id}: A={row.a_passed} B={row.b_passed} "
            f"min_swing={row.min_swing_force_ratio:.3f} max_support={row.max_support_ratio:.3f} "
            f"below045={row.duration_below_0_45} below040={row.duration_below_0_40} "
            f"none={row.contact_none_ratio:.3f} jump={row.jump_count} "
            f"impact={row.impact_post_warmup:.3f} drop={row.base_drop_post_warmup:.4f} "
            f"upright={row.upright:.3f}"
        )
    ranked = sorted(rows, key=lambda item: (item.b_passed, item.a_passed, item.score), reverse=True)
    write_results(args.out_dir / "blue_unload_mechanism_search.csv", rows)
    write_results(args.out_dir / "blue_unload_mechanism_search_top10.csv", ranked[: args.top_k])
    write_summary(args.out_dir / "summary.md", rows)
    for row in ranked[: args.top_k]:
        print(
            "TOP "
            f"{row.candidate_id}: A={row.a_passed} B={row.b_passed} score={row.score:.3f} "
            f"min_swing={row.min_swing_force_ratio:.3f} max_support={row.max_support_ratio:.3f} "
            f"below045={row.duration_below_0_45} below040={row.duration_below_0_40} "
            f"impact={row.impact_post_warmup:.3f} drop={row.base_drop_post_warmup:.4f}"
        )
    for row in ranked[: args.render_top_k]:
        print(
            "rendered: "
            f"{render_candidate(row, steps=args.steps, seed=args.seed, fps=args.fps, width=args.width, height=args.height, out_dir=args.out_dir)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
