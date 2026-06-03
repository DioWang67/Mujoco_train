"""Build Blue-like step reference v2 with preload unload before micro-lift."""

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
from tools.audit_seedon_shuffle_v0 import _count_contact_none_bursts, _load_config, audit_shuffle
from tools.render_seedon_policy_comparison import _make_side_camera, _save_mp4


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "seedon" / "reference_teacher_pose_1_4_imitation.json"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "seedon_debug" / "blue_step_reference_v2"
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
LIFT_VECTOR = np.array([0.18, -0.36, 0.18], dtype=np.float64)


@dataclass(frozen=True)
class V2Candidate:
    """One v2 preload-unload candidate."""

    candidate_id: str
    preload_duration: int
    support_hip_roll: float
    swing_hip_roll: float
    stance_knee: float
    stance_ankle: float
    lift_amplitude: float
    lift_duration: int
    landing_duration: int


@dataclass(frozen=True)
class V2Audit:
    """Aggregate dynamic PD result for one v2 candidate."""

    candidate_id: str
    preload_duration: int
    support_hip_roll: float
    swing_hip_roll: float
    stance_knee: float
    stance_ankle: float
    lift_amplitude: float
    lift_duration: int
    landing_duration: int
    max_clearance: float
    min_swing_force_ratio: float
    support_ratio_at_lift_start: float
    both_contact_ratio: float
    single_contact_ratio: float
    contact_none_ratio: float
    jump_count: int
    peak_support_ratio: float
    landing_impact_post_warmup: float
    base_height_drop_post_warmup: float
    upright: float
    tracking_error: float
    negative_clearance_ratio: float
    passed: bool
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


def _right_preload(c: V2Candidate) -> np.ndarray:
    target = _zero()
    target[L_HIP_ROLL] = c.support_hip_roll
    target[R_HIP_ROLL] = c.swing_hip_roll
    target[L_KNEE] = c.stance_knee
    target[L_ANKLE] = c.stance_ankle
    return target


def _left_preload(c: V2Candidate) -> np.ndarray:
    target = _zero()
    target[R_HIP_ROLL] = -c.support_hip_roll
    target[L_HIP_ROLL] = -c.swing_hip_roll
    target[R_KNEE] = c.stance_knee
    target[R_ANKLE] = c.stance_ankle
    return target


def _right_lift(c: V2Candidate) -> np.ndarray:
    target = _right_preload(c)
    target[[R_HIP_PITCH, R_KNEE, R_ANKLE]] = LIFT_VECTOR * c.lift_amplitude
    return target


def _left_lift(c: V2Candidate) -> np.ndarray:
    target = _left_preload(c)
    target[[L_HIP_PITCH, L_KNEE, L_ANKLE]] = LIFT_VECTOR * c.lift_amplitude
    return target


def _kf(name: str, support_mode: str, target: np.ndarray, duration: int) -> dict[str, Any]:
    return {
        "name": name,
        "support_mode": support_mode,
        "joint_targets": target.tolist(),
        "duration_steps": duration,
    }


def build_seed(c: V2Candidate, neutral_duration: int) -> dict[str, Any]:
    """Build a new unload-gated Blue-step v2 reference seed."""
    right_land = _right_preload(c) * 0.35
    left_land = _left_preload(c) * 0.35
    return {
        "schema": "seedon_gait_seed.v1",
        "target_type": "absolute",
        "description": "Generated Blue-step v2 unload-before-lift reference.",
        "joint_names": JOINT_NAMES,
        "keyframes": [
            _kf("neutral", "double", _zero(), neutral_duration),
            _kf("right_preload_unload", "left", _right_preload(c), c.preload_duration),
            _kf("right_micro_lift", "left", _right_lift(c), c.lift_duration),
            _kf("right_soft_land", "double", right_land, c.landing_duration),
            _kf("neutral_after_right", "double", _zero(), neutral_duration),
            _kf("left_preload_unload", "right", _left_preload(c), c.preload_duration),
            _kf("left_micro_lift", "right", _left_lift(c), c.lift_duration),
            _kf("left_soft_land", "double", left_land, c.landing_duration),
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


def write_candidate_files(c: V2Candidate, *, base_config: Path, out_dir: Path, neutral_duration: int) -> tuple[Path, Path]:
    seed_path = out_dir / "seeds" / f"{c.candidate_id}.json"
    config_path = out_dir / "configs" / f"{c.candidate_id}.json"
    config = _load_json(base_config)
    config["reference_gait_seed_path"] = str(seed_path.relative_to(REPO_ROOT)).replace("\\", "/")
    config["reference_gait_seed_scale"] = 1.0
    _write_json(seed_path, build_seed(c, neutral_duration),)
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


def audit_candidate(
    c: V2Candidate,
    *,
    config_path: Path,
    seed_path: Path,
    baseline_impact: float,
    out_dir: Path,
    steps: int,
    seed: int,
    warmup_steps: int,
) -> V2Audit:
    """Run dynamic PD audit and write a contact-state timeline CSV."""
    config = _load_config(config_path)
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=config)
    total_weight = float(np.sum(env.model.body_mass) * 9.81)
    infos: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    signed_clearances: list[float] = []
    try:
        obs, _ = env.reset(seed=seed)
        del obs
        for step in range(1, steps + 1):
            _, _, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float64)
            )
            info = dict(info)
            phase = env._task_phase_metadata()
            support_z, swing_z = env._task_foot_bottom_heights(phase)
            signed_clearance = float(swing_z - support_z)
            signed_clearances.append(signed_clearance)
            infos.append(info)
            timeline.append(
                {
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
            )
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
        return V2Audit(
            candidate_id=c.candidate_id,
            preload_duration=c.preload_duration,
            support_hip_roll=c.support_hip_roll,
            swing_hip_roll=c.swing_hip_roll,
            stance_knee=c.stance_knee,
            stance_ankle=c.stance_ankle,
            lift_amplitude=c.lift_amplitude,
            lift_duration=c.lift_duration,
            landing_duration=c.landing_duration,
            max_clearance=0.0,
            min_swing_force_ratio=float("inf"),
            support_ratio_at_lift_start=0.0,
            both_contact_ratio=0.0,
            single_contact_ratio=0.0,
            contact_none_ratio=1.0,
            jump_count=1,
            peak_support_ratio=0.0,
            landing_impact_post_warmup=float("inf"),
            base_height_drop_post_warmup=float("inf"),
            upright=0.0,
            tracking_error=float("inf"),
            negative_clearance_ratio=1.0,
            passed=False,
            score=-float("inf"),
            config_path=str(config_path),
            seed_path=str(seed_path),
            timeline_path=str(timeline_path),
        )

    post_infos = infos[warmup_steps:] or infos
    both = sum(1 for info in infos if _contact_state(info) == "both")
    single = sum(1 for info in infos if _contact_state(info) in {"left", "right"})
    none = sum(1 for info in infos if _contact_state(info) == "none")
    preload_rows = [row for row in timeline if "preload" in str(row["phase_name"])]
    lift_rows = [row for row in timeline if "micro_lift" in str(row["phase_name"])]
    support_ratio_at_lift_start = float(lift_rows[0]["force_ratio"]) if lift_rows else 0.0
    contact_forces_post = [
        float(info["left_normal_force"]) + float(info["right_normal_force"])
        for info in post_infos
    ]
    post_initial_base = float(post_infos[0]["base_height"])
    base_drop = max(
        0.0,
        post_initial_base - min(float(info["base_height"]) for info in post_infos),
    )
    tracking_errors = [
        float(np.sqrt(float(info.get("joint_position_error_l2", 0.0)) / 10.0))
        for info in infos
    ]
    min_swing_force_ratio = (
        min(float(row["swing_force_ratio"]) for row in preload_rows)
        if preload_rows
        else float("inf")
    )
    max_clearance = max(max(0.0, value) for value in signed_clearances)
    impact = max(contact_forces_post, default=0.0) / max(total_weight, 1e-6)
    min_upright = min(float(info["upright"]) for info in infos)
    jump_count = _count_contact_none_bursts(infos)
    negative_clearance_ratio = float(np.mean([value < 0.0 for value in signed_clearances]))
    passed = (
        none == 0
        and jump_count == 0
        and min_upright >= 0.99
        and min_swing_force_ratio <= 0.45
        and max_clearance >= 0.0015
        and impact <= baseline_impact * 1.2
        and base_drop <= 0.015
    )
    score = (
        max_clearance * 1000.0
        + max(0.0, 0.45 - min_swing_force_ratio) * 1.5
        + single / max(1, len(infos)) * 0.25
        - max(0.0, impact - baseline_impact) * 2.0
        - max(0.0, base_drop - 0.010) * 30.0
        - none * 10.0
        - negative_clearance_ratio * 0.2
    )
    total = max(1, len(infos))
    return V2Audit(
        candidate_id=c.candidate_id,
        preload_duration=c.preload_duration,
        support_hip_roll=c.support_hip_roll,
        swing_hip_roll=c.swing_hip_roll,
        stance_knee=c.stance_knee,
        stance_ankle=c.stance_ankle,
        lift_amplitude=c.lift_amplitude,
        lift_duration=c.lift_duration,
        landing_duration=c.landing_duration,
        max_clearance=max_clearance,
        min_swing_force_ratio=min_swing_force_ratio,
        support_ratio_at_lift_start=support_ratio_at_lift_start,
        both_contact_ratio=both / total,
        single_contact_ratio=single / total,
        contact_none_ratio=none / total,
        jump_count=jump_count,
        peak_support_ratio=max(float(info["force_ratio"]) for info in infos),
        landing_impact_post_warmup=impact,
        base_height_drop_post_warmup=base_drop,
        upright=min_upright,
        tracking_error=float(np.mean(tracking_errors)),
        negative_clearance_ratio=negative_clearance_ratio,
        passed=passed,
        score=score,
        config_path=str(config_path),
        seed_path=str(seed_path),
        timeline_path=str(timeline_path),
    )


def build_candidates(args: argparse.Namespace) -> list[V2Candidate]:
    """Build sweep candidates."""
    candidates: list[V2Candidate] = []
    for preload_duration, support_roll, swing_roll, knee, ankle, lift, lift_dur, land_dur in product(
        args.preload_durations,
        args.support_hip_rolls,
        args.swing_hip_rolls,
        args.stance_knees,
        args.stance_ankles,
        args.lift_amplitudes,
        args.lift_durations,
        args.landing_durations,
    ):
        candidate_id = (
            f"pd{preload_duration}_sr{_fmt(support_roll)}_wr{_fmt(swing_roll)}"
            f"_sk{_fmt(knee)}_sa{_fmt(ankle)}_lift{_fmt(lift)}"
            f"_ld{lift_dur}_land{land_dur}"
        )
        candidates.append(
            V2Candidate(
                candidate_id,
                preload_duration,
                support_roll,
                swing_roll,
                knee,
                ankle,
                lift,
                lift_dur,
                land_dur,
            )
        )
    return candidates


def write_results(path: Path, rows: list[V2Audit]) -> None:
    """Write aggregate rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def render_candidate(row: V2Audit, *, steps: int, seed: int, fps: int, width: int, height: int, out_dir: Path) -> Path:
    """Render one reference candidate to MP4."""
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


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-warmup-steps", type=int, default=20)
    parser.add_argument("--neutral-duration", type=int, default=30)
    parser.add_argument("--preload-durations", type=_parse_int_list, default="60,90,120")
    parser.add_argument("--support-hip-rolls", type=_parse_float_list, default="0.025,0.05")
    parser.add_argument("--swing-hip-rolls", type=_parse_float_list, default="-0.02")
    parser.add_argument("--stance-knees", type=_parse_float_list, default="-0.03")
    parser.add_argument("--stance-ankles", type=_parse_float_list, default="0.02")
    parser.add_argument("--lift-amplitudes", type=_parse_float_list, default="1.0,1.25,1.5,1.75")
    parser.add_argument("--lift-durations", type=_parse_int_list, default="45,60,90")
    parser.add_argument("--landing-durations", type=_parse_int_list, default="60,90")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--render-top-k", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=368)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build, audit, and render v2 reference candidates."""
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
    rows: list[V2Audit] = []
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
            baseline_impact=baseline.landing_impact_post_warmup,
            out_dir=args.out_dir,
            steps=args.steps,
            seed=args.seed,
            warmup_steps=args.audit_warmup_steps,
        )
        rows.append(row)
        print(
            f"{row.candidate_id}: pass={row.passed} "
            f"unload={row.min_swing_force_ratio:.3f} clear={row.max_clearance:.4f} "
            f"single={row.single_contact_ratio:.3f} none={row.contact_none_ratio:.3f} "
            f"impact={row.landing_impact_post_warmup:.3f} drop={row.base_height_drop_post_warmup:.4f} "
            f"upright={row.upright:.3f}"
        )
    ranked = sorted(rows, key=lambda item: (item.passed, item.score), reverse=True)
    write_results(args.out_dir / "blue_step_reference_v2.csv", rows)
    write_results(args.out_dir / "blue_step_reference_v2_top10.csv", ranked[: args.top_k])
    for row in ranked[: args.top_k]:
        print(
            "TOP "
            f"{row.candidate_id}: pass={row.passed} score={row.score:.3f} "
            f"unload={row.min_swing_force_ratio:.3f} lift_start={row.support_ratio_at_lift_start:.3f} "
            f"clear={row.max_clearance:.4f} impact={row.landing_impact_post_warmup:.3f} "
            f"drop={row.base_height_drop_post_warmup:.4f}"
        )
    for row in ranked[: args.render_top_k]:
        print(
            "rendered: "
            f"{render_candidate(row, steps=args.steps, seed=args.seed, fps=args.fps, width=args.width, height=args.height, out_dir=args.out_dir)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
