"""Build Blue-like step reference v3 with closed-loop preload trigger search."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from sedon_baseline.env import SedonStandingEnv
from tools.audit_sedon_shuffle_v0 import _count_contact_none_bursts, _load_config, audit_shuffle
from tools.render_sedon_policy_comparison import _make_side_camera, _save_mp4


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "sedon" / "reference_teacher_pose_1_4_imitation.json"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "sedon_debug" / "blue_step_reference_v3_closed_loop"
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
class V3Candidate:
    """One v3 closed-loop unload candidate."""

    candidate_id: str
    support_hip_roll: float
    swing_hip_roll: float
    stance_knee: float
    stance_ankle: float
    lift_amplitude: float
    lift_duration: int
    landing_duration: int


@dataclass(frozen=True)
class TriggerResult:
    """Dynamic preload trigger result for one swing side."""

    triggered: bool
    trigger_step: int
    swing_force_ratio: float
    support_ratio: float


@dataclass(frozen=True)
class V3Audit:
    """Aggregate dynamic PD result for one v3 candidate."""

    candidate_id: str
    support_hip_roll: float
    swing_hip_roll: float
    stance_knee: float
    stance_ankle: float
    lift_amplitude: float
    lift_duration: int
    landing_duration: int
    right_triggered: bool
    left_triggered: bool
    unload_trigger_step: int
    right_unload_trigger_step: int
    left_unload_trigger_step: int
    swing_force_ratio_at_lift_start: float
    support_ratio_at_lift_start: float
    right_swing_force_ratio_at_lift_start: float
    left_swing_force_ratio_at_lift_start: float
    max_clearance: float
    single_contact_ratio: float
    contact_none_ratio: float
    jump_count: int
    impact_post: float
    base_drop_post: float
    upright: float
    v3_a_passed: bool
    v3_b_passed: bool
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


def _right_preload(c: V3Candidate) -> np.ndarray:
    target = _zero()
    target[L_HIP_ROLL] = c.support_hip_roll
    target[R_HIP_ROLL] = c.swing_hip_roll
    target[L_KNEE] = c.stance_knee
    target[L_ANKLE] = c.stance_ankle
    return target


def _left_preload(c: V3Candidate) -> np.ndarray:
    target = _zero()
    target[R_HIP_ROLL] = -c.support_hip_roll
    target[L_HIP_ROLL] = -c.swing_hip_roll
    target[R_KNEE] = c.stance_knee
    target[R_ANKLE] = c.stance_ankle
    return target


def _right_lift(c: V3Candidate) -> np.ndarray:
    target = _right_preload(c)
    target[[R_HIP_PITCH, R_KNEE, R_ANKLE]] = LIFT_VECTOR * c.lift_amplitude
    return target


def _left_lift(c: V3Candidate) -> np.ndarray:
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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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


def build_probe_seed(c: V3Candidate, *, neutral_duration: int, max_preload_steps: int) -> dict[str, Any]:
    """Build a preload-only seed used to discover dynamic unload trigger steps."""
    return {
        "schema": "sedon_gait_seed.v1",
        "target_type": "absolute",
        "description": "Generated Blue-step v3 preload trigger probe.",
        "joint_names": JOINT_NAMES,
        "keyframes": [
            _kf("neutral", "double", _zero(), neutral_duration),
            _kf("right_preload_probe", "left", _right_preload(c), max_preload_steps),
            _kf("neutral_after_right", "double", _zero(), neutral_duration),
            _kf("left_preload_probe", "right", _left_preload(c), max_preload_steps),
        ],
    }


def build_final_seed(
    c: V3Candidate,
    *,
    neutral_duration: int,
    right_preload_steps: int,
    left_preload_steps: int,
) -> dict[str, Any]:
    """Build the final reference seed using dynamic trigger-derived preload durations."""
    right_land = _right_preload(c) * 0.35
    left_land = _left_preload(c) * 0.35
    return {
        "schema": "sedon_gait_seed.v1",
        "target_type": "absolute",
        "description": "Generated Blue-step v3 closed-loop unload-before-lift reference.",
        "joint_names": JOINT_NAMES,
        "keyframes": [
            _kf("neutral", "double", _zero(), neutral_duration),
            _kf("right_preload_unload", "left", _right_preload(c), right_preload_steps),
            _kf("right_micro_lift", "left", _right_lift(c), c.lift_duration),
            _kf("right_soft_land", "double", right_land, c.landing_duration),
            _kf("neutral_after_right", "double", _zero(), neutral_duration),
            _kf("left_preload_unload", "right", _left_preload(c), left_preload_steps),
            _kf("left_micro_lift", "right", _left_lift(c), c.lift_duration),
            _kf("left_soft_land", "double", left_land, c.landing_duration),
        ],
    }


def _write_config(base_config: Path, seed_path: Path, config_path: Path) -> None:
    config = _load_json(base_config)
    config["reference_gait_seed_path"] = str(seed_path.relative_to(REPO_ROOT)).replace("\\", "/")
    config["reference_gait_seed_scale"] = 1.0
    _write_json(config_path, config)


def write_probe_files(
    c: V3Candidate,
    *,
    base_config: Path,
    out_dir: Path,
    neutral_duration: int,
    max_preload_steps: int,
) -> tuple[Path, Path]:
    """Write a preload probe seed and matching config."""
    seed_path = out_dir / "probe_seeds" / f"{c.candidate_id}.json"
    config_path = out_dir / "probe_configs" / f"{c.candidate_id}.json"
    _write_json(seed_path, build_probe_seed(c, neutral_duration=neutral_duration, max_preload_steps=max_preload_steps))
    _write_config(base_config, seed_path, config_path)
    return config_path, seed_path


def write_final_files(
    c: V3Candidate,
    *,
    base_config: Path,
    out_dir: Path,
    neutral_duration: int,
    right_preload_steps: int,
    left_preload_steps: int,
) -> tuple[Path, Path]:
    """Write the final closed-loop-triggered seed and matching config."""
    seed_path = out_dir / "seeds" / f"{c.candidate_id}.json"
    config_path = out_dir / "configs" / f"{c.candidate_id}.json"
    _write_json(
        seed_path,
        build_final_seed(
            c,
            neutral_duration=neutral_duration,
            right_preload_steps=right_preload_steps,
            left_preload_steps=left_preload_steps,
        ),
    )
    _write_config(base_config, seed_path, config_path)
    return config_path, seed_path


def find_unload_triggers(
    *,
    config_path: Path,
    seed: int,
    neutral_duration: int,
    max_preload_steps: int,
    swing_force_threshold: float,
    support_ratio_threshold: float,
) -> tuple[TriggerResult, TriggerResult]:
    """Run preload-only dynamic PD and return right/left unload trigger steps."""
    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=_load_config(config_path))
    right = TriggerResult(False, max_preload_steps, float("inf"), 0.0)
    left = TriggerResult(False, max_preload_steps, float("inf"), 0.0)
    right_local_step = 0
    left_local_step = 0
    max_steps = neutral_duration * 2 + max_preload_steps * 2 + 10
    try:
        env.reset(seed=seed)
        for _ in range(max_steps):
            _, _, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float64))
            info = dict(info)
            phase_name = str(info["phase_name"])
            if phase_name == "right_preload_probe":
                right_local_step += 1
                if not right.triggered and (
                    float(info["swing_force_ratio"]) <= swing_force_threshold
                    or float(info["force_ratio"]) >= support_ratio_threshold
                ):
                    right = TriggerResult(
                        True,
                        right_local_step,
                        float(info["swing_force_ratio"]),
                        float(info["force_ratio"]),
                    )
            elif phase_name == "left_preload_probe":
                left_local_step += 1
                if not left.triggered and (
                    float(info["swing_force_ratio"]) <= swing_force_threshold
                    or float(info["force_ratio"]) >= support_ratio_threshold
                ):
                    left = TriggerResult(
                        True,
                        left_local_step,
                        float(info["swing_force_ratio"]),
                        float(info["force_ratio"]),
                    )
            if terminated or truncated:
                break
    finally:
        env.close()
    return right, left


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


def audit_final_candidate(
    c: V3Candidate,
    *,
    config_path: Path,
    seed_path: Path,
    right_trigger: TriggerResult,
    left_trigger: TriggerResult,
    baseline_impact: float,
    out_dir: Path,
    steps: int,
    seed: int,
    warmup_steps: int,
) -> V3Audit:
    """Run final dynamic PD audit and write a contact-state timeline CSV."""
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

    if not infos:
        return V3Audit(
            candidate_id=c.candidate_id,
            support_hip_roll=c.support_hip_roll,
            swing_hip_roll=c.swing_hip_roll,
            stance_knee=c.stance_knee,
            stance_ankle=c.stance_ankle,
            lift_amplitude=c.lift_amplitude,
            lift_duration=c.lift_duration,
            landing_duration=c.landing_duration,
            right_triggered=right_trigger.triggered,
            left_triggered=left_trigger.triggered,
            unload_trigger_step=max(right_trigger.trigger_step, left_trigger.trigger_step),
            right_unload_trigger_step=right_trigger.trigger_step,
            left_unload_trigger_step=left_trigger.trigger_step,
            swing_force_ratio_at_lift_start=float("inf"),
            support_ratio_at_lift_start=0.0,
            right_swing_force_ratio_at_lift_start=float("inf"),
            left_swing_force_ratio_at_lift_start=float("inf"),
            max_clearance=0.0,
            single_contact_ratio=0.0,
            contact_none_ratio=1.0,
            jump_count=1,
            impact_post=float("inf"),
            base_drop_post=float("inf"),
            upright=0.0,
            v3_a_passed=False,
            v3_b_passed=False,
            score=-float("inf"),
            config_path=str(config_path),
            seed_path=str(seed_path),
            timeline_path=str(timeline_path),
        )

    right_lift_rows = [row for row in timeline if str(row["phase_name"]) == "right_micro_lift"]
    left_lift_rows = [row for row in timeline if str(row["phase_name"]) == "left_micro_lift"]
    right_lift_swing = float(right_lift_rows[0]["swing_force_ratio"]) if right_lift_rows else float("inf")
    left_lift_swing = float(left_lift_rows[0]["swing_force_ratio"]) if left_lift_rows else float("inf")
    right_lift_support = float(right_lift_rows[0]["force_ratio"]) if right_lift_rows else 0.0
    left_lift_support = float(left_lift_rows[0]["force_ratio"]) if left_lift_rows else 0.0
    post_infos = infos[warmup_steps:] or infos
    contact_forces_post = [
        float(info["left_normal_force"]) + float(info["right_normal_force"])
        for info in post_infos
    ]
    post_initial_base = float(post_infos[0]["base_height"])
    base_drop = max(
        0.0,
        post_initial_base - min(float(info["base_height"]) for info in post_infos),
    )
    none = sum(1 for info in infos if _contact_state(info) == "none")
    single = sum(1 for info in infos if _contact_state(info) in {"left", "right"})
    jump_count = _count_contact_none_bursts(infos)
    impact = max(contact_forces_post, default=0.0) / max(total_weight, 1e-6)
    max_clearance = max(max(0.0, value) for value in signed_clearances)
    min_upright = min(float(info["upright"]) for info in infos)
    swing_force_at_lift = max(right_lift_swing, left_lift_swing)
    support_ratio_at_lift = min(right_lift_support, left_lift_support)
    trigger_success = right_trigger.triggered and left_trigger.triggered
    v3_a_passed = (
        trigger_success
        and swing_force_at_lift <= 0.40
        and none == 0
        and jump_count == 0
    )
    v3_b_passed = (
        v3_a_passed
        and max_clearance >= 0.0015
        and impact <= baseline_impact * 1.2
        and base_drop <= 0.015
        and min_upright >= 0.99
    )
    score = (
        (10.0 if v3_a_passed else 0.0)
        + (20.0 if v3_b_passed else 0.0)
        + max_clearance * 1000.0
        + max(0.0, 0.40 - swing_force_at_lift) * 2.0
        + single / max(1, len(infos)) * 0.5
        - max(0.0, impact - baseline_impact) * 2.0
        - max(0.0, base_drop - 0.010) * 30.0
        - none * 10.0
    )
    total = max(1, len(infos))
    return V3Audit(
        candidate_id=c.candidate_id,
        support_hip_roll=c.support_hip_roll,
        swing_hip_roll=c.swing_hip_roll,
        stance_knee=c.stance_knee,
        stance_ankle=c.stance_ankle,
        lift_amplitude=c.lift_amplitude,
        lift_duration=c.lift_duration,
        landing_duration=c.landing_duration,
        right_triggered=right_trigger.triggered,
        left_triggered=left_trigger.triggered,
        unload_trigger_step=max(right_trigger.trigger_step, left_trigger.trigger_step),
        right_unload_trigger_step=right_trigger.trigger_step,
        left_unload_trigger_step=left_trigger.trigger_step,
        swing_force_ratio_at_lift_start=swing_force_at_lift,
        support_ratio_at_lift_start=support_ratio_at_lift,
        right_swing_force_ratio_at_lift_start=right_lift_swing,
        left_swing_force_ratio_at_lift_start=left_lift_swing,
        max_clearance=max_clearance,
        single_contact_ratio=single / total,
        contact_none_ratio=none / total,
        jump_count=jump_count,
        impact_post=impact,
        base_drop_post=base_drop,
        upright=min_upright,
        v3_a_passed=v3_a_passed,
        v3_b_passed=v3_b_passed,
        score=score,
        config_path=str(config_path),
        seed_path=str(seed_path),
        timeline_path=str(timeline_path),
    )


def build_candidates(args: argparse.Namespace) -> list[V3Candidate]:
    """Build sweep candidates."""
    candidates: list[V3Candidate] = []
    for support_roll, swing_roll, knee, ankle, lift, lift_dur, land_dur in product(
        args.support_hip_rolls,
        args.swing_hip_rolls,
        args.stance_knees,
        args.stance_ankles,
        args.lift_amplitudes,
        args.lift_durations,
        args.landing_durations,
    ):
        candidate_id = (
            f"sr{_fmt(support_roll)}_wr{_fmt(swing_roll)}"
            f"_sk{_fmt(knee)}_sa{_fmt(ankle)}_lift{_fmt(lift)}"
            f"_ld{lift_dur}_land{land_dur}"
        )
        candidates.append(
            V3Candidate(
                candidate_id,
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


def write_results(path: Path, rows: list[V3Audit]) -> None:
    """Write aggregate CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def render_candidate(
    row: V3Audit,
    *,
    steps: int,
    seed: int,
    fps: int,
    width: int,
    height: int,
    out_dir: Path,
) -> Path:
    """Render one reference candidate to MP4."""
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


def write_summary(path: Path, rows: list[V3Audit]) -> None:
    """Write a compact markdown summary for quick review."""
    v3_a_count = sum(1 for row in rows if row.v3_a_passed)
    v3_b_count = sum(1 for row in rows if row.v3_b_passed)
    ranked = sorted(rows, key=lambda item: (item.v3_b_passed, item.v3_a_passed, item.score), reverse=True)
    lines = [
        "# Blue Step Reference V3 Closed Loop Summary",
        "",
        f"Candidates: {len(rows)}",
        f"v3-A pass: {v3_a_count}",
        f"v3-B pass: {v3_b_count}",
        "",
        "## Top Candidates",
        "",
        "| candidate | A | B | right_trigger | left_trigger | trigger_step | swing_at_lift | support_at_lift | clearance | single | impact | drop | upright |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked[:10]:
        lines.append(
            "| "
            f"{row.candidate_id} | {row.v3_a_passed} | {row.v3_b_passed} | "
            f"{row.right_triggered} | {row.left_triggered} | "
            f"{row.unload_trigger_step} | {row.swing_force_ratio_at_lift_start:.3f} | "
            f"{row.support_ratio_at_lift_start:.3f} | {row.max_clearance:.4f} | "
            f"{row.single_contact_ratio:.3f} | {row.impact_post:.3f} | "
            f"{row.base_drop_post:.4f} | {row.upright:.3f} |"
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
    parser.add_argument("--max-preload-steps", type=int, default=120)
    parser.add_argument("--swing-force-threshold", type=float, default=0.40)
    parser.add_argument("--support-ratio-threshold", type=float, default=0.60)
    parser.add_argument("--support-hip-rolls", type=_parse_float_list, default="0.025,0.05")
    parser.add_argument("--swing-hip-rolls", type=_parse_float_list, default="-0.02")
    parser.add_argument("--stance-knees", type=_parse_float_list, default="-0.03")
    parser.add_argument("--stance-ankles", type=_parse_float_list, default="0.02")
    parser.add_argument("--lift-amplitudes", type=_parse_float_list, default="1.25,1.5,1.75,2.0")
    parser.add_argument("--lift-durations", type=_parse_int_list, default="45,60,90")
    parser.add_argument("--landing-durations", type=_parse_int_list, default="60,90")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--render-top-k", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=368)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build, audit, and optionally render v3 closed-loop reference candidates."""
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
    rows: list[V3Audit] = []
    for candidate in build_candidates(args):
        probe_config_path, _ = write_probe_files(
            candidate,
            base_config=args.base_config,
            out_dir=args.out_dir,
            neutral_duration=args.neutral_duration,
            max_preload_steps=args.max_preload_steps,
        )
        right_trigger, left_trigger = find_unload_triggers(
            config_path=probe_config_path,
            seed=args.seed,
            neutral_duration=args.neutral_duration,
            max_preload_steps=args.max_preload_steps,
            swing_force_threshold=args.swing_force_threshold,
            support_ratio_threshold=args.support_ratio_threshold,
        )
        config_path, seed_path = write_final_files(
            candidate,
            base_config=args.base_config,
            out_dir=args.out_dir,
            neutral_duration=args.neutral_duration,
            right_preload_steps=right_trigger.trigger_step,
            left_preload_steps=left_trigger.trigger_step,
        )
        row = audit_final_candidate(
            candidate,
            config_path=config_path,
            seed_path=seed_path,
            right_trigger=right_trigger,
            left_trigger=left_trigger,
            baseline_impact=baseline.landing_impact_post_warmup,
            out_dir=args.out_dir,
            steps=args.steps,
            seed=args.seed,
            warmup_steps=args.audit_warmup_steps,
        )
        rows.append(row)
        print(
            f"{row.candidate_id}: A={row.v3_a_passed} B={row.v3_b_passed} "
            f"trigger={row.unload_trigger_step} swing={row.swing_force_ratio_at_lift_start:.3f} "
            f"support={row.support_ratio_at_lift_start:.3f} clear={row.max_clearance:.4f} "
            f"single={row.single_contact_ratio:.3f} none={row.contact_none_ratio:.3f} "
            f"impact={row.impact_post:.3f} drop={row.base_drop_post:.4f} upright={row.upright:.3f}"
        )
    ranked = sorted(rows, key=lambda item: (item.v3_b_passed, item.v3_a_passed, item.score), reverse=True)
    write_results(args.out_dir / "blue_step_reference_v3_closed_loop.csv", rows)
    write_results(args.out_dir / "blue_step_reference_v3_closed_loop_top10.csv", ranked[: args.top_k])
    write_summary(args.out_dir / "summary.md", rows)
    for row in ranked[: args.top_k]:
        print(
            "TOP "
            f"{row.candidate_id}: A={row.v3_a_passed} B={row.v3_b_passed} score={row.score:.3f} "
            f"trigger={row.unload_trigger_step} swing={row.swing_force_ratio_at_lift_start:.3f} "
            f"support={row.support_ratio_at_lift_start:.3f} clear={row.max_clearance:.4f} "
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
