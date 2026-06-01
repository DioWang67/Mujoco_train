"""Build and sweep Blue-like visible stepping v1 references with teacher PD."""

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
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "sedon_debug" / "blue_step_reference_v1"
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
RIGHT_HIP_ROLL = 1
RIGHT_HIP_PITCH = 2
RIGHT_KNEE = 3
RIGHT_ANKLE = 4
LEFT_HIP_ROLL = 6
LEFT_HIP_PITCH = 7
LEFT_KNEE = 8
LEFT_ANKLE = 9
BASE_LIFT_VECTOR = np.array([0.18, -0.36, 0.18], dtype=np.float64)


@dataclass(frozen=True)
class BlueStepCandidate:
    """One Blue-step reference candidate."""

    candidate_id: str
    preload_amplitude: float
    lift_amplitude: float
    lift_duration: int
    landing_duration: int


@dataclass(frozen=True)
class BlueStepAudit:
    """Dynamic PD audit result for one Blue-step reference."""

    candidate_id: str
    preload_amplitude: float
    lift_amplitude: float
    lift_duration: int
    landing_duration: int
    steps: int
    max_clearance: float
    both_contact_ratio: float
    single_contact_ratio: float
    contact_none_ratio: float
    jump_count: int
    peak_support_ratio: float
    landing_impact_raw: float
    landing_impact_post_warmup: float
    base_height_drop_raw: float
    base_height_drop_post_warmup: float
    upright: float
    tracking_error: float
    max_tracking_error: float
    negative_clearance_ratio: float
    passed: bool
    score: float
    config_path: str
    seed_path: str


def _fmt(value: float | int) -> str:
    """Return a filename-safe compact value."""
    if isinstance(value, int):
        return str(value)
    return f"{value:.3g}".replace(".", "p")


def _parse_float_list(raw: str) -> list[float]:
    """Parse comma-separated float values."""
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _parse_int_list(raw: str) -> list[int]:
    """Parse comma-separated integer values."""
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("durations must be positive")
    return values


def _zero_target() -> list[float]:
    """Return a neutral absolute joint target."""
    return [0.0 for _ in JOINT_NAMES]


def _right_preload(preload_amplitude: float) -> list[float]:
    """Return a right-swing preload target that shifts load left."""
    target = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    target[RIGHT_HIP_ROLL] = preload_amplitude
    return target.tolist()


def _left_preload(preload_amplitude: float) -> list[float]:
    """Return a left-swing preload target that shifts load right."""
    target = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    target[LEFT_HIP_ROLL] = -preload_amplitude
    return target.tolist()


def _right_lift(preload_amplitude: float, lift_amplitude: float) -> list[float]:
    """Return right micro-lift target after preload."""
    target = np.asarray(_right_preload(preload_amplitude), dtype=np.float64)
    target[[RIGHT_HIP_PITCH, RIGHT_KNEE, RIGHT_ANKLE]] = BASE_LIFT_VECTOR * lift_amplitude
    return target.tolist()


def _left_lift(preload_amplitude: float, lift_amplitude: float) -> list[float]:
    """Return left micro-lift target after preload."""
    target = np.asarray(_left_preload(preload_amplitude), dtype=np.float64)
    target[[LEFT_HIP_PITCH, LEFT_KNEE, LEFT_ANKLE]] = BASE_LIFT_VECTOR * lift_amplitude
    return target.tolist()


def _keyframe(name: str, support: str, targets: list[float], duration: int) -> dict[str, Any]:
    """Build one seed keyframe."""
    return {
        "name": name,
        "support_mode": support,
        "joint_targets": targets,
        "duration_steps": duration,
    }


def build_seed(candidate: BlueStepCandidate, neutral_duration: int, preload_duration: int) -> dict[str, Any]:
    """Build a new Blue-step v1 seed, independent of pose_1..4."""
    return {
        "schema": "sedon_gait_seed.v1",
        "target_type": "absolute",
        "description": "Generated Blue-like visible stepping v1 reference.",
        "joint_names": JOINT_NAMES,
        "keyframes": [
            _keyframe("neutral", "double", _zero_target(), neutral_duration),
            _keyframe(
                "right_preload",
                "left",
                _right_preload(candidate.preload_amplitude),
                preload_duration,
            ),
            _keyframe(
                "right_micro_lift",
                "left",
                _right_lift(candidate.preload_amplitude, candidate.lift_amplitude),
                candidate.lift_duration,
            ),
            _keyframe(
                "right_soft_land",
                "double",
                _right_preload(candidate.preload_amplitude * 0.35),
                candidate.landing_duration,
            ),
            _keyframe("neutral_after_right", "double", _zero_target(), neutral_duration),
            _keyframe(
                "left_preload",
                "right",
                _left_preload(candidate.preload_amplitude),
                preload_duration,
            ),
            _keyframe(
                "left_micro_lift",
                "right",
                _left_lift(candidate.preload_amplitude, candidate.lift_amplitude),
                candidate.lift_duration,
            ),
            _keyframe(
                "left_soft_land",
                "double",
                _left_preload(candidate.preload_amplitude * 0.35),
                candidate.landing_duration,
            ),
        ],
    }


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a formatted JSON object."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_candidate_files(
    candidate: BlueStepCandidate,
    *,
    base_config_path: Path,
    out_dir: Path,
    neutral_duration: int,
    preload_duration: int,
) -> tuple[Path, Path]:
    """Write generated seed and config for one candidate."""
    seed = build_seed(candidate, neutral_duration, preload_duration)
    seed_path = out_dir / "seeds" / f"{candidate.candidate_id}.json"
    config_path = out_dir / "configs" / f"{candidate.candidate_id}.json"
    config = _load_json(base_config_path)
    config["reference_gait_seed_path"] = str(seed_path.relative_to(REPO_ROOT)).replace("\\", "/")
    config["reference_gait_seed_scale"] = 1.0
    _write_json(seed_path, seed)
    _write_json(config_path, config)
    return config_path, seed_path


def audit_candidate(
    candidate: BlueStepCandidate,
    *,
    config_path: Path,
    seed_path: Path,
    baseline_impact: float,
    steps: int,
    seed: int,
    warmup_steps: int,
) -> BlueStepAudit:
    """Run dynamic teacher PD tracking audit for one generated reference."""
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative.")
    config = _load_config(config_path)
    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=config)
    total_weight = float(np.sum(env.model.body_mass) * 9.81)
    infos: list[dict[str, Any]] = []
    clearance_samples: list[float] = []
    negative_clearance_samples: list[bool] = []
    try:
        obs, _ = env.reset(seed=seed)
        del obs
        for _ in range(steps):
            _, _, terminated, truncated, info = env.step(
                np.zeros(env.action_space.shape, dtype=np.float64)
            )
            info = dict(info)
            infos.append(info)
            support_z, swing_z = env._task_foot_bottom_heights(env._task_phase_metadata())
            signed_clearance = float(swing_z - support_z)
            clearance_samples.append(max(0.0, signed_clearance))
            negative_clearance_samples.append(signed_clearance < 0.0)
            if terminated or truncated:
                break
    finally:
        env.close()

    if not infos:
        return BlueStepAudit(
            candidate_id=candidate.candidate_id,
            preload_amplitude=candidate.preload_amplitude,
            lift_amplitude=candidate.lift_amplitude,
            lift_duration=candidate.lift_duration,
            landing_duration=candidate.landing_duration,
            steps=0,
            max_clearance=0.0,
            both_contact_ratio=0.0,
            single_contact_ratio=0.0,
            contact_none_ratio=1.0,
            jump_count=1,
            peak_support_ratio=0.0,
            landing_impact_raw=float("inf"),
            landing_impact_post_warmup=float("inf"),
            base_height_drop_raw=float("inf"),
            base_height_drop_post_warmup=float("inf"),
            upright=0.0,
            tracking_error=float("inf"),
            max_tracking_error=float("inf"),
            negative_clearance_ratio=1.0,
            passed=False,
            score=-float("inf"),
            config_path=str(config_path),
            seed_path=str(seed_path),
        )

    post_infos = infos[warmup_steps:] or infos
    both_steps = 0
    single_steps = 0
    none_steps = 0
    for info in infos:
        left = bool(info["left_contact"])
        right = bool(info["right_contact"])
        if left and right:
            both_steps += 1
        elif left or right:
            single_steps += 1
        else:
            none_steps += 1
    contact_forces_raw = [
        float(info["left_normal_force"]) + float(info["right_normal_force"])
        for info in infos
    ]
    contact_forces_post = [
        float(info["left_normal_force"]) + float(info["right_normal_force"])
        for info in post_infos
    ]
    initial_base_height = float(infos[0]["base_height"])
    post_initial_base_height = float(post_infos[0]["base_height"])
    tracking_errors = [
        float(np.sqrt(float(info.get("joint_position_error_l2", 0.0)) / 10.0))
        for info in infos
    ]
    max_clearance = max(clearance_samples, default=0.0)
    landing_impact_post = max(contact_forces_post, default=0.0) / max(total_weight, 1e-6)
    base_drop_post = max(
        0.0,
        post_initial_base_height - min(float(info["base_height"]) for info in post_infos),
    )
    min_upright = min(float(info["upright"]) for info in infos)
    jump_count = _count_contact_none_bursts(infos)
    negative_clearance_ratio = float(np.mean(negative_clearance_samples))
    passed = (
        none_steps == 0
        and jump_count == 0
        and min_upright >= 0.99
        and max_clearance >= 0.0015
        and landing_impact_post <= baseline_impact * 1.2
        and base_drop_post <= 0.015
    )
    score = (
        max_clearance * 1000.0
        + single_steps / max(1, len(infos)) * 0.25
        - max(0.0, landing_impact_post - baseline_impact) * 2.0
        - max(0.0, base_drop_post - 0.010) * 30.0
        - negative_clearance_ratio * 0.35
        - none_steps * 10.0
    )
    total_steps = max(1, len(infos))
    return BlueStepAudit(
        candidate_id=candidate.candidate_id,
        preload_amplitude=candidate.preload_amplitude,
        lift_amplitude=candidate.lift_amplitude,
        lift_duration=candidate.lift_duration,
        landing_duration=candidate.landing_duration,
        steps=len(infos),
        max_clearance=max_clearance,
        both_contact_ratio=both_steps / total_steps,
        single_contact_ratio=single_steps / total_steps,
        contact_none_ratio=none_steps / total_steps,
        jump_count=jump_count,
        peak_support_ratio=max(float(info["force_ratio"]) for info in infos),
        landing_impact_raw=max(contact_forces_raw, default=0.0) / max(total_weight, 1e-6),
        landing_impact_post_warmup=landing_impact_post,
        base_height_drop_raw=max(
            0.0,
            initial_base_height - min(float(info["base_height"]) for info in infos),
        ),
        base_height_drop_post_warmup=base_drop_post,
        upright=min_upright,
        tracking_error=float(np.mean(tracking_errors)),
        max_tracking_error=max(tracking_errors, default=float("inf")),
        negative_clearance_ratio=negative_clearance_ratio,
        passed=passed,
        score=score,
        config_path=str(config_path),
        seed_path=str(seed_path),
    )


def build_candidates(
    preload_amplitudes: list[float],
    lift_amplitudes: list[float],
    lift_durations: list[int],
    landing_durations: list[int],
) -> list[BlueStepCandidate]:
    """Build all Blue-step candidate settings."""
    candidates: list[BlueStepCandidate] = []
    for preload, lift, lift_duration, landing_duration in product(
        preload_amplitudes,
        lift_amplitudes,
        lift_durations,
        landing_durations,
    ):
        candidate_id = (
            f"pre{_fmt(preload)}_lift{_fmt(lift)}"
            f"_ld{lift_duration}_land{landing_duration}"
        )
        candidates.append(
            BlueStepCandidate(candidate_id, preload, lift, lift_duration, landing_duration)
        )
    return candidates


def write_results(path: Path, rows: list[BlueStepAudit]) -> None:
    """Write audit rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def render_reference(result: BlueStepAudit, *, steps: int, seed: int, fps: int, width: int, height: int, out_dir: Path) -> Path:
    """Render one generated reference candidate to MP4."""
    import mujoco

    config = _load_config(Path(result.config_path))
    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=config)
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
    output_path = out_dir / "render" / f"{result.candidate_id}.mp4"
    _save_mp4(output_path, frames, fps)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-warmup-steps", type=int, default=20)
    parser.add_argument("--neutral-duration", type=int, default=30)
    parser.add_argument("--preload-duration", type=int, default=45)
    parser.add_argument("--preload-amplitudes", type=_parse_float_list, default="0.02,0.04,0.06")
    parser.add_argument("--lift-amplitudes", type=_parse_float_list, default="0.5,0.75,1.0,1.25")
    parser.add_argument("--lift-durations", type=_parse_int_list, default="30,45,60")
    parser.add_argument("--landing-durations", type=_parse_int_list, default="30,45,60")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--render-top-k", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=368)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build, sweep, and render Blue-step reference candidates."""
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
    results: list[BlueStepAudit] = []
    for candidate in build_candidates(
        args.preload_amplitudes,
        args.lift_amplitudes,
        args.lift_durations,
        args.landing_durations,
    ):
        config_path, seed_path = write_candidate_files(
            candidate,
            base_config_path=args.base_config,
            out_dir=args.out_dir,
            neutral_duration=args.neutral_duration,
            preload_duration=args.preload_duration,
        )
        result = audit_candidate(
            candidate,
            config_path=config_path,
            seed_path=seed_path,
            baseline_impact=baseline.landing_impact_post_warmup,
            steps=args.steps,
            seed=args.seed,
            warmup_steps=args.audit_warmup_steps,
        )
        results.append(result)
        print(
            f"{result.candidate_id}: pass={result.passed} "
            f"clear={result.max_clearance:.4f} both={result.both_contact_ratio:.3f} "
            f"single={result.single_contact_ratio:.3f} none={result.contact_none_ratio:.3f} "
            f"impact={result.landing_impact_post_warmup:.3f} "
            f"drop={result.base_height_drop_post_warmup:.4f} "
            f"upright={result.upright:.3f}"
        )
    ranked = sorted(results, key=lambda row: (row.passed, row.score), reverse=True)
    write_results(args.out_dir / "blue_step_reference_v1.csv", results)
    write_results(args.out_dir / "blue_step_reference_v1_top10.csv", ranked[: args.top_k])
    for row in ranked[: args.top_k]:
        print(
            "TOP "
            f"{row.candidate_id}: pass={row.passed} score={row.score:.3f} "
            f"pre={row.preload_amplitude} lift={row.lift_amplitude} "
            f"lift_dur={row.lift_duration} land_dur={row.landing_duration} "
            f"clear={row.max_clearance:.4f} impact={row.landing_impact_post_warmup:.3f} "
            f"drop={row.base_height_drop_post_warmup:.4f}"
        )
    for row in ranked[: args.render_top_k]:
        path = render_reference(
            row,
            steps=args.steps,
            seed=args.seed,
            fps=args.fps,
            width=args.width,
            height=args.height,
            out_dir=args.out_dir,
        )
        print(f"rendered: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
