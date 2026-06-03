"""Sweep Seedon reference lift variants under deterministic teacher PD tracking."""

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
DEFAULT_BASE_SEED = REPO_ROOT / "configs" / "seedon" / "reference_march_pose_1_4_mirrored_seed.json"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "seedon_debug" / "reference_lift_sweep"
SWING_JOINT_INDEXES = {
    "right": (2, 3, 4),
    "left": (7, 8, 9),
}


@dataclass(frozen=True)
class SweepCandidate:
    """One reference lift sweep setting."""

    candidate_id: str
    reference_scale: float
    lift_pose_scale: float
    duration_scale: float


@dataclass(frozen=True)
class SweepResult:
    """Aggregated deterministic teacher PD sweep result."""

    candidate_id: str
    reference_scale: float
    lift_pose_scale: float
    duration_scale: float
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


def _fmt_float(value: float) -> str:
    """Format a float for stable filenames."""
    return f"{value:.3g}".replace(".", "p")


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _infer_swing_side(name: str) -> str | None:
    """Infer swing side from a pose keyframe name."""
    lowered = name.lower()
    if "right_swing" in lowered:
        return "right"
    if "left_swing" in lowered:
        return "left"
    return None


def build_variant_files(
    candidate: SweepCandidate,
    *,
    base_config_path: Path,
    base_seed_path: Path,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Create config/seed files for one sweep candidate."""
    seed = _load_json(base_seed_path)
    keyframes = seed.get("keyframes")
    if not isinstance(keyframes, list):
        raise ValueError(f"Seed missing keyframes list: {base_seed_path}")

    for raw_keyframe in keyframes:
        if not isinstance(raw_keyframe, dict):
            continue
        raw_duration = int(raw_keyframe.get("duration_steps", 60))
        raw_keyframe["duration_steps"] = max(1, int(round(raw_duration * candidate.duration_scale)))
        name = str(raw_keyframe.get("name", ""))
        if "pose_2" not in name and "pose_3" not in name:
            continue
        swing_side = _infer_swing_side(name)
        if swing_side is None:
            continue
        targets = raw_keyframe.get("joint_targets")
        if not isinstance(targets, list):
            raise ValueError(f"Seed keyframe missing joint_targets: {name}")
        for joint_index in SWING_JOINT_INDEXES[swing_side]:
            targets[joint_index] = float(targets[joint_index]) * candidate.lift_pose_scale

    config = _load_json(base_config_path)
    seed_path = out_dir / "seeds" / f"{candidate.candidate_id}.json"
    config_path = out_dir / "configs" / f"{candidate.candidate_id}.json"
    config["reference_gait_seed_path"] = str(seed_path.relative_to(REPO_ROOT)).replace("\\", "/")
    config["reference_gait_seed_scale"] = candidate.reference_scale
    _write_json(seed_path, seed)
    _write_json(config_path, config)
    return config_path, seed_path


def audit_reference_candidate(
    candidate: SweepCandidate,
    *,
    config_path: Path,
    seed_path: Path,
    baseline_impact_post_warmup: float,
    steps: int,
    seed: int,
    audit_warmup_steps: int,
) -> SweepResult:
    """Run deterministic teacher PD tracking and aggregate sweep metrics."""
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if audit_warmup_steps < 0:
        raise ValueError("audit_warmup_steps must be non-negative.")

    config = _load_config(config_path)
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=config)
    infos: list[dict[str, Any]] = []
    total_weight = float(np.sum(env.model.body_mass) * 9.81)
    try:
        obs, _ = env.reset(seed=seed)
        del obs
        for _ in range(steps):
            action = np.zeros(env.action_space.shape, dtype=np.float64)
            _, _, terminated, truncated, info = env.step(action)
            infos.append(dict(info))
            if terminated or truncated:
                break
    finally:
        env.close()

    if not infos:
        return SweepResult(
            candidate_id=candidate.candidate_id,
            reference_scale=candidate.reference_scale,
            lift_pose_scale=candidate.lift_pose_scale,
            duration_scale=candidate.duration_scale,
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
            seed_path="",
        )

    post_infos = infos[audit_warmup_steps:] or infos
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
        float(
            np.sqrt(
                float(info.get("joint_position_error_l2", 0.0))
                / max(int(len(SWING_JOINT_INDEXES) * 5), 1)
            )
        )
        for info in infos
    ]
    max_clearance = max(float(info["foot_clearance"]) for info in infos)
    landing_impact_post = max(contact_forces_post, default=0.0) / max(total_weight, 1e-6)
    base_height_drop_post = max(
        0.0,
        post_initial_base_height - min(float(info["base_height"]) for info in post_infos),
    )
    upright = min(float(info["upright"]) for info in infos)
    jump_count = _count_contact_none_bursts(infos)
    negative_clearance_ratio = float(
        np.mean([float(info["foot_clearance"]) < 0.0 for info in infos])
    )
    passed = (
        none_steps == 0
        and jump_count == 0
        and upright >= 0.99
        and max_clearance >= 0.003
        and landing_impact_post <= baseline_impact_post_warmup * 1.15
        and base_height_drop_post <= 0.015
    )
    score = (
        max_clearance * 1000.0
        - max(0.0, landing_impact_post - baseline_impact_post_warmup) * 2.0
        - max(0.0, base_height_drop_post - 0.008) * 30.0
        - negative_clearance_ratio * 0.5
        - none_steps * 10.0
    )
    total_steps = max(1, len(infos))
    return SweepResult(
        candidate_id=candidate.candidate_id,
        reference_scale=candidate.reference_scale,
        lift_pose_scale=candidate.lift_pose_scale,
        duration_scale=candidate.duration_scale,
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
        base_height_drop_post_warmup=base_height_drop_post,
        upright=upright,
        tracking_error=float(np.mean(tracking_errors)) if tracking_errors else float("inf"),
        max_tracking_error=max(tracking_errors, default=float("inf")),
        negative_clearance_ratio=negative_clearance_ratio,
        passed=passed,
        score=score,
        config_path=str(config_path),
        seed_path=str(seed_path),
    )


def _baseline_landing_impact(config_path: Path, *, steps: int, seed: int, warmup: int) -> float:
    """Return baseline deterministic teacher post-warmup landing impact."""
    result = audit_reference_candidate(
        SweepCandidate("baseline", 0.65, 1.0, 1.0),
        config_path=config_path,
        seed_path=DEFAULT_BASE_SEED,
        baseline_impact_post_warmup=float("inf"),
        steps=steps,
        seed=seed,
        audit_warmup_steps=warmup,
    )
    return result.landing_impact_post_warmup


def write_results(path: Path, rows: list[SweepResult]) -> None:
    """Write sweep rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _render_reference_video(
    result: SweepResult,
    *,
    steps: int,
    seed: int,
    fps: int,
    out_dir: Path,
    width: int,
    height: int,
) -> Path:
    """Render one reference candidate to MP4."""
    import mujoco

    config = _load_config(Path(result.config_path))
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=config)
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


def build_candidates(
    reference_scales: list[float],
    lift_pose_scales: list[float],
    duration_scales: list[float],
) -> list[SweepCandidate]:
    """Build all sweep candidates."""
    candidates = []
    for reference_scale, lift_pose_scale, duration_scale in product(
        reference_scales,
        lift_pose_scales,
        duration_scales,
    ):
        candidate_id = (
            f"ref{_fmt_float(reference_scale)}"
            f"_lift{_fmt_float(lift_pose_scale)}"
            f"_dur{_fmt_float(duration_scale)}"
        )
        candidates.append(
            SweepCandidate(candidate_id, reference_scale, lift_pose_scale, duration_scale)
        )
    return candidates


def _parse_float_list(raw: str) -> list[float]:
    """Parse comma-separated floats."""
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--base-seed", type=Path, default=DEFAULT_BASE_SEED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-warmup-steps", type=int, default=20)
    parser.add_argument("--reference-scales", type=_parse_float_list, default="0.5,0.6,0.7,0.8,1.0")
    parser.add_argument("--lift-pose-scales", type=_parse_float_list, default="1.0,1.2,1.5,2.0")
    parser.add_argument("--duration-scales", type=_parse_float_list, default="1.0,1.2,1.5")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--render-top-k", type=int, default=3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=368)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the reference lift sweep and render the best candidates."""
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline_impact = _baseline_landing_impact(
        args.base_config,
        steps=args.steps,
        seed=args.seed,
        warmup=args.audit_warmup_steps,
    )
    results: list[SweepResult] = []
    for candidate in build_candidates(
        args.reference_scales,
        args.lift_pose_scales,
        args.duration_scales,
    ):
        config_path, seed_path = build_variant_files(
            candidate,
            base_config_path=args.base_config,
            base_seed_path=args.base_seed,
            out_dir=args.out_dir,
        )
        result = audit_reference_candidate(
            candidate,
            config_path=config_path,
            seed_path=seed_path,
            baseline_impact_post_warmup=baseline_impact,
            steps=args.steps,
            seed=args.seed,
            audit_warmup_steps=args.audit_warmup_steps,
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
    write_results(args.out_dir / "reference_lift_sweep.csv", results)
    write_results(args.out_dir / "reference_lift_sweep_top10.csv", ranked[: args.top_k])
    for result in ranked[: args.top_k]:
        print(
            "TOP "
            f"{result.candidate_id}: pass={result.passed} score={result.score:.3f} "
            f"ref={result.reference_scale} lift={result.lift_pose_scale} dur={result.duration_scale} "
            f"clear={result.max_clearance:.4f} impact={result.landing_impact_post_warmup:.3f} "
            f"drop={result.base_height_drop_post_warmup:.4f}"
        )
    for result in ranked[: args.render_top_k]:
        path = _render_reference_video(
            result,
            steps=args.steps,
            seed=args.seed,
            fps=args.fps,
            out_dir=args.out_dir,
            width=args.width,
            height=args.height,
        )
        print(f"rendered: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
