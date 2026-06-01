"""Compare two Sedon teacher-imitation checkpoints around impact spikes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sedon_baseline.env import SedonStandingConfig, SedonStandingEnv
from tools.audit_sedon_shuffle_v0 import _count_contact_none_bursts, _load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "sedon" / "reference_teacher_pose_1_4_imitation.json"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "sedon_debug"


@dataclass(frozen=True)
class CheckpointSpec:
    """One checkpoint plus its VecNormalize stats."""

    label: str
    checkpoint: Path
    vecnorm_path: Path


@dataclass(frozen=True)
class CheckpointStep:
    """One rollout time-series row."""

    label: str
    step: int
    phase_name: str
    contact_state: str
    left_contact: bool
    right_contact: bool
    left_normal_force: float
    right_normal_force: float
    landing_impact: float
    base_height: float
    base_height_drop: float
    left_foot_bottom_z: float
    right_foot_bottom_z: float
    left_foot_velocity_z: float
    right_foot_velocity_z: float
    foot_velocity_near_contact: float
    tracking_error: float
    tracking_error_variance: float
    contact_transition: float
    peak_support_ratio: float
    clearance: float


@dataclass(frozen=True)
class CheckpointSummary:
    """Aggregated comparison metrics for one checkpoint."""

    label: str
    steps: int
    audit_warmup_steps: int
    terminated: bool
    termination_reason: str
    contact_none_ratio: float
    jump_count: int
    peak_support_ratio: float
    clearance: float
    base_height_drop_raw: float
    base_height_drop_post_warmup: float
    base_height_drop: float
    landing_impact_raw: float
    landing_impact_raw_step: int
    landing_impact_post_warmup: float
    landing_impact_post_warmup_step: int
    landing_impact: float
    landing_impact_step: int
    max_contact_force_raw: float
    max_contact_force_raw_step: int
    max_contact_force_post_warmup: float
    max_contact_force_post_warmup_step: int
    max_contact_force: float
    max_contact_force_step: int
    min_base_height: float
    max_foot_velocity_near_contact_raw: float
    max_foot_velocity_near_contact_raw_step: int
    max_foot_velocity_near_contact_post_warmup: float
    max_foot_velocity_near_contact_post_warmup_step: int
    max_foot_velocity_near_contact: float
    max_foot_velocity_near_contact_step: int
    mean_tracking_error: float
    max_tracking_error: float
    tracking_error_at_impact: float
    tracking_error_variance: float
    contact_transition_ratio: float


def _load_model(path: Path) -> Any:
    """Load a Stable-Baselines3 PPO checkpoint."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    from stable_baselines3 import PPO

    return PPO.load(str(path))


def _contact_state(left: bool, right: bool) -> str:
    """Return compact contact state."""
    if left and right:
        return "both"
    if left:
        return "left"
    if right:
        return "right"
    return "none"


def rollout_checkpoint(
    config: SedonStandingConfig,
    spec: CheckpointSpec,
    *,
    steps: int,
    seed: int,
    audit_warmup_steps: int,
) -> tuple[list[CheckpointStep], CheckpointSummary]:
    """Roll out one checkpoint and collect impact diagnostics."""
    if not spec.vecnorm_path.is_file():
        raise FileNotFoundError(f"VecNormalize not found: {spec.vecnorm_path}")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = _load_model(spec.checkpoint)

    def make_env() -> SedonStandingEnv:
        env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=config)
        env.reset(seed=seed)
        return env

    dummy_env = DummyVecEnv([make_env])
    raw_env: SedonStandingEnv = dummy_env.envs[0]
    eval_env = VecNormalize.load(str(spec.vecnorm_path), dummy_env)
    eval_env.training = False
    eval_env.norm_reward = False
    obs = eval_env.reset()
    total_weight = float(np.sum(raw_env.model.body_mass) * 9.81)
    initial_base_height = float(raw_env._base_height())
    previous_bottoms = raw_env._foot_bottom_heights()
    rows: list[CheckpointStep] = []
    infos: list[dict[str, Any]] = []
    terminated = False
    termination_reason = "none"
    try:
        for step in range(1, steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, step_infos = eval_env.step(action)
            info = dict(step_infos[0])
            infos.append(info)
            bottoms = raw_env._foot_bottom_heights()
            foot_velocity = bottoms - previous_bottoms
            previous_bottoms = bottoms.copy()
            left_contact = bool(info["left_contact"])
            right_contact = bool(info["right_contact"])
            landing_impact = (
                float(info["left_normal_force"]) + float(info["right_normal_force"])
            ) / max(total_weight, 1e-6)
            near_contact_velocity = 0.0
            if left_contact:
                near_contact_velocity = max(near_contact_velocity, abs(float(foot_velocity[0])))
            if right_contact:
                near_contact_velocity = max(near_contact_velocity, abs(float(foot_velocity[1])))
            rows.append(
                CheckpointStep(
                    label=spec.label,
                    step=step,
                    phase_name=str(info.get("phase_name", "")),
                    contact_state=_contact_state(left_contact, right_contact),
                    left_contact=left_contact,
                    right_contact=right_contact,
                    left_normal_force=float(info["left_normal_force"]),
                    right_normal_force=float(info["right_normal_force"]),
                    landing_impact=landing_impact,
                    base_height=float(info["base_height"]),
                    base_height_drop=max(0.0, initial_base_height - float(info["base_height"])),
                    left_foot_bottom_z=float(bottoms[0]),
                    right_foot_bottom_z=float(bottoms[1]),
                    left_foot_velocity_z=float(foot_velocity[0]),
                    right_foot_velocity_z=float(foot_velocity[1]),
                    foot_velocity_near_contact=near_contact_velocity,
                    tracking_error=float(info.get("tracking_error", 0.0)),
                    tracking_error_variance=float(info.get("tracking_error_variance", 0.0)),
                    contact_transition=float(info.get("contact_transition", 0.0)),
                    peak_support_ratio=float(info["force_ratio"]),
                    clearance=float(info["foot_clearance"]),
                )
            )
            termination_reason = str(info.get("termination_reason", "none"))
            if bool(dones[0]):
                terminated = True
                break
    finally:
        eval_env.close()

    summary = summarize_rows(
        spec.label,
        rows,
        infos,
        terminated,
        termination_reason,
        audit_warmup_steps=audit_warmup_steps,
    )
    return rows, summary


def summarize_rows(
    label: str,
    rows: list[CheckpointStep],
    infos: list[dict[str, Any]],
    terminated: bool,
    termination_reason: str,
    *,
    audit_warmup_steps: int,
) -> CheckpointSummary:
    """Aggregate rollout rows."""
    if audit_warmup_steps < 0:
        raise ValueError("audit_warmup_steps must be non-negative.")
    if not rows:
        return CheckpointSummary(
            label=label,
            steps=0,
            audit_warmup_steps=audit_warmup_steps,
            terminated=True,
            termination_reason="no_steps",
            contact_none_ratio=1.0,
            jump_count=1,
            peak_support_ratio=0.0,
            clearance=0.0,
            base_height_drop_raw=float("inf"),
            base_height_drop_post_warmup=float("inf"),
            base_height_drop=float("inf"),
            landing_impact_raw=float("inf"),
            landing_impact_raw_step=0,
            landing_impact_post_warmup=float("inf"),
            landing_impact_post_warmup_step=0,
            landing_impact=float("inf"),
            landing_impact_step=0,
            max_contact_force_raw=float("inf"),
            max_contact_force_raw_step=0,
            max_contact_force_post_warmup=float("inf"),
            max_contact_force_post_warmup_step=0,
            max_contact_force=float("inf"),
            max_contact_force_step=0,
            min_base_height=0.0,
            max_foot_velocity_near_contact_raw=float("inf"),
            max_foot_velocity_near_contact_raw_step=0,
            max_foot_velocity_near_contact_post_warmup=float("inf"),
            max_foot_velocity_near_contact_post_warmup_step=0,
            max_foot_velocity_near_contact=float("inf"),
            max_foot_velocity_near_contact_step=0,
            mean_tracking_error=float("inf"),
            max_tracking_error=float("inf"),
            tracking_error_at_impact=float("inf"),
            tracking_error_variance=float("inf"),
            contact_transition_ratio=1.0,
        )

    post_rows = [row for row in rows if row.step > audit_warmup_steps] or rows
    impact_row_raw = max(rows, key=lambda row: row.landing_impact)
    impact_row_post = max(post_rows, key=lambda row: row.landing_impact)
    force_row_raw = max(rows, key=lambda row: row.left_normal_force + row.right_normal_force)
    force_row_post = max(
        post_rows,
        key=lambda row: row.left_normal_force + row.right_normal_force,
    )
    velocity_row_raw = max(rows, key=lambda row: row.foot_velocity_near_contact)
    velocity_row_post = max(post_rows, key=lambda row: row.foot_velocity_near_contact)
    tracking_errors = [row.tracking_error for row in rows]
    post_initial_base_height = post_rows[0].base_height
    base_height_drop_raw = max(row.base_height_drop for row in rows)
    base_height_drop_post_warmup = max(
        0.0,
        post_initial_base_height - min(row.base_height for row in post_rows),
    )
    none_steps = sum(1 for row in rows if row.contact_state == "none")
    return CheckpointSummary(
        label=label,
        steps=len(rows),
        audit_warmup_steps=audit_warmup_steps,
        terminated=terminated,
        termination_reason=termination_reason,
        contact_none_ratio=none_steps / max(1, len(rows)),
        jump_count=_count_contact_none_bursts(infos),
        peak_support_ratio=max(row.peak_support_ratio for row in rows),
        clearance=max(row.clearance for row in rows),
        base_height_drop_raw=base_height_drop_raw,
        base_height_drop_post_warmup=base_height_drop_post_warmup,
        base_height_drop=base_height_drop_post_warmup,
        landing_impact_raw=impact_row_raw.landing_impact,
        landing_impact_raw_step=impact_row_raw.step,
        landing_impact_post_warmup=impact_row_post.landing_impact,
        landing_impact_post_warmup_step=impact_row_post.step,
        landing_impact=impact_row_post.landing_impact,
        landing_impact_step=impact_row_post.step,
        max_contact_force_raw=force_row_raw.left_normal_force + force_row_raw.right_normal_force,
        max_contact_force_raw_step=force_row_raw.step,
        max_contact_force_post_warmup=(
            force_row_post.left_normal_force + force_row_post.right_normal_force
        ),
        max_contact_force_post_warmup_step=force_row_post.step,
        max_contact_force=force_row_post.left_normal_force + force_row_post.right_normal_force,
        max_contact_force_step=force_row_post.step,
        min_base_height=min(row.base_height for row in rows),
        max_foot_velocity_near_contact_raw=velocity_row_raw.foot_velocity_near_contact,
        max_foot_velocity_near_contact_raw_step=velocity_row_raw.step,
        max_foot_velocity_near_contact_post_warmup=(
            velocity_row_post.foot_velocity_near_contact
        ),
        max_foot_velocity_near_contact_post_warmup_step=velocity_row_post.step,
        max_foot_velocity_near_contact=velocity_row_post.foot_velocity_near_contact,
        max_foot_velocity_near_contact_step=velocity_row_post.step,
        mean_tracking_error=float(np.mean(tracking_errors)),
        max_tracking_error=max(tracking_errors),
        tracking_error_at_impact=impact_row_post.tracking_error,
        tracking_error_variance=float(np.var(tracking_errors)),
        contact_transition_ratio=float(np.mean([row.contact_transition for row in rows])),
    )


def write_rows(path: Path, rows: list[CheckpointStep]) -> None:
    """Write per-step rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summaries(path: Path, summaries: list[CheckpointSummary]) -> None:
    """Write summary rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))


def _window(rows: list[CheckpointStep], center_step: int, radius: int) -> list[CheckpointStep]:
    """Return a step window around a center step."""
    return [row for row in rows if abs(row.step - center_step) <= radius]


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--audit-warmup-steps",
        type=int,
        default=20,
        help="Initial rollout steps excluded from post-warmup impact/drop metrics.",
    )
    parser.add_argument("--window-radius", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=3,
        metavar=("LABEL", "MODEL", "VECNORM"),
        required=True,
        help="Checkpoint spec. Repeat as: --checkpoint 25k model.zip vecnorm.pkl",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run checkpoint comparison."""
    args = build_parser().parse_args(argv)
    if args.audit_warmup_steps < 0:
        raise ValueError("--audit-warmup-steps must be non-negative.")
    config = _load_config(args.config)
    all_rows: list[CheckpointStep] = []
    summaries: list[CheckpointSummary] = []
    impact_windows: list[CheckpointStep] = []
    for raw_label, raw_model, raw_vecnorm in args.checkpoint:
        spec = CheckpointSpec(raw_label, Path(raw_model), Path(raw_vecnorm))
        rows, summary = rollout_checkpoint(
            config,
            spec,
            steps=args.steps,
            seed=args.seed,
            audit_warmup_steps=args.audit_warmup_steps,
        )
        all_rows.extend(rows)
        summaries.append(summary)
        impact_windows.extend(_window(rows, summary.landing_impact_raw_step, args.window_radius))
        impact_windows.extend(_window(rows, summary.landing_impact_step, args.window_radius))

    write_rows(args.out_dir / "teacher_imitation_25k_vs_50k_timeseries.csv", all_rows)
    write_rows(args.out_dir / "teacher_imitation_25k_vs_50k_impact_windows.csv", impact_windows)
    write_summaries(args.out_dir / "teacher_imitation_25k_vs_50k_summary.csv", summaries)
    for summary in summaries:
        print(
            f"{summary.label}: "
            f"impact_raw={summary.landing_impact_raw:.3f}@{summary.landing_impact_raw_step} "
            f"impact_post={summary.landing_impact_post_warmup:.3f}@{summary.landing_impact_post_warmup_step} "
            f"force_raw={summary.max_contact_force_raw:.2f}@{summary.max_contact_force_raw_step} "
            f"force_post={summary.max_contact_force_post_warmup:.2f}@{summary.max_contact_force_post_warmup_step} "
            f"base_min={summary.min_base_height:.5f} "
            f"drop_raw={summary.base_height_drop_raw:.5f} "
            f"drop_post={summary.base_height_drop_post_warmup:.5f} "
            f"foot_v_raw={summary.max_foot_velocity_near_contact_raw:.6f}@{summary.max_foot_velocity_near_contact_raw_step} "
            f"foot_v_post={summary.max_foot_velocity_near_contact_post_warmup:.6f}@{summary.max_foot_velocity_near_contact_post_warmup_step} "
            f"track={summary.mean_tracking_error:.5f}/{summary.max_tracking_error:.5f} "
            f"track_at_impact={summary.tracking_error_at_impact:.5f} "
            f"contact_transition={summary.contact_transition_ratio:.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
