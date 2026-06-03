"""Kinematic finite-difference diagnostic for swing-foot vertical authority."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from seedon_baseline.env import SeedonStandingEnv
from tools.audit_seedon_shuffle_v0 import _load_config
from tools.blue_unload_mechanism_search import (
    DEFAULT_BASE_CONFIG,
    JOINT_NAMES,
    L_ANKLE,
    L_HIP_PITCH,
    L_HIP_ROLL,
    L_KNEE,
    R_ANKLE,
    R_HIP_PITCH,
    R_HIP_ROLL,
    R_KNEE,
    REPO_ROOT,
    UnloadCandidate,
    _left_unload_target,
    _right_unload_target,
)


DEFAULT_SOURCE_TOP = (
    REPO_ROOT
    / "artifacts"
    / "seedon_debug"
    / "blue_unload_refine_v2"
    / "blue_unload_refine_v2_top20.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "seedon_debug" / "kinematic_foot_jacobian_diagnostic_v1"
R_HIP_YAW = 0
L_HIP_YAW = 5


@dataclass(frozen=True)
class DiagnosticRow:
    """One kinematic finite-difference result."""

    source_candidate_id: str
    side: str
    test_name: str
    eps: float
    joint_delta_l2: float
    foot_z_delta: float
    foot_xy_delta: float
    dz_per_rad: float
    lateral_drift_per_dz: float
    com_delta: float
    com_lateral_delta: float
    posture_sensitivity_proxy: float
    joint_range_margin: float
    foot_penetration_change: float
    score: float
    recommended: bool


def _parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one float")
    return values


def _read_source_rows(path: Path, limit: int) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source top candidates CSV: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    filtered = [row for row in rows if row.get("a_passed", "True") == "True"]
    if not filtered:
        raise ValueError(f"No A=True candidates found in {path}")
    return filtered[:limit]


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


def _set_joint_positions(env: SeedonStandingEnv, joint_positions: np.ndarray) -> None:
    for joint_index, value in enumerate(joint_positions):
        joint_id = env._joint_ids[joint_index]
        env.data.qpos[env.model.jnt_qposadr[joint_id]] = float(value)
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def _whole_body_com(env: SeedonStandingEnv) -> np.ndarray:
    masses = env.model.body_mass.reshape(-1, 1)
    return np.sum(env.data.xpos * masses, axis=0) / max(float(np.sum(env.model.body_mass)), 1e-9)


def _state(env: SeedonStandingEnv, side: str) -> dict[str, np.ndarray | float]:
    foot_index = 0 if side == "right" else 1
    heights = env._foot_bottom_heights()
    return {
        "foot_pos": env.data.geom_xpos[env._foot_geom_ids[foot_index]].copy(),
        "foot_bottom_z": float(heights[foot_index]),
        "min_foot_bottom_z": float(np.min(heights)),
        "com": _whole_body_com(env),
    }


def _joint_margin(env: SeedonStandingEnv, joint_positions: np.ndarray, delta: np.ndarray) -> float:
    margins: list[float] = []
    for joint_index, change in enumerate(delta):
        if abs(change) <= 0.0:
            continue
        lower, upper = env._joint_range(joint_index)
        value = joint_positions[joint_index] + change
        margins.append(min(value - lower, upper - value))
    return min(margins, default=float("inf"))


def _score(
    *,
    dz_per_rad: float,
    lateral_drift_per_dz: float,
    posture_sensitivity_proxy: float,
    joint_range_margin: float,
    foot_penetration_change: float,
) -> float:
    return (
        dz_per_rad
        - lateral_drift_per_dz * 0.25
        - posture_sensitivity_proxy * 10.0
        + min(joint_range_margin, 0.5) * 0.1
        - max(0.0, foot_penetration_change) * 20.0
    )


def _evaluate_delta(
    env: SeedonStandingEnv,
    *,
    source_candidate_id: str,
    side: str,
    test_name: str,
    base_joint_positions: np.ndarray,
    base_state: dict[str, np.ndarray | float],
    delta: np.ndarray,
    eps: float,
) -> DiagnosticRow:
    _set_joint_positions(env, base_joint_positions + delta)
    changed = _state(env, side)
    foot_delta = np.asarray(changed["foot_pos"]) - np.asarray(base_state["foot_pos"])
    com_delta_vec = np.asarray(changed["com"]) - np.asarray(base_state["com"])
    foot_z_delta = float(foot_delta[2])
    foot_xy_delta = float(np.linalg.norm(foot_delta[:2]))
    joint_delta_l2 = float(np.linalg.norm(delta))
    abs_dz = abs(foot_z_delta)
    dz_per_rad = foot_z_delta / max(joint_delta_l2, 1e-9)
    lateral_drift_per_dz = foot_xy_delta / max(abs_dz, 1e-9)
    com_delta = float(np.linalg.norm(com_delta_vec))
    com_lateral_delta = float(np.linalg.norm(com_delta_vec[:2]))
    posture_proxy = com_lateral_delta / max(abs_dz, 1e-9)
    margin = _joint_margin(env, base_joint_positions, delta)
    penetration_change = max(0.0, -float(changed["min_foot_bottom_z"])) - max(0.0, -float(base_state["min_foot_bottom_z"]))
    score = _score(
        dz_per_rad=dz_per_rad,
        lateral_drift_per_dz=lateral_drift_per_dz,
        posture_sensitivity_proxy=posture_proxy,
        joint_range_margin=margin,
        foot_penetration_change=penetration_change,
    )
    return DiagnosticRow(
        source_candidate_id=source_candidate_id,
        side=side,
        test_name=test_name,
        eps=eps,
        joint_delta_l2=joint_delta_l2,
        foot_z_delta=foot_z_delta,
        foot_xy_delta=foot_xy_delta,
        dz_per_rad=dz_per_rad,
        lateral_drift_per_dz=lateral_drift_per_dz,
        com_delta=com_delta,
        com_lateral_delta=com_lateral_delta,
        posture_sensitivity_proxy=posture_proxy,
        joint_range_margin=margin,
        foot_penetration_change=penetration_change,
        score=score,
        recommended=False,
    )


def _single_joint_tests(side: str) -> list[tuple[str, int]]:
    if side == "right":
        return [
            ("hip_yaw", R_HIP_YAW),
            ("hip_roll", R_HIP_ROLL),
            ("hip_pitch", R_HIP_PITCH),
            ("knee_pitch", R_KNEE),
            ("ankle_pitch", R_ANKLE),
        ]
    return [
        ("hip_yaw", L_HIP_YAW),
        ("hip_roll", L_HIP_ROLL),
        ("hip_pitch", L_HIP_PITCH),
        ("knee_pitch", L_KNEE),
        ("ankle_pitch", L_ANKLE),
    ]


def _combo_delta(side: str, name: str, eps: float) -> np.ndarray:
    delta = np.zeros(len(JOINT_NAMES), dtype=np.float64)
    hip = R_HIP_PITCH if side == "right" else L_HIP_PITCH
    knee = R_KNEE if side == "right" else L_KNEE
    ankle = R_ANKLE if side == "right" else L_ANKLE
    if name == "knee_only":
        delta[knee] = -eps
    elif name == "hip+knee":
        delta[hip] = 0.5 * eps
        delta[knee] = -eps
    elif name == "knee+ankle":
        delta[knee] = -eps
        delta[ankle] = 0.5 * eps
    elif name == "hip+knee+ankle":
        delta[hip] = 0.5 * eps
        delta[knee] = -eps
        delta[ankle] = 0.5 * eps
    elif name == "hip_pitch+knee_pitch_ankle_comp":
        delta[hip] = 0.5 * eps
        delta[knee] = -eps
        delta[ankle] = -0.25 * eps
    else:
        raise ValueError(f"Unsupported combo: {name}")
    return delta


def run_diagnostic(args: argparse.Namespace) -> list[DiagnosticRow]:
    rows: list[DiagnosticRow] = []
    source_rows = _read_source_rows(args.source_top, args.source_top_k)
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=_load_config(args.base_config))
    try:
        env.reset(seed=args.seed)
        default_base_qpos = env.data.qpos.copy()
        for row in source_rows:
            unload = _source_to_unload(row)
            for side in ("right", "left"):
                base_joint_positions = _right_unload_target(unload) if side == "right" else _left_unload_target(unload)
                env.data.qpos[:] = default_base_qpos
                _set_joint_positions(env, base_joint_positions)
                base_state = _state(env, side)
                for eps in args.eps:
                    for joint_name, joint_index in _single_joint_tests(side):
                        for sign in (1.0, -1.0):
                            delta = np.zeros(len(JOINT_NAMES), dtype=np.float64)
                            delta[joint_index] = sign * eps
                            rows.append(
                                _evaluate_delta(
                                    env,
                                    source_candidate_id=unload.candidate_id,
                                    side=side,
                                    test_name=f"{joint_name}_{'plus' if sign > 0 else 'minus'}",
                                    base_joint_positions=base_joint_positions,
                                    base_state=base_state,
                                    delta=delta,
                                    eps=eps,
                                )
                            )
                    for combo in args.combinations:
                        delta = _combo_delta(side, combo, eps)
                        rows.append(
                            _evaluate_delta(
                                env,
                                source_candidate_id=unload.candidate_id,
                                side=side,
                                test_name=combo,
                                base_joint_positions=base_joint_positions,
                                base_state=base_state,
                                delta=delta,
                                eps=eps,
                            )
                        )
                env.data.qpos[:] = default_base_qpos
                mujoco.mj_forward(env.model, env.data)
    finally:
        env.close()
    return rows


def write_results(path: Path, rows: list[DiagnosticRow]) -> None:
    """Write diagnostic CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ranked = rank_rows(rows)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(ranked[0]).keys()))
        writer.writeheader()
        for row in ranked:
            writer.writerow(asdict(row))


def rank_rows(rows: list[DiagnosticRow]) -> list[DiagnosticRow]:
    """Rank rows and mark the top recommendation."""
    positive = [row for row in rows if row.foot_z_delta > 0.0 and row.joint_range_margin > 0.02]
    ranked = sorted(positive or rows, key=lambda row: row.score, reverse=True)
    if not ranked:
        return []
    top = ranked[0]
    return [
        DiagnosticRow(**{**asdict(row), "recommended": row == top})
        for row in ranked
    ]


def write_summary(path: Path, rows: list[DiagnosticRow]) -> None:
    """Write markdown summary."""
    ranked = rank_rows(rows)
    top = ranked[0] if ranked else None
    verdict = "no_recommendation"
    if top and top.foot_z_delta > 1e-5 and top.lateral_drift_per_dz < 5.0 and top.joint_range_margin > 0.02:
        verdict = "mapping_candidate_found"
    elif top and top.foot_z_delta > 1e-5:
        verdict = "z_authority_with_high_sensitivity"
    lines = [
        "# Kinematic Foot Jacobian Diagnostic V1",
        "",
        f"Rows: {len(rows)}",
        f"Verdict: {verdict}",
        "",
    ]
    if top:
        lines.extend(
            [
                "## Recommended Lift Mapping",
                "",
                f"- source: {top.source_candidate_id}",
                f"- side: {top.side}",
                f"- mapping: {top.test_name}",
                f"- eps: {top.eps}",
                f"- foot_z_delta: {top.foot_z_delta:.6f}",
                f"- dz_per_rad: {top.dz_per_rad:.6f}",
                f"- lateral_drift_per_dz: {top.lateral_drift_per_dz:.3f}",
                f"- posture_sensitivity_proxy: {top.posture_sensitivity_proxy:.3f}",
                f"- joint_range_margin: {top.joint_range_margin:.3f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Top Rows",
            "",
            "| source | side | test | eps | dz_per_rad | foot_z | drift/dz | posture | margin | score |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ranked[:10]:
        lines.append(
            "| "
            f"{row.source_candidate_id} | {row.side} | {row.test_name} | {row.eps:.3f} | "
            f"{row.dz_per_rad:.6f} | {row.foot_z_delta:.6f} | "
            f"{row.lateral_drift_per_dz:.3f} | {row.posture_sensitivity_proxy:.3f} | "
            f"{row.joint_range_margin:.3f} | {row.score:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-top", type=Path, default=DEFAULT_SOURCE_TOP)
    parser.add_argument("--source-top-k", type=int, default=3)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=_parse_float_list, default="0.001,0.003,0.005")
    parser.add_argument(
        "--combinations",
        nargs="+",
        default=[
            "knee_only",
            "hip+knee",
            "knee+ankle",
            "hip+knee+ankle",
            "hip_pitch+knee_pitch_ankle_comp",
        ],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run kinematic finite-difference diagnostic."""
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = run_diagnostic(args)
    ranked = rank_rows(rows)
    write_results(args.out_dir / "kinematic_foot_jacobian_diagnostic_v1.csv", rows)
    write_results(args.out_dir / "kinematic_foot_jacobian_diagnostic_v1_top20.csv", ranked[:20])
    write_summary(args.out_dir / "summary.md", rows)
    if ranked:
        top = ranked[0]
        print(
            "recommended="
            f"{top.test_name} side={top.side} source={top.source_candidate_id} "
            f"dz_per_rad={top.dz_per_rad:.6f} drift_per_dz={top.lateral_drift_per_dz:.3f} "
            f"posture={top.posture_sensitivity_proxy:.3f} margin={top.joint_range_margin:.3f}"
        )
    print(f"rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
