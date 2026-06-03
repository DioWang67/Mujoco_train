"""Sweep Seedon preload hip-roll targets for Blue-like in-place load ratios."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from seedon_baseline.env import JOINT_NAMES, SeedonStandingEnv, load_seedon_config_from_env
from tools.seedon_debug_common import (
    DEBUG_OUT_DIR,
    DEFAULT_SCENE_PATH,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    require_scene,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "seedon_preload_sweep.csv"
HIP_ROLL_INDEX = {"right": JOINT_NAMES.index("R_joint_hip_roll"), "left": JOINT_NAMES.index("L_joint_hip_roll")}
TARGET_RATIO_MIN = 0.58
TARGET_RATIO_MAX = 0.65
UPRIGHT_MIN = 0.98
MAX_PENETRATION_M = 0.0015
TOTAL_FORCE_NORM_MIN = 0.8
TOTAL_FORCE_NORM_MAX = 1.2
BASE_HEIGHT_DROP_MAX = 0.015
CONTACT_FORCE_THRESHOLD_N = 5.0
MOVING_AVERAGE_FRAMES = 30


@dataclass(frozen=True)
class PreloadSweepRow:
    """One preload sweep candidate result."""

    side: str
    hip_roll: float
    lean_roll: float
    force_ratio_left: float
    force_ratio_right: float
    base_height: float
    upright: float
    max_penetration: float
    total_force_normalized: float
    support_side_guess: str
    passed: bool
    base_height_drop: float
    score: float


@dataclass(frozen=True)
class FrameContactMetrics:
    """Contact metrics collected for one simulation frame."""

    left_world_z: float
    right_world_z: float
    max_penetration: float
    base_height: float
    upright: float


def _support_side_guess(left_force: float, right_force: float) -> str:
    """Return coarse support side label from left/right vertical contact force."""
    left_contact = left_force > CONTACT_FORCE_THRESHOLD_N
    right_contact = right_force > CONTACT_FORCE_THRESHOLD_N
    if left_contact and right_contact:
        return "double"
    if left_contact:
        return "left"
    if right_contact:
        return "right"
    return "none"


def _foot_world_z_forces(env: SeedonStandingEnv) -> tuple[float, float, float]:
    """Return left/right foot world-z force sums and max foot-floor penetration."""
    left_world_z = 0.0
    right_world_z = 0.0
    max_penetration = 0.0
    wrench = np.zeros(6, dtype=np.float64)
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        geom1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        geom2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        pair = {geom1, geom2}
        if pair not in ({FLOOR_GEOM, LEFT_FOOT_GEOM}, {FLOOR_GEOM, RIGHT_FOOT_GEOM}):
            continue
        mujoco.mj_contactForce(env.model, env.data, contact_index, wrench)
        contact_frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        world_force = contact_frame.T @ wrench[:3]
        world_z = abs(float(world_force[2]))
        max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
        if pair == {FLOOR_GEOM, LEFT_FOOT_GEOM}:
            left_world_z += world_z
        else:
            right_world_z += world_z
    return left_world_z, right_world_z, max_penetration


def _build_target(env: SeedonStandingEnv, side: str, hip_roll: float, lean_roll: float) -> np.ndarray:
    """Return nominal joint target with a small side-specific preload offset."""
    target = env._nominal_joint_qpos.copy()
    if side == "left":
        target[HIP_ROLL_INDEX["left"]] += hip_roll
        target[HIP_ROLL_INDEX["right"]] += lean_roll
        return target
    if side == "right":
        target[HIP_ROLL_INDEX["right"]] += hip_roll
        target[HIP_ROLL_INDEX["left"]] -= lean_roll
        return target
    raise ValueError(f"Unsupported side: {side}")


def _collect_frame_metrics(env: SeedonStandingEnv) -> FrameContactMetrics:
    """Collect one frame of load-transfer metrics."""
    left_world_z, right_world_z, max_penetration = _foot_world_z_forces(env)
    return FrameContactMetrics(
        left_world_z=left_world_z,
        right_world_z=right_world_z,
        max_penetration=max_penetration,
        base_height=float(env.data.xpos[env._base_body_id][2]),
        upright=float(env._base_upright()),
    )


def _mean(values: list[float]) -> float:
    """Return a safe mean for a non-empty metric list."""
    return float(np.mean(values)) if values else 0.0


def _score(row: PreloadSweepRow) -> float:
    """Return lower-is-better ranking score around the target preload band."""
    target_ratio = row.force_ratio_left if row.side == "left" else row.force_ratio_right
    target_mid = 0.5 * (TARGET_RATIO_MIN + TARGET_RATIO_MAX)
    ratio_penalty = abs(target_ratio - target_mid)
    upright_penalty = max(0.0, UPRIGHT_MIN - row.upright) * 5.0
    penetration_penalty = max(0.0, row.max_penetration - MAX_PENETRATION_M) * 100.0
    force_penalty = max(0.0, TOTAL_FORCE_NORM_MIN - row.total_force_normalized)
    force_penalty += max(0.0, row.total_force_normalized - TOTAL_FORCE_NORM_MAX)
    height_penalty = max(0.0, row.base_height_drop - BASE_HEIGHT_DROP_MAX) * 10.0
    return float(ratio_penalty + upright_penalty + penetration_penalty + force_penalty + height_penalty)


def _evaluate_candidate(
    env: SeedonStandingEnv,
    *,
    side: str,
    hip_roll: float,
    lean_roll: float,
    settle_steps: int,
) -> PreloadSweepRow:
    """Run one dynamic preload candidate and return its final-window metrics."""
    env.reset(seed=0)
    initial_base_height = float(env.data.xpos[env._base_body_id][2])
    target = _build_target(env, side, hip_roll, lean_roll)
    frame_metrics: list[FrameContactMetrics] = []
    for _ in range(settle_steps):
        env._do_pd_simulation(target)
        frame_metrics.append(_collect_frame_metrics(env))

    tail = frame_metrics[-MOVING_AVERAGE_FRAMES:]
    left_world_z = _mean([item.left_world_z for item in tail])
    right_world_z = _mean([item.right_world_z for item in tail])
    total_world_z = left_world_z + right_world_z
    force_ratio_left = float(left_world_z / (total_world_z + 1e-6))
    force_ratio_right = float(right_world_z / (total_world_z + 1e-6))
    total_robot_weight = float(np.sum(env.model.body_mass) * 9.81)
    base_height = _mean([item.base_height for item in tail])
    upright = _mean([item.upright for item in tail])
    max_penetration = max((item.max_penetration for item in tail), default=0.0)
    total_force_normalized = float(total_world_z / max(total_robot_weight, 1e-6))
    target_ratio = force_ratio_left if side == "left" else force_ratio_right
    base_height_drop = max(0.0, initial_base_height - base_height)
    passed = (
        TARGET_RATIO_MIN <= target_ratio <= TARGET_RATIO_MAX
        and upright >= UPRIGHT_MIN
        and max_penetration <= MAX_PENETRATION_M
        and TOTAL_FORCE_NORM_MIN <= total_force_normalized <= TOTAL_FORCE_NORM_MAX
        and base_height_drop <= BASE_HEIGHT_DROP_MAX
    )
    row = PreloadSweepRow(
        side=side,
        hip_roll=float(hip_roll),
        lean_roll=float(lean_roll),
        force_ratio_left=force_ratio_left,
        force_ratio_right=force_ratio_right,
        base_height=float(base_height),
        upright=float(upright),
        max_penetration=float(max_penetration),
        total_force_normalized=total_force_normalized,
        support_side_guess=_support_side_guess(left_world_z, right_world_z),
        passed=bool(passed),
        base_height_drop=float(base_height_drop),
        score=0.0,
    )
    return PreloadSweepRow(**{**asdict(row), "score": _score(row)})


def _write_csv(path: Path, rows: list[PreloadSweepRow]) -> None:
    """Write sweep rows to CSV."""
    fieldnames = [
        "side",
        "hip_roll",
        "lean_roll",
        "force_ratio_left",
        "force_ratio_right",
        "base_height",
        "upright",
        "max_penetration",
        "total_force_normalized",
        "support_side_guess",
        "pass",
        "base_height_drop",
        "score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = asdict(row)
            payload["pass"] = payload.pop("passed")
            writer.writerow(payload)


def _top_candidates(rows: list[PreloadSweepRow], top_k: int) -> list[PreloadSweepRow]:
    """Return top candidates, preferring pass rows and then score."""
    return sorted(rows, key=lambda row: (not row.passed, row.score))[:top_k]


def run_sweep(scene_path: Path, out_csv: Path, settle_steps: int, top_k: int) -> list[PreloadSweepRow]:
    """Run the preload sweep and write CSV results."""
    if settle_steps < MOVING_AVERAGE_FRAMES:
        raise ValueError(f"settle_steps must be at least {MOVING_AVERAGE_FRAMES}.")
    env = SeedonStandingEnv(
        scene_path=require_scene(scene_path),
        reset_noise_scale=0.0,
        reward_config=load_seedon_config_from_env(),
    )
    rows: list[PreloadSweepRow] = []
    left_hip_rolls = [-0.001 * index for index in range(16)]
    right_hip_rolls = [0.001 * index for index in range(16)]
    lean_rolls = [0.005 * index for index in range(5)]
    for hip_roll in left_hip_rolls:
        for lean_roll in lean_rolls:
            rows.append(
                _evaluate_candidate(
                    env,
                    side="left",
                    hip_roll=hip_roll,
                    lean_roll=lean_roll,
                    settle_steps=settle_steps,
                )
            )
    for hip_roll in right_hip_rolls:
        for lean_roll in lean_rolls:
            rows.append(
                _evaluate_candidate(
                    env,
                    side="right",
                    hip_roll=hip_roll,
                    lean_roll=lean_roll,
                    settle_steps=settle_steps,
                )
            )
    _write_csv(out_csv, rows)
    top_rows = _top_candidates(rows, top_k)
    print(f"wrote {len(rows)} rows to {out_csv}")
    print(f"top {len(top_rows)} candidates")
    for rank, row in enumerate(top_rows, start=1):
        target_ratio = row.force_ratio_left if row.side == "left" else row.force_ratio_right
        status = "PASS" if row.passed else "fail"
        print(
            f"{rank:>2}. {status} side={row.side:<5} hip_roll={row.hip_roll:+.4f} "
            f"lean_roll={row.lean_roll:+.4f} target_ratio={target_ratio:.3f} "
            f"L={row.force_ratio_left:.3f} R={row.force_ratio_right:.3f} "
            f"upright={row.upright:.3f} pen={row.max_penetration * 1000.0:.2f}mm "
            f"force_norm={row.total_force_normalized:.3f} base_drop={row.base_height_drop:.4f} "
            f"score={row.score:.4f}"
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the Seedon preload sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--settle-steps", type=int, choices=(180, 240), default=240)
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the preload sweep CLI."""
    args = build_parser().parse_args(argv)
    run_sweep(
        scene_path=args.scene,
        out_csv=args.out_csv,
        settle_steps=args.settle_steps,
        top_k=args.top_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
