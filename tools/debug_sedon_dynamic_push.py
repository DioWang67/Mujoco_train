"""Dynamic forward-push contact handoff diagnostic for Sedon foot geometry.

This Class C tool applies an initial floating-base forward velocity and records
center/toe/heel contact forces under zero action. It does not train, modify
rewards, or edit any source scene.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from tools.sedon_debug_common import DEBUG_OUT_DIR, geom_name, require_scene


DEFAULT_OUT_DIR = DEBUG_OUT_DIR / "blue_like_sole_experiments_v5"
DEFAULT_SCENES = (
    DEFAULT_OUT_DIR / "training_scene_v5_a.xml",
    DEFAULT_OUT_DIR / "training_scene_v5_b.xml",
    DEFAULT_OUT_DIR / "training_scene_v5_c.xml",
)
DEFAULT_REPORT = DEFAULT_OUT_DIR / "comparison_report_dynamic_push.md"
FOOT_FORCE_KEYS = (
    "left_center_force",
    "left_toe_force",
    "left_heel_force",
    "right_center_force",
    "right_toe_force",
    "right_heel_force",
)


@dataclass(frozen=True)
class PushRow:
    """One per-step dynamic push sample."""

    scene_name: str
    push_speed: float
    step: int
    base_z: float
    base_pitch_deg: float
    left_center_force: float
    left_toe_force: float
    left_heel_force: float
    right_center_force: float
    right_toe_force: float
    right_heel_force: float
    total_center_force: float
    total_toe_force: float
    total_heel_force: float
    toe_active: bool
    no_contact: bool


@dataclass(frozen=True)
class PushSummary:
    """Summary for one scene and one push speed."""

    scene_name: str
    push_speed: float
    first_toe_contact_step: int | None
    center_to_toe_handoff_ok: bool
    no_contact_steps: int
    min_base_z: float
    passed: bool


def _parse_float_list(raw_value: str) -> list[float]:
    """Parse comma-separated floats."""
    values = [float(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    return values


def _base_pitch_deg(data: mujoco.MjData) -> float:
    """Return base pitch angle in degrees from qpos quaternion."""
    w, x, y, z = [float(value) for value in data.qpos[3:7]]
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    return math.degrees(pitch)


def _force_bucket(geom: str) -> tuple[str, str] | None:
    """Return (side, region) for Sedon foot geoms used in Blue-like soles."""
    if geom.startswith("L_foot_collision"):
        side = "left"
    elif geom.startswith("R_foot_collision"):
        side = "right"
    else:
        return None

    if "toe_rocker" in geom:
        return side, "toe"
    if "heel_rocker" in geom:
        return side, "heel"
    if geom in {"L_foot_collision", "R_foot_collision"}:
        return side, "center"
    return None


def _contact_forces(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    """Sum floor-contact normal force by side and foot region."""
    forces = {key: 0.0 for key in FOOT_FORCE_KEYS}
    wrench = np.zeros(6, dtype=np.float64)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        name_a = geom_name(model, int(contact.geom1))
        name_b = geom_name(model, int(contact.geom2))
        if "floor" not in {name_a, name_b}:
            continue
        foot_geom = name_b if name_a == "floor" else name_a
        bucket = _force_bucket(foot_geom)
        if bucket is None:
            continue
        side, region = bucket
        mujoco.mj_contactForce(model, data, contact_index, wrench)
        forces[f"{side}_{region}_force"] += abs(float(wrench[0]))
    return forces


def _reset_and_settle(model: mujoco.MjModel, data: mujoco.MjData, settle_steps: int) -> None:
    """Reset and settle under zero controls."""
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    for _ in range(settle_steps):
        if data.ctrl.size:
            data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)


def run_dynamic_push(
    scene_path: Path,
    push_speeds: list[float],
    settle_steps: int,
    run_steps: int,
) -> list[PushRow]:
    """Run dynamic push diagnostics for one scene."""
    scene_path = require_scene(scene_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    rows: list[PushRow] = []
    scene_name = scene_path.stem

    for push_speed in push_speeds:
        _reset_and_settle(model, data, settle_steps)
        data.qvel[0] = float(push_speed)
        mujoco.mj_forward(model, data)
        for step in range(1, run_steps + 1):
            if data.ctrl.size:
                data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
            forces = _contact_forces(model, data)
            total_center = forces["left_center_force"] + forces["right_center_force"]
            total_toe = forces["left_toe_force"] + forces["right_toe_force"]
            total_heel = forces["left_heel_force"] + forces["right_heel_force"]
            total_foot = total_center + total_toe + total_heel
            rows.append(
                PushRow(
                    scene_name=scene_name,
                    push_speed=float(push_speed),
                    step=step,
                    base_z=float(data.qpos[2]),
                    base_pitch_deg=_base_pitch_deg(data),
                    left_center_force=forces["left_center_force"],
                    left_toe_force=forces["left_toe_force"],
                    left_heel_force=forces["left_heel_force"],
                    right_center_force=forces["right_center_force"],
                    right_toe_force=forces["right_toe_force"],
                    right_heel_force=forces["right_heel_force"],
                    total_center_force=total_center,
                    total_toe_force=total_toe,
                    total_heel_force=total_heel,
                    toe_active=total_toe > 0.1,
                    no_contact=total_foot < 0.1,
                )
            )
    return rows


def summarize_rows(rows: list[PushRow]) -> list[PushSummary]:
    """Summarize rows by scene and push speed."""
    groups: dict[tuple[str, float], list[PushRow]] = {}
    for row in rows:
        groups.setdefault((row.scene_name, row.push_speed), []).append(row)

    summaries: list[PushSummary] = []
    for (scene_name, push_speed), group in sorted(groups.items()):
        first_toe = next((row.step for row in group if row.toe_active), None)
        center_to_toe = False
        if first_toe is not None:
            before_toe = [row for row in group if row.step < first_toe]
            center_to_toe = bool(before_toe and before_toe[-1].total_center_force > 0.1)
        no_contact_steps = sum(1 for row in group if row.no_contact)
        min_base_z = min(row.base_z for row in group)
        passed = (
            first_toe is not None
            and center_to_toe
            and no_contact_steps <= 5
            and min_base_z > 0.14
        )
        summaries.append(
            PushSummary(
                scene_name=scene_name,
                push_speed=push_speed,
                first_toe_contact_step=first_toe,
                center_to_toe_handoff_ok=center_to_toe,
                no_contact_steps=no_contact_steps,
                min_base_z=min_base_z,
                passed=passed,
            )
        )
    return summaries


def _write_csv(path: Path, rows: list[PushRow]) -> None:
    """Write push rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([row.__dict__ for row in rows])


def _write_report(path: Path, summaries: list[PushSummary]) -> None:
    """Write markdown comparison report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sedon Dynamic Push Contact Handoff",
        "",
        "Task class: Class C geometry diagnostic. No PPO, reward, train.py, or source-scene changes.",
        "",
        "| scene | push_speed | first_toe_contact_step | center_to_toe_handoff_ok | no_contact_steps | min_base_z | PASS/FAIL |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary.scene_name} | {summary.push_speed:.2f} | "
            f"{_fmt_optional_int(summary.first_toe_contact_step)} | "
            f"{summary.center_to_toe_handoff_ok} | {summary.no_contact_steps} | "
            f"{summary.min_base_z:.4f} | {'PASS' if summary.passed else 'FAIL'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_optional_int(value: int | None) -> str:
    """Format optional int for markdown."""
    if value is None:
        return "n/a"
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-path",
        type=Path,
        action="append",
        default=None,
        help="Scene XML path. Repeat to compare multiple scenes.",
    )
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_DIR / "dynamic_push.csv")
    parser.add_argument("--report-md", type=Path, default=DEFAULT_OUT_DIR / "comparison_report_dynamic_push.md")
    parser.add_argument("--push-speeds", type=_parse_float_list, default=_parse_float_list("0.1,0.2,0.3"))
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--run-steps", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run dynamic push diagnostics."""
    args = build_parser().parse_args(argv)
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative.")
    if args.run_steps <= 0:
        raise ValueError("--run-steps must be positive.")
    scene_paths = args.scene_path or list(DEFAULT_SCENES)
    rows: list[PushRow] = []
    for scene_path in scene_paths:
        rows.extend(
            run_dynamic_push(
                scene_path,
                push_speeds=args.push_speeds,
                settle_steps=args.settle_steps,
                run_steps=args.run_steps,
            )
        )
    _write_csv(args.out_csv, rows)
    _write_report(args.report_md, summarize_rows(rows))
    print(f"csv: {args.out_csv}")
    print(f"report: {args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
