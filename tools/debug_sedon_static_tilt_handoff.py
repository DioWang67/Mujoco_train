"""Static pitch-tilt contact handoff diagnostic for Sedon foot geometry.

This Class C tool inspects contact-force handoff across center/toe/heel foot
geoms under fixed sagittal base pitch. It does not train, does not modify
rewards, and does not edit the source scene.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from tools.sedon_debug_common import DEFAULT_SCENE_PATH, DEBUG_OUT_DIR, geom_name, require_scene


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "static_tilt_handoff.csv"
DEFAULT_SUMMARY_MD = DEBUG_OUT_DIR / "static_tilt_handoff_summary.md"
FOOT_FORCE_KEYS = (
    "left_center_force",
    "left_toe_force",
    "left_heel_force",
    "right_center_force",
    "right_toe_force",
    "right_heel_force",
)


@dataclass(frozen=True)
class TiltRow:
    """One static pitch handoff sample."""

    scene_name: str
    pitch_deg: float
    base_z: float
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
    heel_active: bool
    center_active: bool


@dataclass(frozen=True)
class TiltSummary:
    """Aggregated handoff metrics for one tilt sweep."""

    scene_name: str
    pitch0_center_ok: bool
    pitch0_rocker_false_touch: bool
    first_toe_touch_deg: float | None
    first_toe_load_deg: float | None
    max_toe_force: float
    max_center_force: float
    max_heel_force: float
    base_z_at_0deg: float
    min_base_z: float
    base_z_drop_ratio: float
    base_z_drop_ok: bool


def _parse_pitch_degrees(raw_value: str) -> list[float]:
    """Parse comma-separated pitch degrees."""
    values = [float(item.strip()) for item in raw_value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one pitch degree.")
    return values


def _pitch_quat(pitch_rad: float) -> np.ndarray:
    """Return MuJoCo free-joint quaternion for pitch around the y axis."""
    half = 0.5 * pitch_rad
    return np.array([math.cos(half), 0.0, math.sin(half), 0.0], dtype=np.float64)


def _set_base_pitch(model: mujoco.MjModel, data: mujoco.MjData, pitch_deg: float) -> None:
    """Reset the model and set floating-base sagittal pitch."""
    mujoco.mj_resetData(model, data)
    data.qpos[3:7] = _pitch_quat(math.radians(pitch_deg))
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _settle(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    """Let MuJoCo settle contacts without applying active controls."""
    for _ in range(max(0, steps)):
        mujoco.mj_step(model, data)


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
    """Sum floor-contact normal force by side and contact geom region."""
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


def run_sweep(
    scene_path: Path,
    pitch_degrees: list[float],
    settle_steps: int,
) -> list[TiltRow]:
    """Run the static tilt handoff sweep."""
    scene_path = require_scene(scene_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    rows: list[TiltRow] = []
    scene_name = scene_path.stem

    for pitch_deg in pitch_degrees:
        _set_base_pitch(model, data, pitch_deg)
        _settle(model, data, settle_steps)
        forces = _contact_forces(model, data)
        total_center = forces["left_center_force"] + forces["right_center_force"]
        total_toe = forces["left_toe_force"] + forces["right_toe_force"]
        total_heel = forces["left_heel_force"] + forces["right_heel_force"]
        rows.append(
            TiltRow(
                scene_name=scene_name,
                pitch_deg=float(pitch_deg),
                base_z=float(data.qpos[2]),
                left_center_force=forces["left_center_force"],
                left_toe_force=forces["left_toe_force"],
                left_heel_force=forces["left_heel_force"],
                right_center_force=forces["right_center_force"],
                right_toe_force=forces["right_toe_force"],
                right_heel_force=forces["right_heel_force"],
                total_center_force=total_center,
                total_toe_force=total_toe,
                total_heel_force=total_heel,
                toe_active=total_toe > 1e-6,
                heel_active=total_heel > 1e-6,
                center_active=total_center > 1e-6,
            )
        )
    return rows


def _write_csv(path: Path, rows: list[TiltRow]) -> None:
    """Write sweep rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([row.__dict__ for row in rows])


def summarize_rows(rows: list[TiltRow]) -> TiltSummary:
    """Return aggregate static tilt handoff metrics."""
    if not rows:
        raise ValueError("Cannot summarize an empty tilt sweep.")
    pitch_zero = min(rows, key=lambda row: abs(row.pitch_deg))
    first_toe_touch_deg = min(
        (row.pitch_deg for row in rows if row.total_toe_force > 1e-6),
        default=None,
    )
    first_toe_load_deg = min(
        (row.pitch_deg for row in rows if row.total_toe_force > 0.5),
        default=None,
    )
    max_toe_force = max(row.total_toe_force for row in rows)
    max_center_force = max(row.total_center_force for row in rows)
    max_heel_force = max(row.total_heel_force for row in rows)
    min_base_z = min(row.base_z for row in rows)
    base_z_at_0deg = pitch_zero.base_z
    base_z_drop_ratio = (
        max(0.0, base_z_at_0deg - min_base_z) / base_z_at_0deg
        if base_z_at_0deg > 1e-9
        else float("nan")
    )
    return TiltSummary(
        scene_name=rows[0].scene_name,
        pitch0_center_ok=pitch_zero.center_active,
        pitch0_rocker_false_touch=pitch_zero.toe_active or pitch_zero.heel_active,
        first_toe_touch_deg=first_toe_touch_deg,
        first_toe_load_deg=first_toe_load_deg,
        max_toe_force=max_toe_force,
        max_center_force=max_center_force,
        max_heel_force=max_heel_force,
        base_z_at_0deg=base_z_at_0deg,
        min_base_z=min_base_z,
        base_z_drop_ratio=base_z_drop_ratio,
        base_z_drop_ok=base_z_drop_ratio <= 0.10,
    )


def _write_summary(path: Path, rows: list[TiltRow], csv_path: Path) -> None:
    """Write markdown summary for the handoff sweep."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sedon Static Tilt Handoff",
        "",
        "Task class: Class C geometry diagnostic. No PPO, reward, train.py, or source-scene changes.",
        "",
        f"CSV: `{csv_path}`",
        "",
        "## Summary",
        "",
    ]
    if not rows:
        lines.append("No rows were produced.")
    else:
        summary = summarize_rows(rows)
        lines.extend(
            [
                f"- pitch=0 center carries load: {summary.pitch0_center_ok}",
                f"- pitch=0 toe/heel false contact: {summary.pitch0_rocker_false_touch}",
                f"- first toe touch pitch: {_fmt_optional(summary.first_toe_touch_deg)} deg",
                f"- first toe load pitch (>0.5 N): {_fmt_optional(summary.first_toe_load_deg)} deg",
                f"- first_toe_load_deg <= 8: {summary.first_toe_load_deg is not None and summary.first_toe_load_deg <= 8.0}",
                f"- max toe force: {summary.max_toe_force:.4f}",
                f"- max center force: {summary.max_center_force:.4f}",
                f"- max heel force: {summary.max_heel_force:.4f}",
                f"- base_z_at_0deg: {summary.base_z_at_0deg:.4f}",
                f"- min_base_z: {summary.min_base_z:.4f}",
                f"- base_z_drop_ratio <= 0.10: {summary.base_z_drop_ok} ({summary.base_z_drop_ratio:.4f})",
                f"- min_base_z > 0.14: {summary.min_base_z > 0.14}",
                "",
                "## Rows",
                "",
                "| pitch_deg | base_z | center_force | toe_force | heel_force | center | toe | heel |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row.pitch_deg:.2f} | {row.base_z:.4f} | "
                f"{row.total_center_force:.4f} | {row.total_toe_force:.4f} | "
                f"{row.total_heel_force:.4f} | {row.center_active} | "
                f"{row.toe_active} | {row.heel_active} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_rows(path: Path) -> list[TiltRow]:
    """Read tilt rows from one CSV."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            rows.append(
                TiltRow(
                    scene_name=str(raw["scene_name"]),
                    pitch_deg=float(raw["pitch_deg"]),
                    base_z=float(raw["base_z"]),
                    left_center_force=float(raw["left_center_force"]),
                    left_toe_force=float(raw["left_toe_force"]),
                    left_heel_force=float(raw["left_heel_force"]),
                    right_center_force=float(raw["right_center_force"]),
                    right_toe_force=float(raw["right_toe_force"]),
                    right_heel_force=float(raw["right_heel_force"]),
                    total_center_force=float(raw["total_center_force"]),
                    total_toe_force=float(raw["total_toe_force"]),
                    total_heel_force=float(raw["total_heel_force"]),
                    toe_active=str(raw["toe_active"]).lower() == "true",
                    heel_active=str(raw["heel_active"]).lower() == "true",
                    center_active=str(raw["center_active"]).lower() == "true",
                )
            )
        return rows


def write_comparison_report(csv_paths: list[Path], report_path: Path) -> None:
    """Write a markdown comparison report across multiple tilt sweeps."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summaries = []
    for csv_path in csv_paths:
        rows = _read_rows(csv_path)
        if rows:
            summaries.append((csv_path, summarize_rows(rows)))

    lines = [
        "# Blue-Like Sole V4 Static Tilt Comparison",
        "",
        "Task class: Class C geometry diagnostic. No PPO, reward, train.py, or source-scene changes.",
        "",
        "| variant | pitch0_center_ok | pitch0_rocker_false_touch | first_toe_touch_deg | first_toe_load_deg | max_toe_force | base_z_drop_ratio | pass/fail | recommendation |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for _, summary in summaries:
        passes = (
            summary.pitch0_center_ok
            and not summary.pitch0_rocker_false_touch
            and summary.first_toe_load_deg is not None
            and summary.first_toe_load_deg <= 8.0
            and summary.base_z_drop_ratio <= 0.10
        )
        recommendation = _recommendation(summary, passes)
        lines.append(
            f"| {summary.scene_name} | {summary.pitch0_center_ok} | "
            f"{summary.pitch0_rocker_false_touch} | "
            f"{_fmt_optional(summary.first_toe_touch_deg)} | "
            f"{_fmt_optional(summary.first_toe_load_deg)} | "
            f"{summary.max_toe_force:.4f} | {summary.base_z_drop_ratio:.4f} | "
            f"{'pass' if passes else 'fail'} | {recommendation} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recommendation(summary: TiltSummary, passes: bool) -> str:
    """Return a concise recommendation for one geometry variant."""
    if passes:
        return "best candidate for next scripted dynamic preview"
    if not summary.pitch0_center_ok:
        return "reject: center is not primary at zero pitch"
    if summary.pitch0_rocker_false_touch:
        return "reject: rocker touches at zero pitch"
    if summary.first_toe_load_deg is None:
        return "toe never loads in tested pitch range"
    if summary.first_toe_load_deg > 8.0:
        return "toe handoff too late"
    if summary.base_z_drop_ratio > 0.10:
        return "base drops too much during tilt"
    return "borderline; inspect CSV"


def _fmt_optional(value: float | None) -> str:
    """Format optional float."""
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-path", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument(
        "--comparison-csvs",
        type=Path,
        nargs="*",
        default=None,
        help="Existing static tilt CSV files to summarize into --comparison-report.",
    )
    parser.add_argument("--comparison-report", type=Path, default=None)
    parser.add_argument(
        "--pitch-degrees",
        type=_parse_pitch_degrees,
        default=_parse_pitch_degrees("0,2,4,6,8,10,12,15"),
    )
    parser.add_argument("--settle-steps", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run static tilt handoff diagnostic."""
    args = build_parser().parse_args(argv)
    if args.comparison_csvs:
        if args.comparison_report is None:
            raise ValueError("--comparison-report is required with --comparison-csvs.")
        write_comparison_report(args.comparison_csvs, args.comparison_report)
        print(f"comparison: {args.comparison_report}")
        return 0
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative.")
    rows = run_sweep(args.scene_path, args.pitch_degrees, args.settle_steps)
    _write_csv(args.out_csv, rows)
    _write_summary(args.summary_md, rows, args.out_csv)
    print(f"csv: {args.out_csv}")
    print(f"summary: {args.summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
