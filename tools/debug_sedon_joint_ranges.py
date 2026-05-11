"""Compare Sedon URDF joint limits with MuJoCo joint ranges."""

from __future__ import annotations

import argparse
import csv
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco

from sedon_baseline.env import DEFAULT_SCENE_PATH, JOINT_NAMES
from tools.sedon_debug_common import DEBUG_OUT_DIR


DEFAULT_ORIGINAL_URDF = Path("private_assets/sedon/original_urdf_package/urdf/urdf/urdf.urdf")
DEFAULT_PREPARED_URDF = Path("private_assets/sedon/mjcf_source/sedon.urdf")
DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "joint_range_compare.csv"


@dataclass(frozen=True)
class UrdfJointLimit:
    """Relevant joint limit fields parsed from URDF."""

    lower: float | None
    upper: float | None
    effort: float | None
    velocity: float | None
    axis: tuple[float, float, float] | None


def _parse_float(value: str | None) -> float | None:
    """Parse optional XML float attributes."""
    if value is None:
        return None
    return float(value)


def _parse_axis(value: str | None) -> tuple[float, float, float] | None:
    """Parse an optional URDF axis attribute."""
    if value is None:
        return None
    parts = [float(part) for part in value.split()]
    if len(parts) != 3:
        raise ValueError(f"Expected a 3D axis, got: {value}")
    return (parts[0], parts[1], parts[2])


def _load_urdf_limits(path: Path) -> dict[str, UrdfJointLimit]:
    """Load joint limits from one URDF file."""
    if not path.is_file():
        raise FileNotFoundError(f"URDF not found: {path}")
    root = ET.parse(path).getroot()
    limits: dict[str, UrdfJointLimit] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if not name:
            continue
        limit = joint.find("limit")
        axis = joint.find("axis")
        limits[name] = UrdfJointLimit(
            lower=_parse_float(limit.attrib.get("lower")) if limit is not None else None,
            upper=_parse_float(limit.attrib.get("upper")) if limit is not None else None,
            effort=_parse_float(limit.attrib.get("effort")) if limit is not None else None,
            velocity=_parse_float(limit.attrib.get("velocity")) if limit is not None else None,
            axis=_parse_axis(axis.attrib.get("xyz")) if axis is not None else None,
        )
    return limits


def _mujoco_joint_row(model: mujoco.MjModel, joint_name: str) -> dict[str, object]:
    """Return MuJoCo joint range and axis fields for one joint."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"MuJoCo joint not found: {joint_name}")
    return {
        "mujoco_lower": float(model.jnt_range[joint_id][0]),
        "mujoco_upper": float(model.jnt_range[joint_id][1]),
        "mujoco_axis_x": float(model.jnt_axis[joint_id][0]),
        "mujoco_axis_y": float(model.jnt_axis[joint_id][1]),
        "mujoco_axis_z": float(model.jnt_axis[joint_id][2]),
    }


def _same_optional_float(a: float | None, b: float | None, tolerance: float) -> bool:
    """Compare optional floats."""
    if a is None or b is None:
        return a is b
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write comparison rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-urdf", type=Path, default=DEFAULT_ORIGINAL_URDF)
    parser.add_argument("--prepared-urdf", type=Path, default=DEFAULT_PREPARED_URDF)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Compare URDF and MuJoCo joint limits."""
    args = build_parser().parse_args(argv)
    if args.tolerance < 0.0:
        raise ValueError("--tolerance must be non-negative.")

    original = _load_urdf_limits(args.original_urdf)
    prepared = _load_urdf_limits(args.prepared_urdf)
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    rows: list[dict[str, object]] = []

    print("joint original_range prepared_range mujoco_range axis_match range_match")
    for joint_name in JOINT_NAMES:
        original_limit = original.get(joint_name)
        prepared_limit = prepared.get(joint_name)
        if original_limit is None:
            raise ValueError(f"Original URDF missing joint: {joint_name}")
        if prepared_limit is None:
            raise ValueError(f"Prepared URDF missing joint: {joint_name}")

        mj = _mujoco_joint_row(model, joint_name)
        range_match = (
            _same_optional_float(original_limit.lower, prepared_limit.lower, args.tolerance)
            and _same_optional_float(original_limit.upper, prepared_limit.upper, args.tolerance)
            and _same_optional_float(original_limit.lower, mj["mujoco_lower"], args.tolerance)
            and _same_optional_float(original_limit.upper, mj["mujoco_upper"], args.tolerance)
        )
        axis_match = original_limit.axis == prepared_limit.axis and original_limit.axis == (
            mj["mujoco_axis_x"],
            mj["mujoco_axis_y"],
            mj["mujoco_axis_z"],
        )
        row = {
            "joint": joint_name,
            "original_lower": original_limit.lower,
            "original_upper": original_limit.upper,
            "prepared_lower": prepared_limit.lower,
            "prepared_upper": prepared_limit.upper,
            "mujoco_lower": mj["mujoco_lower"],
            "mujoco_upper": mj["mujoco_upper"],
            "original_effort": original_limit.effort,
            "original_velocity": original_limit.velocity,
            "original_axis": original_limit.axis,
            "prepared_axis": prepared_limit.axis,
            "mujoco_axis": (
                mj["mujoco_axis_x"],
                mj["mujoco_axis_y"],
                mj["mujoco_axis_z"],
            ),
            "range_match": range_match,
            "axis_match": axis_match,
        }
        rows.append(row)
        print(
            f"{joint_name} "
            f"[{original_limit.lower:.4f},{original_limit.upper:.4f}] "
            f"[{prepared_limit.lower:.4f},{prepared_limit.upper:.4f}] "
            f"[{mj['mujoco_lower']:.4f},{mj['mujoco_upper']:.4f}] "
            f"{axis_match} {range_match}"
        )

    _write_csv(args.out_csv, rows)
    print(f"\nCSV: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
