"""Inspect settled world z/contact state for Seedon foot-related geoms.

This Class C geometry diagnostic loads MJCF scenes directly with MuJoCo,
settles them briefly with zero controls, and reports foot/toe/heel/sole/center
geom heights. It does not train or modify any scene.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco

from tools.seedon_debug_common import DEBUG_OUT_DIR, geom_name, require_scene


DEFAULT_SCENES = (
    DEBUG_OUT_DIR
    / "blue_like_sole_experiments_v3"
    / "training_scene_v3_rocker_up_005_center_down_002.xml",
    DEBUG_OUT_DIR / "blue_like_sole_experiments_v4" / "training_scene_v4_a.xml",
    DEBUG_OUT_DIR / "blue_like_sole_experiments_v4" / "training_scene_v4_b.xml",
    DEBUG_OUT_DIR / "blue_like_sole_experiments_v4" / "training_scene_v4_c.xml",
)
DEFAULT_REPORT = DEBUG_OUT_DIR / "blue_like_sole_experiments_v4" / "foot_geom_z_report.md"
NAME_TOKENS = ("foot", "toe", "heel", "sole", "center")


@dataclass(frozen=True)
class GeomZRow:
    """Settled z/contact diagnostics for one geom."""

    scene_name: str
    geom_name: str
    geom_type: str
    world_pos_z: float
    estimated_bottom_z: float
    has_contact: bool


def _geom_type_name(model: mujoco.MjModel, geom_id: int) -> str:
    """Return MuJoCo geom type name without prefix."""
    return mujoco.mjtGeom(int(model.geom_type[geom_id])).name.replace("mjGEOM_", "").lower()


def _estimated_bottom_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    """Estimate bottom z for simple foot geoms."""
    geom_type = mujoco.mjtGeom(int(model.geom_type[geom_id]))
    pos_z = float(data.geom_xpos[geom_id][2])
    size = model.geom_size[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return pos_z - float(size[2])
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return pos_z - float(size[0])
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return pos_z - float(size[0])
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return pos_z - float(size[1])
    if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return pos_z - float(size[2])
    return pos_z


def _settle(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    """Run zero-control settle steps."""
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    for _ in range(steps):
        if data.ctrl.size:
            data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)


def _has_contact(data: mujoco.MjData, geom_id: int) -> bool:
    """Return whether a geom appears in the current contact buffer."""
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if int(contact.geom1) == geom_id or int(contact.geom2) == geom_id:
            return True
    return False


def inspect_scene(scene_path: Path, settle_steps: int) -> list[GeomZRow]:
    """Inspect one scene and return sorted foot-related geom rows."""
    scene_path = require_scene(scene_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    _settle(model, data, settle_steps)
    rows: list[GeomZRow] = []
    for geom_id in range(model.ngeom):
        name = geom_name(model, geom_id)
        lower_name = name.lower()
        if not any(token in lower_name for token in NAME_TOKENS):
            continue
        rows.append(
            GeomZRow(
                scene_name=scene_path.stem,
                geom_name=name,
                geom_type=_geom_type_name(model, geom_id),
                world_pos_z=float(data.geom_xpos[geom_id][2]),
                estimated_bottom_z=_estimated_bottom_z(model, data, geom_id),
                has_contact=_has_contact(data, geom_id),
            )
        )
    return sorted(rows, key=lambda row: (row.scene_name, row.geom_name))


def write_report(path: Path, rows: list[GeomZRow]) -> None:
    """Write markdown comparison table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Foot Geom Z Inspection",
        "",
        "Task class: Class C geometry diagnostic. No PPO, reward, train.py, or source-scene changes.",
        "",
        "| scene | geom | type | world_pos_z | estimated_bottom_z | has_contact |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.scene_name} | {row.geom_name} | {row.geom_type} | "
            f"{row.world_pos_z:.5f} | {row.estimated_bottom_z:.5f} | {row.has_contact} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    parser.add_argument("--settle-steps", type=int, default=50)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run foot geom z inspection."""
    args = build_parser().parse_args(argv)
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative.")
    scene_paths = args.scene_path or list(DEFAULT_SCENES)
    rows: list[GeomZRow] = []
    for scene_path in scene_paths:
        rows.extend(inspect_scene(scene_path, args.settle_steps))
    write_report(args.report_md, rows)
    print(f"report: {args.report_md}")
    for row in rows:
        print(
            f"{row.scene_name} {row.geom_name} {row.geom_type} "
            f"z={row.world_pos_z:.5f} bottom={row.estimated_bottom_z:.5f} "
            f"contact={row.has_contact}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
