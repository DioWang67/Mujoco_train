"""Shared helpers for Sedon MuJoCo contact/debug tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_PATH = REPO_ROOT / "private_assets" / "sedon_v5_22" / "training_scene.xml"
DEBUG_OUT_DIR = REPO_ROOT / "artifacts" / "sedon_debug"
FLOOR_GEOM = "floor"
RIGHT_FOOT_GEOM = "R_foot_collision"
LEFT_FOOT_GEOM = "L_foot_collision"
BASE_PROXY_GEOM = "base_proxy"
RELAXED_FOOT_SIZE = (0.16, 0.15, 0.025)
EXPECTED_CONTACT_GEOMS = {
    FLOOR_GEOM,
    RIGHT_FOOT_GEOM,
    LEFT_FOOT_GEOM,
    BASE_PROXY_GEOM,
}


@dataclass(frozen=True)
class GeomSnapshot:
    """World-space geometry diagnostics for one MuJoCo geom."""

    name: str
    geom_type: str
    position: np.ndarray
    size: np.ndarray
    bottom_z: float
    floor_distance: float
    flatness: float | None


def require_scene(path: Path) -> Path:
    """Return a resolved scene path or raise a useful error."""
    scene_path = path.resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"Sedon training scene not found: {scene_path}")
    return scene_path


def geom_id(model: mujoco.MjModel, name: str) -> int:
    """Resolve a geom name to an id."""
    resolved_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if resolved_id < 0:
        raise ValueError(f"Geom not found in model: {name}")
    return int(resolved_id)


def body_id(model: mujoco.MjModel, name: str) -> int:
    """Resolve a body name to an id."""
    resolved_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if resolved_id < 0:
        raise ValueError(f"Body not found in model: {name}")
    return int(resolved_id)


def geom_name(model: mujoco.MjModel, geom: int) -> str:
    """Return a stable geom name for reports."""
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom))
    return name or f"geom_{geom}"


def geom_type_name(model: mujoco.MjModel, geom: int) -> str:
    """Return MuJoCo geom type name without the mjGEOM_ prefix."""
    geom_type = mujoco.mjtGeom(int(model.geom_type[geom])).name
    return geom_type.replace("mjGEOM_", "").lower()


def estimate_geom_bottom_z(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> float:
    """Estimate the lowest world z for simple proxy geoms used in Sedon."""
    geom_type = mujoco.mjtGeom(int(model.geom_type[geom]))
    center_z = float(data.geom_xpos[geom][2])
    size = model.geom_size[geom]
    if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        return 0.0
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return center_z - float(size[2])
    if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return center_z - float(size[2])
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return center_z - float(size[0])
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return center_z - float(size[0] + size[1])
    return center_z


def geom_flatness(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> float:
    """Return alignment between geom local z-axis and world z-axis."""
    del model
    xmat = data.geom_xmat[geom].reshape(3, 3)
    return float(xmat[2, 2])


def snapshot_geom(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    name: str,
    floor_z: float = 0.0,
    include_flatness: bool = True,
) -> GeomSnapshot:
    """Collect world-space diagnostics for a named geom."""
    resolved_id = geom_id(model, name)
    bottom_z = estimate_geom_bottom_z(model, data, resolved_id)
    return GeomSnapshot(
        name=name,
        geom_type=geom_type_name(model, resolved_id),
        position=data.geom_xpos[resolved_id].copy(),
        size=model.geom_size[resolved_id].copy(),
        bottom_z=bottom_z,
        floor_distance=bottom_z - floor_z,
        flatness=geom_flatness(model, data, resolved_id) if include_flatness else None,
    )


def apply_foot_size_override(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_size: tuple[float, float, float],
) -> None:
    """Apply a temporary foot collision box half-size to both Sedon feet."""
    if len(foot_size) != 3:
        raise ValueError("foot_size must contain exactly three half-size values.")
    if any(value <= 0.0 for value in foot_size):
        raise ValueError("foot_size values must be positive.")
    for name in (RIGHT_FOOT_GEOM, LEFT_FOOT_GEOM):
        foot_id = geom_id(model, name)
        model.geom_size[foot_id] = np.array(foot_size, dtype=np.float64)
        model.geom_rbound[foot_id] = float(np.linalg.norm(model.geom_size[foot_id]))
    mujoco.mj_forward(model, data)


def contact_pairs(model: mujoco.MjModel, data: mujoco.MjData) -> list[tuple[str, str, float]]:
    """Return all current geom contact pairs with contact distance."""
    pairs: list[tuple[str, str, float]] = []
    for index in range(data.ncon):
        contact = data.contact[index]
        name_a = geom_name(model, int(contact.geom1))
        name_b = geom_name(model, int(contact.geom2))
        if name_a > name_b:
            name_a, name_b = name_b, name_a
        pairs.append((name_a, name_b, float(contact.dist)))
    return pairs


def count_contacts(
    pairs: Iterable[tuple[str, str, float]]
) -> dict[tuple[str, str], int]:
    """Count normalized contact pairs."""
    counts: dict[tuple[str, str], int] = {}
    for name_a, name_b, _ in pairs:
        key = tuple(sorted((name_a, name_b)))
        counts[key] = counts.get(key, 0) + 1
    return counts


def is_expected_floor_contact(name_a: str, name_b: str) -> bool:
    """Return whether a contact pair is one of the expected floor contacts."""
    pair = set((name_a, name_b))
    return pair in (
        {FLOOR_GEOM, RIGHT_FOOT_GEOM},
        {FLOOR_GEOM, LEFT_FOOT_GEOM},
    )


def is_base_floor_contact(name_a: str, name_b: str) -> bool:
    """Return whether the base proxy touches the floor."""
    return set((name_a, name_b)) == {FLOOR_GEOM, BASE_PROXY_GEOM}
