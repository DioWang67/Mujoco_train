"""Inspect Seedon training-scene proxy geoms and initial contacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from seedon_baseline.env import SeedonStandingEnv
from tools.seedon_debug_common import (
    BASE_PROXY_GEOM,
    DEFAULT_SCENE_PATH,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    body_id,
    contact_pairs,
    geom_id,
    is_base_floor_contact,
    is_expected_floor_contact,
    require_scene,
    snapshot_geom,
)


def _format_vector(values: np.ndarray) -> str:
    """Format a small numeric vector for console output."""
    return " ".join(f"{float(value): .5f}" for value in values)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        "--scene-path",
        dest="scene",
        type=Path,
        default=DEFAULT_SCENE_PATH,
        help="Path to Seedon training_scene.xml.",
    )
    parser.add_argument(
        "--raw-scene-reset",
        action="store_true",
        help="Inspect MuJoCo's raw XML reset instead of SeedonStandingEnv.reset().",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print Seedon contact proxy diagnostics for the reset pose."""
    args = build_parser().parse_args(argv)
    scene_path = require_scene(args.scene)
    env: SeedonStandingEnv | None = None
    if args.raw_scene_reset:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        reset_mode = "raw MuJoCo XML reset"
    else:
        env = SeedonStandingEnv(scene_path=scene_path, reset_noise_scale=0.0)
        env.reset(seed=42)
        model = env.model
        data = env.data
        reset_mode = "SeedonStandingEnv.reset(seed=42)"

    print(f"scene: {scene_path}")
    print(f"reset: {reset_mode}")
    print(
        "model: "
        f"nbody={model.nbody} njnt={model.njnt} ngeom={model.ngeom} "
        f"nq={model.nq} nv={model.nv} nu={model.nu}"
    )

    base_body = body_id(model, "base_link")
    print("\nbase body")
    print(f"  world_pos: {_format_vector(data.xpos[base_body])}")
    print(f"  world_quat: {_format_vector(data.xquat[base_body])}")

    print("\nproxy geoms")
    snapshots = [
        snapshot_geom(model, data, name=RIGHT_FOOT_GEOM),
        snapshot_geom(model, data, name=LEFT_FOOT_GEOM),
        snapshot_geom(model, data, name=BASE_PROXY_GEOM),
    ]
    for item in snapshots:
        flatness = "n/a" if item.flatness is None else f"{item.flatness:.5f}"
        print(f"- {item.name} ({item.geom_type})")
        print(f"  world_pos      : {_format_vector(item.position)}")
        print(f"  size           : {_format_vector(item.size)}")
        print(f"  bottom_z       : {item.bottom_z:.5f}")
        print(f"  floor_distance : {item.floor_distance:.5f}")
        print(f"  flatness       : {flatness}")

    print("\nfoot checks")
    for foot_name in (RIGHT_FOOT_GEOM, LEFT_FOOT_GEOM):
        item = snapshot_geom(model, data, name=foot_name)
        near_floor = abs(item.floor_distance) <= 0.005
        flat = item.flatness is not None and item.flatness >= 0.98
        print(
            f"- {foot_name}: "
            f"near_floor={near_floor} flat={flat} "
            f"distance={item.floor_distance:.5f} flatness={item.flatness:.5f}"
        )

    base_proxy = snapshot_geom(model, data, name=BASE_PROXY_GEOM)
    print("\nbase proxy check")
    print(f"  bottom_z       : {base_proxy.bottom_z:.5f}")
    print(f"  floor_distance : {base_proxy.floor_distance:.5f}")
    if base_proxy.floor_distance <= 0.02:
        print("  warning        : base_proxy is close to the floor at reset.")
    else:
        print("  warning        : none at reset")

    print("\ninitial contacts")
    pairs = contact_pairs(model, data)
    if not pairs:
        print("  none")
    for name_a, name_b, distance in pairs:
        tags: list[str] = []
        if is_base_floor_contact(name_a, name_b):
            tags.append("WARNING_BASE_FLOOR")
        elif not is_expected_floor_contact(name_a, name_b):
            tags.append("UNEXPECTED")
        suffix = f" [{' '.join(tags)}]" if tags else ""
        print(f"  {name_a} <-> {name_b} dist={distance:.6f}{suffix}")

    print("\nrequired geom ids")
    for name in (FLOOR_GEOM, RIGHT_FOOT_GEOM, LEFT_FOOT_GEOM, BASE_PROXY_GEOM):
        print(f"  {name}: {geom_id(model, name)}")
    if env is not None:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
