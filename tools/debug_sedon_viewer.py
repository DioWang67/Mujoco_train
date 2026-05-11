"""Open a Sedon MuJoCo viewer with visible training proxy geoms."""

from __future__ import annotations

import argparse
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from tools.sedon_debug_common import (
    BASE_PROXY_GEOM,
    DEBUG_OUT_DIR,
    DEFAULT_SCENE_PATH,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    require_scene,
)


DEBUG_SCENE_PATH = DEBUG_OUT_DIR / "training_scene_debug_visible_geoms.xml"


def _write_visible_proxy_scene(source: Path, output: Path) -> Path:
    """Write a debug XML where proxy collision geoms are visible."""
    output.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source)
    root = tree.getroot()
    for geom in root.findall(".//geom"):
        name = geom.attrib.get("name")
        if name in (RIGHT_FOOT_GEOM, LEFT_FOOT_GEOM):
            geom.set("rgba", "0.1 0.55 1.0 0.65")
            geom.set("group", "0")
        elif name == BASE_PROXY_GEOM:
            geom.set("rgba", "1.0 0.35 0.05 0.35")
            geom.set("group", "0")
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)

    mesh_source = source.parent / "mjcf_source"
    mesh_target = output.parent / "mjcf_source"
    if mesh_source.is_dir() and not mesh_target.exists():
        shutil.copytree(mesh_source, mesh_target)
    return output


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--debug-scene", type=Path, default=DEBUG_SCENE_PATH)
    parser.add_argument(
        "--no-visible-copy",
        action="store_true",
        help="Load the original scene without rewriting proxy colors.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Open MuJoCo viewer for inspecting Sedon proxy contacts."""
    args = build_parser().parse_args(argv)
    source_scene = require_scene(args.scene)
    scene_path = (
        source_scene
        if args.no_visible_copy
        else _write_visible_proxy_scene(source_scene, args.debug_scene)
    )

    try:
        import mujoco.viewer
    except Exception as exc:  # noqa: BLE001 - provide actionable CLI guidance.
        raise RuntimeError(
            "mujoco.viewer is unavailable in this Python environment. "
            "Install MuJoCo viewer extras or use `python -m tools.trace_zero_action_gait` "
            "for headless contact diagnostics."
        ) from exc

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print(f"viewer scene: {scene_path}")
    print("proxy colors: feet=blue, base_proxy=orange")
    print("In the MuJoCo viewer, use the visualization panel to enable contacts/contact forces.")
    print("Close the viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

