"""Run Open Duck Mini v2 viewer from this repo's virtual environment.

Open Duck's original ``mujoco_infer.py`` imports ``base.py`` only to call
``get_assets()``, but that module also imports ``mujoco.mjx``. Some Windows
MuJoCo wheels do not expose mjx, so this wrapper injects a tiny replacement
``playground.open_duck_mini_v2.base`` module that only provides ``get_assets``.

The original Open Duck inference code is still executed; this file only avoids
the mjx-only training environment dependency.
"""

from __future__ import annotations

import argparse
import runpy
import sys
import types
from pathlib import Path


DEFAULT_DUCK_ROOT = Path("C:/Users/diowang/open_duck_mini_ws/Open_Duck_Playground")
DEFAULT_SCRIPT = DEFAULT_DUCK_ROOT / "playground/open_duck_mini_v2/mujoco_infer.py"
DEFAULT_MODEL = DEFAULT_DUCK_ROOT / "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
DEFAULT_REFERENCE = DEFAULT_DUCK_ROOT / "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"


def collect_assets(duck_root: Path) -> dict[str, bytes]:
    """Collect XML and mesh assets needed by MuJoCo's from_xml_string loader."""
    root = duck_root / "playground" / "open_duck_mini_v2"
    xml_root = root / "xmls"
    asset_root = xml_root / "assets"
    assets: dict[str, bytes] = {}
    if xml_root.exists():
        for path in xml_root.glob("*.xml"):
            assets[path.name] = path.read_bytes()
    if asset_root.exists():
        for path in asset_root.iterdir():
            if path.is_file():
                assets[path.name] = path.read_bytes()
    return assets


def install_base_stub(duck_root: Path) -> None:
    """Inject the minimal base module expected by mujoco_infer_base.py."""
    module = types.ModuleType("playground.open_duck_mini_v2.base")
    module.get_assets = lambda: collect_assets(duck_root)  # type: ignore[attr-defined]
    sys.modules["playground.open_duck_mini_v2.base"] = module


def parse_args() -> argparse.Namespace:
    """Parse wrapper arguments and pass unknown args to original mujoco_infer."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--duck-root", type=Path, default=DEFAULT_DUCK_ROOT)
    args, passthrough = parser.parse_known_args()
    args.passthrough = passthrough
    return args


def main() -> None:
    """Run the Open Duck viewer script with a base.get_assets stub."""
    args = parse_args()
    duck_root = args.duck_root.resolve()
    script = duck_root / "playground/open_duck_mini_v2/mujoco_infer.py"
    if not script.is_file():
        raise FileNotFoundError(f"Open Duck mujoco_infer.py not found: {script}")
    if str(duck_root) not in sys.path:
        sys.path.insert(0, str(duck_root))
    install_base_stub(duck_root)

    passthrough = list(args.passthrough)
    if "--model_path" not in passthrough:
        passthrough.extend(["--model_path", str(DEFAULT_MODEL)])
    if "--reference_data" not in passthrough:
        passthrough.extend(["--reference_data", str(DEFAULT_REFERENCE)])
    sys.argv = [str(script), *passthrough]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
