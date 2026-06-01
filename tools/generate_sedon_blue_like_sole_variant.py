"""Generate conservative Blue-like rounded sole scene variants for Sedon.

This Class C experiment helper does not modify the canonical Sedon scene,
does not change rewards, and does not run PPO. The generated variants keep a
flat central contact patch and add small passive roll helper geoms.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco

from tools.sedon_debug_common import DEFAULT_SCENE_PATH, DEBUG_OUT_DIR, require_scene


OUT_DIR = DEBUG_OUT_DIR / "blue_like_sole_experiments"
V3_OUT_DIR = DEBUG_OUT_DIR / "blue_like_sole_experiments_v3"
V4_OUT_DIR = DEBUG_OUT_DIR / "blue_like_sole_experiments_v4"
GAP_REVIEW_PATH = DEBUG_OUT_DIR / "blue_like_sedon_gap_review.md"
FOOT_GEOM_NAMES = ("R_foot_collision", "L_foot_collision")
BASE_CENTER_Z = -0.061
BASE_ROCKER_BOTTOM_Z = -0.079


@dataclass(frozen=True)
class BlueLikeSoleVariant:
    """One conservative Blue/BDX-like foot approximation."""

    name: str
    scene_filename: str
    center_half_length: float
    center_half_width: float
    center_half_height: float
    rocker_radius: float
    shoulder_radius: float
    lateral_shoulder_offset: float
    sagittal_roll_design: str
    lateral_roll_design: str
    rocker_z_offset: float = 0.0
    center_z_offset: float = 0.0
    toe_rocker_z_extra_offset: float = 0.0
    toe_rocker_x_offset: float = 0.0


def _variants(variant_set: str = "base") -> list[BlueLikeSoleVariant]:
    """Return one conservative variant plus one mild wider-shoulder variant."""
    if variant_set == "v3_contact_ordering":
        return [
            _v3_variant("blue_like_sole_v3_rocker_up_003", 0.003, 0.0),
            _v3_variant("blue_like_sole_v3_rocker_up_005", 0.005, 0.0),
            _v3_variant("blue_like_sole_v3_rocker_up_008", 0.008, 0.0),
            _v3_variant("blue_like_sole_v3_rocker_up_005_center_down_002", 0.005, -0.002),
        ]
    if variant_set == "v4_tilt_handoff":
        return [
            _v4_variant("blue_like_sole_v4_a", rocker_up=0.004, toe_extra=0.0),
            _v4_variant("blue_like_sole_v4_b", rocker_up=0.0035, toe_extra=0.0),
            _v4_variant("blue_like_sole_v4_c", rocker_up=0.004, toe_extra=-0.001),
        ]
    if variant_set == "v5_toe_drop":
        return [
            _v5_variant("blue_like_sole_v5_a", toe_drop_from_v4c=0.004, toe_x_forward=0.0),
            _v5_variant("blue_like_sole_v5_b", toe_drop_from_v4c=0.005, toe_x_forward=0.0),
            _v5_variant("blue_like_sole_v5_c", toe_drop_from_v4c=0.003, toe_x_forward=0.005),
        ]
    return [
        BlueLikeSoleVariant(
            name="blue_like_sole_v1",
            scene_filename="training_scene_blue_like_sole_v1.xml",
            center_half_length=0.045,
            center_half_width=0.032,
            center_half_height=0.018,
            rocker_radius=0.012,
            shoulder_radius=0.007,
            lateral_shoulder_offset=0.038,
            sagittal_roll_design=(
                "toe and heel capsules along local lateral axis; radius 0.012 m; "
                "outer x envelope kept near original flat foot length"
            ),
            lateral_roll_design=(
                "small side shoulder capsules along local sagittal axis; radius 0.007 m; "
                "lower than aggressive ellipsoid curvature to avoid lateral rocking"
            ),
        ),
        BlueLikeSoleVariant(
            name="blue_like_sole_v2_mild_wide",
            scene_filename="training_scene_blue_like_sole_v2_mild_wide.xml",
            center_half_length=0.045,
            center_half_width=0.038,
            center_half_height=0.018,
            rocker_radius=0.012,
            shoulder_radius=0.008,
            lateral_shoulder_offset=0.044,
            sagittal_roll_design=(
                "same toe/heel sagittal rocker as v1; center patch remains short enough "
                "to allow forward roll"
            ),
            lateral_roll_design=(
                "slightly wider and larger side shoulder than v1, still using a flat center "
                "patch to avoid the ellipsoid v1 no-contact failure mode"
            ),
        ),
    ]


def _v3_variant(
    name: str,
    rocker_z_offset: float,
    center_z_offset: float,
) -> BlueLikeSoleVariant:
    """Return one v3 contact-ordering calibration variant."""
    label = name.replace("blue_like_sole_", "")
    return BlueLikeSoleVariant(
        name=name,
        scene_filename=f"training_scene_{label}.xml",
        center_half_length=0.045,
        center_half_width=0.032,
        center_half_height=0.018,
        rocker_radius=0.012,
        shoulder_radius=0.007,
        lateral_shoulder_offset=0.038,
        rocker_z_offset=rocker_z_offset,
        center_z_offset=center_z_offset,
        sagittal_roll_design=(
            f"v3 contact-ordering calibration; toe/heel rocker bottom raised by "
            f"{rocker_z_offset:.3f} m relative to v1 so center patch should contact first"
        ),
        lateral_roll_design=(
            "same lateral shoulder geometry as v1; lateral roll intentionally not changed"
        ),
    )


def _v4_variant(
    name: str,
    *,
    rocker_up: float,
    toe_extra: float,
) -> BlueLikeSoleVariant:
    """Return one v4 static-tilt handoff calibration variant."""
    return BlueLikeSoleVariant(
        name=name,
        scene_filename=f"training_scene_{name.replace('blue_like_sole_', '')}.xml",
        center_half_length=0.045,
        center_half_width=0.032,
        center_half_height=0.018,
        rocker_radius=0.012,
        shoulder_radius=0.007,
        lateral_shoulder_offset=0.038,
        sagittal_roll_design=(
            f"v4 tilt-handoff calibration; center patch down 0.002 m, "
            f"toe/heel rocker up {rocker_up:.4f} m"
            + (
                f", toe rocker extra offset {toe_extra:.4f} m"
                if abs(toe_extra) > 0.0
                else ""
            )
        ),
        lateral_roll_design=(
            "same lateral shoulder geometry as v1/v3; lateral roll intentionally not changed"
        ),
        rocker_z_offset=rocker_up,
        center_z_offset=-0.002,
        toe_rocker_z_extra_offset=toe_extra,
    )


def _v5_variant(
    name: str,
    *,
    toe_drop_from_v4c: float,
    toe_x_forward: float,
) -> BlueLikeSoleVariant:
    """Return one v5 toe-rocker drop calibration variant."""
    v4c_toe_extra = -0.001
    return BlueLikeSoleVariant(
        name=name,
        scene_filename=f"training_scene_{name.replace('blue_like_sole_', '')}.xml",
        center_half_length=0.045,
        center_half_width=0.032,
        center_half_height=0.018,
        rocker_radius=0.012,
        shoulder_radius=0.007,
        lateral_shoulder_offset=0.038,
        sagittal_roll_design=(
            "v5 toe-drop calibration based on v4_c; center down 0.002 m, "
            "heel rocker unchanged from v4_a, toe rocker lowered further by "
            f"{toe_drop_from_v4c:.3f} m relative to v4_c"
            + (f" and moved forward {toe_x_forward:.3f} m" if toe_x_forward else "")
        ),
        lateral_roll_design=(
            "same lateral shoulder geometry as v1/v3/v4; lateral roll intentionally not changed"
        ),
        rocker_z_offset=0.004,
        center_z_offset=-0.002,
        toe_rocker_z_extra_offset=v4c_toe_extra - toe_drop_from_v4c,
        toe_rocker_x_offset=toe_x_forward,
    )


def _fmt(values: tuple[float, ...]) -> str:
    """Format float tuples for MJCF attributes."""
    return " ".join(f"{value:.6g}" for value in values)


def _find_named_geom(root: ET.Element, geom_name: str) -> ET.Element:
    """Return a named geom from the MJCF tree."""
    for geom in root.iter("geom"):
        if geom.get("name") == geom_name:
            return geom
    raise ValueError(f"Foot geom not found in source scene: {geom_name}")


def _set_meshdir_to_source(tree: ET.ElementTree, source_scene: Path) -> None:
    """Make generated scenes loadable from the artifact directory."""
    compiler = tree.getroot().find("compiler")
    if compiler is None:
        return
    meshdir = source_scene.parent / (compiler.get("meshdir") or "")
    compiler.set("meshdir", str(meshdir.resolve()))


def _foot_side_center_y(geom: ET.Element) -> float:
    """Read the local foot center y from the original foot geom position."""
    pos = [float(item) for item in (geom.get("pos") or "").split()]
    if len(pos) != 3:
        raise ValueError(f"Foot geom has invalid pos: {geom.get('pos')}")
    return pos[1]


def _append_capsule(
    parent: ET.Element,
    *,
    name: str,
    fromto: tuple[float, float, float, float, float, float],
    radius: float,
    rgba: str,
    friction: str,
) -> None:
    """Append one passive roll helper capsule."""
    ET.SubElement(
        parent,
        "geom",
        {
            "name": name,
            "type": "capsule",
            "fromto": _fmt(fromto),
            "size": _fmt((radius,)),
            "rgba": rgba,
            "friction": friction,
        },
    )


def _apply_blue_like_sole(root: ET.Element, variant: BlueLikeSoleVariant) -> None:
    """Replace each flat foot with a center patch plus passive roll helpers."""
    for foot_name in FOOT_GEOM_NAMES:
        foot = _find_named_geom(root, foot_name)
        parent = _find_parent(root, foot)
        if parent is None:
            raise ValueError(f"Could not find parent for {foot_name}")

        center_pos = [float(item) for item in (foot.get("pos") or "").split()]
        center_y = _foot_side_center_y(foot)
        friction = foot.get("friction", "1.0 0.005 0.0001")
        rgba = foot.get("rgba", "0.12 0.12 0.12 0")

        center_pos[2] = BASE_CENTER_Z + variant.center_z_offset
        foot.set("type", "box")
        foot.set(
            "size",
            _fmt(
                (
                    variant.center_half_length,
                    variant.center_half_width,
                    variant.center_half_height,
                )
            ),
        )
        foot.set("pos", _fmt(tuple(center_pos)))

        y_min = center_y - variant.center_half_width
        y_max = center_y + variant.center_half_width
        toe_x = 0.095 - variant.rocker_radius + variant.toe_rocker_x_offset
        heel_x = -0.045 + variant.rocker_radius
        toe_rocker_z = (
            BASE_ROCKER_BOTTOM_Z
            + variant.rocker_z_offset
            + variant.toe_rocker_z_extra_offset
            + variant.rocker_radius
        )
        heel_rocker_z = BASE_ROCKER_BOTTOM_Z + variant.rocker_z_offset + variant.rocker_radius
        _append_capsule(
            parent,
            name=f"{foot_name}_toe_rocker",
            fromto=(toe_x, y_min, toe_rocker_z, toe_x, y_max, toe_rocker_z),
            radius=variant.rocker_radius,
            rgba=rgba,
            friction=friction,
        )
        _append_capsule(
            parent,
            name=f"{foot_name}_heel_rocker",
            fromto=(heel_x, y_min, heel_rocker_z, heel_x, y_max, heel_rocker_z),
            radius=variant.rocker_radius,
            rgba=rgba,
            friction=friction,
        )

        shoulder_z = -0.079 + variant.shoulder_radius
        shoulder_x_min = -0.035
        shoulder_x_max = 0.085
        _append_capsule(
            parent,
            name=f"{foot_name}_inner_lateral_shoulder",
            fromto=(
                shoulder_x_min,
                center_y - variant.lateral_shoulder_offset,
                shoulder_z,
                shoulder_x_max,
                center_y - variant.lateral_shoulder_offset,
                shoulder_z,
            ),
            radius=variant.shoulder_radius,
            rgba=rgba,
            friction=friction,
        )
        _append_capsule(
            parent,
            name=f"{foot_name}_outer_lateral_shoulder",
            fromto=(
                shoulder_x_min,
                center_y + variant.lateral_shoulder_offset,
                shoulder_z,
                shoulder_x_max,
                center_y + variant.lateral_shoulder_offset,
                shoulder_z,
            ),
            radius=variant.shoulder_radius,
            rgba=rgba,
            friction=friction,
        )


def _find_parent(root: ET.Element, child: ET.Element) -> ET.Element | None:
    """Return the parent element for ``child``."""
    for parent in root.iter():
        if child in list(parent):
            return parent
    return None


def _validate_scene(scene_path: Path) -> None:
    """Ensure MuJoCo can load the generated scene."""
    mujoco.MjModel.from_xml_path(str(scene_path))


def _write_metadata(
    *,
    source_scene: Path,
    generated_scene: Path,
    variant: BlueLikeSoleVariant,
) -> dict[str, Any]:
    """Write metadata for one generated scene."""
    payload = {
        "source_scene": str(source_scene),
        "generated_scene_path": str(generated_scene),
        "foot_geom_names": list(FOOT_GEOM_NAMES),
        "sole_approximation_method": (
            "named flat center box plus toe/heel sagittal rocker capsules and "
            "small lateral shoulder capsules"
        ),
        "sagittal_roll_design": variant.sagittal_roll_design,
        "lateral_roll_design": variant.lateral_roll_design,
        "half_length": 0.07,
        "half_width": 0.04,
        "half_height": 0.025,
        "center_patch_half_length": variant.center_half_length,
        "center_patch_half_width": variant.center_half_width,
        "center_patch_half_height": variant.center_half_height,
        "center_patch_z": BASE_CENTER_Z + variant.center_z_offset,
        "center_patch_bottom_z": (
            BASE_CENTER_Z + variant.center_z_offset - variant.center_half_height
        ),
        "toe_rocker_z": (
            BASE_ROCKER_BOTTOM_Z
            + variant.rocker_z_offset
            + variant.toe_rocker_z_extra_offset
            + variant.rocker_radius
        ),
        "toe_rocker_x_offset": variant.toe_rocker_x_offset,
        "heel_rocker_z": BASE_ROCKER_BOTTOM_Z + variant.rocker_z_offset + variant.rocker_radius,
        "toe_rocker_bottom_z": (
            BASE_ROCKER_BOTTOM_Z + variant.rocker_z_offset + variant.toe_rocker_z_extra_offset
        ),
        "heel_rocker_bottom_z": BASE_ROCKER_BOTTOM_Z + variant.rocker_z_offset,
        "toe_rocker_center_bottom_relative_height": (
            BASE_ROCKER_BOTTOM_Z
            + variant.rocker_z_offset
            + variant.toe_rocker_z_extra_offset
            - (BASE_CENTER_Z + variant.center_z_offset - variant.center_half_height)
        ),
        "heel_rocker_center_bottom_relative_height": (
            BASE_ROCKER_BOTTOM_Z
            + variant.rocker_z_offset
            - (BASE_CENTER_Z + variant.center_z_offset - variant.center_half_height)
        ),
        "rocker_radius": variant.rocker_radius,
        "shoulder_radius": variant.shoulder_radius,
        "difference_from_original_flat_box": (
            "original single flat box is replaced by a smaller flat center patch; "
            "helper capsule geoms add passive sagittal roll and mild lateral roll shoulders"
        ),
        "variant": asdict(variant),
    }
    metadata_path = generated_scene.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def generate_variants(
    source_scene: Path,
    out_dir: Path,
    *,
    variant_set: str = "base",
) -> list[dict[str, Any]]:
    """Generate conservative Blue-like rounded sole scene variants."""
    source_scene = require_scene(source_scene)
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, Any]] = []
    for variant in _variants(variant_set):
        tree = ET.parse(source_scene)
        _set_meshdir_to_source(tree, source_scene)
        _apply_blue_like_sole(tree.getroot(), variant)
        scene_path = out_dir / variant.scene_filename
        tree.write(scene_path, encoding="utf-8", xml_declaration=True)
        _validate_scene(scene_path)
        payloads.append(
            _write_metadata(
                source_scene=source_scene,
                generated_scene=scene_path,
                variant=variant,
            )
        )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(payloads, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payloads


def write_gap_review(payloads: list[dict[str, Any]], report_path: Path) -> None:
    """Write the Blue/BDX vs Sedon gap review."""
    lines = [
        "# Blue-Like Sedon Gap Review",
        "",
        "Task class: Class C experiment diagnostic. This review does not change reward, train.py, or PPO training.",
        "",
        "## Generated Blue-Like Sole Variants",
        "",
        "| scene | approximation | sagittal roll | lateral roll |",
        "|---|---|---|---|",
    ]
    for payload in payloads:
        lines.append(
            f"| `{Path(payload['generated_scene_path']).name}` | "
            f"{payload['sole_approximation_method']} | "
            f"{payload['sagittal_roll_design']} | "
            f"{payload['lateral_roll_design']} |"
        )

    lines.extend(
        [
            "",
            "## Blue/BDX Dynamic Gait Requirements",
            "",
            "- Rounded sole passive roll: the foot must roll under load without active ankle roll.",
            "- Forward momentum: gait should intentionally carry the body forward rather than freeze into static balance.",
            "- Controlled falling: COM does not need to fully enter a static support polygon before the next step.",
            "- Low-clearance swing: swing foot should clear just enough to avoid scuffing, not hop.",
            "- Swing foot catch: the next foot placement must receive the body before unrecoverable fall.",
            "- Dynamic recovery: hip roll, hip pitch, knee, ankle pitch, and torso motion must recover after catch.",
            "",
            "## Current Sedon Gaps",
            "",
            "- Foot geometry: canonical Sedon uses a single flat box foot, so passive roll is not physically represented.",
            "- Contact behavior: wide flat feet remain stable but do not create Blue-like roll; ellipsoid v1 created roll but lost contact too easily.",
            "- Lateral/forward roll: previous tests mostly exposed lateral unload limits; sagittal forward rocker has not been isolated yet.",
            "- Actuation: Sedon has no active ankle roll, so rounded sole/passive roll must carry that role; hip roll range alone was not enough for static load transfer.",
            "- Reference gait: existing configs include reference-march/shuffle style seeds, but the current objective is not yet a dynamic catch-and-recover Blue-like gait.",
            "",
            "## Tests Not Worth Repeating As Mainline",
            "",
            "- Wide flat foot alone: it improves stance robustness but does not solve passive roll.",
            "- Ellipsoid rounded sole v1: it reduced margin slightly but introduced high no-contact and zero-action instability.",
            "- Pure static support margin: Blue-like gait is not quasi-static single support.",
            "- Pure force-ratio unload: useful as a safety signal, not sufficient for dynamic gait readiness.",
            "- Direct PPO: too expensive before the physical roll/catch preconditions are visible in scripted diagnostics.",
            "",
            "## Fit To Blue-Like Physics",
            "",
            "The new variants are closer to Blue/BDX physical assumptions than both flat-box and full-ellipsoid feet because they keep a stable central patch while adding toe/heel sagittal rocker and mild lateral shoulders. They are still only a contact-shape precondition, not evidence that Sedon can already perform controlled falling or foot catch.",
            "",
            "## Next Metrics",
            "",
            "Use at most these three metrics before reward/PPO work:",
            "",
            "1. Forward roll response under a small scripted forward lean: base x progress and pitch/height recovery without jump/no-contact bursts.",
            "2. Low-clearance swing catch: next foot makes contact before base height/upright crosses failure thresholds.",
            "3. Contact continuity during roll-to-catch: no-contact burst count and duration, not static support margin.",
            "",
            "## Recommendation",
            "",
            "Do not treat both-contact shuffle as success. First verify whether the new Blue-like sole can produce controlled forward roll and catch in scripted diagnostics. If it cannot, reward changes and PPO are premature.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--report-path", type=Path, default=GAP_REVIEW_PATH)
    parser.add_argument(
        "--variant-set",
        choices=("base", "v3_contact_ordering", "v4_tilt_handoff", "v5_toe_drop"),
        default="base",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate scenes and write the gap review."""
    args = build_parser().parse_args(argv)
    payloads = generate_variants(
        args.source_scene,
        args.out_dir,
        variant_set=args.variant_set,
    )
    write_gap_review(payloads, args.report_path)
    print(f"variants: {len(payloads)}")
    print(f"out_dir: {args.out_dir}")
    print(f"report: {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
