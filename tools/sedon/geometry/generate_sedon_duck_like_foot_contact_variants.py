"""Generate Sedon foot-contact prototype scene variants.

This Class C workflow creates artifact-only MJCF variants for contact
diagnostics. It does not modify the private source training scene, train.py,
eval.py, PPO logic, or existing artifacts. Open Duck Mini is used only as a
contact-concept reference; all added patches are low-confidence simulation
prototypes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco

from tools.sedon_debug_common import DEBUG_OUT_DIR, DEFAULT_SCENE_PATH, require_scene


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_PATH = REPO_ROOT / "configs" / "sedon" / "sedon_foot_contact_prototype_profiles.yaml"
DEFAULT_OUT_DIR = DEBUG_OUT_DIR / "foot_contact_variants"
DEFAULT_PLAN_DOC = REPO_ROOT / "docs" / "sedon_foot_contact_prototype_plan.md"
PROTOTYPE_METADATA = {
    "source": "assumption/prototype",
    "confidence": "low",
    "valid_for": "simulation_prototype_only",
}
FOOT_SPECS = {
    "right": {
        "body": "R_link_ankle_pitch",
        "original_geom": "R_foot_collision",
        "prefix": "R",
    },
    "left": {
        "body": "L_link_ankle_pitch",
        "original_geom": "L_foot_collision",
        "prefix": "L",
    },
}


@dataclass(frozen=True)
class FootGeom:
    """Resolved source foot collision geometry fields.

    Args:
        side: Semantic side label, ``left`` or ``right``.
        body_name: MJCF body containing the foot geom.
        geom_name: Original Sedon foot collision geom name.
        pos: Local geom position.
        size: MuJoCo box half-size.
        friction: MJCF friction triplet.
    """

    side: str
    body_name: str
    geom_name: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    friction: str

    @property
    def length(self) -> float:
        """Return full local-x foot length in meters."""

        return 2.0 * self.size[0]

    @property
    def width(self) -> float:
        """Return full local-y foot width in meters."""

        return 2.0 * self.size[1]

    @property
    def bottom_z(self) -> float:
        """Return local bottom z of the source collision box."""

        return self.pos[2] - self.size[2]


def utc_now_iso() -> str:
    """Return a compact UTC timestamp for traceability."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_profile(path: Path) -> dict[str, Any]:
    """Load the JSON-compatible YAML profile file.

    Args:
        path: Profile path.

    Returns:
        Parsed profile payload.

    Raises:
        ValueError: If the file is missing or malformed.
    """

    if not path.is_file():
        raise ValueError(f"Profile file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("profiles"), list):
        raise ValueError(f"Profile must contain a profiles list: {path}")
    return payload


def parse_vec3(raw_value: str, *, field_name: str) -> tuple[float, float, float]:
    """Parse an MJCF vec3 string.

    Args:
        raw_value: Attribute value such as ``"0.07 0.04 0.025"``.
        field_name: Field name used in error messages.

    Returns:
        Three floats.

    Raises:
        ValueError: If the value is not exactly three numeric parts.
    """

    parts = raw_value.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"{field_name} must contain exactly three numbers.")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def fmt_vec(values: tuple[float, float, float]) -> str:
    """Format an MJCF vec3 compactly."""

    return " ".join(f"{value:.9g}" for value in values)


def find_required_body(root: ET.Element, body_name: str) -> ET.Element:
    """Find a required body by name."""

    body = root.find(f".//body[@name='{body_name}']")
    if body is None:
        raise ValueError(f"Source scene has no body named {body_name!r}.")
    return body


def find_required_geom(root: ET.Element, geom_name: str) -> ET.Element:
    """Find a required geom by name."""

    geom = root.find(f".//geom[@name='{geom_name}']")
    if geom is None:
        raise ValueError(f"Source scene has no geom named {geom_name!r}.")
    return geom


def resolve_source_feet(root: ET.Element) -> dict[str, FootGeom]:
    """Resolve current Sedon foot collision boxes from the source scene."""

    feet: dict[str, FootGeom] = {}
    for side, spec in FOOT_SPECS.items():
        find_required_body(root, spec["body"])
        geom = find_required_geom(root, spec["original_geom"])
        if geom.attrib.get("type") != "box":
            raise ValueError(f"{spec['original_geom']} must be a box for normalized patch estimation.")
        pos = parse_vec3(geom.attrib.get("pos", ""), field_name=f"{spec['original_geom']}.pos")
        size = parse_vec3(geom.attrib.get("size", ""), field_name=f"{spec['original_geom']}.size")
        if min(size) <= 0.0:
            raise ValueError(f"{spec['original_geom']} size values must be positive.")
        feet[side] = FootGeom(
            side=side,
            body_name=spec["body"],
            geom_name=spec["original_geom"],
            pos=pos,
            size=size,
            friction=geom.attrib.get("friction", "1.0 0.005 0.0001"),
        )
    return feet


def foot_length_status(feet: dict[str, FootGeom]) -> tuple[bool, str]:
    """Return whether source foot length is reliable enough for patch placement."""

    lengths = [foot.length for foot in feet.values()]
    widths = [foot.width for foot in feet.values()]
    if len(lengths) != 2 or any(value <= 0.0 for value in lengths + widths):
        return False, "manual_required: missing or non-positive foot size."
    relative_delta = abs(lengths[0] - lengths[1]) / max(lengths)
    if relative_delta > 0.05:
        return False, "manual_required: left/right foot length mismatch exceeds 5%."
    return True, "resolved_from_sedon_foot_collision_box_size"


def mark_original_foot_non_contact(root: ET.Element) -> None:
    """Disable contact on the original foot collision geoms in generated variants."""

    for spec in FOOT_SPECS.values():
        geom = find_required_geom(root, spec["original_geom"])
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
        geom.set("group", "1")
        geom.set("rgba", "0.12 0.12 0.12 0.12")


def patch_name(side: str, profile_name: str, region: str) -> str:
    """Build a stable prototype patch geom name."""

    prefix = FOOT_SPECS[side]["prefix"]
    return f"{prefix}_foot_contact_{profile_name}_{region}"


def patch_position(foot: FootGeom, patch: dict[str, Any]) -> tuple[float, float, float]:
    """Compute local patch center from normalized foot ratios."""

    normalized_x = float(patch["normalized_x"])
    normalized_y = float(patch.get("normalized_y", 0.0))
    size_scale = patch["size_scale"]
    half_z = max(0.003, foot.size[2] * float(size_scale[2]))
    return (
        foot.pos[0] + normalized_x * foot.size[0],
        foot.pos[1] + normalized_y * foot.size[1],
        foot.bottom_z + half_z,
    )


def patch_size(foot: FootGeom, patch: dict[str, Any]) -> tuple[float, float, float]:
    """Compute prototype patch half-size from source foot size."""

    scale = patch["size_scale"]
    if not isinstance(scale, list) or len(scale) != 3:
        raise ValueError("patch size_scale must contain three values.")
    return (
        max(0.004, foot.size[0] * float(scale[0])),
        max(0.004, foot.size[1] * float(scale[1])),
        max(0.003, foot.size[2] * float(scale[2])),
    )


def add_patch_metadata(geom: ET.Element, profile_name: str, region: str) -> None:
    """Attach MJCF-valid prototype identifiers to a geom.

    Detailed source/confidence/valid_for metadata is written to the manifest
    because MuJoCo rejects arbitrary geom attributes.
    """

    geom.set("user", "1")
    geom.set("group", "3")


def add_profile_patches(
    root: ET.Element,
    *,
    profile: dict[str, Any],
    feet: dict[str, FootGeom],
) -> list[dict[str, Any]]:
    """Add prototype patches for one profile to an MJCF root."""

    profile_name = str(profile["name"])
    added: list[dict[str, Any]] = []
    for side, foot in feet.items():
        body = find_required_body(root, foot.body_name)
        for patch in profile.get("patches", []):
            region = str(patch["region"])
            geom = ET.SubElement(
                body,
                "geom",
                {
                    "name": patch_name(side, profile_name, region),
                    "type": "box",
                    "pos": fmt_vec(patch_position(foot, patch)),
                    "size": fmt_vec(patch_size(foot, patch)),
                    "rgba": "0.05 0.45 0.95 0.35",
                    "friction": foot.friction,
                    "margin": "0",
                    "gap": "0",
                },
            )
            add_patch_metadata(geom, profile_name, region)
            added.append(
                {
                    "side": side,
                    "region": region,
                    "geom_name": geom.attrib["name"],
                    "body_name": foot.body_name,
                    "pos": geom.attrib["pos"],
                    "size": geom.attrib["size"],
                    **PROTOTYPE_METADATA,
                }
            )
    return added


def compile_scene(path: Path) -> dict[str, Any]:
    """Compile a generated scene and return compact model metadata."""

    model = mujoco.MjModel.from_xml_path(str(path))
    return {
        "compiled": True,
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "njnt": int(model.njnt),
        "nu": int(model.nu),
    }


def rewrite_meshdir_for_output(root: ET.Element, source_scene: Path, output_scene: Path) -> None:
    """Make compiler meshdir valid from the generated scene location."""

    compiler = root.find("compiler")
    if compiler is None:
        return
    meshdir = compiler.attrib.get("meshdir")
    if not meshdir:
        return
    source_meshdir = (source_scene.parent / meshdir).resolve()
    relative_meshdir = os.path.relpath(source_meshdir, output_scene.parent.resolve())
    compiler.set("meshdir", relative_meshdir)


def write_variant_scene(
    source_scene: Path,
    out_dir: Path,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Generate one variant scene and return manifest metadata."""

    tree = ET.parse(source_scene)
    root = tree.getroot()
    feet = resolve_source_feet(root)
    length_ok, length_reason = foot_length_status(feet)
    profile_name = str(profile["name"])
    variant_dir = out_dir / profile_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    scene_path = variant_dir / f"training_scene_{profile_name}.xml"
    rewrite_meshdir_for_output(root, source_scene, scene_path)
    added_patches: list[dict[str, Any]] = []
    status = "generated"

    if profile.get("patches") and not length_ok and profile.get("manual_required_if_foot_length_unknown", True):
        status = "manual_required"
    else:
        if bool(profile.get("disable_original_foot_collision", False)):
            mark_original_foot_non_contact(root)
        added_patches = add_profile_patches(root, profile=profile, feet=feet)

    root.set("model", f"sedon_{profile_name}")
    ET.indent(tree, space="  ")
    tree.write(scene_path, encoding="utf-8", xml_declaration=True)
    compile_result = compile_scene(scene_path) if status == "generated" else {"compiled": False}
    return {
        "name": profile_name,
        "status": status,
        "scene_path": str(scene_path),
        "source_scene_path": str(source_scene),
        "does_not_modify_source_scene": True,
        "disable_original_foot_collision": bool(profile.get("disable_original_foot_collision", False)),
        "foot_length_status": length_reason,
        "prototype_metadata": PROTOTYPE_METADATA,
        "added_patches": added_patches,
        "compile_result": compile_result,
        "description": profile.get("description", ""),
    }


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Write variant manifest JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_plan_doc(path: Path, manifest: dict[str, Any]) -> None:
    """Write the prototype plan markdown."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sedon Foot Contact Prototype Plan",
        "",
        "Task class: Class C experiment workflow. This plan creates artifact-only contact prototypes for MuJoCo diagnostics.",
        "",
        "## Safety Constraints",
        "",
        "- Does not modify `private_assets/sedon/training_scene.xml`.",
        "- Does not modify `sedon_baseline/train.py` or `sedon_baseline/eval.py`.",
        "- Does not delete or move artifacts.",
        "- Does not enter PPO and does not claim walking success.",
        "- Open Duck Mini foot geometry is concept/reference only, not Sedon verified geometry.",
        "- Every added patch is tagged `source=assumption/prototype`, `confidence=low`, `valid_for=simulation_prototype_only`.",
        "",
        "## Variants",
        "",
        "| variant | status | original foot collision | added patches | purpose |",
        "|---|---|---|---:|---|",
    ]
    for variant in manifest["variants"]:
        original = "kept" if not variant["disable_original_foot_collision"] else "non-contact in generated XML"
        lines.append(
            f"| `{variant['name']}` | `{variant['status']}` | {original} | "
            f"{len(variant['added_patches'])} | {variant['description']} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            "The comparison workflow evaluates neutral contact, forward pitch contact, backward pitch contact, left/right support symmetry, raw contact pairs, and contact patch classification.",
            "",
            "Required metrics: `can_classify_center_toe_heel`, `neutral_center_first_left/right`, `forward_pitch_toe_handoff_candidate_left/right`, `toe_heel_bridge_contact_detected_left/right`, `left_right_symmetry`, `contact_model_sufficient_for_rollover_analysis`, and `recommendation_to_mechanical_team`.",
            "",
            "## Trade-off",
            "",
            "The prototype disables original foot collision only inside generated variant XMLs so patch-level contacts can be classified. This improves diagnostic clarity, but it means the variants are not physical Sedon geometry and must stay simulation-prototype-only.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--plan-doc", type=Path, default=DEFAULT_PLAN_DOC)
    return parser.parse_args()


def main() -> int:
    """Generate artifact-only Sedon foot contact prototype variants."""

    args = parse_args()
    source_scene = require_scene(args.source_scene)
    profiles = load_profile(args.profiles)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_copy = args.out_dir / "source_training_scene_snapshot.xml"
    if not source_copy.exists():
        shutil.copy2(source_scene, source_copy)

    variants = [
        write_variant_scene(source_scene, args.out_dir, profile)
        for profile in profiles["profiles"]
    ]
    manifest = {
        "schema_version": 1,
        "generated_at": utc_now_iso(),
        "source_scene_path": str(source_scene),
        "source_scene_snapshot": str(source_copy),
        "profile_path": str(args.profiles),
        "duck_usage": "concept_reference_only_not_verified_geometry",
        "safety": profiles.get("safety", {}),
        "prototype_metadata": PROTOTYPE_METADATA,
        "variants": variants,
    }
    write_manifest(args.out_dir / "manifest.json", manifest)
    write_plan_doc(args.plan_doc, manifest)
    print(f"variants={len(variants)}")
    print(f"manifest={args.out_dir / 'manifest.json'}")
    print(f"plan={args.plan_doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
