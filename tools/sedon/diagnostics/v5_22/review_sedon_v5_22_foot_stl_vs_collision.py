"""Review Sedon v5_22 foot STL geometry against MuJoCo foot collision.

This Class C diagnostic is read-only with respect to source robot assets. It
loads STL vertex geometry, summarizes likely foot-bottom shape, compares it
against current MJCF collision primitives, and writes review artifacts. It does
not use STL meshes as final collision, does not edit XML/URDF/train/eval/env,
does not run PPO, and does not claim walking success.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "sedon" / "sedon_v5_22_foot_stl_review.yaml"
SIDES = ("left", "right")


@dataclass(frozen=True)
class Bounds:
    """Axis-aligned bounding box.

    Args:
        min_xyz: Minimum x/y/z coordinates.
        max_xyz: Maximum x/y/z coordinates.
    """

    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]

    @property
    def dims(self) -> tuple[float, float, float]:
        """Return box dimensions in x/y/z."""

        return tuple(self.max_xyz[index] - self.min_xyz[index] for index in range(3))  # type: ignore[return-value]

    @property
    def center(self) -> tuple[float, float, float]:
        """Return box center."""

        return tuple((self.min_xyz[index] + self.max_xyz[index]) * 0.5 for index in range(3))  # type: ignore[return-value]


@dataclass(frozen=True)
class StlSummary:
    """Computed STL geometry summary."""

    path: Path
    side: str
    role: str
    vertex_count: int
    triangle_count: int
    bounds: Bounds
    bottom: dict[str, Any]
    interpretation: dict[str, Any]


@dataclass(frozen=True)
class CollisionGeom:
    """Current MuJoCo collision geometry summary."""

    side: str
    body_name: str
    name: str
    geom_type: str
    pos: tuple[float, float, float]
    size: tuple[float, ...]
    bounds: Bounds | None
    friction: str | None


def rel_path(path: Path) -> str:
    """Return a repository-relative path string when possible."""

    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def resolve_repo_path(raw_path: str) -> Path:
    """Resolve a repository-relative path."""

    path = Path(raw_path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON-compatible YAML file.

    Args:
        path: Config path.

    Returns:
        Parsed object.

    Raises:
        FileNotFoundError: If config is missing.
        ValueError: If decoded config is not an object.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must decode to a JSON object: {path}")
    return payload


def parse_vec(raw_value: str, field_name: str) -> tuple[float, ...]:
    """Parse an MJCF numeric vector field."""

    parts = raw_value.replace(",", " ").split()
    if not parts:
        raise ValueError(f"{field_name} is empty.")
    return tuple(float(part) for part in parts)


def bounds_from_vertices(vertices: list[tuple[float, float, float]]) -> Bounds:
    """Build bounds from STL vertices."""

    if not vertices:
        raise ValueError("Cannot compute bounds from an empty vertex list.")
    mins = [min(vertex[index] for vertex in vertices) for index in range(3)]
    maxs = [max(vertex[index] for vertex in vertices) for index in range(3)]
    return Bounds(tuple(mins), tuple(maxs))  # type: ignore[arg-type]


def parse_ascii_stl(text: str) -> list[tuple[float, float, float]]:
    """Parse vertex lines from an ASCII STL payload."""

    vertices: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return vertices


def parse_binary_stl(data: bytes) -> list[tuple[float, float, float]]:
    """Parse vertices from a binary STL payload."""

    if len(data) < 84:
        raise ValueError("Binary STL is shorter than the 84 byte header.")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected_size = 84 + triangle_count * 50
    if expected_size > len(data):
        raise ValueError(f"Binary STL is truncated: expected {expected_size} bytes, got {len(data)}.")
    vertices: list[tuple[float, float, float]] = []
    offset = 84
    for _ in range(triangle_count):
        values = struct.unpack_from("<12fH", data, offset)
        vertices.extend(
            [
                (values[3], values[4], values[5]),
                (values[6], values[7], values[8]),
                (values[9], values[10], values[11]),
            ]
        )
        offset += 50
    return vertices


def read_stl_vertices(path: Path) -> tuple[list[tuple[float, float, float]], str]:
    """Read vertices from ASCII or binary STL.

    Args:
        path: STL path.

    Returns:
        A tuple of vertices and detected format.

    Raises:
        ValueError: If no vertices can be parsed.
    """

    data = path.read_bytes()
    vertices: list[tuple[float, float, float]] = []
    detected_format = "binary"
    try:
        text = data.decode("utf-8")
        vertices = parse_ascii_stl(text)
        if vertices:
            detected_format = "ascii"
    except UnicodeDecodeError:
        vertices = []
    if not vertices:
        vertices = parse_binary_stl(data)
        detected_format = "binary"
    if not vertices:
        raise ValueError(f"No STL vertices parsed: {path}")
    return vertices, detected_format


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    """Return existing paths without duplicates, preserving order."""

    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        result.append(path)
    return result


def expand_search_roots(raw_roots: list[str]) -> list[Path]:
    """Expand configured candidate search roots."""

    roots: list[Path] = []
    for raw_root in raw_roots:
        if "*" in raw_root:
            roots.extend(path for path in REPO_ROOT.glob(raw_root) if path.is_dir())
            continue
        root = resolve_repo_path(raw_root)
        if root.is_dir():
            roots.append(root)
    return unique_paths(roots)


def infer_side(path: Path, side_tokens: dict[str, list[str]]) -> str:
    """Infer semantic side from a file name."""

    name = path.name.lower()
    for side, tokens in side_tokens.items():
        for token in tokens:
            if token.lower() in name:
                return side
    return "unknown"


def infer_role(path: Path) -> str:
    """Infer foot-related role from file name."""

    name = path.name.lower()
    if "bottom" in name or "sole" in name or "tpu" in name:
        return "foot_bottom"
    if "toe" in name:
        return "toe"
    if "heel" in name:
        return "heel"
    if "ankle_pitch" in name:
        return "ankle_pitch_visual_candidate"
    if "foot" in name:
        return "foot_visual_candidate"
    return "foot_related_candidate"


def find_foot_stls(config: dict[str, Any]) -> list[Path]:
    """Find configured or auto-discovered foot-related STL files."""

    explicit = [resolve_repo_path(raw_path) for raw_path in config.get("explicit_stl_paths", [])]
    roots = expand_search_roots(list(config.get("candidate_search_roots", [])))
    tokens = [str(token).lower() for token in config.get("foot_related_name_tokens", [])]
    discovered: list[Path] = []
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".stl":
                name = path.name.lower()
                if any(token in name for token in tokens):
                    discovered.append(path)
    return unique_paths([*explicit, *discovered])


def x_bin_minima(
    vertices: list[tuple[float, float, float]],
    bounds: Bounds,
    x_bins: int,
) -> list[dict[str, Any]]:
    """Compute minimum z per x bin."""

    min_x, max_x = bounds.min_xyz[0], bounds.max_xyz[0]
    span = max(max_x - min_x, 1e-12)
    bins: list[list[float]] = [[] for _ in range(x_bins)]
    for x, _, z in vertices:
        index = min(x_bins - 1, max(0, int((x - min_x) / span * x_bins)))
        bins[index].append(z)
    rows: list[dict[str, Any]] = []
    for index, values in enumerate(bins):
        start = min_x + span * index / x_bins
        end = min_x + span * (index + 1) / x_bins
        rows.append(
            {
                "bin": index,
                "x_start": start,
                "x_end": end,
                "min_z": min(values) if values else None,
                "sample_count": len(values),
            }
        )
    return rows


def summarize_bottom(
    vertices: list[tuple[float, float, float]],
    bounds: Bounds,
    settings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Summarize lowest-z vertex distribution and bottom-shape candidates."""

    tolerance = float(settings["bottom_vertex_tolerance"])
    x_bins = int(settings["x_bins"])
    min_z = bounds.min_xyz[2]
    bottom_vertices = [vertex for vertex in vertices if vertex[2] <= min_z + tolerance]
    bottom_bounds = bounds_from_vertices(bottom_vertices) if bottom_vertices else None
    bin_rows = x_bin_minima(vertices, bounds, x_bins)
    valid_minima = [row["min_z"] for row in bin_rows if row["min_z"] is not None]
    z_range = max(valid_minima) - min(valid_minima) if valid_minima else math.nan
    front_min_z = valid_minima[-1] if valid_minima else math.nan
    rear_min_z = valid_minima[0] if valid_minima else math.nan
    center_min_z = valid_minima[len(valid_minima) // 2] if valid_minima else math.nan
    dims = bounds.dims
    flat_threshold = float(settings["flat_z_range_threshold"])
    curved_threshold = float(settings["curved_z_range_threshold"])
    protrusion_ratio = float(settings["protrusion_ratio_threshold"])
    bottom_x_span = bottom_bounds.dims[0] if bottom_bounds else 0.0
    bottom_y_span = bottom_bounds.dims[1] if bottom_bounds else 0.0
    bottom_span_ratio = bottom_x_span / dims[0] if dims[0] > 0 else 0.0
    front_extent_ratio = (bounds.max_xyz[0] - (bottom_bounds.max_xyz[0] if bottom_bounds else bounds.max_xyz[0])) / dims[0] if dims[0] > 0 else 0.0
    rear_extent_ratio = ((bottom_bounds.min_xyz[0] if bottom_bounds else bounds.min_xyz[0]) - bounds.min_xyz[0]) / dims[0] if dims[0] > 0 else 0.0
    flat_bottom = bool(z_range <= flat_threshold)
    curved_bottom = bool(z_range >= curved_threshold)
    rocker_like = bool(curved_bottom and center_min_z <= front_min_z and center_min_z <= rear_min_z)
    continuous_bottom = bool(bottom_span_ratio >= 0.55 or flat_bottom)
    discrete_patch = bool(bottom_span_ratio < 0.35 and not flat_bottom)
    toe_area = bool(bottom_bounds is not None and (bounds.max_xyz[0] - bottom_bounds.max_xyz[0]) / max(dims[0], 1e-12) <= protrusion_ratio)
    heel_area = bool(bottom_bounds is not None and (bottom_bounds.min_xyz[0] - bounds.min_xyz[0]) / max(dims[0], 1e-12) <= protrusion_ratio)
    bottom = {
        "lowest_z": min_z,
        "bottom_vertex_tolerance": tolerance,
        "bottom_vertex_count": len(bottom_vertices),
        "bottom_vertex_ratio": len(bottom_vertices) / max(len(vertices), 1),
        "bottom_bbox": bounds_payload(bottom_bounds) if bottom_bounds else None,
        "x_bin_min_z": bin_rows,
        "x_min_z_range": z_range,
        "front_min_z": front_min_z,
        "center_min_z": center_min_z,
        "rear_min_z": rear_min_z,
        "front_extent": bounds.max_xyz[0],
        "rear_extent": bounds.min_xyz[0],
        "bottom_x_span_ratio": bottom_span_ratio,
        "bottom_y_span": bottom_y_span,
        "front_non_bottom_extent_ratio": front_extent_ratio,
        "rear_non_bottom_extent_ratio": rear_extent_ratio,
    }
    interpretation = {
        "flat_bottom_candidate": flat_bottom,
        "rocker_like_candidate": rocker_like,
        "continuous_bottom_candidate": continuous_bottom,
        "discrete_patch_candidate": discrete_patch,
        "toe_contact_area_candidate": toe_area,
        "heel_contact_area_candidate": heel_area,
        "bottom_shape": "flat" if flat_bottom else ("rocker" if rocker_like else ("curved" if curved_bottom else "unknown")),
        "heuristic_source": "stl_vertex_lowest_z_distribution",
        "source": "assumption",
        "confidence": "low",
    }
    return bottom, interpretation


def summarize_stl(path: Path, config: dict[str, Any]) -> StlSummary:
    """Build a summary for one STL file."""

    vertices, _ = read_stl_vertices(path)
    bounds = bounds_from_vertices(vertices)
    bottom, interpretation = summarize_bottom(vertices, bounds, config["stl_bottom_analysis"])
    return StlSummary(
        path=path,
        side=infer_side(path, config["side_name_tokens"]),
        role=infer_role(path),
        vertex_count=len(vertices),
        triangle_count=len(vertices) // 3,
        bounds=bounds,
        bottom=bottom,
        interpretation=interpretation,
    )


def bounds_payload(bounds: Bounds | None) -> dict[str, Any] | None:
    """Serialize bounds."""

    if bounds is None:
        return None
    dims = bounds.dims
    return {
        "min": list(bounds.min_xyz),
        "max": list(bounds.max_xyz),
        "center": list(bounds.center),
        "dimensions": {"x": dims[0], "y": dims[1], "z": dims[2]},
        "length_x": dims[0],
        "width_y": dims[1],
        "height_z": dims[2],
    }


def summarize_collision_geoms(model_path: Path, config: dict[str, Any]) -> list[CollisionGeom]:
    """Read current MuJoCo foot collision geoms from MJCF XML."""

    root = ET.parse(model_path).getroot()
    collision_names: dict[str, str] = config["collision_geom_names"]
    geoms: list[CollisionGeom] = []
    for side in SIDES:
        name = collision_names[side]
        geom = root.find(f".//geom[@name='{name}']")
        if geom is None:
            geoms.append(CollisionGeom(side, "unknown", name, "missing", (math.nan, math.nan, math.nan), (), None, None))
            continue
        parent_body = find_parent_body(root, geom)
        geom_type = geom.attrib.get("type", "sphere")
        pos_raw = geom.attrib.get("pos", "0 0 0")
        size_raw = geom.attrib.get("size", "")
        pos = parse_vec(pos_raw, f"{name}.pos")
        if len(pos) != 3:
            raise ValueError(f"{name}.pos must have three values.")
        size = parse_vec(size_raw, f"{name}.size")
        bounds = collision_bounds(geom_type, pos, size)
        geoms.append(
            CollisionGeom(
                side=side,
                body_name=parent_body.attrib.get("name", "unknown") if parent_body is not None else "unknown",
                name=name,
                geom_type=geom_type,
                pos=(pos[0], pos[1], pos[2]),
                size=size,
                bounds=bounds,
                friction=geom.attrib.get("friction"),
            )
        )
    return geoms


def find_parent_body(root: ET.Element, target: ET.Element) -> ET.Element | None:
    """Find the parent body for a geom element."""

    for body in root.iter("body"):
        if target in list(body):
            return body
    return None


def collision_bounds(geom_type: str, pos: tuple[float, float, float], size: tuple[float, ...]) -> Bounds | None:
    """Derive a local collision bbox for simple MuJoCo geom types."""

    if geom_type == "box" and len(size) >= 3:
        half = (size[0], size[1], size[2])
    elif geom_type in {"ellipsoid", "sphere"} and len(size) >= 3:
        half = (size[0], size[1], size[2])
    elif geom_type == "capsule" and len(size) >= 2:
        half = (size[1] + size[0], size[0], size[0])
    else:
        return None
    return Bounds(
        tuple(pos[index] - half[index] for index in range(3)),  # type: ignore[arg-type]
        tuple(pos[index] + half[index] for index in range(3)),  # type: ignore[arg-type]
    )


def choose_side_stl(summaries: list[StlSummary], side: str) -> StlSummary | None:
    """Choose the best available visual STL for a side."""

    candidates = [summary for summary in summaries if summary.side == side]
    if not candidates:
        return None
    role_priority = {
        "foot_bottom": 0,
        "foot_visual_candidate": 1,
        "ankle_pitch_visual_candidate": 2,
        "toe": 3,
        "heel": 4,
        "foot_related_candidate": 5,
    }
    candidates.sort(key=lambda summary: (role_priority.get(summary.role, 9), path_priority(summary.path), len(rel_path(summary.path))))
    return candidates[0]


def path_priority(path: Path) -> int:
    """Prefer v5_22 assets over older duplicate Sedon assets."""

    normalized = rel_path(path).lower()
    if "sedon_v5_22" in normalized:
        return 0
    if "seedon_urdf_5_22" in normalized:
        return 1
    return 2


def compare_bounds(stl: StlSummary | None, collision: CollisionGeom, thresholds: dict[str, Any]) -> dict[str, Any]:
    """Compare STL and collision bboxes for one side."""

    if stl is None or collision.bounds is None:
        return {
            "side": collision.side,
            "stl_path": rel_path(stl.path) if stl else None,
            "collision_geom": collision.name,
            "status": "insufficient_data",
            "mismatch_score": None,
            "collision_over_simplified": "unknown",
        }
    stl_dims = stl.bounds.dims
    collision_dims = collision.bounds.dims
    dimension_errors: list[float] = []
    for index in range(3):
        denominator = max(abs(stl_dims[index]), abs(collision_dims[index]), 1e-12)
        dimension_errors.append(abs(stl_dims[index] - collision_dims[index]) / denominator)
    center_offsets: list[float] = []
    for index in range(3):
        denominator = max(abs(stl_dims[index]), abs(collision_dims[index]), 1e-12)
        center_offsets.append(abs(stl.bounds.center[index] - collision.bounds.center[index]) / denominator)
    primitive_penalty = 0.25 if collision.geom_type == "box" and stl.interpretation["bottom_shape"] in {"curved", "rocker"} else 0.0
    mismatch_score = min(1.0, sum(dimension_errors) / 3.0 * 0.65 + sum(center_offsets) / 3.0 * 0.35 + primitive_penalty)
    over_simplified = bool(
        mismatch_score >= float(thresholds["score_over_simplified"])
        or primitive_penalty > 0.0
        or max(dimension_errors) >= float(thresholds["bbox_dimension_ratio_warn"])
    )
    return {
        "side": collision.side,
        "stl_path": rel_path(stl.path),
        "stl_role": stl.role,
        "collision_geom": collision.name,
        "collision_type": collision.geom_type,
        "visual_stl_bbox": bounds_payload(stl.bounds),
        "collision_bbox": bounds_payload(collision.bounds),
        "dimension_relative_errors": {
            "x": dimension_errors[0],
            "y": dimension_errors[1],
            "z": dimension_errors[2],
        },
        "center_offset_relative": {
            "x": center_offsets[0],
            "y": center_offsets[1],
            "z": center_offsets[2],
        },
        "mismatch_score": mismatch_score,
        "collision_over_simplified": over_simplified,
        "status": "compared",
        "source": "assumption",
        "confidence": "low",
    }


def aggregate_conclusion(comparisons: list[dict[str, Any]], summaries: list[StlSummary]) -> dict[str, Any]:
    """Build engineering conclusion from geometry and collision comparison."""

    compared = [row for row in comparisons if row["status"] == "compared"]
    bottom_specific_roles = {"foot_bottom", "foot_visual_candidate", "toe", "heel"}
    bottom_specific = [summary for summary in summaries if summary.side in SIDES and summary.role in bottom_specific_roles]
    side_summaries = bottom_specific if bottom_specific else [summary for summary in summaries if summary.side in SIDES]
    bottom_shapes = {summary.interpretation["bottom_shape"] for summary in side_summaries}
    has_curved_or_rocker = bool(bottom_shapes & {"curved", "rocker"})
    has_flat = "flat" in bottom_shapes
    over_simplified_values = [row["collision_over_simplified"] for row in compared]
    over_simplified = any(value is True for value in over_simplified_values) if over_simplified_values else "unknown"
    collision_needs_update: bool | str
    mechanical_redesign_needed: bool | str
    if not bottom_specific:
        collision_needs_update = "unknown"
        mechanical_redesign_needed = "unknown"
        physical_rollover: bool | str = "unknown"
        next_step = "request_cad_step_or_bottom_specific_mesh"
    elif has_curved_or_rocker and over_simplified is True:
        collision_needs_update = True
        mechanical_redesign_needed = False
        physical_rollover = True
        next_step = "update_mujoco_collision_first"
    elif has_flat and over_simplified is False:
        collision_needs_update = False
        mechanical_redesign_needed = True
        physical_rollover = False
        next_step = "mechanical_foot_bottom_redesign_discussion"
    else:
        collision_needs_update = "unknown"
        mechanical_redesign_needed = "unknown"
        physical_rollover = True if has_curved_or_rocker else ("unknown" if not has_flat else False)
        next_step = "request_cad_step_or_bottom_specific_mesh"
    return {
        "collision_over_simplified": over_simplified,
        "physical_foot_may_support_rollover": physical_rollover,
        "MuJoCo_collision_needs_update": collision_needs_update,
        "mechanical_redesign_needed": mechanical_redesign_needed,
        "next_step": next_step,
        "bottom_specific_mesh_found": bool(bottom_specific),
        "do_not_claim_walking_success": True,
        "do_not_use_stl_as_final_collision": True,
    }


def recommendation(conclusion: dict[str, Any], summaries: list[StlSummary]) -> list[str]:
    """Generate action recommendations."""

    bottom_specific_roles = {"foot_bottom", "foot_visual_candidate", "toe", "heel"}
    bottom_specific = [summary for summary in summaries if summary.side in SIDES and summary.role in bottom_specific_roles]
    if not bottom_specific:
        return [
            "Request CAD/STEP or a bottom-specific mesh because the discovered STL files are ankle-pitch visual candidates, not explicit sole/bottom geometry.",
            "Do not directly use the STL mesh as final MuJoCo collision.",
            "Use mechanical review to decide whether to update collision or redesign the foot bottom.",
        ]
    shapes = {summary.interpretation["bottom_shape"] for summary in bottom_specific}
    if bool({"curved", "rocker"} & shapes) and conclusion["collision_over_simplified"] is True:
        return [
            "Update MuJoCo collision first using a simplified reviewed collision proxy, not raw STL as final collision.",
            "Rerun contact persistence diagnostics only after the collision proxy represents the visual bottom better.",
            "Keep STL/CAD review separate from PPO; do not claim walking success from this result.",
        ]
    if shapes == {"flat"} or ("flat" in shapes and len(shapes) == 1):
        return [
            "Treat the physical foot bottom as likely too simple for rollover until mechanical review says otherwise.",
            "Discuss mechanical foot bottom redesign or TPU/sole geometry before more controller tuning.",
            "Request measured bottom dimensions and material friction assumptions.",
        ]
    return [
        "Request CAD/STEP or a bottom-specific mesh because the current STL candidate is not conclusive.",
        "Do not directly use the STL mesh as final MuJoCo collision.",
        "Keep the next diagnostic read-only or artifact-only until mechanical geometry is clarified.",
    ]


def stl_payload(summary: StlSummary) -> dict[str, Any]:
    """Serialize an STL summary."""

    return {
        "path": rel_path(summary.path),
        "side": summary.side,
        "role": summary.role,
        "vertex_count": summary.vertex_count,
        "triangle_count": summary.triangle_count,
        "bbox": bounds_payload(summary.bounds),
        "bottom": summary.bottom,
        "interpretation": summary.interpretation,
    }


def collision_payload(collision: CollisionGeom) -> dict[str, Any]:
    """Serialize a collision geometry summary."""

    return {
        "side": collision.side,
        "body_name": collision.body_name,
        "name": collision.name,
        "type": collision.geom_type,
        "pos": list(collision.pos),
        "size": list(collision.size),
        "friction": collision.friction,
        "bbox": bounds_payload(collision.bounds),
    }


def write_json_yaml(path: Path, payload: Any) -> None:
    """Write deterministic YAML-compatible JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write CSV rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    """Write the markdown review report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    found_rows = "\n".join(
        f"| `{item['side']}` | `{item['role']}` | `{item['path']}` | `{item['interpretation']['bottom_shape']}` | "
        f"{item['bbox']['length_x']:.6g} | {item['bbox']['width_y']:.6g} | {item['bbox']['height_z']:.6g} |"
        for item in payload["stl_summaries"]
    )
    comparison_rows = "\n".join(
        f"| `{row['side']}` | `{row.get('collision_geom')}` | `{row.get('collision_type')}` | "
        f"`{row.get('stl_path')}` | `{row.get('mismatch_score')}` | `{row.get('collision_over_simplified')}` |"
        for row in payload["comparisons"]
    )
    recommendations = "\n".join(f"- {item}" for item in payload["recommendations"])
    conclusion = payload["engineering_conclusion"]
    lines = [
        "# Sedon v5_22 Foot STL vs Collision Review",
        "",
        "Task class: Class C read-only geometry diagnostic. This review does not modify source XML/URDF/train/eval/env, does not run PPO, does not use STL as final collision, and does not claim walking success.",
        "",
        "## Summary",
        "",
        f"- Status: `{payload['status']}`",
        f"- STL candidates found: `{len(payload['stl_summaries'])}`",
        f"- Collision over simplified: `{conclusion['collision_over_simplified']}`",
        f"- Physical foot may support rollover: `{conclusion['physical_foot_may_support_rollover']}`",
        f"- MuJoCo collision needs update: `{conclusion['MuJoCo_collision_needs_update']}`",
        f"- Mechanical redesign needed: `{conclusion['mechanical_redesign_needed']}`",
        f"- Next step: `{conclusion['next_step']}`",
        "",
        "## Found Foot-Related STL",
        "",
        "| side | role | path | bottom shape | length x | width y | height z |",
        "|---|---|---|---|---:|---:|---:|",
        found_rows or "| `none` | `none` | `none` | `unknown` | 0 | 0 | 0 |",
        "",
        "## MuJoCo Collision Comparison",
        "",
        "| side | collision geom | type | compared STL | mismatch score | over simplified |",
        "|---|---|---|---|---:|---|",
        comparison_rows,
        "",
        "## Foot Bottom Interpretation",
        "",
        "- `flat_bottom_candidate`, `rocker_like_candidate`, `continuous_bottom_candidate`, `discrete_patch_candidate`, `toe_contact_area_candidate`, and `heel_contact_area_candidate` are vertex-distribution heuristics.",
        "- Any threshold value marked with `source=assumption` uses `confidence=low` and is valid for this review only.",
        "- `ankle_pitch_visual_candidate` means the mesh is attached to the same ankle-pitch link as the current foot collision, not that it is a bottom-specific CAD surface.",
        "",
        "## Recommendations",
        "",
        recommendations,
        "",
        "## Artifacts",
        "",
        f"- `foot_stl_summary.yaml`: `{payload['artifact_paths']['foot_stl_summary']}`",
        f"- `foot_stl_bbox.csv`: `{payload['artifact_paths']['foot_stl_bbox']}`",
        f"- `collision_comparison.yaml`: `{payload['artifact_paths']['collision_comparison']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_review(config: dict[str, Any]) -> dict[str, Any]:
    """Build the full read-only STL vs collision review payload."""

    model_path = resolve_repo_path(config["model_path"])
    if not model_path.is_file():
        raise FileNotFoundError(f"Model XML not found: {model_path}")
    stl_paths = find_foot_stls(config)
    summaries = [summarize_stl(path, config) for path in stl_paths]
    collisions = summarize_collision_geoms(model_path, config)
    comparisons = [
        compare_bounds(choose_side_stl(summaries, collision.side), collision, config["mismatch_thresholds"])
        for collision in collisions
    ]
    conclusion = aggregate_conclusion(comparisons, summaries)
    recs = recommendation(conclusion, summaries)
    artifacts_dir = resolve_repo_path(config["artifacts_dir"])
    artifact_paths = {
        "foot_stl_summary": rel_path(artifacts_dir / "foot_stl_summary.yaml"),
        "foot_stl_bbox": rel_path(artifacts_dir / "foot_stl_bbox.csv"),
        "collision_comparison": rel_path(artifacts_dir / "collision_comparison.yaml"),
    }
    return {
        "schema_version": 1,
        "status": "FOOT_STL_VS_COLLISION_REVIEW_COMPLETE",
        "version": config["version"],
        "model_path": rel_path(model_path),
        "valid_for": config["valid_for"],
        "invalid_for": config["invalid_for"],
        "search_roots": [rel_path(path) for path in expand_search_roots(config["candidate_search_roots"])],
        "stl_summaries": [stl_payload(summary) for summary in summaries],
        "collision_geoms": [collision_payload(collision) for collision in collisions],
        "comparisons": comparisons,
        "engineering_conclusion": conclusion,
        "recommendations": recs,
        "artifact_paths": artifact_paths,
    }


def write_artifacts(payload: dict[str, Any], config: dict[str, Any]) -> None:
    """Write YAML-compatible JSON, CSV, and markdown outputs."""

    artifacts_dir = resolve_repo_path(config["artifacts_dir"])
    write_json_yaml(artifacts_dir / "foot_stl_summary.yaml", payload)
    bbox_rows: list[dict[str, Any]] = []
    for summary in payload["stl_summaries"]:
        bbox = summary["bbox"]
        bbox_rows.append(
            {
                "side": summary["side"],
                "role": summary["role"],
                "path": summary["path"],
                "min_x": bbox["min"][0],
                "min_y": bbox["min"][1],
                "min_z": bbox["min"][2],
                "max_x": bbox["max"][0],
                "max_y": bbox["max"][1],
                "max_z": bbox["max"][2],
                "length_x": bbox["length_x"],
                "width_y": bbox["width_y"],
                "height_z": bbox["height_z"],
                "bottom_shape": summary["interpretation"]["bottom_shape"],
                "flat_bottom_candidate": summary["interpretation"]["flat_bottom_candidate"],
                "rocker_like_candidate": summary["interpretation"]["rocker_like_candidate"],
                "continuous_bottom_candidate": summary["interpretation"]["continuous_bottom_candidate"],
                "toe_contact_area_candidate": summary["interpretation"]["toe_contact_area_candidate"],
                "heel_contact_area_candidate": summary["interpretation"]["heel_contact_area_candidate"],
            }
        )
    write_csv(
        artifacts_dir / "foot_stl_bbox.csv",
        bbox_rows,
        [
            "side",
            "role",
            "path",
            "min_x",
            "min_y",
            "min_z",
            "max_x",
            "max_y",
            "max_z",
            "length_x",
            "width_y",
            "height_z",
            "bottom_shape",
            "flat_bottom_candidate",
            "rocker_like_candidate",
            "continuous_bottom_candidate",
            "toe_contact_area_candidate",
            "heel_contact_area_candidate",
        ],
    )
    write_json_yaml(
        artifacts_dir / "collision_comparison.yaml",
        {
            "schema_version": 1,
            "version": payload["version"],
            "model_path": payload["model_path"],
            "collision_geoms": payload["collision_geoms"],
            "comparisons": payload["comparisons"],
            "engineering_conclusion": payload["engineering_conclusion"],
            "recommendations": payload["recommendations"],
        },
    )
    write_report(resolve_repo_path(config["report_path"]), payload)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    """Run the read-only review."""

    args = parse_args()
    config = load_json_object(args.config)
    payload = build_review(config)
    write_artifacts(payload, config)
    print(f"status={payload['status']}")
    print(f"stl_candidates={len(payload['stl_summaries'])}")
    print(f"collision_over_simplified={payload['engineering_conclusion']['collision_over_simplified']}")
    print(f"next_step={payload['engineering_conclusion']['next_step']}")


if __name__ == "__main__":
    main()
