"""Generate Seedon foot-geometry scene variants and optional diagnostics.

This is an experiment helper only. It never overwrites the canonical
``private_assets/seedon/training_scene.xml`` and does not invoke PPO training.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco

from tools.seedon_debug_common import DEFAULT_SCENE_PATH, DEBUG_OUT_DIR, require_scene


OUT_DIR = DEBUG_OUT_DIR / "rounded_sole_experiments"
FOOT_GEOM_NAMES = ("R_foot_collision", "L_foot_collision")
BASELINE_SIZE = (0.07, 0.04, 0.025)


@dataclass(frozen=True)
class FootVariant:
    """One generated Seedon foot geometry variant."""

    name: str
    foot_type: str
    half_length: float
    half_width: float
    half_height: float
    rounded_method: str
    scene_filename: str


def _variants() -> list[FootVariant]:
    """Return the small fixed experiment matrix requested for this audit."""
    return [
        FootVariant(
            name="baseline_flat_box",
            foot_type="baseline_flat_box",
            half_length=0.07,
            half_width=0.04,
            half_height=0.025,
            rounded_method="none; original MuJoCo box foot collision",
            scene_filename="training_scene_baseline_flat_box.xml",
        ),
        *[
            FootVariant(
                name=f"flat_w{int(width * 100):03d}",
                foot_type="wide_flat_box",
                half_length=0.07,
                half_width=width,
                half_height=0.025,
                rounded_method="none; widened MuJoCo box foot collision",
                scene_filename=f"training_scene_flat_w{int(width * 100):03d}.xml",
            )
            for width in (0.08, 0.10, 0.12)
        ],
        *[
            FootVariant(
                name=f"rounded_w{int(width * 100):03d}",
                foot_type="rounded_sole" if width <= 0.06 else "wide_rounded_sole",
                half_length=0.07,
                half_width=width,
                half_height=0.025,
                rounded_method=(
                    "single named MuJoCo ellipsoid replacing the flat box; "
                    "ellipsoid x/y/z size preserves foot length, lateral width, and sole height"
                ),
                scene_filename=f"training_scene_rounded_w{int(width * 100):03d}.xml",
            )
            for width in (0.06, 0.08, 0.10)
        ],
    ]


def _format_floats(values: tuple[float, ...]) -> str:
    """Format MJCF float tuples compactly."""
    return " ".join(f"{value:.6g}" for value in values)


def _find_named_geom(root: ET.Element, geom_name: str) -> ET.Element:
    """Return a named geom element or raise a useful error."""
    for geom in root.iter("geom"):
        if geom.get("name") == geom_name:
            return geom
    raise ValueError(f"Foot geom not found in source scene: {geom_name}")


def _set_meshdir_to_source(tree: ET.ElementTree, source_scene: Path) -> None:
    """Keep generated scenes loadable when written outside the asset folder."""
    compiler = tree.getroot().find("compiler")
    if compiler is None:
        return
    meshdir = source_scene.parent / (compiler.get("meshdir") or "")
    compiler.set("meshdir", str(meshdir.resolve()))


def _apply_variant(root: ET.Element, variant: FootVariant) -> None:
    """Mutate foot geoms in an MJCF tree for one variant."""
    size = _format_floats(
        (variant.half_length, variant.half_width, variant.half_height)
    )
    for geom_name in FOOT_GEOM_NAMES:
        geom = _find_named_geom(root, geom_name)
        geom.set("size", size)
        if "rounded" in variant.foot_type:
            geom.set("type", "ellipsoid")
        else:
            geom.set("type", "box")


def _write_metadata(
    *,
    source_scene: Path,
    variant: FootVariant,
    generated_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    """Write one metadata JSON file and return its payload."""
    payload = {
        "source_scene": str(source_scene),
        "variant": variant.name,
        "foot_type": variant.foot_type,
        "half_length": variant.half_length,
        "half_width": variant.half_width,
        "half_height": variant.half_height,
        "rounded_approximation_method": variant.rounded_method,
        "generated_path": str(generated_path),
    }
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def generate_variants(source_scene: Path, out_dir: Path) -> list[dict[str, Any]]:
    """Generate all foot-geometry scene variants.

    Args:
        source_scene: Canonical Seedon scene to read.
        out_dir: Directory that receives generated XML and metadata JSON files.

    Returns:
        Metadata payloads for all generated variants.

    Raises:
        FileNotFoundError: If ``source_scene`` does not exist.
        ValueError: If the expected Seedon foot geoms are missing.
    """
    source_scene = require_scene(source_scene)
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, Any]] = []
    for variant in _variants():
        tree = ET.parse(source_scene)
        _set_meshdir_to_source(tree, source_scene)
        _apply_variant(tree.getroot(), variant)
        scene_path = out_dir / variant.scene_filename
        metadata_path = out_dir / f"{variant.name}.metadata.json"
        tree.write(scene_path, encoding="utf-8", xml_declaration=True)
        _validate_scene(scene_path)
        payloads.append(
            _write_metadata(
                source_scene=source_scene,
                variant=variant,
                generated_path=scene_path,
                metadata_path=metadata_path,
            )
        )
    return payloads


def _validate_scene(scene_path: Path) -> None:
    """Validate that MuJoCo can load a generated scene."""
    mujoco.MjModel.from_xml_path(str(scene_path))


def _run_command(command: list[str], log_path: Path) -> None:
    """Run one diagnostic command and save combined output."""
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.write_text(
        "$ " + " ".join(command) + "\n\n" + result.stdout + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Diagnostic failed ({result.returncode}): {' '.join(command)}")


def run_diagnostics(payloads: list[dict[str, Any]], out_dir: Path) -> None:
    """Run existing Seedon debug diagnostics for every generated variant."""
    for payload in payloads:
        variant_dir = out_dir / str(payload["variant"])
        variant_dir.mkdir(parents=True, exist_ok=True)
        scene_path = str(payload["generated_path"])
        foot_sizes = (
            f"{payload['half_length']},{payload['half_width']},{payload['half_height']}"
        )
        commands = [
            (
                "contacts",
                [
                    sys.executable,
                    "-m",
                    "tools.debug_seedon_contacts",
                    "--scene-path",
                    scene_path,
                ],
            ),
            (
                "zero_action",
                [
                    sys.executable,
                    "-m",
                    "tools.trace_zero_action_gait",
                    "--scene-path",
                    scene_path,
                    "--steps",
                    "80",
                    "--print-every",
                    "80",
                    "--out-csv",
                    str(variant_dir / "zero_action_trace.csv"),
                ],
            ),
            (
                "lateral_controllability",
                [
                    sys.executable,
                    "-m",
                    "tools.debug_seedon_lateral_controllability",
                    "--scene-path",
                    scene_path,
                    "--ramp-steps",
                    "40",
                    "--hold-steps",
                    "80",
                    "--unload-scales",
                    "0,0.5,1.0",
                    "--out-csv",
                    str(variant_dir / "lateral_controllability.csv"),
                ],
            ),
            (
                "single_support_left",
                [
                    sys.executable,
                    "-m",
                    "tools.debug_seedon_single_support_load_transfer",
                    "--scene-path",
                    scene_path,
                    "--support-side",
                    "left",
                    "--load-steps",
                    "80",
                    "--lift-steps",
                    "40",
                    "--print-every",
                    "80",
                    "--out-csv",
                    str(variant_dir / "single_support_left.csv"),
                ],
            ),
            (
                "single_support_right",
                [
                    sys.executable,
                    "-m",
                    "tools.debug_seedon_single_support_load_transfer",
                    "--scene-path",
                    scene_path,
                    "--support-side",
                    "right",
                    "--load-steps",
                    "80",
                    "--lift-steps",
                    "40",
                    "--print-every",
                    "80",
                    "--out-csv",
                    str(variant_dir / "single_support_right.csv"),
                ],
            ),
            (
                "static_stability",
                [
                    sys.executable,
                    "-m",
                    "tools.debug_seedon_static_stability",
                    "--scene-path",
                    scene_path,
                    "--plan",
                    "both",
                    "--ramp-steps",
                    "80",
                    "--hold-steps",
                    "80",
                    "--support-roll",
                    "0.10",
                    "--foot-sizes",
                    foot_sizes,
                    "--csv-path",
                    str(variant_dir / "static_stability_summary.csv"),
                ],
            ),
            (
                "foot_contact_geometry_sweep",
                [
                    sys.executable,
                    "-m",
                    "tools.debug_seedon_foot_contact_geometry_sweep",
                    "--scene-path",
                    scene_path,
                    "--steps",
                    "80",
                    "--support-roll",
                    "0.10",
                    "--out-csv",
                    str(variant_dir / "foot_contact_geometry_sweep.csv"),
                ],
            ),
        ]
        for name, command in commands:
            _run_command(command, variant_dir / f"{name}.log")


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file if it exists."""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    """Parse a float from a CSV row."""
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _variant_summary(payload: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Summarize diagnostic CSV outputs for one variant."""
    variant_dir = out_dir / str(payload["variant"])
    zero_rows = _read_csv(variant_dir / "zero_action_trace.csv")
    lateral_rows = _read_csv(variant_dir / "lateral_controllability.csv")
    left_rows = _read_csv(variant_dir / "single_support_left.csv")
    right_rows = _read_csv(variant_dir / "single_support_right.csv")
    static_rows = _read_csv(variant_dir / "static_stability_summary.csv")
    foot_rows = _read_csv(variant_dir / "foot_contact_geometry_sweep.csv")

    single_rows = left_rows + right_rows
    no_contact_steps = sum(1 for row in single_rows if row.get("contact_state") == "none")
    lift_started = any(row.get("stage") == "micro_lift" for row in single_rows)
    margins = [_float(row, "support_margin_y", 999.0) for row in single_rows]
    valid_ratio_rows = [
        row
        for row in single_rows
        if row.get("contact_state") != "none"
        and _float(row, "support_normal_force", 0.0) > 1e-3
        and _float(row, "swing_normal_force", 0.0) > 1e-3
    ]
    ratios = [_float(row, "support_force_ratio", 0.0) for row in valid_ratio_rows]
    zero_both = sum(1 for row in zero_rows if row.get("contact_state") == "both")
    zero_none = sum(1 for row in zero_rows if row.get("contact_state") == "none")
    zero_base = sum(
        1 for row in zero_rows if str(row.get("floor_base_proxy_contact")) == "True"
    )

    return {
        **payload,
        "zero_action_rows": len(zero_rows),
        "zero_action_both_steps": zero_both,
        "zero_action_none_steps": zero_none,
        "zero_action_base_proxy_steps": zero_base,
        "min_support_margin_y": min(margins) if margins else None,
        "max_support_force_ratio": max(ratios) if ratios else None,
        "valid_force_ratio_samples": len(valid_ratio_rows),
        "lift_stage_started": lift_started,
        "single_support_no_contact_ratio": (
            no_contact_steps / len(single_rows) if single_rows else None
        ),
        "max_lateral_com_delta": max(
            (_float(row, "max_abs_com_y_delta", 0.0) for row in lateral_rows),
            default=None,
        ),
        "static_any_com_inside": any(
            row.get("com_inside_support") == "True" for row in static_rows
        ),
        "foot_sweep_max_com_delta": max(
            (_float(row, "max_abs_com_y_delta", 0.0) for row in foot_rows),
            default=None,
        ),
    }


def write_report(payloads: list[dict[str, Any]], out_dir: Path) -> Path:
    """Write the markdown experiment report."""
    summaries = [_variant_summary(payload, out_dir) for payload in payloads]
    stable_candidates = [
        item
        for item in summaries
        if item["zero_action_rows"] == 80
        and item["zero_action_both_steps"] == 80
        and item["zero_action_none_steps"] == 0
        and _default_if_none(item["single_support_no_contact_ratio"], 1.0) <= 0.10
    ]
    ranked = sorted(
        stable_candidates,
        key=lambda item: (
            item["lift_stage_started"],
            -_default_if_none(item["min_support_margin_y"], 999.0),
            _default_if_none(item["max_support_force_ratio"], 0.0),
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    lines = [
        "# Seedon Rounded Sole Foot Geometry Experiment",
        "",
        "Task class: Class C experiment diagnostic. No PPO training was run and the canonical training scene was not overwritten.",
        "",
        "## Variants",
        "",
        "| variant | type | half_length | half_width | half_height | method |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['variant']} | {item['foot_type']} | {item['half_length']:.3f} | "
            f"{item['half_width']:.3f} | {item['half_height']:.3f} | "
            f"{item['rounded_approximation_method']} |"
        )

    lines.extend(
        [
            "",
            "## Diagnostic Summary",
            "",
            "| variant | zero stable | min support margin y | max force ratio | lift stage | no-contact ratio | max lateral COM delta | static COM inside |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summaries:
        zero_stable = (
            item["zero_action_rows"] == 80
            and item["zero_action_both_steps"] == 80
            and item["zero_action_none_steps"] == 0
            and item["zero_action_base_proxy_steps"] == 0
        )
        lines.append(
            f"| {item['variant']} | {zero_stable} | {_fmt_optional(item['min_support_margin_y'])} | "
            f"{_fmt_optional(item['max_support_force_ratio'])} | {item['lift_stage_started']} | "
            f"{_fmt_optional(item['single_support_no_contact_ratio'])} | "
            f"{_fmt_optional(item['max_lateral_com_delta'])} | {item['static_any_com_inside']} |"
        )

    lines.extend(["", "## Conclusion", ""])
    rounded_items = [item for item in summaries if "rounded" in item["foot_type"]]
    best_rounded = min(
        rounded_items,
        key=lambda item: item["min_support_margin_y"] or 999.0,
        default=None,
    )
    best_stable = ranked[0] if ranked else None
    if best_rounded is not None:
        lines.append(
            f"Rounded sole reduced the best observed support margin to "
            f"{_fmt_optional(best_rounded['min_support_margin_y'])} m on "
            f"`{best_rounded['variant']}`, but it also produced a high no-contact ratio "
            f"({_fmt_optional(best_rounded['single_support_no_contact_ratio'])})."
        )
    if best_stable is not None:
        lines.append(
            f"Among variants that stayed contact-stable, `{best_stable['variant']}` had the best usable load-transfer signal, "
            f"but its support margin remained {_fmt_optional(best_stable['min_support_margin_y'])} m and it never entered lift stage."
        )
    if best is None:
        lines.append(
            "No variant met the minimum physical precondition filter: zero-action stable, no-contact ratio <= 10%, and improved usable load transfer."
        )
    else:
        lines.append(
            f"Best stable fallback candidate: `{best['variant']}`. "
            "It is not Blue-like because it is still a flat-foot variant and it did not enter lift stage."
        )
        if not best["lift_stage_started"]:
            lines.append(
                "None of the variants reached the lift gate in this diagnostic set, so rounded sole alone is not yet enough to justify PPO training."
            )
        else:
            lines.append(
                "At least one variant reached the lift stage; that is enough to justify the next reward/config iteration before PPO."
            )
        lines.append(
            "Read this report as a physics precheck only. Reward changes and PPO should wait until a variant shows better lateral load transfer without worse contact loss."
        )

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _fmt_optional(value: Any) -> str:
    """Format optional floats for markdown tables."""
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _default_if_none(value: Any, default: float) -> float:
    """Return ``default`` only for None, preserving valid zero values."""
    if value is None:
        return default
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--run-diagnostics", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate variants and optionally run diagnostics/reporting."""
    args = build_parser().parse_args(argv)
    payloads = generate_variants(args.source_scene, args.out_dir)
    manifest = args.out_dir / "manifest.json"
    manifest.write_text(
        json.dumps(payloads, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.run_diagnostics:
        run_diagnostics(payloads, args.out_dir)
    if args.write_report:
        report_path = write_report(payloads, args.out_dir)
        print(f"report: {report_path}")
    print(f"variants: {len(payloads)}")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
