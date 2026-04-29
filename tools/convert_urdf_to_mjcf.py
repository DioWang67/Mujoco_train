"""Convert a ROS-style URDF package into a MuJoCo-loadable MJCF file."""

from __future__ import annotations

import argparse
import math
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import mujoco

STL_HEADER_BYTES = 80
STL_FACE_COUNT_BYTES = 4
STL_TRIANGLE_BYTES = 50


@dataclass(frozen=True)
class MeshConversionResult:
    """Result for one STL mesh prepared for MuJoCo compilation.

    Args:
        source: Original STL path.
        destination: Prepared STL path.
        original_faces: Number of faces in the original mesh.
        output_faces: Number of faces in the prepared mesh.
        decimated: Whether faces were sampled down.
    """

    source: Path
    destination: Path
    original_faces: int
    output_faces: int
    decimated: bool


def read_binary_stl_face_count(path: Path) -> int:
    """Return the face count from a binary STL file.

    Args:
        path: STL file to inspect.

    Returns:
        Number of triangles declared in the STL header.

    Raises:
        ValueError: If the file is not a valid binary STL by size.
    """
    actual_size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(STL_HEADER_BYTES)
        count_bytes = handle.read(STL_FACE_COUNT_BYTES)

    if len(header) != STL_HEADER_BYTES or len(count_bytes) != STL_FACE_COUNT_BYTES:
        raise ValueError(f"STL file is too small: {path}")

    face_count = struct.unpack("<I", count_bytes)[0]
    expected_size = STL_HEADER_BYTES + STL_FACE_COUNT_BYTES + face_count * STL_TRIANGLE_BYTES
    if actual_size != expected_size:
        raise ValueError(
            f"Only binary STL files are supported. Size mismatch for {path}: "
            f"expected {expected_size}, got {actual_size}."
        )
    return face_count


def prepare_binary_stl_for_mujoco(
    source: Path,
    destination: Path,
    max_faces: int,
) -> MeshConversionResult:
    """Copy or downsample one binary STL so MuJoCo can compile it.

    MuJoCo rejects a single mesh with too many faces. For an initial conversion,
    uniform face sampling is enough to produce a loadable visual model. It is not
    a substitute for proper collision mesh cleanup.

    Args:
        source: Original STL file.
        destination: Destination STL file.
        max_faces: Maximum number of faces to keep.

    Returns:
        Conversion metadata for reporting.

    Raises:
        ValueError: If ``max_faces`` is not positive or the STL is invalid.
    """
    if max_faces <= 0:
        raise ValueError("max_faces must be positive.")

    face_count = read_binary_stl_face_count(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if face_count <= max_faces:
        shutil.copy2(source, destination)
        return MeshConversionResult(source, destination, face_count, face_count, False)

    stride = math.ceil(face_count / max_faces)
    kept_faces = 0

    with source.open("rb") as src, destination.open("wb") as dst:
        original_header = src.read(STL_HEADER_BYTES)
        src.read(STL_FACE_COUNT_BYTES)
        header = original_header[:72] + b"mjcfconv"
        dst.write(header[:STL_HEADER_BYTES].ljust(STL_HEADER_BYTES, b"\0"))
        dst.write(struct.pack("<I", 0))

        for face_index in range(face_count):
            triangle = src.read(STL_TRIANGLE_BYTES)
            if len(triangle) != STL_TRIANGLE_BYTES:
                raise ValueError(f"Unexpected EOF while reading {source}")
            if face_index % stride == 0 and kept_faces < max_faces:
                dst.write(triangle)
                kept_faces += 1

        dst.seek(STL_HEADER_BYTES)
        dst.write(struct.pack("<I", kept_faces))

    return MeshConversionResult(source, destination, face_count, kept_faces, True)


def prepare_urdf_package(
    urdf_path: Path,
    mesh_dir: Path,
    work_dir: Path,
    max_faces: int,
) -> tuple[Path, list[MeshConversionResult]]:
    """Prepare a flat URDF work directory for MuJoCo's URDF compiler.

    Args:
        urdf_path: Source URDF file.
        mesh_dir: Directory containing STL meshes referenced by the URDF.
        work_dir: Output directory used for compilation.
        max_faces: Maximum face count per STL mesh.

    Returns:
        Tuple of prepared URDF path and per-mesh conversion results.

    Raises:
        FileNotFoundError: If required inputs are missing.
        ValueError: If no STL meshes are found.
    """
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")

    mesh_paths_by_path = {
        mesh_path.resolve(): mesh_path
        for pattern in ("*.STL", "*.stl")
        for mesh_path in mesh_dir.glob(pattern)
    }
    mesh_paths = sorted(mesh_paths_by_path.values())
    if not mesh_paths:
        raise ValueError(f"No STL meshes found under: {mesh_dir}")

    work_dir.mkdir(parents=True, exist_ok=True)
    prepared_urdf = work_dir / "sedon.urdf"
    shutil.copy2(urdf_path, prepared_urdf)

    results = [
        prepare_binary_stl_for_mujoco(mesh_path, work_dir / mesh_path.name, max_faces)
        for mesh_path in mesh_paths
    ]
    return prepared_urdf, results


def convert_urdf_to_mjcf(
    urdf_path: Path,
    mesh_dir: Path,
    output_dir: Path,
    max_faces: int,
) -> tuple[Path, list[MeshConversionResult], mujoco.MjModel]:
    """Convert URDF + STL assets into a MuJoCo MJCF XML file.

    Args:
        urdf_path: Source URDF file.
        mesh_dir: Directory containing source STL meshes.
        output_dir: Private output directory for prepared assets and scene.xml.
        max_faces: Maximum face count per mesh before downsampling.

    Returns:
        Generated scene path, mesh conversion results, and compiled MuJoCo model.
    """
    prepared_urdf, results = prepare_urdf_package(
        urdf_path=urdf_path,
        mesh_dir=mesh_dir,
        work_dir=output_dir,
        max_faces=max_faces,
    )
    model = mujoco.MjModel.from_xml_path(str(prepared_urdf))
    scene_path = output_dir / "scene.xml"
    mujoco.mj_saveLastXML(str(scene_path), model)
    return scene_path, results, model


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(
        description="Convert a URDF package with STL meshes into a MuJoCo MJCF scene."
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("private_assets/sedon/original_urdf_package/urdf/urdf/urdf.urdf"),
        help="Source URDF file.",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=Path("private_assets/sedon/original_urdf_package/urdf/meshes"),
        help="Directory containing STL meshes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("private_assets/sedon/mjcf_source"),
        help="Private output directory for prepared assets and scene.xml.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=190_000,
        help="Maximum faces per STL mesh. Keep below MuJoCo's 200000 limit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the URDF to MJCF conversion command."""
    args = build_parser().parse_args(argv)
    scene_path, results, model = convert_urdf_to_mjcf(
        urdf_path=args.urdf,
        mesh_dir=args.mesh_dir,
        output_dir=args.output_dir,
        max_faces=args.max_faces,
    )

    print(f"Saved MJCF: {scene_path}")
    print(f"Model: nbody={model.nbody} njnt={model.njnt} ngeom={model.ngeom} nu={model.nu}")
    for result in results:
        status = "decimated" if result.decimated else "copied"
        print(
            f"{status}: {result.source.name} "
            f"{result.original_faces} -> {result.output_faces} faces"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
