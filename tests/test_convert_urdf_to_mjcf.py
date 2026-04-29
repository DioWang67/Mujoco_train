from __future__ import annotations

import struct
from pathlib import Path

from tools.convert_urdf_to_mjcf import (
    STL_HEADER_BYTES,
    STL_TRIANGLE_BYTES,
    prepare_binary_stl_for_mujoco,
    read_binary_stl_face_count,
)


def _write_binary_stl(path: Path, face_count: int) -> None:
    header = b"test-stl".ljust(STL_HEADER_BYTES, b"\0")
    triangle = b"\0" * STL_TRIANGLE_BYTES
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(struct.pack("<I", face_count))
        for _ in range(face_count):
            handle.write(triangle)


def test_read_binary_stl_face_count_returns_declared_faces(tmp_path: Path) -> None:
    stl_path = tmp_path / "mesh.STL"
    _write_binary_stl(stl_path, face_count=3)

    assert read_binary_stl_face_count(stl_path) == 3


def test_prepare_binary_stl_copies_mesh_under_limit(tmp_path: Path) -> None:
    source = tmp_path / "source.STL"
    destination = tmp_path / "prepared.STL"
    _write_binary_stl(source, face_count=2)

    result = prepare_binary_stl_for_mujoco(source, destination, max_faces=5)

    assert result.decimated is False
    assert result.original_faces == 2
    assert result.output_faces == 2
    assert read_binary_stl_face_count(destination) == 2


def test_prepare_binary_stl_decimates_mesh_over_limit(tmp_path: Path) -> None:
    source = tmp_path / "source.STL"
    destination = tmp_path / "prepared.STL"
    _write_binary_stl(source, face_count=10)

    result = prepare_binary_stl_for_mujoco(source, destination, max_faces=4)

    assert result.decimated is True
    assert result.original_faces == 10
    assert result.output_faces <= 4
    assert read_binary_stl_face_count(destination) == result.output_faces
