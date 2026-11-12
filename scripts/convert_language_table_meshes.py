#!/usr/bin/env python3
"""Convert Language Table's block meshes (OBJ) to binary STL for MuJoCo 2.1, which cannot load OBJ.

Each mesh is an outline extruded into a slab. The output is centered on its bounding box, scaled so the outline's
half-width is 1 (the env scales it by ``target_size``, like the sphere radius and cube half-size), and rotated so
the outline lies in the xy-plane and the extrusion runs along z.

Example:
  python scripts/convert_language_table_meshes.py \
    --blocks ~/code/language-table/language_table/environments/assets/blocks \
    --outdir multi_object_fetch/assets/meshes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SHAPES = ("moon", "star", "pentagon")
# OBJ axes: outline in x-z, extrusion along y. Rotation (det +1) to outline in x-y, extrusion along z.
ROTATION = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def read_obj(path: Path):
    vertices, triangles = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f":
            ids = [int(p.split("/")[0]) - 1 for p in parts[1:]]
            triangles += [(ids[0], ids[k], ids[k + 1]) for k in range(1, len(ids) - 1)]  # fan (faces are convex)
    return np.array(vertices), np.array(triangles)


def normalize(vertices: np.ndarray) -> np.ndarray:
    vertices = vertices - (vertices.min(0) + vertices.max(0)) / 2
    vertices = vertices / np.abs(vertices[:, [0, 2]]).max()
    return vertices @ ROTATION.T


def write_stl(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    tri = vertices[triangles].astype(np.float32)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    record = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    data = np.zeros(len(tri), dtype=record)
    data["normal"], data["vertices"] = normals, tri
    with open(path, "wb") as f:
        f.write(b"Language Table block mesh".ljust(80, b" "))
        f.write(np.uint32(len(tri)).tobytes())
        f.write(data.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=Path, required=True, help="language_table/environments/assets/blocks")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parents[1]
                        / "multi_object_fetch" / "assets" / "meshes")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    for shape in SHAPES:
        vertices, triangles = read_obj(args.blocks.expanduser() / f"{shape}.obj")
        vertices = normalize(vertices)
        write_stl(args.outdir / f"{shape}.stl", vertices, triangles)
        print(f"{shape}: {len(triangles)} triangles, extent {np.ptp(vertices, 0).round(3)}, "
              f"radius {np.linalg.norm(vertices, axis=1).max():.3f}")


if __name__ == "__main__":
    main()
