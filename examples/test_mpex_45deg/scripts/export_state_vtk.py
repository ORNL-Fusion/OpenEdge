#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def iter_frames(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            step = int(handle.readline().strip())
            if not handle.readline().startswith("ITEM: NUMBER OF ATOMS"):
                raise RuntimeError(f"{path}: malformed NUMBER OF ATOMS block at step {step}")
            natoms = int(handle.readline().strip())

            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError(f"{path}: malformed BOX BOUNDS block at step {step}")
            handle.readline()
            handle.readline()
            handle.readline()

            header = handle.readline().strip().split()[2:]
            col = {name: idx for idx, name in enumerate(header)}
            needed = ["id", "type", "x", "y", "z"]
            missing = [name for name in needed if name not in col]
            if missing:
                raise RuntimeError(f"{path}: missing columns {missing} at step {step}")

            rows = []
            for _ in range(natoms):
                parts = handle.readline().split()
                if len(parts) != len(header):
                    raise RuntimeError(f"{path}: row width mismatch at step {step}")
                rows.append(
                    {
                        "id": int(parts[col["id"]]),
                        "type": int(parts[col["type"]]),
                        "x": float(parts[col["x"]]),
                        "y": float(parts[col["y"]]),
                        "z": float(parts[col["z"]]),
                    }
                )
            yield step, rows


def write_vtp(points, ids, types, outpath: Path):
    vtk = ET.Element(
        "VTKFile",
        type="PolyData",
        version="0.1",
        byte_order="LittleEndian",
    )
    poly = ET.SubElement(vtk, "PolyData")
    piece = ET.SubElement(
        poly,
        "Piece",
        NumberOfPoints=str(len(points)),
        NumberOfVerts=str(len(points)),
        NumberOfLines="0",
        NumberOfStrips="0",
        NumberOfPolys="0",
    )

    pdata = ET.SubElement(piece, "PointData", Scalars="type")
    arr_type = ET.SubElement(
        pdata, "DataArray", type="Int32", Name="type", format="ascii"
    )
    arr_type.text = " ".join(str(v) for v in types)
    arr_id = ET.SubElement(
        pdata, "DataArray", type="Int32", Name="id", format="ascii"
    )
    arr_id.text = " ".join(str(v) for v in ids)

    cdata = ET.SubElement(piece, "CellData")

    pts = ET.SubElement(piece, "Points")
    arr_pts = ET.SubElement(
        pts, "DataArray", type="Float64", NumberOfComponents="3", format="ascii"
    )
    arr_pts.text = " ".join(f"{x:.16e} {y:.16e} {z:.16e}" for x, y, z in points)

    verts = ET.SubElement(piece, "Verts")
    connectivity = ET.SubElement(
        verts, "DataArray", type="Int32", Name="connectivity", format="ascii"
    )
    connectivity.text = " ".join(str(i) for i in range(len(points)))
    offsets = ET.SubElement(
        verts, "DataArray", type="Int32", Name="offsets", format="ascii"
    )
    offsets.text = " ".join(str(i + 1) for i in range(len(points)))

    ET.indent(vtk)
    outpath.write_text(ET.tostring(vtk, encoding="unicode"))


def main():
    parser = argparse.ArgumentParser(description="Convert a particle state dump into VTK point-cloud frames.")
    parser.add_argument("--input", required=True, help="Path to particle dump, e.g. output/state.45.dump")
    parser.add_argument("--outdir", default="vtk/state", help="Directory for .vtp frames")
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Write only the latest frame instead of a full time series.",
    )
    args = parser.parse_args()

    inpath = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = list(iter_frames(inpath))
    if not frames:
        raise SystemExit(f"No particle frames found in {inpath}")

    if args.latest_only:
        frames = [frames[-1]]

    for step, rows in frames:
        points = [(row["x"], row["y"], row["z"]) for row in rows]
        ids = [row["id"] for row in rows]
        types = [row["type"] for row in rows]
        outpath = outdir / f"{inpath.stem}_t{step:08d}.vtp"
        write_vtp(points, ids, types, outpath)
        print(f"Wrote {outpath}")


if __name__ == "__main__":
    main()
