#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def iter_blocks(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                return
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            step = int(handle.readline().strip())
            if not handle.readline().startswith("ITEM: NUMBER OF SURFS"):
                raise RuntimeError(f"{path}: malformed NUMBER OF SURFS block at step {step}")
            nsurf = int(handle.readline().strip())
            if not handle.readline().startswith("ITEM: BOX BOUNDS"):
                raise RuntimeError(f"{path}: malformed BOX BOUNDS block at step {step}")
            handle.readline()
            handle.readline()
            handle.readline()

            header = handle.readline().strip().split()[2:]
            col = {name: idx for idx, name in enumerate(header)}
            rows = []
            for _ in range(nsurf):
                parts = handle.readline().split()
                if len(parts) != len(header) + 1:
                    raise RuntimeError(f"{path}: row width mismatch at step {step}")
                rows.append((int(parts[0]), [float(v) for v in parts[1:]]))
            yield step, header, col, rows


def extract_value_map(header, col, rows, scalar_name):
    if scalar_name not in col:
        raise RuntimeError(
            f"Scalar '{scalar_name}' not found. Available columns include: {header[:20]}"
        )
    idx = col[scalar_name]
    values = {sid: vals[idx] for sid, vals in rows}
    return values


def write_vtp(points, triangles, surf_ids, scalars, scalar_name, outpath: Path):
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
        NumberOfVerts="0",
        NumberOfLines="0",
        NumberOfStrips="0",
        NumberOfPolys=str(len(triangles)),
    )

    pdata = ET.SubElement(piece, "PointData")
    cdata = ET.SubElement(piece, "CellData", Scalars=scalar_name)
    arr_id = ET.SubElement(cdata, "DataArray", type="Int32", Name="surf_id", format="ascii")
    arr_id.text = " ".join(str(v) for v in surf_ids)
    arr_scalar = ET.SubElement(cdata, "DataArray", type="Float64", Name=scalar_name, format="ascii")
    arr_scalar.text = " ".join(f"{v:.16e}" for v in scalars)

    pts = ET.SubElement(piece, "Points")
    arr_pts = ET.SubElement(
        pts, "DataArray", type="Float64", NumberOfComponents="3", format="ascii"
    )
    arr_pts.text = " ".join(f"{x:.16e} {y:.16e} {z:.16e}" for x, y, z in points)

    polys = ET.SubElement(piece, "Polys")
    conn = ET.SubElement(polys, "DataArray", type="Int32", Name="connectivity", format="ascii")
    conn.text = " ".join(" ".join(str(v) for v in tri) for tri in triangles)
    offsets = ET.SubElement(polys, "DataArray", type="Int32", Name="offsets", format="ascii")
    offsets.text = " ".join(str(3 * (i + 1)) for i in range(len(triangles)))

    ET.indent(vtk)
    outpath.write_text(ET.tostring(vtk, encoding="unicode"))


def main():
    parser = argparse.ArgumentParser(description="Convert a surface dump to VTK PolyData.")
    parser.add_argument("--input", required=True, help="Path to surface dump, e.g. output/surface_diag.45.surf")
    parser.add_argument(
        "--scalar",
        default="f_fSurfAll[1]",
        help="Surface scalar to export from the dump header.",
    )
    parser.add_argument(
        "--timestep",
        default="last",
        help="Step number to export, or 'last' for the latest block.",
    )
    parser.add_argument("--output", required=True, help="Output .vtp path")
    args = parser.parse_args()

    blocks = list(iter_blocks(Path(args.input)))
    if not blocks:
        raise SystemExit(f"No surface blocks found in {args.input}")

    if args.timestep == "last":
        step, header, col, rows = blocks[-1]
    else:
        step_wanted = int(args.timestep)
        matches = [blk for blk in blocks if blk[0] == step_wanted]
        if not matches:
            raise SystemExit(f"Timestep {step_wanted} not found in {args.input}")
        step, header, col, rows = matches[-1]

    values = extract_value_map(header, col, rows, args.scalar)

    points = []
    triangles = []
    surf_ids = []
    scalars = []
    point_index = 0
    for surf_id, vals in rows:
        v1 = tuple(vals[col["v1x"] : col["v1z"] + 1])
        v2 = tuple(vals[col["v2x"] : col["v2z"] + 1])
        v3 = tuple(vals[col["v3x"] : col["v3z"] + 1])
        points.extend([v1, v2, v3])
        triangles.append((point_index, point_index + 1, point_index + 2))
        point_index += 3
        surf_ids.append(surf_id)
        scalars.append(values[surf_id])

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    write_vtp(points, triangles, surf_ids, scalars, args.scalar, outpath)
    print(f"Wrote {outpath} from timestep {step}")


if __name__ == "__main__":
    main()
