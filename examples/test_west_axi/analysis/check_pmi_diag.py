#!/usr/bin/env python3
"""
Summarize OpenEdge/SPARTA PMI surface diagnostics.

This parses the `dump surf` written by:

    dump dpmi surf all ... id c_cpmiD[*] c_cpmiO[*]

and maps the surface IDs onto wall-segment centroids from `input/wall.surf`.

Example:
    python3 analysis/check_pmi_diag.py
    python3 analysis/check_pmi_diag.py --timestep last --top 12
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re

import numpy as np



def parse_surface(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()

    npts = nlines = None
    i_points = i_lines = None
    for i, raw in enumerate(lines):
        s = raw.strip().lower()
        if s.endswith(" points"):
            npts = int(s.split()[0])
        elif s.endswith(" lines"):
            nlines = int(s.split()[0])
        elif s == "points":
            i_points = i + 1
        elif s == "lines":
            i_lines = i + 1

    if None in (npts, nlines, i_points, i_lines):
        raise RuntimeError(f"Could not parse surface file {path}")

    pts = {}
    k = i_points
    while k < len(lines) and len(pts) < npts:
        parts = lines[k].split()
        k += 1
        if len(parts) < 3:
            continue
        try:
            pts[int(parts[0])] = (float(parts[1]), float(parts[2]))
        except ValueError:
            continue

    seg = {}
    k = i_lines
    nread = 0
    while k < len(lines) and nread < nlines:
        parts = lines[k].split()
        k += 1
        if len(parts) < 3:
            continue
        try:
            sid = int(parts[0])
            p1 = int(parts[1])
            p2 = int(parts[2])
        except ValueError:
            continue
        if p1 not in pts or p2 not in pts:
            continue
        r1, z1 = pts[p1]
        r2, z2 = pts[p2]
        seg[sid] = {
            "r1": r1,
            "z1": z1,
            "r2": r2,
            "z2": z2,
            "rc": 0.5 * (r1 + r2),
            "zc": 0.5 * (z1 + z2),
            "length": math.hypot(r2 - r1, z2 - z1),
        }
        nread += 1

    return seg


def parse_surf_dump(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    blocks = {}
    headers = {}

    while i < len(lines):
        if lines[i].strip() != "ITEM: TIMESTEP":
            i += 1
            continue
        ts = int(lines[i + 1].strip())
        i += 2

        if lines[i].strip() != "ITEM: NUMBER OF SURFS":
            raise RuntimeError("Bad dump format: NUMBER OF SURFS")
        nsurf = int(lines[i + 1].strip())
        i += 2

        if not lines[i].startswith("ITEM: BOX BOUNDS"):
            raise RuntimeError("Bad dump format: BOX BOUNDS")
        i += 4

        if not lines[i].startswith("ITEM: SURFS"):
            raise RuntimeError("Bad dump format: SURFS header")
        header = lines[i].split()[2:]
        i += 1

        rows = []
        for _ in range(nsurf):
            parts = lines[i].split()
            i += 1
            row = {header[j]: float(parts[j]) for j in range(len(header))}
            row["id"] = int(row["id"])
            rows.append(row)
        blocks[ts] = rows
        headers[ts] = header

    if not blocks:
        raise RuntimeError(f"No SURFS blocks found in {path}")
    return blocks, headers


def read_species_names(plasma_h5: Path):
    try:
        import h5py  # type: ignore
    except ImportError:
        return None

    try:
        with h5py.File(str(plasma_h5), "r") as f:
            if "names" not in f:
                return None
            raw = f["names"][...]
    except Exception:
        return None

    names = []
    for item in raw:
        if isinstance(item, bytes):
            names.append(item.decode("utf-8"))
        else:
            names.append(str(item))
    return names


def header_indices(header, prefix: str):
    pat = re.compile(rf"^{re.escape(prefix)}\[(\d+)\]$")
    found = []
    for name in header:
        m = pat.match(name)
        if m:
            found.append((int(m.group(1)), name))
    found.sort()
    return [name for _, name in found]


def split_cpmi_columns(header, prefix: str, nspec: int, nproj: int):
    cols = header_indices(header, prefix)
    if len(cols) == 5:
        return {
            "flux_cols": [cols[0]],
            "angle_cols": [cols[1]],
            "energy_cols": [cols[2]],
            "sputter_cols": [cols[3]],
            "total_col": cols[4],
        }

    expected_all = 3 * nspec + nproj + 1
    if len(cols) != expected_all:
        raise RuntimeError(
            f"{prefix}: unsupported column count {len(cols)}; "
            f"expected either 5 (single-spec) or {expected_all} (all-spec layout)"
        )

    i = 0
    flux_cols = cols[i : i + nspec]
    i += nspec
    angle_cols = cols[i : i + nspec]
    i += nspec
    energy_cols = cols[i : i + nspec]
    i += nspec
    sputter_cols = cols[i : i + nproj]
    i += nproj
    total_col = cols[i]
    return {
        "flux_cols": flux_cols,
        "angle_cols": angle_cols,
        "energy_cols": energy_cols,
        "sputter_cols": sputter_cols,
        "total_col": total_col,
    }


def pick_timestep(blocks, timestep_arg: str):
    if timestep_arg == "last":
        return max(blocks)
    ts = int(timestep_arg)
    if ts not in blocks:
        raise KeyError(f"Timestep {ts} not found; available: {sorted(blocks)}")
    return ts


def summarize(rows, segments, cpmi_d_layout, cpmi_o_layout, species_names):
    proj_species = species_names[1:]
    enriched = []
    for row in rows:
        sid = row["id"]
        if sid not in segments:
            continue
        seg = segments[sid]

        if cpmi_d_layout is not None:
            d_flux_all = np.array([row[c] for c in cpmi_d_layout["flux_cols"]], dtype=float)
            d_angles_all = np.array([row[c] for c in cpmi_d_layout["angle_cols"]], dtype=float)
            d_energy_all = np.array([row[c] for c in cpmi_d_layout["energy_cols"]], dtype=float)
            d_sputter_all = np.array([row[c] for c in cpmi_d_layout["sputter_cols"]], dtype=float)
            d_flux = d_flux_all[0]
            d_angle = d_angles_all[0]
            d_energy = d_energy_all[0]
            d_sputter = d_sputter_all[0] if d_sputter_all.size else 0.0
            d_total = row[cpmi_d_layout["total_col"]]
        else:
            d_flux = o_flux_all = None
            d_angle = 0.0
            d_energy = 0.0
            d_sputter = 0.0
            d_total = 0.0

        o_flux_all = np.array([row[c] for c in cpmi_o_layout["flux_cols"]], dtype=float)
        o_angles_all = np.array([row[c] for c in cpmi_o_layout["angle_cols"]], dtype=float)
        o_energy_all = np.array([row[c] for c in cpmi_o_layout["energy_cols"]], dtype=float)
        o_sputter = np.array([row[c] for c in cpmi_o_layout["sputter_cols"]], dtype=float)

        # cpmiO requests "all" for flux/angle/energy, so those arrays include D+
        # in slot 1. Restrict O summaries to projectile slots 2..N.
        o_flux = o_flux_all[1:]
        o_angles = o_angles_all[1:]
        o_energy = o_energy_all[1:]
        dom_idx = int(np.argmax(o_sputter if np.any(o_sputter > 0.0) else o_flux))
        enriched.append(
            {
                **seg,
                "id": sid,
                "d_flux": d_flux if cpmi_d_layout is not None else o_flux_all[0],
                "d_angle": d_angle if cpmi_d_layout is not None else o_angles_all[0],
                "d_energy": d_energy if cpmi_d_layout is not None else o_energy_all[0],
                "d_sputter": d_sputter,
                "d_total": d_total,
                "o_flux_sum": float(np.sum(o_flux)),
                "o_total": row[cpmi_o_layout["total_col"]],
                "o_flux": o_flux,
                "o_angles": o_angles,
                "o_energy": o_energy,
                "o_sputter": o_sputter,
                "dom_species": proj_species[dom_idx],
                "dom_flux": o_flux[dom_idx],
                "dom_energy": o_energy[dom_idx],
                "dom_sputter": o_sputter[dom_idx],
                "dom_angle": o_angles[dom_idx],
            }
        )
    return enriched


def print_global_summary(data):
    lengths = np.array([d["length"] for d in data], dtype=float)
    d_flux = np.array([d["d_flux"] for d in data], dtype=float)
    o_flux = np.array([d["o_flux_sum"] for d in data], dtype=float)
    d_sput = np.array([d["d_total"] for d in data], dtype=float)
    o_sput = np.array([d["o_total"] for d in data], dtype=float)

    d_flux_int = float(np.sum(d_flux * lengths))
    o_flux_int = float(np.sum(o_flux * lengths))
    d_sput_int = float(np.sum(d_sput * lengths))
    o_sput_int = float(np.sum(o_sput * lengths))
    total_flux = d_flux_int + o_flux_int
    total_sput = d_sput_int + o_sput_int

    print("Integrated Along Wall")
    print(
        f"  D Bohm flux integral      = {d_flux_int:.6e} [m^-1 s^-1]"
    )
    print(
        f"  O Bohm flux integral      = {o_flux_int:.6e} [m^-1 s^-1]"
    )
    print(
        f"  D sputter source integral = {d_sput_int:.6e} [m^-1 s^-1]"
    )
    print(
        f"  O sputter source integral = {o_sput_int:.6e} [m^-1 s^-1]"
    )
    if total_flux > 0.0:
        print(f"  D flux fraction           = {d_flux_int / total_flux:.3%}")
        print(f"  O flux fraction           = {o_flux_int / total_flux:.3%}")
    if total_sput > 0.0:
        print(f"  D sputter fraction        = {d_sput_int / total_sput:.3%}")
        print(f"  O sputter fraction        = {o_sput_int / total_sput:.3%}")


def print_top_table(data, key: str, label: str, top: int):
    order = sorted(data, key=lambda d: d[key], reverse=True)
    print()
    print(label)
    for d in order[:top]:
        print(
            f"  surf={d['id']:3d} R={d['rc']:.4f} Z={d['zc']:.4f} "
            f"D_flux={d['d_flux']:.3e} O_flux={d['o_flux_sum']:.3e} "
            f"D_sput={d['d_total']:.3e} O_sput={d['o_total']:.3e} "
            f"domO={d['dom_species']:>3s} domO_flux={d['dom_flux']:.3e} "
            f"domO_E={d['dom_energy']:.3e}eV domO_sput={d['dom_sputter']:.3e}"
        )


def print_regional_summary(data):
    regions = {
        "lower_outer": lambda d: d["zc"] < -0.40 and d["rc"] < 2.45,
        "upper_outer": lambda d: d["zc"] > 0.40 and d["rc"] < 2.45,
        "lower_limiter": lambda d: d["zc"] < -0.35 and d["rc"] > 2.70,
        "upper_limiter": lambda d: d["zc"] > 0.35 and d["rc"] > 2.70,
    }
    print()
    print("Regional Integrals")
    for name, pred in regions.items():
        subset = [d for d in data if pred(d)]
        if not subset:
            continue
        length = np.array([d["length"] for d in subset], dtype=float)
        d_sput = float(np.sum(np.array([d["d_total"] for d in subset]) * length))
        o_sput = float(np.sum(np.array([d["o_total"] for d in subset]) * length))
        d_flux = float(np.sum(np.array([d["d_flux"] for d in subset]) * length))
        o_flux = float(np.sum(np.array([d["o_flux_sum"] for d in subset]) * length))
        print(
            f"  {name:13s} D_flux={d_flux:.3e} O_flux={o_flux:.3e} "
            f"D_sput={d_sput:.3e} O_sput={o_sput:.3e}"
        )


def main():
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--surf-dump", default=str(here / "output" / "pmi_diag.west_o.surf"))
    ap.add_argument("--wall", default=str(here / "input" / "wall.surf"))
    ap.add_argument("--plasma-h5", default=str(here / "input" / "plasma.h5"))
    ap.add_argument("--timestep", default="last", help="'last' or explicit timestep")
    ap.add_argument("--top", type=int, default=10, help="rows to print in each top table")
    args = ap.parse_args()

    blocks, headers = parse_surf_dump(Path(args.surf_dump))
    ts = pick_timestep(blocks, args.timestep)
    segments = parse_surface(Path(args.wall))
    species_names = read_species_names(Path(args.plasma_h5))
    if not species_names:
        species_names = ["D+", "O+", "O2+", "O3+", "O4+", "O5+", "O6+", "O7+", "O8+"]

    cpmi_d_cols = header_indices(headers[ts], "c_cpmiD")
    cpmi_d_layout = None
    if cpmi_d_cols:
        cpmi_d_layout = split_cpmi_columns(headers[ts], "c_cpmiD", len(species_names), 1)
    cpmi_o_layout = split_cpmi_columns(headers[ts], "c_cpmiO", len(species_names), len(species_names) - 1)
    data = summarize(blocks[ts], segments, cpmi_d_layout, cpmi_o_layout, species_names)

    print(f"Timestep: {ts}")
    print(f"Segments parsed: {len(data)}")
    print(f"Plasma species: {', '.join(species_names)}")
    print(f"cpmiO projectile species: {', '.join(species_names[1:])}")
    print_global_summary(data)
    print_regional_summary(data)
    if cpmi_d_layout is not None:
        print_top_table(data, "d_total", f"Top {args.top} D-Driven Sputter Segments", args.top)
    print_top_table(data, "o_total", f"Top {args.top} O-Driven Sputter Segments", args.top)


if __name__ == "__main__":
    main()
