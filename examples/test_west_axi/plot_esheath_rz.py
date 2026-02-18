#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def read_last_grid_block(path: Path):
    lines = path.read_text().splitlines()
    i = 0
    last_rows = None
    last_cols = None

    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue

        i += 2
        if i >= len(lines) or not lines[i].startswith("ITEM: NUMBER OF CELLS"):
            break
        i += 1
        ncell = int(lines[i].strip())
        i += 1

        if i >= len(lines) or not lines[i].startswith("ITEM: BOX BOUNDS"):
            break
        i += 4

        if i >= len(lines) or not lines[i].startswith("ITEM: CELLS"):
            break
        cols = lines[i].split()[2:]
        i += 1

        rows = []
        for _ in range(ncell):
            if i >= len(lines) or lines[i].startswith("ITEM:"):
                break
            parts = lines[i].split()
            if len(parts) == len(cols):
                rows.append(parts)
            i += 1

        last_rows = rows
        last_cols = cols

    if not last_rows or not last_cols:
        raise RuntimeError(f"Could not parse grid dump: {path}")
    return last_cols, last_rows


def build_mesh(cols, rows, field_names: list[str]):
    try:
        ix = cols.index("xc")
        iy = cols.index("yc")
    except ValueError as exc:
        raise RuntimeError("Grid dump must include xc and yc") from exc

    colmap = {c: j for j, c in enumerate(cols)}
    for f in field_names:
        if f not in colmap:
            raise RuntimeError(f"Missing required field column: {f}")

    r_vals = sorted({float(r[ix]) for r in rows})
    z_vals = sorted({float(r[iy]) for r in rows})

    nr = len(r_vals)
    nz = len(z_vals)
    R = np.array(r_vals, dtype=float).reshape(1, nr).repeat(nz, axis=0)
    Z = np.array(z_vals, dtype=float).reshape(nz, 1).repeat(nr, axis=1)

    rowmap = {(float(r[ix]), float(r[iy])): r for r in rows}
    meshes = {}
    for f in field_names:
        F = np.full((nz, nr), np.nan, dtype=float)
        j = colmap[f]
        for iz, zv in enumerate(z_vals):
            for ir, rv in enumerate(r_vals):
                rec = rowmap.get((rv, zv))
                if rec is not None:
                    F[iz, ir] = float(rec[j])
        meshes[f] = F

    return R, Z, meshes


def read_surface_polygon(surface_file: Path):
    points = {}
    lines = []
    section = None

    with surface_file.open("r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if low == "points":
                section = "points"
                continue
            if low == "lines":
                section = "lines"
                continue

            parts = line.split()
            if section == "points" and len(parts) >= 3 and parts[0].isdigit():
                points[int(parts[0])] = (float(parts[1]), float(parts[2]))
            elif section == "lines" and len(parts) >= 3 and parts[0].isdigit():
                lines.append((int(parts[1]), int(parts[2])))

    if not points:
        return None

    if not lines:
        return [points[i] for i in sorted(points)]

    chain = [lines[0][0], lines[0][1]]
    remaining = lines[1:]
    while remaining:
        last = chain[-1]
        next_idx = None
        for i, (a, b) in enumerate(remaining):
            if a == last:
                chain.append(b)
                next_idx = i
                break
            if b == last:
                chain.append(a)
                next_idx = i
                break
        if next_idx is None:
            break
        remaining.pop(next_idx)
        if chain[-1] == chain[0]:
            break

    if chain[0] != chain[-1]:
        chain.append(chain[0])
    return [points[idx] for idx in chain if idx in points]


def main():
    ap = argparse.ArgumentParser(description="Plot WEST sheath E-field magnitude on R-Z grid")
    ap.add_argument("--dump", default="output/sheath.grid")
    ap.add_argument("--out", default="output/west_esheath.rz.png")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--log", action="store_true", help="Use log10 scale for positive |E| values")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    dumpf = (base / args.dump).resolve() if not Path(args.dump).is_absolute() else Path(args.dump)

    cols, rows = read_last_grid_block(dumpf)
    R, Z, meshes = build_mesh(cols, rows, ["c_csheath[1]", "c_csheath[2]", "c_csheath[3]", "c_csheath[8]"])

    Ex = meshes["c_csheath[1]"]
    Ey = meshes["c_csheath[2]"]
    Ez = meshes["c_csheath[3]"]
    active = meshes["c_csheath[8]"]
    Emag = np.sqrt(np.maximum(0.0, Ex * Ex + Ey * Ey + Ez * Ez))
    Emag = np.where(active > 0.5, Emag, np.nan)

    wall = read_surface_polygon(base / "input" / "wall.txt")
    core = read_surface_polygon(base / "input" / "core.txt")

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.2), dpi=200, constrained_layout=True)
    fig.patch.set_facecolor("#d9d9d9")
    ax.set_facecolor("#d9d9d9")

    finite = Emag[np.isfinite(Emag)]
    if finite.size == 0:
        raise RuntimeError("No active sheath cells found (all |E| are NaN after active-mask).")

    if args.log:
        pos = finite[finite > 0.0]
        if pos.size == 0:
            raise RuntimeError("Cannot use --log: no positive |E| values")
        vmin = float(np.nanpercentile(pos, 5.0))
        vmax = float(np.nanpercentile(pos, 99.0))
        if vmax <= vmin:
            vmax = vmin * 10.0
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        im = ax.pcolormesh(R, Z, np.ma.masked_invalid(Emag), shading="auto", cmap="inferno", norm=norm)
    else:
        vmax = float(np.nanpercentile(finite, 99.0))
        if vmax <= 0.0:
            vmax = float(np.nanmax(finite))
        levels = np.linspace(0.0, vmax, 181)
        im = ax.contourf(R, Z, np.ma.masked_invalid(Emag), levels=levels, cmap="inferno", extend="max")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$|E_{sheath}|$ [V/m]")

    if core:
        cp = np.array(core, dtype=float)
        ax.plot(cp[:, 0], cp[:, 1], color="#3ea6ff", lw=1.2, zorder=4)
    if wall:
        wp = np.array(wall, dtype=float)
        ax.plot(wp[:, 0], wp[:, 1], color="black", lw=1.4, zorder=4)

    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.set_title("WEST case: |E_sheath| (R-Z)")

    out = (base / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    print(f"Wrote: {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
