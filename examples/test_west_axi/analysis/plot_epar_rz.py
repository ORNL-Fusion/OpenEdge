#!/usr/bin/env python3
from pathlib import Path


def read_last_grid_block(path: Path):
    lines = path.read_text().splitlines()
    i = 0
    last_rows = None
    last_cols = None

    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue

        i += 2  # skip timestep header + value
        if i >= len(lines) or not lines[i].startswith("ITEM: NUMBER OF CELLS"):
            break
        i += 1
        ncell = int(lines[i].strip())
        i += 1

        if i >= len(lines) or not lines[i].startswith("ITEM: BOX BOUNDS"):
            break
        i += 4  # header + 3 bounds lines

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


def build_mesh(cols, rows):
    try:
        ix = cols.index("xc")
        iy = cols.index("yc")
    except ValueError as exc:
        raise RuntimeError("Grid dump must include xc and yc") from exc

    fields = {}
    for j, name in enumerate(cols):
        if name.startswith("c_") or name.startswith("v_") or name.startswith("f_"):
            fields[name] = j
    if not fields:
        raise RuntimeError("No field columns found (c_*/v_*/f_*)")

    sample_triples = [(float(r[ix]), float(r[iy]), 0.0) for r in rows]
    r_vals = sorted({t[0] for t in sample_triples})
    z_vals = sorted({t[1] for t in sample_triples})

    import numpy as np

    nr = len(r_vals)
    nz = len(z_vals)
    R = np.array(r_vals, dtype=float).reshape(1, nr).repeat(nz, axis=0)
    Z = np.array(z_vals, dtype=float).reshape(nz, 1).repeat(nr, axis=1)
    meshes = {}
    for name, j in fields.items():
        val_map = {(float(r[ix]), float(r[iy])): float(r[j]) for r in rows}
        F = np.full((nz, nr), float("nan"), dtype=float)
        for iz, zv in enumerate(z_vals):
            for ir, rv in enumerate(r_vals):
                F[iz, ir] = val_map.get((rv, zv), float("nan"))
        meshes[name] = F
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
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np

    base = Path(__file__).resolve().parent
    dumpf = base / "output" / "epar.west.grid"
    cols, rows = read_last_grid_block(dumpf)
    R, Z, meshes = build_mesh(cols, rows)

    required = ["c_cwest[4]", "c_cwest[5]", "c_cwest[6]", "c_cwest[7]"]
    missing = [k for k in required if k not in meshes]
    if missing:
        raise RuntimeError("Missing required columns in dump: " + ", ".join(missing))

    Epar = meshes["c_cwest[4]"]
    Ne = meshes["c_cwest[5]"]
    Gtr = meshes["c_cwest[6]"]
    Gtz = meshes["c_cwest[7]"]
    Gt = np.sqrt(np.maximum(0.0, Gtr * Gtr + Gtz * Gtz))
    LogNe = np.full_like(Ne, np.nan)
    pos = Ne > 0.0
    LogNe[pos] = np.log10(Ne[pos])

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.7), dpi=200, constrained_layout=True)
    fig.patch.set_facecolor("#d9d9d9")
    for ax in axes:
        ax.set_facecolor("#d9d9d9")

    core = read_surface_polygon(base / "input" / "core.txt")
    wall = read_surface_polygon(base / "input" / "wall.txt")

    def decorate(ax):
        if core:
            cp = np.array(core, dtype=float)
            ax.plot(cp[:, 0], cp[:, 1], color="#3ea6ff", lw=1.2, zorder=4)
        if wall:
            wp = np.array(wall, dtype=float)
            ax.plot(wp[:, 0], wp[:, 1], color="black", lw=1.2, zorder=4)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.25)
        ax.minorticks_on()
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

    # Epar panel
    e_finite = Epar[np.isfinite(Epar)]
    e_vmax = float(np.nanpercentile(np.abs(e_finite), 99.0)) if e_finite.size else 1.0
    if e_vmax <= 0.0:
        e_vmax = 1.0
    e_norm = mpl.colors.TwoSlopeNorm(vmin=-e_vmax, vcenter=0.0, vmax=e_vmax)
    e_levels = np.linspace(-e_vmax, e_vmax, 181)
    im0 = axes[0].contourf(
        R, Z, np.ma.masked_invalid(Epar),
        levels=e_levels, cmap="coolwarm", norm=e_norm, extend="both"
    )
    cbar0 = fig.colorbar(im0, ax=axes[0])
    cbar0.set_label("Epar [V/m]")
    axes[0].set_title("Epar")
    decorate(axes[0])

    # |grad Te| panel
    g_finite = Gt[np.isfinite(Gt)]
    g_vmax = float(np.nanpercentile(g_finite, 99.0)) if g_finite.size else 1.0
    if g_vmax <= 0.0:
        g_vmax = 1.0
    g_levels = np.linspace(0.0, g_vmax, 181)
    im1 = axes[1].contourf(
        R, Z, np.ma.masked_invalid(Gt),
        levels=g_levels, cmap="viridis", extend="max"
    )
    cbar1 = fig.colorbar(im1, ax=axes[1])
    cbar1.set_label("|grad Te| [eV/m]")
    axes[1].set_title("|grad Te|")
    decorate(axes[1])

    # log10(ne) panel
    n_finite = LogNe[np.isfinite(LogNe)]
    if n_finite.size == 0:
        raise RuntimeError("No positive ne values in dump")
    n_vmin = float(np.nanpercentile(n_finite, 2.0))
    n_vmax = float(np.nanpercentile(n_finite, 98.0))
    if n_vmax <= n_vmin:
        n_vmin = float(np.nanmin(n_finite))
        n_vmax = float(np.nanmax(n_finite))
    n_levels = np.linspace(n_vmin, n_vmax, 181)
    im2 = axes[2].contourf(
        R, Z, np.ma.masked_invalid(LogNe),
        levels=n_levels, cmap="inferno", extend="both"
    )
    cbar2 = fig.colorbar(im2, ax=axes[2])
    cbar2.set_label("log10(ne [m^-3])")
    axes[2].set_title("log10(ne)")
    decorate(axes[2])

    fig.suptitle("WEST case fields (R-Z)", fontsize=14)

    out = base / "output" / "west_fields.rz.png"
    fig.savefig(out, dpi=220)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
