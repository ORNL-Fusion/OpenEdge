"""Diagnostic: plot SOLPS fort.31 wall-face ion flux density along the wall.

Reads /fort31/fnixb, /fort31/fniyb, /mesh/wall_face_area from plasma.h5,
projects the staggered fluxes onto the three wall families
(ix=1 inner target, ix=nx outer target, iy=ny outer SOL) with sign so
positive = into wall, divides by the toroidally-integrated face area to
get m^-2 s^-1, then plots vs poloidal arc length along the wall.

If this is smooth, the jaggedness in c_cpmiLi is downstream
(compute surface/physical/sputter reconstructing flux from n*cs*sin_alpha).
If it's jagged, the converter is the source.
"""
import argparse
import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plasma", default="plasma.h5")
    ap.add_argument("--out", default="wall_face_gamma.png")
    ap.add_argument("--species", default="D+,Ne+,Li+",
                    help="comma-separated species names to plot")
    args = ap.parse_args()

    with h5py.File(args.plasma, "r") as f:
        if "b2/nx" in f:
            nx = int(f["b2/nx"][()])
            ny = int(f["b2/ny"][()])
        else:
            nx = int(f["fort31"].attrs["nx"])
            ny = int(f["fort31"].attrs["ny"])
        ion_names = [n.decode() if isinstance(n, bytes) else n
                     for n in f["ion_species/names"][:]]
        fnixb = f["fort31/fnixb"][...]   # (nx+2, ny+2, nion)
        fniyb = f["fort31/fniyb"][...]
        wall_face_area = f["mesh/wall_face_area"][...]  # flat (ny+2)*(nx+2), F-order
        teb = f["fort31/teb"][...]
        tib = f["fort31/tib"][...]
        if "b2/r_center" in f:
            rc = f["b2/r_center"][...]
            zc = f["b2/z_center"][...]
        else:
            rc = f["fort31/rrb"][...].reshape(-1, order="F")  # geometric R, m
            # No z_center in this file; reconstruct from EIRENE mesh per-cell mean
            # via mesh/cell_index -> mesh/vtx_z. Fall back: use teb proxy if absent.
            if "mesh/cell_index" in f and "mesh/vtx_z" in f:
                cidx = f["mesh/cell_index"][...]
                vz = f["mesh/vtx_z"][...]
                tri = f["mesh/triangles"][...]
                tri_z = vz[tri].mean(axis=1)
                ncell = (nx + 2) * (ny + 2)
                zc = np.zeros(ncell)
                cnt = np.zeros(ncell, dtype=int)
                for t, c in enumerate(cidx):
                    if 0 <= c < ncell:
                        zc[c] += tri_z[t]
                        cnt[c] += 1
                with np.errstate(invalid="ignore"):
                    zc = np.where(cnt > 0, zc / np.maximum(cnt, 1), 0.0)
            else:
                zc = np.zeros_like(rc)

    nion = fnixb.shape[2]
    nxg, nyg = nx + 2, ny + 2
    ncell_flat = nxg * nyg
    assert wall_face_area.shape[0] == ncell_flat

    # Build per-species wall-face flux density (m^-2 s^-1, positive into wall).
    # fort31 staggered convention (verified empirically — read_fort31.py docstring
    # says "left face" but the data shows fnixb[ix, iy, s] is the RIGHT face of
    # cell (ix, iy)): fnixb[nx+1] / fniyb[:, ny+1] are zero ("beyond the wall"),
    # while the actual wall fluxes live at fnixb[0]/fnixb[nx]/fniyb[:, ny].
    flux = np.zeros((nxg, nyg, nion), dtype=np.float64)
    flux[1,  :, :] += -fnixb[0,  :, :]      # inner target (ix=1, wall = right face of ix=0)
    flux[nx, :, :] += +fnixb[nx, :, :]      # outer target (ix=nx, wall = right face of ix=nx)
    flux[:, ny, :] += +fniyb[:, ny, :]      # outer SOL    (iy=ny, wall = top face of iy=ny)

    safe_area = np.where(wall_face_area > 0.0, wall_face_area, 1.0)
    gamma = np.zeros((nion, ncell_flat), dtype=np.float64)
    for s in range(nion):
        flat_s = flux[:, :, s].reshape(ncell_flat, order="F")
        gamma[s] = np.where(wall_face_area > 0.0, flat_s / safe_area, 0.0)

    # Wall-cell ordering: walk the boundary in poloidal order, pieced from
    # the three families. Use cell centers (R, Z) as the 1D coord seed.
    rc2 = rc.reshape(nyg, nxg)   # rc is (ncell_flat,) F-order -> reshape to (ny+2, nx+2)
    zc2 = zc.reshape(nyg, nxg)
    teb2 = teb  # already (nx+2, ny+2)
    tib2 = tib

    wall_cells = []   # list of (ix, iy, family_label)
    # inner target: ix=1, iy = 1..ny (skip guard rows)
    for iy in range(1, ny + 1):
        wall_cells.append((1, iy, "inner_target"))
    # outer SOL: iy=ny, ix = 1..nx
    for ix in range(1, nx + 1):
        wall_cells.append((ix, ny, "outer_sol"))
    # outer target: ix=nx, iy = ny..1 (reverse so the path closes)
    for iy in range(ny, 0, -1):
        wall_cells.append((nx, iy, "outer_target"))

    # 1D coordinate: arc length along the wall path
    rs = np.array([rc2[iy, ix] for ix, iy, _ in wall_cells])
    zs = np.array([zc2[iy, ix] for ix, iy, _ in wall_cells])
    ds = np.hypot(np.diff(rs), np.diff(zs))
    s_coord = np.concatenate([[0.0], np.cumsum(ds)])

    cell_flat = np.array(
        [iy * nxg + ix for ix, iy, _ in wall_cells], dtype=np.int64)
    family = np.array([fam for _, _, fam in wall_cells])

    # Pull flux per species along the path
    species_to_plot = [s.strip() for s in args.species.split(",") if s.strip()]
    sidx = {n: i for i, n in enumerate(ion_names)}

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_g, ax_t = axes

    for sp in species_to_plot:
        if sp not in sidx:
            print(f"WARN: species {sp!r} not in plasma.h5; choices = {ion_names}")
            continue
        s = sidx[sp]
        g_along = gamma[s, cell_flat]
        ax_g.plot(s_coord, g_along, lw=1.0, label=sp)

    # Te/Ti on same axis
    te_along = np.array([teb2[ix, iy] for ix, iy, _ in wall_cells])
    ti_along = np.array([tib2[ix, iy] for ix, iy, _ in wall_cells])
    ax_t.plot(s_coord, te_along, lw=1.0, label=r"$T_e$ (eV)")
    ax_t.plot(s_coord, ti_along, lw=1.0, label=r"$T_i$ (eV)")

    # Mark family boundaries
    for ax in axes:
        # transitions where family changes
        for i in range(1, len(family)):
            if family[i] != family[i-1]:
                ax.axvline(s_coord[i], color="k", lw=0.5, alpha=0.4)

    ax_g.set_ylabel(r"$\Gamma_i$ at wall face (m$^{-2}$ s$^{-1}$)")
    ax_g.set_yscale("symlog", linthresh=1e16)
    ax_g.legend(frameon=False, fontsize=9)
    ax_g.set_title(f"SOLPS fort31-derived wall-face ion flux  ({args.plasma})")

    ax_t.set_ylabel("Temperature (eV)")
    ax_t.set_xlabel("Poloidal arc length along wall (m)")
    ax_t.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print(f"Wrote {args.out}")
    print(f"Wall-cell counts: inner_target={int((family=='inner_target').sum())} "
          f"outer_sol={int((family=='outer_sol').sum())} "
          f"outer_target={int((family=='outer_target').sum())}")
    print(f"Per-species peak |gamma| at wall (m^-2 s^-1):")
    for sp in species_to_plot:
        if sp in sidx:
            s = sidx[sp]
            g_along = gamma[s, cell_flat]
            print(f"  {sp:>6}: peak={np.abs(g_along).max():.3e}  "
                  f"mean={np.abs(g_along).mean():.3e}")


if __name__ == "__main__":
    main()
