#!/usr/bin/env python3
"""
SOLPS heatflux -> OpenEdge converter (no quixote dependency).

Extracts heat-flux density from SOLPS face-centered fluxes read directly from
b2fstate/b2fgmtry, and writes heatflux.h5.

Usage:
    python solps_heatflux2openedge.py /path/to/solps_run \\
        --heatflux-out heatflux.h5 --method total --nr 300 --nz 300
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import griddata

# Import SOLPS parsers from the main converter
from convert_solps_plasma import b2f_read_dims, b2f_extract, read_b2fgmtry, \
    _cell_centers_from_corners, _regular_grid, _interp_field


def _read_b2mn_value(run_path: Path, key: str, default: float = 0.0) -> float:
    """Return a numeric b2mn.dat switch (quoted 'key' 'value' lines)."""
    b2mn = run_path / "b2mn.dat"
    if not b2mn.exists():
        return default
    pat = re.compile(r"^\s*'" + re.escape(key) + r"'\s+'([^']*)'")
    with b2mn.open("r", errors="ignore") as handle:
        for line in handle:
            m = pat.match(line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    return default
    return default


def _ionization_potentials() -> list[np.ndarray]:
    """Per-element single-step ionization potentials [eV]; row Z-1 holds Z values.

    Table copied from ORNL-Fusion SOLPS-routines (Matlab/utility)."""
    path = Path(__file__).resolve().parent / "ionization_potentials.txt"
    table = []
    with path.open("r") as handle:
        for line in handle:
            vals = [float(t.replace("d", "e").replace("D", "E")) for t in line.split()]
            table.append(np.asarray(vals, dtype=float))
    return table


def _b2_neighbour_index(gmtry: dict, name: str, nx: int, ny: int) -> np.ndarray:
    """leftix/leftiy/bottomix/bottomiy as 0-based (nx+2, ny+2) arrays; -1 = none."""
    arr = np.asarray(gmtry[name], dtype=float).reshape(nx + 2, ny + 2, order="F")
    return arr.astype(int) + 1


def compute_fht(run_path: Path, b2fgmtry_path: Path | None = None,
                b2fstate_path: Path | None = None) -> tuple[np.ndarray, dict]:
    """
    Total SOLPS energy flux through cell faces, W, shape (nx+2, ny+2, 2).

    Port of calc_fht from ORNL-Fusion SOLPS-routines (J. Lore):
      fht = fhe + fhi + fhj + fnt + fhp + fhm
    with fhj the current term (-fch*phi), fnt the kinetic enthalpy carried
    by the particle fluxes, fhp the ionization potential energy and fhm the
    drift kinetic energy.  Returns (fht, components).  If b2fstate already
    carries fht it is returned as is with the components that exist.
    """
    b2fstate = b2fstate_path if b2fstate_path is not None else run_path / "b2fstate"
    b2fgmtry = b2fgmtry_path if b2fgmtry_path is not None else run_path / "b2fgmtry"
    nx, ny, ns = b2f_read_dims(b2fstate)
    nx2, ny2 = nx + 2, ny + 2

    def field(name):
        return b2f_extract(name, b2fstate)

    fhe = field("fhe")
    fhi = field("fhi")
    components = {"fhe": fhe, "fhi": fhi}
    try:
        fht = field("fht")
        print("fht read directly from b2fstate")
        return fht, components
    except KeyError:
        pass

    te = field("te")           # J
    ti = field("ti")
    po = field("po")
    fch = field("fch")          # A, (nx2, ny2, 2)
    kinrgy = field("kinrgy")    # J, (nx2, ny2, ns)
    if kinrgy.ndim == 2:
        kinrgy = kinrgy[:, :, None]
    fna_raw = field("fna")      # (nx2, ny2, 2*ns): index 2*s + dir
    fna = fna_raw.reshape(nx2, ny2, 2, ns, order="F")
    try:
        ua = field("ua")
        if ua.ndim == 2:
            ua = ua[:, :, None]
    except KeyError:
        ua = np.zeros((nx2, ny2, ns))
    zamin = np.asarray(b2f_extract("zamin", b2fstate, reshape=False), dtype=float)
    zamax = np.asarray(b2f_extract("zamax", b2fstate, reshape=False), dtype=float)
    zn = np.asarray(b2f_extract("zn", b2fstate, reshape=False), dtype=float)

    # fne reconstructed from the current and the charge-weighted ion fluxes
    # (SOLPS-routines read_b2fstate; exact only when rza is uniform).
    ev = 1.602176487e-19
    fne = -fch / ev
    for s in range(ns):
        fne = fne + zamax[s] * fna[:, :, :, s]

    gm = read_b2fgmtry(str(b2fgmtry))
    leftix = _b2_neighbour_index(gm, "leftix", nx, ny)
    leftiy = _b2_neighbour_index(gm, "leftiy", nx, ny)
    bottomix = _b2_neighbour_index(gm, "bottomix", nx, ny)
    bottomiy = _b2_neighbour_index(gm, "bottomiy", nx, ny)
    boris = _read_b2mn_value(run_path, "b2news_BoRiS", 0.0)

    table = _ionization_potentials()
    rpt = np.zeros(ns)
    for s in range(ns):
        charge, z = int(round(zamin[s])), int(round(zn[s]))
        if charge > 0:
            rpt[s] = table[z - 1][charge - 1]

    # neighbour-averaged quantities on the west (dir 0) and south (dir 1) faces
    def west(a):
        out = np.zeros_like(a)
        ok = leftix >= 0
        out[ok] = a[leftix[ok], leftiy[ok]]
        return out, ok

    def south(a):
        out = np.zeros_like(a)
        ok = bottomiy >= 0
        out[ok] = a[bottomix[ok], bottomiy[ok]]
        return out, ok

    te_w, okw = west(te)
    te_s, oks = south(te)
    ti_w, _ = west(ti)
    ti_s, _ = south(ti)
    po_w, _ = west(po)
    po_s, _ = south(po)

    fhj = np.zeros((nx2, ny2, 2))
    fnt = np.zeros((nx2, ny2, 2))
    fhj[:, :, 0] = np.where(okw, -0.5 * fch[:, :, 0] * (po + po_w), 0.0)
    fhj[:, :, 1] = np.where(oks, -0.5 * fch[:, :, 1] * (po + po_s), 0.0)
    fnt[:, :, 0] = np.where(okw, 0.5 * fne[:, :, 0] * (te + te_w), 0.0)
    fnt[:, :, 1] = np.where(oks, 0.5 * fne[:, :, 1] * (te + te_s), 0.0)
    fnt_eonly = fnt.copy()

    fhm = np.zeros((nx2, ny2, 2))
    fhp = np.zeros((nx2, ny2, 2))
    for s in range(ns):
        k_w, _ = west(kinrgy[:, :, s])
        k_s, _ = south(kinrgy[:, :, s])
        # cvsa (viscous kinetic term) is not written to b2fstate: zero, as in
        # SOLPS-routines when the field is absent.
        fhm[:, :, 0] += np.where(okw, 0.5 * fna[:, :, 0, s] * (k_w + kinrgy[:, :, s]), 0.0)
        fhm[:, :, 1] += np.where(oks, 0.5 * fna[:, :, 1, s] * (k_s + kinrgy[:, :, s]), 0.0)
        fhp[:, :, 0] += np.where(okw, fna[:, :, 0, s] * rpt[s] * ev, 0.0)
        fhp[:, :, 1] += np.where(oks, fna[:, :, 1, s] * rpt[s] * ev, 0.0)
        fnt[:, :, 0] += np.where(okw, 0.5 * fna[:, :, 0, s] * (ti_w + ti), 0.0)
        fnt[:, :, 1] += np.where(oks, 0.5 * fna[:, :, 1, s] * (ti_s + ti), 0.0)

    fnt = fnt * (1.0 - boris)
    fhm = fhm * (1.0 - boris)
    fht = fhe + fhi + fhj + fnt + fhp + fhm
    components.update({"fhj": fhj, "fnt": fnt, "fnt_eonly": fnt_eonly,
                       "fhp": fhp, "fhm": fhm})
    print("fht reconstructed (fhe+fhi+fhj+fnt+fhp+fhm; SOLPS-routines calc_fht)")
    return fht, components


def compute_heatflux_cell_center(run_path: Path, method: str = "total",
                                 b2fgmtry_path: Path | None = None,
                                 b2fstate_path: Path | None = None) -> tuple:
    """
    Compute heat-flux density from SOLPS face fluxes on a cell-centered grid.

    Methods (all use the parallel contact area |pbs_x| = sx*Bpol/B):
      - total:        |fht_x / pbs_x|, fht = total energy flux (compute_fht)
      - no_jv:        |(fht_x - fhj_x) / pbs_x|, total without the current term
      - fluid:        |(fhe_x + fhi_x) / pbs_x|, electron+ion fluid fluxes only
      - vector_mag:   sqrt((fht_x/sx)^2 + (fht_y/sy)^2) (deposited-style, legacy)
    Legacy aliases: jeremy_total -> total, jeremy_no_jv -> no_jv.

    Returns (rc, zc, qmag) where rc/zc are cell centers and qmag is heat flux.
    """
    aliases = {"jeremy_total": "total", "jeremy_no_jv": "no_jv"}
    method = aliases.get(method, method)
    if method not in ("total", "no_jv", "fluid", "vector_mag"):
        raise ValueError(f"Unknown heatflux method: {method}")

    b2fstate = b2fstate_path if b2fstate_path is not None else run_path / "b2fstate"
    b2fgmtry = b2fgmtry_path if b2fgmtry_path is not None else (run_path / "b2fgmtry")

    nx, ny, ns = b2f_read_dims(b2fstate)
    gmtry = read_b2fgmtry(str(b2fgmtry))
    rc, zc = _cell_centers_from_corners(gmtry["crx"], gmtry["cry"], nx, ny)

    if method == "fluid":
        flux = b2f_extract("fhe", b2fstate) + b2f_extract("fhi", b2fstate)
        print("heatflux=fluid: fhe + fhi only (no current, kinetic or "
              "potential-energy terms)")
    else:
        try:
            fht, comp = compute_fht(run_path, b2fgmtry, b2fstate)
        except KeyError as exc:
            print(f"WARNING: cannot reconstruct fht ({exc}); falling back to "
                  "fhe + fhi, which under-estimates the target heat flux")
            fht = b2f_extract("fhe", b2fstate) + b2f_extract("fhi", b2fstate)
            comp = {}
        flux = fht
        if method == "no_jv":
            if "fhj" in comp:
                flux = fht - comp["fhj"]
            else:
                print("fhj unavailable; no_jv equals total")

    # Parallel contact area from geometry: pbs(:,:,0) = (B_pol/|B|) * sx.
    try:
        pbs = b2f_extract("pbs", b2fgmtry)
        if pbs.ndim == 2:
            pbs = pbs.reshape(nx + 2, ny + 2, -1, order="F")
        sxprl = np.abs(pbs[:, :, 0])
    except KeyError as exc:
        raise KeyError(
            "Variable 'pbs' not found in b2fgmtry; cannot compute SOLPS "
            "parallel heat flux."
        ) from exc

    # Face -> cell: a cell's value is the mean of its two poloidal faces
    # (west face ix, east face ix+1).  Average on the full (nx+2, ny+2)
    # array first so the last real column keeps its target face, then drop
    # the guard cells (ix=0,nx+1 and iy=0,ny+1).
    s = np.s_[1:-1, 1:-1]
    rc, zc = rc[s], zc[s]

    def _cell_mean_x(face):
        return 0.5 * (face[:-1, :] + face[1:, :])[1:nx + 1, 1:ny + 1]

    def _cell_mean_y(face):
        return 0.5 * (face[:, :-1] + face[:, 1:])[1:nx + 1, 1:ny + 1]

    if method in ("total", "no_jv", "fluid"):
        qx = np.divide(flux[:, :, 0], sxprl, out=np.full_like(sxprl, np.nan),
                       where=sxprl > 0)
        return rc, zc, np.abs(_cell_mean_x(qx))

    # vector_mag: poloidal/radial face areas
    def _face_area(state_name, gs_col, h_name):
        try:
            return b2f_extract(state_name, b2fstate)
        except KeyError:
            try:
                gs = b2f_extract("gs", b2fgmtry)
                if gs.ndim == 2:
                    gs = gs.reshape(nx + 2, ny + 2, -1, order="F")
                print(f"Using gs(:,:,{gs_col}) from b2fgmtry as face area")
                return gs[:, :, gs_col]
            except KeyError:
                print(f"WARNING: Using {h_name} from b2fgmtry as face area fallback")
                return b2f_extract(h_name, b2fgmtry)
    sx = _face_area("sx", 0, "hx")
    sy = _face_area("sy", 1, "hy")
    qx = np.divide(flux[:, :, 0], sx, out=np.full_like(sx, np.nan), where=sx > 0)
    qy = np.divide(flux[:, :, 1], sy, out=np.full_like(sy, np.nan), where=sy > 0)
    return rc, zc, np.hypot(_cell_mean_x(qx), _cell_mean_y(qy))


def convert_solps_heatflux(
    run_path: Path,
    heatflux_out: Path,
    nr: int = 300,
    nz: int = 300,
    rmin: float | None = None,
    rmax: float | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    method: str = "total",
    plot: bool = False,
    plot_out: Path | None = None,
) -> None:
    rc, zc, qmag = compute_heatflux_cell_center(run_path, method)
    r, z, rr, zz = _regular_grid(rc, zc, nr, nz, rmin, rmax, zmin, zmax)

    src_pts = np.column_stack((rc.reshape(-1), zc.reshape(-1)))
    tgt_pts = np.column_stack((rr.reshape(-1), zz.reshape(-1)))
    qmag_grid = _interp_field(src_pts, qmag, tgt_pts, nz, nr)
    qmag_grid = np.nan_to_num(qmag_grid, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Compute grad_Te for rocket force direction ---
    # Read Te from b2fstate (in eV), interpolate to regular grid, take gradient
    b2fstate = run_path / "b2fstate"
    nx_d, ny_d, ns_d = b2f_read_dims(b2fstate)
    try:
        te_raw = b2f_extract("te", b2fstate)  # (nx+2, ny+2)
        if te_raw.ndim == 3:
            te_raw = te_raw[:, :, 0]  # take first species if multi-species
        te_cc = te_raw[1:-1, 1:-1]  # clip guard cells
        te_grid = _interp_field(src_pts, te_cc, tgt_pts, nz, nr)
        te_grid = np.nan_to_num(te_grid, nan=0.0)

        dR = r[1] - r[0]
        dZ = z[1] - z[0]
        grad_te_r_grid = np.gradient(te_grid, dR, axis=1)  # eV/m
        grad_te_z_grid = np.gradient(te_grid, dZ, axis=0)  # eV/m
        has_grad_te = True
        print(f"  Computed grad_Te: |grad_Te_r|_max={np.nanmax(np.abs(grad_te_r_grid)):.1f} eV/m, "
              f"|grad_Te_z|_max={np.nanmax(np.abs(grad_te_z_grid)):.1f} eV/m")
    except (KeyError, Exception) as e:
        print(f"  Warning: could not compute grad_Te ({e}), rocket force data not written")
        has_grad_te = False

    with h5py.File(heatflux_out, "w") as f:
        f.create_dataset("r", data=r)
        f.create_dataset("z", data=z)
        gg = f.create_group("grid")
        gf = f.create_group("fields")
        rr2d, zz2d = np.meshgrid(r, z)
        gg.create_dataset("R", data=rr2d)
        gg.create_dataset("Z", data=zz2d)
        gg.create_dataset("Rc", data=r)
        gg.create_dataset("Zc", data=z)
        gf.create_dataset("q_mag", data=qmag_grid)
        f.create_dataset("R", data=rr2d)
        f.create_dataset("Z", data=zz2d)
        f.create_dataset("q_mag", data=qmag_grid)
        if has_grad_te:
            f.create_dataset("grad_te_r", data=grad_te_r_grid)
            f.create_dataset("grad_te_z", data=grad_te_z_grid)

    print(f"Wrote heatflux: {heatflux_out} (method={method})")
    print(f"Grid: nr={nr}, nz={nz}, R=[{r[0]:.4f},{r[-1]:.4f}], Z=[{z[0]:.4f},{z[-1]:.4f}]")

    if plot:
        import matplotlib.pyplot as plt
        extent = [float(r.min()), float(r.max()), float(z.min()), float(z.max())]
        fig, ax = plt.subplots(1, 1, figsize=(6, 8), constrained_layout=True)
        log_q = np.full_like(qmag_grid, np.nan)
        m = qmag_grid > 0
        log_q[m] = np.log10(qmag_grid[m])
        im = ax.imshow(log_q, origin="lower", extent=extent, aspect="auto", cmap="inferno")
        ax.set_title(f"log10(q_mag) [W/m^2] ({method})")
        ax.set_xlabel("R [m]"); ax.set_ylabel("Z [m]")
        fig.colorbar(im, ax=ax, shrink=0.9)
        out_png = plot_out or Path("solps_heatflux.png")
        fig.savefig(out_png, dpi=180); plt.close(fig)
        print(f"Wrote plot: {out_png}")


def main():
    p = argparse.ArgumentParser(description="Extract SOLPS heatflux to OpenEdge HDF5 (no quixote)")
    p.add_argument("run_path", type=Path, help="SOLPS run directory")
    p.add_argument("--heatflux-out", type=Path, default=Path("heatflux.h5"))
    p.add_argument("--nr", type=int, default=300)
    p.add_argument("--nz", type=int, default=300)
    p.add_argument("--rmin", type=float, default=None)
    p.add_argument("--rmax", type=float, default=None)
    p.add_argument("--zmin", type=float, default=None)
    p.add_argument("--zmax", type=float, default=None)
    p.add_argument("--method", default="total",
                   choices=["total", "no_jv", "fluid", "vector_mag",
                            "jeremy_no_jv", "jeremy_total"])
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-out", type=Path, default=None)
    args = p.parse_args()
    convert_solps_heatflux(**vars(args))


if __name__ == "__main__":
    main()
