#!/usr/bin/env python3
"""Export the LTX neutral sources of in.openedge to one R-Z grid file.

Reads the last output/source_Li.*, output/source_H.* and output/wall.* dumps
(units flight: the chem-fix columns are already the steady-state source) and
writes output/ltx_li_sources.h5 (+ .csv):

  r, z, cell_volume, psi_n           uniform grid, 5 mm, (nz, nr) arrays
  S_p [m^-3 s^-1]                    Li+ particle source
  S_m_r, S_m_z, S_m_phi, S_m_par     momentum source [N m^-3]
  Q_e, Q_i [W m^-3]                  electron / ion energy source
  n_Li0 [m^-3]                       neutral Li density
  H/*                                the same for H+, with n_H0, n_H2, n_H2plus
  wall/*                             per element: erosion flux and rate, H+ flux, impact energy, yield
  convergence/*, H/convergence/*     build-up of the integrals along the flight
  totals (attributes)                volume integrals
"""
from __future__ import annotations
import argparse, glob, os, sys
import numpy as np

CASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TWO_PI = 2.0 * np.pi
NAMES = ["S_p", "S_m_z", "S_m_r", "S_m_phi", "Q_e", "Q_i"]          # chem-fix columns 1-6 (x = Z, y = R)
UNITS = {"S_p": "m^-3 s^-1", "S_m_r": "N m^-3", "S_m_z": "N m^-3", "S_m_phi": "N m^-3",
         "S_m_par": "N m^-3", "Q_e": "W m^-3", "Q_i": "W m^-3", "n_Li0": "m^-3",
         "n_H0": "m^-3", "n_H2": "m^-3", "n_H2plus": "m^-3", "psi_n": "1"}

def read_dump(path):
    """One SPARTA grid/surf dump snapshot -> (step, header list, array)."""
    with open(path) as fh:
        lines = fh.readlines()
    step = None
    for i, ln in enumerate(lines):
        if ln.startswith("ITEM: TIMESTEP"):
            step = int(lines[i + 1])
        elif ln.startswith("ITEM: CELLS") or ln.startswith("ITEM: SURFS"):
            cols = ln.split()[2:]
            data = np.loadtxt(lines[i + 1:], ndmin=2)
            return step, cols, data
    raise RuntimeError(f"no CELLS/SURFS section in {path}")


def sorted_files(outdir, prefix):
    files = glob.glob(os.path.join(outdir, prefix + ".*"))
    files = [f for f in files if f.rsplit(".", 1)[1].isdigit()]
    return sorted(files, key=lambda p: int(p.rsplit(".", 1)[1]))


def deposit(geom, vals, zedges, redges):
    """Conservative deposition of per-cell intensive values onto the uniform
    grid: target = sum_i value_i * V(overlap_i) / V(target), with the
    axisymmetric volume V = pi (r2^2 - r1^2) dz. Returns (nz, nr, nval)."""
    nz, nr = len(zedges) - 1, len(redges) - 1
    out = np.zeros((nz, nr, vals.shape[1]))
    vtar = np.zeros((nz, nr))
    dz = zedges[1] - zedges[0]
    dr = redges[1] - redges[0]
    r1g, r2g = redges[:-1], redges[1:]
    vtar[:] = (np.pi * (r2g**2 - r1g**2) * dz)[None, :]
    xlo, ylo, xhi, yhi, vol = geom[:, 2], geom[:, 3], geom[:, 4], geom[:, 5], geom[:, 6]
    # cells cut by the wall: only the flow volume carries the source, so the
    # amount value*vol is spread over the box in proportion to the overlap
    vbox = np.pi * (yhi**2 - ylo**2) * (xhi - xlo)
    frac = np.where(vbox > 0, np.minimum(vol / np.maximum(vbox, 1e-300), 1.0), 0.0)
    for i in range(len(vals)):
        if not np.any(vals[i]):
            continue
        iz0 = max(int(np.floor((xlo[i] - zedges[0]) / dz)), 0)
        iz1 = min(int(np.ceil((xhi[i] - zedges[0]) / dz)), nz)
        ir0 = max(int(np.floor((ylo[i] - redges[0]) / dr)), 0)
        ir1 = min(int(np.ceil((yhi[i] - redges[0]) / dr)), nr)
        for iz in range(iz0, iz1):
            oz = min(xhi[i], zedges[iz + 1]) - max(xlo[i], zedges[iz])
            if oz <= 0:
                continue
            for ir in range(ir0, ir1):
                ra, rb = max(ylo[i], redges[ir]), min(yhi[i], redges[ir + 1])
                if rb <= ra:
                    continue
                out[iz, ir] += vals[i] * frac[i] * np.pi * (rb**2 - ra**2) * oz
    out /= vtar[:, :, None]
    return out, vtar


def psi_and_b_on_grid(plasma_file, rc, zc):
    """psi_n and the unit B vector on the export grid from equilibrium/*."""
    import h5py
    from scipy.interpolate import RegularGridInterpolator as RGI
    with h5py.File(plasma_file, "r") as f:
        eq = f["equilibrium"]
        r, z, psi = eq["r"][...], eq["z"][...], eq["psi"][...]
        br, bz, bt = eq["br"][...], eq["bz"][...], eq["bt"][...]
        psib = float(np.asarray(eq["psib"][...]).reshape(-1)[0])
        psia = float(np.asarray(eq["psi_axis"][...]).reshape(-1)[0])
    if r.ndim == 2:
        r1, z1 = r[0, :], z[:, 0]
    else:
        r1, z1 = r, z
    pts = np.stack(np.meshgrid(zc, rc, indexing="ij"), axis=-1).reshape(-1, 2)
    def interp(a):
        return RGI((z1, r1), a, bounds_error=False, fill_value=np.nan)(pts).reshape(len(zc), len(rc))
    psin = (interp(psi) - psia) / (psib - psia)
    b = np.stack([interp(br), interp(bz), interp(bt)])
    return psin, b / np.maximum(np.linalg.norm(b, axis=0), 1e-30)



def last_dump(outdir, prefix):
    files = [f for f in sorted_files(outdir, prefix) if int(f.rsplit(".", 1)[1]) > 0]
    if not files:
        sys.exit(f"no {prefix}.* in {outdir}; run in.openedge first")
    return files


def read_sources(files, ncol):
    """Last dump -> (geom, values[ncell, ncol]); every dump -> integrals (build-up)."""
    hist = []
    for f in files:
        step, cols, data = read_dump(f)
        idx = {c: k for k, c in enumerate(cols)}
        data = data[np.argsort(data[:, idx["id"]])]
        fcols = [c for c in cols if c.startswith("f_")]
        if len(fcols) < ncol:
            raise RuntimeError(f"{f}: expected {ncol} fix columns, found {fcols}")
        vol = data[:, idx["vol"]]
        vals = data[:, [idx[c] for c in fcols[:ncol]]]
        hist.append((step, (vals * vol[:, None]).sum(0)))
    geom = data[:, [idx[c] for c in ("xc", "yc", "xlo", "ylo", "xhi", "yhi", "vol")]]
    return geom, vals, hist


def wall_table(fn):
    step, cols, data = read_dump(fn)
    idx = {c: k for k, c in enumerate(cols)}
    data = data[np.argsort(data[:, idx["id"]])]
    g = lambda c: data[:, idx[c]]
    z1, r1, z2, r2 = g("v1x"), g("v1y"), g("v2x"), g("v2y")
    length = np.hypot(z2 - z1, r2 - r1)
    arc = np.concatenate([[0.0], np.cumsum(length)[:-1]]) + 0.5 * length
    rm = 0.5 * (r1 + r2)
    c = [k for k in cols if k.startswith("c_")]
    return {"R": rm, "Z": 0.5 * (z1 + z2), "arc": arc, "area": TWO_PI * rm * length,
            "erosion_flux": g(c[0]), "erosion_rate": g(c[1]), "flux_Hplus": g(c[2]),
            "E_impact_Hplus": g(c[3]), "yield_H_on_Li": g(c[4])}


def fields_from(geom, vals, zedges, redges, dens_names, bhat):
    grid, vtar = deposit(geom, vals, zedges, redges)
    f = {n: grid[:, :, k] for k, n in enumerate(NAMES + dens_names)}
    f["S_m_par"] = f["S_m_r"] * bhat[0] + f["S_m_z"] * bhat[1] + f["S_m_phi"] * bhat[2]
    return f, vtar


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="output")
    ap.add_argument("--dr", type=float, default=0.005)
    ap.add_argument("--dz", type=float, default=0.005)
    ap.add_argument("--plasma", default="input/plasma_ltx.h5")
    a = ap.parse_args()
    outdir = os.path.join(CASE, a.out)

    li_files = last_dump(outdir, "source_Li")
    h_files = last_dump(outdir, "source_H")
    geom, li, li_hist = read_sources(li_files, 7)
    hgeom, hv, h_hist = read_sources(h_files, 9)
    wall = wall_table(last_dump(outdir, "wall")[-1])

    redges = np.arange(0.10, 0.70 + 0.5 * a.dr, a.dr)
    zedges = np.arange(-0.40, 0.40 + 0.5 * a.dz, a.dz)
    rc, zc = 0.5 * (redges[1:] + redges[:-1]), 0.5 * (zedges[1:] + zedges[:-1])
    psin, bhat = psi_and_b_on_grid(os.path.join(CASE, a.plasma), rc, zc)
    F, vtar = fields_from(geom, li, zedges, redges, ["n_Li0"], bhat)
    H, _ = fields_from(hgeom, hv, zedges, redges, ["n_H0", "n_H2", "n_H2plus"], bhat)
    F["psi_n"] = psin

    tot = {}
    for k, n in enumerate(NAMES):
        tot[n] = float(li_hist[-1][1][k]); tot["H_" + n] = float(h_hist[-1][1][k])
    tot["N_Li0"] = float(li_hist[-1][1][6])
    tot["wall_erosion_rate"] = float(wall["erosion_rate"].sum())
    tot["last_step"] = int(li_hist[-1][0])
    print(f"Li: int S_p {tot['S_p']:.4e} /s  Q_e {tot['Q_e']:.4e} W  Q_i {tot['Q_i']:.4e} W  "
          f"(eroded by H+ {tot['wall_erosion_rate']:.4e} /s)")
    print(f"H : int S_p {tot['H_S_p']:.4e} /s  Q_e {tot['H_Q_e']:.4e} W  Q_i {tot['H_Q_i']:.4e} W  flight {tot['last_step']} steps")

    import h5py
    h5 = os.path.join(outdir, "ltx_li_sources.h5")
    with h5py.File(h5, "w") as f:
        f.attrs["description"] = ("LTX-beta neutral sources from OpenEdge on the SOLEDGE3X background "
                                  "(shot 103955). Plasma-frame source moments, positive = gain for the "
                                  "Li+ / H+ fluid, electrons, ions. Puff 6.1e20 atoms/s (effective value "
                                  "of the reference run); wall ion flux 7.1e20 /s (reference 6.8e20).")
        f.attrs["layout"] = "arrays are (nz, nr); momentum components in the (R, Z, phi) cylindrical frame"
        f.create_dataset("r", data=rc).attrs["units"] = "m"
        f.create_dataset("z", data=zc).attrs["units"] = "m"
        f.create_dataset("cell_volume", data=vtar).attrs["units"] = "m^3 (full toroidal revolution)"
        for n, arr in F.items():
            f.create_dataset(n, data=arr).attrs["units"] = UNITS[n]
        hg = f.create_group("H")
        for n, arr in H.items():
            hg.create_dataset(n, data=arr).attrs["units"] = UNITS[n]
        for grp, hist in ((f, li_hist), (hg, h_hist)):
            c = grp.create_group("convergence")
            c.attrs["note"] = "integrals accumulated along the flight; the last entry is the source"
            c.create_dataset("step", data=[h[0] for h in hist])
            for k, n in enumerate(NAMES):
                c.create_dataset("int_" + n, data=[h[1][k] for h in hist])
        t = f.create_group("totals")
        for k, v in tot.items():
            t.attrs[k] = v
        w = f.create_group("wall")
        w.attrs["note"] = "per wall element; arc = distance along the wall from element 1"
        for k, v in wall.items():
            w.create_dataset(k, data=v)
    print("wrote", h5)

    names = ["S_p", "S_m_r", "S_m_z", "S_m_phi", "S_m_par", "Q_e", "Q_i", "n_Li0", "psi_n"]
    ZZ, RR = np.meshgrid(zc, rc, indexing="ij")
    table = np.column_stack([RR.ravel(), ZZ.ravel()] + [F[n].ravel() for n in names])
    csv = os.path.join(outdir, "ltx_li_sources.csv")
    np.savetxt(csv, table, delimiter=",", header="R_m,Z_m," + ",".join(f"{n}[{UNITS[n]}]" for n in names),
               comments="", fmt="%.6e")
    print("wrote", csv)


if __name__ == "__main__":
    main()
