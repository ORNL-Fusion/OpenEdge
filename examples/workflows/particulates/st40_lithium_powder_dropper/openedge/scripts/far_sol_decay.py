#!/usr/bin/env python3
"""Replace the SOLPS far-SOL plateau by an exponential decay away from the separatrix.

    python scripts/far_sol_decay.py input/plasma_13589.h5 3d/input/plasma_farsol.h5 --psi-cut 1.1 --lambda-n 0.02 --lambda-t 0.04

Beyond psi_N = psi_cut, n_e, n_i (and the ion densities) are multiplied by
exp(-(d - d_cut)/lambda_n) and T_e, T_i by exp(-(d - d_cut)/lambda_t), where d is the
distance from the separatrix estimated as (psi_N - 1)/|grad psi_N| at the cell. Flow,
B and E are left as they are; the drag on grains scales with the density."""
import argparse, shutil
import h5py, numpy as np
from scipy.interpolate import RegularGridInterpolator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--psi-cut", type=float, default=1.1)
    ap.add_argument("--lambda-n", type=float, default=0.02, help="density decay length [m]")
    ap.add_argument("--lambda-t", type=float, default=0.04, help="temperature decay length [m]")
    a = ap.parse_args()
    shutil.copy(a.src, a.dst)
    with h5py.File(a.dst, "r+") as f:
        eq = f["equilibrium"]; psi = eq["psi"][...]; r = eq["r"][...]; z = eq["z"][...]
        pa, pb = float(eq["psi_axis"][()]), float(eq["psib"][()])
        P = psi if psi.shape == (len(z), len(r)) else psi.T
        psin = (P - pa) / (pb - pa)
        gz, gr = np.gradient(psin, z, r)
        f_psin = RegularGridInterpolator((z, r), psin, bounds_error=False, fill_value=None)
        f_grad = RegularGridInterpolator((z, r), np.hypot(gr, gz), bounds_error=False, fill_value=None)
        g = f["mesh"]; tri = g["triangles"][...]; vr = g["vtx_r"][...]; vz = g["vtx_z"][...]; cidx = g["cell_index"][...]
        cr, cz = vr[tri].mean(1), vz[tri].mean(1)
        ncell = len(g["temp_e"]); R = np.zeros(ncell); Z = np.zeros(ncell); cnt = np.zeros(ncell)
        ok = (cidx >= 0) & (cidx < ncell)
        np.add.at(R, cidx[ok], cr[ok]); np.add.at(Z, cidx[ok], cz[ok]); np.add.at(cnt, cidx[ok], 1)
        R /= np.maximum(cnt, 1); Z /= np.maximum(cnt, 1)
        pn = f_psin(np.column_stack([Z, R])); grad = np.maximum(f_grad(np.column_stack([Z, R])), 1e-6)
        d = (pn - 1.0) / grad; d_cut = (a.psi_cut - 1.0) / grad
        far = (pn > a.psi_cut) & (cnt > 0)
        fn = np.ones(ncell); ft = np.ones(ncell)
        fn[far] = np.exp(-(d[far] - d_cut[far]) / a.lambda_n); ft[far] = np.exp(-(d[far] - d_cut[far]) / a.lambda_t)
        for name in ("mesh/dens_e", "mesh/dens_i", "b2/dens_e", "b2/dens_i"):
            if name in f and f[name].shape[0] == ncell: f[name][...] = f[name][...] * fn
        if "mesh/ions/dens" in f: f["mesh/ions/dens"][...] = f["mesh/ions/dens"][...] * fn
        for name in ("mesh/temp_e", "mesh/temp_i", "b2/temp_e", "b2/temp_i"):
            if name in f and f[name].shape[0] == ncell: f[name][...] = f[name][...] * ft
        f.attrs["far_sol_decay"] = f"psi_cut {a.psi_cut}, lambda_n {a.lambda_n} m, lambda_t {a.lambda_t} m"
        print(f"{far.sum()} of {ncell} cells beyond psi_N {a.psi_cut}; density factor 10/50/90%: "
              f"{np.percentile(fn[far], [10, 50, 90]).round(4)}; min {fn[far].min():.2e}")


if __name__ == "__main__":
    main()
