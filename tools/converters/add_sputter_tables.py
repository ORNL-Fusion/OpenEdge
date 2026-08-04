#!/usr/bin/env python3
"""Populate /surface/sputter/<pair>/{E,theta,Y} tables in processes.h5.

Data source: W. Eckstein, "Calculated Sputtering, Reflection and Range
Values", IPP 9/132 (June 2002), TRIM.SP tables, W target (sbe = 8.68 eV):

  d_on_w : p.211, full 9-angle table (rows with complete angular scans).
  n_on_w : p.221, full 9-angle table.
  o_on_w : p.224 gives NORMAL INCIDENCE ONLY (TPP 9/82, low fluence).
           The stored table is a documented hybrid: Y_O(E,0) from p.224
           scaled by the N->W angular shape Y_N(E,th)/Y_N(E,0) (nearest
           full-table projectile, m=14 vs 16). Replace with RustBCA /
           SDTrimSP data when available.

Angles are degrees from the surface NORMAL. Blank entries in the report
mean Y < 5e-6 and are stored as 0. Yields are atoms/ion.

Usage:  python3 add_sputter_tables.py [processes.h5]
"""
import sys
import numpy as np
import h5py

TH = [0.0, 15.0, 30.0, 45.0, 55.0, 65.0, 75.0, 80.0, 85.0]

# --- D -> W (IPP 9/132 p.211), rows with complete angular scans ----------
D_E = [250, 270, 300, 350, 400, 500, 600, 700, 1000]
D_Y = [
 [2.34e-5, 2.33e-5, 2.50e-5, 3.08e-5, 3.00e-5, 2.64e-5, 1.43e-5, 5.19e-6, 2.86e-6],
 [7.63e-5, 8.21e-5, 8.33e-5, 1.00e-4, 9.70e-5, 9.17e-5, 5.54e-5, 2.16e-5, 3.40e-6],
 [2.08e-4, 2.21e-4, 2.37e-4, 2.82e-4, 2.73e-4, 2.32e-4, 1.53e-4, 7.83e-5, 9.13e-6],
 [5.98e-4, 6.04e-4, 6.83e-4, 7.34e-4, 7.85e-4, 7.52e-4, 4.75e-4, 2.60e-4, 4.13e-5],
 [1.11e-3, 1.16e-3, 1.18e-3, 1.33e-3, 1.53e-3, 1.39e-3, 9.83e-4, 5.91e-4, 9.55e-5],
 [2.20e-3, 2.32e-3, 2.49e-3, 2.74e-3, 2.93e-3, 3.08e-3, 2.50e-3, 1.72e-3, 3.56e-4],
 [3.39e-3, 3.31e-3, 3.42e-3, 4.11e-3, 4.55e-3, 4.76e-3, 4.73e-3, 3.78e-3, 1.12e-3],
 [4.22e-3, 4.14e-3, 4.84e-3, 5.23e-3, 6.38e-3, 7.10e-3, 7.42e-3, 6.92e-3, 2.52e-3],
 [6.55e-3, 7.11e-3, 7.78e-3, 9.22e-3, 1.07e-2, 1.26e-2, 1.82e-2, 2.04e-2, 1.15e-2],
]

# --- N -> W (IPP 9/132 p.221) --------------------------------------------
N_E = [50, 52, 55, 60, 70, 80, 90, 100, 120, 140, 200, 300, 500, 1000]
N_Y = [
 [5.70e-5, 5.25e-5, 3.93e-5, 2.11e-5, 1.07e-5, 2.85e-6, 0.0,     0.0,     0.0    ],
 [1.35e-4, 1.28e-4, 9.18e-5, 5.06e-5, 2.86e-5, 9.30e-6, 0.0,     0.0,     0.0    ],
 [3.60e-4, 3.30e-4, 2.56e-4, 1.52e-4, 8.47e-5, 2.75e-5, 3.08e-6, 0.0,     0.0    ],
 [9.73e-4, 9.57e-4, 7.55e-4, 4.74e-4, 2.76e-4, 1.08e-4, 1.52e-5, 2.20e-6, 0.0    ],
 [3.26e-3, 3.26e-3, 2.84e-3, 2.02e-3, 1.31e-3, 5.87e-4, 1.08e-4, 1.98e-5, 1.50e-6],
 [7.00e-3, 6.70e-3, 6.06e-3, 4.61e-3, 3.18e-3, 1.61e-3, 3.52e-4, 7.13e-5, 6.20e-6],
 [1.17e-2, 1.14e-2, 1.06e-2, 8.58e-3, 6.13e-3, 3.36e-3, 8.56e-4, 2.12e-4, 1.76e-5],
 [1.72e-2, 1.70e-2, 1.60e-2, 1.30e-2, 1.03e-2, 6.07e-3, 1.78e-3, 4.90e-4, 4.29e-5],
 [2.77e-2, 2.79e-2, 2.82e-2, 2.53e-2, 2.17e-2, 1.43e-2, 5.43e-3, 1.70e-3, 1.23e-4],
 [3.99e-2, 4.07e-2, 4.14e-2, 4.06e-2, 3.50e-2, 2.65e-2, 1.14e-2, 3.67e-3, 2.47e-4],
 [7.57e-2, 8.00e-2, 8.32e-2, 8.70e-2, 9.07e-2, 7.77e-2, 4.11e-2, 1.50e-2, 8.58e-4],
 [1.32e-1, 1.35e-1, 1.48e-1, 1.71e-1, 1.85e-1, 1.80e-1, 1.15e-1, 4.56e-2, 2.65e-3],
 [2.13e-1, 2.21e-1, 2.52e-1, 3.10e-1, 3.51e-1, 3.68e-1, 2.66e-1, 1.26e-1, 8.67e-3],
 [3.39e-1, 3.58e-1, 4.22e-1, 5.35e-1, 6.24e-1, 6.89e-1, 5.80e-1, 3.49e-1, 4.00e-2],
]

# --- O -> W normal incidence (IPP 9/132 p.224, TPP 9/82 low fluence) -----
O_E  = [50, 100, 200, 300, 500, 1000, 2000, 5000, 6000]
O_Y0 = [3.63e-4, 2.17e-2, 9.00e-2, 1.52e-1, 2.45e-1, 3.71e-1, 5.33e-1, 6.89e-1, 7.64e-1]


def o_on_w_hybrid():
    """Y_O(E,th) = Y_O(E,0) * [Y_N(E,th)/Y_N(E,0)], N-shape log-E interp,
    clamped to the N table energy range."""
    nE = np.array(N_E, float)
    nY = np.array(N_Y, float)
    shape = nY / nY[:, :1]                      # angular shape per N energy
    out = np.zeros((len(O_E), len(TH)))
    for i, (e, y0) in enumerate(zip(O_E, O_Y0)):
        le = np.clip(np.log(e), np.log(nE[0]), np.log(nE[-1]))
        k = np.clip(np.searchsorted(np.log(nE), le) - 1, 0, len(nE) - 2)
        t = (le - np.log(nE[k])) / (np.log(nE[k+1]) - np.log(nE[k]))
        sh = (1 - t) * shape[k] + t * shape[k+1]
        out[i] = y0 * sh
    return np.array(O_E, float), out


def write(f, pair, E, Y, source):
    g = f.require_group(f"surface/sputter/{pair}")
    for name in ("E", "theta", "Y"):
        if name in g:
            del g[name]
    g.create_dataset("E", data=np.asarray(E, float))
    g.create_dataset("theta", data=np.asarray(TH, float))
    g.create_dataset("Y", data=np.asarray(Y, float))
    g.attrs["source"] = source
    g.attrs["units"] = "E: eV; theta: deg from surface normal; Y: atoms/ion"
    print(f"  {pair}: Y{np.asarray(Y).shape}  E=[{E[0]:g}..{E[-1]:g}] eV")


def write_grid(f, pair, E, THETA, Y, source):
    g = f.require_group(f"surface/sputter/{pair}")
    for name in ("E", "theta", "Y"):
        if name in g:
            del g[name]
    g.create_dataset("E", data=np.asarray(E, float))
    g.create_dataset("theta", data=np.asarray(THETA, float))
    g.create_dataset("Y", data=np.clip(np.asarray(Y, float), 0.0, None))
    g.attrs["source"] = source
    g.attrs["units"] = "E: eV; theta: deg from surface normal; Y: atoms/ion"
    print(f"  {pair}: Y{np.asarray(Y).shape}  E=[{E[0]:g}..{E[-1]:g}] eV "
          f"theta=[{THETA[0]:g}..{THETA[-1]:g}] deg")


def import_bca(f, pair, path, source):
    """Import a RustBCA/FTridyn surface file (E, A, spyld) as a table."""
    with h5py.File(path, "r") as b:
        E = b["E"][:]; A = b["A"][:]; Y = b["spyld"][:]
    write_grid(f, pair, E, A, Y, source)


# surface binding energies [eV] stamped as Es_eV on compound tables so the
# C++ side can Thompson-sample without an eckstein_sputter_data.h entry
ES_EV = {"b": 5.77, "w": 8.9, "o": 2.58, "c": 7.37, "be": 3.32}


def import_bca_compound(f, base, path, source):
    """Import a compound-target file (E, A, C, spyld_<Elem>...) as one 3D
    table per emitted species: /surface/sputter/<base>_<elem>/{E,theta,C,Y}
    with Y shaped (NC, NE, NTHETA)."""
    with h5py.File(path, "r") as b:
        E = b["E"][:]; A = b["A"][:]; C = b["C"][:]
        species = [k for k in b if k.startswith("spyld_")]
        for k in species:
            elem = k.split("_", 1)[1].lower()
            Y = np.clip(b[k][:], 0.0, None)
            name = f"{base}_{elem}"
            g = f.require_group(f"surface/sputter/{name}")
            for d in ("E", "theta", "C", "Y"):
                if d in g:
                    del g[d]
            g.create_dataset("E", data=np.asarray(E, float))
            g.create_dataset("theta", data=np.asarray(A, float))
            g.create_dataset("C", data=np.asarray(C, float))
            g.create_dataset("Y", data=np.asarray(Y, float))
            g.attrs["source"] = source
            g.attrs["units"] = ("E: eV; theta: deg from surface normal; "
                                "C: lead-element atomic fraction; Y: atoms/ion")
            if elem in ES_EV:
                g.attrs["Es_eV"] = ES_EV[elem]
            print(f"  {name}: Y{Y.shape}  E=[{E[0]:g}..{E[-1]:g}] eV "
                  f"C=[{C[0]:g}..{C[-1]:g}]")
        # composition-resolved reflection coefficient as a 3D table
        # (T-channel `rtable` override; probability only, outgoing
        # spectrum stays with the pure TRIM table)
        if "rfyld" in b:
            R = np.clip(b["rfyld"][:], 0.0, 1.0)
            name = f"{base}_refl"
            g = f.require_group(f"surface/sputter/{name}")
            for d in ("E", "theta", "C", "Y"):
                if d in g:
                    del g[d]
            g.create_dataset("E", data=np.asarray(E, float))
            g.create_dataset("theta", data=np.asarray(A, float))
            g.create_dataset("C", data=np.asarray(C, float))
            g.create_dataset("Y", data=np.asarray(R, float))
            g.attrs["source"] = source
            g.attrs["units"] = ("E: eV; theta: deg from surface normal; "
                                "C: lead-element atomic fraction; "
                                "Y: particle reflection coefficient")
            print(f"  {name}: R{R.shape}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/42d/OpenEdge/database/processes.h5"
    bca = "/Users/42d/Projects/code/OpenEdge/database/surface"
    with h5py.File(path, "r+") as f:
        write(f, "d_on_w", D_E, D_Y,
              "TRIM.SP, Eckstein IPP 9/132 (2002) p.211")
        write(f, "n_on_w", N_E, N_Y,
              "TRIM.SP, Eckstein IPP 9/132 (2002) p.221")
        import_bca(f, "o_on_w", f"{bca}/O_on_W.h5",
                   "RustBCA O->W (user-generated; database/surface/O_on_W.h5)")
        import_bca(f, "w_on_w", f"{bca}/74_on_74.h5",
                   "RustBCA W->W (user-generated; database/surface/74_on_74.h5)")
        bca2 = "/Users/42d/OpenEdge_/database/surface"
        import_bca(f, "b_on_b", f"{bca2}/B_on_B.h5",
                   "BCA B->B (user-generated; OpenEdge_ database/surface/B_on_B.h5)")
        import_bca(f, "d_on_c", f"{bca2}/D_on_C.h5",
                   "BCA D->C (user-generated; OpenEdge_ database/surface/D_on_C.h5)")
        # test_slag W/B mixed-material pairs, RustBCA 2.8.4 python bindings
        # (generated by RustBCA scripts/generate_openedge_pairs.py)
        bca3 = "/Users/42d/OpenEdge/database/surface"
        for pair in ("d_on_b", "o_on_b", "b_on_w", "w_on_b"):
            import_bca(f, pair, f"{bca3}/{pair}.h5",
                       f"RustBCA 2.8.4 {pair} (database/surface/{pair}.h5)")
        # compound B(c_B)W(1-c_B) targets -> per-species 3D tables
        # (skipped with a note while the generation runs are in flight)
        import os
        for base in ("d_on_bw", "o_on_bw", "b_on_bw", "w_on_bw",
                     "d_on_ow", "o_on_ow", "w_on_ow"):
            src = f"{bca3}/{base}.h5"
            if os.path.exists(src):
                import_bca_compound(f, base, src,
                    f"RustBCA 2.8.4 compound {base} (database/surface/{base}.h5)")
            else:
                print(f"  {base}: {src} not found, skipped")
    print(f"wrote sputter tables to {path}")


if __name__ == "__main__":
    main()
