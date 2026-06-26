#!/usr/bin/env python3
"""Diagnose the plasma.h5 -> wall.surf gap that's underfeeding the
OpenEdge sputter compute.

For each divertor wall segment (lower divertor only), sample the
plasma.h5 mesh values (Te, ne, ni, parr_flow) at the segment centroid
the way fix_background does internally:
  1. find the SOLPS mesh triangle CONTAINING the centroid; if no
     triangle contains it (centroid sits outside the SOLPS mesh —
     i.e. in the gap between mesh boundary and the wall), fall back
     to the nearest triangle by barycenter distance and report the
     gap distance.
  2. read /mesh/{temp_e, dens_e, dens_i, parr_flow} at that triangle.
  3. compute predicted Bohm flux Γ_Bohm = ne * c_s.
  4. compare with the SOLPS target-cell value.

Reads:
  ../coupled_test/openedge/plasma.h5
  ../input/wall.surf
  ../coupled_test/solps/v0_1/b2pl.exe.dir/{fp_tg_o.dat, ld_tg_o.dat}

Usage:  python3 diagnose_wall_plasma.py
"""

from pathlib import Path
import sys
import numpy as np
import h5py
from scipy.spatial import cKDTree


def parse_wall(path):
    text = Path(path).read_text().splitlines()
    pts, lns, sec = {}, [], None
    for line in text:
        s = line.strip()
        if s == "Points": sec = "p"; continue
        if s == "Lines":  sec = "l"; continue
        if not s or not s[0].isdigit(): continue
        tok = s.split()
        if sec == "p":
            pts[int(tok[0])] = (float(tok[1]), float(tok[2]))
        elif sec == "l":
            lns.append((int(tok[0]), int(tok[1]), int(tok[2])))
    return pts, lns


def point_in_triangle(p, a, b, c):
    """Sign-of-cross-product test for 2D point-in-triangle."""
    def s(u, v, w):
        return (u[0]-w[0]) * (v[1]-w[1]) - (v[0]-w[0]) * (u[1]-w[1])
    d1 = s(p, a, b)
    d2 = s(p, b, c)
    d3 = s(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def eckstein_d_on_li(E, theta_deg, Q=0.16, Eth=6.92, ETF=209.0):
    """Bohdansky-1993 + Yamamura, matching OpenEdge eckstein_sputter.h."""
    th = float(np.clip(theta_deg, 0, 89.5))
    c = max(np.cos(np.radians(th)), 1e-3)
    if E <= Eth:
        return 0.0
    r = Eth / E
    eps = E / ETF
    f1 = Q * (1 - r**(2/3)) * (1 - r)**2
    num = 0.5 * np.log1p(1.2288 * eps)
    den = eps + 0.1728 * np.sqrt(eps) + 0.008 * eps**0.1504
    Y0 = max(f1 * (num / den), 0.0)
    arg = -2.0 * np.log(c) + 2.0 * (1.0 - 1.0 / c) * 0.26
    arg = max(min(arg, 500), -500)
    return Y0 * float(np.exp(arg))


def main():
    base = Path(__file__).resolve().parent
    project = base.parent
    plasma_h5 = project / "coupled_test" / "openedge" / "plasma.h5"
    wall = project / "input" / "wall.surf"
    b2pl = project / "coupled_test" / "solps" / "v0_1" / "b2pl.exe.dir"

    if not plasma_h5.exists():
        sys.exit(f"missing {plasma_h5}")
    if not wall.exists():
        sys.exit(f"missing {wall}")

    # --- Load plasma.h5 mesh ---
    with h5py.File(plasma_h5, 'r') as f:
        vtx_r = f['mesh/vtx_r'][:]
        vtx_z = f['mesh/vtx_z'][:]
        tri   = f['mesh/triangles'][:]
        cell_index = f['mesh/cell_index'][:]
        mesh_te = f['mesh/temp_e'][:]
        mesh_ne = f['mesh/dens_e'][:]
        mesh_ni = f['mesh/dens_i'][:]
        mesh_upar = f['mesh/parr_flow'][:] if 'mesh/parr_flow' in f else None
        # B at vertices: needed to compute B at segment centroid +
        # sin_alpha = |B.n|/|B|, exactly as OE compute does.
        vtx_br = f['mesh/vtx_br'][:] if 'mesh/vtx_br' in f else None
        vtx_bz = f['mesh/vtx_bz'][:] if 'mesh/vtx_bz' in f else None
        vtx_bt = f['mesh/vtx_bt'][:] if 'mesh/vtx_bt' in f else None
    print(f"[plasma.h5] mesh: {len(vtx_r)} vertices, {tri.shape[0]} triangles, "
          f"{len(mesh_te)} cells, "
          f"B at vtx: {'yes' if vtx_br is not None else 'NO'}")
    print(f"  Te range  [{mesh_te.min():.3e}, {mesh_te.max():.3e}] eV")
    print(f"  ne range  [{mesh_ne.min():.3e}, {mesh_ne.max():.3e}] m^-3")

    # Triangle barycenters (used for nearest-fallback lookup)
    bary_r = (vtx_r[tri[:, 0]] + vtx_r[tri[:, 1]] + vtx_r[tri[:, 2]]) / 3.0
    bary_z = (vtx_z[tri[:, 0]] + vtx_z[tri[:, 1]] + vtx_z[tri[:, 2]]) / 3.0
    tree = cKDTree(np.column_stack([bary_r, bary_z]))

    # --- Load wall and pick lower divertor segments (Z < -3) ---
    pts, lns = parse_wall(wall)
    div = []
    for sid, p1, p2 in lns:
        z = 0.5 * (pts[p1][0] + pts[p2][0])
        r = 0.5 * (pts[p1][1] + pts[p2][1])
        if z < -3.0:        # lower divertor only
            div.append((sid, r, z))
    div.sort(key=lambda x: x[0])
    print(f"\n[wall] {len(div)} lower-divertor segments")

    # --- For each segment, sample plasma at centroid ---
    AMU = 1.66053906660e-27
    eV = 1.602176634e-19
    m_D = 2.014 * AMU

    print(f"\n{'sid':>4} {'R':>7} {'Z':>7} "
          f"{'Te':>8} {'ni':>10} {'|B|':>6} "
          f"{'sinα':>7} {'θ_deg':>6} "
          f"{'E_imp':>7} {'Γ_proj':>10} {'Y':>9} {'Γ_ero':>10}")
    n_outside = 0; gap_max = 0.0
    rows = []
    # Yamamura constants used by OE compute_surface_physical_sputter
    for sid_, r, z in div:
        # Find triangle containing segment centroid (or nearest)
        dists, idxs = tree.query([r, z], k=20)
        found = -1
        for k in idxs:
            a = (vtx_r[tri[k, 0]], vtx_z[tri[k, 0]])
            b = (vtx_r[tri[k, 1]], vtx_z[tri[k, 1]])
            c = (vtx_r[tri[k, 2]], vtx_z[tri[k, 2]])
            if point_in_triangle((r, z), a, b, c):
                found = k; break
        if found < 0:
            n_outside += 1
            found = int(idxs[0])
            gap_max = max(gap_max, float(dists[0]))
        ci = int(cell_index[found])
        Te = float(mesh_te[ci]) if 0 <= ci < len(mesh_te) else 0.0
        ni = float(mesh_ni[ci]) if 0 <= ci < len(mesh_ni) else 0.0
        Ti = Te  # Ti not directly available; use Te as proxy (OE does similarly)

        # B at centroid: barycentric average of triangle vertex B
        if vtx_br is not None:
            v0, v1, v2 = tri[found]
            Br = float(np.mean([vtx_br[v0], vtx_br[v1], vtx_br[v2]]))
            Bz = float(np.mean([vtx_bz[v0], vtx_bz[v1], vtx_bz[v2]]))
            Bt = float(np.mean([vtx_bt[v0], vtx_bt[v1], vtx_bt[v2]]))
        else:
            Br = Bz = Bt = 0.0
        Bmag = float(np.sqrt(Br**2 + Bz**2 + Bt**2))

        # Surface normal in (R, Z) plane: perpendicular to the segment,
        # pointing into the plasma. Use segment endpoints.
        sid_lookup = {sid: (p1, p2) for sid, p1, p2 in lns}
        p1id, p2id = sid_lookup[sid_]
        z1, r1 = pts[p1id]; z2, r2 = pts[p2id]
        # Tangent (Z, R) -> normal in (R, Z) is rotated 90 deg.
        tan_z, tan_r = z2 - z1, r2 - r1
        nlen = np.hypot(tan_z, tan_r)
        if nlen > 0:
            n_r = -tan_z / nlen
            n_z =  tan_r / nlen
        else:
            n_r = n_z = 0.0
        # Force normal to point INTO plasma: toward the mesh barycenter
        if (Br * n_r + Bz * n_z) < 0:
            n_r, n_z = -n_r, -n_z

        # sin_alpha = |B.n|/|B|  (n has no toroidal component for axi wall)
        Bdotn = abs(Br * n_r + Bz * n_z)
        sin_alpha = (Bdotn / Bmag) if Bmag > 0 else 0.0
        alpha_deg = float(np.degrees(np.arcsin(min(sin_alpha, 1.0))))
        theta_inc = 90.0 - alpha_deg          # angle from normal

        # Bohm projectile flux: ni * cs * sin_alpha, cs = sqrt((Te+Ti)/m_D)
        cs2 = (Te + Ti) * eV / m_D
        cs = np.sqrt(max(cs2, 0.0))
        G_proj = ni * cs * sin_alpha

        # Chankin sheath impact energy
        Te_safe = max(Te, 1e-30)
        ti_ratio = Ti / Te_safe
        psi_over_Te = 0.5 * np.log((m_D / 9.1093837015e-31) /
                                    (2 * np.pi * (1 + ti_ratio)))
        E_imp = 2.0 * Ti + 1.0 * psi_over_Te * Te
        # Eckstein D_on_Li yield
        Y = eckstein_d_on_li(E_imp, theta_inc)
        G_ero = G_proj * Y
        rows.append(dict(sid=sid_, R=r, Z=z, Te=Te, ni=ni, Bmag=Bmag,
                         sin_alpha=sin_alpha, theta_inc=theta_inc,
                         E_imp=E_imp, G_proj=G_proj, Y=Y, G_ero=G_ero))
        print(f"{sid_:>4d} {r:>7.3f} {z:>7.3f} "
              f"{Te:>8.2f} {ni:>10.3e} {Bmag:>6.2f} "
              f"{sin_alpha:>7.4f} {theta_inc:>6.2f} "
              f"{E_imp:>7.2f} {G_proj:>10.3e} {Y:>9.3e} {G_ero:>10.3e}")

    G_ero_arr = np.array([r['G_ero'] for r in rows])
    G_proj_arr = np.array([r['G_proj'] for r in rows])
    Y_arr = np.array([r['Y'] for r in rows])
    print(f"\nSUMMARY")
    print(f"  segments outside SOLPS mesh: {n_outside} / {len(div)}")
    print(f"  max gap: {gap_max*1000:.2f} mm")
    if Y_arr.max() > 0:
        kmax = int(np.argmax(G_ero_arr))
        print(f"  PEAK Γ_ero (Python-OE pipeline) at sid={rows[kmax]['sid']}: "
              f"{G_ero_arr[kmax]:.3e} m^-2 s^-1")
        print(f"    Te={rows[kmax]['Te']:.2f} eV  E_imp={rows[kmax]['E_imp']:.2f} eV  "
              f"Y={Y_arr[kmax]:.3e}  Γ_proj={G_proj_arr[kmax]:.3e}")

    # --- Compare with SOLPS target peak (outer lower) ---
    if (b2pl / 'fp_tg_o.dat').exists():
        sys.path.insert(0, str(base))
        from read_solps_target import read_target
        d = read_target(b2pl, 'outer')
        kmax = int(np.argmax(d['D_flux_ion']))
        Te_sps = float(d['Te'][kmax])
        ne_sps = float(d['ne'][kmax])
        D_sps  = float(d['D_flux_ion'][kmax])
        print(f"\nSOLPS outer target peak (cell with max D_flux_ion):")
        print(f"  R, Z = ({d['Rclfc'][kmax]:.3f}, {d['Zclfc'][kmax]:.3f}) m")
        print(f"  Te = {Te_sps:.3f} eV   ne = {ne_sps:.3e} m^-3   "
              f"D_flux_ion = {D_sps:.3e} m^-2 s^-1")


if __name__ == "__main__":
    main()
