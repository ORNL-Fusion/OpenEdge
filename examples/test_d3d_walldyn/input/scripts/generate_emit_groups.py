#!/usr/bin/env python3
"""Generate an emit_wall surface subgroup for SPARTA from SOLPS plasma data.

Computes Bohm sputtering flux at each poloidal wall segment midpoint, then
selects segments above a relative threshold to form the emit_wall group.
The full wall remains for particle-wall collisions; only the plasma-wetted
subset emits sputtered particles.

Outputs:
  - emit_wall_groups.sparta   SPARTA include file with group commands
  - Updated metadata JSON with emit_wall triangle ID ranges

Usage:
    python generate_emit_groups.py \\
        --meta d3d_refined_meta.json \\
        --mesh-extra mesh.extra \\
        --plasma-h5 plasma.h5 \\
        --bfield-h5 bfield.h5 \\
        --threshold 0.01 \\
        --mass-amu 2.0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np

Point2 = Tuple[float, float]

# Physical constants
QE = 1.602176634e-19   # elementary charge [C]
AMU = 1.66053906660e-27  # atomic mass unit [kg]


# ---------------------------------------------------------------------------
# Mesh.extra parsing (same as refine_wedge_geometry.py)
# ---------------------------------------------------------------------------

def _qkey(p, tol):
    return (int(round(float(p[0]) / tol)), int(round(float(p[1]) / tol)))


def read_mesh_segments(mesh_extra: Path, tol: float = 1e-8):
    from collections import Counter, defaultdict
    id_of = {}
    points = []
    edges = []

    def vid(p):
        k = _qkey(p, tol)
        if k not in id_of:
            id_of[k] = len(points)
            points.append((float(p[0]), float(p[1])))
        return id_of[k]

    with mesh_extra.open("r") as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            cols = s.split()
            if len(cols) < 4:
                continue
            r1, z1, r2, z2 = map(float, cols[:4])
            i = vid((r1, z1))
            j = vid((r2, z2))
            if i != j:
                edges.append((i, j))

    edge_count = Counter(tuple(sorted(e)) for e in edges)
    boundary_edges = [e for e in edge_count if edge_count[e] == 1]
    adj = defaultdict(list)
    edge_set = set()
    for i, j in boundary_edges:
        adj[i].append(j)
        adj[j].append(i)
        edge_set.add(tuple(sorted((i, j))))

    unused = set(edge_set)
    loops = []
    while unused:
        e0 = next(iter(unused))
        start, cur = e0
        loop = [start, cur]
        unused.remove(e0)
        prev = start
        while True:
            nbrs = adj[cur]
            nxt = None
            for cand in nbrs:
                if cand == prev:
                    continue
                ek = tuple(sorted((cur, cand)))
                if ek in unused:
                    nxt = cand
                    break
            if nxt is None:
                if cur == start:
                    break
                if start in nbrs:
                    break
                break
            loop.append(nxt)
            unused.remove(tuple(sorted((cur, nxt))))
            prev, cur = cur, nxt
            if cur == start:
                break
        if len(loop) >= 4 and loop[0] == loop[-1]:
            loops.append(loop[:-1])

    def perimeter(loop):
        n = len(loop)
        return sum(math.hypot(points[loop[(i+1) % n]][0] - points[loop[i]][0],
                              points[loop[(i+1) % n]][1] - points[loop[i]][1])
                   for i in range(n))

    outer = max(loops, key=perimeter)
    rz = [points[i] for i in outer]

    area = sum(rz[i][0]*rz[(i+1) % len(rz)][1] - rz[(i+1) % len(rz)][0]*rz[i][1]
               for i in range(len(rz))) * 0.5
    if area < 0:
        rz = list(reversed(rz))
    return rz


def is_in_divertor_region(r, z, rmin, rmax, zmin, zmax):
    return rmin <= r <= rmax and zmin <= z <= zmax


def subdivide_polyline(rz, n_default, n_divertor, rmin, rmax, zmin, zmax):
    refined = []
    n = len(rz)
    for i in range(n):
        r1, z1 = rz[i]
        r2, z2 = rz[(i + 1) % n]
        mid_r = 0.5 * (r1 + r2)
        mid_z = 0.5 * (z1 + z2)
        if is_in_divertor_region(mid_r, mid_z, rmin, rmax, zmin, zmax):
            nsub = n_divertor
        else:
            nsub = n_default
        for k in range(nsub):
            t = k / nsub
            refined.append((r1 + t * (r2 - r1), z1 + t * (z2 - z1)))
    return refined


# ---------------------------------------------------------------------------
# Eirene mesh lookup (replaces regular-grid interpolation)
# ---------------------------------------------------------------------------

class EireneMesh:
    """Point-in-triangle lookup on the SOLPS Eirene triangulation.

    Only triangles mapped to a B2.5 plasma cell (cell_index >= 0) return
    valid plasma data.  Wall segments whose centroids fall outside the
    mapped mesh — or inside vacuum/PFR triangles — get zero flux.

    This is the same logic as C++ find_mesh_triangle() + mesh_cell_idx[].
    """

    def __init__(self, plasma_h5: Path):
        with h5py.File(plasma_h5, "r") as f:
            self.vtx_r = np.asarray(f["mesh/vtx_r"])
            self.vtx_z = np.asarray(f["mesh/vtx_z"])
            self.tri = np.asarray(f["mesh/triangles"])   # (ntri, 3)
            self.cell_idx = np.asarray(f["mesh/cell_index"])  # (ntri,)
            self.mesh_te = np.asarray(f["mesh/temp_e"])
            self.mesh_ne = np.asarray(f["mesh/dens_e"])
            self.mesh_ti = np.asarray(f["mesh/temp_i"])
            self.mesh_ni = np.asarray(f["mesh/dens_i"])

        # Precompute bounding boxes for fast rejection
        r0 = self.vtx_r[self.tri[:, 0]]
        r1 = self.vtx_r[self.tri[:, 1]]
        r2 = self.vtx_r[self.tri[:, 2]]
        z0 = self.vtx_z[self.tri[:, 0]]
        z1 = self.vtx_z[self.tri[:, 1]]
        z2 = self.vtx_z[self.tri[:, 2]]
        self.bb_rmin = np.minimum(np.minimum(r0, r1), r2)
        self.bb_rmax = np.maximum(np.maximum(r0, r1), r2)
        self.bb_zmin = np.minimum(np.minimum(z0, z1), z2)
        self.bb_zmax = np.maximum(np.maximum(z0, z1), z2)

        self.ntri = len(self.tri)

        # Precompute centroids of mapped triangles for nearest-neighbor fallback
        mapped_mask = self.cell_idx >= 0
        mapped_indices = np.where(mapped_mask)[0]
        self.mapped_cr = np.array([
            (self.vtx_r[self.tri[t, 0]] + self.vtx_r[self.tri[t, 1]] +
             self.vtx_r[self.tri[t, 2]]) / 3.0 for t in mapped_indices])
        self.mapped_cz = np.array([
            (self.vtx_z[self.tri[t, 0]] + self.vtx_z[self.tri[t, 1]] +
             self.vtx_z[self.tri[t, 2]]) / 3.0 for t in mapped_indices])
        self.mapped_idx = mapped_indices
        self.n_mapped = len(mapped_indices)

    def find_triangle(self, r: float, z: float) -> int:
        """Return index of mapped triangle containing (r,z), or -1."""
        for t in range(self.ntri):
            if self.cell_idx[t] < 0:
                continue
            if (r < self.bb_rmin[t] or r > self.bb_rmax[t] or
                    z < self.bb_zmin[t] or z > self.bb_zmax[t]):
                continue
            a, b, c = self.tri[t]
            ar, az = self.vtx_r[a], self.vtx_z[a]
            br_, bz_ = self.vtx_r[b], self.vtx_z[b]
            cr, cz = self.vtx_r[c], self.vtx_z[c]
            # Barycentric coordinate test
            d = (br_ - ar) * (cz - az) - (cr - ar) * (bz_ - az)
            if abs(d) < 1e-30:
                continue
            u = ((r - ar) * (cz - az) - (z - az) * (cr - ar)) / d
            v = ((z - az) * (br_ - ar) - (r - ar) * (bz_ - az)) / d
            if u >= 0 and v >= 0 and (u + v) <= 1.0:
                return t
        return -1

    def find_nearest_mapped(self, r: float, z: float,
                            max_dist: float = 0.05) -> int:
        """Nearest mapped triangle centroid within max_dist [m].

        Wall centroids typically fall in the vacuum gap between the B2.5
        domain and the physical wall (Eirene fills this with unmapped
        triangles).  This fallback uses the nearest mapped cell's data.
        """
        if self.n_mapped == 0:
            return -1
        d2 = (self.mapped_cr - r)**2 + (self.mapped_cz - z)**2
        imin = np.argmin(d2)
        if d2[imin] <= max_dist * max_dist:
            return self.mapped_idx[imin]
        return -1

    def plasma_at(self, r: float, z: float):
        """Return (ni, ti, te) at (r,z) from mesh lookup, or None.

        Tries exact point-in-triangle first, then nearest-neighbor fallback
        for wall points in the vacuum gap.
        """
        t = self.find_triangle(r, z)
        if t < 0:
            t = self.find_nearest_mapped(r, z)
        if t < 0:
            return None
        cell = self.cell_idx[t]
        if cell < 0:
            return None
        return (self.mesh_ni[cell], self.mesh_ti[cell], self.mesh_te[cell])


def load_regular_grid(h5path, field_name):
    """Load a 2D field on a regular (r, z) grid from HDF5."""
    with h5py.File(h5path, "r") as f:
        r = np.array(f["r"])
        z = np.array(f["z"])
        data = np.array(f[field_name])
    return r, z, data


def interp2d_regular(r_grid, z_grid, data, r_pt, z_pt):
    """Bilinear interpolation on a regular grid. Returns 0 if out of bounds."""
    nr, nz = len(r_grid), len(z_grid)
    dr = r_grid[1] - r_grid[0]
    dz = z_grid[1] - z_grid[0]

    fi = (r_pt - r_grid[0]) / dr
    fj = (z_pt - z_grid[0]) / dz

    i = int(math.floor(fi))
    j = int(math.floor(fj))

    if i < 0 or i >= nr - 1 or j < 0 or j >= nz - 1:
        return 0.0

    tr = fi - i
    tz = fj - j

    v00 = data[i, j]
    v10 = data[i + 1, j]
    v01 = data[i, j + 1]
    v11 = data[i + 1, j + 1]

    return (v00 * (1 - tr) * (1 - tz) +
            v10 * tr * (1 - tz) +
            v01 * (1 - tr) * tz +
            v11 * tr * tz)


# ---------------------------------------------------------------------------
# Bohm flux computation per poloidal segment
# ---------------------------------------------------------------------------

def compute_segment_flux(rz_refined: List[Point2],
                         plasma_h5: Path, bfield_h5: Path,
                         mass_amu: float) -> np.ndarray:
    """Compute Bohm sputtering flux at each poloidal segment midpoint.

    Uses the Eirene mesh triangulation from plasma.h5 for plasma lookup
    (not regular-grid interpolation).  Only wall segments whose centroids
    fall inside a mapped Eirene triangle (cell_index >= 0) get nonzero flux.
    B-field is still interpolated on the regular grid (always valid).

    Returns array of shape (n_rz,) with Gamma = ni * cs * sin(alpha_B).
    """
    n_rz = len(rz_refined)

    # Load Eirene mesh for plasma lookup
    mesh = EireneMesh(plasma_h5)
    print(f"  Eirene mesh: {mesh.ntri} triangles, "
          f"{np.sum(mesh.cell_idx >= 0)} mapped to B2.5")

    # B-field still on regular grid (always covers the full domain)
    r_b, z_b, br = load_regular_grid(bfield_h5, "br")
    _, _, bt = load_regular_grid(bfield_h5, "bt")
    _, _, bz_fld = load_regular_grid(bfield_h5, "bz")

    flux = np.zeros(n_rz)
    n_mapped = 0

    for i in range(n_rz):
        r1, z1 = rz_refined[i]
        r2, z2 = rz_refined[(i + 1) % n_rz]

        # Segment midpoint
        rm = 0.5 * (r1 + r2)
        zm = 0.5 * (z1 + z2)

        # Plasma lookup via Eirene mesh (None = outside mapped domain)
        plasma = mesh.plasma_at(rm, zm)
        if plasma is None:
            continue
        ni_loc, ti_loc, te_loc = plasma
        if ni_loc <= 0.0:
            continue
        n_mapped += 1

        # Outward normal (perpendicular to segment, CCW winding -> outward)
        dr = r2 - r1
        dz = z2 - z1
        seg_len = math.hypot(dr, dz)
        if seg_len < 1e-15:
            continue
        nr_s = dz / seg_len    # normal_R = +dz
        nz_s = -dr / seg_len   # normal_Z = -dr

        # B-field lookup (regular grid — always valid)
        br_loc = interp2d_regular(r_b, z_b, br, rm, zm)
        bt_loc = interp2d_regular(r_b, z_b, bt, rm, zm)
        bz_loc = interp2d_regular(r_b, z_b, bz_fld, rm, zm)
        bmag = math.sqrt(br_loc**2 + bt_loc**2 + bz_loc**2)

        if bmag <= 0.0:
            continue

        # Poloidal incidence angle (same as C++ code: |Bp . n| / |B|)
        bp_dot_n = br_loc * nr_s + bz_loc * nz_s
        sin_alpha = abs(bp_dot_n) / bmag

        # Bohm sound speed: cs = sqrt((Te + Ti) * e / (2 * m_amu * AMU))
        cs_arg = (te_loc + ti_loc) * QE / (2.0 * mass_amu * AMU)
        cs = math.sqrt(cs_arg) if cs_arg > 0 else 0.0

        # Bohm flux
        flux[i] = ni_loc * cs * sin_alpha

    print(f"  Segments inside mapped Eirene mesh: {n_mapped} / {n_rz}")
    return flux


# ---------------------------------------------------------------------------
# Triangle ID generation
# ---------------------------------------------------------------------------

def active_segments_to_triangle_ids(active_indices: List[int],
                                    n_rz: int, n_tor: int) -> List[Tuple[int, int]]:
    """Convert active poloidal indices to (first, last) 1-based triangle ID ranges.

    Wall triangle ordering (from build_wall_tris):
      For strip k, poloidal index i, sub-triangle s (0,1):
        0-based = k * (n_rz * 2) + i * 2 + s
        1-based = above + 1

    Returns sorted, merged ranges for SPARTA group commands.
    """
    raw_ids = []
    for k in range(n_tor):
        base = k * (n_rz * 2)
        for i in active_indices:
            raw_ids.append(base + i * 2 + 1)      # sub-tri 0, 1-based
            raw_ids.append(base + i * 2 + 2)      # sub-tri 1, 1-based

    raw_ids.sort()

    # Merge consecutive IDs into ranges
    if not raw_ids:
        return []

    ranges = []
    lo = hi = raw_ids[0]
    for tid in raw_ids[1:]:
        if tid == hi + 1:
            hi = tid
        else:
            ranges.append((lo, hi))
            lo = hi = tid
    ranges.append((lo, hi))
    return ranges


def write_sparta_include(path: Path, ranges: List[Tuple[int, int]],
                         wall_first: int, wall_last: int):
    """Write SPARTA include file that defines emit_wall as a surface subgroup."""
    with path.open("w") as f:
        f.write("# Auto-generated emit_wall subgroup (plasma-wetted wall segments)\n")
        f.write("# Defines emit_wall as a subset of wall for sputtering emission.\n")
        f.write("# Full wall group is still used for particle-wall collisions.\n")
        f.write("#\n")
        f.write(f"# Wall triangles: {wall_first} - {wall_last}\n")
        f.write(f"# Emit wall ranges: {len(ranges)} contiguous blocks, "
                f"{sum(hi - lo + 1 for lo, hi in ranges)} triangles total\n\n")

        for lo, hi in ranges:
            f.write(f"group emit_wall surf id <> {lo} {hi}\n")

    print(f"Wrote {path}  ({len(ranges)} ranges, "
          f"{sum(hi - lo + 1 for lo, hi in ranges)} triangles)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", type=Path, required=True,
                   help="Refined wedge metadata JSON")
    p.add_argument("--mesh-extra", type=Path, required=True,
                   help="SOLPS mesh.extra file")
    p.add_argument("--plasma-h5", type=Path, required=True,
                   help="Plasma HDF5 file (ne, Te, ni, Ti)")
    p.add_argument("--bfield-h5", type=Path, required=True,
                   help="B-field HDF5 file (br, bt, bz)")
    p.add_argument("--threshold", type=float, default=0.01,
                   help="Relative flux threshold (fraction of peak, default 0.01)")
    p.add_argument("--mass-amu", type=float, default=2.0,
                   help="Background ion mass in amu (default 2.0 for D+)")
    p.add_argument("--output", type=Path, default=None,
                   help="Output SPARTA include file (default: emit_wall_groups.sparta)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load metadata
    meta = json.loads(args.meta.read_text())
    n_rz = meta["n_rz_refined"]
    n_tor = meta["n_toroidal_segments"]
    wall_first = meta["triangle_groups"]["wall"]["first"]
    wall_last = meta["triangle_groups"]["wall"]["last"]

    div = meta["divertor_region"]

    # Reconstruct R-Z polyline (must match refine_wedge_geometry.py exactly)
    rz_raw = read_mesh_segments(args.mesh_extra)
    rz = subdivide_polyline(rz_raw,
                            meta["wall_refine"], meta["divertor_refine"],
                            div["rmin"], div["rmax"], div["zmin"], div["zmax"])
    assert len(rz) == n_rz, (
        f"Reconstructed polyline has {len(rz)} vertices, expected {n_rz}")

    # Compute Bohm flux per poloidal segment
    flux = compute_segment_flux(rz, args.plasma_h5, args.bfield_h5, args.mass_amu)

    peak = flux.max()
    if peak <= 0:
        print("WARNING: No positive flux found. Check plasma/bfield data.")
        return

    cutoff = args.threshold * peak
    active = [i for i in range(n_rz) if flux[i] >= cutoff]

    print(f"Peak Bohm flux: {peak:.3e} m^-2 s^-1")
    print(f"Threshold: {args.threshold:.1%} of peak = {cutoff:.3e} m^-2 s^-1")
    print(f"Active poloidal segments: {len(active)} / {n_rz} "
          f"({100*len(active)/n_rz:.1f}%)")

    if not active:
        print("WARNING: No active segments above threshold.")
        return

    # Print poloidal extent of active region(s)
    active_rz = [(rz[i][0], rz[i][1]) for i in active]
    r_min = min(p[0] for p in active_rz)
    r_max = max(p[0] for p in active_rz)
    z_min = min(p[1] for p in active_rz)
    z_max = max(p[1] for p in active_rz)
    print(f"Active region R: [{r_min:.3f}, {r_max:.3f}] m")
    print(f"Active region Z: [{z_min:.3f}, {z_max:.3f}] m")

    # Generate triangle ID ranges
    ranges = active_segments_to_triangle_ids(active, n_rz, n_tor)

    n_emit_tris = sum(hi - lo + 1 for lo, hi in ranges)
    n_wall_tris = wall_last - wall_first + 1
    print(f"Emit wall triangles: {n_emit_tris} / {n_wall_tris} "
          f"({100*n_emit_tris/n_wall_tris:.1f}%)")

    # Write SPARTA include file
    out_path = args.output or args.meta.parent / "emit_wall_groups.sparta"
    write_sparta_include(out_path, ranges, wall_first, wall_last)

    # Update metadata JSON with emit_wall group
    meta["triangle_groups"]["emit_wall"] = {
        "first_ranges": [[lo, hi] for lo, hi in ranges],
        "n_triangles": n_emit_tris,
        "threshold": args.threshold,
        "n_active_poloidal": len(active),
    }
    args.meta.write_text(json.dumps(meta, indent=2))
    print(f"Updated {args.meta} with emit_wall group info")


if __name__ == "__main__":
    main()
