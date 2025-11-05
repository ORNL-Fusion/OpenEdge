import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

# ---------------- parse your dump ----------------
def read_cells_dump(path):
    """
    Reads a LAMMPS-like 'ITEM: CELLS' dump with:
      ITEM: CELLS id xc yc zc f_Fdist[1] f_Fdist[2]
    Returns dict with arrays.
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # find the CELLS header line
    hdr_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("ITEM: CELLS"):
            hdr_idx = i
            break
    if hdr_idx is None:
        raise ValueError("Could not find 'ITEM: CELLS' header.")

    # read column names from header (optional)
    cols = lines[hdr_idx].split()[2:]  # after 'ITEM: CELLS'
    # read subsequent numeric rows until next ITEM or EOF
    data = []
    for ln in lines[hdr_idx+1:]:
        if ln.startswith("ITEM:"):
            break
        parts = ln.split()
        if len(parts) < 6:  # id xc yc zc f1 f2
            continue
        data.append([float(p) for p in parts[:6]])

    arr = np.array(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError("Parsed data has wrong shape.")

    out = {
        "id":  arr[:, 0].astype(int),
        "xc":  arr[:, 1],
        "yc":  arr[:, 2],
        "zc":  arr[:, 3],
        "f1":  arr[:, 4],  # f_Fdist[1]
        "f2":  arr[:, 5],  # f_Fdist[2]
        "cols": cols,
    }
    return out

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_3d_scatter(
    xc, yc, zc, val,
    tris, pts,
    ids=None,                    # array of cell ids (same length as xc)
    annotate=False,              # turn on labels
    annotate_every=1,            # label every Nth point
    fontsize=7,                  # label size
    cmap="viridis",
    title="3D scatter (cell centers)"
):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection="3d")

    finite = np.isfinite(val)
    vmin, vmax = np.nanpercentile(val[finite], [2, 98])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    sc = ax.scatter(xc, yc, zc, c=val, s=8, marker='o', cmap=cmap, norm=norm, alpha=0.9)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("f_Fdist[2]")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)

    # Equal-ish aspect box
    mins = np.array([xc.min(), yc.min(), zc.min()])
    maxs = np.array([xc.max(), yc.max(), zc.max()])
    ctr  = 0.5*(mins + maxs)
    span = (maxs - mins).max()
    ax.set_xlim(ctr[0]-span/2, ctr[0]+span/2)
    ax.set_ylim(ctr[1]-span/2, ctr[1]+span/2)
    ax.set_zlim(ctr[2]-span/2, ctr[2]+span/2)

    polys = [pts[t] for t in tris]
    poly = Poly3DCollection(polys, alpha=0.9, linewidths=0.2)
    ax.add_collection3d(poly)
    
    
    # Optional ID labels
    if annotate and ids is not None:
        # small offset relative to plot span
        dx, dy, dz = (0.01*span, 0.01*span, 0.01*span)
        # label every Nth point
        for i in range(0, len(xc), max(1, annotate_every)):
            ax.text(
                xc[i] + dx, yc[i] + dy, zc[i] + dz,
                str(ids[i]),
                fontsize=fontsize, color="black",
                ha="left", va="bottom"
            )

    plt.tight_layout()
    plt.show()


# ------------- utilities -----------------
def centers_to_edges(c):
    c = np.asarray(c)
    e = np.empty(c.size + 1, dtype=c.dtype)
    e[1:-1] = 0.5*(c[:-1] + c[1:])
    e[0]    = c[0] - 0.5*(c[1] - c[0])
    e[-1]   = c[-1] + 0.5*(c[-1] - c[-2])
    return e

def is_rectilinear(c):
    d = np.diff(c)
    return np.allclose(d, d[0], rtol=1e-6, atol=1e-12)


def plot_slices(xc, yc, zc, val, pts, tris, zs_to_plot=None, tol=None, cmap="plasma"):
    """
    Make 2D pcolormesh slices at selected z-planes.
    Infers x/y edges from centers for clean cell plots.
    """
    x = np.unique(xc); y = np.unique(yc); z = np.unique(zc)
    if zs_to_plot is None:
        # choose 3 representative z planes
        picks = np.unique(np.linspace(0, len(z)-1, 3).round().astype(int))
        zs_to_plot = z[picks]

    if tol is None:
        tol = 0.5*min(np.diff(np.unique(zc)).min(), 1e9) if len(z) > 1 else 1e-9

    xe, ye = centers_to_edges(x), centers_to_edges(y)

    for z0 in zs_to_plot:
        mask = np.isclose(zc, z0, atol=tol)
        if not np.any(mask):
            continue
        # grid the slice
        ix = np.searchsorted(x, xc[mask])
        iy = np.searchsorted(y, yc[mask])
        G = np.full((y.size, x.size), np.nan)
        G[iy, ix] = val[mask]

        finite = np.isfinite(G)
        vmin, vmax = np.nanpercentile(G[finite], [2, 98]) if np.any(finite) else (None, None)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax) if vmin is not None else None

        fig, ax = plt.subplots(figsize=(7,6))
        pcm = ax.pcolormesh(xe, ye, G, shading="auto", cmap=cmap, norm=norm)
        
                # add surface
        polys = [pts[t] for t in tris]
#        poly = Poly3DCollection(polys, alpha=0.9, linewidths=0.2)
#        ax.add_collection3d(poly)
    
        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label("f_Fdist[2]")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Slice at z = {z0:g} m")
        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()

def plot_voxels(xc, yc, zc, val, pts, tris, cmap="viridis"):
    """
    Voxel volume rendering for rectilinear grids (regular spacing in x,y,z).
    """
    x = np.unique(xc); y = np.unique(yc); z = np.unique(zc)
    if not (is_rectilinear(x) and is_rectilinear(y) and (len(z) > 1 and is_rectilinear(z))):
        print("Voxels skipped: grid not strictly rectilinear in all dims (or only one z-plane).")
        return

    # Build grid
    ix = np.searchsorted(x, xc)
    iy = np.searchsorted(y, yc)
    iz = np.searchsorted(z, zc)
    G = np.full((z.size, y.size, x.size), np.nan)  # (k,j,i)
    G[iz, iy, ix] = val

    finite = np.isfinite(G)
    vmin, vmax = np.nanpercentile(G[finite], [2, 98])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    # mask & colors
    filled = np.isfinite(G)
    facecolors = np.zeros(filled.shape + (4,), dtype=float)
    facecolors[filled] = cmap_obj(norm(G[filled]))

    # We’ll render on an index grid and relabel axes with physical ticks
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(filled, facecolors=facecolors, edgecolor='k', linewidth=0.1, alpha=0.9)

    # tick labels mapped to physical coords
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("Voxel rendering (rectilinear grid)")
    # build a colorbar manually
    mapp = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    mapp.set_array([])
    cbar = fig.colorbar(mapp, ax=ax, pad=0.02)
    cbar.set_label("f_Fdist[2]")

    # add surface
    polys = [pts[t] for t in tris]
    poly = Poly3DCollection(polys, alpha=0.9, linewidths=0.2)
    ax.add_collection3d(poly)
    
    
    plt.tight_layout()
    plt.show()

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def parse_sparta_surface(path):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]

    try:
        pts_idx = lines.index("Points") + 1
        tri_idx = lines.index("Triangles") + 1
    except ValueError as e:
        raise RuntimeError("Could not find 'Points' or 'Triangles' section") from e

    pts = []
    i = pts_idx
    while i < tri_idx - 1:
        parts = lines[i].split()
        if len(parts) == 4:
            try:
                _id = int(parts[0])
                x, y, z = map(float, parts[1:])
                pts.append((x, y, z))
            except ValueError:
                pass
        i += 1

    tris = []
    for j in range(tri_idx, len(lines)):
        parts = lines[j].split()
        if len(parts) == 4:
            try:
                _id = int(parts[0])
                i1, i2, i3 = map(int, parts[1:])
                tris.append((i1 - 1, i2 - 1, i3 - 1))
            except ValueError:
                pass

    if not pts or not tris:
        raise RuntimeError("Parsed zero points or zero triangles; check file format.")
    return np.array(pts, dtype=float), np.array(tris, dtype=int)


def plot_surface(pts, tris, save=None, figsize=(8, 6)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    polys = [pts[t] for t in tris]
    poly = Poly3DCollection(polys, alpha=0.9, linewidths=0.2)
    ax.add_collection3d(poly)

    xyz_min = pts.min(axis=0)
    xyz_max = pts.max(axis=0)
    ranges = xyz_max - xyz_min
    max_range = max(ranges)
    mid = (xyz_max + xyz_min) / 2.0
    ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
    ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
    ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Surface mesh (SPARTA)")
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    else:
        plt.show()
# ---------------- run ----------------
if __name__ == "__main__":
    path = "tmp.grid"   # <-- put your file path
    d = read_cells_dump(path)

    # read surf
            # read surface
    pts, tris = parse_sparta_surface("surfaces/flatSurface.txt")
    
    # choose which field to color by:
    field_col = "f2"  # or "f1"

    xc, yc, zc, val = d["xc"], d["yc"], d["zc"], d[field_col]

    d = read_cells_dump("tmp.grid")
    plot_3d_scatter(
        d["xc"], d["yc"], d["zc"], d["f2"], tris, pts,
        ids=d["id"], annotate=True, annotate_every=1, fontsize=8,
        title="Cell centers with IDs (every 5th)"
    )


#    # 1) 3D scatter of centers
#    plot_3d_scatter(xc, yc, zc, val,  pts, tris, cmap="viridis", title="Cell centers colored by f_Fdist[2]")
##
###    # 2) 2D slices at several z-planes
#    plot_slices(xc, yc, zc, val,  pts, tris, zs_to_plot=None, cmap="plasma")
###
##  
##
##
##    # 3) Voxel volume (only if x,y,z spacing is strictly regular and multiple z planes exist)
#    plot_voxels(xc, yc, zc, val,pts, tris, cmap="viridis")
#
