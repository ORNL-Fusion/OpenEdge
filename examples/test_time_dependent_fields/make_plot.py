# Let's build a robust R-Z (poloidal) plot from the file by taking a single y-slice (the layer closest to y=0).
import numpy as np
import matplotlib.pyplot as plt

# Load rows
rows = []
with open("tmp.forces", "r", errors="ignore") as f:
    for line in f:
        if line.strip() and line[0].isdigit():
            parts = line.split()
            if len(parts) == 7:
                rows.append([float(p) for p in parts])
arr = np.array(rows)  # id, xc, yc, zc, Bx, By, Bz

# Choose the y-layer closest to 0
yc_vals = arr[:,2]
unique_y = np.unique(yc_vals)
y0 = unique_y[np.argmin(np.abs(unique_y))]

mask = np.isclose(yc_vals, y0)
slice_arr = arr[mask]

xc = slice_arr[:,1]
zc = slice_arr[:,3]
Bx = slice_arr[:,4]
By = slice_arr[:,5]
Bz = slice_arr[:,6]

# Build structured grid for X-Z
xgrid = np.unique(xc)
zgrid = np.unique(zc)
nx, nz = len(xgrid), len(zgrid)
def make_grid(values):
    grid = np.full((nz, nx), np.nan)
    # use exact matches (values are on a regular grid)
    for xi, xval in enumerate(xgrid):
        for zi, zval in enumerate(zgrid):
            m = (np.isclose(xc, xval) & np.isclose(zc, zval))
            if np.any(m):
                grid[zi, xi] = values[m][0]
    return grid

Gx = make_grid(Bx)
Gy = make_grid(By)
Gz = make_grid(Bz)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
for ax, G, ttl in zip(axes, [Gx, Gy, Gz], ["Bx", "By", "Bz"]):
    pcm = ax.pcolormesh(xgrid, zgrid, G, shading="nearest")
    fig.colorbar(pcm, ax=ax, label=ttl + " (norm)")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(f"{ttl} at y≈{y0:.2f}")
    ax.set_aspect("equal")

plt.show()

