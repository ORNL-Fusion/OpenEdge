"""WEST grid figures for manual section 4.10: ni_jump refinement and rank partition.

usage: python plot_grid.py <run dir with grid.dump, wall.dump> <output dir>
"""
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

d, out = sys.argv[1], sys.argv[2]
H5 = 'examples/workflows/impurity_transport/west_tungsten_transport/input/plasma.h5'
NZ, NR = 50, 88                                # level-1 grid of in.grid
Z0, Z1, R0, R1 = -1.0, 0.85, 0.0, 3.25

g = np.loadtxt(f'{d}/grid.dump', skiprows=9)
proc, zlo, rlo, zhi, rhi = g[:, 1], g[:, 2], g[:, 3], g[:, 4], g[:, 5]   # x = Z, y = R
level = np.rint(np.log2((Z1 - Z0) / NZ / (zhi - zlo))).astype(int) + 1
nlev = int(level.max())

lines = open(f'{d}/wall.dump').read().split('\n')
ip, il = lines.index('Points'), lines.index('Lines')
pts = {int(l.split()[0]): (float(l.split()[1]), float(l.split()[2]))
       for l in lines[ip + 1:il] if l.strip()}
segs = [[pts[int(l.split()[1])][::-1], pts[int(l.split()[2])][::-1]]
        for l in lines[il + 1:] if len(l.split()) >= 3]

e = h5py.File(H5)['equilibrium']
psin = (e['psi'][:] - e['psi_axis'][()]) / (e['psib'][()] - e['psi_axis'][()])


def polys(rl, rh, zl, zh):
    return np.stack([np.c_[rl, zl], np.c_[rh, zl], np.c_[rh, zh], np.c_[rl, zh]], axis=1)


def frame(ax, xl, yl, title):
    ax.contour(e['r'][:], e['z'][:], psin, levels=[1.0], colors='orange', linewidths=1.2)
    ax.add_collection(LineCollection(segs, colors='crimson', linewidths=0.9))
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect('equal')
    ax.set_xlabel('R (m)'); ax.set_ylabel('Z (m)'); ax.set_title(title)


full, zoom = ((1.8, 3.25), (Z0, Z1)), ((2.2, 2.9), (-0.85, -0.35))
cmap = ListedColormap(plt.cm.Blues(np.linspace(0.1, 0.85, nlev)))
norm = BoundaryNorm(np.arange(0.5, nlev + 1.5), nlev)

# Figure 1: uniform start, ni_jump refinement, X-point zoom
fig, ax = plt.subplots(1, 3, figsize=(14, 6.4), gridspec_kw={'width_ratios': [1, 1, 1.3]})
ur, uz = np.meshgrid(np.linspace(R0, R1, NR + 1), np.linspace(Z0, Z1, NZ + 1))
ax[0].add_collection(PolyCollection(
    polys(ur[:-1, :-1].ravel(), ur[:-1, 1:].ravel(), uz[:-1, :-1].ravel(), uz[1:, :-1].ravel()),
    facecolors='#eef3fa', edgecolors='k', linewidths=0.2))
frame(ax[0], *full, 'uniform level 1 (%dx%d)' % (NZ, NR))
for a, (xl, yl), t in ((ax[1], full, 'refined on ni_jump > 0.5'),
                       (ax[2], zoom, 'X-point and lower divertor')):
    pc = PolyCollection(polys(rlo, rhi, zlo, zhi), array=level, cmap=cmap, norm=norm,
                        edgecolors='k', linewidths=0.15)
    a.add_collection(pc)
    frame(a, xl, yl, t)
cb = fig.colorbar(pc, ax=ax[2], ticks=range(1, nlev + 1), shrink=0.7)
cb.set_label('grid level')
fig.tight_layout()
fig.savefig(f'{out}/west_refine_grid.png', dpi=130)

# Figure 2: rank ownership after balance_grid rcb cell
nproc = int(proc.max()) + 1
pcm = ListedColormap(plt.cm.tab10(np.arange(nproc) % 10))
pnorm = BoundaryNorm(np.arange(-0.5, nproc + 0.5), nproc)
fig, a = plt.subplots(figsize=(6.4, 7))
pc = PolyCollection(polys(rlo, rhi, zlo, zhi), array=proc, cmap=pcm, norm=pnorm,
                    edgecolors='k', linewidths=0.1)
a.add_collection(pc)
frame(a, *full, 'balance_grid rcb cell, %d ranks' % nproc)
cb = fig.colorbar(pc, ax=a, ticks=range(nproc), shrink=0.6)
cb.set_label('owning rank')
fig.tight_layout()
fig.savefig(f'{out}/west_partition.png', dpi=130)

print('cells', len(level), 'per level', np.bincount(level)[1:],
      'per rank', np.bincount(proc.astype(int)))
