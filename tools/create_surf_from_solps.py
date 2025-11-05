import numpy as np
import os
from collections import Counter, defaultdict
from matplotlib import pyplot as plt

data_path='/Users/42d/ORNL Dropbox/Abdou DIaw/addLi/fnacore=6.00e22_pheat=90.00MW_cont_dt=1e-6_te_up'

fname = os.path.join(data_path, "mesh.extra")

d = np.loadtxt(fname)                      # columns: R1 Z1 R2 Z2
p1 = d[:, [0,1]]
p2 = d[:, [2,3]]
pts = np.vstack([p1, p2])

# --- dedup vertices with a small tolerance ---
dec = 8  # decimals to keep; adjust if needed
key = lambda p: tuple(np.round(p, dec))
id_map, coords = {}, []
def vid(p):
    k = key(p)
    if k not in id_map:
        id_map[k] = len(coords)
        coords.append(p.astype(float))
    return id_map[k]

v1 = np.array([vid(p) for p in p1])
v2 = np.array([vid(p) for p in p2])
coords = np.array(coords)                  # shape (Nvert, 2)

# --- boundary edges = edges that appear only once (interior edges appear twice) ---
edges = [tuple(sorted(e)) for e in zip(v1, v2)]
cnt = Counter(edges)
b_edges = [e for e,c in cnt.items() if c == 1]

# --- adjacency for the boundary graph ---
adj = defaultdict(list)
for a,b in b_edges:
    adj[a].append(b); adj[b].append(a)

# --- walk one boundary loop (pick the leftmost vertex as start) ---
start = min(adj.keys(), key=lambda i: (coords[i,0], coords[i,1]))
path = [start]; prev = None; cur = start
while True:
    nbrs = adj[cur]
    nxt = nbrs[0] if nbrs[0] != prev else (nbrs[1] if len(nbrs) > 1 else None)
    if nxt is None or nxt == start:
        break
    path.append(nxt)
    prev, cur = cur, nxt

# (optional) choose the longest loop if multiple exist
# You can repeat the walk from other start nodes and keep the max length.

# --- write SPARTA surf file (Points then Lines, closing the loop) ---
out = "wall_from_mesh.surf"
with open(out, "w") as f:
    n = len(path)
    f.write("\n")
    f.write(f"{n} points\n{n} lines\n\nPoints\n\n")
    for i, vidx in enumerate(path, start=1):
        R, Z = coords[vidx]
        f.write(f"{i} {R:.8f} {Z:.8f}\n")
    f.write("\n")
    f.write("\nLines\n")
    f.write("\n")
    for i in range(1, n):
        f.write(f"{i} {i} {i+1}\n")
    f.write(f"{n} {n} 1\n")  # close the loop

print(f"wrote {out} with {len(path)} boundary points")


Rb, Zb = coords[path, 0], coords[path, 1]
plt.figure()
plt.plot(Rb, Zb, 'k-', lw=2)
plt.axis('equal'); plt.tight_layout(); plt.show()
