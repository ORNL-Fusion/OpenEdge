"""Compare inner vs outer droplet trajectories from case.drag.mode.1.

Particles are separated by species type:
  type=1 → inner divertor (species I)   — plotted in blues
  type=2 → outer divertor (species O)   — plotted in reds/yellows

Each trajectory is colored by normalized radius r/r₀
  1.0 (blue/red) = initial size
  0.0 (yellow)   = fully evaporated

Arrows indicate the centrifugal direction (always outward, +R).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from utils import surface


DUMP_FILE = "case.drag.mode.1"
RHO       = 534.0

CMAP_INNER = "Blues_r"    # dark = large r (initial), light = evaporated
CMAP_OUTER = "plasma_r"   # bright = large r, dark = evaporated


# ── parser ────────────────────────────────────────────────────────────────────

def parse_file(filename):
    """Read SPARTA dump → arrays per data row, including particle type."""
    t, pid, typ = [], [], []
    x, y, z, vx, vy, vz, m, temp, rad = [], [], [], [], [], [], [], [], []
    with open(filename) as f:
        lines = f.readlines()
    i = ts = n = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "ITEM: TIMESTEP":
            ts = int(lines[i + 1]); i += 2
        elif s == "ITEM: NUMBER OF ATOMS":
            n = int(lines[i + 1]); i += 2
        elif s.startswith("ITEM: ATOMS"):
            headers = lines[i].split()[2:]
            idx = {h: j for j, h in enumerate(headers)}
            for _ in range(n):
                p = lines[i + 1].split()
                pid.append(int(p[idx["id"]]))
                t.append(ts)
                typ.append(int(p[idx["type"]]))
                x.append(float(p[idx["x"]]))
                y.append(float(p[idx["y"]]))
                z.append(float(p[idx["z"]]))
                vx.append(float(p[idx["vx"]]))
                vy.append(float(p[idx["vy"]]))
                vz.append(float(p[idx["vz"]]))
                m.append(float(p[idx["v_pmass"]]))
                temp.append(float(p[idx["temp"]]))
                rad.append(float(p[idx["radius"]]))
                i += 1
            i += 1
        else:
            i += 1
    return {
        "t":   np.array(t,   dtype=int),
        "pid": np.array(pid, dtype=int),
        "typ": np.array(typ, dtype=int),
        "x":   np.array(x),
        "y":   np.array(y),
        "vx":  np.array(vx),
        "vy":  np.array(vy),
        "vz":  np.array(vz),
        "rad": np.array(rad),
    }


def all_trajs(data, species_type):
    """Extract list of trajectory dicts for a given species type (1=inner, 2=outer)."""
    mask_type = data["typ"] == species_type
    pids = np.unique(data["pid"][mask_type])
    trajs = []
    for uid in pids:
        mask = mask_type & (data["pid"] == uid)
        order = np.argsort(data["t"][mask])
        xi  = data["x"][mask][order]
        yi  = data["y"][mask][order]
        ri  = data["rad"][mask][order]
        vxi = data["vx"][mask][order]
        vyi = data["vy"][mask][order]
        vzi = data["vz"][mask][order]
        rv  = ri[ri > 0]
        r0  = rv[0] if rv.size else 1.0
        trajs.append(dict(x=xi, y=yi, vx=vxi, vy=vyi, vz=vzi,
                          r=ri, rn=ri / r0, r0=r0))
    return trajs


def make_lc(x, y, rn, cmap, norm, lw=1.5, alpha=0.85):
    """LineCollection for one trajectory, colored by r/r₀."""
    pts  = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc   = LineCollection(segs, cmap=cmap, norm=norm, linewidth=lw, alpha=alpha)
    lc.set_array(0.5 * (rn[:-1] + rn[1:]))
    return lc


def summary(trajs, label):
    print(f"\n{label}: {len(trajs)} droplets")
    final_R = [tr["x"][-1] for tr in trajs]
    final_Z = [tr["y"][-1] for tr in trajs]
    birth_R = [tr["x"][0]  for tr in trajs]
    birth_Z = [tr["y"][0]  for tr in trajs]
    mass_loss = []
    for tr in trajs:
        rv = tr["r"][tr["r"] > 0]
        if rv.size >= 2:
            mass_loss.append(100.0 * (1.0 - (rv[-1]/rv[0])**3))
    if mass_loss:
        print(f"  Mass loss: {np.mean(mass_loss):.1f}% ± {np.std(mass_loss):.1f}%")
    dr = np.array(final_R) - np.array(birth_R)
    print(f"  ΔR (final−birth): mean={np.mean(dr):+.3f} m  "
          f"[{np.min(dr):+.3f}, {np.max(dr):+.3f}]")
    print(f"  Birth R: [{np.min(birth_R):.3f}, {np.max(birth_R):.3f}] m")
    print(f"  Final R: [{np.min(final_R):.3f}, {np.max(final_R):.3f}] m")
    print(f"  Final Z: [{np.min(final_Z):.3f}, {np.max(final_Z):.3f}] m")


# ── load ──────────────────────────────────────────────────────────────────────

if not os.path.exists(DUMP_FILE):
    raise FileNotFoundError(f"Missing: {DUMP_FILE}")

data  = parse_file(DUMP_FILE)
inner = all_trajs(data, species_type=1)
outer = all_trajs(data, species_type=2)

print(f"Loaded {len(inner)+len(outer)} total droplets  "
      f"({len(inner)} inner, {len(outer)} outer)")

summary(inner, "Inner divertor droplets (type=1)")
summary(outer, "Outer divertor droplets (type=2)")

# ── surfaces ──────────────────────────────────────────────────────────────────

wall = surface("wall.surf", "2D")
core = surface("core.surf", "2D")
Rwall, Zwall = wall.polygon.exterior.xy
Rcore, Zcore = core.polygon.exterior.xy


# ── figure: combined view ─────────────────────────────────────────────────────

norm = Normalize(vmin=0.0, vmax=1.0)
fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

for ax, trajs, cmap_name, label, color in [
    (axes[0], inner, CMAP_INNER, "Inner (type=1)", "steelblue"),
    (axes[1], outer, CMAP_OUTER, "Outer (type=2)", "tomato"),
]:
    ax.plot(Rwall, Zwall, "k-",            lw=2.0)
    ax.plot(Rcore, Zcore, color="#2f7d32", lw=2.0)

    for tr in trajs:
        if len(tr["x"]) < 2:
            ax.scatter(tr["x"], tr["y"], s=12, c=color)
            continue
        lc = make_lc(tr["x"], tr["y"], tr["rn"], cmap_name, norm)
        ax.add_collection(lc)
        # birth (circle) and death/final (cross)
        ax.scatter(tr["x"][0],  tr["y"][0],  s=20, marker="o",
                   color="k", zorder=4, linewidths=0)
        ax.scatter(tr["x"][-1], tr["y"][-1], s=25, marker="x",
                   color=color, zorder=4, linewidths=1.5)

    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(r"$r\,/\,r_0$", fontsize=9)
    cb.ax.set_yticks([0.0, 0.5, 1.0])
    cb.ax.set_yticklabels(["0 (evap.)", "0.5", "1.0 (init.)"], fontsize=8)

    ax.set_title(f"{label} droplets", fontsize=11)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.grid(alpha=0.2, linestyle="--")
    ax.autoscale_view()
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylim(-4.2, -1.5)

# shared legend
wall_patch  = mpatches.Patch(color="k",         label="Wall")
core_patch  = mpatches.Patch(color="#2f7d32",   label="Core")
birth_patch = plt.Line2D([0], [0], marker="o", color="k",
                         label="Birth", linestyle="None", markersize=5)
end_patch   = plt.Line2D([0], [0], marker="x", color="grey",
                         label="Final pos.", linestyle="None", markersize=6)
fig.legend(handles=[wall_patch, core_patch, birth_patch, end_patch],
           loc="upper center", ncol=4, fontsize=9, frameon=True)

fig.suptitle("Droplet trajectories — inner vs outer divertor\n"
             "cylindrical yes  (DUSTT-style: centrifugal + angular-momentum)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])

os.makedirs("Figs", exist_ok=True)
fig.savefig("Figs/trajs_inner_outer.png", dpi=300,
            bbox_inches="tight", facecolor="white")
print("\nSaved Figs/trajs_inner_outer.png")
plt.show()
