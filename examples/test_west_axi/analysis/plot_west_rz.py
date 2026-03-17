#!/usr/bin/env python3
import math
from pathlib import Path

from shapely.geometry import Point, Polygon
import glob


class surface:
    """Class to represent and process surface geometry files.

    Attributes:
        filename (str): Path to the surface file.
        geom_type (str): Type of geometry represented in the file.
        points (dict): Dictionary of point IDs and their coordinates.
        lines (list): List of line definitions including line ID, materials, and point IDs.
        polygon (Polygon): Shapely Polygon object created from the points.
        id (str): Identifier for the surface, derived from filename.
    """

    def __init__(self, filename, geom_type):
        """Initialize a Surface object by reading geometry data from a file."""
        # read surface file
        with open(filename, 'r') as f:
            # skip first two lines
            data = f.readlines()[2:]
            # read the number of points and lines
            num_points = int(data[0].split()[0])
            num_lines = int(data[1].split()[0])
            print('Number of points: ', num_points)
            print('Number of lines: ', num_lines)
            # read points
            self.points = {}
            for line in data[5:num_points+5]:
                point_id, *point_coords = map(float, line.split())
                self.points[int(point_id)] = tuple(point_coords)
            # read lines
            self.lines = []
            for line in data[num_points+8:]:
                line_parts = line.split()
                materials = line_parts[-1]
                line_id = int(line_parts[0])
                line_points = [int(line_parts[1]), int(line_parts[2])]
                self.lines.append((line_id, str(materials), line_points))
        # create Polygon object from points
        points_list = [list(p) for p in self.points.values()]
        self.polygon = Polygon(points_list)
        # set surface ID
        self.id = filename.split('.')[0]
        

def read_series(path: Path):
    lines = path.read_text().splitlines()
    i = 0
    out = {}
    rlo = rhi = zlo = zhi = None
    while i < len(lines):
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue
        step = int(lines[i + 1].strip())
        i += 2
        if i >= len(lines) or not lines[i].startswith("ITEM: NUMBER OF ATOMS"):
            raise RuntimeError("Bad dump format: NUMBER OF ATOMS")
        nat = int(lines[i + 1].strip())
        i += 2
        if i >= len(lines) or not lines[i].startswith("ITEM: BOX BOUNDS"):
            raise RuntimeError("Bad dump format: BOX BOUNDS")
        i += 1
        # Read BOX BOUNDS lines (2D/3D safe) until ATOMS header.
        bounds = []
        while i < len(lines) and not lines[i].startswith("ITEM: ATOMS"):
            parts = lines[i].split()
            if len(parts) >= 2:
                try:
                    bounds.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
            i += 1
        if bounds and rlo is None:
            rlo, rhi = bounds[0]
            zlo, zhi = bounds[1] if len(bounds) > 1 else (None, None)
        if i >= len(lines) or not lines[i].startswith("ITEM: ATOMS"):
            raise RuntimeError("Bad dump format: ATOMS")
        cols = lines[i].split()[2:]
        i += 1
        colmap = {name: idx for idx, name in enumerate(cols)}
        if "id" not in colmap or "x" not in colmap or "y" not in colmap:
            raise RuntimeError("ATOMS header must contain id,x,y columns")

        # Read exactly nat atom rows for this timestep block.
        for _ in range(nat):
            if i >= len(lines) or lines[i].startswith("ITEM:"):
                raise RuntimeError("Bad dump format: atom row count does not match NUMBER OF ATOMS")
            parts = lines[i].split()
            pid = int(parts[colmap["id"]])
            x = float(parts[colmap["x"]])
            y = float(parts[colmap["y"]])
            if pid not in out:
                out[pid] = []
            out[pid].append((step, x, y))
            i += 1
    for pid in out:
        # keep file order to preserve contiguous trajectory blocks
        pass
    return out, (rlo, rhi, zlo, zhi)


def main():
    import matplotlib.pyplot as plt

    base = Path(__file__).resolve().parent
    dumpf = base / "output" / "state.west"
    series_by_id, bounds = read_series(dumpf)
    if not series_by_id:
        raise RuntimeError("No particle trajectory found in output/state.west")

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 5.2), dpi=220, constrained_layout=True)
    rlo, rhi, zlo, zhi = bounds
    rspan = (rhi - rlo) if (rlo is not None and rhi is not None) else 1.0
    zspan = (zhi - zlo) if (zlo is not None and zhi is not None) else 1.0
    jump_r = 0.25 * abs(rspan)
    jump_z = 0.25 * abs(zspan)

    def split_segments(traj):
        segs = []
        cur = []
        prev = None
        for p in traj:
            if prev is not None:
                step_dec = p[0] <= prev[0]
                jump = abs(p[1] - prev[1]) > jump_r or abs(p[2] - prev[2]) > jump_z
                if step_dec or jump:
                    if len(cur) > 1:
                        segs.append(cur)
                    cur = []
            cur.append(p)
            prev = p
        if len(cur) > 1:
            segs.append(cur)
        return segs

    ntraj = len(series_by_id)
    for idx, (pid, traj) in enumerate(sorted(series_by_id.items())):
        segs = split_segments(traj)
        for iseg, seg in enumerate(segs):
            r = [s[1] for s in seg]
            z = [s[2] for s in seg]
            ax.plot(
                r,
                z,
                linestyle="-",
                linewidth=1.2,
                marker="o",
                markersize=1.4,
                markerfacecolor="none",
                markeredgewidth=0.7,
                markevery=max(1, len(r) // 200),
                alpha=0.9,
                label=(f"id {pid}" if (ntraj <= 8 and iseg == 0) else None),
            )
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(f"WEST trajectories (R-Z), B-only, N={ntraj}")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    if ntraj <= 8:
        ax.legend(frameon=True, framealpha=0.9, loc="best")


    core = surface('input/core.txt', "2D")
    domain = core.polygon
    rcore, zcore = domain.exterior.xy
    ax.plot(rcore, zcore, 'g', lw=2.5)
   
    wall = surface('input/wall.txt', "2D")
    domain = wall.polygon
    rwall, zwall = domain.exterior.xy
    ax.plot(rwall, zwall, 'k', lw=2.5)
    
    out = base / "output" / "trajectory.west.rz.png"
    fig.savefig(out, dpi=220)
    print(f"Wrote: {out}")
    print(f"Plotted trajectories: {ntraj}")
    plt.show()

if __name__ == "__main__":
    main()
