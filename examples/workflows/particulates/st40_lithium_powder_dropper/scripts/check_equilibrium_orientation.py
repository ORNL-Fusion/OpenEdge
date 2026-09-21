#!/usr/bin/env python3
"""Fail-closed ST40 equilibrium/mesh orientation check."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def boundary_loops(triangles: np.ndarray) -> list[list[int]]:
    counts = Counter(
        tuple(sorted((int(a), int(b))))
        for tri in triangles
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
    )
    adj: dict[int, list[int]] = defaultdict(list)
    for (a, b), count in counts.items():
        if count == 1:
            adj[a].append(b)
            adj[b].append(a)
    if not adj or any(len(v) != 2 for v in adj.values()):
        raise RuntimeError("plasma-mesh boundary is not a set of closed loops")

    loops: list[list[int]] = []
    seen: set[int] = set()
    for start in adj:
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        previous = None
        current = start
        while True:
            following = adj[current][0]
            if following == previous:
                following = adj[current][1]
            if following == start:
                break
            if following in seen:
                raise RuntimeError("plasma-mesh boundary self-intersects")
            loop.append(following)
            seen.add(following)
            previous, current = current, following
        loops.append(loop)
    return loops


def signed_area(r: np.ndarray, z: np.ndarray) -> float:
    return 0.5 * float(np.sum(r * np.roll(z, -1) - np.roll(r, -1) * z))


def check(path: Path) -> dict[str, float]:
    with h5py.File(path, "r") as h5:
        triangles = h5["mesh/triangles"][:]
        rv = h5["mesh/vtx_r"][:]
        zv = h5["mesh/vtx_z"][:]
        br = h5["mesh/vtx_br"][:]
        bz = h5["mesh/vtx_bz"][:]
        loops = boundary_loops(triangles)
        inner = min(loops, key=lambda q: signed_area(rv[q], zv[q]))

        r = h5["equilibrium/r"][:]
        z = h5["equilibrium/z"][:]
        psi = h5["equilibrium/psi"][:]
        if psi.shape != (len(z), len(r)):
            raise RuntimeError(
                f"equilibrium/psi shape {psi.shape} != (nZ,nR) "
                f"= ({len(z)},{len(r)})")
        psi_axis = float(h5["equilibrium/psi_axis"][()])
        psi_boundary = float(h5["equilibrium/psib"][()])
        interp = RegularGridInterpolator((z, r), psi, bounds_error=True)
        psi_n = (
            interp(np.column_stack((zv[inner], rv[inner]))) - psi_axis
        ) / (psi_boundary - psi_axis)

        alignment = []
        for k, vertex in enumerate(inner):
            before = inner[k - 1]
            after = inner[(k + 1) % len(inner)]
            tangent = np.array((rv[after] - rv[before], zv[after] - zv[before]))
            tangent /= np.linalg.norm(tangent)
            bpol = np.array((br[vertex], bz[vertex]))
            bpol /= np.linalg.norm(bpol)
            alignment.append(abs(float(np.dot(tangent, bpol))))

    result = {
        "inner_vertices": float(len(inner)),
        "psi_n_median": float(np.median(psi_n)),
        "psi_n_std": float(np.std(psi_n)),
        "psi_n_min": float(np.min(psi_n)),
        "psi_n_max": float(np.max(psi_n)),
        "b_tangent_median": float(np.median(alignment)),
        "b_tangent_min": float(np.min(alignment)),
    }
    result["pass"] = float(
        result["psi_n_std"] < 2.0e-3
        and result["b_tangent_median"] > 0.999
        and result["b_tangent_min"] > 0.995
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plasma", type=Path)
    args = parser.parse_args()
    result = check(args.plasma)
    for key, value in result.items():
        print(f"{key}: {value}")
    if not result["pass"]:
        raise SystemExit("FAIL: equilibrium R/Z orientation gate")
    print("PASS: equilibrium R/Z orientation gate")


if __name__ == "__main__":
    main()
