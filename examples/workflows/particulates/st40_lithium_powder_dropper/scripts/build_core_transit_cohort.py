#!/usr/bin/env python3
"""Build a deterministic ST40 grain cohort and its manifest.

OpenEdge axisymmetric storage is ``(x, y, z) = (Z, R, phi)``.  User-facing
launch angles in the manifest remain physical: zero degrees is vertically
downward (-Z), and positive angles point toward increasing R.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WALL = ROOT / "input" / "st40_wall_axi.surf"
PARTICLES = ROOT / "input" / "core_transit_cohort.particles"
MANIFEST = ROOT / "input" / "core_transit_cohort.csv"

DROP_LINES = (16, 17)
SPEEDS_M_S = (1.0, 1.25, 1.5)
ANGLES_DEG = (-15.0, -7.5, 0.0, 7.5, 15.0)
INWARD_OFFSET_M = 1.0e-3


def read_surface(path: Path) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[int, int]]]:
    lines = path.read_text().splitlines()
    ip = lines.index("Points") + 1
    while not lines[ip].strip():
        ip += 1
    points: dict[int, tuple[float, float]] = {}
    while ip < len(lines) and lines[ip].strip():
        idx, z, r = lines[ip].split()
        points[int(idx)] = (float(z), float(r))
        ip += 1

    il = lines.index("Lines") + 1
    while not lines[il].strip():
        il += 1
    segments: dict[int, tuple[int, int]] = {}
    while il < len(lines) and lines[il].strip():
        idx, p1, p2 = lines[il].split()
        segments[int(idx)] = (int(p1), int(p2))
        il += 1
    return points, segments


def polygon_contains(z: float, r: float, points: dict[int, tuple[float, float]]) -> bool:
    """Odd/even point-in-polygon test for the closed, ordered ST40 wall."""
    poly = [points[i] for i in sorted(points)]
    inside = False
    j = len(poly) - 1
    for i, (zi, ri) in enumerate(poly):
        zj, rj = poly[j]
        if (ri > r) != (rj > r):
            z_cross = (zj - zi) * (r - ri) / (rj - ri) + zi
            if z < z_cross:
                inside = not inside
        j = i
    return inside


def build() -> list[dict[str, float | int]]:
    points, segments = read_surface(WALL)
    rows: list[dict[str, float | int]] = []
    particle_id = 1

    for line_id in DROP_LINES:
        p1, p2 = segments[line_id]
        z1, r1 = points[p1]
        z2, r2 = points[p2]
        dz, dr = z2 - z1, r2 - r1
        length = math.hypot(dz, dr)
        # The wall is CCW in (Z,R), so (-dR,+dZ) is the documented inward
        # normal.  Offset the read-in particles into the gas by 1 mm.
        nz, nr = -dr / length, dz / length
        z0 = 0.5 * (z1 + z2) + INWARD_OFFSET_M * nz
        r0 = 0.5 * (r1 + r2) + INWARD_OFFSET_M * nr
        if not polygon_contains(z0, r0, points):
            raise RuntimeError(f"line {line_id}: inward-offset start is outside the vessel")

        for speed in SPEEDS_M_S:
            for angle_deg in ANGLES_DEG:
                angle = math.radians(angle_deg)
                # Physical launch: dZ=-cos(theta), dR=+sin(theta).
                vz = -speed * math.cos(angle)
                vr = speed * math.sin(angle)
                rows.append(
                    {
                        "id": particle_id,
                        "dropper_line": line_id,
                        "speed_m_s": speed,
                        "angle_deg": angle_deg,
                        "z0_m": z0,
                        "r0_m": r0,
                        "vz0_m_s": vz,
                        "vr0_m_s": vr,
                    }
                )
                particle_id += 1
    return rows


def write(rows: list[dict[str, float | int]]) -> None:
    with MANIFEST.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    with PARTICLES.open("w") as stream:
        stream.write(
            "ITEM: TIMESTEP\n0\n"
            "ITEM: NUMBER OF ATOMS\n"
            f"{len(rows)}\n"
            "ITEM: BOX BOUNDS oo ao pp\n"
            "-1.0 1.0\n0.0 1.05\n-0.05 0.05\n"
            "ITEM: ATOMS id type x y z vx vy vz\n"
        )
        for row in rows:
            # Axisymmetric OpenEdge slots are x=Z, y=R, z=phi.
            stream.write(
                f"{row['id']} 1 {row['z0_m']:.14g} {row['r0_m']:.14g} 0 "
                f"{row['vz0_m_s']:.14g} {row['vr0_m_s']:.14g} 0\n"
            )


def main() -> None:
    rows = build()
    write(rows)
    print(f"wrote {PARTICLES.relative_to(ROOT)}: {len(rows)} deterministic grains")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    print(
        f"launch grid: {len(DROP_LINES)} dropper segments x {len(SPEEDS_M_S)} speeds "
        f"x {len(ANGLES_DEG)} angles"
    )


if __name__ == "__main__":
    main()
