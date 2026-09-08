#!/usr/bin/env python3
"""Independent oracle for the local-normal psi reflection."""

from pathlib import Path

import h5py
import numpy as np


THRESHOLD = 0.51
DT = 5.0e-3


def read_frames(path: Path):
    lines = path.read_text().splitlines()
    frames = []
    i = 0
    while i < len(lines):
        if lines[i] != "ITEM: TIMESTEP":
            i += 1
            continue
        step = int(lines[i + 1])
        count = int(lines[i + 3])
        j = i + 4
        while not lines[j].startswith("ITEM: ATOMS"):
            j += 1
        cols = lines[j].split()[2:]
        rows = [dict(zip(cols, line.split()))
                for line in lines[j + 1 : j + 1 + count]]
        frames.append((step, rows))
        i = j + 1 + count
    return frames


with h5py.File("../../input/plasma_st40_solps_corefill.h5", "r") as h5:
    rgrid = h5["equilibrium/r"][...]
    zgrid = h5["equilibrium/z"][...]
    psi = h5["equilibrium/psi"][...]
    psi_axis = float(h5["equilibrium/psi_axis"][()])
    psi_bry = float(h5["equilibrium/psib"][()])


def sample(r_value, z_value, gradient=False):
    i = int(np.clip(np.searchsorted(rgrid, r_value, side="right") - 1,
                    0, len(rgrid) - 2))
    j = int(np.clip(np.searchsorted(zgrid, z_value, side="right") - 1,
                    0, len(zgrid) - 2))
    dr = rgrid[i + 1] - rgrid[i]
    dz = zgrid[j + 1] - zgrid[j]
    t = np.clip((r_value - rgrid[i]) / dr, 0.0, 1.0)
    u = np.clip((z_value - zgrid[j]) / dz, 0.0, 1.0)
    p00, p10 = psi[j, i], psi[j, i + 1]
    p01, p11 = psi[j + 1, i], psi[j + 1, i + 1]
    denom = psi_bry - psi_axis
    value = ((1-t)*(1-u)*p00 + t*(1-u)*p10 +
             (1-t)*u*p01 + t*u*p11 - psi_axis) / denom
    if not gradient:
        return value
    grad_r = ((1-u)*(p10-p00) + u*(p11-p01)) / (dr*denom)
    grad_z = ((1-t)*(p01-p00) + t*(p11-p10)) / (dz*denom)
    return value, np.array([grad_z, grad_r], dtype=float)


frames = read_frames(Path("output_reflect/state"))
if len(frames) != 2 or any(len(rows) != 1 for _, rows in frames):
    raise SystemExit(f"FAIL: expected one particle at steps 0 and 1, got {frames}")

initial = frames[0][1][0]
final = frames[1][1][0]
x0 = np.array([float(initial["x"]), float(initial["y"])])
v0 = np.array([float(initial["vx"]), float(initial["vy"])])
proposed = x0 + DT*v0

lo, hi = 0.0, 1.0
for _ in range(80):
    mid = 0.5*(lo+hi)
    point = x0 + mid*(proposed-x0)
    if sample(point[1], point[0]) >= THRESHOLD:
        lo = mid
    else:
        hi = mid
fraction = 0.5*(lo+hi)
crossing = x0 + fraction*(proposed-x0)
_, normal = sample(crossing[1], crossing[0], gradient=True)
normal /= np.linalg.norm(normal)

expected_v = v0 - 2.0*np.dot(v0, normal)*normal
remaining = proposed-crossing
expected_x = crossing + remaining - 2.0*np.dot(remaining, normal)*normal

actual_x = np.array([float(final["x"]), float(final["y"])])
actual_v = np.array([float(final["vx"]), float(final["vy"])])

if not np.allclose(actual_x, expected_x, rtol=0.0, atol=2.0e-11):
    raise SystemExit(f"FAIL: reflected position {actual_x}, expected {expected_x}")
if not np.allclose(actual_v, expected_v, rtol=0.0, atol=2.0e-10):
    raise SystemExit(f"FAIL: reflected velocity {actual_v}, expected {expected_v}")
if sample(actual_x[1], actual_x[0]) < THRESHOLD:
    raise SystemExit("FAIL: reflected endpoint remains inside psi boundary")
if abs(np.dot(actual_v, normal) + np.dot(v0, normal)) > 2.0e-10:
    raise SystemExit("FAIL: normal velocity was not reversed")
if abs(np.linalg.norm(actual_v)-np.linalg.norm(v0)) > 2.0e-10:
    raise SystemExit("FAIL: reflection changed kinetic speed")

print(
    "PASS: crossing root, reflected endpoint, local-normal velocity, and "
    "kinetic speed match the independent psi oracle"
)
