"""Shared helpers for writing /wall_flux/* into plasma.h5.

The /wall_flux/ group is a plasma-source-agnostic representation of
per-species ion fluxes onto the wall (positive = into wall), evaluated
at scattered (R, Z) sample points. Each plasma converter (SOLPS,
SOLEDGE3X, OEDGE, ...) extracts wall fluxes from its own boundary-face
data and writes the same schema, so the OpenEdge runtime can query at
any geometric (R, Z) without knowing which solver produced the data.

Schema:
  /wall_flux/r            (N,)            m
  /wall_flux/z            (N,)            m
  /wall_flux/s            (N,)            m       arc length, optional
  /wall_flux/gamma_i      (nion, N)       m^-2 s^-1
  /wall_flux/te           (N,)            eV      optional
  /wall_flux/ti           (N,)            eV      optional
  /wall_flux/area         (N,)            m^2     optional
  /wall_flux/normal_r,z   (N,)            -       outward unit normal, optional
  /wall_flux/b_r,b_z,b_t  (N,)            T       B-field at sample, optional
                                                  (used by consumers to compute
                                                  smooth incidence angle)

Group attrs:
  source         : str
  extraction     : str
  species_names  : array of str
"""
from __future__ import annotations

import math
from typing import Optional

import h5py
import numpy as np


def build_wall_flux_solps(ft31, nx: int, ny: int, nion: int,
                          crx4: np.ndarray, cry4: np.ndarray) -> dict:
    """Build wall_flux from SOLPS-ITER fort.31 staggered fluxes.

    Walks the three SOLPS wall families in canonical order
    (inner target -> outer SOL -> outer target). Each B2 wall-boundary
    face contributes one (R, Z) sample. Per-species flux density is
    fnixb / fniyb projected onto the face area, with sign flipped so
    positive = into wall (regardless of which face).

    crx4, cry4 are the cell-corner arrays shaped (nx+2, ny+2, 4) with
    SOLPS corner ordering: 0=ll, 1=lr, 2=ul, 3=ur.
    """
    rs, zs, ar, gs, tes, tis, nrs, nzs = [], [], [], [], [], [], [], []

    def push(R_a, Z_a, R_b, Z_b, gamma_per_s, te, ti, sign):
        Rm = 0.5 * (R_a + R_b)
        Zm = 0.5 * (Z_a + Z_b)
        L = math.hypot(R_b - R_a, Z_b - Z_a)
        a = 2.0 * math.pi * Rm * L
        # Outward normal: rotate edge tangent by -90deg, then flip with sign.
        # tangent (tR, tZ) = (Rb-Ra, Zb-Za)/L; -90 rotation gives (tZ, -tR).
        if L > 0.0:
            tR = (R_b - R_a) / L
            tZ = (Z_b - Z_a) / L
            # Use sign to choose which side is "outward".
            n_R = sign *  tZ
            n_Z = sign * (-tR)
        else:
            n_R = n_Z = 0.0
        rs.append(Rm); zs.append(Zm); ar.append(a)
        gs.append(np.asarray(gamma_per_s, dtype=np.float64) /
                  max(a, 1e-30))
        tes.append(float(te)); tis.append(float(ti))
        nrs.append(n_R); nzs.append(n_Z)

    # 1) Inner target: ix=1, iy=1..ny. Wall edge = corners 0-2 (left side).
    #    fnixb[0, iy, :] is the right face of guard cell ix=0 = the wall.
    #    Sign in -x direction; ions flowing wallward give negative fnixb,
    #    so flip sign to make gamma>0 = into wall.
    for iy in range(1, ny + 1):
        push(crx4[1, iy, 0], cry4[1, iy, 0],
             crx4[1, iy, 2], cry4[1, iy, 2],
             -ft31.fnixb[0, iy, :],
             ft31.teb[1, iy], ft31.tib[1, iy], sign=-1)

    # 2) Outer SOL: iy=ny, ix=1..nx. Wall edge = corners 2-3 (top side).
    #    fniyb[ix, ny, :] is the right face of cell iy=ny in +y direction
    #    = the wall.  Sign positive flow = into wall.
    for ix in range(1, nx + 1):
        push(crx4[ix, ny, 2], cry4[ix, ny, 2],
             crx4[ix, ny, 3], cry4[ix, ny, 3],
             +ft31.fniyb[ix, ny, :],
             ft31.teb[ix, ny], ft31.tib[ix, ny], sign=+1)

    # 3) Outer target: ix=nx, iy=1..ny. Wall edge = corners 1-3 (right
    #    side). fnixb[nx, iy, :] is the right face of cell ix=nx (the
    #    wall). Sign positive flow = into wall.
    for iy in range(1, ny + 1):
        push(crx4[nx, iy, 1], cry4[nx, iy, 1],
             crx4[nx, iy, 3], cry4[nx, iy, 3],
             +ft31.fnixb[nx, iy, :],
             ft31.teb[nx, iy], ft31.tib[nx, iy], sign=+1)

    rs = np.asarray(rs, dtype=np.float64)
    zs = np.asarray(zs, dtype=np.float64)
    ar = np.asarray(ar, dtype=np.float64)
    g = np.stack(gs, axis=1)            # (nion, N)
    tes = np.asarray(tes, dtype=np.float64)
    tis = np.asarray(tis, dtype=np.float64)
    nrs = np.asarray(nrs, dtype=np.float64)
    nzs = np.asarray(nzs, dtype=np.float64)

    seg_lens = np.hypot(np.diff(rs), np.diff(zs))
    s_arc = np.concatenate([[0.0], np.cumsum(seg_lens)])

    return {
        "r": rs, "z": zs, "s": s_arc,
        "gamma_i": g,
        "te": tes, "ti": tis, "area": ar,
        "normal_r": nrs, "normal_z": nzs,
    }


def write_wall_flux_h5(f: h5py.File, wf: dict,
                       source: str, extraction: str,
                       species_names) -> None:
    g = f.create_group("wall_flux")
    g.attrs["source"] = source
    g.attrs["extraction"] = extraction
    sdt = h5py.string_dtype(encoding="utf-8")
    g.attrs.create("species_names",
                   np.array(list(species_names), dtype=object), dtype=sdt)
    for k, v in wf.items():
        g.create_dataset(k, data=v)
