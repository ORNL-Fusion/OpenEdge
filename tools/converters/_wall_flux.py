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
  /wall_flux/projection_distance (N,)      m       source face midpoint to
                                                  exact wall, optional

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


def _as_2d(dataset) -> np.ndarray:
    """Read an S3X 2-D field, accepting its usual trailing singleton axis."""
    values = np.asarray(dataset[...], dtype=np.float64)
    if values.ndim == 3 and values.shape[2] == 1:
        values = values[:, :, 0]
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D S3X field, got shape {values.shape}")
    return values


def _s3x_metric_on_faces(values: np.ndarray, direction: str,
                          ni: int, nj: int, nghost: int) -> np.ndarray:
    """Average a cell-centred S3X metric onto physical-zone faces."""
    expected = (ni + 2 * nghost, nj + 2 * nghost)
    if values.shape != expected:
        raise ValueError(
            f"S3X metric shape {values.shape} does not match {expected} "
            f"for a ({ni}, {nj}) zone with {nghost} ghost cells"
        )
    if direction == "psi":
        low = values[nghost - 1:nghost + ni, nghost:nghost + nj]
        high = values[nghost:nghost + ni + 1, nghost:nghost + nj]
    elif direction == "theta":
        low = values[nghost:nghost + ni, nghost - 1:nghost + nj]
        high = values[nghost:nghost + ni, nghost:nghost + nj + 1]
    else:
        raise ValueError(f"unknown S3X face direction {direction!r}")
    return 0.5 * (low + high)


def _s3x_cell_field(dataset, ni: int, nj: int,
                    nghost: int) -> np.ndarray:
    """Read an S3X cell field and strip solver ghost cells when present."""
    values = _as_2d(dataset)
    if values.shape == (ni, nj):
        return values
    expected = (ni + 2 * nghost, nj + 2 * nghost)
    if values.shape != expected:
        raise ValueError(
            f"S3X cell field shape {values.shape} is neither ({ni}, {nj}) "
            f"nor {expected}"
        )
    return values[nghost:nghost + ni, nghost:nghost + nj]


def _project_to_wall(r: np.ndarray, z: np.ndarray,
                     wall_r: np.ndarray, wall_z: np.ndarray):
    """Project scattered points onto a polyline and return R, Z, s, distance."""
    wr = np.asarray(wall_r, dtype=np.float64).reshape(-1)
    wz = np.asarray(wall_z, dtype=np.float64).reshape(-1)
    if wr.size != wz.size or wr.size < 2:
        raise ValueError("S3X wall polyline must contain at least two R/Z points")
    if not (np.isclose(wr[0], wr[-1]) and np.isclose(wz[0], wz[-1])):
        wr = np.concatenate([wr, wr[:1]])
        wz = np.concatenate([wz, wz[:1]])

    starts = np.column_stack([wr[:-1], wz[:-1]])
    vectors = np.column_stack([np.diff(wr), np.diff(wz)])
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 0.0
    starts = starts[valid]
    vectors = vectors[valid]
    lengths = lengths[valid]
    if lengths.size == 0:
        raise ValueError("S3X wall polyline contains no nonzero-length segments")
    length2 = lengths * lengths
    s0 = np.concatenate([[0.0], np.cumsum(lengths[:-1])])

    projected = np.empty((r.size, 2), dtype=np.float64)
    s = np.empty(r.size, dtype=np.float64)
    distance = np.empty(r.size, dtype=np.float64)
    for k, point in enumerate(np.column_stack([r, z])):
        t = np.sum((point - starts) * vectors, axis=1) / length2
        t = np.clip(t, 0.0, 1.0)
        candidates = starts + t[:, None] * vectors
        d2 = np.sum((point - candidates) ** 2, axis=1)
        j = int(np.argmin(d2))
        projected[k] = candidates[j]
        s[k] = s0[j] + t[j] * lengths[j]
        distance[k] = math.sqrt(float(d2[j]))
    return projected[:, 0], projected[:, 1], s, distance


def build_wall_flux_s3x(mesh_h5: h5py.File, metric_h5: h5py.File,
                        plasma_h5: h5py.File, ref_h5: h5py.File,
                        ion_spec_indices, main_ion_spec: int,
                        wall_r: np.ndarray, wall_z: np.ndarray,
                        nghost: int = 2) -> dict:
    """Build wall flux samples from native SOLEDGE3X staggered fluxes.

    SOLEDGE3X stores ``fluxn/psi`` and ``fluxn/theta`` on zone faces as
    metric-integrated particle rates.  A physical wall face is either an
    internal face across which ``chi`` changes between plasma (0) and
    material (1), or a zone boundary whose neighbour code is -1.  Dividing
    by ``J*sqrt(g^ii)`` removes the face metric; the sign of ``J`` and the
    plasma-to-material direction are both retained so positive output means
    flux into the wall.

    Face midpoints are projected onto the exact S3X wall polyline.  This
    removes the small penalisation-grid offset before another code queries
    the scattered values on its own representation of the same wall.
    """
    ion_spec_indices = [int(i) for i in ion_spec_indices]
    if not ion_spec_indices:
        raise ValueError("at least one S3X ion species is required")
    if main_ion_spec not in ion_spec_indices:
        raise ValueError(f"main ion spec{main_ion_spec} is not an ion species")

    n0 = float(np.asarray(ref_h5["n0"][...]).reshape(-1)[0])
    c0 = float(np.asarray(ref_h5["c0"][...]).reshape(-1)[0])
    rho0 = float(np.asarray(ref_h5["rho0"][...]).reshape(-1)[0])
    t0 = float(np.asarray(ref_h5["T0"][...]).reshape(-1)[0])
    b0 = float(np.asarray(ref_h5["B0"][...]).reshape(-1)[0])
    r0 = float(np.asarray(mesh_h5["R0"][...]).reshape(-1)[0])
    a0 = float(np.asarray(mesh_h5["a0"][...]).reshape(-1)[0])
    nzones = int(np.asarray(mesh_h5["NZones"][...]).reshape(-1)[0])
    neighbor_key = "neighbors" if "neighbors" in mesh_h5 else "neighb"
    neighbors = np.asarray(mesh_h5[neighbor_key][...], dtype=np.int64)
    if neighbors.shape[0] < 4 or neighbors.shape[1] != nzones:
        raise ValueError(f"unexpected S3X neighbours shape {neighbors.shape}")

    flux_scale = n0 * c0 * rho0 ** 2
    records = []

    for izone in range(1, nzones + 1):
        zone = f"zone{izone}"
        chi = _as_2d(mesh_h5[f"{zone}/chi"]) > 0.5
        ni, nj = chi.shape
        r_corner = r0 + a0 * _as_2d(mesh_h5[f"{zone}/Rcorners"])
        z_corner = a0 * _as_2d(mesh_h5[f"{zone}/Zcorners"])
        if r_corner.shape != (ni + 1, nj + 1):
            raise ValueError(
                f"{zone} corner shape {r_corner.shape} does not match chi {chi.shape}"
            )

        j_metric = _as_2d(metric_h5[f"{zone}/J"])
        psi_metric = _s3x_metric_on_faces(
            j_metric, "psi", ni, nj, nghost
        ) * np.sqrt(np.maximum(_s3x_metric_on_faces(
            _as_2d(metric_h5[f"{zone}/g11"]),
            "psi", ni, nj, nghost), 0.0)) * rho0 ** 2
        theta_metric = _s3x_metric_on_faces(
            j_metric, "theta", ni, nj, nghost
        ) * np.sqrt(np.maximum(_s3x_metric_on_faces(
            _as_2d(metric_h5[f"{zone}/g22"]),
            "theta", ni, nj, nghost), 0.0)) * rho0 ** 2

        flux_psi = []
        flux_theta = []
        for spec in ion_spec_indices:
            base = f"{zone}/spec{spec}/fluxn"
            flux_psi.append(_as_2d(plasma_h5[f"{base}/psi"]))
            flux_theta.append(_as_2d(plasma_h5[f"{base}/theta"]))
        flux_psi = np.stack(flux_psi)
        flux_theta = np.stack(flux_theta)
        if flux_psi.shape[1:] != (ni + 1, nj):
            raise ValueError(f"{zone} psi flux shape {flux_psi.shape} is invalid")
        if flux_theta.shape[1:] != (ni, nj + 1):
            raise ValueError(f"{zone} theta flux shape {flux_theta.shape} is invalid")

        te = _s3x_cell_field(plasma_h5[f"{zone}/spec0/T"], ni, nj,
                             nghost) * t0
        ti = _s3x_cell_field(
            plasma_h5[f"{zone}/spec{main_ion_spec}/T"], ni, nj,
            nghost) * t0
        ne = _s3x_cell_field(plasma_h5[f"{zone}/spec0/n"], ni, nj,
                             nghost) * n0
        br = _s3x_cell_field(mesh_h5[f"{zone}/Br"], ni, nj, nghost) * b0
        bz = _s3x_cell_field(mesh_h5[f"{zone}/Bz"], ni, nj, nghost) * b0
        bt = _s3x_cell_field(mesh_h5[f"{zone}/Bphi"], ni, nj, nghost) * b0

        def push(direction, i, j, plasma_i, plasma_j, wall_sign):
            if direction == "psi":
                pa = np.array([r_corner[i, j], z_corner[i, j]])
                pb = np.array([r_corner[i, j + 1], z_corner[i, j + 1]])
                face_metric = psi_metric[i, j]
                raw_flux = flux_psi[:, i, j]
            else:
                pa = np.array([r_corner[i, j], z_corner[i, j]])
                pb = np.array([r_corner[i + 1, j], z_corner[i + 1, j]])
                face_metric = theta_metric[i, j]
                raw_flux = flux_theta[:, i, j]
            if not np.isfinite(face_metric) or abs(face_metric) <= 1.0e-300:
                raise ValueError(f"zero/invalid S3X metric on {zone} {direction} face")

            midpoint = 0.5 * (pa + pb)
            tangent = pb - pa
            length = float(np.linalg.norm(tangent))
            if length <= 0.0:
                raise ValueError(f"zero-length S3X wall face in {zone}")
            normal = np.array([tangent[1], -tangent[0]]) / length
            cell_center = 0.25 * np.array([
                r_corner[plasma_i, plasma_j]
                + r_corner[plasma_i + 1, plasma_j]
                + r_corner[plasma_i, plasma_j + 1]
                + r_corner[plasma_i + 1, plasma_j + 1],
                z_corner[plasma_i, plasma_j]
                + z_corner[plasma_i + 1, plasma_j]
                + z_corner[plasma_i, plasma_j + 1]
                + z_corner[plasma_i + 1, plasma_j + 1],
            ])
            if np.dot(normal, midpoint - cell_center) < 0.0:
                normal *= -1.0

            gamma = wall_sign * raw_flux * flux_scale / face_metric
            records.append({
                "r": midpoint[0], "z": midpoint[1],
                "gamma": gamma,
                "te": te[plasma_i, plasma_j],
                "ti": ti[plasma_i, plasma_j],
                "ne": ne[plasma_i, plasma_j],
                "br": br[plasma_i, plasma_j],
                "bz": bz[plasma_i, plasma_j],
                "bt": bt[plasma_i, plasma_j],
                "area": abs(face_metric),
                "nr": normal[0], "nz": normal[1],
            })

        # Internal material interfaces.  For an increasing-coordinate face,
        # chi_high-chi_low is +1 when +direction points into material and -1
        # in the opposite orientation.
        for i in range(1, ni):
            for j in range(nj):
                if chi[i - 1, j] == chi[i, j]:
                    continue
                if chi[i, j]:
                    push("psi", i, j, i - 1, j, +1.0)
                else:
                    push("psi", i, j, i, j, -1.0)
        for i in range(ni):
            for j in range(1, nj):
                if chi[i, j - 1] == chi[i, j]:
                    continue
                if chi[i, j]:
                    push("theta", i, j, i, j - 1, +1.0)
                else:
                    push("theta", i, j, i, j, -1.0)

        # Body-fitted physical boundaries use neighbour code -1.  Only add
        # them when the adjacent in-domain cell is plasma; a material cell
        # has already placed the physical interface farther inside the zone.
        if neighbors[0, izone - 1] == -1:
            for j in np.flatnonzero(~chi[0, :]):
                push("psi", 0, int(j), 0, int(j), -1.0)
        if neighbors[1, izone - 1] == -1:
            for j in np.flatnonzero(~chi[-1, :]):
                push("psi", ni, int(j), ni - 1, int(j), +1.0)
        if neighbors[2, izone - 1] == -1:
            for i in np.flatnonzero(~chi[:, 0]):
                push("theta", int(i), 0, int(i), 0, -1.0)
        if neighbors[3, izone - 1] == -1:
            for i in np.flatnonzero(~chi[:, -1]):
                push("theta", int(i), nj, int(i), nj - 1, +1.0)

    if not records:
        raise ValueError("no S3X plasma/material wall faces were found")

    raw_r = np.asarray([x["r"] for x in records], dtype=np.float64)
    raw_z = np.asarray([x["z"] for x in records], dtype=np.float64)
    r, z, s, projection_distance = _project_to_wall(
        raw_r, raw_z, wall_r, wall_z
    )
    order = np.argsort(s, kind="stable")
    gamma_i = np.stack([x["gamma"] for x in records], axis=1)

    def ordered(key):
        return np.asarray([x[key] for x in records], dtype=np.float64)[order]

    return {
        "r": r[order], "z": z[order], "s": s[order],
        "gamma_i": gamma_i[:, order],
        "te": ordered("te"), "ti": ordered("ti"), "ne": ordered("ne"),
        "area": ordered("area"),
        "normal_r": ordered("nr"), "normal_z": ordered("nz"),
        "b_r": ordered("br"), "b_z": ordered("bz"), "b_t": ordered("bt"),
        "projection_distance": projection_distance[order],
    }


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
