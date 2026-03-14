#!/usr/bin/env python3
"""
SOLPS -> OpenEdge converter.

This script mirrors the core post-processing flow used in Jeremy's MATLAB tools:
- Load a SOLPS case (via quixote.SolpsData)
- Build a regular (R,Z) grid
- Interpolate plasma and magnetic fields
- Compute heat-flux density using Jeremy's SOLPS convention (qdep_fht or qdep_fht_no_jV)
- Write OpenEdge-compatible plasma.h5, bfield.h5, and heatflux.h5

Output conventions:
- plasma.h5 uses 1D r/z and 2D fields shaped (nz, nr)
- bfield.h5 uses 1D r/z and 2D fields shaped (nz, nr)
- heatflux.h5 uses 2D grid + fields:
    /grid/R : 2D R mesh (nz, nr)
    /grid/Z : 2D Z mesh (nz, nr)
    /grid/Rc, /grid/Zc : legacy 1D axes
    /fields/q_mag : 2D (nz, nr)
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import sys
import warnings
from pathlib import Path
from typing import Iterable, Tuple

import h5py
import numpy as np
from scipy.interpolate import griddata

# If quixote-master is installed locally, add its path here or install via pip.
# Example: QUIXOTE_MASTER = Path("/path/to/quixote-master")
QUIXOTE_MASTER = Path(os.environ.get("QUIXOTE_PATH", "quixote-master"))
if QUIXOTE_MASTER.exists():
    qpath = str(QUIXOTE_MASTER)
    if qpath not in sys.path:
        sys.path.insert(0, qpath)

try:
    import quixote
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "quixote is required. Install/activate your quixote environment before running this script."
    ) from exc

# If another package named "quixote" is imported from site-packages,
# force-load the intended local quixote-master package.
if not hasattr(quixote, "SolpsData"):
    qinit = QUIXOTE_MASTER / "quixote" / "__init__.py"
    if qinit.exists():
        spec = importlib.util.spec_from_file_location("quixote_local_master", qinit)
        if spec and spec.loader:
            qlocal = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(qlocal)
            if hasattr(qlocal, "SolpsData"):
                quixote = qlocal


def _load_solps_data(run_path: Path, case_name: str):
    """
    Load SOLPS case from quixote across API variants.
    """
    tried = []

    def _try_ctor(ctor, label: str):
        # Try common constructor signatures.
        for args, kwargs in [
            ((str(run_path),), {"name": case_name}),
            ((str(run_path),), {}),
        ]:
            try:
                return ctor(*args, **kwargs)
            except TypeError as exc:
                tried.append(f"{label}{args},{kwargs} -> TypeError: {exc}")
            except Exception as exc:
                tried.append(f"{label}{args},{kwargs} -> {type(exc).__name__}: {exc}")
        return None

    ctor_candidates = []

    # Most common APIs.
    for nm in ("SolpsData", "SOLPSData"):
        if hasattr(quixote, nm):
            ctor_candidates.append((getattr(quixote, nm), f"quixote.{nm}"))

    # Submodule variants.
    for mod_name in ("quixote.solps", "quixote.data", "quixote.solps_data"):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for nm in ("SolpsData", "SOLPSData"):
            if hasattr(mod, nm):
                ctor_candidates.append((getattr(mod, nm), f"{mod_name}.{nm}"))

    # Heuristic fallback: any class with both "solps" and "data" in name.
    for nm in dir(quixote):
        obj = getattr(quixote, nm)
        if inspect.isclass(obj):
            low = nm.lower()
            if "solps" in low and "data" in low:
                ctor_candidates.append((obj, f"quixote.{nm}"))

    seen = set()
    for ctor, label in ctor_candidates:
        if label in seen:
            continue
        seen.add(label)
        shot = _try_ctor(ctor, label)
        if shot is not None:
            return shot

    avail = [a for a in dir(quixote) if "solp" in a.lower() or "data" in a.lower() or "case" in a.lower()]
    raise RuntimeError(
        "Could not construct a SOLPS case from quixote.\\n"
        f"quixote module: {quixote.__file__}\\n"
        f"Relevant quixote attrs: {avail[:60]}\\n"
        "Tried constructors:\\n- " + "\\n- ".join(tried) + "\\n"
        "This usually means a different 'quixote' package is installed in this environment."
    )


def _first_attr(obj, names: Iterable[str]):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    candidates = []
    patterns = [name.lower() for name in names]
    for attr in dir(obj):
        low = attr.lower()
        if any(pat in low for pat in patterns):
            candidates.append(attr)
    if candidates:
        return getattr(obj, candidates[0])
    raise AttributeError(f"None of attributes found: {list(names)}")


def _as_2d_center_coords(shot) -> Tuple[np.ndarray, np.ndarray]:
    # Prefer explicit SOLPS cell-center arrays if present.
    if hasattr(shot, "cx") and hasattr(shot, "cy"):
        rc = np.asarray(shot.cx, dtype=np.float64)
        zc = np.asarray(shot.cy, dtype=np.float64)
        if rc.shape == zc.shape and rc.ndim == 2:
            return rc, zc

    # Fallback used by existing scripts.
    if hasattr(shot, "crx") and hasattr(shot, "cry"):
        crx = np.asarray(shot.crx, dtype=np.float64)
        cry = np.asarray(shot.cry, dtype=np.float64)
        if crx.ndim == 3 and cry.ndim == 3:
            return crx[:, :, -1], cry[:, :, -1]

    raise RuntimeError("Could not infer SOLPS cell-center coordinates from quixote object")


def _regular_grid_from_centers(
    rc: np.ndarray,
    zc: np.ndarray,
    nr: int,
    nz: int,
    rmin: float | None,
    rmax: float | None,
    zmin: float | None,
    zmax: float | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rvals = rc[np.isfinite(rc)]
    zvals = zc[np.isfinite(zc)]
    if rvals.size == 0 or zvals.size == 0:
        raise RuntimeError("No finite SOLPS coordinates found")

    if rmin is None:
        rmin = float(np.min(rvals))
    if rmax is None:
        rmax = float(np.max(rvals))
    if zmin is None:
        zmin = float(np.min(zvals))
    if zmax is None:
        zmax = float(np.max(zvals))

    # Symmetric z-range is often used in MATLAB post-processing for tokamak views.
    zabs = max(abs(zmin), abs(zmax))
    zmin, zmax = -zabs, zabs

    r = np.linspace(rmin, rmax, nr)
    z = np.linspace(zmin, zmax, nz)
    rr, zz = np.meshgrid(r, z)
    return r, z, rr, zz


def _interp_field(
    points_rz: np.ndarray,
    values: np.ndarray,
    grid_points_rz: np.ndarray,
    nz: int,
    nr: int,
) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    lin = griddata(points_rz, v, grid_points_rz, method="linear")
    nn = griddata(points_rz, v, grid_points_rz, method="nearest")
    out = np.where(np.isfinite(lin), lin, nn)
    return out.reshape(nz, nr)


def _potential_field_components(shot, te_shape):
    try:
        pot = np.asarray(_first_attr(shot, ["pot", "phi", "potential"]), dtype=np.float64)
    except AttributeError:
        return None, None, None, None
    if pot.ndim == 3:
        pot = pot[:, :, -1]
    if pot.shape != te_shape:
        raise RuntimeError(f"Expected potential shape {te_shape}, got {pot.shape}")
    grad_pot = np.asarray(shot.gradient(pot), dtype=np.float64)
    efull_z = -grad_pot[:, :, 0]
    efull_r = -grad_pot[:, :, 1]
    efull_t = -grad_pot[:, :, 2] if grad_pot.shape[2] > 2 else np.zeros_like(efull_z)
    return pot, efull_r, efull_t, efull_z


def _extract_main_fields(shot):
    te = np.asarray(_first_attr(shot, ["te"]), dtype=np.float64)
    ne = np.asarray(_first_attr(shot, ["ne"]), dtype=np.float64)
    ti = np.asarray(_first_attr(shot, ["ti"]), dtype=np.float64)
    ni = np.asarray(_first_attr(shot, ["ni"]), dtype=np.float64)
    pot, efull_r, efull_t, efull_z = _potential_field_components(shot, te.shape)
    if pot is None:
        warnings.warn(
            "SOLPS shot lacks pot/phi/potential attributes; falling back to electric_field.",
            stacklevel=2,
        )
        efield = np.asarray(
            _first_attr(shot, ["electric_field", "electric_force"]), dtype=np.float64
        )
        if efield.ndim != 3 or efield.shape[2] < 3:
            raise RuntimeError("Expected electric_field array with shape (nx, ny, >=3)")
        if efield.shape[0] != te.shape[0] or efield.shape[1] != te.shape[1]:
            raise RuntimeError(
                f"electric_field shape {efield.shape[:2]} does not match te shape {te.shape}"
            )
        efull_z = efield[:, :, 0]
        efull_r = efield[:, :, 1]
        efull_t = efield[:, :, 2]
        pot = np.full_like(te, np.nan)

    ua = np.asarray(_first_attr(shot, ["ua"]), dtype=np.float64)
    if ua.ndim == 3:
        # Keep same convention used in existing OpenEdge SOLPS scripts.
        ua = ua[:, :, -1]

    bb = np.asarray(_first_attr(shot, ["bb"]), dtype=np.float64)
    if bb.ndim != 3 or bb.shape[2] < 3:
        raise RuntimeError("Expected magnetic field array bb with shape (nx, ny, 3)")

    # Existing OpenEdge scripts map bb components as:
    #   bb[:,:,0] -> Bt, bb[:,:,1] -> Br, bb[:,:,2] -> Bz
    bt = bb[:, :, 0]
    br = bb[:, :, 1]
    bz = bb[:, :, 2]

    bmag = np.sqrt(br * br + bt * bt + bz * bz)
    safe = np.where(bmag > 0.0, bmag, 1.0)
    upar = ua
    upar_r = ua * br / safe
    upar_t = ua * bt / safe
    upar_z = ua * bz / safe
    upar_r = np.where(bmag > 0.0, upar_r, 0.0)
    upar_t = np.where(bmag > 0.0, upar_t, 0.0)
    upar_z = np.where(bmag > 0.0, upar_z, 0.0)

    # Gradient ordering used by existing scripts:
    #   grad[:,:,0] -> z, grad[:,:,1] -> r, grad[:,:,2] -> toroidal
    grad_te = np.asarray(shot.gradient(te), dtype=np.float64)
    grad_ti = np.asarray(shot.gradient(ti), dtype=np.float64)

    grad_te_z = grad_te[:, :, 0]
    grad_te_r = grad_te[:, :, 1]
    grad_te_t = grad_te[:, :, 2] if grad_te.shape[2] > 2 else np.zeros_like(grad_te_z)

    grad_ti_z = grad_ti[:, :, 0]
    grad_ti_r = grad_ti[:, :, 1]
    grad_ti_t = grad_ti[:, :, 2] if grad_ti.shape[2] > 2 else np.zeros_like(grad_ti_z)

    e_par = (efull_r * br + efull_t * bt + efull_z * bz) / safe
    er = efull_r
    et = efull_t
    ez = efull_z
    e_par = np.where(bmag > 0.0, e_par, 0.0)
    er = np.where(np.isfinite(er), er, 0.0)
    et = np.where(np.isfinite(et), et, 0.0)
    ez = np.where(np.isfinite(ez), ez, 0.0)

    return {
        "temp_e": te,
        "dens_e": ne,
        "temp_i": ti,
        "dens_i": ni,
        "pot": pot,
        "parr_flow": upar,
        "parr_flow_r": upar_r,
        "parr_flow_t": upar_t,
        "parr_flow_z": upar_z,
        "e_par": e_par,
        "er": er,
        "et": et,
        "ez": ez,
        "grad_te_r": grad_te_r,
        "grad_te_t": grad_te_t,
        "grad_te_z": grad_te_z,
        "grad_ti_r": grad_ti_r,
        "grad_ti_t": grad_ti_t,
        "grad_ti_z": grad_ti_z,
        "br": br,
        "bt": bt,
        "bz": bz,
    }


def _compute_heatflux_cell_center(shot, method: str = "jeremy_no_jv") -> np.ndarray:
    """
    Compute heat-flux field from SOLPS face fluxes on a cell-centered grid.

    Methods:
      - jeremy_no_jv: |(fht_x - fhj_x)/sx| (recommended; matches qdep_fht_no_jV)
      - jeremy_total: |fht_x/sx|
      - vector_mag:   sqrt((fht_x/sx)^2 + (fht_y/sy)^2)

    The x-face quantity is centered to cell centers by averaging adjacent x-faces,
    consistent with how we map other face-centered fields onto cell centers.
    """
    fht = np.asarray(_first_attr(shot, ["fht"]), dtype=np.float64)
    sx = np.asarray(_first_attr(shot, ["sx"]), dtype=np.float64)

    if fht.ndim != 3 or fht.shape[2] < 2:
        raise RuntimeError("Expected fht with shape (nx, ny, >=2)")

    qx = np.divide(fht[:, :, 0], sx, out=np.full_like(sx, np.nan), where=sx > 0)

    if method == "jeremy_no_jv":
        fhj = np.asarray(_first_attr(shot, ["fhj"]), dtype=np.float64)
        if fhj.shape[:2] != fht.shape[:2] or fhj.shape[2] < 1:
            raise RuntimeError("Expected fhj with shape compatible with fht")
        qx = np.divide(fht[:, :, 0] - fhj[:, :, 0], sx, out=np.full_like(sx, np.nan), where=sx > 0)

    if method in ("jeremy_no_jv", "jeremy_total"):
        qx_c = 0.5 * (qx + np.roll(qx, -1, axis=0))
        qx_c[-1, :] = np.nan
        return np.abs(qx_c)

    if method == "vector_mag":
        sy = np.asarray(_first_attr(shot, ["sy"]), dtype=np.float64)
        qy = np.divide(fht[:, :, 1], sy, out=np.full_like(sy, np.nan), where=sy > 0)
        qx_c = 0.5 * (qx + np.roll(qx, -1, axis=0))
        qx_c[-1, :] = np.nan
        qy_c = 0.5 * (qy + np.roll(qy, -1, axis=1))
        qy_c[:, -1] = np.nan
        return np.sqrt(qx_c * qx_c + qy_c * qy_c)

    raise ValueError(f"Unknown heatflux method: {method}")
def _read_equilibrium_bfield(equ_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read SOLPS/GITR .equ file and reconstruct (Br,Bt,Bz) from psi.
    Matches the approach in solpsProcessing.py:
      Br = -dpsi/dz / R
      Bz =  dpsi/dr / R
      Bt = (btf*rtf) / R
    """
    if not equ_file.exists():
        raise FileNotFoundError(f"Equilibrium file not found: {equ_file}")

    jm = km = None
    btf = rtf = None
    read_r = read_z = read_psi = False
    r_vals: list[float] = []
    z_vals: list[float] = []
    psi_vals: list[float] = []

    with equ_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if len(tok) >= 3 and tok[0] == "jm" and tok[1] == "=":
                jm = int(tok[2]); continue
            if len(tok) >= 3 and tok[0] == "km" and tok[1] == "=":
                km = int(tok[2]); continue
            if len(tok) >= 3 and tok[0] == "btf" and tok[1] == "=":
                btf = float(tok[2]); continue
            if len(tok) >= 3 and tok[0] == "rtf" and tok[1] == "=":
                rtf = float(tok[2]); continue

            if tok[0] == "r(1:jm);":
                read_r, read_z, read_psi = True, False, False
                continue
            if tok[0] == "z(1:km);":
                read_r, read_z, read_psi = False, True, False
                continue
            if tok[0].startswith("((psi(j,k)-psib,j=1,jm),k=1,km)"):
                read_r, read_z, read_psi = False, False, True
                continue

            if read_r:
                try:
                    r_vals.extend(float(x) for x in tok)
                except ValueError:
                    pass
                continue
            if read_z:
                try:
                    z_vals.extend(float(x) for x in tok)
                except ValueError:
                    pass
                continue
            if read_psi:
                try:
                    psi_vals.extend(float(x) for x in tok)
                except ValueError:
                    pass
                continue

    if jm is None or km is None or btf is None or rtf is None:
        raise RuntimeError(f"Missing required headers in equilibrium file: {equ_file}")
    if len(r_vals) < jm or len(z_vals) < km or len(psi_vals) < jm * km:
        raise RuntimeError(
            f"Incomplete equilibrium arrays in {equ_file}: "
            f"len(r)={len(r_vals)} (jm={jm}), len(z)={len(z_vals)} (km={km}), len(psi)={len(psi_vals)}"
        )

    r = np.asarray(r_vals[:jm], dtype=np.float64)
    z = np.asarray(z_vals[:km], dtype=np.float64)
    psi = np.asarray(psi_vals[: jm * km], dtype=np.float64).reshape((km, jm))

    grad_z, grad_r = np.gradient(psi, z[1] - z[0], r[1] - r[0])
    rr = np.meshgrid(r, z)[0]
    safe_r = np.where(np.abs(rr) > 1.0e-12, rr, 1.0e-12)
    br = -grad_z / safe_r
    bz = grad_r / safe_r
    bt = (btf * rtf) / safe_r
    return r, z, br, bt, bz


def _read_geqdsk_bfield(gfile: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (Br,Bt,Bz) from GEQDSK using the same convention supplied by user code:
      Br =  dpsi/dZ / R
      Bz = -dpsi/dR / R
      Bt = bcentr * rcentr / R
    """
    if not gfile.exists():
        raise FileNotFoundError(f"GEQDSK file not found: {gfile}")
    try:
        from freeqdsk import geqdsk
    except Exception as exc:
        raise RuntimeError(
            "GEQDSK support requested but Python package 'freeqdsk' is not available."
        ) from exc

    with gfile.open("r", encoding="utf-8", errors="ignore") as f:
        data = geqdsk.read(f)

    nx = int(data["nx"])
    ny = int(data["ny"])
    rdim = float(data["rdim"])
    rmin = float(data["rleft"])
    rmax = rmin + rdim
    zmin = float(data["zmid"]) - float(data["zdim"]) / 2.0
    zmax = float(data["zmid"]) + float(data["zdim"]) / 2.0

    # Use linspace to avoid endpoint drift and keep exact dimensions.
    rs = np.linspace(rmin, rmax, nx)
    zs = np.linspace(zmin, zmax, ny)
    r2d, z2d = np.meshgrid(rs, zs)

    psi_arr = data.get("psi", data.get("psirz", None))
    if psi_arr is None:
        raise RuntimeError("GEQDSK does not contain psi/psirz array")
    flux2d = np.array(psi_arr, dtype=np.float64)
    if flux2d.shape != (ny, nx):
        flux2d = flux2d.reshape((ny, nx))

    dpsidR = np.zeros_like(flux2d)
    dpsidZ = np.zeros_like(flux2d)

    # dpsi/dR (central diff in interior, one-sided at boundaries)
    dpsidR[:, 1:-1] = (flux2d[:, 2:] - flux2d[:, :-2]) / (r2d[:, 2:] - r2d[:, :-2])
    dpsidR[:, 0] = (flux2d[:, 1] - flux2d[:, 0]) / (r2d[:, 1] - r2d[:, 0])
    dpsidR[:, -1] = (flux2d[:, -1] - flux2d[:, -2]) / (r2d[:, -1] - r2d[:, -2])

    # dpsi/dZ
    dpsidZ[1:-1, :] = (flux2d[2:, :] - flux2d[:-2, :]) / (z2d[2:, :] - z2d[:-2, :])
    dpsidZ[0, :] = (flux2d[1, :] - flux2d[0, :]) / (z2d[1, :] - z2d[0, :])
    dpsidZ[-1, :] = (flux2d[-1, :] - flux2d[-2, :]) / (z2d[-1, :] - z2d[-2, :])

    safe_r = np.where(np.abs(r2d) > 1.0e-12, r2d, 1.0e-12)
    br = dpsidZ / safe_r
    bz = -dpsidR / safe_r
    bt = (float(data["bcentr"]) * float(data["rcentr"])) / safe_r
    return rs, zs, br, bt, bz


def _write_sparta_wall_from_mesh_extra(mesh_extra: Path, wall_out: Path, tol: float = 1.0e-8) -> None:
    """
    Build SPARTA wall surface file from SOLPS mesh.extra segments.
    mesh.extra is expected to have at least 4 columns: R1 Z1 R2 Z2.
    """
    if not mesh_extra.exists():
        raise FileNotFoundError(f"mesh.extra not found: {mesh_extra}")

    d = np.loadtxt(mesh_extra)
    d = np.atleast_2d(d)
    if d.shape[1] < 4:
        raise RuntimeError(f"mesh.extra requires >=4 columns, got shape {d.shape}")

    seg_a = d[:, 0:2].astype(np.float64)
    seg_b = d[:, 2:4].astype(np.float64)
    nseg = seg_a.shape[0]

    used = np.zeros(nseg, dtype=bool)
    pts = [seg_a[0], seg_b[0]]
    used[0] = True

    def _dist(p, q):
        return float(np.hypot(p[0] - q[0], p[1] - q[1]))

    for _ in range(nseg - 1):
        cur = pts[-1]
        best_i = -1
        best_side = 0
        best_d = 1.0e99
        for i in range(nseg):
            if used[i]:
                continue
            d1 = _dist(cur, seg_a[i])
            d2 = _dist(cur, seg_b[i])
            if d1 < best_d:
                best_d = d1
                best_i = i
                best_side = 1
            if d2 < best_d:
                best_d = d2
                best_i = i
                best_side = 2

        if best_i < 0 or best_d > max(tol, 1.0e-5):
            break

        used[best_i] = True
        nxt = seg_b[best_i] if best_side == 1 else seg_a[best_i]
        pts.append(nxt)

    # Fallback if chaining failed: use first endpoint list directly
    if len(pts) < 3:
        pts = [p for p in seg_a]
        pts.append(seg_b[-1])

    # Remove consecutive duplicates
    clean = [pts[0]]
    for p in pts[1:]:
        if np.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > tol:
            clean.append(p)

    if np.hypot(clean[0][0] - clean[-1][0], clean[0][1] - clean[-1][1]) > tol:
        clean.append(clean[0])

    # Write SPARTA 2D surface format
    wall_out.parent.mkdir(parents=True, exist_ok=True)
    r = np.array([p[0] for p in clean[:-1]], dtype=np.float64)
    z = np.array([p[1] for p in clean[:-1]], dtype=np.float64)
    n = int(r.size)
    if n < 3:
        raise RuntimeError("mesh.extra did not produce a valid closed wall polygon")

    with wall_out.open("w", encoding="utf-8") as f:
        f.write("surface geometry\n\n")
        f.write(f"{n} points\n")
        f.write(f"{n} lines\n\n")
        f.write("Points\n\n")
        for i in range(n):
            f.write(f"{i+1} {r[i]:.12g} {z[i]:.12g}\n")
        f.write("\nLines\n\n")
        for i in range(n):
            f.write(f"{i+1} {i+1} {(i+1) % n + 1}\n")


def _save_plots(r: np.ndarray, z: np.ndarray, interp: dict, plot_prefix: Path) -> None:
    import matplotlib.pyplot as plt

    extent = [float(r.min()), float(r.max()), float(z.min()), float(z.max())]

    plasma_panels = [
        ("dens_e", "dens_e [m^-3]", "inferno", True),
        ("temp_e", "temp_e [eV]", "magma", False),
        ("dens_i", "dens_i [m^-3]", "inferno", True),
        ("temp_i", "temp_i [eV]", "magma", False),
        ("parr_flow", "parr_flow [m/s]", "RdBu_r", False),
        ("grad_te_r", "grad_te_r [eV/m]", "viridis", False),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, (k, title, cmap, log10) in zip(axes.flat, plasma_panels):
        arr = np.array(interp[k], copy=True)
        if log10:
            arr = np.where(arr > 0.0, np.log10(arr), np.nan)
        im = ax.imshow(arr, origin="lower", extent=extent, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        fig.colorbar(im, ax=ax, shrink=0.9)
    plasma_png = Path(f"{plot_prefix}_plasma.png")
    fig.savefig(plasma_png, dpi=180)
    plt.close(fig)

    bfield_panels = [
        ("br", "Br [T]"),
        ("bt", "Bt [T]"),
        ("bz", "Bz [T]"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, (k, title) in zip(axes.flat, bfield_panels):
        arr = np.array(interp[k], copy=True)
        vmax = float(np.nanmax(np.abs(arr))) if np.any(np.isfinite(arr)) else 1.0
        if not np.isfinite(vmax) or vmax == 0.0:
            vmax = 1.0
        im = ax.imshow(
            arr, origin="lower", extent=extent, aspect="auto",
            cmap="RdBu_r", vmin=-vmax, vmax=vmax
        )
        ax.set_title(title)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        fig.colorbar(im, ax=ax, shrink=0.9)
    bfield_png = Path(f"{plot_prefix}_bfield.png")
    fig.savefig(bfield_png, dpi=180)
    plt.close(fig)

    print(f"Wrote plot   : {plasma_png}")
    print(f"Wrote plot   : {bfield_png}")


def convert_solps_to_openedge(
    run_path: Path,
    plasma_out: Path,
    bfield_out: Path,
    heatflux_out: Path,
    nr: int = 300,
    nz: int = 300,
    rmin: float | None = None,
    rmax: float | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    case_name: str = "d3d",
    gfile: Path | None = None,
    equ_file: Path | None = None,
    plot: bool = False,
    plot_prefix: Path | None = None,
    heatflux_method: str = "jeremy_no_jv",
    wall_out: Path | None = None,
    mesh_extra: Path | None = None,
) -> None:
    shot = _load_solps_data(run_path, case_name)

    rc, zc = _as_2d_center_coords(shot)
    r, z, rr, zz = _regular_grid_from_centers(rc, zc, nr, nz, rmin, rmax, zmin, zmax)

    src_points = np.column_stack((rc.reshape(-1), zc.reshape(-1)))
    tgt_points = np.column_stack((rr.reshape(-1), zz.reshape(-1)))

    fields = _extract_main_fields(shot)

    interp = {}
    for key, arr in fields.items():
        interp[key] = _interp_field(src_points, np.asarray(arr), tgt_points, nz, nr)

    # B-field must come from external equilibrium file, not quixote bb.
    if gfile is None and equ_file is None:
        raise RuntimeError(
            "B-field source is required. Provide --gfile (preferred) or --equ-file. "
            "Quixote bb is intentionally not used for bfield.h5."
        )

    if gfile is not None:
        rg, zg, br_g, bt_g, bz_g = _read_geqdsk_bfield(gfile)
        rr_g, zz_g = np.meshgrid(rg, zg)
        b_points = np.column_stack((rr_g.reshape(-1), zz_g.reshape(-1)))
        binterp = {
            "br": _interp_field(b_points, br_g, tgt_points, nz, nr),
            "bt": _interp_field(b_points, bt_g, tgt_points, nz, nr),
            "bz": _interp_field(b_points, bz_g, tgt_points, nz, nr),
        }
    else:
        req, zeq, br_eq, bt_eq, bz_eq = _read_equilibrium_bfield(equ_file)
        rr_eq, zz_eq = np.meshgrid(req, zeq)
        b_points = np.column_stack((rr_eq.reshape(-1), zz_eq.reshape(-1)))
        binterp = {
            "br": _interp_field(b_points, br_eq, tgt_points, nz, nr),
            "bt": _interp_field(b_points, bt_eq, tgt_points, nz, nr),
            "bz": _interp_field(b_points, bz_eq, tgt_points, nz, nr),
        }

    # Ensure any downstream plotting uses equilibrium-based B only.
    interp["br"] = binterp["br"]
    interp["bt"] = binterp["bt"]
    interp["bz"] = binterp["bz"]

    # If gradients are not finite in some areas, recompute from interpolated temperatures.
    dz = z[1] - z[0]
    dr = r[1] - r[0]
    gte_z, gte_r = np.gradient(interp["temp_e"], dz, dr)
    gti_z, gti_r = np.gradient(interp["temp_i"], dz, dr)
    interp["grad_te_r"] = np.where(np.isfinite(interp["grad_te_r"]), interp["grad_te_r"], gte_r)
    interp["grad_te_z"] = np.where(np.isfinite(interp["grad_te_z"]), interp["grad_te_z"], gte_z)
    interp["grad_ti_r"] = np.where(np.isfinite(interp["grad_ti_r"]), interp["grad_ti_r"], gti_r)
    interp["grad_ti_z"] = np.where(np.isfinite(interp["grad_ti_z"]), interp["grad_ti_z"], gti_z)
    interp["grad_te_t"] = np.nan_to_num(interp["grad_te_t"], nan=0.0)
    interp["grad_ti_t"] = np.nan_to_num(interp["grad_ti_t"], nan=0.0)

    qmag_src = _compute_heatflux_cell_center(shot, method=heatflux_method)
    qmag_grid = _interp_field(src_points, qmag_src, tgt_points, nz, nr)
    qmag_grid = np.nan_to_num(qmag_grid, nan=0.0, posinf=0.0, neginf=0.0)

    # Write plasma.h5
    with h5py.File(plasma_out, "w") as f:
        f.create_dataset("r", data=r)
        f.create_dataset("z", data=z)
        for key in [
            "dens_e",
            "dens_i",
            "temp_e",
            "temp_i",
            "pot",
            "parr_flow",
            "parr_flow_r",
            "parr_flow_t",
            "parr_flow_z",
            "e_par",
            "er",
            "et",
            "ez",
            "grad_te_r",
            "grad_te_t",
            "grad_te_z",
            "grad_ti_r",
            "grad_ti_t",
            "grad_ti_z",
        ]:
            f.create_dataset(key, data=np.nan_to_num(interp[key], nan=0.0))

        # Optional compatibility mirrors used by some legacy analysis tooling.
        f.create_dataset("n_e/dens", data=np.nan_to_num(interp["dens_e"], nan=0.0))
        f.create_dataset("n_e/temp", data=np.nan_to_num(interp["temp_e"], nan=0.0))
        f.create_dataset("n_i/dens", data=np.nan_to_num(interp["dens_i"], nan=0.0))
        f.create_dataset("n_i/temp", data=np.nan_to_num(interp["temp_i"], nan=0.0))
        f.create_dataset("n_i/parr_flow", data=np.nan_to_num(interp["parr_flow"], nan=0.0))

    # Write bfield.h5
    with h5py.File(bfield_out, "w") as f:
        f.create_dataset("r", data=r)
        f.create_dataset("z", data=z)
        f.create_dataset("br", data=np.nan_to_num(binterp["br"], nan=0.0))
        f.create_dataset("bt", data=np.nan_to_num(binterp["bt"], nan=0.0))
        f.create_dataset("bz", data=np.nan_to_num(binterp["bz"], nan=0.0))

    # Write heatflux.h5 with both 2D grid (new) and 1D axes (legacy compatibility).
    with h5py.File(heatflux_out, "w") as f:
        ggrid = f.create_group("grid")
        gfld = f.create_group("fields")
        rr2d, zz2d = np.meshgrid(r, z)
        ggrid.create_dataset("R", data=rr2d)
        ggrid.create_dataset("Z", data=zz2d)
        ggrid.create_dataset("Rc", data=r)
        ggrid.create_dataset("Zc", data=z)
        gfld.create_dataset("q_mag", data=qmag_grid)
        # Root-level mirrors for simple readers
        f.create_dataset("R", data=rr2d)
        f.create_dataset("Z", data=zz2d)
        f.create_dataset("q_mag", data=qmag_grid)

    if wall_out is not None:
        mesh_path = mesh_extra if mesh_extra is not None else (run_path / "mesh.extra")
        _write_sparta_wall_from_mesh_extra(mesh_path, wall_out)
        print(f"Wrote wall   : {wall_out} (from {mesh_path})")

    print(f"Wrote plasma : {plasma_out}")
    print(f"Wrote bfield : {bfield_out}")
    print(f"Wrote heatflux: {heatflux_out}")
    print(
        f"Grid: nr={nr}, nz={nz}, R=[{r[0]:.4f},{r[-1]:.4f}], Z=[{z[0]:.4f},{z[-1]:.4f}]"
    )

    if plot:
        prefix = plot_prefix if plot_prefix is not None else Path("solps2openedge")
        _save_plots(r, z, interp, prefix)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert SOLPS case to OpenEdge plasma/bfield/heatflux HDF5")
    p.add_argument("run_path", type=Path, help="SOLPS run directory (contains b2fstate/b2fplasmf etc.)")
    p.add_argument("--plasma-out", type=Path, default=Path("plasma.h5"))
    p.add_argument("--bfield-out", type=Path, default=Path("bfield.h5"))
    p.add_argument("--heatflux-out", type=Path, default=Path("heatflux.h5"))
    p.add_argument("--nr", type=int, default=300)
    p.add_argument("--nz", type=int, default=300)
    p.add_argument("--rmin", type=float, default=None)
    p.add_argument("--rmax", type=float, default=None)
    p.add_argument("--zmin", type=float, default=None)
    p.add_argument("--zmax", type=float, default=None)
    p.add_argument("--case-name", type=str, default="d3d", help="quixote geometry label")
    p.add_argument("--gfile", type=Path, default=None, help="Path to GEQDSK file for Br/Bt/Bz reconstruction")
    p.add_argument("--equ-file", type=Path, default=None, help="Path to *.equ file used to reconstruct Br/Bt/Bz from psi")
    p.add_argument("--plot", action="store_true", help="Write quick-look PNGs for plasma and bfield")
    p.add_argument("--heatflux-method", type=str, default="jeremy_no_jv",
                   choices=["jeremy_no_jv", "jeremy_total", "vector_mag"],
                   help="Heatflux definition: Jeremy no-jV (default), Jeremy total, or vector magnitude")
    p.add_argument("--plot-prefix", type=Path, default=Path("solps2openedge"), help="Prefix for plot PNGs")
    p.add_argument("--wall-out", type=Path, default=Path("wall.txt"), help="Output SPARTA wall file built from mesh.extra")
    p.add_argument("--mesh-extra", type=Path, default=None, help="Optional path to mesh.extra (defaults to <run_path>/mesh.extra)")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    convert_solps_to_openedge(
        run_path=args.run_path,
        plasma_out=args.plasma_out,
        bfield_out=args.bfield_out,
        heatflux_out=args.heatflux_out,
        nr=args.nr,
        nz=args.nz,
        rmin=args.rmin,
        rmax=args.rmax,
        zmin=args.zmin,
        zmax=args.zmax,
        case_name=args.case_name,
        gfile=args.gfile,
        equ_file=args.equ_file,
        plot=args.plot,
        plot_prefix=args.plot_prefix,
        heatflux_method=args.heatflux_method,
        wall_out=args.wall_out,
        mesh_extra=args.mesh_extra,
    )


if __name__ == "__main__":
    main()
