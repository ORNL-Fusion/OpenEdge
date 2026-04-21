import os
import re
import h5py
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from matplotlib.path import Path
import matplotlib.pyplot as plt
from utilities import surface


QE = 1.602176634e-19
AMU = 1.66053906660e-27
DEFAULT_SHEATH_MASS_AMU = 2.01410177811


def _find_species_indices(data_h5):
    """Return sorted species indices from /triangles/spec* groups."""
    tri = data_h5['/triangles']
    inds = []
    for key in tri.keys():
        m = re.fullmatch(r'spec(\d+)', key)
        if m:
            inds.append(int(m.group(1)))
    return sorted(inds)


def _safe_read_field(data_h5, spec_idx, field_name, scale=1.0):
    """Read /triangles/spec{idx}/{field_name}; return None if missing."""
    path = f'/triangles/spec{spec_idx}/{field_name}'
    if path not in data_h5:
        return None
    return data_h5[path][...].flatten() * scale


def _interpolate_field(centroids, grid_points, nZ, nR, field_data):
    lin = griddata(centroids, field_data, grid_points, method='linear')
    nst = griddata(centroids, field_data, grid_points, method='nearest')
    out = np.where(np.isnan(lin), nst, lin)
    return out.reshape(nZ, nR)


def _mask_outside_wall(arr2d, mask_outside, nZ, nR, fill=0.0):
    arr2d[mask_outside.reshape(nZ, nR)] = fill
    return arr2d


def _read_scalar(h5obj, key, default=None):
    if key not in h5obj:
        return default
    val = h5obj[key][...]
    try:
        return float(np.asarray(val).reshape(-1)[0])
    except Exception:
        return default


def _wall_from_mesh(mesh_file, ref_file):
    """
    Build wall geometry directly from SOLEDGE mesh wall arrays and rescale
    with mesh/reference (R0,a0), following legacy getWallInfo logic.
    """
    with h5py.File(ref_file, "r") as ref, h5py.File(mesh_file, "r") as mesh:
        if "/wall/R" not in mesh or "/wall/Z" not in mesh:
            raise RuntimeError("Mesh file does not contain /wall/R and /wall/Z")

        rwall = np.asarray(mesh["/wall/R"][...], dtype=np.float64).reshape(-1)
        zwall = np.asarray(mesh["/wall/Z"][...], dtype=np.float64).reshape(-1)
        if rwall.size == 0 or zwall.size == 0 or rwall.size != zwall.size:
            raise RuntimeError("Invalid wall arrays in mesh file")

        r0_init = _read_scalar(mesh, "/R0", 0.0)
        a0_init = _read_scalar(mesh, "/a0", 1.0)
        r0_ref = _read_scalar(ref, "/R0", r0_init)
        a0_ref = _read_scalar(ref, "/a0", a0_init)

        a0_init = max(float(a0_init), 1.0e-99)
        rwall = (rwall - r0_init) / a0_init * a0_ref + r0_ref
        zwall = zwall / a0_init * a0_ref

        # Ensure closed polygon for Path() masking.
        if not (np.isclose(rwall[0], rwall[-1]) and np.isclose(zwall[0], zwall[-1])):
            rwall = np.concatenate([rwall, rwall[:1]])
            zwall = np.concatenate([zwall, zwall[:1]])

        return rwall, zwall


def _wall_and_config_from_mesh(mesh_file, ref_file):
    """
    Read wall + config (r,z,psi,psisep) from a SOLEDGE mesh/config file and
    rescale to physical coordinates using the same logic as getWallInfo().
    """
    with h5py.File(ref_file, "r") as ref, h5py.File(mesh_file, "r") as mesh:
        wall_is_absolute_coords = False
        if "/wall/R" in mesh and "/wall/Z" in mesh:
            rwall = np.asarray(mesh["/wall/R"][...], dtype=np.float64).reshape(-1)
            zwall = np.asarray(mesh["/wall/Z"][...], dtype=np.float64).reshape(-1)
        elif "walls/wall1/R" in mesh and "walls/wall1/Z" in mesh:
            # SOLEDGE3X mesh_raptorX layout
            rwall = np.asarray(mesh["walls/wall1/R"][...], dtype=np.float64).reshape(-1)
            zwall = np.asarray(mesh["walls/wall1/Z"][...], dtype=np.float64).reshape(-1)
            # walls/wall1 coordinates are typically physical machine coords in cm.
            # Convert cm->m only; do NOT apply R0/a0 remap on this branch.
            if np.nanmax(np.abs(rwall)) > 20.0 or np.nanmax(np.abs(zwall)) > 20.0:
                rwall = rwall / 100.0
                zwall = zwall / 100.0
            wall_is_absolute_coords = True
        else:
            raise RuntimeError("Mesh file does not contain supported wall datasets (/wall/R,Z or walls/wall1/R,Z)")
        if rwall.size == 0 or zwall.size == 0 or rwall.size != zwall.size:
            raise RuntimeError("Invalid wall arrays in mesh file")

        have_cfg = all(k in mesh for k in ["/config/r", "/config/z", "/config/psi"])
        r2d = z2d = psi2d = psisep = psicore = None
        if have_cfg:
            r2d = np.asarray(mesh["/config/r"][...], dtype=np.float64)
            z2d = np.asarray(mesh["/config/z"][...], dtype=np.float64)
            psi2d = np.asarray(mesh["/config/psi"][...], dtype=np.float64)
            if "/config/psisep1" in mesh:
                psisep = float(np.asarray(mesh["/config/psisep1"][...]).reshape(-1)[0])
            if "/config/psicore" in mesh:
                psicore = float(np.asarray(mesh["/config/psicore"][...]).reshape(-1)[0])

        if not wall_is_absolute_coords:
            r0_init = _read_scalar(mesh, "/R0", 0.0)
            a0_init = _read_scalar(mesh, "/a0", 1.0)
            r0_ref = _read_scalar(ref, "/R0", r0_init)
            a0_ref = _read_scalar(ref, "/a0", a0_init)

            a0_init = max(float(a0_init), 1.0e-99)
            rwall = (rwall - r0_init) / a0_init * a0_ref + r0_ref
            zwall = zwall / a0_init * a0_ref
            if r2d is not None:
                r2d = (r2d - r0_init) / a0_init * a0_ref + r0_ref
                z2d = z2d / a0_init * a0_ref

        # Ensure closed wall polygon.
        if not (np.isclose(rwall[0], rwall[-1]) and np.isclose(zwall[0], zwall[-1])):
            rwall = np.concatenate([rwall, rwall[:1]])
            zwall = np.concatenate([zwall, zwall[:1]])

        return rwall, zwall, r2d, z2d, psi2d, psisep, psicore


def _extract_core_contour(r2d, z2d, psi2d, psi_level):
    """Extract a closed core contour polyline (R,Z) from psi=const."""
    if r2d is None or z2d is None or psi2d is None or psi_level is None:
        return None, None

    fig, ax = plt.subplots(1, 1)
    try:
        cs = ax.contour(r2d, z2d, psi2d, levels=[float(psi_level)])
        if len(cs.allsegs) == 0 or len(cs.allsegs[0]) == 0:
            return None, None
        # Pick longest contour segment.
        segs = cs.allsegs[0]
        seg = max(segs, key=lambda s: s.shape[0])
        if seg.shape[0] < 3:
            return None, None
        rc = np.asarray(seg[:, 0], dtype=np.float64)
        zc = np.asarray(seg[:, 1], dtype=np.float64)
        if not (np.isclose(rc[0], rc[-1]) and np.isclose(zc[0], zc[-1])):
            rc = np.concatenate([rc, rc[:1]])
            zc = np.concatenate([zc, zc[:1]])
        return rc, zc
    finally:
        plt.close(fig)


def _write_sparta_surface_polyline(path, rvals, zvals, title="surface geometry"):
    """Write a closed polyline into SPARTA 2D surface text format.

    SOLEDGE3X is an axisymmetric edge code, so we always write the file
    in SPARTA's true axisymmetric layout: column 1 = Z (axial), column 2
    = R (radial). Pairs with `boundary o ao p`, `create_box ... 0 R_max
    ... `, where SPARTA computes correct cylindrical cell volumes and
    2*pi*R*L surface areas internally.
    """
    if rvals is None or zvals is None:
        return
    r = np.asarray(rvals, dtype=np.float64).reshape(-1)
    z = np.asarray(zvals, dtype=np.float64).reshape(-1)
    if r.size < 3 or r.size != z.size:
        return

    # Remove final duplicate point for SPARTA points/lines indexing.
    if np.isclose(r[0], r[-1]) and np.isclose(z[0], z[-1]):
        r = r[:-1]
        z = z[:-1]

    # Remove consecutive duplicate vertices that would otherwise create
    # zero-length segments and fail SPARTA's watertight checks.
    keep = np.ones(r.size, dtype=bool)
    keep[1:] = ~(np.isclose(r[1:], r[:-1]) & np.isclose(z[1:], z[:-1]))
    r = r[keep]
    z = z[keep]

    # Re-check closure after deduplication and drop the repeated endpoint if
    # the cleaned polyline still ends where it begins.
    if r.size >= 2 and np.isclose(r[0], r[-1]) and np.isclose(z[0], z[-1]):
        r = r[:-1]
        z = z[:-1]

    n = int(r.size)
    if n < 3:
        return

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n\n")
        f.write(f"{n} points\n")
        f.write(f"{n} lines\n\n")
        f.write("Points\n\n")
        for i in range(n):
            f.write(f"{i+1} {z[i]:.12g} {r[i]:.12g}\n")
        f.write("\nLines\n\n")
        for i in range(n):
            j = i + 1
            k = (i + 1) % n
            f.write(f"{j} {j} {k+1}\n")


def _build_ion_metadata(ion_inds, ion_metadata=None):
    """
    Build per-ion metadata arrays aligned with ion_inds.

    ion_metadata format:
      {spec_index: {"name": str, "mass_amu": float, "z": int}}
    """
    # WEST/SOLEDGE default convention:
    # spec1=D+, spec2..spec9=O+..O8+
    defaults = {
        1: {"name": "D+", "mass_amu": 2.01410177811, "z": 1},
        2: {"name": "O+", "mass_amu": 15.999, "z": 1},
        3: {"name": "O2+", "mass_amu": 15.999, "z": 2},
        4: {"name": "O3+", "mass_amu": 15.999, "z": 3},
        5: {"name": "O4+", "mass_amu": 15.999, "z": 4},
        6: {"name": "O5+", "mass_amu": 15.999, "z": 5},
        7: {"name": "O6+", "mass_amu": 15.999, "z": 6},
        8: {"name": "O7+", "mass_amu": 15.999, "z": 7},
        9: {"name": "O8+", "mass_amu": 15.999, "z": 8},
    }
    if ion_metadata:
        defaults.update(ion_metadata)

    names = []
    masses_amu = np.zeros(len(ion_inds), dtype=np.float64)
    charge_state_z = np.zeros(len(ion_inds), dtype=np.int32)

    for k, sidx in enumerate(ion_inds):
        meta = defaults.get(sidx, None)
        if meta is None:
            names.append(f"spec{sidx}")
            masses_amu[k] = np.nan
            charge_state_z[k] = -1
            continue
        zval = int(meta["z"])
        names.append(str(meta["name"]))
        masses_amu[k] = float(meta["mass_amu"])
        charge_state_z[k] = zval
    return names, masses_amu, charge_state_z


def _segment_normals(rvals, zvals):
    """Return segment midpoints and unit normals for a closed wall polyline."""
    r = np.asarray(rvals, dtype=np.float64).reshape(-1)
    z = np.asarray(zvals, dtype=np.float64).reshape(-1)
    if r.size < 2:
        raise ValueError("Wall polyline needs at least 2 points")
    if not (np.isclose(r[0], r[-1]) and np.isclose(z[0], z[-1])):
        r = np.concatenate([r, r[:1]])
        z = np.concatenate([z, z[:1]])
    dr = np.diff(r)
    dz = np.diff(z)
    seglen = np.hypot(dr, dz)
    good = seglen > 1.0e-14
    if not np.any(good):
        raise ValueError("Wall polyline has no non-zero segments")
    mid_r = 0.5 * (r[:-1] + r[1:])
    mid_z = 0.5 * (z[:-1] + z[1:])
    nr = dz.copy()
    nz = -dr.copy()
    nr[good] /= seglen[good]
    nz[good] /= seglen[good]
    return mid_r[good], mid_z[good], nr[good], nz[good]


def _nearest_wall_normals(sample_r, sample_z, wall_r, wall_z):
    """Assign each sample point the normal of the nearest wall segment midpoint."""
    mid_r, mid_z, nr_seg, nz_seg = _segment_normals(wall_r, wall_z)
    sr = np.asarray(sample_r, dtype=np.float64).reshape(-1)
    sz = np.asarray(sample_z, dtype=np.float64).reshape(-1)
    best_d2 = np.full(sr.size, np.inf, dtype=np.float64)
    best_idx = np.zeros(sr.size, dtype=np.int32)
    for i in range(mid_r.size):
        d2 = (sr - mid_r[i]) ** 2 + (sz - mid_z[i]) ** 2
        mask = d2 < best_d2
        best_d2[mask] = d2[mask]
        best_idx[mask] = i
    return nr_seg[best_idx], nz_seg[best_idx]


def _bfield_incidence_sinalpha(br, bz, bt, nr, nz):
    bmag = np.sqrt(br * br + bz * bz + bt * bt)
    bp_dot_n = br * nr + bz * nz
    out = np.zeros_like(bmag, dtype=np.float64)
    mask = bmag > 1.0e-30
    out[mask] = np.abs(bp_dot_n[mask]) / bmag[mask]
    return np.clip(out, 0.0, 1.0)


def _compute_bohm_flux_maps(
    r,
    z,
    wall_r,
    wall_z,
    dens_i_all,
    temp_i_all,
    temp_e,
    br,
    bt,
    bz,
    sheath_mass_amu=DEFAULT_SHEATH_MASS_AMU,
):
    rr, zz = np.meshgrid(r, z)
    nr_wall, nz_wall = _nearest_wall_normals(rr.ravel(), zz.ravel(), wall_r, wall_z)
    nr_wall = nr_wall.reshape(rr.shape)
    nz_wall = nz_wall.reshape(rr.shape)
    sin_alpha = _bfield_incidence_sinalpha(br, bz, bt, nr_wall, nz_wall)
    cs_arg = (np.maximum(temp_e, 0.0)[None, :, :] + np.maximum(temp_i_all, 0.0))
    cs_arg = cs_arg * QE / (2.0 * sheath_mass_amu * AMU)
    cs = np.sqrt(np.maximum(cs_arg, 0.0))
    gamma = np.maximum(dens_i_all, 0.0) * cs * sin_alpha[None, :, :]
    return gamma, sin_alpha


def _compute_bohm_flux_on_wall(
    vv_r,
    vv_z,
    ion_names,
    r,
    z,
    dens_i_all,
    temp_i_all,
    temp_e,
    br,
    bt,
    bz,
    sheath_mass_amu=DEFAULT_SHEATH_MASS_AMU,
):
    sample_pts = np.column_stack((vv_z, vv_r))
    nr_wall, nz_wall = _nearest_wall_normals(vv_r, vv_z, vv_r, vv_z)
    interp_te = RegularGridInterpolator((z, r), temp_e, method='linear', bounds_error=False, fill_value=0.0)
    interp_br = RegularGridInterpolator((z, r), br, method='linear', bounds_error=False, fill_value=0.0)
    interp_bt = RegularGridInterpolator((z, r), bt, method='linear', bounds_error=False, fill_value=0.0)
    interp_bz = RegularGridInterpolator((z, r), bz, method='linear', bounds_error=False, fill_value=0.0)
    te_w = np.maximum(interp_te(sample_pts), 0.0)
    br_w = interp_br(sample_pts)
    bt_w = interp_bt(sample_pts)
    bz_w = interp_bz(sample_pts)
    sin_alpha = _bfield_incidence_sinalpha(br_w, bz_w, bt_w, nr_wall, nz_wall)
    gamma_all = np.zeros((dens_i_all.shape[0], vv_r.size), dtype=np.float64)
    for k in range(dens_i_all.shape[0]):
        interp_n = RegularGridInterpolator((z, r), dens_i_all[k], method='linear', bounds_error=False, fill_value=0.0)
        interp_ti = RegularGridInterpolator((z, r), temp_i_all[k], method='linear', bounds_error=False, fill_value=0.0)
        n_w = np.maximum(interp_n(sample_pts), 0.0)
        ti_w = np.maximum(interp_ti(sample_pts), 0.0)
        cs = np.sqrt(np.maximum((te_w + ti_w) * QE / (2.0 * sheath_mass_amu * AMU), 0.0))
        gamma_all[k] = n_w * cs * sin_alpha

    if gamma_all.shape[0] > 1:
        other = np.sum(gamma_all[1:], axis=0)
        dom_idx = 1 + np.argmax(gamma_all[1:], axis=0)
        peak_idx = np.argsort(other)[-5:][::-1]
        print("Top non-D Bohm-flux wall peaks:")
        for idx in peak_idx:
            name = ion_names[dom_idx[idx]] if dom_idx[idx] < len(ion_names) else f"ion{dom_idx[idx]}"
            print(
                f"  s={idx:4d} R={vv_r[idx]:.4f} Z={vv_z[idx]:.4f} "
                f"Gamma_Osum={other[idx]:.3e} dominant={name} "
                f"Gamma_dom={gamma_all[dom_idx[idx], idx]:.3e} sin(alpha_B)={sin_alpha[idx]:.3e}"
            )
    return gamma_all, sin_alpha


def _plot_debug_fields(
    out_png,
    r,
    z,
    Rwall,
    Zwall,
    dens_e,
    temp_e,
    dens_i,
    temp_i,
    parr_flow,
    parr_flow_r,
    parr_flow_z,
    grad_te_r,
    grad_te_z,
    br,
    bt,
    bz,
):
    extent = [r.min(), r.max(), z.min(), z.max()]
    log_ne = np.full_like(dens_e, np.nan, dtype=np.float64)
    log_ni = np.full_like(dens_i, np.nan, dtype=np.float64)
    mne = dens_e > 0.0
    mni = dens_i > 0.0
    log_ne[mne] = np.log10(dens_e[mne])
    log_ni[mni] = np.log10(dens_i[mni])

    panels = [
        (log_ne, "log10(ne) [m^-3]", "inferno"),
        (temp_e, "Te [eV]", "magma"),
        (log_ni, "log10(ni_main) [m^-3]", "inferno"),
        (temp_i, "Ti_main [eV]", "magma"),
        (parr_flow, "parr_flow_main [m/s]", "coolwarm"),
        (parr_flow_r, "parr_flow_r_main [m/s]", "coolwarm"),
        (parr_flow_z, "parr_flow_z_main [m/s]", "coolwarm"),
        (grad_te_r, "grad_te_r [eV/m]", "viridis"),
        (grad_te_z, "grad_te_z [eV/m]", "viridis"),
        (br, "Br [T]", "RdBu_r"),
        (bt, "Bt [T]", "RdBu_r"),
        (bz, "Bz [T]", "RdBu_r"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(18, 11), constrained_layout=True)
    for ax, (arr, title, cmap) in zip(axes.flat, panels):
        im = ax.imshow(arr, origin='lower', extent=extent, cmap=cmap, aspect='auto')
        ax.plot(Rwall, Zwall, 'k-', lw=1.0)
        ax.set_title(title)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        fig.colorbar(im, ax=ax, shrink=0.88)
    fig.suptitle("SOLEDGE -> OpenEdge field sanity check", fontsize=15)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_flux_fields(
    out_total_png,
    out_species_png,
    r,
    z,
    Rwall,
    Zwall,
    ion_names,
    dens_i_all,
    temp_i_all,
    temp_e,
    br,
    bt,
    bz,
):
    """
    Plot cpmi-like Bohm wall-flux diagnostics:
      Gamma_s = n_s * c_s * sin(alpha_B)
    """
    extent = [r.min(), r.max(), z.min(), z.max()]
    gamma_all, sin_alpha = _compute_bohm_flux_maps(
        r, z, Rwall, Zwall, dens_i_all, temp_i_all, temp_e, br, bt, bz
    )
    gamma_sum = np.sum(gamma_all, axis=0)
    log_sum = np.full_like(gamma_sum, np.nan, dtype=np.float64)
    m = gamma_sum > 0.0
    log_sum[m] = np.log10(gamma_sum[m])

    # Total Bohm-flux proxy + incidence factor.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    im0 = axes[0].imshow(
        log_sum, origin='lower', extent=extent, aspect='auto', cmap='inferno'
    )
    axes[0].plot(Rwall, Zwall, 'k-', lw=1.0)
    axes[0].set_title("log10(sum Bohm flux) [m^-2 s^-1]")
    axes[0].set_xlabel("R [m]")
    axes[0].set_ylabel("Z [m]")
    fig.colorbar(im0, ax=axes[0], shrink=0.9)

    im1 = axes[1].imshow(
        sin_alpha, origin='lower', extent=extent, aspect='auto', cmap='viridis'
    )
    axes[1].plot(Rwall, Zwall, 'k-', lw=1.0)
    axes[1].set_title(r"$\sin(\alpha_B)$ from nearest wall normal")
    axes[1].set_xlabel("R [m]")
    axes[1].set_ylabel("Z [m]")
    fig.colorbar(im1, ax=axes[1], shrink=0.9)
    fig.savefig(out_total_png, dpi=180)
    plt.close(fig)

    # Per-species flux panel
    ns = gamma_all.shape[0]
    ncols = 3
    nrows = int(np.ceil(ns / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.9 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    for k in range(nrows * ncols):
        ax = axes.flat[k]
        if k >= ns:
            ax.axis('off')
            continue
        gk = gamma_all[k]
        log_gk = np.full_like(gk, np.nan, dtype=np.float64)
        mask = gk > 0.0
        log_gk[mask] = np.log10(gk[mask])
        im = ax.imshow(
            log_gk, origin='lower', extent=extent, aspect='auto', cmap='inferno'
        )
        ax.plot(Rwall, Zwall, 'k-', lw=0.9)
        ax.set_title(f"{ion_names[k]}: log10(Bohm flux) [m^-2 s^-1]")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        fig.colorbar(im, ax=ax, shrink=0.86)
    fig.savefig(out_species_png, dpi=180)
    plt.close(fig)


def _plot_wall_flux_vs_coord(
    vv_csv_file,
    out_png,
    ion_names,
    r,
    z,
    dens_i_all,
    temp_i_all,
    temp_e,
    br,
    bt,
    bz,
):
    """
    Plot cpmi-like Bohm flux sampled along wall coordinates from vv_values.csv.
    CSV expected columns: R, Z (at least first two columns).
    """
    if not vv_csv_file or not os.path.exists(vv_csv_file):
        return

    # Accept both whitespace-separated and comma-separated formats.
    try:
        data = np.loadtxt(vv_csv_file, unpack=True)
    except Exception:
        data = np.loadtxt(vv_csv_file, delimiter=',', unpack=True)

    if data.ndim == 1 or data.shape[0] < 2:
        return
    vv_r = np.asarray(data[0], dtype=np.float64)
    vv_z = np.asarray(data[1], dtype=np.float64)
    if vv_r.size < 2:
        return

    # Arc-length coordinate along wall points
    wall_coord = np.zeros(vv_r.size, dtype=np.float64)
    for i in range(1, vv_r.size):
        d_r = vv_r[i] - vv_r[i - 1]
        d_z = vv_z[i] - vv_z[i - 1]
        wall_coord[i] = wall_coord[i - 1] + np.sqrt(d_r * d_r + d_z * d_z)

    gamma_all, sin_alpha = _compute_bohm_flux_on_wall(
        vv_r, vv_z, ion_names, r, z, dens_i_all, temp_i_all, temp_e, br, bt, bz
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    # Panel 1: main ion versus impurity sum.
    gamma_d = gamma_all[0]
    gamma_o = np.sum(gamma_all[1:], axis=0) if gamma_all.shape[0] > 1 else np.zeros_like(gamma_d)
    d_name = ion_names[0] if ion_names else "main_ion"
    axes[0].plot(wall_coord, gamma_d, lw=2.0, color='tab:blue', label=d_name)
    axes[0].plot(wall_coord, gamma_o, lw=2.0, color='tab:red', label='sum(other ions)')
    axes[0].set_title("Wall Bohm Flux: D vs impurities")
    axes[0].set_xlabel("Wall coordinate [m]")
    axes[0].set_ylabel(r"$\Gamma_B$ [m^-2 s^-1]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # Panel 2: individual impurity species.
    if gamma_all.shape[0] > 1:
        for k in range(1, gamma_all.shape[0]):
            label = ion_names[k] if k < len(ion_names) else f"ion{k}"
            axes[1].plot(wall_coord, gamma_all[k], lw=1.6, label=label)
        axes[1].legend(fontsize=9, ncol=2)
    else:
        axes[1].text(0.5, 0.5, "No other ion species", ha='center', va='center')

    axes[1].set_title("Wall Bohm Flux: Other Ions")
    axes[1].set_xlabel("Wall coordinate [m]")
    axes[1].set_ylabel(r"$\Gamma_B$ [m^-2 s^-1]")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: incidence and D share.
    d_frac = np.divide(gamma_d, gamma_d + gamma_o, out=np.zeros_like(gamma_d), where=(gamma_d + gamma_o) > 0.0)
    axes[2].plot(wall_coord, sin_alpha, lw=1.8, color='tab:green', label=r'$\sin(\alpha_B)$')
    axes[2].plot(wall_coord, d_frac, lw=1.8, color='tab:purple', label='D fraction of total Bohm flux')
    axes[2].set_title("Limiter Diagnostic")
    axes[2].set_xlabel("Wall coordinate [m]")
    axes[2].set_ylabel("dimensionless")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9)

    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def interpolate_and_save_plasma_field(
    ref_file,
    mesh_file,
    bfield_file,
    data_file,
    wall_file=None,
    plasma_out_file=None,
    bfield_out_file=None,
    debug_plot_file=None,
    flux_total_plot_file=None,
    flux_species_plot_file=None,
    wall_flux_plot_file=None,
    wall_flux_csv_file=None,
    nR=200,
    nZ=200,
    main_ion_spec=1,
    ion_metadata=None,
    g_as_flux=True,
    n_norm_floor=1.0e-8,
    use_mesh_wall=True,
    wall_sparta_file=None,
    core_sparta_file=None,
    core_psi_level=None,
    equ_file=None,
    gfile=None,
    config_file=None,
):
    """
    Convert SOLEDGE3X plasma/mesh data to OpenEdge-style HDF5.

    B-field source priority:
      1. GEQDSK file (--gfile) — reconstructed from psi, most trustworthy
      2. .equ file (--equ-file) — reconstructed from psi
      3. mesh_raptorX.h5 triangles — S3X equilibrium on the simulation mesh (default)

    Backward-compatible legacy fields are still written:
      dens_i/temp_i/parr_flow(/r/t/z) from main ion species.

    New multi-ion fields are written under:
      ion_species/* and ions/*

    Native SOLEDGE triangle-mesh fields are also written under:
      mesh/*
    """

    # Reference scalings
    if plasma_out_file is None:
        raise ValueError("plasma_out_file must be provided")

    with h5py.File(ref_file, 'r') as ref:
        n0 = float(ref['/n0'][...])
        T0 = float(ref['/T0'][...])
        c0 = float(ref['/c0'][...])
        B0 = float(ref['/B0'][...])

    # Wall geometry + config: prefer SOLEDGE mesh/config source.
    # Optional fallback to wall_file only if explicitly provided.
    Rwall = Zwall = None
    r2d_cfg = z2d_cfg = psi2d_cfg = None
    psisep_cfg = psicore_cfg = None
    if use_mesh_wall:
        # Try mesh.h5 (carries /config/psi in SOLEDGE3X layouts) first, then
        # the usual knot/bfield files, then the plasma snapshot as last resort.
        candidates = []
        if config_file:
            candidates.append(config_file)
        candidates.extend([bfield_file, mesh_file, data_file])
        for candidate in candidates:
            try:
                Rwall, Zwall, r2d_cfg, z2d_cfg, psi2d_cfg, psisep_cfg, psicore_cfg = _wall_and_config_from_mesh(candidate, ref_file)
                print(f"Using wall/config geometry from mesh file: {candidate}")
                break
            except Exception:
                continue
    if Rwall is None or Zwall is None:
        if wall_file:
            print(f"WARNING: mesh wall read failed; falling back to wall file: {wall_file}")
            wall = surface(wall_file, "2D")
            domain = wall.polygon
            Rwall, Zwall = domain.exterior.xy
        else:
            raise RuntimeError(
                "Could not read wall geometry from mesh files, and no wall_file fallback was provided."
            )

    # Mesh (triangles + knots)
    with h5py.File(mesh_file, 'r') as mesh_eirene:
        tri_knots = mesh_eirene['/triangles/tri_knots'][...] - 1  # 0-based
        Rk = mesh_eirene['/knots/R'][...] / 100.0
        Zk = mesh_eirene['/knots/Z'][...] / 100.0

    points = np.vstack((Rk, Zk)).T
    triangle_points = np.array([points[tri] for tri in tri_knots.T])
    centroids = triangle_points.mean(axis=1)

    r = np.round(np.linspace(1.8, 3.2, nR), 4)
    z = np.round(np.linspace(-1.0, 1.0, nZ), 4)
    grid_r, grid_z = np.meshgrid(r, z)
    grid_points = np.vstack((grid_r.flatten(), grid_z.flatten())).T

    # Magnetic field: prefer equilibrium file over mesh triangles.
    if gfile is not None:
        from convert_solps_plasma import _read_geqdsk_bfield
        rg, zg, br_eq, bt_eq, bz_eq = _read_geqdsk_bfield(gfile)
        rr_eq, zz_eq = np.meshgrid(rg, zg)
        b_pts = np.column_stack((rr_eq.reshape(-1), zz_eq.reshape(-1)))
        b_r_grid = _interpolate_field(b_pts, grid_points, nZ, nR, br_eq.reshape(-1))
        b_t_grid = _interpolate_field(b_pts, grid_points, nZ, nR, bt_eq.reshape(-1))
        b_z_grid = _interpolate_field(b_pts, grid_points, nZ, nR, bz_eq.reshape(-1))
        print(f"B-field from GEQDSK: {gfile}")
    elif equ_file is not None:
        from convert_solps_plasma import _read_equilibrium_bfield
        rg, zg, br_eq, bt_eq, bz_eq = _read_equilibrium_bfield(equ_file)
        rr_eq, zz_eq = np.meshgrid(rg, zg)
        b_pts = np.column_stack((rr_eq.reshape(-1), zz_eq.reshape(-1)))
        b_r_grid = _interpolate_field(b_pts, grid_points, nZ, nR, br_eq.reshape(-1))
        b_t_grid = _interpolate_field(b_pts, grid_points, nZ, nR, bt_eq.reshape(-1))
        b_z_grid = _interpolate_field(b_pts, grid_points, nZ, nR, bz_eq.reshape(-1))
        print(f"B-field from equilibrium: {equ_file}")
    else:
        # Fallback: B-field from S3X mesh_raptorX.h5 triangles
        with h5py.File(bfield_file, 'r') as bfile:
            b_r_tri = bfile['/triangles/Br'][...].flatten() * B0
            b_z_tri = bfile['/triangles/Bz'][...].flatten() * B0
            b_t_tri = bfile['/triangles/Bphi'][...].flatten() * B0
        b_r_grid = _interpolate_field(centroids, grid_points, nZ, nR, b_r_tri)
        b_z_grid = _interpolate_field(centroids, grid_points, nZ, nR, b_z_tri)
        b_t_grid = _interpolate_field(centroids, grid_points, nZ, nR, b_t_tri)
        print(f"B-field from S3X mesh: {bfield_file} (consider using --gfile or --equ-file)")

    b_mag = np.sqrt(b_r_grid * b_r_grid + b_z_grid * b_z_grid + b_t_grid * b_t_grid)
    eps = 1.0e-12

    # Psi on the target (R,Z) grid, interpolated from /config/psi via griddata.
    # Written alongside the plasma fields so fix plasma/data consumers can
    # read psi directly without a second equilibrium file.
    psi_grid = None
    psi_norm_grid = None
    if psi2d_cfg is not None and r2d_cfg is not None and z2d_cfg is not None:
        if r2d_cfg.ndim == 1 and z2d_cfg.ndim == 1:
            rr_cfg, zz_cfg = np.meshgrid(r2d_cfg, z2d_cfg)
        else:
            rr_cfg, zz_cfg = r2d_cfg, z2d_cfg
        psi_pts = np.column_stack((rr_cfg.reshape(-1), zz_cfg.reshape(-1)))
        psi_vals = np.asarray(psi2d_cfg, dtype=np.float64).reshape(-1)
        finite = np.isfinite(psi_vals)
        if np.any(finite):
            psi_grid = _interpolate_field(
                psi_pts[finite], grid_points, nZ, nR, psi_vals[finite]
            )
            if (psisep_cfg is not None and psicore_cfg is not None
                    and np.isfinite(psisep_cfg) and np.isfinite(psicore_cfg)
                    and psisep_cfg != psicore_cfg):
                # Normalized flux: 0 at core, 1 at primary separatrix.
                # Uses SOLEDGE psicore (axis) and psisep1 (separatrix).
                psi_norm_grid = (psi_grid - float(psicore_cfg)) / (float(psisep_cfg) - float(psicore_cfg))
            print("Interpolated /config/psi onto plasma grid")

    # Optional SPARTA geometry exports from mesh-derived contours.
    if wall_sparta_file:
        _write_sparta_surface_polyline(wall_sparta_file, Rwall, Zwall,
                                        title="surface geometry")
        print(f"Wrote SPARTA wall surface (axi: x=Z, y=R): {wall_sparta_file}")

    if core_sparta_file:
        psi_level = psisep_cfg if core_psi_level is None else float(core_psi_level)
        rc, zc = _extract_core_contour(r2d_cfg, z2d_cfg, psi2d_cfg, psi_level)
        if rc is not None:
            _write_sparta_surface_polyline(core_sparta_file, rc, zc,
                                            title="surface geometry")
            print(f"Wrote SPARTA core surface (axi): {core_sparta_file} (psi={psi_level})")
        else:
            print("WARNING: could not extract core contour from config/psi; core surface not written.")

    # Wall mask
    print(
        f"Wall extent used for masking: R=[{np.nanmin(Rwall):.6g},{np.nanmax(Rwall):.6g}] "
        f"Z=[{np.nanmin(Zwall):.6g},{np.nanmax(Zwall):.6g}]"
    )
    wall_path = Path(np.vstack((Rwall, Zwall)).T)
    mask_outside_wall = ~wall_path.contains_points(grid_points)

    # Species data
    with h5py.File(data_file, 'r') as data:
        spec_inds = _find_species_indices(data)
        if not spec_inds:
            raise RuntimeError("No /triangles/spec* groups found in plasma file")

        # Convention for SOLEDGE3X currently: spec0 = electrons
        if 0 not in spec_inds:
            raise RuntimeError("Expected spec0 (electron) in SOLEDGE plasma file")

        ion_inds = [s for s in spec_inds if s != 0]
        if not ion_inds:
            raise RuntimeError("No ion species found (spec1+) in SOLEDGE plasma file")

        # Electron fields
        temp_e_tri = _safe_read_field(data, 0, 'T', T0)
        dens_e_tri = _safe_read_field(data, 0, 'n', n0)
        if temp_e_tri is None or dens_e_tri is None:
            raise RuntimeError("Missing electron fields /triangles/spec0/{T,n}")

        ntri_mesh = centroids.shape[0]
        if temp_e_tri.size != ntri_mesh or dens_e_tri.size != ntri_mesh:
            raise RuntimeError(
                "SOLEDGE triangle field size does not match meshEIRENE triangulation"
            )

        temp_e_grid = _interpolate_field(centroids, grid_points, nZ, nR, temp_e_tri)
        dens_e_grid = _interpolate_field(centroids, grid_points, nZ, nR, dens_e_tri)

        # Multi-ion arrays (Nion, Nz, Nr)
        nion = len(ion_inds)
        dens_i_all = np.zeros((nion, nZ, nR), dtype=np.float64)
        temp_i_all = np.zeros((nion, nZ, nR), dtype=np.float64)
        flow_i_par_all = np.zeros((nion, nZ, nR), dtype=np.float64)
        flow_i_r_all = np.zeros((nion, nZ, nR), dtype=np.float64)
        flow_i_t_all = np.zeros((nion, nZ, nR), dtype=np.float64)
        flow_i_z_all = np.zeros((nion, nZ, nR), dtype=np.float64)
        mesh_dens_i_all = np.zeros((nion, ntri_mesh), dtype=np.float64)
        mesh_temp_i_all = np.zeros((nion, ntri_mesh), dtype=np.float64)
        mesh_flow_i_par_all = np.zeros((nion, ntri_mesh), dtype=np.float64)
        ion_names, ion_mass_amu, ion_charge_state_z = _build_ion_metadata(
            ion_inds, ion_metadata=ion_metadata
        )

        for k, sidx in enumerate(ion_inds):
            n_tri = _safe_read_field(data, sidx, 'n', n0)
            T_tri = _safe_read_field(data, sidx, 'T', T0)
            # SOLEDGE convention:
            # - G is often normalized parallel particle flux-like quantity
            # - n is normalized density
            # Then u_par [m/s] = c0 * (G_norm / n_norm)
            # Fallback (g_as_flux=False): u_par = c0 * G_norm
            G_tri_norm = _safe_read_field(data, sidx, 'G', 1.0)
            n_tri_norm = _safe_read_field(data, sidx, 'n', 1.0)

            if n_tri is None:
                raise RuntimeError(f"Missing ion density for spec{sidx}: /triangles/spec{sidx}/n")
            if T_tri is None:
                T_tri = np.zeros_like(n_tri)
            if G_tri_norm is None:
                G_tri_norm = np.zeros_like(n_tri)
            if n_tri_norm is None:
                n_tri_norm = np.zeros_like(n_tri)

            if g_as_flux:
                denom = np.maximum(np.abs(n_tri_norm), n_norm_floor)
                upar_tri = c0 * (G_tri_norm / denom)
            else:
                upar_tri = c0 * G_tri_norm

            if n_tri.size != ntri_mesh or T_tri.size != ntri_mesh or upar_tri.size != ntri_mesh:
                raise RuntimeError(
                    f"SOLEDGE triangle field size mismatch for spec{sidx}"
                )

            n_grid = _interpolate_field(centroids, grid_points, nZ, nR, n_tri)
            T_grid = _interpolate_field(centroids, grid_points, nZ, nR, T_tri)
            G_grid = _interpolate_field(centroids, grid_points, nZ, nR, upar_tri)

            # Resolve parallel flow into cylindrical components along Bhat
            bhat_r = np.where(b_mag > eps, b_r_grid / b_mag, 0.0)
            bhat_t = np.where(b_mag > eps, b_t_grid / b_mag, 0.0)
            bhat_z = np.where(b_mag > eps, b_z_grid / b_mag, 0.0)

            vr_grid = G_grid * bhat_r
            vt_grid = G_grid * bhat_t
            vz_grid = G_grid * bhat_z

            dens_i_all[k] = n_grid
            temp_i_all[k] = T_grid
            flow_i_par_all[k] = G_grid
            flow_i_r_all[k] = vr_grid
            flow_i_t_all[k] = vt_grid
            flow_i_z_all[k] = vz_grid
            mesh_dens_i_all[k] = np.nan_to_num(np.asarray(n_tri, dtype=np.float64), nan=0.0)
            mesh_temp_i_all[k] = np.nan_to_num(np.asarray(T_tri, dtype=np.float64), nan=0.0)
            mesh_flow_i_par_all[k] = np.nan_to_num(np.asarray(upar_tri, dtype=np.float64), nan=0.0)

    # Legacy main-ion choice
    if main_ion_spec in ion_inds:
        main_k = ion_inds.index(main_ion_spec)
    else:
        main_k = 0

    dens_i_grid = dens_i_all[main_k]
    temp_i_grid = temp_i_all[main_k]
    parr_flow_i_grid = flow_i_par_all[main_k]
    parr_flow_i_r_grid = flow_i_r_all[main_k]
    parr_flow_i_t_grid = flow_i_t_all[main_k]
    parr_flow_i_z_grid = flow_i_z_all[main_k]
    mesh_vtx_r = np.asarray(Rk, dtype=np.float64)
    mesh_vtx_z = np.asarray(Zk, dtype=np.float64)
    mesh_tri = np.asarray(tri_knots.T, dtype=np.int32)
    mesh_cell_idx = np.arange(mesh_tri.shape[0], dtype=np.int32)
    mesh_dens_e = np.nan_to_num(np.asarray(dens_e_tri, dtype=np.float64), nan=0.0)
    mesh_temp_e = np.nan_to_num(np.asarray(temp_e_tri, dtype=np.float64), nan=0.0)
    mesh_dens_i = mesh_dens_i_all[main_k]
    mesh_temp_i = mesh_temp_i_all[main_k]
    mesh_parr_flow = mesh_flow_i_par_all[main_k]
    if mesh_tri.shape[0] != mesh_dens_e.size:
        raise RuntimeError("mesh/triangles count does not match SOLEDGE plasma triangle count")
    mpar = np.isfinite(parr_flow_i_grid)
    if np.any(mpar):
        print(
            f"Main-ion u_par stats [m/s]: min={np.nanmin(parr_flow_i_grid[mpar]):.6e}, "
            f"max={np.nanmax(parr_flow_i_grid[mpar]):.6e}"
        )

    # Temperature gradients from interpolated fields
    dz = z[1] - z[0]
    dr = r[1] - r[0]
    grad_te_z, grad_te_r = np.gradient(temp_e_grid, dz, dr)
    grad_ti_z, grad_ti_r = np.gradient(temp_i_grid, dz, dr)

    # Wall masking
    for arr in [
        temp_e_grid, dens_e_grid,
        temp_i_grid, dens_i_grid,
        parr_flow_i_grid, parr_flow_i_r_grid, parr_flow_i_t_grid, parr_flow_i_z_grid,
        b_r_grid, b_t_grid, b_z_grid,
        grad_te_r, grad_te_z,
        grad_ti_r, grad_ti_z,
    ]:
        _mask_outside_wall(arr, mask_outside_wall, nZ, nR, fill=0.0)

    for arr3 in [dens_i_all, temp_i_all, flow_i_par_all, flow_i_r_all, flow_i_t_all, flow_i_z_all]:
        for k in range(arr3.shape[0]):
            _mask_outside_wall(arr3[k], mask_outside_wall, nZ, nR, fill=0.0)

    # Write OpenEdge plasma + bfield files (separate)
    try:
        # plasma.h5
        with h5py.File(plasma_out_file, 'w') as f:
            f.create_dataset('r', data=r)
            f.create_dataset('z', data=z)
            f.create_dataset('dens_e', data=dens_e_grid)
            f.create_dataset('temp_e', data=temp_e_grid)
            f.create_dataset('dens_i', data=dens_i_grid)
            f.create_dataset('temp_i', data=temp_i_grid)

            f.create_dataset('parr_flow', data=parr_flow_i_grid)
            f.create_dataset('parr_flow_r', data=parr_flow_i_r_grid)
            f.create_dataset('parr_flow_t', data=parr_flow_i_t_grid)
            f.create_dataset('parr_flow_z', data=parr_flow_i_z_grid)

            f.create_dataset('grad_te_r', data=grad_te_r)
            f.create_dataset('grad_te_t', data=np.zeros_like(grad_te_r))
            f.create_dataset('grad_te_z', data=grad_te_z)
            f.create_dataset('grad_ti_r', data=grad_ti_r)
            f.create_dataset('grad_ti_t', data=np.zeros_like(grad_ti_r))
            f.create_dataset('grad_ti_z', data=grad_ti_z)

            # Magnetic field on the plasma grid (was previously a separate
            # bfield.h5). Downstream consumers (fix plasma/data, Boris pusher,
            # compute plasma/fields) pick these up by dataset name.
            f.create_dataset('br', data=b_r_grid)
            f.create_dataset('bt', data=b_t_grid)
            f.create_dataset('bz', data=b_z_grid)

            # Psi (and normalized psi) from SOLEDGE /config/psi, interpolated
            # onto the plasma grid. Optional — absent if mesh lacks /config.
            if psi_grid is not None:
                f.create_dataset('psi', data=psi_grid)
                if psi_norm_grid is not None:
                    f.create_dataset('psi_norm', data=psi_norm_grid)
                if psisep_cfg is not None and np.isfinite(psisep_cfg):
                    f.create_dataset('psisep', data=np.float64(psisep_cfg))
                if psicore_cfg is not None and np.isfinite(psicore_cfg):
                    f.create_dataset('psicore', data=np.float64(psicore_cfg))

            # Multi-ion extension
            sdt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('ion_species/names', data=np.array(ion_names, dtype=object), dtype=sdt)
            f.create_dataset('ion_species/spec_index', data=np.array(ion_inds, dtype=np.int32))
            f.create_dataset('ion_species/main_ion_spec_index', data=np.array([ion_inds[main_k]], dtype=np.int32))
            f.create_dataset('ion_species/mass_amu', data=ion_mass_amu)
            f.create_dataset('ion_species/charge_state_z', data=ion_charge_state_z)
            f.create_dataset('ions/dens', data=dens_i_all)
            f.create_dataset('ions/temp', data=temp_i_all)
            f.create_dataset('ions/parr_flow', data=flow_i_par_all)
            f.create_dataset('ions/parr_flow_r', data=flow_i_r_all)
            f.create_dataset('ions/parr_flow_t', data=flow_i_t_all)
            f.create_dataset('ions/parr_flow_z', data=flow_i_z_all)

            # Compatibility mirror groups
            f.create_dataset('solps_like/r', data=r)
            f.create_dataset('solps_like/z', data=z)
            f.create_dataset('n_e/temp', data=temp_e_grid)
            f.create_dataset('n_e/dens', data=dens_e_grid)
            f.create_dataset('n_i/temp', data=temp_i_grid)
            f.create_dataset('n_i/dens', data=dens_i_grid)
            f.create_dataset('n_i/parr_flow', data=parr_flow_i_grid)

            # Native SOLEDGE triangle mesh for direct point-in-cell lookup.
            # For SOLEDGE3X the plasma fields are already triangle-centered,
            # so the cell mapping is the identity.
            f.create_dataset('mesh/vtx_r', data=mesh_vtx_r)
            f.create_dataset('mesh/vtx_z', data=mesh_vtx_z)
            f.create_dataset('mesh/triangles', data=mesh_tri)
            f.create_dataset('mesh/cell_index', data=mesh_cell_idx)
            f.create_dataset('mesh/dens_e', data=mesh_dens_e)
            f.create_dataset('mesh/temp_e', data=mesh_temp_e)
            f.create_dataset('mesh/dens_i', data=mesh_dens_i)
            f.create_dataset('mesh/temp_i', data=mesh_temp_i)
            f.create_dataset('mesh/parr_flow', data=mesh_parr_flow)
            f.create_dataset('mesh/ions/dens', data=mesh_dens_i_all)
            f.create_dataset('mesh/ions/temp', data=mesh_temp_i_all)
            f.create_dataset('mesh/ions/parr_flow', data=mesh_flow_i_par_all)

        # Legacy standalone bfield.h5 — only written if the caller asks for
        # it. The default flow now keeps B on the plasma.h5 grid.
        if bfield_out_file:
            with h5py.File(bfield_out_file, 'w') as f:
                f.create_dataset('r', data=r)
                f.create_dataset('z', data=z)
                f.create_dataset('br', data=b_r_grid)
                f.create_dataset('bt', data=b_t_grid)
                f.create_dataset('bz', data=b_z_grid)

        print(f"Wrote OpenEdge plasma file: {plasma_out_file}")
        if bfield_out_file:
            print(f"Wrote OpenEdge bfield file: {bfield_out_file}")
        print(f"Detected species: electron=spec0, ions={ion_inds} (Nion={len(ion_inds)}), main ion=spec{ion_inds[main_k]}")
        if debug_plot_file:
            _plot_debug_fields(
                debug_plot_file,
                r, z, Rwall, Zwall,
                dens_e_grid, temp_e_grid,
                dens_i_grid, temp_i_grid,
                parr_flow_i_grid, parr_flow_i_r_grid, parr_flow_i_z_grid,
                grad_te_r, grad_te_z,
                b_r_grid, b_t_grid, b_z_grid,
            )
            print(f"Wrote debug plot: {debug_plot_file}")
        if flux_total_plot_file and flux_species_plot_file:
            _plot_flux_fields(
                flux_total_plot_file,
                flux_species_plot_file,
                r, z, Rwall, Zwall,
                ion_names,
                dens_i_all,
                temp_i_all,
                temp_e_grid,
                b_r_grid,
                b_t_grid,
                b_z_grid,
            )
            print(f"Wrote flux plot (total): {flux_total_plot_file}")
            print(f"Wrote flux plot (species): {flux_species_plot_file}")
        if wall_flux_plot_file and wall_flux_csv_file:
            _plot_wall_flux_vs_coord(
                wall_flux_csv_file,
                wall_flux_plot_file,
                ion_names,
                r,
                z,
                dens_i_all,
                temp_i_all,
                temp_e_grid,
                b_r_grid,
                b_t_grid,
                b_z_grid,
            )
            print(f"Wrote wall-flux plot: {wall_flux_plot_file}")

    except Exception as e:
        print(f"Error writing output files: {e}")
        raise


# Example batch
if __name__ == '__main__':
    # Default: 3MW dataset in /home/cloud/3MW, plasma_00010.h5 as the
    # final plasma snapshot. Override base_dir/data_name per case.
    cases = [('1p5MW', '/Users/42d/Downloads/1p5MW', 'plasma_00060.h5')]
    for case, base_dir, data_name in cases:
        _here = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(_here, '..', '..', 'examples', 'test_west_axi', 'input')

        ref_file = os.path.join(base_dir, 'refParam_raptorX.h5')
        mesh_file = os.path.join(base_dir, 'meshEIRENE.h5')
        data_file = os.path.join(base_dir, data_name)
        bfield_file = os.path.join(base_dir, 'mesh_raptorX.h5')
        config_file = os.path.join(base_dir, 'mesh.h5')
        wall_file = os.path.join(out_dir, 'wall.surf')
        plasma_out_file = os.path.join(out_dir, 'plasma.h5')
        debug_plot_file = os.path.join(out_dir, f'soledge_fields_{case}.png')
        flux_total_plot_file = os.path.join(out_dir, f'soledge_flux_total_{case}.png')
        flux_species_plot_file = os.path.join(out_dir, f'soledge_flux_species_{case}.png')
        wall_flux_plot_file = os.path.join(out_dir, f'soledge_flux_wallcoord_{case}.png')
        wall_flux_csv_file = os.path.join(out_dir, 'vv_values.csv')

        interpolate_and_save_plasma_field(
            ref_file, mesh_file, bfield_file, data_file, None,
            plasma_out_file, None,
            debug_plot_file=debug_plot_file,
            flux_total_plot_file=flux_total_plot_file,
            flux_species_plot_file=flux_species_plot_file,
            wall_flux_plot_file=wall_flux_plot_file,
            wall_flux_csv_file=wall_flux_csv_file,
            nR=200, nZ=200, main_ion_spec=1,
            use_mesh_wall=True,
            wall_sparta_file=wall_file,
            core_sparta_file=None,
            core_psi_level=None,
            config_file=config_file,
        )
