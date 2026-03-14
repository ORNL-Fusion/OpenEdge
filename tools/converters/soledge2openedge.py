import os
import re
import h5py
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from matplotlib.path import Path
import matplotlib.pyplot as plt
from utilities import surface

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
        r2d = z2d = psi2d = psisep = None
        if have_cfg:
            r2d = np.asarray(mesh["/config/r"][...], dtype=np.float64)
            z2d = np.asarray(mesh["/config/z"][...], dtype=np.float64)
            psi2d = np.asarray(mesh["/config/psi"][...], dtype=np.float64)
            if "/config/psisep1" in mesh:
                psisep = float(np.asarray(mesh["/config/psisep1"][...]).reshape(-1)[0])

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

        return rwall, zwall, r2d, z2d, psi2d, psisep


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
    """Write a closed polyline into SPARTA 2D surface text format."""
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

    n = int(r.size)
    if n < 3:
        return

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n\n")
        f.write(f"{n} points\n")
        f.write(f"{n} lines\n\n")
        f.write("Points\n\n")
        for i in range(n):
            f.write(f"{i+1} {r[i]:.12g} {z[i]:.12g}\n")
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
    flow_i_par_all,
):
    """
    Plot ion parallel particle flux diagnostics.
    Flux definition used here:
      Gamma_s = n_s * v_par_s   [m^-2 s^-1] (signed)
    """
    extent = [r.min(), r.max(), z.min(), z.max()]
    gamma_all = dens_i_all * flow_i_par_all  # (Nion, Nz, Nr)
    gamma_sum = np.sum(gamma_all, axis=0)
    gamma_abs_sum = np.sum(np.abs(gamma_all), axis=0)

    # Total flux panel (signed + abs-sum)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    vmax = np.nanmax(np.abs(gamma_sum))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0
    im0 = axes[0].imshow(
        gamma_sum, origin='lower', extent=extent, aspect='auto',
        cmap='RdBu_r', vmin=-vmax, vmax=vmax
    )
    axes[0].plot(Rwall, Zwall, 'k-', lw=1.0)
    axes[0].set_title("Gamma_total = sum(n_i v_par) [m^-2 s^-1]")
    axes[0].set_xlabel("R [m]")
    axes[0].set_ylabel("Z [m]")
    fig.colorbar(im0, ax=axes[0], shrink=0.9)

    # log10 of absolute summed flux
    log_abs = np.full_like(gamma_abs_sum, np.nan, dtype=np.float64)
    m = gamma_abs_sum > 0.0
    log_abs[m] = np.log10(gamma_abs_sum[m])
    im1 = axes[1].imshow(
        log_abs, origin='lower', extent=extent, aspect='auto', cmap='inferno'
    )
    axes[1].plot(Rwall, Zwall, 'k-', lw=1.0)
    axes[1].set_title("log10(sum|n_i v_par|) [m^-2 s^-1]")
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
        vk = np.nanmax(np.abs(gk))
        if not np.isfinite(vk) or vk == 0.0:
            vk = 1.0
        im = ax.imshow(
            gk, origin='lower', extent=extent, aspect='auto',
            cmap='RdBu_r', vmin=-vk, vmax=vk
        )
        ax.plot(Rwall, Zwall, 'k-', lw=0.9)
        ax.set_title(f"{ion_names[k]}: n v_par [m^-2 s^-1]")
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
    flow_i_par_all,
):
    """
    Plot species ion flux sampled along wall coordinates from vv_values.csv.
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

    # Flux per species on grid
    gamma_all = dens_i_all * flow_i_par_all  # (Nion, Nz, Nr)
    sample_pts = np.column_stack((vv_z, vv_r))  # interpolator axes are (z, r)

    # Two-panel wall-coordinate plot:
    # panel 1 = main ion (D), panel 2 = all other ions.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # Panel 1: main ion only (positive flux magnitude on linear scale)
    interp_d = RegularGridInterpolator(
        (z, r), gamma_all[0], method='linear', bounds_error=False, fill_value=0.0
    )
    gamma_d = np.abs(interp_d(sample_pts))
    d_name = ion_names[0] if ion_names else "main_ion"
    axes[0].plot(wall_coord, gamma_d, lw=2.0, color='tab:blue', label=d_name)
    axes[0].set_title("Wall Flux: Main Ion (D)")
    axes[0].set_xlabel("Wall coordinate [m]")
    axes[0].set_ylabel("|Gamma| [m^-2 s^-1]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # Panel 2: all other ions on same axes
    if gamma_all.shape[0] > 1:
        for k in range(1, gamma_all.shape[0]):
            interp_k = RegularGridInterpolator(
                (z, r), gamma_all[k], method='linear', bounds_error=False, fill_value=0.0
            )
            gamma_k = np.abs(interp_k(sample_pts))
            label = ion_names[k] if k < len(ion_names) else f"ion{k}"
            axes[1].plot(wall_coord, gamma_k, lw=1.6, label=label)
        axes[1].legend(fontsize=9, ncol=2)
    else:
        axes[1].text(0.5, 0.5, "No other ion species", ha='center', va='center')

    axes[1].set_title("Wall Flux: Other Ions")
    axes[1].set_xlabel("Wall coordinate [m]")
    axes[1].set_ylabel("|Gamma| [m^-2 s^-1]")
    axes[1].grid(True, alpha=0.3)

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
):
    """
    Convert SOLEDGE3X plasma/mesh data to OpenEdge-style HDF5.

    Backward-compatible legacy fields are still written:
      dens_i/temp_i/parr_flow(/r/t/z) from main ion species.

    New multi-ion fields are written under:
      ion_species/* and ions/*
    """

    # Reference scalings
    if plasma_out_file is None or bfield_out_file is None:
        raise ValueError("plasma_out_file and bfield_out_file must be provided")

    with h5py.File(ref_file, 'r') as ref:
        n0 = float(ref['/n0'][...])
        T0 = float(ref['/T0'][...])
        c0 = float(ref['/c0'][...])
        B0 = float(ref['/B0'][...])

    # Wall geometry + config: prefer SOLEDGE mesh/config source.
    # Optional fallback to wall_file only if explicitly provided.
    Rwall = Zwall = None
    r2d_cfg = z2d_cfg = psi2d_cfg = None
    psisep_cfg = None
    if use_mesh_wall:
        # Try bfield/config mesh first (often contains /config/*), then mesh_file.
        for candidate in [bfield_file, mesh_file, data_file]:
            try:
                Rwall, Zwall, r2d_cfg, z2d_cfg, psi2d_cfg, psisep_cfg = _wall_and_config_from_mesh(candidate, ref_file)
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

    # Magnetic field on triangles
    with h5py.File(bfield_file, 'r') as bfile:
        b_r_tri = bfile['/triangles/Br'][...].flatten() * B0
        b_z_tri = bfile['/triangles/Bz'][...].flatten() * B0
        b_t_tri = bfile['/triangles/Bphi'][...].flatten() * B0

    points = np.vstack((Rk, Zk)).T
    triangle_points = np.array([points[tri] for tri in tri_knots.T])
    centroids = triangle_points.mean(axis=1)

    r = np.round(np.linspace(1.8, 3.2, nR), 4)
    z = np.round(np.linspace(-1.0, 1.0, nZ), 4)
    grid_r, grid_z = np.meshgrid(r, z)
    grid_points = np.vstack((grid_r.flatten(), grid_z.flatten())).T

    # Interpolate B once (used for all species velocity components)
    b_r_grid = _interpolate_field(centroids, grid_points, nZ, nR, b_r_tri)
    b_z_grid = _interpolate_field(centroids, grid_points, nZ, nR, b_z_tri)
    b_t_grid = _interpolate_field(centroids, grid_points, nZ, nR, b_t_tri)
    b_mag = np.sqrt(b_r_grid * b_r_grid + b_z_grid * b_z_grid + b_t_grid * b_t_grid)
    eps = 1.0e-12

    # Optional SPARTA geometry exports from mesh-derived contours.
    if wall_sparta_file:
        _write_sparta_surface_polyline(wall_sparta_file, Rwall, Zwall, title="surface geometry")
        print(f"Wrote SPARTA wall surface: {wall_sparta_file}")

    if core_sparta_file:
        psi_level = psisep_cfg if core_psi_level is None else float(core_psi_level)
        rc, zc = _extract_core_contour(r2d_cfg, z2d_cfg, psi2d_cfg, psi_level)
        if rc is not None:
            _write_sparta_surface_polyline(core_sparta_file, rc, zc, title="surface geometry")
            print(f"Wrote SPARTA core surface: {core_sparta_file} (psi={psi_level})")
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

        # bfield.h5
        with h5py.File(bfield_out_file, 'w') as f:
            f.create_dataset('r', data=r)
            f.create_dataset('z', data=z)
            f.create_dataset('br', data=b_r_grid)
            f.create_dataset('bt', data=b_t_grid)
            f.create_dataset('bz', data=b_z_grid)

        print(f"Wrote OpenEdge plasma file: {plasma_out_file}")
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
                flow_i_par_all,
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
                flow_i_par_all,
            )
            print(f"Wrote wall-flux plot: {wall_flux_plot_file}")

    except Exception as e:
        print(f"Error writing output files: {e}")
        raise


# Example batch
if __name__ == '__main__':
    cases = ['1p5MW']
    for case in cases:
        base_dir = f'/path/to/soledge/{case}/run_dir'  # UPDATE: set your SOLEDGE3X output path
        _here = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(_here, '..', '..', 'examples', 'test_west_axi', 'input')

        ref_file = os.path.join(base_dir, 'refParam_raptorX.h5')
        mesh_file = os.path.join(base_dir, 'meshEIRENE.h5')
        data_file = os.path.join(base_dir, 'plasmaFinal.h5')
        bfield_file = os.path.join(base_dir, 'mesh_raptorX.h5')
        wall_file = os.path.join(out_dir, 'wall.txt')
        plasma_out_file = os.path.join(out_dir, 'plasma.h5')
        bfield_out_file = os.path.join(out_dir, 'bfield.h5')
        debug_plot_file = os.path.join(out_dir, f'soledge_fields_{case}.png')
        flux_total_plot_file = os.path.join(out_dir, f'soledge_flux_total_{case}.png')
        flux_species_plot_file = os.path.join(out_dir, f'soledge_flux_species_{case}.png')
        wall_flux_plot_file = os.path.join(out_dir, f'soledge_flux_wallcoord_{case}.png')
        wall_flux_csv_file = os.path.join(out_dir, 'vv_values.csv')

        interpolate_and_save_plasma_field(
            ref_file, mesh_file, bfield_file, data_file, None,
            plasma_out_file, bfield_out_file,
            debug_plot_file=debug_plot_file,
            flux_total_plot_file=flux_total_plot_file,
            flux_species_plot_file=flux_species_plot_file,
            wall_flux_plot_file=wall_flux_plot_file,
            wall_flux_csv_file=wall_flux_csv_file,
            nR=200, nZ=200, main_ion_spec=1,
            use_mesh_wall=True,
            wall_sparta_file=wall_file,
            core_sparta_file=None,
            core_psi_level=None
        )
