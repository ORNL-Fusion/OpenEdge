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


def _read_scalar(h5obj, key, default=None):
    if key not in h5obj:
        return default
    val = h5obj[key][...]
    try:
        return float(np.asarray(val).reshape(-1)[0])
    except Exception:
        return default



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


def _write_sparta_surface_polyline(path, rvals, zvals, title="surface geometry",
                                    geometry="cart"):
    """Write a closed polyline into SPARTA 2D surface text format.

    geometry='cart' (default): write (x=R, y=Z). Pairs with
        `boundary o o p`, `create_box R_min R_max Z_min Z_max ...`. Apply
        2*pi*R post-processing externally for axisymmetric integrals.
    geometry='axi': write (x=Z, y=R) for SPARTA's true axisymmetric layout
        with `boundary o ao p`, `create_box Z_min Z_max 0 R_max ...`,
        SPARTA handles cylindrical cell volumes and 2*pi*R*L surface
        areas internally.

    Vertex order is forced CCW in the chosen (x,y) so SPARTA normals
    point inward (into the fluid) and the deck does not need `invert`.
    """
    if rvals is None or zvals is None:
        return
    r = np.asarray(rvals, dtype=np.float64).reshape(-1)
    z = np.asarray(zvals, dtype=np.float64).reshape(-1)
    if r.size < 3 or r.size != z.size:
        return

    if np.isclose(r[0], r[-1]) and np.isclose(z[0], z[-1]):
        r = r[:-1]
        z = z[:-1]

    keep = np.ones(r.size, dtype=bool)
    keep[1:] = ~(np.isclose(r[1:], r[:-1]) & np.isclose(z[1:], z[:-1]))
    r = r[keep]
    z = z[keep]

    if r.size >= 2 and np.isclose(r[0], r[-1]) and np.isclose(z[0], z[-1]):
        r = r[:-1]
        z = z[:-1]

    n = int(r.size)
    if n < 3:
        return

    if geometry == "axi":
        x = z; y = r
    elif geometry == "cart":
        x = r; y = z
    else:
        raise ValueError(f"geometry must be 'cart' or 'axi', got {geometry!r}")

    # Shoelace signed area; flip to CCW so SPARTA normals point inward.
    signed_area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if signed_area < 0.0:
        x = x[::-1]
        y = y[::-1]

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n\n")
        f.write(f"{n} points\n")
        f.write(f"{n} lines\n\n")
        f.write("Points\n\n")
        for i in range(n):
            f.write(f"{i+1} {x[i]:.12g} {y[i]:.12g}\n")
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



def _load_zone_native(config_file, data_file, ref_file):
    """Zone-native SOLEDGE3X loader (fluid-neutral runs: no meshEIRENE.h5,
    no /triangles in the snapshot). Builds a triangulation from the zone
    quads (2 tris per quad, chi>0 penalized cells dropped) and returns the
    same intermediates the /triangles path produces. Zone corner coords are
    normalized: R = R0 + a0*x, Z = a0*z (R0/a0 from mesh.h5)."""
    with h5py.File(ref_file, 'r') as ref:
        n0 = float(ref['/n0'][...]); T0 = float(ref['/T0'][...])
        c0 = float(ref['/c0'][...]); B0 = float(ref['/B0'][...])
        phi0 = _read_scalar(ref, '/phi0', T0)   # potential norm [V]

    mesh = h5py.File(config_file, 'r')
    data = h5py.File(data_file, 'r')
    R0m = float(mesh['R0'][0]); a0m = float(mesh['a0'][0])
    nz_zones = int(mesh['NZones'][0])

    # ion metadata from species_raptorX.txt (row 0 = electrons)
    spfile = os.path.join(os.path.dirname(ref_file), 'species_raptorX.txt')
    rows = [l.split() for l in open(spfile) if l.strip()]
    ions = rows[1:]
    ion_names = [f"{r[0]}+" if int(r[2]) == 1 else f"{r[0]}{int(r[2])}+" for r in ions]
    ion_mass_amu = np.array([float(r[1]) for r in ions])
    ion_charge_state_z = np.array([int(r[2]) for r in ions], dtype=np.int32)
    nion = len(ions)

    vidx = {}
    vr, vz, tris, zc = [], [], [], []

    def vid(r, z):
        key = (round(float(r), 9), round(float(z), 9))
        if key not in vidx:
            vidx[key] = len(vr); vr.append(float(r)); vz.append(float(z))
        return vidx[key]

    for z in range(1, nz_zones + 1):
        R = R0m + a0m * mesh[f'zone{z}/Rcorners'][:, :, 0]
        Z = a0m * mesh[f'zone{z}/Zcorners'][:, :, 0]
        chi = mesh[f'zone{z}/chi'][:, :, 0]
        ni, nj = R.shape[0] - 1, R.shape[1] - 1
        for i in range(ni):
            for j in range(nj):
                if chi[i, j] > 0.5:
                    continue
                v00 = vid(R[i, j], Z[i, j]); v10 = vid(R[i+1, j], Z[i+1, j])
                v01 = vid(R[i, j+1], Z[i, j+1]); v11 = vid(R[i+1, j+1], Z[i+1, j+1])
                tris.append((v00, v10, v11)); zc.append((z, i, j))
                tris.append((v00, v11, v01)); zc.append((z, i, j))

    ntri = len(tris)
    print(f"Zone-native SOLEDGE load: {nz_zones} zones -> {ntri} triangles, {len(vr)} vertices")

    def zfield(z, path):
        a = data[f'zone{z}/{path}'][:, :, 0]
        return a[2:-2, 2:-2]                      # strip 2 ghost layers

    cache = {}
    for z in range(1, nz_zones + 1):
        ne = zfield(z, 'spec0/n'); te = zfield(z, 'spec0/T')
        # electrostatic potential (zone-level, only nonzero when the
        # SOLEDGE3X run solved the drift/vorticity model)
        phi = zfield(z, 'PHI') if f'zone{z}/PHI' in data else None
        # fluid-neutral atomic D density (Nspec1 = element 1 = D in
        # fluid-neutral runs; used as the impurity-CX partner density)
        nn = zfield(z, 'Nspec1/n') if f'zone{z}/Nspec1/n' in data else None
        tn = zfield(z, 'Nspec1/T') if f'zone{z}/Nspec1/T' in data else None
        # per-element fluid-neutral densities (NspecE = element E)
        nn_elt = [zfield(z, f'Nspec{e}/n')
                  if f'zone{z}/Nspec{e}/n' in data else None
                  for e in range(1, 10) if f'zone{z}/Nspec{e}' in data]
        per = []
        for sp in range(1, nion + 1):
            n = zfield(z, f'spec{sp}/n'); T = zfield(z, f'spec{sp}/T')
            G = zfield(z, f'spec{sp}/G')
            per.append((n, T, G))
        Br = mesh[f'zone{z}/Br'][:, :, 0]; Bz = mesh[f'zone{z}/Bz'][:, :, 0]
        Bt = mesh[f'zone{z}/Bphi'][:, :, 0]
        cache[z] = (ne, te, per, Br, Bz, Bt, phi, nn, tn, nn_elt)

    dens_e = np.empty(ntri); temp_e = np.empty(ntri)
    d_all = np.zeros((nion, ntri)); t_all = np.zeros((nion, ntri)); u_all = np.zeros((nion, ntri))
    br = np.empty(ntri); bz = np.empty(ntri); bt = np.empty(ntri)
    pob = np.zeros(ntri)
    nn_tri = np.zeros(ntri)
    tn_tri = np.zeros(ntri)
    nelt = max(len(c[9]) for c in cache.values())
    nn_elt_tri = np.zeros((nelt, ntri))
    for k, (z, i, j) in enumerate(zc):
        ne, te, per, Br, Bz, Bt, phi, nn, tn, nn_elt = cache[z]
        dens_e[k] = ne[i, j] * n0; temp_e[k] = te[i, j] * T0
        br[k] = Br[i, j] * B0; bz[k] = Bz[i, j] * B0; bt[k] = Bt[i, j] * B0
        if phi is not None:
            pob[k] = phi[i, j] * phi0
        if nn is not None:
            nn_tri[k] = max(nn[i, j], 0.0) * n0
        if tn is not None:
            tn_tri[k] = max(tn[i, j], 0.0) * T0
        for e, ne_e in enumerate(nn_elt):
            if ne_e is not None:
                nn_elt_tri[e, k] = max(ne_e[i, j], 0.0) * n0
        for sp in range(nion):
            n, T, G = per[sp]
            d_all[sp, k] = n[i, j] * n0
            t_all[sp, k] = T[i, j] * T0
            u_all[sp, k] = (G[i, j] / max(abs(n[i, j]), 1e-30)) * c0

    mesh.close(); data.close()
    return dict(
        Rk=np.array(vr), Zk=np.array(vz),
        tri_knots=np.array(tris, dtype=np.int64).T,
        dens_e_tri=dens_e, temp_e_tri=temp_e,
        dens_i_all=d_all, temp_i_all=t_all, flow_i_par_all=u_all,
        tri_br=br, tri_bz=bz, tri_bt=bt, pob_tri=pob,
        dens_n_tri=nn_tri, temp_n_tri=tn_tri,
        dens_n_elt_tri=nn_elt_tri,
        ion_names=ion_names, ion_mass_amu=ion_mass_amu,
        ion_charge_state_z=ion_charge_state_z,
        ion_inds=list(range(1, nion + 1)),
    )


def interpolate_and_save_plasma_field(
    ref_file,
    mesh_file,
    bfield_file,
    data_file,
    wall_file=None,
    plasma_out_file=None,
    main_ion_spec=1,
    ion_metadata=None,
    g_as_flux=True,
    n_norm_floor=1.0e-8,
    use_mesh_wall=True,
    wall_sparta_file=None,
    equ_file=None,
    gfile=None,
    config_file=None,
    geometry="cart",
    heatflux="none",
    sheath_gamma=7.0,
    efield="auto",
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

    # Zone-native fallback: fluid-neutral runs carry no meshEIRENE.h5 and
    # no /triangles group in the plasma snapshot (NeutralModel = 2).
    zone_native = not (mesh_file and os.path.exists(mesh_file))
    if not zone_native:
        with h5py.File(data_file, 'r') as _d:
            zone_native = 'triangles' not in _d
    zn = None
    if zone_native:
        zn = _load_zone_native(config_file, data_file, ref_file)
        Rk, Zk = zn['Rk'], zn['Zk']
        tri_knots = zn['tri_knots']
    else:
        # Mesh (triangles + knots)
        with h5py.File(mesh_file, 'r') as mesh_eirene:
            tri_knots = mesh_eirene['/triangles/tri_knots'][...] - 1  # 0-based
            Rk = mesh_eirene['/knots/R'][...] / 100.0
            Zk = mesh_eirene['/knots/Z'][...] / 100.0

    points = np.vstack((Rk, Zk)).T
    triangle_points = np.array([points[tri] for tri in tri_knots.T])
    centroids = triangle_points.mean(axis=1)

    # Mesh-native. No regular (R, Z) grid is built. Plasma / B-field are
    # carried on the native SOLEDGE3X triangulation; psi is carried on
    # the native /config/psi grid from mesh.h5. --nr/--nz are not needed.

    # Read B-field directly from mesh.h5 /triangles/{Br,Bz,Bphi}. This is
    # the native SOLEDGE3X equilibrium on the simulation mesh. Fallback
    # to --equ-file / --gfile only if /triangles/B* is missing (rare).
    # All values end up interpolated / averaged onto mesh vertices later.
    tri_br_src = tri_bz_src = tri_bt_src = None
    equ_r_src = equ_z_src = equ_br_src = equ_bz_src = equ_bt_src = None
    if zone_native:
        tri_br_src, tri_bz_src, tri_bt_src = zn['tri_br'], zn['tri_bz'], zn['tri_bt']
        print("B-field from SOLEDGE zones (cell-centered, x B0)")
    elif gfile is not None:
        from convert_solps_plasma import _read_geqdsk_bfield
        equ_r_src, equ_z_src, equ_br_src, equ_bt_src, equ_bz_src = _read_geqdsk_bfield(gfile)
        print(f"B-field from GEQDSK: {gfile}")
    elif equ_file is not None:
        from convert_solps_plasma import _read_equilibrium_bfield
        equ_r_src, equ_z_src, equ_br_src, equ_bt_src, equ_bz_src = _read_equilibrium_bfield(equ_file)
        print(f"B-field from equilibrium: {equ_file}")
    else:
        with h5py.File(bfield_file, 'r') as bfile:
            tri_br_src = np.asarray(bfile['/triangles/Br'][...],   dtype=np.float64).flatten() * B0
            tri_bz_src = np.asarray(bfile['/triangles/Bz'][...],   dtype=np.float64).flatten() * B0
            tri_bt_src = np.asarray(bfile['/triangles/Bphi'][...], dtype=np.float64).flatten() * B0
        print(f"B-field from S3X mesh: {bfield_file}")

    # Optional SPARTA geometry exports from mesh-derived contours.
    if wall_sparta_file:
        _write_sparta_surface_polyline(wall_sparta_file, Rwall, Zwall,
                                        title="surface geometry",
                                        geometry=geometry)
        layout = "x=Z, y=R" if geometry == "axi" else "x=R, y=Z"
        print(f"Wrote SPARTA wall surface ({geometry}: {layout}): {wall_sparta_file}")

    # Wall extent (informational).
    print(
        f"Wall extent: R=[{np.nanmin(Rwall):.6g},{np.nanmax(Rwall):.6g}] "
        f"Z=[{np.nanmin(Zwall):.6g},{np.nanmax(Zwall):.6g}]"
    )

    # Species data
    if zone_native:
        temp_e_tri = zn['temp_e_tri']; dens_e_tri = zn['dens_e_tri']
        ion_inds = zn['ion_inds']; nion = len(ion_inds)
        mesh_dens_i_all = zn['dens_i_all']
        mesh_temp_i_all = zn['temp_i_all']
        mesh_flow_i_par_all = zn['flow_i_par_all']
        ion_names = zn['ion_names']
        ion_mass_amu = zn['ion_mass_amu']
        ion_charge_state_z = zn['ion_charge_state_z']
    else:
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

        # Multi-ion per-triangle arrays (no regular grid).
        nion = len(ion_inds)
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
            # Then u_par [m/s] = c0 * (G_norm / n_norm).
            # Fallback (g_as_flux=False): u_par = c0 * G_norm.
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

            mesh_dens_i_all[k]     = np.nan_to_num(np.asarray(n_tri,    dtype=np.float64), nan=0.0)
            mesh_temp_i_all[k]     = np.nan_to_num(np.asarray(T_tri,    dtype=np.float64), nan=0.0)
            mesh_flow_i_par_all[k] = np.nan_to_num(np.asarray(upar_tri, dtype=np.float64), nan=0.0)

    # Main-ion choice.
    if main_ion_spec in ion_inds:
        main_k = ion_inds.index(main_ion_spec)
    else:
        main_k = 0
    mesh_vtx_r = np.asarray(Rk, dtype=np.float64)
    mesh_vtx_z = np.asarray(Zk, dtype=np.float64)
    mesh_tri = np.asarray(tri_knots.T, dtype=np.int32)
    mesh_cell_idx = np.arange(mesh_tri.shape[0], dtype=np.int32)
    mesh_dens_e = np.nan_to_num(np.asarray(dens_e_tri, dtype=np.float64), nan=0.0)
    mesh_temp_e = np.nan_to_num(np.asarray(temp_e_tri, dtype=np.float64), nan=0.0)
    mesh_dens_i = mesh_dens_i_all[main_k]
    mesh_temp_i = mesh_temp_i_all[main_k]
    mesh_parr_flow = mesh_flow_i_par_all[main_k]

    # Per-triangle mesh-native gradients via weighted least-squares fit
    # in physical (R, Z) space. Mirrors the SOLPS converter's _grad_rz_lsq
    # (cell-stencil) pattern, but adapted to the SOLEDGE3X unstructured
    # triangulation: for each triangle, the stencil is the set of triangles
    # sharing at least one vertex with it. Fit T(R,Z) = T0 + b*(R-R0) + c*(Z-Z0)
    # via inverse-distance-weighted normal equations; (b, c) = (grad_R, grad_Z).
    tri_R = mesh_vtx_r[mesh_tri].mean(axis=1)
    tri_Z = mesh_vtx_z[mesh_tri].mean(axis=1)

    # Build vertex -> triangles map, then for each triangle gather its
    # vertex-neighbor triangles (one-ring stencil).
    ntri = mesh_tri.shape[0]
    nvtx = mesh_vtx_r.size
    vtx_tris: list[list[int]] = [[] for _ in range(nvtx)]
    for t in range(ntri):
        for k in range(3):
            vtx_tris[mesh_tri[t, k]].append(t)

    def _tri_lsq_gradient(field_tri):
        """Per-triangle (grad_R, grad_Z) via one-ring vertex-neighbor LSQ fit."""
        gR = np.zeros(ntri, dtype=np.float64)
        gZ = np.zeros(ntri, dtype=np.float64)
        f_tri = np.asarray(field_tri, dtype=np.float64)
        for t in range(ntri):
            f0 = f_tri[t]
            r0 = tri_R[t]; z0 = tri_Z[t]
            if not (np.isfinite(f0) and np.isfinite(r0) and np.isfinite(z0)):
                continue
            # Gather unique vertex-neighbor triangles (exclude self).
            stencil = set()
            for k in range(3):
                stencil.update(vtx_tris[mesh_tri[t, k]])
            stencil.discard(t)
            a11 = a12 = a22 = 0.0
            b1 = b2 = 0.0
            npt = 0
            for s in stencil:
                f1 = f_tri[s]
                if not np.isfinite(f1):
                    continue
                dR = tri_R[s] - r0
                dZ = tri_Z[s] - z0
                ds2 = dR * dR + dZ * dZ
                if ds2 <= 1e-20:
                    continue
                w = 1.0 / ds2
                df = f1 - f0
                a11 += w * dR * dR
                a12 += w * dR * dZ
                a22 += w * dZ * dZ
                b1  += w * dR * df
                b2  += w * dZ * df
                npt += 1
            det = a11 * a22 - a12 * a12
            if npt < 3 or abs(det) <= 1e-24:
                continue
            gR[t] = ( a22 * b1 - a12 * b2) / det
            gZ[t] = (-a12 * b1 + a11 * b2) / det
        return gR, gZ

    mesh_grad_te_r, mesh_grad_te_z = _tri_lsq_gradient(mesh_temp_e)
    mesh_grad_ti_r, mesh_grad_ti_z = _tri_lsq_gradient(mesh_temp_i)
    for arr in [mesh_grad_te_r, mesh_grad_te_z,
                mesh_grad_ti_r, mesh_grad_ti_z]:
        arr[~np.isfinite(arr)] = 0.0

    # Electric field on the mesh. Priority: (1) E = -grad(phi) from the
    # real SOLEDGE3X potential zone*/PHI when the run solved the
    # drift/vorticity model (pob_tri); (2) parallel Ohm's-law field
    # E_par = -grad_par(pe)/(e ne) - 0.71 grad_par(Te)/e, written as the
    # field-aligned vector E_par*b_hat so ExB is identically zero —
    # this is the presheath field implicit in any no-drift fluid run;
    # (3) --efield zero.
    # fluid-neutral D density (zone-native only; zeros otherwise). CX
    # partner density for fix volume/chem/adas impurity channels.
    mesh_dens_n = np.zeros(mesh_tri.shape[0], dtype=np.float64)
    if zone_native and zn.get('dens_n_tri') is not None:
        mesh_dens_n = np.nan_to_num(
            np.asarray(zn['dens_n_tri'], dtype=np.float64), nan=0.0)
        print(f"mesh/dens_n (fluid-neutral D): max {mesh_dens_n.max():.3g} "
              f"m^-3, nonzero {int((mesh_dens_n > 0).sum())}/{mesh_dens_n.size}")
    # per-element fluid-neutral densities (dens_n_D/_O/_W ...), element
    # order = first appearance in the ion list
    mesh_dens_n_elt = {}
    if zone_native and zn.get('dens_n_elt_tri') is not None:
        _elts = []
        for _n in zn['ion_names']:
            _e = re.sub(r'[+\-\d]+$', '', _n)
            if _e not in _elts: _elts.append(_e)
        for _k, _e in enumerate(_elts):
            _arr = np.asarray(zn['dens_n_elt_tri'])
            if _k < _arr.shape[0]:
                mesh_dens_n_elt[_e] = np.nan_to_num(
                    np.asarray(_arr[_k], dtype=np.float64), nan=0.0)
                print(f"mesh/dens_n_{_e}: max {mesh_dens_n_elt[_e].max():.3g} "
                      f"m^-3, nonzero "
                      f"{int((mesh_dens_n_elt[_e] > 0).sum())}/{_arr.shape[1]}")
    mesh_temp_n = np.zeros(mesh_tri.shape[0], dtype=np.float64)
    if zone_native and zn.get('temp_n_tri') is not None:
        mesh_temp_n = np.nan_to_num(
            np.asarray(zn['temp_n_tri'], dtype=np.float64), nan=0.0)
    mesh_pob = np.zeros(mesh_tri.shape[0], dtype=np.float64)
    if efield != "zero" and zone_native and zn.get('pob_tri') is not None:
        mesh_pob = np.nan_to_num(
            np.asarray(zn['pob_tri'], dtype=np.float64), nan=0.0)
    ntri_m = mesh_tri.shape[0]
    mesh_e_r = np.zeros(ntri_m, dtype=np.float64)
    mesh_e_z = np.zeros(ntri_m, dtype=np.float64)
    mesh_e_t = np.zeros(ntri_m, dtype=np.float64)
    if np.abs(mesh_pob).max() > 0.0:
        print(f"mesh/pob from SOLEDGE zone*/PHI: "
              f"[{mesh_pob.min():.3g}, {mesh_pob.max():.3g}] V")
        gR, gZ = _tri_lsq_gradient(mesh_pob)
        gR[~np.isfinite(gR)] = 0.0
        gZ[~np.isfinite(gZ)] = 0.0
        mesh_e_r = -gR
        mesh_e_z = -gZ
    elif efield != "zero" and tri_br_src is not None:
        # Ohm's-law parallel field from electron force balance (the same
        # closure a no-drift fluid run uses implicitly). Te in eV so
        # gradients are directly V/m; axisym -> b.grad = bR dR + bZ dZ.
        bmag = np.sqrt(tri_br_src**2 + tri_bz_src**2 + tri_bt_src**2)
        bmag = np.where(bmag > 0.0, bmag, 1.0)
        bR_h, bZ_h, bT_h = (tri_br_src / bmag, tri_bz_src / bmag,
                            tri_bt_src / bmag)
        gpR, gpZ = _tri_lsq_gradient(mesh_dens_e * mesh_temp_e)
        for arr in (gpR, gpZ):
            arr[~np.isfinite(arr)] = 0.0
        ne_safe = np.maximum(mesh_dens_e, 1.0e14)
        e_par = (-(bR_h * gpR + bZ_h * gpZ) / ne_safe
                 - 0.71 * (bR_h * mesh_grad_te_r + bZ_h * mesh_grad_te_z))
        e_par[~np.isfinite(e_par) | (mesh_dens_e <= 0.0)] = 0.0
        mesh_e_r = e_par * bR_h
        mesh_e_z = e_par * bZ_h
        mesh_e_t = e_par * bT_h
        pct = np.percentile(np.abs(e_par), [50, 95, 99.9])
        print(f"mesh E from Ohm E_par (PHI absent): |E_par| median "
              f"{pct[0]:.3g}, p95 {pct[1]:.3g}, p99.9 {pct[2]:.3g} V/m")
        # sign check: in strong-flow regions the presheath field should
        # accelerate ions toward the target, i.e. e*E_par along u_par
        strong = np.abs(mesh_parr_flow) > 1.0e4
        if strong.any():
            agree = np.sign(e_par[strong]) == np.sign(mesh_parr_flow[strong])
            print(f"  E_par/u_par same-sign in strong-flow triangles: "
                  f"{100.0*agree.mean():.0f}% ({int(strong.sum())} tris)")
    elif efield != "zero":
        print("WARNING: mesh E=0 (PHI absent and no per-triangle B for "
              "the Ohm E_par fallback).")
    else:
        print("mesh E=0 (--efield zero).")
    if mesh_tri.shape[0] != mesh_dens_e.size:
        raise RuntimeError("mesh/triangles count does not match SOLEDGE plasma triangle count")

    # ---- Per-vertex magnetic field on the EIRENE triangulation ----
    # Primary path: the native per-triangle B from mesh.h5
    # (/triangles/Br,Bz,Bphi) area-weighted onto the vertices.
    # Fallback: if B came from --gfile / --equ-file, interpolate the
    # equilibrium B directly onto vertex positions.
    if tri_br_src is not None:
        v0 = np.column_stack([mesh_vtx_r[mesh_tri[:, 0]], mesh_vtx_z[mesh_tri[:, 0]]])
        v1 = np.column_stack([mesh_vtx_r[mesh_tri[:, 1]], mesh_vtx_z[mesh_tri[:, 1]]])
        v2 = np.column_stack([mesh_vtx_r[mesh_tri[:, 2]], mesh_vtx_z[mesh_tri[:, 2]]])
        tri_area = 0.5 * np.abs(
            (v1[:, 0]-v0[:, 0]) * (v2[:, 1]-v0[:, 1])
            - (v2[:, 0]-v0[:, 0]) * (v1[:, 1]-v0[:, 1])
        )
        nv = mesh_vtx_r.size
        wsum = np.zeros(nv)
        br_v = np.zeros(nv); bz_v = np.zeros(nv); bt_v = np.zeros(nv)
        for k in range(3):
            idx = mesh_tri[:, k]
            np.add.at(wsum, idx, tri_area)
            np.add.at(br_v,  idx, tri_area * tri_br_src)
            np.add.at(bz_v,  idx, tri_area * tri_bz_src)
            np.add.at(bt_v,  idx, tri_area * tri_bt_src)
        nz = wsum > 0
        mesh_vtx_br = np.zeros(nv); mesh_vtx_bz = np.zeros(nv); mesh_vtx_bt = np.zeros(nv)
        mesh_vtx_br[nz] = br_v[nz] / wsum[nz]
        mesh_vtx_bz[nz] = bz_v[nz] / wsum[nz]
        mesh_vtx_bt[nz] = bt_v[nz] / wsum[nz]
        print(f"Per-vertex B from mesh.h5 /triangles/B[rzφ] (area-avg to {nv} vertices)")
    else:
        from convert_solps_plasma import _interp_field_points
        rr_eq, zz_eq = np.meshgrid(equ_r_src, equ_z_src)
        eq_pts = np.column_stack([rr_eq.reshape(-1), zz_eq.reshape(-1)])
        tgt = np.column_stack([mesh_vtx_r, mesh_vtx_z])
        mesh_vtx_br = _interp_field_points(eq_pts, np.asarray(equ_br_src, dtype=np.float64).reshape(-1), tgt)
        mesh_vtx_bz = _interp_field_points(eq_pts, np.asarray(equ_bz_src, dtype=np.float64).reshape(-1), tgt)
        mesh_vtx_bt = _interp_field_points(eq_pts, np.asarray(equ_bt_src, dtype=np.float64).reshape(-1), tgt)
        print(f"Per-vertex B interpolated from equilibrium ({len(equ_r_src)}x{len(equ_z_src)} source)")
    mesh_vtx_bpol = np.sqrt(mesh_vtx_br**2 + mesh_vtx_bz**2)
    mesh_vtx_bmag = np.sqrt(mesh_vtx_bpol**2 + mesh_vtx_bt**2)

    # Provenance / wall-gap mask — SOLEDGE3X meshes extend to the wall
    # natively, so no flood-fill; every triangle is "native".
    mesh_tri_source_kind = np.ones(mesh_tri.shape[0], dtype=np.int32)
    mesh_wall_gap_mask = np.zeros(mesh_tri.shape[0], dtype=np.int32)
    mpar = np.isfinite(mesh_flow_i_par_all[main_k])
    if np.any(mpar):
        uvals = mesh_flow_i_par_all[main_k][mpar]
        print(f"Main-ion u_par stats [m/s]: min={uvals.min():.6e}, max={uvals.max():.6e}")

    # Write OpenEdge plasma + bfield files (separate)
    try:
        # plasma.h5 — mesh-only output (matches SOLPS converter schema).
        # OpenEdge C++ reads the per-cell plasma from /mesh/* and the
        # equilibrium from /equilibrium/*. No top-level regular-grid
        # datasets are written; /config/psi is handed to /equilibrium/
        # directly on its native grid.
        with h5py.File(plasma_out_file, 'w') as f:
            # /equilibrium from SOLEDGE /config/{r, z, psi} written verbatim
            # on its native grid — no resampling. SOLEDGE3X does not expose
            # btf/rtf, so those are omitted (OpenEdge reads them optionally).
            if psi2d_cfg is not None and r2d_cfg is not None and z2d_cfg is not None:
                if r2d_cfg.ndim == 1 and z2d_cfg.ndim == 1:
                    equ_r = np.asarray(r2d_cfg, dtype=np.float64)
                    equ_z = np.asarray(z2d_cfg, dtype=np.float64)
                    equ_psi = np.asarray(psi2d_cfg, dtype=np.float64)
                else:
                    # SOLEDGE3X /config/{r, z, psi} are 2D meshgrid-style
                    # arrays. Detect which axis varies with R vs Z by looking
                    # at what's constant along each axis — rather than guessing
                    # from shape, which silently breaks on square grids.
                    r_varies_rows = not np.allclose(r2d_cfg[0, :], r2d_cfg[0, 0])
                    r_varies_cols = not np.allclose(r2d_cfg[:, 0], r2d_cfg[0, 0])
                    if r_varies_rows and not r_varies_cols:
                        # Row 0 sweeps R -> r axis is r2d[0,:], z axis is z2d[:,0]
                        # and psi is already in (iZ, iR) = (row, col) order.
                        equ_r = np.asarray(r2d_cfg[0, :], dtype=np.float64)
                        equ_z = np.asarray(z2d_cfg[:, 0], dtype=np.float64)
                        equ_psi = np.asarray(psi2d_cfg, dtype=np.float64)
                    elif r_varies_cols and not r_varies_rows:
                        # Column 0 sweeps R -> r axis is r2d[:,0], z axis is
                        # z2d[0,:], psi is (iR, iZ) and needs transposing to
                        # match the OpenEdge convention psi[iZ, iR].
                        equ_r = np.asarray(r2d_cfg[:, 0], dtype=np.float64)
                        equ_z = np.asarray(z2d_cfg[0, :], dtype=np.float64)
                        equ_psi = np.asarray(psi2d_cfg, dtype=np.float64).T
                    else:
                        raise RuntimeError(
                            "convert_s3x_plasma: cannot determine R/Z axis "
                            "orientation in mesh.h5 /config/{r,z} — both or "
                            "neither axis varies. Inspect the mesh file.")
                f.create_dataset('equilibrium/r',    data=equ_r)
                f.create_dataset('equilibrium/z',    data=equ_z)
                f.create_dataset('equilibrium/psi',  data=equ_psi)
                if psisep_cfg is not None and np.isfinite(psisep_cfg):
                    f.create_dataset('equilibrium/psib', data=np.float64(psisep_cfg))
                if psicore_cfg is not None and np.isfinite(psicore_cfg):
                    f.create_dataset('equilibrium/psi_axis',
                                      data=np.float64(psicore_cfg))

            # Multi-ion metadata.
            sdt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('ion_species/names', data=np.array(ion_names, dtype=object), dtype=sdt)
            ion_elements = [re.sub(r'[+\-\d]+$', '', n) for n in ion_names]
            f.create_dataset('ion_species/elements', data=np.array(ion_elements, dtype=object), dtype=sdt)
            f.create_dataset('ion_species/spec_index', data=np.array(ion_inds, dtype=np.int32))
            f.create_dataset('ion_species/main_ion_spec_index', data=np.array([ion_inds[main_k]], dtype=np.int32))
            f.create_dataset('ion_species/mass_amu', data=ion_mass_amu)
            f.create_dataset('ion_species/charge_state_z', data=ion_charge_state_z)

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
            f.create_dataset('mesh/grad_te_r', data=mesh_grad_te_r)
            f.create_dataset('mesh/grad_te_z', data=mesh_grad_te_z)
            f.create_dataset('mesh/grad_ti_r', data=mesh_grad_ti_r)
            f.create_dataset('mesh/grad_ti_z', data=mesh_grad_ti_z)
            f.create_dataset('mesh/e_r', data=mesh_e_r)
            f.create_dataset('mesh/e_z', data=mesh_e_z)
            f.create_dataset('mesh/e_t', data=mesh_e_t)
            f.create_dataset('mesh/pob', data=mesh_pob)
            f.create_dataset('mesh/dens_n', data=mesh_dens_n)
            f.create_dataset('mesh/temp_n', data=mesh_temp_n)
            for _e, _arr in mesh_dens_n_elt.items():
                f.create_dataset(f'mesh/dens_n_{_e}', data=_arr)
            if heatflux == "sheath":
                # Sheath-limited parallel heat flux estimate from the
                # per-cell plasma state (SOLEDGE3X snapshots carry no heat
                # flux directly): q_par = gamma_sh * n_e * c_s * e * T_e,
                # c_s = sqrt(e (Te + Ti)/m_i). Consumed by fix
                # grain/ablate heating=flux; the OML heating mode does
                # not need it.
                _QE = 1.602176634e-19
                _mi = ion_mass_amu[main_k] * 1.66053906660e-27
                _cs = np.sqrt(_QE * (np.maximum(mesh_temp_e, 0.0)
                                     + np.maximum(mesh_temp_i, 0.0)) / _mi)
                _qp = sheath_gamma * np.maximum(mesh_dens_e, 0.0) * _cs \
                      * _QE * np.maximum(mesh_temp_e, 0.0)
                f.create_dataset('mesh/q_par', data=_qp)
                f.create_dataset('mesh/q_perp', data=np.zeros_like(_qp))
                print(f"mesh/q_par written (sheath-limited, gamma="
                      f"{sheath_gamma}): peak {_qp.max():.3e} W/m^2")
            f.create_dataset('mesh/ions/dens', data=mesh_dens_i_all)
            f.create_dataset('mesh/ions/temp', data=mesh_temp_i_all)
            f.create_dataset('mesh/ions/parr_flow', data=mesh_flow_i_par_all)
            # Per-vertex B-field (matches SOLPS schema). OpenEdge reads
            # these for bmag/bpol/btor without needing to re-derive from
            # the embedded equilibrium.
            f.create_dataset('mesh/vtx_br',   data=mesh_vtx_br)
            f.create_dataset('mesh/vtx_bz',   data=mesh_vtx_bz)
            f.create_dataset('mesh/vtx_bt',   data=mesh_vtx_bt)
            f.create_dataset('mesh/vtx_bmag', data=mesh_vtx_bmag)
            f.create_dataset('mesh/vtx_bpol', data=mesh_vtx_bpol)
            f.create_dataset('mesh/vtx_btor', data=mesh_vtx_bt)
            # Provenance stubs for schema uniformity with SOLPS output.
            f.create_dataset('mesh/tri_source_kind', data=mesh_tri_source_kind)
            f.create_dataset('mesh/wall_gap_mask',   data=mesh_wall_gap_mask)

        # B-field is embedded in plasma.h5 above (br/bt/bz datasets); no
        # separate bfield.h5 is produced.

        print(f"Wrote OpenEdge plasma file: {plasma_out_file}")
        print(f"Detected species: electron=spec0, ions={ion_inds} (Nion={len(ion_inds)}), main ion=spec{ion_inds[main_k]}")

    except Exception as e:
        print(f"Error writing output files: {e}")
        raise


def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        description="Convert SOLEDGE3X run to OpenEdge plasma.h5 (mesh-native, "
                    "reads B and psi from mesh.h5; no separate equilibrium required)."
    )
    p.add_argument("run_path", type=str,
                   help="SOLEDGE3X run directory "
                        "(refParam_raptorX.h5, meshEIRENE.h5, mesh_raptorX.h5, "
                        "mesh.h5, plasma_*.h5).")
    p.add_argument("--plasma-snapshot", required=True, type=str,
                   help="Filename of the plasma snapshot (e.g. plasma_00010.h5 "
                        "or plasmaFinal.h5).")
    p.add_argument("--plasma-out", required=True, type=str)
    p.add_argument("--wall-out", required=True, type=str,
                   help="Output SPARTA wall.surf path.")
    p.add_argument("--main-ion-spec", type=int, default=1,
                   help="SOLEDGE3X species index of the main ion (spec0=electrons; "
                        "typically spec1 for D+). Used to label the 'main ion' "
                        "metadata in plasma.h5.")
    p.add_argument("--gfile", type=str, default=None,
                   help="Optional GEQDSK override for B-field (fallback path).")
    p.add_argument("--equ-file", type=str, default=None,
                   help="Optional .equ override for B-field (fallback path).")
    p.add_argument("--efield", choices=["auto", "zero"], default="auto",
                   help="mesh E field: auto = E=-grad(phi) from real "
                        "zone*/PHI when nonzero, else the field-aligned "
                        "Ohm E_par = -grad_par(pe)/(e ne) - 0.71 "
                        "grad_par(Te)/e (presheath field, zero ExB) "
                        "or all-zero; zero = force zeros")
    p.add_argument("--heatflux", choices=["none", "sheath"], default="none",
                   help="write mesh/q_par as the sheath-limited estimate "
                        "gamma*ne*cs*e*Te (SOLEDGE snapshots carry no flux)")
    p.add_argument("--sheath-gamma", type=float, default=7.0)
    p.add_argument("--geometry", choices=["cart", "axi"], default="cart",
                   help="SPARTA wall.surf slot layout. cart (default): "
                        "x=R, y=Z; pairs with 'boundary o o p'. axi: "
                        "x=Z, y=R; pairs with 'boundary o ao p'.")
    return p


def main():
    args = _build_parser().parse_args()
    base = args.run_path
    interpolate_and_save_plasma_field(
        ref_file=os.path.join(base, "refParam_raptorX.h5"),
        mesh_file=os.path.join(base, "meshEIRENE.h5"),
        bfield_file=os.path.join(base, "mesh_raptorX.h5"),
        data_file=os.path.join(base, args.plasma_snapshot),
        wall_file=None,
        plasma_out_file=args.plasma_out,
        main_ion_spec=args.main_ion_spec,
        use_mesh_wall=True,
        wall_sparta_file=args.wall_out,
        config_file=os.path.join(base, "mesh.h5"),
        equ_file=args.equ_file,
        gfile=args.gfile,
        geometry=args.geometry,
        heatflux=args.heatflux,
        sheath_gamma=args.sheath_gamma,
        efield=args.efield,
    )


if __name__ == "__main__":
    main()
