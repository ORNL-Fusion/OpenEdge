import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import h5py

def parse_tmp_surf(filename, start_col=1, end_col=None):
    """
    Parse a OpenEdge surf dump of the form:
      ITEM: TIMESTEP
      <t>
      ITEM: NUMBER OF SURFS
      <N>
      ITEM: BOX BOUNDS ...
      ...
      ITEM: SURFS id f_save[1] f_save[2] ...
      <id> <val1> <val2> ...

    Parameters
    ----------
    filename : str
    start_col : int
        1-based index of the first f_save column to extract (i.e., 1 -> f_save[1]).
    end_col : int or None
        1-based index of the last f_save column to extract. If None, inferred from header.

    Returns
    -------
    dict
        results[timestep] = [ids] + [vals_for_fsave_start, vals_for_fsave_start+1, ...]
        where each list has length N (number of surfs at that timestep).
    """
    results = {}
    with open(filename, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]

    i = 0
    L = len(lines)
    while i < L:
        if lines[i].startswith("ITEM: TIMESTEP"):
            # TIMESTEP
            if i + 1 >= L:
                break
            timestep = int(lines[i + 1])
            i += 2

            # NUMBER OF SURFS
            while i < L and not lines[i].startswith("ITEM: NUMBER OF SURFS"):
                i += 1
            if i + 1 >= L:
                break
            n_surfs = int(lines[i + 1])
            i += 2

            # skip BOX BOUNDS block (3 lines after the header)
            while i < L and not lines[i].startswith("ITEM: BOX BOUNDS"):
                i += 1
            if i < L and lines[i].startswith("ITEM: BOX BOUNDS"):
                i += 1 + 3  # header + 3 bound lines

            # SURFS header
            while i < L and not lines[i].startswith("ITEM: SURFS"):
                i += 1
            if i >= L:
                break

            header = lines[i].split()
            # header looks like: ["ITEM:", "SURFS", "id", "f_save[1]", "f_save[2]", ...]
            # find how many f_save columns exist
            # Everything after 'id' is treated as a data column
            try:
                id_index = header.index("id")
            except ValueError:
                # Some dumps put 'id' implicitly as the first column; assume 0
                id_index = 2  # conservative fallback

            # Number of data columns (excluding the leading 'ITEM: SURFS ...')
            # Realistically, dataparsing uses the data lines, so infer from them too.
            # We'll set end_col if not provided after reading the first data line.
            i += 1  # move to first data row

            ids = []
            fsave_cols = None
            selected = None

            # Prepare containers after we know how many columns the data lines have
            # We’ll peek at the first line (if present)
            if i < L:
                first_data = lines[i].split()
                # Data layout per row: [id, f_save[1], f_save[2], ...]
                # So the total f_save count = len(first_data) - 1
                total_fsave = max(0, len(first_data) - 1)
                if end_col is None:
                    end = total_fsave
                else:
                    end = end_col
                if start_col < 1 or end < start_col:
                    raise ValueError(f"Invalid column range: start_col={start_col}, end_col={end_col}")
                # clamp to available
                end = min(end, total_fsave)
                fsave_cols = list(range(start_col, end + 1))
                selected = [[] for _ in fsave_cols]

            # Read N surf lines (or until next ITEM)
            read_count = 0
            while i < L and read_count < n_surfs:
                if lines[i].startswith("ITEM: "):
                    break
                parts = lines[i].split()
                if len(parts) >= 1:
                    try:
                        sid = int(parts[0])
                    except ValueError:
                        sid = None
                    if sid is not None and fsave_cols is not None:
                        ids.append(sid)
                        # f_save[k] is at parts[k], since parts[0] is id and f_save[1] is parts[1]
                        for idx, k in enumerate(fsave_cols):
                            if k < len(parts):
                                try:
                                    selected[idx].append(float(parts[k]))
                                except ValueError:
                                    selected[idx].append(float("nan"))
                            else:
                                selected[idx].append(float("nan"))
                        read_count += 1
                i += 1

            # Store results for this timestep
            if selected is None:
                results[timestep] = [ids]
            else:
                results[timestep] = [ids] + selected

            # continue loop without skipping the next ITEM line if we stopped early
            continue

        # advance if not at a TIMESTEP block
        i += 1

    return results


import re
import numpy as np

def _unwrap_midpoint(a, b, L, periodic):
    """Midpoint along one axis with optional minimum-image unwrapping."""
    if not periodic or not np.isfinite(L) or L <= 0:
        return 0.5*(a + b)
    d = b - a
    d -= np.round(d / L) * L
    return a + 0.5*d

def parse_tmp_surf(filename, start_col=1, end_col=None):
    """
    Parse an OpenEdge/SPARTA-style surf dump and also compute surf centers (xc,yc).

    Returns
    -------
    dict
        results[timestep] = [ids, fsave[start], ..., fsave[end], xc, yc]
        where each inner list has length N (number of surfs at that timestep).
        If vertices aren't present, xc,yc are filled with NaN.
    """
    results = {}
    with open(filename, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]

    i = 0
    L = len(lines)
    while i < L:
        if not lines[i].startswith("ITEM: TIMESTEP"):
            i += 1
            continue

        # TIMESTEP
        if i + 1 >= L:
            break
        timestep = int(lines[i + 1])
        i += 2

        # NUMBER OF SURFS
        while i < L and not lines[i].startswith("ITEM: NUMBER OF SURFS"):
            i += 1
        if i + 1 >= L:
            break
        n_surfs = int(lines[i + 1])
        i += 2

        # BOX BOUNDS (+ periodic flags)
        xlo = xhi = ylo = yhi = None
        px = py = False
        while i < L and not lines[i].startswith("ITEM: BOX BOUNDS"):
            i += 1
        if i < L and lines[i].startswith("ITEM: BOX BOUNDS"):
            # Parse flags like "ITEM: BOX BOUNDS pp pp pp"
            parts = lines[i].split()
            # flags may be at the end; tolerate missing flags
            flags = parts[3:6] if len(parts) >= 6 else []
            px = (len(flags) >= 1 and flags[0] == "pp")
            py = (len(flags) >= 2 and flags[1] == "pp")
            # read 3 bound lines (x,y,z)
            if i + 3 < L:
                xlo, xhi = map(float, lines[i+1].split()[:2])
                ylo, yhi = map(float, lines[i+2].split()[:2])
                # z bounds present but unused for 2D
            i += 4  # header + 3 lines

        Lx = (xhi - xlo) if (xlo is not None and xhi is not None) else np.nan
        Ly = (yhi - ylo) if (ylo is not None and yhi is not None) else np.nan

        # SURFS header
        while i < L and not lines[i].startswith("ITEM: SURFS"):
            i += 1
        if i >= L:
            break

        header_tokens = lines[i].split()
        # Strip leading ["ITEM:", "SURFS"]
        if len(header_tokens) >= 2 and header_tokens[0] == "ITEM:" and header_tokens[1] == "SURFS":
            data_cols = header_tokens[2:]
        else:
            # Fallback: assume everything after the first two tokens are column names
            data_cols = header_tokens[2:] if len(header_tokens) > 2 else []

        col_index = {name: idx for idx, name in enumerate(data_cols)}

        # Identify columns
        id_idx = col_index.get("id", 0)  # default to first column if 'id' missing

        # Vertex columns (optional; present for 2D segments or 3-vertex polygons)
        v1x_idx = col_index.get("v1x", None)
        v1y_idx = col_index.get("v1y", None)
        v2x_idx = col_index.get("v2x", None)
        v2y_idx = col_index.get("v2y", None)
        v3x_idx = col_index.get("v3x", None)
        v3y_idx = col_index.get("v3y", None)

        # f_save-like columns: match names ending with [k]; sort by k
        fsave_named_cols = []
        for name, idx in col_index.items():
            m = re.search(r"\[(\d+)\]$", name)  # captures the trailing index
            if m:
                fsave_named_cols.append((int(m.group(1)), idx))
        fsave_named_cols.sort(key=lambda x: x[0])  # sort by the numeric bracket index

        # Determine which f_save indices to keep (1-based start_col..end_col)
        if fsave_named_cols:
            max_fsave = fsave_named_cols[-1][0]
            end = max_fsave if end_col is None else end_col
            if start_col < 1 or end < start_col:
                raise ValueError(f"Invalid column range: start_col={start_col}, end_col={end_col}")
            end = min(end, max_fsave)
            wanted_range = set(range(start_col, end + 1))
            fsave_keep = [(k, idx) for (k, idx) in fsave_named_cols if k in wanted_range]
            # keep in increasing k order
            fsave_keep.sort(key=lambda x: x[0])
        else:
            # Legacy format: assume columns after 'id' are f_save
            # We'll infer the count from the first data line below.
            fsave_keep = None

        # Move to first data row
        i += 1

        # Prepare accumulators
        ids = []
        selected = None  # list of lists for chosen f_save columns
        xc_list, yc_list = [], []

        # If legacy f_save layout, peek at the first row to decide range
        if fsave_keep is None and i < L and not lines[i].startswith("ITEM: "):
            first_parts = lines[i].split()
            ncols = len(first_parts)
            # Estimate how many columns correspond to data_cols
            # If we have header names, use them; otherwise assume [id] + fsave...
            if data_cols:
                # f_save are any columns except 'id' and vertices (v?x/v?y/v?z)
                def is_vertex(name): return name.startswith("v") and name[2:] in ("x", "y", "z")
                fsave_guess = [j for j, name in enumerate(data_cols)
                               if name != "id" and not is_vertex(name)]
                # Treat them as a contiguous block in input order
                fsave_keep = [(k+1, j) for k, j in enumerate(fsave_guess)]  # fake 1..K
            else:
                # No header names: treat columns 1..(ncols-1) as f_save[1..]
                total_fsave = max(0, ncols - 1)
                end = total_fsave if end_col is None else min(end_col, total_fsave)
                fsave_keep = [(k, k) for k in range(1, end + 1)]  # (k, idx=k) since parts[0] is id

        # Build empty containers for f_save selection
        selected = [[] for _ in range(len(fsave_keep))] if fsave_keep is not None else []

        # Read N surf lines
        read_count = 0
        while i < L and read_count < n_surfs:
            if lines[i].startswith("ITEM: "):
                break
            parts = lines[i].split()
            if not parts:
                i += 1
                continue

            # Basic fields
            try:
                sid = int(parts[id_idx])
            except Exception:
                sid = None
            if sid is None:
                i += 1
                continue
            ids.append(sid)

            # f_save extraction
            for list_idx, (_, col_idx_real) in enumerate(fsave_keep):
                try:
                    selected[list_idx].append(float(parts[col_idx_real]))
                except Exception:
                    selected[list_idx].append(float("nan"))

            # Centers: 2D segment midpoint, or triangle centroid if v3 exists
            def _getf(idx):
                try:
                    return float(parts[idx]) if idx is not None else float("nan")
                except Exception:
                    return float("nan")

            v1x = _getf(v1x_idx); v1y = _getf(v1y_idx)
            v2x = _getf(v2x_idx); v2y = _getf(v2y_idx)
            v3x = _getf(v3x_idx); v3y = _getf(v3y_idx)

            if np.isfinite(v1x) and np.isfinite(v1y) and np.isfinite(v2x) and np.isfinite(v2y):
                # If triangle present, use triangle centroid; else segment midpoint
                if np.isfinite(v3x) and np.isfinite(v3y):
                    xc_raw = (v1x + v2x + v3x) / 3.0
                    yc_raw = (v1y + v2y + v3y) / 3.0
                    xc = xc_raw if not px else ((xc_raw - xlo) % Lx + xlo) if np.isfinite(Lx) else xc_raw
                    yc = yc_raw if not py else ((yc_raw - ylo) % Ly + ylo) if np.isfinite(Ly) else yc_raw
                else:
                    # Minimum-image midpoint for periodic boxes
                    mx = _unwrap_midpoint(v1x, v2x, Lx, px)
                    my = _unwrap_midpoint(v1y, v2y, Ly, py)
                    # Wrap back into box extent if periodic
                    xc = ((mx - xlo) % Lx + xlo) if px and np.isfinite(Lx) else mx
                    yc = ((my - ylo) % Ly + ylo) if py and np.isfinite(Ly) else my
            else:
                xc = yc = float("nan")

            xc_list.append(xc)
            yc_list.append(yc)

            read_count += 1
            i += 1

        # Store results
        # results[t] = [ids] + chosen f_save columns + [xc, yc]
        if selected is None:
            results[timestep] = [ids, xc_list, yc_list]
        else:
            results[timestep] = [ids] + selected + [xc_list, yc_list]

        # continue loop (don't skip potential next ITEM)
        continue

    return results

import numpy as np

# ---------- 1) Pull arrays from your parsed dict ----------
def unpack_surf_timestep(data, t):
    """
    data[t] = [ids, fsave1, fsave2, ..., xc, yc]
    returns: ids, fsave_cols(list of arrays), xc, yc
    """
    row = data[t]
    ids = np.asarray(row[0], dtype=int)
    xc  = np.asarray(row[-2], dtype=float)
    yc  = np.asarray(row[-1], dtype=float)
    fsave_cols = [np.asarray(col, dtype=float) for col in row[1:-2]]
    return ids, fsave_cols, xc, yc

# ---------- 2) Build arclength for your wall vertices ----------
def wall_arclength(vv_r, vv_z):
    """
    vv_r, vv_z : arrays of wall vertices (length M)
    returns:
      s_nodes : arclength at each vertex (length M)
      s_seg   : arclength at each segment midpoint (length M-1)
      Rmid, Zmid : segment midpoints (length M-1)
    """
    vv_r = np.asarray(vv_r, float)
    vv_z = np.asarray(vv_z, float)
    dr = np.diff(vv_r)
    dz = np.diff(vv_z)
    seg_len = np.sqrt(dr*dr + dz*dz)
    s_nodes = np.zeros_like(vv_r)
    s_nodes[1:] = np.cumsum(seg_len)
    s_seg = 0.5*(s_nodes[:-1] + s_nodes[1:])
    Rmid = 0.5*(vv_r[:-1] + vv_r[1:])
    Zmid = 0.5*(vv_z[:-1] + vv_z[1:])
    return s_nodes, s_seg, Rmid, Zmid

# ---------- 3) Project points (xc,yc) onto the polyline to get s ----------
def project_points_onto_polyline(x, y, xv, yv, s_nodes):
    """
    For each point (x,y), find closest point on polyline defined by (xv,yv),
    return arclength coordinate s_closest (same length as x).
    Complexity O(N*M) which is fine for a few thousand.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    xv = np.asarray(xv, float); yv = np.asarray(yv, float)

    # Precompute per-segment vectors and lengths
    sx = xv[1:] - xv[:-1]
    sy = yv[1:] - yv[:-1]
    L2 = sx*sx + sy*sy
    L = np.sqrt(L2)
    s0 = s_nodes[:-1]  # arclength at start of each segment

    s_out = np.empty_like(x)
    for i in range(x.size):
        px = x[i] - xv[:-1]
        py = y[i] - yv[:-1]
        # projection parameter t on each segment (clamped to [0,1])
        # handle zero-length segments safely
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (px*sx + py*sy) / L2
        t = np.clip(np.where(L2 > 0, t, 0.0), 0.0, 1.0)
        cx = xv[:-1] + t*sx
        cy = yv[:-1] + t*sy
        d2 = (x[i]-cx)**2 + (y[i]-cy)**2
        j = np.argmin(d2)
        s_out[i] = s0[j] + t[j]*L[j]
    return s_out

# ---------- 4) Make a 1D profile along wall and (optionally) resample ----------
def wall_profile_from_surfs(vv_r, vv_z, xc, yc, values, s_samples=None):
    """
    Map surf-centered 'values' to arclength and optionally resample.
    Returns:
      s_surf, v_surf  (sorted by s)
      and if s_samples is given: v_interp at s_samples
    """
    s_nodes, s_seg, Rmid, Zmid = wall_arclength(vv_r, vv_z)
    s_surf = project_points_onto_polyline(xc, yc, vv_r, vv_z, s_nodes)
    # sort along arclength
    order = np.argsort(s_surf)
    s_surf = s_surf[order]
    v_surf = np.asarray(values, float)[order]

    if s_samples is None:
        return s_surf, v_surf
    # 1D linear interpolation, with edge fill by nearest
    v_interp = np.interp(s_samples, s_surf, v_surf, left=v_surf[0], right=v_surf[-1])
    return s_surf, v_surf, v_interp



def align_by_common_ids(data_dict):
    """Return sorted timesteps, sorted common_ids, and a 3D array vals[t,i,q]."""
    timesteps = sorted(data_dict.keys())
    # find IDs present at all timesteps
    id_sets = [set(data_dict[t][0]) for t in timesteps]
    common_ids = sorted(set.intersection(*id_sets))
    if not common_ids:
        raise ValueError("No common surface IDs across timesteps; can’t align.")

    # number of quantities (f_save columns)
    n_q = len(data_dict[timesteps[0]]) - 1

    # build vals[t, i, q]
    T = len(timesteps)
    I = len(common_ids)
    vals = np.full((T, I, n_q), np.nan, dtype=float)

    id_to_idx = {sid: k for k, sid in enumerate(common_ids)}
    for ti, t in enumerate(timesteps):
        ids = data_dict[t][0]
        # map this timestep's ids to indices
        local_map = {sid: i for i, sid in enumerate(ids)}
        for sid in common_ids:
            j_global = id_to_idx[sid]
            j_local = local_map[sid]
            for q in range(n_q):
                arr = data_dict[t][q+1]  # +1 because [0] is ids
                vals[ti, j_global, q] = arr[j_local]
    return timesteps, np.array(common_ids), vals

def plot_profiles_over_id(
    timesteps, ids, vals, names=None, sample=8, use_log=('nflux_incident','etot')
):
    """
    Plot profiles y(id) for multiple times with SI labels.

    Parameters
    ----------
    timesteps : (T,) array-like
        Usually step indices or times [s]. Used for colorbar.
    ids : (I,) array-like
        Surface IDs (x-axis).
    vals : (T, I, Q) ndarray
        Values for each quantity q at each id and time.
    names : list[str] or None
        Names of the Q quantities in vals' last axis. Examples:
        ["nflux_incident","mflux","etot","press"].
    sample : int
        Number of evenly spaced timesteps to plot (plus the last 5).
    use_log : iterable[str]
        Quantities (by name) to show on log-y if present.
    """
    vals = np.asarray(vals)
    T, I, Q = vals.shape

    # default names if not provided
    if names is None or len(names) != Q:
        names = [f"q{j}" for j in range(Q)]

    # SI units mapping (edit as needed)
    units = {
        "nflux_incident": r"m$^{-2}$ s$^{-1}$",  # particle flux Γ
        "mflux":          r"N m$^{-2}$",         # momentum flux (= Pa)
        "etot":           r"W m$^{-2}$",         # energy flux
        "press":          r"Pa",                 # pressure
    }

    # pick times: uniform subset + last few (always include last)
    base_idx = np.linspace(0, T-1, num=min(sample, T), dtype=int)
    tail     = np.arange(max(0, T-5), T, dtype=int)
    picked   = sorted(set(base_idx.tolist() + tail.tolist()))
    cmap = get_cmap("viridis")
    norm = Normalize(vmin=float(timesteps[0]), vmax=float(timesteps[-1]))

    fig, axes = plt.subplots(Q, 1, figsize=(10, 2.6*Q + 2), sharex=True)
    if Q == 1:
        axes = [axes]

    # plot each quantity
    for q in range(Q):
        ax = axes[q]
        name = names[q]
        for ti in picked:
            color = cmap(norm(timesteps[ti]))
            lw    = 1.25 if ti != (T-1) else 2.8
            alpha = 0.85 if ti == (T-1) else 0.6
            # only label a couple of curves to keep legend small
            label = None
            if q == 0 and (ti in (picked[0], picked[-1], T-1)):
                label = f"t={timesteps[ti]}"
            ax.plot(ids, vals[ti, :, q], color=color, lw=lw, alpha=alpha, label=label)

        # y label with SI
        unit = units.get(name, "")
        ax.set_ylabel(f"{name} [{unit}]".strip())
        ax.grid(True, alpha=0.3, linestyle="--")

        # optional log scale for some positive-definite quantities
        if name in use_log:
            # only set log if all positive over picked to avoid warnings
            ypick = vals[picked, :, q]
            if np.all(ypick > 0):
                ax.set_yscale("log")

    axes[-1].set_xlabel("Surface ID")

    # time colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", pad=0.01, fraction=0.03)
    cbar.set_label("Timestep")  # change to "Time [s]" if timesteps are seconds

    # compact legend on first axis if any labels exist
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="best", frameon=False, ncols=3)

    fig.suptitle("Profiles over surface ID across time (bold = final)")
    fig.tight_layout()
    return fig, axes


def compute_convergence_metrics(vals):
    """
    Max over IDs of absolute change between successive timesteps, per quantity.
    Returns delta[T-1, Q], where delta[t] compares t and t-1 (delta[0]=nan).
    """
    diffs = np.abs(np.diff(vals, axis=0))  # shape: (T-1, I, Q)
    delta_max = np.nanmax(diffs, axis=1)   # (T-1, Q)
    # prepend a row of NaNs for the first timestep
    delta_max = np.vstack([np.full((1, vals.shape[2]), np.nan), delta_max])
    return delta_max  # (T, Q)

def plot_convergence(timesteps, delta_max, eps=None, M=5):
    """
    Plot max|Δ| vs time for each quantity. Optionally show an epsilon threshold.
    If eps provided, also shade the region where the last M steps are below eps.
    """
    T, Q = delta_max.shape
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    for q in range(Q):
        ax.plot(timesteps, delta_max[:, q], label=f"max|Δ| f_save[{q+1}]")
    ax.set_yscale("log")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Max change across IDs")
    ax.grid(True, which="both", alpha=0.3)
    if eps is not None:
        ax.axhline(eps, ls="--", lw=1, color="k", alpha=0.6, label=f"ε={eps:g}")
        # steady detection: last M steps all below eps
        if T >= M and np.all(delta_max[-M:, :] < eps):
            ax.axvspan(timesteps[-M], timesteps[-1], color="green", alpha=0.08, label=f"steady last {M} steps")
    ax.legend(frameon=False, ncols=min(3, Q+1))
    ax.set_title("Convergence: max change across IDs vs time (log scale)")
    fig.tight_layout()
    return fig, ax


import numpy as np
import h5py
# from scipy.spatial import cKDTree  # not used in this path

# --- unpack surf dump ---
data = parse_tmp_surf("tmp.all.surf", start_col=1, end_col=None)
tmax = max(data.keys())
ids, fcols, xc, yc = unpack_surf_timestep(data, tmax)

# --- wall vertices (vv_r, vv_z) ---
vv_r, vv_z = np.loadtxt("input/vv_values.csv", unpack=True)

def wall_arclength(vr, vz):
    vr = np.asarray(vr, float); vz = np.asarray(vz, float)
    dr, dz = np.diff(vr), np.diff(vz)
    seg = np.sqrt(dr*dr + dz*dz)
    s_nodes = np.zeros_like(vr); s_nodes[1:] = np.cumsum(seg)
    rmid = 0.5*(vr[:-1] + vr[1:]); zmid = 0.5*(vz[:-1] + vz[1:])
    smid = 0.5*(s_nodes[:-1] + s_nodes[1:])
    return s_nodes, rmid, zmid, smid

def project_points_to_s(x, y, vr, vz, s_nodes):
    vr = np.asarray(vr); vz = np.asarray(vz)
    sx = vr[1:] - vr[:-1]; sy = vz[1:] - vz[:-1]
    L2 = sx*sx + sy*sy
    s0 = s_nodes[:-1]
    out = np.empty_like(x, dtype=float)
    for i, (px, py) in enumerate(zip(x, y)):
        dx = px - vr[:-1]; dy = py - vz[:-1]
        t = (dx*sx + dy*sy) / np.where(L2 > 0, L2, 1.0)
        t = np.clip(t, 0.0, 1.0)
        cx = vr[:-1] + t*sx; cy = vz[:-1] + t*sy
        j = np.argmin((px - cx)**2 + (py - cy)**2)
        out[i] = s0[j] + t[j]*np.sqrt(L2[j])
    return out

# --- flux_oe.h5 ---
with h5py.File("input/flux_oe.h5", "r") as h5:
    r = np.asarray(h5["r"]).reshape(-1)
    z = np.asarray(h5["z"]).reshape(-1)
    F = np.asarray(h5["eroded_flux"]).reshape(-1)  # [m^-2 s^-1]

# 1) arclength geometry
s_nodes, rmid, zmid, smid = wall_arclength(vv_r, vv_z)

# 2) project both datasets to arclength
s_surf = project_points_to_s(xc, yc, vv_r, vv_z, s_nodes)
s_flux = project_points_to_s(r,  z,  vv_r, vv_z, s_nodes)

# 3) sort flux by s and 1D interpolate to surf positions
ordf = np.argsort(s_flux)
sF, F_sorted = s_flux[ordf], F[ordf]
Gamma_emit_by_surf = np.interp(s_surf, sF, F_sorted, left=F_sorted[0], right=F_sorted[-1])

# 4) diagnostics
print("s range [m]:", float(s_nodes[0]), float(s_nodes[-1]))
dss = np.diff(np.sort(s_surf))
print("surf spacing median [m]:", float(np.median(dss[dss>0])))

# 5) optional: write ID-indexed file your emitter can read directly
import h5py
with h5py.File("input/flux_by_surf.h5", "w") as h5w:
    h5w.create_dataset("surf_id", data=ids, compression="gzip")
    h5w.create_dataset("Gamma_emit", data=Gamma_emit_by_surf, compression="gzip")
print("wrote: input/flux_by_surf.h5")


import matplotlib.pyplot as plt
# s_surf from your script; Gamma_emit_by_surf as computed
order = s_surf.argsort()
plt.figure(figsize=(9,3.6))
plt.plot(s_surf[order], Gamma_emit_by_surf[order], '-', lw=1.2)
plt.xlabel('Wall arclength s [m]'); plt.ylabel('Γ_emit [m$^{-2}$ s$^{-1}$]')
plt.tight_layout(); plt.show()


import numpy as np
from scipy.spatial import cKDTree

# segment lengths from wall vertices
dr, dz = np.diff(vv_r), np.diff(vv_z)
Lseg = np.sqrt(dr*dr + dz*dz)          # length of each wall segment
s_nodes = np.zeros_like(vv_r); s_nodes[1:] = np.cumsum(Lseg)
s_mid   = 0.5*(s_nodes[:-1] + s_nodes[1:])

# map each surf to nearest segment midpoint -> use that segment length
j = cKDTree(np.c_[0.5*(vv_r[:-1]+vv_r[1:]), 0.5*(vv_z[:-1]+vv_z[1:])]).query(np.c_[xc,yc])[1]
A_surf = Lseg[j]                        # "area" per-surf in 2D

# totals
Gamma_emit = Gamma_emit_by_surf         # [m^-2 s^-1]
R_phys = float(np.sum(Gamma_emit * A_surf))   # [s^-1] real particles per second
dt   = 1e-8      # your timestep
fnum = 5e05      # your particle weight
N_step = R_phys * dt / fnum
print(f"Real emission rate ≈ {R_phys:.3e} 1/s")
print(f"Expected inserts per step ≈ {N_step:.3f} particles/step")
print(f"Expected inserts per second ≈ {R_phys/fnum:.3e} sim-particles/s")


#
#exit()
## map file order to dump order
#order = np.argsort(sid_file)
#sid_sorted = sid_file[order]
#Gamma_emit_sorted = Gamma_emit_file[order]
#idx = np.searchsorted(sid_sorted, ids)
#Gamma_emit = Gamma_emit_sorted[idx]
#
## 3) optional: compute arclength along wall for pretty plotting
#def wall_arclength(vv_r, vv_z):
#    dr = np.diff(vv_r); dz = np.diff(vv_z)
#    seg = np.sqrt(dr*dr + dz*dz)
#    s_nodes = np.zeros_like(vv_r); s_nodes[1:] = np.cumsum(seg)
#    # project surf centers to arclength (nearest segment midpoint proxy)
#    Rmid = 0.5*(vv_r[:-1] + vv_r[1:]); Zmid = 0.5*(vv_z[:-1] + vv_z[1:])
#    s_seg = 0.5*(s_nodes[:-1] + s_nodes[1:])
#    # nearest segment by distance
#    from scipy.spatial import cKDTree
#    tree = cKDTree(np.c_[Rmid, Zmid])
#    _, j = tree.query(np.c_[xc, yc])
#    s_surf = s_seg[j]
#    return s_surf
#
## supply your wall vertex arrays:
## vv_r, vv_z = ...
## s = wall_arclength(vv_r, vv_z)
#s = np.arange(len(ids))  # fallback: index if you don't have vertices here
#
## 4) compare profiles
#plt.figure(figsize=(9,4))
#plt.plot(s, Gamma_inc, '.', ms=3, label='incident Γ_inc (dump)')
#plt.plot(s, Gamma_emit, '.', ms=3, label='emit Γ_emit (file)')
#plt.xlabel('wall arclength s [m] (or index)')
#plt.ylabel('flux [m$^{-2}$ s$^{-1}$]')
#plt.legend(); plt.tight_layout(); plt.show()
#
## If you know Y at each surf, you can overlay Y * Gamma_inc too:
## plt.plot(s, Y * Gamma_inc, '-', lw=1.5, label='Y * Γ_inc (expected)')
