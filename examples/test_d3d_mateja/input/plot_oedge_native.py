#!/usr/bin/env python3
"""
Plot OEDGE/DIVIMP fields on the native polygon mesh.

This avoids interpolating cell-centred data onto a regular (R, Z) grid,
which can smear across the separatrix, X-point, and private-flux regions.

Examples
--------
python plot_oedge_native.py /Users/42d/OEGDE/d3d-204953-bkg-v25.nc --field KVHS
python plot_oedge_native.py /Users/42d/OEGDE/d3d-204953-bkg-v25.nc --field ring
python plot_oedge_native.py /Users/42d/OEGDE/d3d-204953-bkg-v25.nc --field KVHS --ring 70
python plot_oedge_native.py /Users/42d/OEGDE/d3d-204953-bkg-v25.nc --list-fields
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

try:
    import netCDF4 as nc
except ImportError:
    nc = None

try:
    import h5py
except ImportError:
    h5py = None


SPECIAL_FIELDS = {"ring", "ir", "knot", "ik"}
FLOW_FIELDS = {"KVHS", "OKVHS"}


def _variables(ds: Any) -> Any:
    return ds.variables if hasattr(ds, "variables") else ds


def open_dataset(path: Path) -> Any:
    if nc is not None:
        return nc.Dataset(str(path), "r")
    if h5py is not None:
        return h5py.File(str(path), "r")
    raise RuntimeError("Neither netCDF4 nor h5py is available.")


def _read_scalar(ds: Any, name: str) -> int | float:
    vars_ = _variables(ds)
    item = vars_[name]
    try:
        data = item[:]
    except (TypeError, ValueError):
        data = item[()]
    return np.asarray(data).squeeze().item()


def _read_array(ds: Any, name: str) -> np.ndarray:
    vars_ = _variables(ds)
    item = vars_[name]
    try:
        data = item[:]
    except TypeError:
        data = item[()]
    return np.asarray(data)


def list_cell_fields(ds: Any) -> list[str]:
    vars_ = _variables(ds)
    korpg_shape = vars_["KORPG"].shape
    names = []
    for name, var in vars_.items():
        if getattr(var, "shape", None) == korpg_shape:
            names.append(name)
    return sorted(names)


def detect_polygon_index_offset(korpg: np.ndarray, nvertp: np.ndarray) -> int:
    positive = korpg[np.isfinite(korpg) & (korpg > 0)].astype(int)
    if positive.size == 0:
        raise RuntimeError("No positive KORPG entries found.")

    n_poly = len(nvertp)
    if positive.min() >= 1 and positive.max() <= n_poly:
        return 1
    if positive.min() >= 0 and positive.max() < n_poly:
        return 0

    # Fall back to the more common Fortran-style 1-based indexing.
    return 1


def extract_native_cells(ds: Any, field_name: str) -> dict[str, np.ndarray | list[np.ndarray] | float | str]:
    korpg = _read_array(ds, "KORPG")
    nvertp = _read_array(ds, "NVERTP")
    rvertp = _read_array(ds, "RVERTP")
    zvertp = _read_array(ds, "ZVERTP")
    rs = _read_array(ds, "RS")
    zs = _read_array(ds, "ZS")

    offset = detect_polygon_index_offset(korpg, nvertp)

    field_arr = None
    qtim = None
    display_name = field_name
    if field_name not in SPECIAL_FIELDS:
        if field_name not in _variables(ds):
            fields = ", ".join(list_cell_fields(ds))
            raise KeyError(f"Field '{field_name}' not found. Cell fields: {fields}")
        field_arr = _read_array(ds, field_name)
        if field_arr.shape != korpg.shape:
            raise ValueError(
                f"Field '{field_name}' has shape {field_arr.shape}, expected {korpg.shape}."
            )
        if field_name in FLOW_FIELDS:
            if "QTIM" not in _variables(ds):
                raise KeyError(f"Field '{field_name}' requires QTIM, but QTIM was not found.")
            qtim = float(_read_scalar(ds, "QTIM"))
            if qtim <= 0.0:
                raise RuntimeError(f"Invalid OEDGE QTIM={qtim}")
            field_arr = field_arr / qtim
            display_name = f"{field_name}/QTIM [m/s]"

    polygons: list[np.ndarray] = []
    values = []
    rings = []
    knots = []
    centers_r = []
    centers_z = []

    nr, nk = korpg.shape
    for ir in range(nr):
        for ik in range(nk):
            raw_pid = int(korpg[ir, ik])
            if raw_pid <= 0:
                continue

            pid = raw_pid - offset
            if pid < 0 or pid >= len(nvertp):
                continue

            nvert = int(nvertp[pid])
            if nvert < 3:
                continue

            poly = np.column_stack((rvertp[pid, :nvert], zvertp[pid, :nvert]))
            if not np.isfinite(poly).all():
                continue

            ring_id = ir + 1
            knot_id = ik + 1

            if field_name in {"ring", "ir"}:
                value = ring_id
            elif field_name in {"knot", "ik"}:
                value = knot_id
            else:
                value = float(field_arr[ir, ik])
                if not np.isfinite(value):
                    continue

            polygons.append(poly)
            values.append(value)
            rings.append(ring_id)
            knots.append(knot_id)
            centers_r.append(float(rs[ir, ik]))
            centers_z.append(float(zs[ir, ik]))

    if not polygons:
        raise RuntimeError(f"No valid polygons found for field '{field_name}'.")

    return {
        "polygons": polygons,
        "values": np.asarray(values, dtype=float),
        "rings": np.asarray(rings, dtype=int),
        "knots": np.asarray(knots, dtype=int),
        "r": np.asarray(centers_r, dtype=float),
        "z": np.asarray(centers_z, dtype=float),
        "qtim": qtim,
        "display_name": display_name,
    }


def choose_limits(values: np.ndarray, symmetric: bool | None) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0

    if symmetric is None:
        symmetric = np.nanmin(finite) < 0.0 < np.nanmax(finite)

    if symmetric:
        vlim = np.nanpercentile(np.abs(finite), 98.0)
        vlim = max(vlim, 1e-12)
        return -vlim, vlim

    vmin = np.nanpercentile(finite, 2.0)
    vmax = np.nanpercentile(finite, 98.0)
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin), 1.0) * 1e-6
        vmin -= pad
        vmax += pad
    return vmin, vmax


def plot_native_mesh(
    nc_file: Path,
    field_name: str,
    output: Path | None,
    ring: int | None,
    show_centers: bool,
    symmetric: bool | None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    with open_dataset(nc_file) as ds:
        data = extract_native_cells(ds, field_name)
        vars_ = _variables(ds)
        irsep = int(_read_scalar(ds, "IRSEP")) if "IRSEP" in vars_ else None
        irwall = int(_read_scalar(ds, "IRWALL")) if "IRWALL" in vars_ else None
        irtrap = int(_read_scalar(ds, "IRTRAP")) if "IRTRAP" in vars_ else None
        ksb = _read_array(ds, "KSB") if "KSB" in vars_ else None

    values = data["values"]
    label = str(data["display_name"])
    cmap = "RdBu_r" if (symmetric or (symmetric is None and np.nanmin(values) < 0.0 < np.nanmax(values))) else "viridis"
    vmin, vmax = choose_limits(values, symmetric)

    has_profile = ring is not None and field_name not in SPECIAL_FIELDS
    if has_profile:
        fig, (ax_mesh, ax_prof) = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    else:
        fig, ax_mesh = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
        ax_prof = None

    coll = PolyCollection(
        data["polygons"],
        array=values,
        cmap=cmap,
        edgecolors="none",
        linewidths=0.0,
    )
    coll.set_clim(vmin, vmax)
    ax_mesh.add_collection(coll)
    ax_mesh.autoscale_view()
    ax_mesh.set_aspect("equal")
    ax_mesh.set_xlabel("R [m]")
    ax_mesh.set_ylabel("Z [m]")
    title = f"{label} on native OEDGE polygons"
    if irsep is not None:
        title += f"\nIRSEP={irsep}"
        if irwall is not None:
            title += f", IRWALL={irwall}"
        if irtrap is not None:
            title += f", IRTRAP={irtrap}"
    ax_mesh.set_title(title)
    cbar = fig.colorbar(coll, ax=ax_mesh, shrink=0.9)
    cbar.set_label(label)

    if show_centers:
        ax_mesh.scatter(data["r"], data["z"], s=0.2, c="k", alpha=0.25)

    if ring is not None:
        mask = data["rings"] == ring
        ax_mesh.scatter(data["r"][mask], data["z"][mask], s=3.0, c="white", alpha=0.9)

    if has_profile and ax_prof is not None:
        mask = data["rings"] == ring
        if not np.any(mask):
            raise RuntimeError(f"Ring {ring} was not found in the native cell list.")

        knot = data["knots"][mask]
        order = np.argsort(knot)
        y = values[mask][order]
        x = knot[order].astype(float)
        xlabel = "knot index"

        ring_idx = ring - 1
        knot_idx = knot[order] - 1
        if ksb is not None and 0 <= ring_idx < ksb.shape[0]:
            s_centers = 0.5 * (ksb[ring_idx, :-1] + ksb[ring_idx, 1:])
            if np.all((0 <= knot_idx) & (knot_idx < len(s_centers))):
                x = s_centers[knot_idx]
                xlabel = "parallel coordinate s [m]"

        ax_prof.plot(x, y, "-o", ms=2.5, lw=1.0)
        ax_prof.set_xlabel(xlabel)
        ax_prof.set_ylabel(label)
        ax_prof.set_title(f"Ring {ring} profile")
        ax_prof.grid(True, alpha=0.3)

    print(f"File: {nc_file}")
    print(f"Field: {field_name}")
    if data["qtim"] is not None:
        print(f"QTIM: {float(data['qtim']):.6e} s")
        print("Scaling: plotted KVHS/QTIM in m/s")
    print(f"Cells plotted: {len(values)}")
    print(f"R range: {np.nanmin(data['r']):.4f} .. {np.nanmax(data['r']):.4f} m")
    print(f"Z range: {np.nanmin(data['z']):.4f} .. {np.nanmax(data['z']):.4f} m")
    print(f"Value range: {np.nanmin(values):.6g} .. {np.nanmax(values):.6g}")

    if output is not None:
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved {output}")
    else:
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot an OEDGE field on the native polygon mesh."
    )
    parser.add_argument("nc_file", type=Path, help="OEDGE background NetCDF file")
    parser.add_argument(
        "--field",
        default="KVHS",
        help="Native cell field to plot, or one of: ring, knot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write plot to file instead of opening a window",
    )
    parser.add_argument(
        "--ring",
        type=int,
        default=None,
        help="Also plot a 1D profile along the selected ring",
    )
    parser.add_argument(
        "--show-centers",
        action="store_true",
        help="Overlay cell centers on the mesh plot",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="Force symmetric color limits around zero",
    )
    parser.add_argument(
        "--list-fields",
        action="store_true",
        help="Print all native cell fields with KORPG shape and exit",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    with open_dataset(args.nc_file) as ds:
        if args.list_fields:
            print("Special fields: ring, knot")
            print("NetCDF fields on the native cell mesh:")
            for name in list_cell_fields(ds):
                print(name)
            return

    plot_native_mesh(
        nc_file=args.nc_file,
        field_name=args.field,
        output=args.output,
        ring=args.ring,
        show_centers=args.show_centers,
        symmetric=True if args.symmetric else None,
    )


if __name__ == "__main__":
    main()
