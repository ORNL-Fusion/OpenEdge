#!/usr/bin/env python3
"""
Postprocess OpenEdge CPC benchmark output and plot surface deposition.

Inputs:
  - output/pmi_impacts.csv.rank*   for per-surface PMI outcomes
  - output/tmp.all.surf.cpc        for triangle geometry

Default output:
  - output/openedge_gross_deposition.png

Default plotted quantity:
  gross deposited impurity mass [amu/s]
"""

from __future__ import annotations

import argparse
import csv
import glob
from dataclasses import dataclass
from pathlib import Path

IMPORT_ERROR = None
matplotlib = None
plt = None
np = None
LogNorm = None
Poly3DCollection = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot OpenEdge surface deposition")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Defaults to <script_dir>/output",
    )
    parser.add_argument(
        "--csv-base",
        default=None,
        help="PMI CSV base path. Defaults to <outdir>/pmi_impacts.csv",
    )
    parser.add_argument(
        "--surf-geom",
        default=None,
        help="Surface geometry dump. Defaults to <outdir>/tmp.all.surf.cpc",
    )
    parser.add_argument(
        "--quantity",
        choices=("mass-rate", "mass-per-source", "count"),
        default="mass-rate",
        help="Surface quantity to color by",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0e-10,
        help="Simulation timestep [s]",
    )
    parser.add_argument(
        "--source-particles",
        type=float,
        default=100000.0,
        help="Total injected source particles for per-source normalization",
    )
    parser.add_argument(
        "--w-amu",
        type=float,
        default=183.84,
        help="Tungsten atomic mass in amu",
    )
    parser.add_argument(
        "--keep-ymin-wall",
        action="store_true",
        help="Keep the ymin wall instead of masking it out",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title",
    )
    parser.add_argument(
        "--png",
        default=None,
        help="PNG output path. Defaults to <outdir>/openedge_gross_deposition.png",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Summary output path. Defaults to <outdir>/openedge_gross_deposition_summary.txt",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Saved PNG resolution",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=23.0,
        help="3D camera elevation in degrees",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=238.0,
        help="3D camera azimuth in degrees",
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=True,
        help="Display the figure interactively if supported",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Save PNG only using a non-interactive backend",
    )
    return parser.parse_args()


def ensure_dependencies(show: bool) -> None:
    global IMPORT_ERROR, matplotlib, plt, np, LogNorm, Poly3DCollection

    if plt is not None and np is not None:
        return

    try:
        import matplotlib as _matplotlib
        if not show:
            _matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        import numpy as _np
        from matplotlib.colors import LogNorm as _LogNorm
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _Poly3DCollection
    except ModuleNotFoundError as exc:
        IMPORT_ERROR = exc
    else:
        matplotlib = _matplotlib
        plt = _plt
        np = _np
        LogNorm = _LogNorm
        Poly3DCollection = _Poly3DCollection
        IMPORT_ERROR = None
        return

    missing = []
    if np is None:
        missing.append("numpy")
    if plt is None:
        missing.append("matplotlib")
    missing_str = ", ".join(missing) if missing else str(IMPORT_ERROR)
    raise SystemExit(
        "Missing Python package(s): "
        f"{missing_str}. Install them in your environment and rerun."
    )


@dataclass
class SurfaceGeometry:
    surf_ids: "np.ndarray"
    x: "np.ndarray"
    y: "np.ndarray"
    z: "np.ndarray"
    area: "np.ndarray"
    timestep: int


@dataclass
class SurfaceCounts:
    absorb: "np.ndarray"
    reflect: "np.ndarray"
    sputter: "np.ndarray"


def _find_pmi_files(csv_base: Path) -> list[Path]:
    files = sorted(csv_base.parent.glob(f"{csv_base.name}.rank*"))
    if files:
        return files
    if csv_base.exists():
        return [csv_base]
    return []


def read_last_surface_geometry(path: Path) -> SurfaceGeometry:
    if not path.exists():
        raise FileNotFoundError(f"Could not open {path}")

    lines = path.read_text().splitlines()
    block_starts = [i for i, line in enumerate(lines) if line == "ITEM: TIMESTEP"]
    if not block_starts:
        raise ValueError(f"No surface geometry blocks found in {path}")

    start = block_starts[-1]
    timestep = int(lines[start + 1].strip())

    surf_header_idx = None
    for idx in range(start, len(lines)):
        if lines[idx].startswith("ITEM: SURFS"):
            surf_header_idx = idx
            break
    if surf_header_idx is None:
        raise ValueError(f"Malformed SURFS block in {path}")

    header = lines[surf_header_idx].split()[2:]
    n_surfs = int(lines[start + 3].strip())
    data_lines = lines[surf_header_idx + 1 : surf_header_idx + 1 + n_surfs]
    if len(data_lines) != n_surfs:
        raise ValueError(f"Unexpected EOF while reading {path}")

    data = np.array([[float(tok) for tok in row.split()] for row in data_lines], dtype=float)

    def col(name: str) -> int:
        try:
            return header.index(name)
        except ValueError as exc:
            raise ValueError(f"Column {name} not found in surface dump header") from exc

    surf_ids = data[:, col("id")].astype(int)
    x = data[:, [col("v1x"), col("v2x"), col("v3x")]]
    y = data[:, [col("v1y"), col("v2y"), col("v3y")]]
    z = data[:, [col("v1z"), col("v2z"), col("v3z")]]
    area = data[:, col("area")]
    return SurfaceGeometry(surf_ids=surf_ids, x=x, y=y, z=z, area=area, timestep=timestep)


def read_pmi_surface_counts(csv_base: Path, surf_ids: "np.ndarray") -> SurfaceCounts:
    files = _find_pmi_files(csv_base)
    if not files:
        raise FileNotFoundError(f"No PMI CSV files found for base path {csv_base}")

    id_to_index = {int(sid): i for i, sid in enumerate(surf_ids)}
    absorb = np.zeros(len(surf_ids), dtype=float)
    reflect = np.zeros(len(surf_ids), dtype=float)
    sputter = np.zeros(len(surf_ids), dtype=float)

    for fname in files:
        with open(fname, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                surf_raw = row.get("surf_id")
                outcome_raw = row.get("outcome")
                if surf_raw is None or outcome_raw is None:
                    continue
                surf_raw = surf_raw.strip()
                outcome = outcome_raw.strip()
                if not surf_raw or not outcome:
                    continue
                try:
                    surf_id = int(surf_raw)
                except ValueError:
                    continue
                idx = id_to_index.get(surf_id)
                if idx is None:
                    continue
                if outcome == "A":
                    absorb[idx] += 1.0
                elif outcome == "R":
                    reflect[idx] += 1.0
                elif outcome == "S":
                    sputter[idx] += 1.0

    return SurfaceCounts(absorb=absorb, reflect=reflect, sputter=sputter)


def remove_ymin_wall_mask(y: "np.ndarray") -> "np.ndarray":
    ymin_wall = (
        (np.abs(y[:, 0] + 0.015) < 1.0e-4)
        & (np.abs(y[:, 1] + 0.015) < 1.0e-3)
        & (np.abs(y[:, 2] + 0.015) < 1.0e-3)
    )
    return ~ymin_wall


def compute_quantity(
    quantity: str,
    absorb_count: "np.ndarray",
    physical_time_s: float,
    w_amu: float,
    source_particles: float,
) -> tuple["np.ndarray", str]:
    if quantity == "mass-rate":
        return absorb_count * (w_amu / physical_time_s), "Gross deposited W (amu/s)"
    if quantity == "mass-per-source":
        return absorb_count * (w_amu / source_particles), "Deposited W mass [amu/source particle]"
    return absorb_count.copy(), "Absorbed particle count"


def default_title(quantity: str, physical_time_s: float) -> str:
    if quantity == "mass-rate":
        return f"Gross Deposited W (amu/s), t = {physical_time_s:.3g} s"
    if quantity == "mass-per-source":
        return f"Gross Deposited W Mass, t = {physical_time_s:.3g} s"
    return f"Gross Deposition (PMI absorbs), t = {physical_time_s:.3g} s"


def write_summary(
    path: Path,
    geom: SurfaceGeometry,
    counts: SurfaceCounts,
    physical_time_s: float,
    quantity: str,
    values: "np.ndarray",
    plotted_surfaces: int,
) -> None:
    nonzero = values[np.isfinite(values) & (values > 0.0)]
    total_quantity = float(np.sum(nonzero)) if nonzero.size else 0.0
    lines = [
        f"geometry_dump_timestep: {geom.timestep}",
        f"physical_time_s: {physical_time_s:.12g}",
        f"surfaces_total: {len(geom.surf_ids)}",
        f"surfaces_plotted: {plotted_surfaces}",
        f"total_absorbs: {int(np.sum(counts.absorb))}",
        f"total_reflects: {int(np.sum(counts.reflect))}",
        f"total_sputters: {int(np.sum(counts.sputter))}",
        f"max_surface_absorb: {int(np.max(counts.absorb)) if counts.absorb.size else 0}",
        f"quantity: {quantity}",
        f"total_quantity: {total_quantity:.12g}",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot_surface(
    geom: SurfaceGeometry,
    values: "np.ndarray",
    title: str,
    colorbar_label: str,
    png_path: Path,
    dpi: int,
    elev: float,
    azim: float,
    show: bool,
) -> None:
    finite = np.isfinite(values) & (values > 0.0)
    plot_values = np.where(finite, values, np.nan)
    vertices = np.stack((geom.x, geom.y, geom.z), axis=2)

    fig = plt.figure(figsize=(10.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    # Base translucent mesh so the full geometry stays visible without
    # competing with the deposition colors.
    base_collection = Poly3DCollection(
        vertices,
        linewidths=0.4,
        edgecolors=(0.1, 0.1, 0.1, 0.16),
        facecolors=(0.82, 0.86, 0.91, 0.10),
    )
    ax.add_collection3d(base_collection)

    if finite.any():
        active_vertices = vertices[finite]
        active_values = plot_values[finite]
        collection = Poly3DCollection(
            active_vertices,
            linewidths=0.55,
            edgecolors=(0.05, 0.05, 0.05, 0.22),
            alpha=0.72,
            cmap="viridis",
        )
        vmin = float(np.nanmin(plot_values[finite]))
        vmax = float(np.nanmax(plot_values[finite]))
        collection.set_norm(LogNorm(vmin=vmin, vmax=vmax))
        collection.set_array(np.ma.masked_invalid(active_values))
        ax.add_collection3d(collection)
    else:
        collection = None

    ax.set_xlim(float(np.min(geom.x)), float(np.max(geom.x)))
    ax.set_ylim(float(np.min(geom.y)), float(np.max(geom.y)))
    ax.set_zlim(float(np.min(geom.z)), float(np.max(geom.z)))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(False)
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.set_box_aspect((
        float(np.max(geom.x) - np.min(geom.x)),
        float(np.max(geom.y) - np.min(geom.y)),
        float(np.max(geom.z) - np.min(geom.z)),
    ))

    if collection is not None:
        cbar = fig.colorbar(collection, ax=ax, pad=0.08, shrink=0.72)
        cbar.set_label(colorbar_label)

    fig.tight_layout()
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    args = parse_args()
    ensure_dependencies(args.show)

    script_dir = Path(__file__).resolve().parent
    outdir = Path(args.outdir) if args.outdir else script_dir / "output"
    csv_base = Path(args.csv_base) if args.csv_base else outdir / "pmi_impacts.csv"
    surf_geom = Path(args.surf_geom) if args.surf_geom else outdir / "tmp.all.surf.cpc"
    png_path = Path(args.png) if args.png else outdir / "openedge_gross_deposition.png"
    summary_path = (
        Path(args.summary)
        if args.summary
        else outdir / "openedge_gross_deposition_summary.txt"
    )

    geom = read_last_surface_geometry(surf_geom)
    counts = read_pmi_surface_counts(csv_base, geom.surf_ids)
    physical_time_s = geom.timestep * args.dt
    if physical_time_s <= 0.0:
        raise SystemExit(
            f"Non-positive physical time inferred from timestep {geom.timestep} and dt={args.dt:g}"
        )

    values, colorbar_label = compute_quantity(
        quantity=args.quantity,
        absorb_count=counts.absorb,
        physical_time_s=physical_time_s,
        w_amu=args.w_amu,
        source_particles=args.source_particles,
    )

    if not args.keep_ymin_wall:
        mask = remove_ymin_wall_mask(geom.y)
        geom = SurfaceGeometry(
            surf_ids=geom.surf_ids[mask],
            x=geom.x[mask, :],
            y=geom.y[mask, :],
            z=geom.z[mask, :],
            area=geom.area[mask],
            timestep=geom.timestep,
        )
        counts = SurfaceCounts(
            absorb=counts.absorb[mask],
            reflect=counts.reflect[mask],
            sputter=counts.sputter[mask],
        )
        values = values[mask]

    title = args.title if args.title else default_title(args.quantity, physical_time_s)

    write_summary(
        path=summary_path,
        geom=geom,
        counts=counts,
        physical_time_s=physical_time_s,
        quantity=args.quantity,
        values=values,
        plotted_surfaces=len(geom.surf_ids),
    )

    plot_surface(
        geom=geom,
        values=values,
        title=title,
        colorbar_label=colorbar_label,
        png_path=png_path,
        dpi=args.dpi,
        elev=args.elev,
        azim=args.azim,
        show=args.show,
    )

    total_absorbs = int(np.sum(counts.absorb))
    total_reflects = int(np.sum(counts.reflect))
    total_sputters = int(np.sum(counts.sputter))
    max_surface_absorb = int(np.max(counts.absorb)) if counts.absorb.size else 0
    finite = values[np.isfinite(values) & (values > 0.0)]
    total_quantity = float(np.sum(finite)) if finite.size else 0.0

    print("OpenEdge deposition summary")
    print(f"  Geometry dump timestep: {geom.timestep}")
    print(f"  Physical time [s]:      {physical_time_s:.6g}")
    print(f"  Surfaces plotted:       {len(geom.surf_ids)}")
    print(f"  Total absorbs:          {total_absorbs}")
    print(f"  Total reflects:         {total_reflects}")
    print(f"  Total sputters:         {total_sputters}")
    print(f"  Max surface absorb:     {max_surface_absorb}")
    print(f"  Total {args.quantity}:         {total_quantity:.6g}")
    print(f"  Saved {png_path}")
    print(f"  Saved {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
