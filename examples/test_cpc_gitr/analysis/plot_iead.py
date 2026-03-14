#!/usr/bin/env python3
"""
Plot the ion energy-angle distribution (IEAD) for the CPC benchmark.

The script supports two input sources:
  1. `output/iead.dat` if a direct IEAD file exists
  2. `output/pmi_impacts.csv.rank*` by reconstructing the IEAD from
     the per-impact PMI log columns `E_in_eV` and `theta_deg`

Outputs:
  - output/iead_2d.png
  - output/iead_1d_energy.png
  - output/iead_1d_angle.png
  - output/iead_summary.txt

Examples:
  python3 plot_iead.py
  python3 plot_iead.py --source pmi --species W+ W2+
  python3 plot_iead.py --source iead --input output/iead.dat
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import glob
import math
from pathlib import Path

IMPORT_ERROR = None
matplotlib = None
plt = None
LogNorm = None
np = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot IEAD for test_cpc_gitr")
    parser.add_argument(
        "--source",
        choices=("auto", "pmi", "iead"),
        default="auto",
        help="Input source: auto-detect, PMI CSV logs, or direct iead.dat",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input file/prefix. Defaults: output/iead.dat or output/pmi_impacts.csv",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=None,
        help="Optional species filter, e.g. --species W+ W2+",
    )
    parser.add_argument(
        "--outcomes",
        nargs="+",
        default=None,
        help="Optional PMI outcome filter, e.g. --outcomes A R S",
    )
    parser.add_argument(
        "--nbins-e",
        type=int,
        default=100,
        help="Number of energy bins",
    )
    parser.add_argument(
        "--nbins-a",
        type=int,
        default=90,
        help="Number of angle bins",
    )
    parser.add_argument(
        "--emax",
        type=float,
        default=None,
        help="Energy axis max in eV; default is inferred from data",
    )
    parser.add_argument(
        "--amax",
        type=float,
        default=90.0,
        help="Angle axis max in degrees",
    )
    parser.add_argument(
        "--outdir",
        default="output",
        help="Directory for plot outputs",
    )
    parser.add_argument(
        "--species-breakdown",
        action="store_true",
        help="Also write charge-resolved energy plots and summary from PMI logs",
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        default=True,
        help="Display the plot window if a GUI backend is available",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Save PNG output only using a non-interactive backend",
    )
    return parser.parse_args()


def ensure_dependencies(show: bool) -> None:
    global IMPORT_ERROR, matplotlib, plt, LogNorm, np

    if plt is not None and np is not None:
        return

    try:
        import matplotlib as _matplotlib
        if not show:
            _matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        from matplotlib.colors import LogNorm as _LogNorm
        import numpy as _np
    except ModuleNotFoundError as exc:
        IMPORT_ERROR = exc
    else:
        matplotlib = _matplotlib
        plt = _plt
        LogNorm = _LogNorm
        np = _np
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
        f"{missing_str}. Install them in your OpenEdge environment and rerun."
    )


def _nice_upper_bound(values: np.ndarray, floor: float) -> float:
    vmax = float(np.nanmax(values)) if values.size else floor
    vmax = max(vmax, floor)
    target = 1.05 * vmax
    exponent = math.floor(math.log10(target)) if target > 0 else 0
    scale = 10.0 ** exponent
    return math.ceil(target / scale) * scale


def _find_pmi_files(prefix: str) -> list[str]:
    files = sorted(glob.glob(f"{prefix}.rank*"))
    if files:
        return files
    path = Path(prefix)
    if path.exists():
        return [str(path)]
    return []


def _iter_pmi_samples(prefix: str):
    files = _find_pmi_files(prefix)
    if not files:
        raise FileNotFoundError(f"No PMI files found matching {prefix}.rank*")

    for fname in files:
        with open(fname, newline="") as fh:
            reader = csv.DictReader(fh)
            required = {"species", "E_in_eV", "theta_deg", "outcome"}
            if not required.issubset(reader.fieldnames or set()):
                raise ValueError(f"{fname} is missing required PMI IEAD columns")
            for row in reader:
                species_raw = row.get("species")
                outcome_raw = row.get("outcome")
                energy_raw = row.get("E_in_eV")
                angle_raw = row.get("theta_deg")

                # Some rank files can end with a truncated line; skip incomplete rows.
                if species_raw is None or outcome_raw is None:
                    continue
                if energy_raw is None or angle_raw is None:
                    continue

                species = species_raw.strip()
                outcome = outcome_raw.strip()
                if not species or not outcome:
                    continue

                try:
                    energy = float(energy_raw)
                    angle = float(angle_raw)
                except ValueError:
                    continue

                yield species, outcome, energy, angle


def read_pmi_logs(
    prefix: str,
    species_filter: set[str] | None,
    outcome_filter: set[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energies: list[float] = []
    angles: list[float] = []
    weights: list[float] = []

    for species, outcome, energy, angle in _iter_pmi_samples(prefix):
        if species_filter and species not in species_filter:
            continue
        if outcome_filter and outcome not in outcome_filter:
            continue
        energies.append(energy)
        angles.append(angle)
        weights.append(1.0)

    if not energies:
        raise ValueError("No IEAD samples remain after filtering PMI logs")

    return np.array(energies), np.array(angles), np.array(weights)


def _species_sort_key(species: str) -> tuple[int, str]:
    if species == "W":
        return (0, species)
    if species.startswith("W") and species.endswith("+"):
        charge_text = species[1:-1]
        charge = int(charge_text) if charge_text else 1
        return (charge, species)
    return (10**6, species)


def group_pmi_by_species(
    prefix: str,
    species_filter: set[str] | None,
    outcome_filter: set[str] | None,
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for species, outcome, energy, _angle in _iter_pmi_samples(prefix):
        if species_filter and species not in species_filter:
            continue
        if outcome_filter and outcome not in outcome_filter:
            continue
        grouped[species].append(energy)

    if not grouped:
        raise ValueError("No PMI samples remain for species breakdown after filtering")

    return {
        species: np.array(grouped[species], dtype=float)
        for species in sorted(grouped, key=_species_sort_key)
    }


def _numeric_tokens(line: str) -> list[float]:
    vals: list[float] = []
    for token in line.replace(",", " ").split():
        try:
            vals.append(float(token))
        except ValueError:
            return []
    return vals


def read_iead_file(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            numeric = _numeric_tokens(stripped)
            if len(numeric) >= 2:
                rows.append(numeric)

    if not rows:
        raise ValueError(f"No numeric IEAD data found in {path}")

    min_cols = min(len(row) for row in rows)
    data = np.array([row[:min_cols] for row in rows], dtype=float)

    energy = data[:, 0]
    angle = data[:, 1]
    if np.nanmax(energy) <= 90.0 and np.nanmax(angle) > 90.0:
        energy, angle = angle, energy

    if min_cols >= 3:
        weights = data[:, 2]
    else:
        weights = np.ones_like(energy)

    mask = np.isfinite(energy) & np.isfinite(angle) & np.isfinite(weights)
    mask &= weights > 0.0
    if not np.any(mask):
        raise ValueError(f"No valid IEAD rows found in {path}")
    return energy[mask], angle[mask], weights[mask]


def load_samples(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    source = args.source
    iead_path = args.input if (source == "iead" and args.input) else "output/iead.dat"
    pmi_prefix = args.input if (source == "pmi" and args.input) else "output/pmi_impacts.csv"

    species_filter = set(args.species) if args.species else None
    outcome_filter = set(args.outcomes) if args.outcomes else None

    if source in ("auto", "iead") and Path(iead_path).exists():
        energy, angle, weights = read_iead_file(iead_path)
        return energy, angle, weights, f"iead:{iead_path}"

    if source in ("auto", "pmi"):
        energy, angle, weights = read_pmi_logs(pmi_prefix, species_filter, outcome_filter)
        return energy, angle, weights, f"pmi:{pmi_prefix}"

    if source == "iead":
        raise FileNotFoundError(f"IEAD file not found: {iead_path}")
    raise FileNotFoundError(
        "Could not find output/iead.dat, and no PMI logs were found to reconstruct the IEAD"
    )


def build_histogram(
    energy: np.ndarray,
    angle: np.ndarray,
    weights: np.ndarray,
    emax: float,
    amax: float,
    nbins_e: int,
    nbins_a: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e_edges = np.linspace(0.0, emax, nbins_e + 1)
    a_edges = np.linspace(0.0, amax, nbins_a + 1)
    hist, _, _ = np.histogram2d(
        angle,
        energy,
        bins=[a_edges, e_edges],
        weights=weights,
    )
    return hist, e_edges, a_edges


def plot_2d(hist: np.ndarray, e_edges: np.ndarray, a_edges: np.ndarray, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    h_plot = np.ma.masked_where(hist <= 0.0, hist)
    vmax = max(float(hist.max()), 1.0)
    im = ax.pcolormesh(
        e_edges,
        a_edges,
        h_plot,
        norm=LogNorm(vmin=1.0, vmax=vmax),
        cmap="plasma",
        shading="auto",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Counts per bin")
    ax.set_xlabel("Energy [eV]")
    ax.set_ylabel("Angle [deg]")
    ax.set_xlim(e_edges[0], e_edges[-1])
    ax.set_ylim(a_edges[0], a_edges[-1])
    ax.set_title("Ion energy-angle distribution")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return



def plot_1d_energy(hist: np.ndarray, e_edges: np.ndarray, out_png: Path) -> None:
    e_centers = 0.5 * (e_edges[:-1] + e_edges[1:])
    e_dist = hist.sum(axis=0)
    total = e_dist.sum()
    if total > 0.0:
        e_dist = e_dist / (total * (e_edges[1] - e_edges[0]))

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(e_centers, e_dist, lw=2.0)
    ax.set_xlabel("Energy [eV]")
    ax.set_ylabel("PDF [1/eV]")
    ax.set_xlim(e_edges[0], e_edges[-1])
    ax.set_title("IEAD energy marginal")
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_1d_angle(hist: np.ndarray, a_edges: np.ndarray, out_png: Path) -> None:
    a_centers = 0.5 * (a_edges[:-1] + a_edges[1:])
    a_dist = hist.sum(axis=1)
    total = a_dist.sum()
    if total > 0.0:
        a_dist = a_dist / (total * (a_edges[1] - a_edges[0]))

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(a_centers, a_dist, lw=2.0)
    ax.set_xlabel("Angle [deg]")
    ax.set_ylabel("PDF [1/deg]")
    ax.set_xlim(a_edges[0], a_edges[-1])
    ax.set_title("IEAD angle marginal")
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_species_energy(grouped: dict[str, np.ndarray], emax: float, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins = np.linspace(0.0, emax, 100 + 1)

    for species in sorted(grouped, key=_species_sort_key):
        energies = grouped[species]
        if energies.size == 0:
            continue
        ax.hist(
            energies,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=f"{species}  N={energies.size}",
        )

    ax.set_xlabel("Energy [eV]")
    ax.set_ylabel("PDF [1/eV]")
    ax.set_xlim(0.0, emax)
    ax.set_title("IEAD energy by species")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")


def write_species_summary(grouped: dict[str, np.ndarray], out_txt: Path) -> None:
    total = sum(values.size for values in grouped.values())
    lines = [
        "species,count,fraction,mean_energy_eV,std_energy_eV,min_energy_eV,max_energy_eV",
    ]
    for species in sorted(grouped, key=_species_sort_key):
        energies = grouped[species]
        fraction = energies.size / total if total > 0 else 0.0
        lines.append(
            f"{species},{energies.size},{fraction:.6f},"
            f"{float(np.mean(energies)):.6f},{float(np.std(energies)):.6f},"
            f"{float(np.min(energies)):.6f},{float(np.max(energies)):.6f}"
        )
    out_txt.write_text("\n".join(lines) + "\n")


def write_summary(
    energy: np.ndarray,
    angle: np.ndarray,
    weights: np.ndarray,
    hist: np.ndarray,
    source_label: str,
    out_txt: Path,
) -> None:
    total_weight = float(np.sum(weights))
    mean_energy = float(np.average(energy, weights=weights))
    mean_angle = float(np.average(angle, weights=weights))
    lines = [
        f"source: {source_label}",
        f"samples: {len(energy)}",
        f"total_weight: {total_weight:.6g}",
        f"mean_energy_eV: {mean_energy:.6f}",
        f"mean_angle_deg: {mean_angle:.6f}",
        f"energy_min_eV: {float(np.min(energy)):.6f}",
        f"energy_max_eV: {float(np.max(energy)):.6f}",
        f"angle_min_deg: {float(np.min(angle)):.6f}",
        f"angle_max_deg: {float(np.max(angle)):.6f}",
        f"nonzero_bins: {int(np.count_nonzero(hist))}",
    ]
    out_txt.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    ensure_dependencies(args.show)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    energy, angle, weights, source_label = load_samples(args)

    emax = args.emax if args.emax is not None else _nice_upper_bound(energy, floor=10.0)
    hist, e_edges, a_edges = build_histogram(
        energy=energy,
        angle=angle,
        weights=weights,
        emax=emax,
        amax=args.amax,
        nbins_e=args.nbins_e,
        nbins_a=args.nbins_a,
    )

    plot_2d(hist, e_edges, a_edges, outdir / "iead_2d.png")
    plot_1d_energy(hist, e_edges, outdir / "iead_1d_energy.png")
    plot_1d_angle(hist, a_edges, outdir / "iead_1d_angle.png")
    write_summary(energy, angle, weights, hist, source_label, outdir / "iead_summary.txt")

    if args.species_breakdown:
        if source_label.startswith("pmi:"):
            pmi_prefix = args.input if (args.source == "pmi" and args.input) else "output/pmi_impacts.csv"
            species_filter = set(args.species) if args.species else None
            outcome_filter = set(args.outcomes) if args.outcomes else None
            grouped = group_pmi_by_species(pmi_prefix, species_filter, outcome_filter)
            plot_species_energy(grouped, emax, outdir / "iead_species_energy.png")
            write_species_summary(grouped, outdir / "iead_species_summary.txt")
            print(f"Saved {outdir / 'iead_species_energy.png'}")
            print(f"Saved {outdir / 'iead_species_summary.txt'}")
        else:
            print("Species breakdown skipped: direct iead.dat input does not contain species labels.")

    if args.show:
        plt.show()
    else:
        plt.close("all")

    print(f"Loaded {len(energy)} IEAD samples from {source_label}")
    print(f"Saved {outdir / 'iead_2d.png'}")
    print(f"Saved {outdir / 'iead_1d_energy.png'}")
    print(f"Saved {outdir / 'iead_1d_angle.png'}")
    print(f"Saved {outdir / 'iead_summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
