#!/usr/bin/env python3
"""Consolidate all OpenEdge atomic / surface data into database/processes.h5.

Reads existing per-element / per-pair files and re-packs them under the
canonical /volume/ + /surface/ schema (see docs/database_schema.md).
Idempotent: overwrites the target each run.  Preserves the legacy
per-element files for now (consumers read them as fallback).

Named after what it contains (elementary volume + surface PROCESSES)
rather than after the consumer code, because the data originates from
external sources (open-ADAS, TRIM, literature) -- we curate, we do not
author.

Usage:
  python3 database/ingest/build_processes_h5.py

Output:
  database/processes.h5

Attributes required on every leaf dataset are `units`, `source`,
`method`.  The root carries `schema_version`, `generated`, `sources`.
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DB   = ROOT / "database"
OUT  = DB / "processes.h5"

SCHEMA_VERSION = "1.0"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def z_from_filename(name: str) -> int:
    # ADAS_Rates_<Z>.h5 -> <Z>
    return int(name.split("_")[-1].split(".")[0])


ELEMENT_SYMBOLS = {
    1:"h", 2:"he", 3:"li", 4:"be", 5:"b", 6:"c", 7:"n", 8:"o",
    10:"ne", 18:"ar", 26:"fe", 36:"kr", 42:"mo", 54:"xe", 73:"ta", 74:"w",
}


def copy_dataset(src: h5py.Dataset, dst_group: h5py.Group, name: str,
                 units: str, source: str, method: str) -> None:
    """Copy a dataset with required metadata attrs."""
    ds = dst_group.create_dataset(name, data=src[...],
                                  compression="gzip", compression_opts=6)
    ds.attrs["units"]  = units
    ds.attrs["source"] = source
    ds.attrs["method"] = method


def ingest_volume_adas(fout: h5py.File) -> dict:
    """Copy scd / acd / ccd / plt / prb / ionization_potentials for every
    ADAS_Rates_<Z>.h5 under /volume/rates, /volume/radiation,
    /volume/thresholds."""
    rates_root     = fout.require_group("volume/rates")
    radiation_root = fout.require_group("volume/radiation")
    thresh_root    = fout.require_group("volume/thresholds")

    rate_sources = [
        ("IonizationRateCoeff",       "Ionization",      "rates", "scd",
         "log10(sigma_v) [cm^3/s]"),
        ("RecombinationRateCoeff",    "Recombination",   "rates", "acd",
         "log10(sigma_v) [cm^3/s]"),
        ("ChargeExchangeRateCoeff",   "ChargeExchange",  "rates", "ccd",
         "log10(sigma_v) [cm^3/s]"),
        ("LineRadiationPowerCoeff",   "LineRadiation",   "radiation", "plt",
         "log10(power) [W cm^3]"),
        ("RecombRadiationPowerCoeff", "RecombRadiation", "radiation", "prb",
         "log10(power) [W cm^3]"),
    ]

    stats = {"elements": [], "rate_tables": 0, "radiation_tables": 0,
             "ip_vectors": 0}
    for path in sorted(DB.glob("adas/ADAS_Rates_*.h5")):
        Z = z_from_filename(path.name)
        sym = ELEMENT_SYMBOLS.get(Z)
        if sym is None:
            print(f"[skip] Z={Z} not in ELEMENT_SYMBOLS map", file=sys.stderr)
            continue
        stats["elements"].append((sym, Z))

        with h5py.File(path, "r") as fin:
            for ds_name, grid_tag, top_group, short, units in rate_sources:
                if ds_name not in fin:
                    continue
                # target group: /volume/<top_group>/<short>/<sym>/
                grp = fout.require_group(f"volume/{top_group}/{short}/{sym}")
                copy_dataset(fin[ds_name], grp, "coefficient",
                             units=units,
                             source=f"open-ADAS adf11 {short}89",
                             method="log10 values on "
                                    "(log10 Te [eV], log10 ne [cm^-3])")
                # companion grid axes
                for axis in ("Temperature", "Density", "ChargeState"):
                    key = f"grid{axis}_{grid_tag}"
                    if key in fin:
                        ax_name = axis.lower()
                        if ax_name in grp:
                            del grp[ax_name]
                        grp.create_dataset(ax_name, data=fin[key][...])
                if top_group == "rates":
                    stats["rate_tables"] += 1
                else:
                    stats["radiation_tables"] += 1

            # Ionization potentials -> /volume/thresholds/ionization/<sym>
            if "IonizationPotential" in fin:
                tg = fout.require_group(f"volume/thresholds/ionization/{sym}")
                copy_dataset(fin["IonizationPotential"], tg, "energy",
                             units="eV",
                             source="open-ADAS ionization-potential file",
                             method="scalar per charge state "
                                    "(index = pre-ionization charge)")
                stats["ip_vectors"] += 1

    # Molecular bond energies (currently just D2, hand-curated)
    mol = fout.require_group("volume/thresholds/dissociation")
    d2 = mol.create_dataset("d2", data=np.array([4.478]))
    d2.attrs["units"]  = "eV"
    d2.attrs["source"] = "Huber & Herzberg 1979, molecular-spectra tables"
    d2.attrs["method"] = "scalar bond dissociation energy D2 -> 2D"
    return stats


def ingest_reactions(fout: h5py.File) -> dict:
    """Ingest per-element reaction catalogs (plain text in
    database/adas/reactions/) as opaque string datasets under
    /volume/reactions/<sym>/."""
    grp_root = fout.require_group("volume/reactions")
    stats = {"reaction_files": 0}
    for path in sorted(DB.glob("adas/reactions/*.reactions")):
        sym = path.stem.lower()
        text = path.read_text()
        grp = fout.require_group(f"volume/reactions/{sym}")
        if "catalog" in grp:
            del grp["catalog"]
        ds = grp.create_dataset("catalog", data=text)
        ds.attrs["units"]  = "-"
        ds.attrs["source"] = f"database/adas/reactions/{path.name}"
        ds.attrs["method"] = ("plain-text reaction list; columns parsed "
                              "by fix_chem_adas::readfile")
        stats["reaction_files"] += 1
    return stats


def ingest_surface_trim(fout: h5py.File) -> dict:
    """Copy every TRIM sputter / reflection table into /surface/sputter/ and
    /surface/reflection/, keyed by <projectile>_on_<target>.

    Schema-group names reflect the physical process (sputter, reflection),
    not the data source (TRIM) or the data type (yields).  Fix names
    follow the same convention: fix surface/emit/sputter reads /sputter/,
    surf_collide reflect-style consumers read /reflection/, etc.
    """
    grp_sputter = fout.require_group("surface/sputter")
    grp_refl    = fout.require_group("surface/reflection")
    stats = {"pairs": 0}
    for path in sorted((DB / "surface" / "trim").glob("*.h5")):
        pair = path.stem.lower()  # e.g. "d_on_w"
        stats["pairs"] += 1
        with h5py.File(path, "r") as fin:
            # Sputter-side: E, theta (yields themselves are implicit in
            # the outgoing distribution -- consumers compute Y from the
            # moments).  Kept here so a single pair-group has the (E,
            # theta) binning.
            s = grp_sputter.require_group(pair)
            for k in ("E", "theta"):
                if k in fin and k not in s:
                    s.create_dataset(k, data=fin[k][...])
            # Reflection: R_N, Eout_*, cos_polar_q, cos_azim_q, polar_*
            r = grp_refl.require_group(pair)
            for k in ("R_N", "Eout_max", "Eout_min", "Eout_q",
                      "cos_polar_q", "cos_azim_q",
                      "polar_max", "polar_min", "raar", "E", "theta"):
                if k in fin and k not in r:
                    r.create_dataset(k, data=fin[k][...])
            # Tag metadata at the pair-level group
            for tgt in (s, r):
                tgt.attrs["source"] = f"TRIM table {path.name}"
                tgt.attrs["method"] = ("binned in (E, theta); outgoing "
                                       "distribution moments stored as 'q' "
                                       "quantiles")
                if "units" not in tgt.attrs:
                    tgt.attrs["units"] = "E [eV], theta [rad], R_N [-]"
    return stats


def main(argv=None) -> int:
    now = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    sha = git_sha()

    print(f"building {OUT}")
    print(f"  root   : {ROOT}")
    print(f"  version: {SCHEMA_VERSION}")
    print(f"  commit : {sha}")

    with h5py.File(OUT, "w") as f:
        f.attrs["schema_version"]  = SCHEMA_VERSION
        f.attrs["generated"]       = now
        f.attrs["openedge_commit"] = sha
        f.attrs["sources"]         = json.dumps([
            "open-ADAS adf11 (scd/acd/ccd/plt/prb year 89)",
            "TRIM yield tables (pre-computed per projectile/target pair)",
            "In-repo reaction catalogs (database/adas/reactions/)",
        ])

        stats_vol_adas = ingest_volume_adas(f)
        stats_vol_rxn  = ingest_reactions(f)
        stats_surf     = ingest_surface_trim(f)

    print(f"\nwrote {OUT} "
          f"({OUT.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  volume.rates tables      : {stats_vol_adas['rate_tables']}")
    print(f"  volume.radiation tables  : {stats_vol_adas['radiation_tables']}")
    print(f"  volume.thresholds vectors: {stats_vol_adas['ip_vectors']}")
    print(f"  volume.reactions files   : {stats_vol_rxn['reaction_files']}")
    print(f"  surface.{{sputter,reflection}} pairs: {stats_surf['pairs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
