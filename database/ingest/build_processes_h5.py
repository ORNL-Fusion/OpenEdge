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

# Pull the adf11 text-file parser + element / class tables from adas.py.
# Phase 2 removes the per-element ADAS_Rates_<Z>.h5 intermediate, so
# this ingest now reads adf11 text directly (via parse_adf11) and
# writes straight into the canonical /volume/ schema.
sys.path.insert(0, str(DB / "adas"))
from adas import (                                          # noqa: E402
    parse_adf11, load_ionization_potentials, ip_filename,
    DEFAULT_ELEMENTS, CLASSES,
)


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
    """Parse open-ADAS adf11 text files directly (via parse_adf11) and
    write them under the canonical /volume/rates, /volume/radiation,
    /volume/thresholds schema.  No intermediate per-element HDF5.

    For each element in adas.DEFAULT_ELEMENTS and each class in
    adas.CLASSES (scd, acd, ccd, plt, prb), read
    ${DB}/adas/adf11/<cls>89/<cls>89_<sym>.dat if present.  Required
    classes (scd, acd) missing from an element skip that element.

    Ionization potentials come from
    ${DB}/adas/ionization_potentials/ADAS_ionization_potentials_<Elt>
    when the file exists (hydrogenic isotopes resolve to 'D').
    """
    adf11_dir = DB / "adas" / "adf11"
    ip_dir    = DB / "adas" / "ionization_potentials"

    # class short name -> (top-group "rates" or "radiation", units, year)
    CLASS_META = {
        "scd": ("rates",     "log10(sigma_v) [cm^3/s]",  "89"),
        "acd": ("rates",     "log10(sigma_v) [cm^3/s]",  "89"),
        "ccd": ("rates",     "log10(sigma_v) [cm^3/s]",  "89"),
        "plt": ("radiation", "log10(power) [W cm^3]",    "89"),
        "prb": ("radiation", "log10(power) [W cm^3]",    "89"),
    }

    stats = {"elements": [], "rate_tables": 0, "radiation_tables": 0,
             "ip_vectors": 0}

    for sym, Z in DEFAULT_ELEMENTS.items():
        elem_loaded_any = False
        for class_name, base, coeff_name, required, units in CLASSES:
            top, units_str, year = CLASS_META[class_name]
            adf11_path = adf11_dir / f"{class_name}{year}" / \
                         f"{class_name}{year}_{sym}.dat"
            if not adf11_path.exists():
                if required and not elem_loaded_any:
                    # non-fatal: element is simply not shipped here
                    pass
                continue
            logQ, logDens, logTe, iZmin, iZmax = parse_adf11(adf11_path)
            # adf11 layout: logQ.shape = (nDens, nTe, nZ).  Canonical
            # storage for downstream consumers is (nZ, nTe, nDens) so
            # dims[0] is the charge-state axis.
            coef = np.transpose(logQ, (2, 1, 0))  # -> (nZ, nTe, nDens)
            grp = fout.require_group(f"volume/{top}/{class_name}/{sym}")
            if "coefficient" in grp:
                del grp["coefficient"]
            ds = grp.create_dataset("coefficient", data=coef,
                                    compression="gzip", compression_opts=6)
            ds.attrs["units"]  = units_str
            ds.attrs["source"] = f"open-ADAS adf11 {class_name}{year}"
            ds.attrs["method"] = ("log10 values on "
                                  "(log10 Te [eV], log10 ne [cm^-3])")
            for ax_name, ax_data in (("temperature", logTe),
                                     ("density", logDens)):
                if ax_name in grp:
                    del grp[ax_name]
                grp.create_dataset(ax_name, data=ax_data)
            if top == "rates":
                stats["rate_tables"] += 1
            else:
                stats["radiation_tables"] += 1
            elem_loaded_any = True
        if elem_loaded_any:
            stats["elements"].append((sym, Z))

        # Ionization potentials (optional per element)
        ip_path = ip_dir / ip_filename(sym)
        if ip_path.exists():
            ip_vals = load_ionization_potentials(ip_path, Z)
            tg = fout.require_group(f"volume/thresholds/ionization/{sym}")
            if "energy" in tg:
                del tg["energy"]
            dsi = tg.create_dataset("energy", data=ip_vals)
            dsi.attrs["units"]  = "eV"
            dsi.attrs["source"] = f"open-ADAS ionization_potentials ({ip_path.name})"
            dsi.attrs["method"] = ("scalar per charge state "
                                   "(index = pre-ionization charge q=0,1,...,Z-1)")
            stats["ip_vectors"] += 1

    # Molecular bond energies (currently just D2, hand-curated)
    mol = fout.require_group("volume/thresholds/dissociation")
    if "d2" in mol:
        del mol["d2"]
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


def ingest_volume_pec(fout: h5py.File) -> dict:
    """Copy per-line photon-emission coefficients (PEC) from
    database/pec/*.h5 into /volume/pec/<element>/<line_id>.

    Input file naming: ``<element>_<pec_id>_pec.h5``.  Inside, the
    dataset keyed as ``<ion_stage>_<wavelength>_pec`` holds PEC
    values (photons m^3 s^-1) on a (Te, ne) grid.  We store each
    line's coefficient as its own leaf so synthetic-emission computes
    can request them by (element, line_id) directly.
    """
    grp_root = fout.require_group("volume/pec")
    stats = {"lines": 0}
    for path in sorted((DB / "pec").glob("*_pec.h5")):
        parts = path.stem.split("_")          # e.g. ["w", "4009", "pec"]
        if len(parts) < 3:
            continue
        element = parts[0].lower()
        pec_id  = parts[1]
        with h5py.File(path, "r") as fin:
            te = fin["te"][...] if "te" in fin else None
            ne = fin["ne"][...] if "ne" in fin else None
            for key in fin:
                if not key.endswith("_pec"):
                    continue
                stats["lines"] += 1
                g = grp_root.require_group(f"{element}/{pec_id}/{key}")
                ds = g.create_dataset("coefficient", data=fin[key][...],
                                      compression="gzip", compression_opts=6)
                ds.attrs["units"]  = "photons m^3 s^-1"
                ds.attrs["source"] = f"open-ADAS adf15 (EIRENE distribution) {path.name}"
                ds.attrs["method"] = ("bilinear in log10(Te) x log10(ne); "
                                      "multiply by n_e * n_ion to get "
                                      "photons m^-3 s^-1")
                if te is not None and "temperature" not in g:
                    g.create_dataset("temperature", data=te)
                if ne is not None and "density" not in g:
                    g.create_dataset("density", data=ne)
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
        stats_vol_pec  = ingest_volume_pec(f)
        stats_surf     = ingest_surface_trim(f)

    print(f"\nwrote {OUT} "
          f"({OUT.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  volume.rates tables      : {stats_vol_adas['rate_tables']}")
    print(f"  volume.radiation tables  : {stats_vol_adas['radiation_tables']}")
    print(f"  volume.thresholds vectors: {stats_vol_adas['ip_vectors']}")
    print(f"  volume.reactions files   : {stats_vol_rxn['reaction_files']}")
    print(f"  volume.pec lines         : {stats_vol_pec['lines']}")
    print(f"  surface.{{sputter,reflection}} pairs: {stats_surf['pairs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
