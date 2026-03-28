#!/usr/bin/env python3
"""Build transfer-matrix CSV files from SPARTA collision tally dumps."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

WALL_REGIONS = {
    "lfs_wall": list(range(1, 50)) + list(range(151, 157)),
    "up_outer_div": list(range(50, 64)),
    "crown": list(range(64, 83)),
    "up_inner_div": list(range(83, 91)),
    "hfs_wall": list(range(91, 110)),
    "lo_inner_div": list(range(110, 118)),
    "dome": list(range(118, 137)),
    "lo_outer_div": list(range(137, 151)),
}

DEST_ORDER = [
    "lfs_wall",
    "up_outer_div",
    "crown",
    "up_inner_div",
    "hfs_wall",
    "lo_inner_div",
    "dome",
    "lo_outer_div",
    "core",
]

_surf_to_region = {sid: name for name, ids in WALL_REGIONS.items() for sid in ids}


def parse_collision_events(path: Path, source_name: str):
    """
    Parse compute: id id/surf time
    -> columns [1]=particle_id, [2]=surf_id, [3]=collision_time
    """
    events = []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() != "ITEM: TIMESTEP":
            i += 1
            continue

        timestep = int(lines[i + 1].strip())
        i += 2

        if lines[i].strip() != "ITEM: NUMBER OF TALLIES":
            raise ValueError(f"{path}: malformed NUMBER OF TALLIES near step {timestep}")
        ntally = int(lines[i + 1].strip())
        i += 2

        if not lines[i].startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"{path}: missing BOX BOUNDS near step {timestep}")
        i += 4

        if not lines[i].startswith("ITEM: TALLIES"):
            raise ValueError(f"{path}: missing TALLIES near step {timestep}")
        i += 1

        for _ in range(ntally):
            parts = lines[i].split()
            if len(parts) < 3:
                raise ValueError(f"{path}: expected >=3 tally fields near step {timestep}")
            events.append(
                {
                    "source": source_name,
                    "timestep": timestep,
                    "particle_id": int(float(parts[0])),
                    "surf_id": int(float(parts[1])),
                    "collision_time": float(parts[2]),
                }
            )
            i += 1

    return events


def parse_surf_running_dump(path: Path):
    """
    Parse dump surf file with columns: id f_f_hit_I[1] f_f_hit_O[1] (running averages).
    Returns counts_by_source, surf_ids, last_ts.
    """
    lines = path.read_text().splitlines()
    i = 0
    last_ts = None
    last_rows = []
    last_cols = []

    while i < len(lines):
        if lines[i].strip() != "ITEM: TIMESTEP":
            i += 1
            continue
        ts = int(lines[i + 1].strip())
        i += 2

        if lines[i].strip() != "ITEM: NUMBER OF SURFS":
            raise ValueError(f"{path}: malformed NUMBER OF SURFS near step {ts}")
        nsurf = int(lines[i + 1].strip())
        i += 2

        if not lines[i].startswith("ITEM: BOX BOUNDS"):
            raise ValueError(f"{path}: missing BOX BOUNDS near step {ts}")
        i += 4

        if not lines[i].startswith("ITEM: SURFS"):
            raise ValueError(f"{path}: missing SURFS header near step {ts}")
        cols = lines[i].split()[2:]
        i += 1

        rows = []
        for _ in range(nsurf):
            parts = lines[i].split()
            if len(parts) != len(cols):
                raise ValueError(f"{path}: bad SURFS row near step {ts}")
            rows.append(parts)
            i += 1

        if ts > 0:
            last_ts = ts
            last_rows = rows
            last_cols = cols

    if last_ts is None:
        return {"I": Counter(), "O": Counter()}, [], 0

    # Map columns
    id_idx = None
    i_idx = None
    o_idx = None
    for k, c in enumerate(last_cols):
        if c == "id":
            id_idx = k
        elif "f_f_hit_I" in c:
            i_idx = k
        elif "f_f_hit_O" in c:
            o_idx = k
    if id_idx is None or i_idx is None or o_idx is None:
        raise ValueError(
            f"{path}: expected columns 'id', 'f_f_hit_I[...]', 'f_f_hit_O[...]' in ITEM: SURFS"
        )

    counts_by_source = {"I": Counter(), "O": Counter()}
    surf_ids = []
    for row in last_rows:
        sid = int(float(row[id_idx]))
        surf_ids.append(sid)
        i_val = float(row[i_idx])
        o_val = float(row[o_idx])
        counts_by_source["I"][sid] = i_val * last_ts
        counts_by_source["O"][sid] = o_val * last_ts

    return counts_by_source, sorted(set(surf_ids)), last_ts


def parse_particle_first_hits(path: Path, source_name: str):
    """
    Parse particle dump with columns including:
      id, p_hit_flag, p_hit_surf_id
    Returns:
      counts (Counter of first-hit surf IDs),
      events (list[dict]),
      launched_detected (# unique particle IDs seen),
      last_ts (max timestep in file)
    """
    lines = path.read_text().splitlines()
    i = 0
    cols = None
    first_hit = {}
    seen_ids = set()
    last_ts = 0

    while i < len(lines):
        s = lines[i].strip()
        if s == "ITEM: TIMESTEP":
            last_ts = int(lines[i + 1].strip())
            i += 2
            continue
        if s.startswith("ITEM: ATOMS"):
            cols = s.split()[2:]
            i += 1
            continue
        if s.startswith("ITEM:"):
            i += 1
            continue
        if not s or cols is None:
            i += 1
            continue

        parts = s.split()
        if len(parts) != len(cols):
            i += 1
            continue
        row = {c: parts[k] for k, c in enumerate(cols)}

        if "id" not in row:
            i += 1
            continue
        pid = int(float(row["id"]))
        seen_ids.add(pid)

        if "p_hit_flag" in row and "p_hit_surf_id" in row:
            hit_flag = int(float(row["p_hit_flag"]))
            hit_sid = int(float(row["p_hit_surf_id"]))
            if hit_flag > 0 and hit_sid > 0 and pid not in first_hit:
                first_hit[pid] = hit_sid
        i += 1

    counts = Counter()
    events = []
    for pid, sid in first_hit.items():
        counts[sid] += 1
        events.append(
            {
                "source": source_name,
                "timestep": last_ts,
                "particle_id": pid,
                "surf_id": sid,
                "collision_time": 0.0,
            }
        )
    return counts, events, len(seen_ids), last_ts




def parse_vanish_hit_csv(paths, source_name: str):
    """
    Parse one or more collide_vanish_hits CSV files (rank-split).
    Expected columns include at least: timestep,id,surf_id.
    Returns:
      counts (Counter by surf_id),
      events (list of dict rows),
      vanished_detected (# unique particle IDs in vanish logs),
      last_ts (max timestep).
    """
    if not paths:
        return Counter(), [], 0, 0

    by_particle = {}
    last_ts = 0

    for pth in paths:
        if not pth.exists():
            raise FileNotFoundError(f"Missing vanish-hit file: {pth}")
        with pth.open(newline="") as f:
            reader = csv.DictReader(f)
            req = {"timestep", "id", "surf_id"}
            missing = req - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{pth}: missing required columns: {sorted(missing)}")
            for row in reader:
                try:
                    ts = int(float(row["timestep"]))
                    pid = int(float(row["id"]))
                    sid = int(float(row["surf_id"]))
                except Exception:
                    continue
                if sid <= 0:
                    continue
                isp = None
                if "ispecies" in row and str(row["ispecies"]).strip() != "":
                    try:
                        isp = int(float(row["ispecies"]))
                    except Exception:
                        isp = None
                last_ts = max(last_ts, ts)
                prev = by_particle.get(pid)
                # Keep the latest vanish row if duplicate IDs appear.
                if (prev is None) or (ts >= prev["timestep"]):
                    by_particle[pid] = {
                        "source": source_name,
                        "timestep": ts,
                        "particle_id": pid,
                        "surf_id": sid,
                        "collision_time": 0.0,
                        "ispecies": isp,
                    }

    counts = Counter()
    events = []
    for pid in sorted(by_particle):
        ev = by_particle[pid]
        counts[ev["surf_id"]] += 1
        events.append(ev)

    return counts, events, len(by_particle), last_ts

def parse_particle_ids(path: Path):
    """Return unique particle IDs found in a particle dump with ITEM: ATOMS headers."""
    lines = path.read_text().splitlines()
    i = 0
    cols = None
    ids = set()
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("ITEM: ATOMS"):
            cols = s.split()[2:]
            i += 1
            continue
        if s.startswith("ITEM:"):
            i += 1
            continue
        if not s or cols is None:
            i += 1
            continue
        parts = s.split()
        if len(parts) != len(cols):
            i += 1
            continue
        try:
            row = {c: parts[k] for k, c in enumerate(cols)}
            if "id" in row:
                ids.add(int(float(row["id"])))
        except Exception:
            pass
        i += 1
    return ids


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def get_cell_value(row_dict, col_key):
    if isinstance(col_key, str) and col_key.isdigit():
        return row_dict.get(int(col_key), 0)
    return row_dict.get(col_key, 0)


def surf_id_to_region(surf_id: int, n_wall: int = 156) -> str:
    if surf_id > n_wall:
        return "core"
    return _surf_to_region.get(surf_id, "unknown")


def infer_tag(path_i: Path, path_o: Path) -> str:
    pat = re.compile(r"mode[^.]*\.nemit[^.]*")
    for p in (path_i, path_o):
        m = pat.search(p.name)
        if m:
            return m.group(0)
    return "auto"



def plot_incidence_angle_speed(case_path: Path, droplet_class: str, out_path: Path):
    if plt is None:
        raise RuntimeError("matplotlib is required for incidence plotting")
    from utils import angle_vnorm_df
    df = angle_vnorm_df(str(case_path), droplet_class=droplet_class)
    if df.empty:
        print(f"Warning: no rows in incidence dump: {case_path}")
        return

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    fig, ax = plt.subplots(figsize=(4, 4), dpi=400)
    ax.scatter(df["angle_deg"], df["vnorm"], s=14, alpha=0.6, edgecolors="none")
    ax.set_xlabel(r"Incident angle $\theta$ (deg)")
    ax.set_ylabel(r"Incident velocity (m/s)")
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title(f"{droplet_class} incidence")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved incidence plot: {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-I", type=Path, default=None)
    p.add_argument("--in-O", type=Path, default=None)
    p.add_argument("--surf-dump", type=Path, default=None, help="Use dump surf running-average file")
    p.add_argument("--case-I", type=Path, default=None, help="Particle dump for inner source with p_hit_* fields")
    p.add_argument("--case-O", type=Path, default=None, help="Particle dump for outer source with p_hit_* fields")
    p.add_argument("--vanish-I", type=Path, nargs="+", default=None, help="collide_vanish_hits CSV files for inner source (rank files allowed)")
    p.add_argument("--vanish-O", type=Path, nargs="+", default=None, help="collide_vanish_hits CSV files for outer source (rank files allowed)")
    p.add_argument("--vanish-all", type=Path, nargs="+", default=None, help="collide_vanish_hits CSV files when inner/outer are mixed")
    p.add_argument("--idmap-I", type=Path, default=None, help="Particle dump used to map IDs to inner source (for --vanish-all)")
    p.add_argument("--idmap-O", type=Path, default=None, help="Particle dump used to map IDs to outer source (for --vanish-all)")
    p.add_argument("--ispecies-I", type=int, default=0, help="ispecies value mapped to inner source for --vanish-all when idmaps are omitted")
    p.add_argument("--ispecies-O", type=int, default=1, help="ispecies value mapped to outer source for --vanish-all when idmaps are omitted")
    p.add_argument("--launched-I", type=int, default=0)
    p.add_argument("--launched-O", type=int, default=0)
    p.add_argument("--source-region-I", type=str, default="lo_inner_div")
    p.add_argument("--source-region-O", type=str, default="lo_outer_div")
    p.add_argument(
        "--subtract-emission",
        action="store_true",
        default=True,
        help="For --surf-dump mode: subtract source-surface emission counts before normalization",
    )
    p.add_argument(
        "--no-subtract-emission",
        dest="subtract_emission",
        action="store_false",
        help="Disable source-emission subtraction in --surf-dump mode",
    )
    p.add_argument(
        "--enforce-one-hit",
        action="store_true",
        default=True,
        help="For --surf-dump mode: scale row hits to <= launched count",
    )
    p.add_argument(
        "--no-enforce-one-hit",
        dest="enforce_one_hit",
        action="store_false",
        help="Disable one-hit scaling in --surf-dump mode",
    )
    p.add_argument("--out-prefix", type=Path, default=None)
    p.add_argument("--incidence-dump", type=Path, default=None,
                  help="Particle dump path for angle-vnorm scatter (e.g. case.drag.mode.1)")
    p.add_argument("--incidence-class", type=str, default="droplet")
    p.add_argument("--incidence-out", type=Path, default=None,
                  help="Output PNG for incidence scatter")
    p.add_argument("--no-segment", action="store_true", help="Skip auto-plot via segment.py")
    args = p.parse_args()

    use_case_dump = (args.case_I is not None) or (args.case_O is not None)
    use_surf_dump = args.surf_dump is not None
    use_vanish_dump = (args.vanish_I is not None) or (args.vanish_O is not None)
    use_vanish_all = args.vanish_all is not None

    active_modes = int(use_case_dump) + int(use_surf_dump) + int(use_vanish_dump) + int(use_vanish_all)
    if active_modes > 1:
        raise SystemExit("Choose only one input mode: --case-*, --surf-dump, --vanish-*, or --vanish-all")

    if use_case_dump:
        if args.case_I is None or args.case_O is None:
            raise SystemExit("Provide both --case-I and --case-O for ID-based mode")
        if not args.case_I.exists():
            raise SystemExit(f"Missing input file: {args.case_I}")
        if not args.case_O.exists():
            raise SystemExit(f"Missing input file: {args.case_O}")
        tag = infer_tag(args.case_I, args.case_O)
    elif use_surf_dump:
        if not args.surf_dump.exists():
            raise SystemExit(f"Missing input file: {args.surf_dump}")
        tag = infer_tag(args.surf_dump, args.surf_dump)
    elif use_vanish_dump:
        if args.vanish_I:
            for fp in args.vanish_I:
                if not fp.exists():
                    raise SystemExit(f"Missing input file: {fp}")
        if args.vanish_O:
            for fp in args.vanish_O:
                if not fp.exists():
                    raise SystemExit(f"Missing input file: {fp}")
        # one-sided use is allowed (e.g., only outer-source run)
        vanI = args.vanish_I[0] if args.vanish_I else Path("vanishI")
        vanO = args.vanish_O[0] if args.vanish_O else Path("vanishO")
        tag = infer_tag(vanI, vanO)
    elif use_vanish_all:
        if (args.idmap_I is None) ^ (args.idmap_O is None):
            raise SystemExit("For --vanish-all with ID mapping, provide both --idmap-I and --idmap-O")
        if args.idmap_I is not None and not args.idmap_I.exists():
            raise SystemExit(f"Missing input file: {args.idmap_I}")
        if args.idmap_O is not None and not args.idmap_O.exists():
            raise SystemExit(f"Missing input file: {args.idmap_O}")
        for fp in args.vanish_all:
            if not fp.exists():
                raise SystemExit(f"Missing input file: {fp}")
        if args.idmap_I is not None and args.idmap_O is not None:
            tag = infer_tag(args.idmap_I, args.idmap_O)
        else:
            tag = infer_tag(args.vanish_all[0], args.vanish_all[-1])
    else:
        if args.in_I is None or args.in_O is None:
            raise SystemExit("Provide either --surf-dump OR both --in-I and --in-O OR --vanish-I/--vanish-O")
        if not args.in_I.exists():
            raise SystemExit(f"Missing input file: {args.in_I}")
        if not args.in_O.exists():
            raise SystemExit(f"Missing input file: {args.in_O}")
        tag = infer_tag(args.in_I, args.in_O)
    out_prefix = args.out_prefix if args.out_prefix else Path(f"transfer_matrix.{tag}")

    launched_detected = {"I": 0, "O": 0}

    if use_case_dump:
        cI, eI, nI, tsI = parse_particle_first_hits(args.case_I, "I")
        cO, eO, nO, tsO = parse_particle_first_hits(args.case_O, "O")
        launched_detected["I"] = nI
        launched_detected["O"] = nO

        counts_by_source = {"I": cI, "O": cO}
        events = eI + eO
        events_rows = [
            [e["source"], e["timestep"], e["particle_id"], e["surf_id"], f'{e["collision_time"]:.8g}']
            for e in events
        ]
        surf_ids = sorted(set(cI.keys()) | set(cO.keys()))
        last_ts = max(tsI, tsO)
    elif use_vanish_dump:
        cI, eI, nI, tsI = parse_vanish_hit_csv(args.vanish_I or [], "I")
        cO, eO, nO, tsO = parse_vanish_hit_csv(args.vanish_O or [], "O")
        launched_detected["I"] = nI
        launched_detected["O"] = nO

        counts_by_source = {"I": cI, "O": cO}
        events = eI + eO
        events_rows = [
            [e["source"], e["timestep"], e["particle_id"], e["surf_id"], f'{e["collision_time"]:.8g}']
            for e in events
        ]
        surf_ids = sorted(set(cI.keys()) | set(cO.keys()))
        last_ts = max(tsI, tsO)
    elif use_vanish_all:
        _, eall, _, ts_all = parse_vanish_hit_csv(args.vanish_all or [], "M")

        cI, cO = Counter(), Counter()
        eI, eO = [], []
        nunmapped = 0

        if args.idmap_I is not None and args.idmap_O is not None:
            idsI = parse_particle_ids(args.idmap_I)
            idsO = parse_particle_ids(args.idmap_O)
            launched_detected["I"] = len(idsI)
            launched_detected["O"] = len(idsO)

            for e in eall:
                pid = e["particle_id"]
                if pid in idsI:
                    ee = dict(e); ee["source"] = "I"
                    eI.append(ee); cI[ee["surf_id"]] += 1
                elif pid in idsO:
                    ee = dict(e); ee["source"] = "O"
                    eO.append(ee); cO[ee["surf_id"]] += 1
                else:
                    nunmapped += 1

            if nunmapped > 0:
                print(f"Warning: {nunmapped} vanish hits could not be mapped to I/O by particle ID.")
        else:
            # No ID maps available: split mixed vanish logs using ispecies.
            # This is the recommended path for vanish-only sweeps.
            idsI_seen = set()
            idsO_seen = set()
            for e in eall:
                isp = e.get("ispecies", None)
                if isp == int(args.ispecies_I):
                    ee = dict(e); ee["source"] = "I"
                    eI.append(ee); cI[ee["surf_id"]] += 1
                    idsI_seen.add(ee["particle_id"])
                elif isp == int(args.ispecies_O):
                    ee = dict(e); ee["source"] = "O"
                    eO.append(ee); cO[ee["surf_id"]] += 1
                    idsO_seen.add(ee["particle_id"])
                else:
                    nunmapped += 1

            launched_detected["I"] = len(idsI_seen)
            launched_detected["O"] = len(idsO_seen)
            if nunmapped > 0:
                print(
                    f"Warning: {nunmapped} vanish hits could not be mapped by ispecies "
                    f"(I={args.ispecies_I}, O={args.ispecies_O})."
                )

        counts_by_source = {"I": cI, "O": cO}
        events = eI + eO
        events_rows = [
            [e["source"], e["timestep"], e["particle_id"], e["surf_id"], f'{e["collision_time"]:.8g}']
            for e in events
        ]
        surf_ids = sorted(set(cI.keys()) | set(cO.keys()))
        last_ts = ts_all
    elif use_surf_dump:
        counts_by_source, surf_ids, last_ts = parse_surf_running_dump(args.surf_dump)
        # Minimal events file for downstream plot title timestep.
        events = []
        events_rows = [
            ["I", last_ts, -1, -1, "0.0"],
            ["O", last_ts, -1, -1, "0.0"],
        ]
    else:
        events = parse_collision_events(args.in_I, "I") + parse_collision_events(args.in_O, "O")
        events_rows = [
            [e["source"], e["timestep"], e["particle_id"], e["surf_id"], f'{e["collision_time"]:.8g}']
            for e in events
        ]
        surf_ids = sorted({e["surf_id"] for e in events})
        counts_by_source = {"I": Counter(), "O": Counter()}
        for e in events:
            counts_by_source[e["source"]][e["surf_id"]] += 1
    write_csv(
        Path(f"{out_prefix}.events.csv"),
        ["source", "timestep", "particle_id", "surf_id", "collision_time"],
        events_rows,
    )

    launched = {"I": max(0, int(args.launched_I)), "O": max(0, int(args.launched_O))}
    if use_case_dump or use_vanish_dump or use_vanish_all:
        # For vanish-all, launched_detected comes from idmap files (best estimate).
        # For vanish-only files, detected counts are lower bounds (only vanished particles).
        if use_vanish_dump and args.idmap_I is not None and args.idmap_I.exists() and launched["I"] <= 0:
            launched_detected["I"] = len(parse_particle_ids(args.idmap_I))
        if use_vanish_dump and args.idmap_O is not None and args.idmap_O.exists() and launched["O"] <= 0:
            launched_detected["O"] = len(parse_particle_ids(args.idmap_O))

        if launched["I"] <= 0:
            launched["I"] = launched_detected["I"]
            if use_vanish_dump and args.idmap_I is None:
                print(f"Info: launched-I inferred from vanished IDs only = {launched['I']} (lower bound)")
            else:
                print(f"Info: launched-I inferred from particle IDs = {launched['I']}")
        if launched["O"] <= 0:
            launched["O"] = launched_detected["O"]
            if use_vanish_dump and args.idmap_O is None:
                print(f"Info: launched-O inferred from vanished IDs only = {launched['O']} (lower bound)")
            else:
                print(f"Info: launched-O inferred from particle IDs = {launched['O']}")
    matrix_counts = defaultdict(dict)
    for src in ["I", "O"]:
        row_sum = 0
        for sid in surf_ids:
            c = counts_by_source[src][sid]
            matrix_counts[src][sid] = c
            row_sum += c
        if launched[src] > 0:
            matrix_counts[src]["evaporated"] = max(0, launched[src] - row_sum)

    count_cols = [str(s) for s in surf_ids]
    if "evaporated" in matrix_counts["I"] or "evaporated" in matrix_counts["O"]:
        count_cols.append("evaporated")
    count_rows = []
    for src in ["I", "O"]:
        row = [src] + [get_cell_value(matrix_counts[src], c) for c in count_cols]
        count_rows.append(row)
    write_csv(Path(f"{out_prefix}.counts.csv"), ["source"] + count_cols, count_rows)

    pct_rows = []
    for src in ["I", "O"]:
        denom = float(launched[src])
        vals = []
        for c in count_cols:
            v = float(get_cell_value(matrix_counts[src], c))
            vals.append((100.0 * v / denom) if denom > 0 else 0.0)
        pct_rows.append([src] + [f"{v:.6f}" for v in vals])
    write_csv(Path(f"{out_prefix}.pct.csv"), ["source"] + count_cols, pct_rows)

    # Region-aggregated matrix from surf-id counts.
    region_counts = defaultdict(dict)
    for src in ["I", "O"]:
        for region in DEST_ORDER:
            region_counts[src][region] = 0
        for sid in surf_ids:
            reg = surf_id_to_region(sid, n_wall=156)
            if reg not in region_counts[src]:
                region_counts[src][reg] = 0
            region_counts[src][reg] += counts_by_source[src][sid]
        if "evaporated" in matrix_counts[src]:
            region_counts[src]["evaporated"] = matrix_counts[src]["evaporated"]

    # In surf-dump mode, compute surf n includes emit/surf-created events on source
    # surfaces. Subtract them so matrix reflects destination hits of launched particles.
    if use_surf_dump and args.subtract_emission:
        submap = {
            "I": (args.source_region_I, max(0, int(args.launched_I))),
            "O": (args.source_region_O, max(0, int(args.launched_O))),
        }
        for src, (region, nlaunch) in submap.items():
            if region not in region_counts[src]:
                continue
            region_counts[src][region] = max(0.0, float(region_counts[src][region]) - float(nlaunch))

    region_cols = DEST_ORDER[:]
    if "evaporated" in region_counts["I"] or "evaporated" in region_counts["O"]:
        region_cols.append("evaporated")

    # Recompute evaporated using region counts, so it is consistent with subtraction.
    if "evaporated" in region_cols:
        for src in ["I", "O"]:
            hit_sum = 0
            for c in region_cols:
                if c == "evaporated":
                    continue
                hit_sum += float(region_counts[src].get(c, 0))
            if launched[src] > 0:
                region_counts[src]["evaporated"] = max(0.0, float(launched[src]) - hit_sum)

    # Optional physical constraint for sticking model: each launched particle
    # contributes to at most one destination hit.
    if use_surf_dump and args.enforce_one_hit:
        for src in ["I", "O"]:
            nlaunch = float(launched[src])
            if nlaunch <= 0:
                continue
            hit_cols = [c for c in region_cols if c != "evaporated"]
            hit_sum = sum(float(region_counts[src].get(c, 0.0)) for c in hit_cols)
            if hit_sum > nlaunch and hit_sum > 0.0:
                scale = nlaunch / hit_sum
                for c in hit_cols:
                    region_counts[src][c] = float(region_counts[src].get(c, 0.0)) * scale
                region_counts[src]["evaporated"] = 0.0
                print(
                    f"Info: scaled {src} row by {scale:.6f} to enforce <= launched hits "
                    f"({hit_sum:.3f} -> {nlaunch:.3f})."
                )

    region_count_rows = []
    for src in ["I", "O"]:
        region_count_rows.append([src] + [region_counts[src].get(c, 0) for c in region_cols])
    write_csv(Path(f"{out_prefix}.region.counts.csv"), ["source"] + region_cols, region_count_rows)

    region_pct_rows = []
    for src in ["I", "O"]:
        denom = float(launched[src])
        vals = []
        for c in region_cols:
            v = float(region_counts[src].get(c, 0))
            vals.append((100.0 * v / denom) if denom > 0 else 0.0)
        region_pct_rows.append([src] + [f"{v:.6f}" for v in vals])
    write_csv(Path(f"{out_prefix}.region.pct.csv"), ["source"] + region_cols, region_pct_rows)

    print(f"Wrote: {out_prefix}.events.csv")
    print(f"Wrote: {out_prefix}.counts.csv")
    print(f"Wrote: {out_prefix}.pct.csv")
    print(f"Wrote: {out_prefix}.region.counts.csv")
    print(f"Wrote: {out_prefix}.region.pct.csv")
    print(f"Event rows: {len(events)}")
    if use_case_dump and len(events) == 0:
        print("Warning: no first-hit rows found in case dumps.")
        print("Check that dump includes p_hit_flag and p_hit_surf_id fields.")
    elif use_vanish_dump and len(events) == 0:
        print("Warning: no vanish-hit rows found in vanish CSV files.")
        print("Check that files include columns timestep,id,surf_id and non-empty data.")
    elif (not use_surf_dump) and len(events) == 0:
        print("Warning: no collision tally rows found in input files.")
        print("Check compute group/mixture filters (e.g., surfID vs surfID2, I/O mixture).")

    if not args.no_segment:
        segment_script = Path(__file__).with_name("segment.py")
        fig_out = Path("Figs") / f"transfer_matrix_percentage_{tag}.png"
        counts_for_plot = f"{out_prefix}.counts.csv"
        if use_surf_dump:
            counts_for_plot = f"{out_prefix}.region.counts.csv"
        cmd = [
            sys.executable,
            str(segment_script),
            "--counts",
            counts_for_plot,
            "--events",
            f"{out_prefix}.events.csv",
            "--launched-I",
            str(launched["I"]),
            "--launched-O",
            str(launched["O"]),
            "--source-region-I",
            str(args.source_region_I),
            "--source-region-O",
            str(args.source_region_O),
            "--out",
            str(fig_out),
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception as exc:
            print(f"Auto segment failed: {exc}")
        else:
            print(f"Auto segment done: {fig_out}")

    incidence_dump = args.incidence_dump
    if incidence_dump is None:
        auto = Path("case.drag.mode.1")
        if auto.exists():
            incidence_dump = auto

    if incidence_dump is not None:
        if not incidence_dump.exists():
            print(f"Warning: incidence dump not found: {incidence_dump}")
        else:
            inc_out = args.incidence_out
            if inc_out is None:
                inc_out = Path("Figs") / f"incidence_angle_vnorm_{tag}.png"
            try:
                plot_incidence_angle_speed(incidence_dump, args.incidence_class, inc_out)
            except Exception as exc:
                print(f"Incidence plot failed: {exc}")


if __name__ == "__main__":
    main()
#python3 build_transfer_matrix.py --surf-dump tmp.all.surf --launched-I 199 --launched-O 199 --wall-label-every 2 --core-label-every 10 --label-fontsize 5
# python3 build_transfer_matrix.py --surf-dump tmp.all.surf --launched-I 199 --launched-O 199
#
#  python3 build_transfer_matrix.py \
#    --vanish-all collide_vanish_hits.csv.rank* \
#    --idmap-I case.inner \
#    --idmap-O case.outer \
#    --out-prefix transfer_matrix.vanish_io_auto
