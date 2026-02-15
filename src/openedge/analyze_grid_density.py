#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_grid_dump(path):
    blocks = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f]

    i = 0
    n = len(lines)
    while i < n:
        if lines[i] != "ITEM: TIMESTEP":
            i += 1
            continue

        ts = int(lines[i + 1])
        i += 2

        if lines[i] != "ITEM: NUMBER OF CELLS":
            raise ValueError(f"Malformed dump near timestep {ts}")
        ncells = int(lines[i + 1])
        i += 2

        if lines[i] != "ITEM: BOX BOUNDS pp pp pp":
            raise ValueError(f"Malformed bounds near timestep {ts}")
        i += 4  # header + 3 bounds lines

        if not lines[i].startswith("ITEM: CELLS"):
            raise ValueError(f"Missing CELLS header near timestep {ts}")
        header = lines[i].split()[2:]  # id xc yc f_fixID[*]
        i += 1

        rows = []
        for _ in range(ncells):
            parts = lines[i].split()
            i += 1
            rows.append(parts)

        blocks.append((ts, header, rows))

    return blocks


def to_records(header, rows):
    idx = {name: j for j, name in enumerate(header)}
    if "id" not in idx or "xc" not in idx or "yc" not in idx:
        raise ValueError("Expected columns: id xc yc ...")

    # density is the first non-coordinate column
    dens_col = None
    for name in header:
        if name not in ("id", "xc", "yc", "zc"):
            dens_col = name
            break
    if dens_col is None:
        raise ValueError("No density-like column found in grid dump")

    out = []
    for r in rows:
        rec = {
            "id": int(float(r[idx["id"]])),
            "xc": float(r[idx["xc"]]),
            "yc": float(r[idx["yc"]]),
            "nrho": float(r[idx[dens_col]]),
        }
        out.append(rec)
    return out, dens_col


def main():
    ap = argparse.ArgumentParser(description="Analyze SPARTA/OpenEdge grid density dump.")
    ap.add_argument("--infile", default="output/tmp.grid.density")
    ap.add_argument("--timestep", default="last", help="'last' or explicit integer timestep")
    ap.add_argument("--outcsv", default="output/grid_density_last.csv")
    ap.add_argument("--plot", action="store_true", help="save scatter plot png")
    ap.add_argument("--outpng", default="output/grid_density_last.png")
    args = ap.parse_args()

    infile = Path(args.infile)
    blocks = parse_grid_dump(infile)
    if not blocks:
        raise RuntimeError(f"No timestep blocks found in {infile}")

    if args.timestep == "last":
        ts, header, rows = blocks[-1]
    else:
        tsel = int(args.timestep)
        found = [b for b in blocks if b[0] == tsel]
        if not found:
            raise RuntimeError(f"Timestep {tsel} not found")
        ts, header, rows = found[-1]

    recs, dens_col = to_records(header, rows)
    vals = [r["nrho"] for r in recs]
    nnz = sum(v > 0.0 for v in vals)

    print(f"Timestep: {ts}")
    print(f"Cells: {len(vals)}")
    print(f"Column used: {dens_col}")
    print(f"Nonzero cells: {nnz}")
    print(f"Min/Max nrho: {min(vals):.6e} / {max(vals):.6e}")
    print(f"Mean nrho: {sum(vals)/len(vals):.6e}")

    outcsv = Path(args.outcsv)
    outcsv.parent.mkdir(parents=True, exist_ok=True)
    with open(outcsv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "R", "Z", "nrho_m^-3"])
        for r in recs:
            w.writerow([r["id"], r["xc"], r["yc"], r["nrho"]])
    print(f"Wrote CSV: {outcsv}")

    if args.plot:
        import matplotlib.pyplot as plt

        xs = [r["xc"] for r in recs]
        ys = [r["yc"] for r in recs]
        cs = [r["nrho"] for r in recs]
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(xs, ys, c=cs, s=55, cmap="inferno")
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("nrho [m$^{-3}$]")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title(f"Grid density at timestep {ts}")
        fig.tight_layout()
        outpng = Path(args.outpng)
        outpng.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpng, dpi=160)
        print(f"Wrote plot: {outpng}")


if __name__ == "__main__":
    main()
