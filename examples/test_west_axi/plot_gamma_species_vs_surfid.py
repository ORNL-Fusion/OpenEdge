#!/usr/bin/env python3
"""Plot per-species incident plasma flux vs surface ID from output/gamma.only."""

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np


def parse_gamma_dump(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing dump file: {path}")

    lines = path.read_text().splitlines()
    i = 0
    last_block = None

    while i < len(lines):
        if lines[i].strip() == "ITEM: TIMESTEP":
            if i + 8 >= len(lines):
                break
            timestep = int(lines[i + 1].strip())
            if lines[i + 2].strip() != "ITEM: NUMBER OF SURFS":
                i += 1
                continue
            nsurf = int(lines[i + 3].strip())
            # lines i+4..i+7 are bounds header + 3 bounds lines
            header = lines[i + 8].strip()
            if not header.startswith("ITEM: SURFS"):
                i += 1
                continue

            cols = header.split()[2:]  # skip ITEM: SURFS
            block = []
            start = i + 9
            end = min(start + nsurf, len(lines))
            for k in range(start, end):
                parts = lines[k].split()
                if len(parts) < len(cols):
                    continue
                block.append([float(x) for x in parts[: len(cols)]])

            if block:
                arr = np.array(block, dtype=float)
                last_block = (timestep, cols, arr)
            i = end
        else:
            i += 1

    if last_block is None:
        raise RuntimeError("No SURFS block found in dump file")
    return last_block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="output/gamma.only", help="Path to surf dump")
    ap.add_argument("--out", default="output/gamma_species_vs_surfid.png", help="Output PNG")
    ap.add_argument("--show", action="store_true", help="Show plot window")
    args = ap.parse_args()

    timestep, cols, arr = parse_gamma_dump(Path(args.dump))
    surf_id = arr[:, 0].astype(int)

    fig, ax = plt.subplots(figsize=(10, 6))

    # c_cgam[*] columns are species fluxes when using nflux_species all
    for j in range(1, arr.shape[1]):
        y = arr[:, j]
        ax.plot(surf_id, y, lw=1.2, label=f"species slot {j}")

    ax.set_title(f"Incident Species Flux vs Surf ID (timestep={timestep})")
    ax.set_xlabel("surf id")
    ax.set_ylabel("Gamma_species_normal [1/m^2/s]")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"Wrote: {out.resolve()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
