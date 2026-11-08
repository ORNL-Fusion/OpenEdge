#!/usr/bin/env python3
"""Compare WEST particle states and area-integrated wall ledgers by monoblock.

An escape inventory inferred from net wall loss is reported explicitly as such;
these decks do not independently tally escaped real-particle weights. The default
is a statistical report; --require-exact also rejects any trajectory difference.
"""
import csv
import json
from pathlib import Path
import re
import sys
import numpy as np

def last_dump(path, kind):
    result = None
    with path.open() as stream:
        while line := stream.readline():
            if not line.startswith("ITEM: TIMESTEP"):
                continue
            step = int(stream.readline())
            line = stream.readline()
            assert line.startswith("ITEM: NUMBER OF")
            n = int(stream.readline())
            while not (line := stream.readline()).startswith("ITEM: " + kind):
                if not line:
                    raise ValueError("Truncated dump")
            columns = line.split()[2:]
            rows = np.array([[float(v) for v in stream.readline().split()] for _ in range(n)])
            rows = rows.reshape(n, len(columns))
            result = step, columns, rows
    if result is None:
        raise ValueError("No frames in " + str(path))
    return result

def wall(run):
    paths = list(run.glob("mb_adens.*.dump"))
    path = max(paths, key=lambda p: int(p.name.split(".")[-2]))
    step, cols, data = last_dump(path, "SURFS")
    lines = [l.strip() for l in (run / "surface_geometry.surf").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    npnt, ntri = int(lines[0].split()[0]), int(lines[1].split()[0])
    start = lines.index("Points") + 1
    points = {int(t[0]): np.array(t[1:4], float)
              for t in (l.split() for l in lines[start:start+npnt])}
    start = lines.index("Triangles") + 1
    areas = {}
    for line in lines[start:start+ntri]:
        t = list(map(int, line.split()))
        a, b, c = [points[k] for k in t[1:4]]
        areas[t[0]] = .5 * np.linalg.norm(np.cross(b-a, c-a))
    with (run / "surface_groups.csv").open() as f:
        blocks = {int(r["surf_id"]): int(re.search(r"MB(\d+)", r["owner"])[1])
                  for r in csv.DictReader(f)}
    ids = data[:, cols.index("id")].astype(int)
    order = np.argsort(ids)
    data, ids = data[order], ids[order]
    area = np.array([areas[i] for i in ids])
    mb = np.array([blocks[i] for i in ids])
    ledgers = {}
    for b in range(1, 36):
        mask = mb == b
        ledgers[str(b)] = {k: float(np.sum(area[mask]*data[mask, cols.index("s_adens_"+k)]))
                           for k in ("net", "dep", "ero")}
    return step, cols, data, ledgers

def main():
    before, after = map(Path, sys.argv[1:3])
    a = last_dump(before / "particles.dump", "ATOMS")
    b = last_dump(after / "particles.dump", "ATOMS")
    assert a[:2] == b[:2]
    # Lexicographic row ordering also handles possible duplicate random IDs.
    def ordered(x):
        return x[np.lexsort(x.T[::-1])]
    particles_equal = a[2].shape == b[2].shape and np.array_equal(ordered(a[2]), ordered(b[2]))
    wa, wb = wall(before), wall(after)
    assert wa[:2] == wb[:2]
    walls_equal = np.array_equal(wa[2], wb[2])
    inventory_a = float(a[2][:, a[1].index("p_pweight")].sum())
    inventory_b = float(b[2][:, b[1].index("p_pweight")].sum())
    totals = {k: sum(row[k] for row in wa[3].values()) for k in ("net", "dep", "ero")}
    differences = {}
    for k in ("net", "dep", "ero"):
        ref = np.array([row[k] for row in wa[3].values()])
        new = np.array([row[k] for row in wb[3].values()])
        differences[k] = dict(max_absolute_atoms=float(np.max(np.abs(new-ref))),
                              normalized_L1=float(np.sum(np.abs(new-ref))/max(np.sum(np.abs(ref)),1)),
                              total_relative=float(abs(new.sum()-ref.sum())/max(abs(ref.sum()),1)))
    result = dict(step=a[0], particle_count=len(a[2]), particles_exact=particles_equal,
                  surface_fields_exact=walls_equal, before_monoblock_atoms=wa[3],
                  after_monoblock_atoms=wb[3], before_particle_W_atoms=inventory_a,
                  after_particle_W_atoms=inventory_b, before_surface_W_atoms=totals,
                  wall_ledger_residual_atoms=totals["net"]-totals["dep"]+totals["ero"],
                  inferred_escaped_W_atoms=-totals["net"]-inventory_a,
                  monoblock_differences=differences,
                  particle_inventory_relative_difference=abs(inventory_a-inventory_b)/max(abs(inventory_a),1),
                  independent_escape_budget_available=False)
    print(json.dumps(result, indent=2))
    if "--require-exact" in sys.argv and not (particles_equal and walls_equal):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
