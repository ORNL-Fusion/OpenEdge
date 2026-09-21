#!/usr/bin/env python3
"""Check that Li is absorbed and the grain crosses the psi boundary."""

from pathlib import Path


def frames(path: Path):
    lines = path.read_text().splitlines()
    out = []
    i = 0
    while i < len(lines):
        if lines[i] != "ITEM: TIMESTEP":
            i += 1
            continue
        step = int(lines[i + 1])
        assert lines[i + 2] == "ITEM: NUMBER OF ATOMS"
        count = int(lines[i + 3])
        j = i + 4
        while not lines[j].startswith("ITEM: ATOMS"):
            j += 1
        columns = lines[j].split()[2:]
        rows = []
        for line in lines[j + 1 : j + 1 + count]:
            rows.append(dict(zip(columns, line.split())))
        out.append((step, rows))
        i = j + 1 + count
    return out


history = frames(Path("output_absorb/state"))
if not history:
    raise SystemExit("FAIL: no particle frames")

step0 = history[0][1]
final_step, final = history[-1]
initial_types = sorted(int(row["type"]) for row in step0)
final_types = sorted(int(row["type"]) for row in final)

if initial_types != [1, 2, 3, 4, 5]:
    raise SystemExit(
        f"FAIL: expected initial grain plus four Li charge states, got {initial_types}"
    )
if final_types != [1]:
    raise SystemExit(f"FAIL: Li was not selectively absorbed; final types {final_types}")

grain = final[0]
z_final = float(grain["x"])
r_final = float(grain["y"])
if abs(z_final - 0.23) > 1.0e-10 or abs(r_final - 0.50) > 1.0e-10:
    raise SystemExit(
        f"FAIL: grain did not cross ballistically: Z={z_final}, R={r_final}"
    )

print(
    "PASS: Li through Li3+ crossed psi_N=0.51 and were absorbed; "
    f"grain remained and reached (R,Z)=({r_final:.2f},{z_final:.2f}) at step {final_step}"
)

# The final ave/time block has 12 flattened entries:
# [events, physical pweight, mean rate] x [Li, Li+, Li2+, Li3+].
flux_lines = [line for line in Path("output_absorb/core_flux").read_text().splitlines()
              if line and not line.startswith("#")]
block_start = max(i for i, line in enumerate(flux_lines)
                  if line.split() == [str(final_step), "12"])
values = {
    int(line.split()[0]): float(line.split()[1])
    for line in flux_lines[block_start + 1 : block_start + 13]
}
for row, name in enumerate(("Li", "Li+", "Li2+", "Li3+")):
    base = 3 * row + 1
    if values[base] != 1.0 or values[base + 1] != 1.0:
        raise SystemExit(
            f"FAIL: {name} ledger expected one marker and pweight=1, "
            f"got events={values[base]}, physical={values[base + 1]}"
        )
    if abs(values[base + 2] - 200.0) > 1.0e-12:
        raise SystemExit(
            f"FAIL: {name} mean removal rate expected 200/s, "
            f"got {values[base + 2]}"
        )

print("PASS: core ledger resolves Li, Li+, Li2+, and Li3+: pweight=1 and mean rate=200 s^-1 each")
