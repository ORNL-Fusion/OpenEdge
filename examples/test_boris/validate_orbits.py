#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


def prepare_input(template: str, source_name: str, steps: int) -> str:
    lines = []
    for raw in template.splitlines():
        s = raw.strip()
        if s.startswith("read_particles"):
            # keep only selected source active
            if source_name in s:
                lines.append(f"read_particles    {source_name} 0")
            elif s.startswith("#read_particles"):
                if source_name in s:
                    lines.append(f"read_particles    {source_name} 0")
                else:
                    lines.append(raw)
            else:
                lines.append(f"# {raw}")
            continue
        if s.startswith("#read_particles") and source_name in s:
            lines.append(f"read_particles    {source_name} 0")
            continue
        if s.startswith("variable numStep"):
            lines.append(f"variable numStep    equal {steps}")
            continue
        if s.startswith("variable dumpFreq1"):
            dumpfreq = max(1, steps // 500)
            lines.append(f"variable dumpFreq1  equal {dumpfreq}")
            continue
        lines.append(raw)
    return "\n".join(lines) + "\n"


def run_case(example_dir: Path, exe: str, np: int, source_name: str, steps: int) -> Path:
    in_template = (example_dir / "in.orbits").read_text()
    inp = prepare_input(in_template, source_name, steps)
    inp_path = example_dir / f"in.tmp.{source_name}"
    dump_path = example_dir / "output" / f"state.{source_name}"

    # Force case-specific dump name by replacing default dump target.
    inp = inp.replace("dump      10 particle all ${dumpFreq1} state id type x y z vx vy vz",
                      f"dump      10 particle all ${{dumpFreq1}} output/state.{source_name} id type x y z vx vy vz")

    inp_path.write_text(inp)
    cmd = ["mpirun", "-np", str(np), str((example_dir / exe).resolve())]
    with inp_path.open("r") as f:
        proc = subprocess.run(cmd, stdin=f, cwd=example_dir)
    if proc.returncode != 0:
        raise RuntimeError(f"Case {source_name} failed with exit code {proc.returncode}")
    return dump_path


def read_xy(path: Path):
    xs, ys = [], []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].startswith("ITEM: ATOMS"):
            i += 1
            continue
        cols = lines[i].split()[2:]
        i += 1
        # one particle in these tests
        row = lines[i].split()
        vals = {cols[k]: float(row[k]) for k in range(len(cols))}
        xs.append(vals["x"])
        ys.append(vals["y"])
        i += 1
    return xs, ys


def plot_cases(example_dir: Path, banana_dump: Path, passing_dump: Path):
    import matplotlib.pyplot as plt

    xb, yb = read_xy(banana_dump)
    xp, yp = read_xy(passing_dump)

    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=150)
    ax.plot(xb, yb, label="banana", lw=1.3)
    ax.plot(xp, yp, label="passing", lw=1.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("test_boris: banana vs passing")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()

    out = example_dir / "output" / "orbits_compare.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote: {out}")


def main():
    p = argparse.ArgumentParser(description="Run banana/passing orbit validation")
    p.add_argument("--exe", default="../../src/spa_mpi")
    p.add_argument("--np", type=int, default=4)
    p.add_argument("--steps", type=int, default=30000)
    args = p.parse_args()

    example_dir = Path(__file__).resolve().parent
    (example_dir / "output").mkdir(parents=True, exist_ok=True)

    banana = run_case(example_dir, args.exe, args.np, "source_banana_orb", args.steps)
    passing = run_case(example_dir, args.exe, args.np, "source_passing_orb", args.steps)

    try:
        plot_cases(example_dir, banana, passing)
    except Exception as exc:
        raise RuntimeError("Plot failed (matplotlib required)") from exc


if __name__ == "__main__":
    main()
