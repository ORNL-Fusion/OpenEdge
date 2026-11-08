#!/usr/bin/env python3
"""Exec'd by run_case.sh in place of the OpenEdge binary (cwd = pilot dir).
Edits only generated copies under OE_PROFILE_DIR: `post no` -> `post yes` in the
schedule (SPARTA Finish timing table), optional Kokkos args, optional nsys prefix.
Physics and production decks unchanged (same approach as tools/performance_review/profile_wrapper.py)."""
import os, shlex, sys
from pathlib import Path
args = sys.argv[1:]
dest = Path(os.environ["OE_PROFILE_DIR"]); dest.mkdir(parents=True, exist_ok=True)
i = args.index("-in") + 1
deck = Path(args[i]).read_text()
variables = {args[j + 1]: args[j + 2] for j, a in enumerate(args) if a == "-var"}
schedule = Path(variables["schedule"]).read_text().replace("post no", "post yes")
(dest / "schedule.inc").write_text(schedule)
deck = deck.replace("include ${schedule}", f"include {dest / 'schedule.inc'}")
setup_sed = os.environ.get("OE_SETUP_SED")   # diagnostic experiments on a GENERATED copy of in.setup only
if setup_sed:
    import subprocess
    src = Path("production/west_mb8_continuous_ops_20260922/in.setup")
    setup = subprocess.run(["sed", "-e", setup_sed], input=src.read_text(), capture_output=True, text=True, check=True).stdout
    (dest / "setup.inc").write_text(setup)
    deck = deck.replace("include production/west_mb8_continuous_ops_20260922/in.setup", f"include {dest / 'setup.inc'}")
(dest / "profile.in").write_text(deck)
args[i] = str(dest / "profile.in")
if os.environ.get("OE_KOKKOS_ARGS"):
    args += shlex.split(os.environ["OE_KOKKOS_ARGS"])
binary = os.environ["OE_PROFILE_BINARY"]
cmd = [binary, *args]
rank = int(os.environ.get("SLURM_PROCID", "0"))
nsys_ranks = os.environ.get("OE_NSYS_RANKS")  # e.g. "0" or "0,1,2,3"
if nsys_ranks and str(rank) in nsys_ranks.split(","):
    nsys = os.environ.get("NSYS", "nsys")
    extra = shlex.split(os.environ.get("OE_NSYS_OPTS", "-t cuda,nvtx -s none --cpuctxsw=none"))
    cmd = [nsys, "profile", *extra, "--force-overwrite=true", "-o", str(dest / f"nsys_r{rank}"), *cmd]
with open(dest / f"cmd_r{rank}.txt", "w") as f:
    f.write(" ".join(shlex.quote(c) for c in cmd) + "\n")
os.execvp(cmd[0], cmd)
