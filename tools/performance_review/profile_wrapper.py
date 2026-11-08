#!/usr/bin/env python3
"""Wrap run_case.sh without modifying the production input decks.

Set OE_PROFILE_BINARY and OE_PROFILE_DIR; OPENEDGE_BIN points at this file.
Only generated copies in OE_PROFILE_DIR are edited. Physics is unchanged.
"""
import os
from pathlib import Path
import shlex
import sys

args = sys.argv[1:]
dest = Path(os.environ["OE_PROFILE_DIR"])
dest.mkdir(parents=True, exist_ok=True)
i = args.index("-in") + 1
restart = os.environ.get("OE_PROFILE_WARM_RESTART")
deck = (Path("production/west_mb8_continuous_ops_20260922/in.restart_history")
        if restart else Path(args[i])).read_text()
if restart:
    args += ["-var", "restartfile", restart]
variables = {args[j + 1]: args[j + 2] for j, a in enumerate(args) if a == "-var"}
schedule = Path(variables["schedule"]).read_text().replace("post no", "post yes")
(dest / "schedule.inc").write_text(schedule)
deck = deck.replace("include ${schedule}", f"include {dest / 'schedule.inc'}")
setup = Path("production/west_mb8_continuous_ops_20260922/in.setup").read_text()
setup = setup.replace("static ${bgstatic}", "static ${bgstatic} mesh_lookup_diagnostics yes")
(dest / "setup.inc").write_text(setup)
deck = deck.replace("include production/west_mb8_continuous_ops_20260922/in.setup",
                    f"include {dest / 'setup.inc'}")
(dest / "profile.in").write_text(deck)
args[i] = str(dest / "profile.in")
if os.environ.get("OE_PROFILE_KOKKOS_ARGS"):
    args += shlex.split(os.environ["OE_PROFILE_KOKKOS_ARGS"])
elif os.environ.get("OE_KK_PROFILE"):
    args += ["-k", "on", "t", "1", "-sf", "kk", "-pk", "kokkos", "react/retry", "yes"]
binary = os.environ["OE_PROFILE_BINARY"]
os.execv(binary, [binary, *args])
