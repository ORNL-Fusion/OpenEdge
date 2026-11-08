#!/usr/bin/env python3
"""Build the offline probe with the CPU binary's existing library link flags."""
from pathlib import Path
import shlex
import subprocess
import sys

source, build = map(lambda x: Path(x).resolve(), sys.argv[1:3])
probe = sys.argv[3] if len(sys.argv) > 3 else "background_audit"
work = build / "src"
links = [p for p in work.glob("CMakeFiles/spa_*.dir/link.txt")
         if (work / p.parent.name.removesuffix(".dir")).is_file()]
if len(links) != 1:
    raise SystemExit("Expected one CPU binary link.txt")
link = shlex.split(links[0].read_text())
obj = work / (probe + ".o")
subprocess.run([link[0], "-std=c++17", "-O3", "-g", "-pg", "-DSPARTA_MAP",
                "-I" + str(source / "src"), "-c",
                str(source / ("tools/performance_review/" + probe + ".cpp")),
                "-o", str(obj)], check=True)
link = [str(obj) if x.endswith("main.cpp.o") else x for x in link]
link[link.index("-o") + 1] = str(work / probe)
subprocess.run(link, cwd=work, check=True)
