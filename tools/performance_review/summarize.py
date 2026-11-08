#!/usr/bin/env python3
"""Summarize completed profile and paired-timing logs under OE_REVIEW_ROOT."""
import json
from pathlib import Path
import re
import statistics
import sys

def read_run(path):
    text = path.read_text()
    loops = [m for m in re.finditer(r"Loop time of (\S+) on (\d+) procs for (\d+) steps with (\d+) particles", text)
             if int(m[3]) > 0]
    if not loops:
        return None
    last = loops[-1]
    seconds, ranks, steps, particles = float(last[1]), int(last[2]), int(last[3]), int(last[4])
    sections = {}
    for line in text[last.end():].splitlines():
        m = re.match(r"(\w+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)", line)
        if m:
            sections[m[1]] = dict(zip(("min_s", "mean_s", "max_s", "varavg", "percent"), map(float, m.groups()[1:])))
    stats = []
    header = None
    for line in text.splitlines():
        t = line.split()
        if t[:3] == ["Step", "CPU", "Np"]:
            header = t
        elif header and len(t) == len(header):
            try:
                row = dict(zip(header, map(float, t)))
                if row["Step"] >= 0:
                    stats.append(row)
            except ValueError:
                pass
    diagnostics = [line.strip() for line in text.splitlines() if line.startswith("[background/perf]")][-3:]
    counts = {}
    for label, count in re.findall(r"^(Particle moves|Particles exiting|Particles stuck|Particles lost at periodic caps)\s*=\s*(\d+)", text, re.M):
        counts[label] = int(count)
    return dict(seconds=seconds, ranks=ranks, steps=steps, final_particles=particles,
                milliseconds_per_step=1000*seconds/steps, sections=sections,
                diagnostics=diagnostics, counts=counts, stats=stats)

root = Path(sys.argv[1])
result = {"profiles": {}, "release_timing_runs": {}, "release_timing_summary": {}}
for prefix in ("baseline", "patched", "kkhost"):
    for path in sorted(root.glob(prefix + "_np*/teprobe_fixed_off/run.log")):
        data = read_run(path)
        if data:
            result["profiles"][path.parent.parent.name] = data
for path in sorted(root.glob("timing_*_np*_rep*/teprobe_fixed_off/run.log")):
    data = read_run(path)
    if data:
        result["release_timing_runs"][path.parent.parent.name] = data
for ranks in (8, 48):
    summary = {}
    for variant in ("baseline", "patched"):
        samples = [v["milliseconds_per_step"] for k,v in result["release_timing_runs"].items()
                   if k.startswith(f"timing_{variant}_np{ranks}_")]
        if samples:
            summary[variant] = dict(n=len(samples), median_ms=statistics.median(samples), min_ms=min(samples), max_ms=max(samples))
    if "baseline" in summary and "patched" in summary:
        summary["speedup"] = summary["baseline"]["median_ms"] / summary["patched"]["median_ms"]
    result["release_timing_summary"][str(ranks)] = summary
print(json.dumps(result, indent=2))
