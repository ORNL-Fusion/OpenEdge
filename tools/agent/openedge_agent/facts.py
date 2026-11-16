"""Facts about a run, derived from the commands of a read deck."""
import json
import os

from . import paths
from .deck import fmt_number

with open(os.path.join(paths.DATA_DIR, "rules.json")) as _f:
    RULES = json.load(_f)

INPUT_EXT = (".surf", ".species", ".recycle", ".h5", ".inc", ".reactions", ".txt", ".dat",
             ".equ", ".grid", ".csv")


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _kw(args, key):
    for i, a in enumerate(args[:-1]):
        if a == key:
            return args[i + 1]
    return None


class Facts(dict):
    pass


def extract(deck, index=None):
    f = Facts(dt=None, runs=[], fixes=[], computes=[], species_files=[], recycle_files=[],
              input_files=[], output_paths=[], pusher={}, step_settings=[], surf_react=[],
              background_files=[], dumps=[], stats=None, mixtures={}, unknown=list(deck.unknown))
    dt, fixes = None, {}
    for c in deck.commands:
        a = c.words
        if c.name == "timestep" and a:
            dt = _num(a[0])
            f["dt"] = dt if f["dt"] is None else f["dt"]
        elif c.name == "run" and a:
            n = _num(a[0])
            f["runs"].append({"steps": int(n) if n is not None else None, "dt": dt,
                              "upto": "upto" in a, "where": c.where,
                              "conditional": c.conditional})
        elif c.name == "fix" and len(a) >= 2:
            fx = {"id": a[0], "style": a[1], "args": a[2:], "where": c.where,
                  "conditional": c.conditional, "order": len(f["fixes"])}
            f["fixes"].append(fx)
            fixes[a[0]] = fx
            if a[1] == "background":
                fl = _kw(a, "file")
                if fl:
                    f["background_files"].append((fl, c.where))
        elif c.name == "unfix" and a:
            for fx in f["fixes"]:
                if fx["id"] == a[0]:
                    fx["unfixed_at"] = c.where
        elif c.name == "compute" and len(a) >= 2:
            f["computes"].append({"id": a[0], "style": a[1], "args": a[2:], "where": c.where})
        elif c.name == "species" and a:
            f["species_files"].append((a[0], c.where))
            f["species"] = a[1:]
        elif c.name == "mixture" and a:
            f["mixtures"].setdefault(a[0], []).extend(x for x in a[1:] if x[:1].isalpha())
        elif c.name == "surf_react" and len(a) >= 3:
            sr = {"id": a[0], "style": a[1], "file": a[2], "args": a[3:], "where": c.where,
                  "conditional": c.conditional}
            f["surf_react"].append(sr)
            if a[1] == "surface/pwi":
                f["recycle_files"].append((a[2], c.where))
        elif c.name == "global" and a and a[0] == "pusher":
            p = f["pusher"]
            p["where"] = c.where
            for key in ("mode", "subcycles", "bad_dt_check", "bad_dt_limit", "sheath"):
                v = _kw(a, key)
                if v is not None:
                    p[key] = v
        elif c.name == "stats" and a:
            f["stats"] = _num(a[0])
        elif c.name == "dump" and len(a) >= 5:
            f["dumps"].append({"id": a[0], "style": a[1], "every": _num(a[3]), "file": a[4],
                               "where": c.where})
            f["output_paths"].append((a[4], c.where))
        elif c.name == "restart" and len(a) >= 2:
            f["output_paths"].append((a[1], c.where))
        elif c.name in ("write_restart", "write_grid", "write_surf", "log") and a:
            f["output_paths"].append((a[0], c.where))
        if c.name in ("read_surf", "include", "read_restart", "read_grid", "read_particles",
                      "read_isurf") and a:
            f["input_files"].append((a[0], c.where, c.conditional))
    f["fix_by_id"] = fixes
    steps = [r["steps"] for r in f["runs"]]
    f["total_steps"] = None if any(s is None for s in steps) else sum(s for s in steps)
    f["physical_time_s"] = (f["total_steps"] * f["dt"]) if f["total_steps"] and f["dt"] else None
    f["step_settings"] = step_settings(f, index)
    return f


def step_settings(f, index):
    """Every setting counted in timesteps, with its physical interval."""
    out, dt = [], f["dt"]
    physics = set(RULES["physics_fix_styles"])

    def add(where, what, steps, phys, cond, note=None):
        n = _num(steps)
        out.append({"where": where, "setting": what, "steps": n,
                    "interval_s": (n * dt) if (n is not None and dt) else None,
                    "physics": phys, "conditional": cond, "note": note})

    for fx in f["fixes"]:
        style, args = fx["style"], fx["args"]
        pos = RULES["positional_steps"].get(style)
        if pos is None and index is not None:
            e = index["styles"].get("fix", {}).get(style) or {}
            pos = e.get("step_args")
        for name, i in pos or []:
            if i < len(args):
                add(fx["where"], "fix %s %s %s" % (fx["id"], style, name), args[i],
                    style in physics, fx["conditional"])
        for kw in RULES["keyword_steps"]:
            if kw["command"] == "fix" and kw["style"] == style:
                v = _kw(args, kw["keyword"])
                add(fx["where"], "fix %s %s %s" % (fx["id"], style, kw["keyword"]),
                    v if v is not None else kw["default"], kw["physics"], fx["conditional"],
                    kw["note"] + ("" if v is not None else " (default)"))
    for sr in f["surf_react"]:
        for kw in RULES["keyword_steps"]:
            if kw["command"] == "surf_react" and kw["style"] == sr["style"]:
                v = _kw(sr["args"], kw["keyword"])
                add(sr["where"], "surf_react %s %s" % (sr["id"], kw["keyword"]),
                    v if v is not None else kw["default"], kw["physics"], sr["conditional"],
                    kw["note"] + ("" if v is not None else " (default)"))
    for d in f["dumps"]:
        add(d["where"], "dump %s every" % d["id"], d["every"], False, False)
    if f["stats"] is not None:
        add("", "stats", f["stats"], False, False)
    return out


def summary(f):
    return {
        "dt": f["dt"], "total_steps": f["total_steps"],
        "physical_time_s": f["physical_time_s"],
        "runs": f["runs"],
        "pusher": f["pusher"],
        "fixes": [{"id": x["id"], "style": x["style"], "where": x["where"],
                   "conditional": x["conditional"]} for x in f["fixes"]],
        "surf_react": [{"id": s["id"], "style": s["style"], "file": s["file"]}
                       for s in f["surf_react"]],
        "species": f.get("species"),
        "step_settings": [dict(s, interval_s=fmt_number(s["interval_s"]) if s["interval_s"] else None)
                          for s in f["step_settings"]],
        "unknown": f["unknown"],
    }
