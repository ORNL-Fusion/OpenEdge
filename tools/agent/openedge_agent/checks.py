"""Preflight checks. Each returns results {check, level, message, where, fix, catalog}.

level: error (the run will fail or is wrong), warn (likely wrong or costly),
unknown (could not be decided statically), info, ok.
"""
import json
import os
import re

from . import database, paths
from .deck import fmt_number

with open(os.path.join(paths.DATA_DIR, "failures.json")) as _f:
    CATALOG = {e["id"]: e for e in json.load(_f)["entries"]}

KIND = {"fix": "fix", "compute": "compute", "surf_react": "surfreact",
        "surf_collide": "surfcollide", "dump": "dump"}
LONG_RUN_STEPS = 1_000_000


def R(check, level, message, where="", catalog=None, fix=None):
    if fix is None and catalog in CATALOG:
        fix = CATALOG[catalog]["fix"]
    return {"check": check, "level": level, "message": message, "where": where,
            "catalog": catalog, "fix": fix}


def _abs(ctx, p):
    return p if os.path.isabs(p) else os.path.join(ctx["deck"].run_dir, p)


def check_parse(ctx):
    out = [R("parse", "unknown", u["what"], u["where"]) for u in ctx["deck"].unknown]
    return out or [R("parse", "ok", "deck read completely (%d commands)" % len(ctx["deck"].commands))]


def check_runs(ctx):
    f = ctx["facts"]
    if not f["runs"]:
        return [R("runs", "error", "no run command found")]
    out = []
    for r in f["runs"]:
        if r["steps"] is None:
            out.append(R("runs", "unknown", "run length not determined", r["where"]))
    if f["dt"] is None:
        if any(c.name == "timestep" for c in ctx["deck"].commands):
            out.append(R("runs", "unknown", "timestep not determined"))
        else:
            out.append(R("runs", "warn", "no timestep command: the code's default timestep is used"))
    if not any(r["level"] in ("unknown", "warn") for r in out):
        out.append(R("runs", "ok", "%d run command(s), %s steps, dt %.4g s, %s s simulated" % (
            len(f["runs"]), f["total_steps"], f["dt"],
            ("%.4g" % f["physical_time_s"]) if f["physical_time_s"] else "?")))
    return out


def check_styles(ctx):
    idx, out = ctx.get("index"), []
    if idx is None:
        return [R("styles", "unknown", "no OpenEdge source tree found: styles not checked")]
    for c in ctx["deck"].commands:
        kind = KIND.get(c.name)
        if not kind:
            continue
        if len(c.words) < 2:
            continue
        style = c.words[1]  # <command> ID style ...
        if style not in idx["styles"].get(kind, {}):
            lvl = "unknown" if c.conditional else "error"
            out.append(R("styles", lvl, "%s style '%s' is not in the source tree (%s)" % (
                c.name, style, (idx.get("git_head") or "?")[:9]), c.where, "unknown-style"))
    return out or [R("styles", "ok", "every style exists in the source tree")]


def check_files(ctx):
    f, out, seen = ctx["facts"], [], set()
    items = [(p, w, False) for p, w in f["species_files"]]
    items += [(p, w, False) for p, w in f["recycle_files"]]
    items += [(p, w, False) for p, w in f["background_files"]]
    items += list(f["input_files"])
    for p, where, cond in items:
        if (p, where) in seen:
            continue
        seen.add((p, where))
        if "*" in p or "%" in p:
            continue
        if not os.path.exists(_abs(ctx, p)):
            out.append(R("files", "unknown" if cond else "error",
                         "input file '%s' not found (relative to %s)" % (p, ctx["deck"].run_dir),
                         where, "wrong-directory"))
    return out or [R("files", "ok", "%d input files present" % len(seen))]


def analytic_sputter_names(root):
    """Pair names of the analytic sputter catalog (src/eckstein_sputter_data.h)."""
    if not root:
        return None
    p = os.path.join(root, "src", "eckstein_sputter_data.h")
    if not os.path.isfile(p):
        return None
    with open(p, errors="replace") as f:
        return set(re.findall(r'\{"([A-Za-z]+_on_[A-Za-z]+)"', f.read()))


def check_database(ctx):
    f, out = ctx["facts"], []
    needs = f["recycle_files"] or any(x["style"] == "volume/chem/adas" for x in f["fixes"])
    if not needs:
        return [R("database", "ok", "deck uses no database tables")]
    db, source = ctx["db_dir"], ctx["db_source"]
    if db is None:
        return [R("database", "unknown", "database directory unknown: %s" % source)]
    h5path = os.path.join(db, "processes.h5")
    if not os.path.isfile(h5path):
        return [R("database", "error", "processes.h5 not found at %s (from %s)" % (h5path, source),
                  catalog="db-file")]
    h5 = database.H5(h5path)
    if h5.backend is None:
        return [R("database", "unknown", "neither h5py nor h5ls available: tables not checked")]
    if h5.groups("/") is None:
        return [R("database", "error", "%s is not readable as HDF5" % h5path, catalog="db-file")]
    out.append(R("database", "info", "binary reads %s (%s)" % (h5path, source)))
    for p, where in f["recycle_files"]:
        full = _abs(ctx, p)
        if not os.path.isfile(full):
            continue
        refl = h5.groups("/surface/reflection")
        sput = h5.groups("/surface/sputter")
        for kind, name, line in database.recycle_tables(full):
            pair = name.lower()
            w = "%s:%d" % (os.path.basename(p), line)
            if kind == "T":
                if refl is None:
                    out.append(R("database", "unknown", "reflection tables not listed", w))
                elif pair not in refl:
                    out.append(R("database", "error", "reflection table '%s' not in %s "
                                 "(/surface/reflection/%s): the run stops" % (name, h5path, pair),
                                 w, "reflection-table"))
            else:
                if sput is None:
                    out.append(R("database", "unknown", "sputter tables not listed", w))
                elif pair not in sput:
                    analytic = analytic_sputter_names(ctx.get("root"))
                    if analytic is None:
                        out.append(R("database", "unknown", "sputter table '%s' not in %s and the "
                                     "analytic catalog was not found" % (name, h5path), w))
                    elif name in analytic:
                        out.append(R("database", "warn", "sputter table '%s' not in %s: the "
                                     "analytic Eckstein/Yamamura yield is used" % (name, h5path),
                                     w, "sputter-analytic"))
                    else:
                        out.append(R("database", "error", "sputter entry '%s' is neither a table "
                                     "in %s nor in the analytic catalog (names are case-"
                                     "sensitive): the run stops" % (name, h5path), w,
                                     "sputter-analytic"))
    for fx in f["fixes"]:
        if fx["style"] != "volume/chem/adas" or len(fx["args"]) < 3:
            continue
        elem, src = fx["args"][1], fx["args"][2]
        literal = "/" in src or src.startswith(".") or src.endswith((".h5", ".reactions"))
        if literal:
            if not os.path.isfile(_abs(ctx, src)):
                out.append(R("database", "error", "reactions file '%s' not found" % src,
                             fx["where"], "no-reactions"))
            continue
        e = (elem if src == "auto" else src).lower()
        have = h5.groups("/volume/reactions")
        spec = elem if src == "auto" else src
        ingest = os.path.join(db, "ingest", "reactions", spec + ".reactions")
        if have is None:
            out.append(R("database", "unknown", "volume reactions not listed", fx["where"]))
        elif e not in have and not os.path.isfile(ingest):
            out.append(R("database", "error", "no reactions for element '%s' in %s" % (e, h5path),
                         fx["where"], "no-reactions"))
    if not any(r["level"] in ("error", "warn", "unknown") for r in out):
        out.append(R("database", "ok", "every table the deck names is in the database"))
    return out


def check_dt_monitor(ctx):
    f = ctx["facts"]
    mons = [x for x in f["fixes"] if x["style"] == "dt/monitor"]
    chem = [x for x in f["fixes"] if x["style"] == "volume/chem/adas"]
    if not mons:
        long_run = (f["total_steps"] or 0) >= LONG_RUN_STEPS
        return [R("dt_monitor", "warn" if long_run else "info",
                  "no fix dt/monitor: gyromotion and reaction intervals are not checked",
                  catalog="gyro-resolution",
                  fix="Add `fix dtm dt/monitor <N>` after the last fix volume/chem/adas.")]
    out = []
    m = mons[0]
    late = [c for c in chem if c["order"] > m["order"]]
    if late:
        out.append(R("dt_monitor", "error", "fix dt/monitor is defined before fix volume/chem/adas "
                     "%s: the run stops" % late[0]["id"], m["where"], "dt-monitor-order"))
    if m["conditional"]:
        out.append(R("dt_monitor", "unknown", "fix dt/monitor is conditional (if/variable)",
                     m["where"]))
    return out or [R("dt_monitor", "ok", "fix dt/monitor present and after the chemistry fixes")]


def check_step_settings(ctx):
    f, ref = ctx["facts"], ctx.get("reference")
    rows = f["step_settings"]
    out = [R("step_settings", "info", "%d settings are counted in steps" % len(rows))]
    if ref is None:
        return out
    if not (f["dt"] and ref["dt"]) or f["dt"] == ref["dt"]:
        return out + [R("step_settings", "info", "reference has the same timestep")]
    refmap = {r["setting"]: r for r in ref["step_settings"]}
    bad = []
    for r in rows:
        q = refmap.get(r["setting"])
        if not q or not r["physics"] or r["interval_s"] is None or q["interval_s"] is None:
            continue
        if abs(r["interval_s"] - q["interval_s"]) > 1e-6 * max(r["interval_s"], q["interval_s"]):
            if r["steps"] == 1 and q["steps"] == 1:
                continue  # every step stays every step
            bad.append(r["setting"])
            out.append(R("step_settings", "warn", "%s: %g steps = %.4g s here, %g steps = %.4g s "
                         "in the reference: physical interval changed with dt" % (
                             r["setting"], r["steps"], r["interval_s"], q["steps"], q["interval_s"]),
                         r["where"], "step-rescale"))
    if not bad:
        out.append(R("step_settings", "ok", "every physics setting keeps its physical interval "
                     "(or stays every step) against the reference"))
    return out


def check_dt_pair(ctx):
    f = ctx["facts"]
    if (f["total_steps"] or 0) < LONG_RUN_STEPS:
        return []
    return [R("dt_pair", "warn", "long run (%s steps): confirm a dt/2 comparison over a short "
              "interval before launching" % f["total_steps"], catalog="dt-convergence")]


def _kw(args, key):
    for i, a in enumerate(args[:-1]):
        if a == key:
            return args[i + 1]
    return None


def check_physics_notes(ctx):
    f, out, np_ = ctx["facts"], [], ctx.get("np") or 1
    for fx in f["fixes"]:
        if fx["style"] == "surface/emit/source":
            n = _kw(fx["args"], "nevery")
            if n and n != "1":
                out.append(R("emit_nevery", "info", "fix %s nevery %s: source window of %s steps "
                             "(meaning changed in 8ffdc1f)" % (fx["id"], n, n), fx["where"],
                             "emit-nevery"))
        if fx["style"] == "population/control":
            out.append(R("population_control", "info", "fix %s population/control: watch max/mean "
                         "weight and effective sample size" % fx["id"], fx["where"],
                         "population-weights"))
        if fx["style"] == "cross_field_diffusion" and f["surf_react"]:
            out.append(R("cross_diffusion", "info", "cross-field diffusion with wall reactions: "
                         "include it in the dt/2 comparison", fx["where"], "crossdiff-near-wall"))
    for sr in f["surf_react"]:
        if sr["style"] != "surface/pwi":
            continue
        n = _kw(sr["args"], "adens_nevery")
        if np_ > 1 and (n is None or float(n) < 100):
            out.append(R("adens_nevery", "warn", "surf_react %s adens_nevery %s on %d ranks: global "
                         "reduction every %s step(s)" % (sr["id"], n or "1 (default)", np_, n or 1),
                         sr["where"], "adens-nevery-cost"))
    return out


def check_outputs(ctx):
    f, out = ctx["facts"], []
    for p, where in f["output_paths"]:
        full = _abs(ctx, p)
        d = os.path.dirname(full)
        if not os.path.isdir(d):
            continue
        prefix = os.path.basename(full).split("*")[0]
        hits = [x for x in os.listdir(d) if x.startswith(prefix)] if prefix else os.listdir(d)
        if hits:
            out.append(R("outputs", "warn", "output '%s' would overwrite or mix with %d existing "
                         "file(s) in %s" % (p, len(hits), d), where,
                         fix="Move or rename the old outputs, or point the deck's output "
                             "variable at a new directory (--var)."))
    return out or [R("outputs", "ok", "no existing outputs at the output paths")]


ALL = [check_parse, check_runs, check_styles, check_files, check_database, check_dt_monitor,
       check_step_settings, check_dt_pair, check_physics_notes, check_outputs]


def run_all(ctx):
    results = []
    for chk in ALL:
        try:
            results.extend(chk(ctx))
        except Exception as e:  # a broken check must not read as a pass
            results.append(R(chk.__name__[6:], "unknown", "check failed: %r" % e))
    levels = {r["level"] for r in results}
    verdict = "fail" if "error" in levels else ("unknown" if "unknown" in levels else "pass")
    return verdict, results
