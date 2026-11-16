"""Operations shared by the CLI and the MCP server. Every function returns JSON-able data.

Gates (enforced here, not by prompts):
  * submit needs an approval of that exact plan id; approval needs an interactive
    terminal (see approve()); any change to the deck or inputs after planning
    invalidates the plan;
  * one job per plan; smoke runs are capped at SMOKE_CAP steps;
  * nothing here edits src/ or runs git commit/push; nothing polls in the background.
"""
import getpass
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time

from . import checks, database, facts as factsmod, index as indexmod, paths
from .deck import read_deck

SMOKE_CAP = 10000


# ---------------------------------------------------------------- helpers

def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        n = 0
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
            if limit and n >= limit:
                break
    return h.hexdigest()


def parse_vars(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError("variable '%s' must be NAME=VALUE" % it)
        k, v = it.split("=", 1)
        out[k] = v
    return out


def load_profile(name):
    for d in (os.path.join(paths.state_dir(), "profiles"), os.path.join(paths.DATA_DIR, "profiles")):
        p = os.path.join(d, name + ".json")
        if os.path.isfile(p):
            with open(p) as f:
                prof = json.load(f)
            prof["_file"] = p
            return prof
    raise ValueError("no machine profile '%s' (looked in ~/.openedge-agent/profiles and the shipped "
                     "profiles: %s)" % (name, ", ".join(list_profiles())))


def list_profiles():
    names = set()
    for d in (os.path.join(paths.state_dir(), "profiles"), os.path.join(paths.DATA_DIR, "profiles")):
        if os.path.isdir(d):
            names.update(x[:-5] for x in os.listdir(d) if x.endswith(".json"))
    return sorted(names)


def _context(deck_path, var=None, run_dir=None, binary=None, root=None, np=None,
             reference_var=None):
    overrides = parse_vars(var)
    deck = read_deck(deck_path, overrides, run_dir)
    root = paths.openedge_root(root)
    idx = indexmod.load(root) if root else None
    f = factsmod.extract(deck, idx)
    ref = None
    if reference_var:
        rv = dict(overrides)
        rv.update(parse_vars(reference_var))
        ref = factsmod.extract(read_deck(deck_path, rv, run_dir), idx)
    db_dir, db_source = database.database_dir(binary, deck.run_dir)
    if db_dir is None and root:
        db_dir, db_source = os.path.join(root, "database"), "source tree (no binary given; " \
            "the binary may read another database)"
    return {"deck": deck, "facts": f, "index": idx, "root": root, "binary": binary,
            "db_dir": db_dir, "db_source": db_source, "np": np, "reference": ref}


# ---------------------------------------------------------------- read-only operations

def inspect(deck, var=None, run_dir=None):
    ctx = _context(deck, var, run_dir)
    s = factsmod.summary(ctx["facts"])
    s["deck"] = ctx["deck"].path
    s["run_dir"] = ctx["deck"].run_dir
    s["files_read"] = ctx["deck"].files
    s["variables"] = {k: v[1] for k, v in ctx["deck"].variables.items()}
    return s


def preflight(deck, var=None, run_dir=None, binary=None, np=None, reference_var=None, root=None):
    ctx = _context(deck, var, run_dir, binary, root, np, reference_var)
    verdict, results = checks.run_all(ctx)
    return {"verdict": verdict, "deck": ctx["deck"].path, "run_dir": ctx["deck"].run_dir,
            "binary": binary, "database": ctx["db_dir"],
            "counts": {lvl: sum(r["level"] == lvl for r in results)
                       for lvl in ("error", "warn", "unknown", "info", "ok")},
            "results": results,
            "step_settings": ctx["facts"]["step_settings"]}


def explain(log_text):
    hits = []
    for e in checks.CATALOG.values():
        sig = e.get("signature")
        if not sig:
            continue
        for m in re.finditer(sig, log_text):
            line = log_text[log_text.rfind("\n", 0, m.start()) + 1:log_text.find("\n", m.end())]
            hits.append({"catalog": e["id"], "title": e["title"], "line": line.strip()[:400],
                         "cause": e["cause"], "fix": e["fix"]})
            break
    errs = [ln.strip() for ln in log_text.splitlines() if ln.lstrip().startswith("ERROR")]
    return {"matches": hits, "error_lines": errs[:20],
            "unmatched": bool(errs) and not hits}


def catalog(query=None):
    out = []
    for e in checks.CATALOG.values():
        txt = json.dumps(e).lower()
        if not query or all(w in txt for w in query.lower().split()):
            out.append(e)
    with open(os.path.join(paths.DATA_DIR, "changes.json")) as f:
        ch = json.load(f)["entries"]
    if query:
        ch = [c for c in ch if all(w in json.dumps(c).lower() for w in query.lower().split())]
    return {"failures": out, "changes": ch}


def style_info(kind, style, root=None):
    root = paths.openedge_root(root)
    if not root:
        raise ValueError("OpenEdge source tree not found: pass --root or set OPENEDGE_ROOT")
    idx = indexmod.load(root)
    kind = {"surf_react": "surfreact", "surf_collide": "surfcollide"}.get(kind, kind)
    e = indexmod.lookup(idx, kind, style)
    if e is None:
        near = [s for s in idx["styles"].get(kind, {}) if style.split("/")[0] in s]
        return {"found": False, "kind": kind, "style": style, "similar": near[:20],
                "git_head": idx["git_head"]}
    doc = None
    if e.get("doc"):
        with open(os.path.join(root, e["doc"]), errors="replace") as f:
            doc = f.read()
    return {"found": True, "kind": kind, "style": style, "git_head": idx["git_head"],
            "entry": e, "doc_text": doc}


def health(log_path):
    """Progress and problems from a log or stdout file (read once, no polling)."""
    with open(log_path, errors="replace") as f:
        text = f.read()
    rows, header = [], None
    for ln in text.splitlines():
        w = ln.split()
        if w and w[0] == "Step":
            header = w
            continue
        if header and w and len(w) == len(header) and re.fullmatch(r"\d+", w[0]):
            rows.append(dict(zip(header, w)))
        elif header and w and not re.fullmatch(r"\d+", w[0]):
            header = None if w[0] in ("Loop", "ERROR") else header
    loops = re.findall(r"Loop time of (\S+) on (\d+) procs for (\d+) steps with (\d+) particles", text)
    out = {"log": log_path, "stats_rows": len(rows), "last": rows[-1] if rows else None,
           "loops": [{"seconds": float(a), "procs": int(b), "steps": int(c), "particles": int(d),
                      "steps_per_s": (int(c) / float(a)) if float(a) > 0 else None}
                     for a, b, c, d in loops],
           "warnings": sorted(set(ln.strip()[:300] for ln in text.splitlines()
                                  if "WARNING" in ln))[:30],
           "dt_monitor": [ln.strip() for ln in text.splitlines() if "status:" in ln][:40],
           "explain": explain(text)}
    if len(rows) >= 2 and "Np" in rows[-1]:
        out["np_first_last"] = [rows[0]["Np"], rows[-1]["Np"]]
    if len(rows) >= 2 and "CPU" in rows[-1]:
        try:
            ds = int(rows[-1]["Step"]) - int(rows[0]["Step"])
            dc = float(rows[-1]["CPU"]) - float(rows[0]["CPU"])
            if dc > 0 and ds > 0:
                out["steps_per_s_recent"] = ds / dc
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------- plans

def _plans_dir():
    d = os.path.join(paths.state_dir(), "plans")
    os.makedirs(d, exist_ok=True)
    return d


def _input_hashes(ctx):
    files = list(ctx["deck"].files)
    fc = ctx["facts"]
    for p, _ in fc["species_files"] + fc["recycle_files"] + fc["background_files"]:
        files.append(p)
    files += [p for p, _, _ in fc["input_files"]]
    out = {}
    for p in files:
        full = p if os.path.isabs(p) else os.path.join(ctx["deck"].run_dir, p)
        if os.path.isfile(full):
            # large binary inputs: hash the first 64 MB plus size and mtime
            st = os.stat(full)
            big = st.st_size > (64 << 20)
            out[os.path.abspath(full)] = (sha256(full, 64 << 20) + ("+%d" % st.st_size if big else ""))
    return out


def _smoke_deck(ctx, cap, out_path):
    """Copy of the deck with a step cap (fix halt) inserted before the first run."""
    d, fc = ctx["deck"], ctx["facts"]
    if any(c.name == "read_restart" for c in d.commands):
        raise ValueError("smoke cap not applied: the deck reads a restart file (step numbers do "
                         "not start at 0); run a short deck instead")
    first = next((c for c in d.commands if c.name == "run"), None)
    if first is None:
        raise ValueError("no run command")
    if os.path.abspath(first.file) != d.path:
        raise ValueError("the first run command is in an included file (%s): smoke cap not "
                         "applied" % first.where)
    with open(d.path) as f:
        lines = f.read().splitlines()
    ins = ["variable openedge_agent_step equal step",
           "fix openedge_agent_cap halt 1 v_openedge_agent_step >= %d error soft message yes" % cap]
    lines[first.line - 1:first.line - 1] = ins
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return {"inserted_before": first.where, "cap": cap,
            "note": "each later run command executes at most one step after the cap"}


def plan(deck, profile="laptop", np=1, binary=None, var=None, run_dir_mode="in-place",
         smoke=None, steps_per_second=None, reference_var=None, label=None, root=None):
    prof = load_profile(profile)
    if np < 1 or np > prof.get("max_np", 1 << 20):
        raise ValueError("np %d outside the profile's limit (max_np %s)" % (np, prof.get("max_np")))
    if not binary or not os.path.isfile(binary) or not os.access(binary, os.X_OK):
        raise ValueError("binary '%s' is not an executable file" % binary)
    if smoke is not None and not (0 < smoke <= SMOKE_CAP):
        raise ValueError("smoke steps must be in 1..%d" % SMOKE_CAP)
    pf = preflight(deck, var, None, binary, np, reference_var, root)
    if pf["verdict"] == "fail":
        return {"planned": False, "reason": "preflight failed", "preflight": pf}
    ctx = _context(deck, var, None, binary, root, np)
    fc = ctx["facts"]
    steps = fc["total_steps"]
    if smoke is not None and steps is not None:
        steps = min(steps, smoke)
    est = None
    if steps_per_second and steps:
        wall_h = steps / float(steps_per_second) / 3600.0
        nodes = (prof.get("sbatch") or {}).get("nodes", 1) if prof["scheduler"] == "slurm" else 0
        est = {"wall_hours": round(wall_h, 3), "node_hours": round(wall_h * nodes, 3) if nodes else None,
               "basis": "%g steps/s supplied (measure it with a smoke run)" % steps_per_second}
    body = {
        "label": label, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "deck": ctx["deck"].path, "case_dir": ctx["deck"].run_dir, "vars": parse_vars(var),
        "profile": prof["name"], "profile_file": prof["_file"], "scheduler": prof["scheduler"],
        "np": np, "binary": os.path.abspath(binary), "binary_sha256": sha256(binary),
        "database": pf["database"], "run_dir_mode": run_dir_mode, "smoke_steps": smoke,
        "steps": steps, "dt": fc["dt"], "jobs": 1, "estimate": est,
        "preflight": {"verdict": pf["verdict"], "counts": pf["counts"],
                      "attention": [r for r in pf["results"] if r["level"] in ("warn", "unknown")]},
        "inputs": _input_hashes(ctx), "env": prof.get("env", {}),
        "launcher": prof["launcher"].format(np=np), "binary_args": prof.get("binary_args", []),
    }
    pid = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    body["plan_id"] = pid
    with open(os.path.join(_plans_dir(), pid + ".json"), "w") as f:
        json.dump(body, f, indent=1)
    return {"planned": True, "plan_id": pid, "plan": body,
            "next": "A person approves it in a terminal: openedge-agent approve %s" % pid}


def get_plan(pid):
    p = os.path.join(_plans_dir(), pid + ".json")
    if not re.fullmatch(r"[0-9a-f]{16}", pid or "") or not os.path.isfile(p):
        raise ValueError("no plan '%s'" % pid)
    with open(p) as f:
        return json.load(f)


def _approval_path(pid):
    return os.path.join(_plans_dir(), pid + ".approved")


def approve(pid, stdin=None, stdout=None):
    """Interactive approval. Refuses without a terminal: an agent driving the CLI
    through a non-interactive shell cannot approve. This is a speed bump, not a
    security boundary: anyone with a terminal on this account can approve."""
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    if not (stdin.isatty() and stdout.isatty()):
        raise PermissionError("approval needs an interactive terminal; ask the person to run "
                              "`openedge-agent approve %s` themselves" % pid)
    p = get_plan(pid)
    print(render_plan(p), file=stdout)
    stdout.write("To approve, type %s (the first 6 characters of the plan id): " % pid[:6])
    stdout.flush()
    ans = stdin.readline().strip()
    if ans != pid[:6]:
        raise PermissionError("not approved")
    rec = {"plan_id": pid, "plan_sha256": sha256(os.path.join(_plans_dir(), pid + ".json")),
           "by": getpass.getuser(), "host": socket.gethostname(),
           "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(_approval_path(pid), "w") as f:
        json.dump(rec, f)
    return rec


def approved(pid):
    """An approval record for this plan id and this exact plan file. The record is a file:
    anyone (or any agent) with write access to the state directory could forge it."""
    ap = _approval_path(pid)
    if not os.path.isfile(ap):
        return False
    try:
        with open(ap) as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return False
    return (rec.get("plan_id") == pid and
            rec.get("plan_sha256") == sha256(os.path.join(_plans_dir(), pid + ".json")))


def render_plan(p):
    lines = ["Plan %s  %s" % (p["plan_id"], p.get("label") or ""),
             "  deck      %s" % p["deck"],
             "  vars      %s" % (" ".join("%s=%s" % kv for kv in p["vars"].items()) or "-"),
             "  machine   %s (%s), %d ranks, %d job" % (p["profile"], p["scheduler"], p["np"], p["jobs"]),
             "  binary    %s (sha256 %s)" % (p["binary"], p["binary_sha256"][:12]),
             "  steps     %s%s, dt %s" % (p["steps"], " (smoke cap)" if p["smoke_steps"] else "", p["dt"]),
             "  cost      %s" % (("%.2f wall h, %s node h" % (p["estimate"]["wall_hours"],
                                  p["estimate"]["node_hours"])) if p["estimate"] else
                                 "unknown: no measured throughput"),
             "  preflight %s %s" % (p["preflight"]["verdict"], p["preflight"]["counts"])]
    for r in p["preflight"]["attention"]:
        lines.append("    [%s] %s %s" % (r["level"], r["where"], r["message"]))
    return "\n".join(lines)


def _runs_dir():
    d = os.path.join(paths.state_dir(), "runs")
    os.makedirs(d, exist_ok=True)
    return d


def submit(pid, root=None):
    p = get_plan(pid)
    if not approved(pid):
        raise PermissionError("plan %s is not approved; a person must run `openedge-agent approve %s`"
                              % (pid, pid))
    if os.path.isfile(os.path.join(_runs_dir(), pid, "manifest.json")):
        raise PermissionError("plan %s was already submitted (one job per approval)" % pid)
    changed = [f for f, h in p["inputs"].items()
               if not os.path.isfile(f) or (sha256(f, 64 << 20) +
                                            ("+%d" % os.path.getsize(f) if os.path.getsize(f) > (64 << 20) else "")) != h]
    if changed or sha256(p["binary"]) != p["binary_sha256"]:
        raise PermissionError("inputs or binary changed since planning: %s; make a new plan"
                              % (changed or ["binary"]))
    rd = os.path.join(_runs_dir(), pid)
    os.makedirs(rd, exist_ok=True)
    deck_in = p["deck"]
    smoke = None
    if p["smoke_steps"]:
        ctx = _context(p["deck"], ["%s=%s" % kv for kv in p["vars"].items()], None, p["binary"], root)
        deck_in = os.path.join(rd, "in.smoke")
        smoke = _smoke_deck(ctx, p["smoke_steps"], deck_in)
    cmd = shlex.split(p["launcher"]) + [p["binary"]] + p["binary_args"] + ["-in", deck_in]
    for k, v in p["vars"].items():
        cmd += ["-var", k, str(v)]
    cmd += ["-log", os.path.join(rd, "log.openedge")]
    env = dict(os.environ)
    env.update({k: str(v) for k, v in p["env"].items()})
    manifest = {"plan": p, "approval": json.load(open(_approval_path(pid))), "command": cmd,
                "cwd": p["case_dir"], "run_record": rd, "smoke": smoke,
                "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if p["scheduler"] == "local":
        if shutil.which(cmd[0]) is None:
            raise RuntimeError("launcher '%s' not found" % cmd[0])
        out = open(os.path.join(rd, "stdout.txt"), "w")
        proc = subprocess.Popen(cmd, cwd=p["case_dir"], env=env, stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True)
        manifest["pid"] = proc.pid
    elif p["scheduler"] == "slurm":
        prof = load_profile(p["profile"])
        script = os.path.join(rd, "job.sbatch")
        with open(script, "w") as f:
            f.write(_sbatch(prof, p, cmd, rd))
        if shutil.which("sbatch") is None:
            raise RuntimeError("sbatch not found on this machine; the job script is at %s" % script)
        r = subprocess.run(["sbatch", "--parsable", script], cwd=p["case_dir"], capture_output=True,
                           text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError("sbatch failed: %s" % r.stderr.strip())
        manifest["job_id"] = r.stdout.strip().split(";")[0]
    else:
        raise ValueError("unknown scheduler '%s'" % p["scheduler"])
    with open(os.path.join(rd, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return {"submitted": True, "plan_id": pid, "run_record": rd,
            "pid": manifest.get("pid"), "job_id": manifest.get("job_id"),
            "log": os.path.join(rd, "log.openedge")}


def _sbatch(prof, p, cmd, rd):
    """Job script for a Slurm profile. Untested in this package's own test suite."""
    lines = ["#!/bin/bash"]
    for k, v in (prof.get("sbatch") or {}).items():
        if v is None:
            raise ValueError("profile %s: sbatch option '%s' is not set" % (prof["name"], k))
        lines.append("#SBATCH --%s=%s" % (k, str(v).format(np=p["np"])))
    lines += ["#SBATCH --job-name=oe-%s" % p["plan_id"][:8],
              "#SBATCH --output=%s/stdout.txt" % rd, "set -euo pipefail"]
    lines += ["module load %s" % m for m in prof.get("modules", [])]
    lines += prof.get("setup", [])
    lines += ["export %s=%s" % (k, shlex.quote(str(v))) for k, v in p["env"].items()]
    lines += ["cd %s" % shlex.quote(p["case_dir"]), " ".join(shlex.quote(c) for c in cmd)]
    return "\n".join(lines) + "\n"


def status(pid):
    rd = os.path.join(_runs_dir(), pid)
    mf = os.path.join(rd, "manifest.json")
    if not os.path.isfile(mf):
        raise ValueError("no submitted run for plan %s" % pid)
    with open(mf) as f:
        m = json.load(f)
    st = {"plan_id": pid, "run_record": rd}
    if "pid" in m:
        try:  # reap our own child (MCP server) so a finished run does not read as running
            done, code = os.waitpid(m["pid"], os.WNOHANG)
            if done:
                st["exit_code"] = os.waitstatus_to_exitcode(code) if hasattr(os, "waitstatus_to_exitcode") else code
        except ChildProcessError:
            pass
        try:
            os.kill(m["pid"], 0)
            ps = subprocess.run(["ps", "-o", "stat=", "-p", str(m["pid"])], capture_output=True,
                                text=True, timeout=30).stdout.strip()
            st["state"] = "finished" if (not ps or ps.startswith("Z")) else "running"
        except ProcessLookupError:
            st["state"] = "finished"
        except PermissionError:
            st["state"] = "running (other user)"
    elif "job_id" in m and shutil.which("squeue"):
        r = subprocess.run(["squeue", "-h", "-j", m["job_id"], "-o", "%T"], capture_output=True,
                           text=True, timeout=60)
        st["state"] = r.stdout.strip()
        if not st["state"] and shutil.which("sacct"):  # left the queue: final state from accounting
            r = subprocess.run(["sacct", "-n", "-X", "-P", "-j", m["job_id"], "-o", "State,ExitCode"],
                               capture_output=True, text=True, timeout=60)
            st["state"] = (r.stdout.strip().splitlines() or [""])[0].replace("|", " exit ")
        st["state"] = st["state"] or "not in queue (finished or failed)"
        st["job_id"] = m["job_id"]
    log = os.path.join(rd, "log.openedge")
    if os.path.isfile(log):
        st["health"] = health(log)
    return st


def collect(pid):
    """Run record: manifest, log summary, list of output files newer than submission."""
    st = status(pid)
    rd = st["run_record"]
    with open(os.path.join(rd, "manifest.json")) as f:
        m = json.load(f)
    t0 = os.path.getmtime(os.path.join(rd, "manifest.json")) - 5
    outs = []
    for base, _, files in os.walk(m["cwd"]):
        if len(outs) > 2000:
            break
        for fn in files:
            fp = os.path.join(base, fn)
            try:
                if os.path.getmtime(fp) >= t0:
                    outs.append(fp)
            except OSError:
                pass
    rec = {"status": st, "outputs": sorted(outs)[:2000], "manifest": m}
    with open(os.path.join(rd, "record.json"), "w") as f:
        json.dump(rec, f, indent=1, default=str)
    return rec
