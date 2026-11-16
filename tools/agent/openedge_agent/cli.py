"""openedge-agent command line. Every command prints JSON (or text with --text where offered)."""
import argparse
import json
import sys

from . import __version__, core


def _common(p):
    p.add_argument("deck", help="input deck (run directory = its directory)")
    p.add_argument("--var", action="append", metavar="NAME=VALUE",
                   help="as -var NAME VALUE on the command line of the code (repeatable)")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["chat"]:
        from . import chat
        return chat.main(argv[1:])
    ap = argparse.ArgumentParser(prog="openedge-agent", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chat", help="terminal chat driven by Claude (needs anthropic, ANTHROPIC_API_KEY)")

    p = sub.add_parser("inspect", help="what the deck will do: dt, steps, fixes, step settings")
    _common(p)

    p = sub.add_parser("preflight", help="checks before a run; exit 0 pass, 1 fail, 2 unknown")
    _common(p)
    p.add_argument("--binary", help="binary to be used (locates its database)")
    p.add_argument("--np", type=int, default=1)
    p.add_argument("--reference-var", action="append", metavar="NAME=VALUE",
                   help="compare step settings with the deck under these variables, e.g. dt=2e-9")
    p.add_argument("--root", help="OpenEdge source tree (default: $OPENEDGE_ROOT or this checkout)")
    p.add_argument("--text", action="store_true", help="human-readable output")

    p = sub.add_parser("explain", help="match a log against the failure catalog")
    p.add_argument("log")

    p = sub.add_parser("catalog", help="search failure modes and behaviour changes")
    p.add_argument("query", nargs="?")

    p = sub.add_parser("style", help="syntax and doc of a style, from the source tree")
    p.add_argument("kind", choices=["fix", "compute", "surf_react", "surf_collide", "dump", "command"])
    p.add_argument("style")
    p.add_argument("--root")

    p = sub.add_parser("health", help="progress and problems in a log file (read once)")
    p.add_argument("log")

    p = sub.add_parser("profiles", help="list machine profiles")

    p = sub.add_parser("plan", help="make a run plan (preflight + cost); prints a plan id")
    _common(p)
    p.add_argument("--profile", default="laptop")
    p.add_argument("--np", type=int, default=1)
    p.add_argument("--binary", required=True)
    p.add_argument("--smoke", type=int, metavar="STEPS", help="cap the run at STEPS (<= %d)" % core.SMOKE_CAP)
    p.add_argument("--steps-per-second", type=float, help="measured throughput, for the cost")
    p.add_argument("--reference-var", action="append", metavar="NAME=VALUE")
    p.add_argument("--label")
    p.add_argument("--root")

    p = sub.add_parser("show", help="show a plan")
    p.add_argument("plan_id")

    p = sub.add_parser("approve", help="approve a plan (interactive terminal only)")
    p.add_argument("plan_id")

    p = sub.add_parser("submit", help="launch an approved plan")
    p.add_argument("plan_id")

    p = sub.add_parser("status", help="state and log health of a submitted plan")
    p.add_argument("plan_id")

    p = sub.add_parser("collect", help="write the run record and list outputs")
    p.add_argument("plan_id")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "inspect":
            out = core.inspect(a.deck, a.var)
        elif a.cmd == "preflight":
            out = core.preflight(a.deck, a.var, None, a.binary, a.np, a.reference_var, a.root)
            if a.text:
                print(render_preflight(out))
                return {"pass": 0, "fail": 1, "unknown": 2}[out["verdict"]]
            print(json.dumps(out, indent=1, default=str))
            return {"pass": 0, "fail": 1, "unknown": 2}[out["verdict"]]
        elif a.cmd == "explain":
            with open(a.log, errors="replace") as f:
                out = core.explain(f.read())
        elif a.cmd == "catalog":
            out = core.catalog(a.query)
        elif a.cmd == "style":
            out = core.style_info(a.kind, a.style, a.root)
        elif a.cmd == "health":
            out = core.health(a.log)
        elif a.cmd == "profiles":
            out = core.list_profiles()
        elif a.cmd == "plan":
            out = core.plan(a.deck, a.profile, a.np, a.binary, a.var, smoke=a.smoke,
                            steps_per_second=a.steps_per_second, reference_var=a.reference_var,
                            label=a.label, root=a.root)
            if out.get("planned"):
                print(core.render_plan(out["plan"]), file=sys.stderr)
        elif a.cmd == "show":
            print(core.render_plan(core.get_plan(a.plan_id)))
            return 0
        elif a.cmd == "approve":
            out = core.approve(a.plan_id)
        elif a.cmd == "submit":
            out = core.submit(a.plan_id)
        elif a.cmd == "status":
            out = core.status(a.plan_id)
        elif a.cmd == "collect":
            out = core.collect(a.plan_id)
    except (ValueError, PermissionError, RuntimeError, FileNotFoundError) as e:
        print(json.dumps({"error": type(e).__name__, "message": str(e)}), file=sys.stderr)
        return 3
    print(json.dumps(out, indent=1, default=str))
    return 0


def render_preflight(out):
    order = {"error": 0, "unknown": 1, "warn": 2, "info": 3, "ok": 4}
    lines = ["preflight: %s  %s" % (out["verdict"].upper(), out["counts"]),
             "deck: %s" % out["deck"], "database: %s" % out["database"]]
    for r in sorted(out["results"], key=lambda r: order[r["level"]]):
        lines.append("[%-7s] %-16s %s%s" % (r["level"], r["check"], r["message"],
                                           ("  (%s)" % r["where"]) if r["where"] else ""))
        if r["level"] in ("error", "warn") and r.get("fix"):
            lines.append("          fix: %s" % r["fix"])
    rows = [s for s in out["step_settings"] if s["physics"]]
    if rows:
        lines.append("physics settings counted in steps:")
        for s in rows:
            lines.append("   %-55s %8g steps  %s s" % (s["setting"], s["steps"],
                                                       ("%.4g" % s["interval_s"]) if s["interval_s"] else "?"))
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
