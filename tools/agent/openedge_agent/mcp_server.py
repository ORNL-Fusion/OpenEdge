"""MCP server over the openedge-agent core (stdio). Needs the `mcp` package (2.x).

Approval is deliberately not exposed: a person approves plans in a terminal
with `openedge-agent approve <plan_id>`.
"""
import json
import os

try:
    from mcp.server.mcpserver import MCPServer
    from mcp_types import ToolAnnotations
except ImportError as e:  # pragma: no cover
    raise SystemExit("openedge-agent MCP server needs the mcp package 2.x (pip install 'mcp>=2'): %s" % e)

from . import core

# Loaded into every session that has this server registered: scoped to these tools only.
INSTRUCTIONS = """Tools to check, plan and launch OpenEdge simulation runs.
- A person approves each plan in their own terminal (`openedge-agent approve <plan_id>`).
  There is no approve tool: never approve, write under ~/.openedge-agent, or launch the
  binary directly (mpirun/srun/sbatch) to get around it. Give the person the plan id.
- One job per plan; smoke runs at most 10000 steps; a plan is void if the deck, an input or
  the binary changed.
- preflight `fail`: do not plan. `unknown`: say what could not be decided; it is not a pass.
- Read run status only when the person asks; never poll on a schedule.
- Cost only from a measured throughput (log_health steps/s), otherwise say it is unknown.
Full guidance for running OpenEdge with these tools: AGENTS.md next to this server."""

server = MCPServer(name="openedge", instructions=INSTRUCTIONS, version="0.1.0")
RO = ToolAnnotations(read_only_hint=True)


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except (ValueError, PermissionError, RuntimeError, FileNotFoundError) as e:
        return {"error": type(e).__name__, "message": str(e)}


@server.tool(annotations=RO)
def inspect_deck(deck: str, var: list[str] | None = None) -> dict:
    """What an OpenEdge input deck will do: timestep, run lengths, fixes, settings counted in
    steps, unresolved parts. deck: path; var: ["NAME=VALUE", ...] as -var on the command line."""
    return _safe(core.inspect, deck, var)


@server.tool(annotations=RO)
def preflight(deck: str, binary: str | None = None, np: int = 1, var: list[str] | None = None,
              reference_var: list[str] | None = None) -> dict:
    """Checks before a run. verdict pass/fail/unknown with results (error, warn, unknown, info,
    ok), each with location, catalog id and fix. reference_var (e.g. ["dt=2e-9"]) compares the
    physical interval of every step-counted setting with the deck under those variables."""
    return _safe(core.preflight, deck, var, None, binary, np, reference_var)


@server.tool(annotations=RO)
def explain_log(log_path: str) -> dict:
    """Match a log/stdout file against the catalog of known OpenEdge failures."""
    def run():
        with open(log_path, errors="replace") as f:
            return core.explain(f.read())
    return _safe(run)


@server.tool(annotations=RO)
def search_catalog(query: str = "") -> dict:
    """Known failure modes (cause, check, fix) and behaviour changes by commit."""
    return core.catalog(query or None)


@server.tool(annotations=RO)
def style_info(kind: str, style: str) -> dict:
    """Syntax and documentation of a style (kind: fix, compute, surf_react, surf_collide, dump)
    from the OpenEdge source tree, with the tree's git commit."""
    return _safe(core.style_info, kind, style)


@server.tool(annotations=RO)
def log_health(log_path: str) -> dict:
    """Progress (steps, particles, steps/s), warnings, dt/monitor status and catalog matches of a
    log. Reads the file once; call again when the person asks, never on a schedule."""
    return _safe(core.health, log_path)


@server.tool(annotations=RO)
def list_profiles() -> list[str]:
    """Machine profiles available for plans."""
    return core.list_profiles()


@server.tool()
def make_plan(deck: str, binary: str, profile: str = "laptop", np: int = 1,
              var: list[str] | None = None, smoke_steps: int | None = None,
              steps_per_second: float | None = None, reference_var: list[str] | None = None,
              label: str | None = None) -> dict:
    """Make a run plan (runs preflight; refuses on errors). Returns plan_id and the plan text.
    Launching needs a person to approve it in a terminal: `openedge-agent approve <plan_id>`.
    smoke_steps caps a test run (<= 10000)."""
    out = _safe(core.plan, deck, profile, np, binary, var, smoke=smoke_steps,
                steps_per_second=steps_per_second, reference_var=reference_var, label=label)
    if out.get("planned"):
        out["text"] = core.render_plan(out["plan"])
    return out


@server.tool(annotations=RO)
def show_plan(plan_id: str) -> dict:
    """A plan and whether it is approved."""
    def run():
        p = core.get_plan(plan_id)
        return {"plan": p, "text": core.render_plan(p),
                "approved": core.approved(plan_id)}
    return _safe(run)


@server.tool()
def submit_plan(plan_id: str) -> dict:
    """Launch an approved plan (one job). Refuses if not approved, already submitted, or if the
    deck, inputs or binary changed since planning."""
    return _safe(core.submit, plan_id)


@server.tool(annotations=RO)
def run_status(plan_id: str) -> dict:
    """State of a submitted plan and the health of its log (read once)."""
    return _safe(core.status, plan_id)


@server.tool()
def collect_run(plan_id: str) -> dict:
    """Write the run record (manifest, status, outputs newer than submission) and return it."""
    return json.loads(json.dumps(_safe(core.collect, plan_id), default=str))


def main():
    server.run("stdio")


if __name__ == "__main__":
    main()
