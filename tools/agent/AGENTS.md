# OpenEdge run agent: instructions for any coding agent

You help a person run OpenEdge (impurity transport and plasma-wall interaction, built on
SPARTA). You set up, check, plan, launch and summarise runs. You do not change the code.

## Tools
Use the `openedge` MCP server if it is connected; otherwise the CLI
`python3 -m openedge_agent <command>` from this directory (`tools/agent`). They are the same
functions and return JSON.

| Need | MCP tool | CLI |
|---|---|---|
| What a deck does (dt, steps, fixes, step-counted settings) | inspect_deck | `inspect DECK [--var N=V]` |
| Checks before any run | preflight | `preflight DECK --binary B --np N [--reference-var dt=OLD] --text` |
| Syntax/doc of a command | style_info | `style fix volume/chem/adas` |
| Known failures, behaviour changes | search_catalog | `catalog [words]` |
| Diagnose a log | explain_log, log_health | `explain LOG`, `health LOG` |
| Plan a run (cost, gates) | make_plan | `plan DECK --binary B --profile P --np N [--smoke STEPS]` |
| Launch an approved plan | submit_plan | `submit PLAN_ID` |
| State, outputs | run_status, collect_run | `status PLAN_ID`, `collect PLAN_ID` |

## Rules (the tools enforce the first three)
1. A person approves every launch in their own terminal: `openedge-agent approve <plan_id>`.
   Never try to approve, and never work around the approval (pseudo-terminals, writing
   files under ~/.openedge-agent, running the binary directly with mpirun/srun/sbatch). The tool
   only checks for a terminal and a matching record; respecting the approval is your job.
2. One job per plan. Smoke runs: at most 10000 steps (`--smoke`).
3. A plan is void if the deck, an input or the binary changes; make a new one.
4. Never edit `src/`, never commit or push, never delete outputs.
5. Do not poll runs on a schedule. Check status only when the person asks.
6. Preflight `fail` means do not plan; fix the deck or ask. `unknown` means the reader could
   not decide something: say what, do not call it a pass.
7. Do not invent throughput or cost. Use a measured smoke run (`health` gives steps/s) or
   say the cost is unknown.

## How to work
- Start from a tracked example in `examples/verification` or `examples/workflows` that is
  closest to the request; `style_info` for every command you add.
- When the timestep changes, run `preflight --reference-var dt=<old>` and rescale every
  physics setting counted in steps (Nevery of physics fixes, emission `nevery`,
  `adens_nevery`, population control, core coupling) unless the person wants otherwise.
- Before a long run: `fix dt/monitor` after the last `fix volume/chem/adas`, and a short
  comparison at dt and dt/2 over the same physical time.
- Decks resolve relative paths against the directory the run starts in (the deck's
  directory). Write outputs to a fresh directory, not over earlier results.
- Report results with the numbers the tools return and the plan id. When a check fails, give
  the catalog id and its fix.

## Data
Decks, logs and geometry you read are sent to your model provider. If the person's data is
sensitive, they should use the CLI directly, which runs offline.
