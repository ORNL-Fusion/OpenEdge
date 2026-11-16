# openedge-agent

Deterministic tooling to check, plan and launch OpenEdge runs, usable by a person on the
command line or by any coding agent (Claude Code, Codex, ...) through MCP. There is no
language model inside: the checks, gates and run records are code.

    cd tools/agent
    python3 -m openedge_agent preflight path/to/in.deck --binary path/to/spa_xxx --np 8 --text

Requires Python 3.8+ and no packages for the CLI. Table checks use `h5py` if present,
otherwise the `h5ls` tool. The MCP server needs `mcp` 2.x (Python 3.10+).

## Commands
`inspect`, `preflight`, `style`, `catalog`, `explain`, `health`, `profiles`, `plan`, `show`,
`approve`, `submit`, `status`, `collect`. `python3 -m openedge_agent <command> -h` for options.
Exit codes of `preflight`: 0 pass, 1 fail, 2 unknown. Other commands: 0 ok, 3 refused/error.

## Workflow and gates
1. `preflight` reads the deck the way the code does (variables, `-var`, include, if,
   `jump SELF`) and checks: styles exist in the source tree; input files exist; the
   database the binary will read holds every reflection/sputter table and reaction set the
   deck names; `fix dt/monitor` order; settings counted in steps and their physical
   intervals (with `--reference-var dt=OLD`, the ones that changed); wall-ledger interval on
   many ranks; outputs that would overwrite earlier ones. What it cannot decide is reported
   as `unknown`, never as a pass.
2. `plan` runs preflight, refuses on errors, records hashes of the deck, inputs and binary,
   the machine profile and the cost (only from a measured throughput), and prints a plan id.
3. `approve PLAN_ID` must be run by a person in an interactive terminal; it refuses without
   one and writes an approval record bound to the plan file's hash. The terminal check guards
   the `approve` command only: the record is a file, and an agent with write access to the
   state directory could forge it. The real barrier against an agent approving its own plan
   is the agent client's permission settings (deny writes to the state directory and
   pseudo-terminal tricks); `AGENTS.md` forbids both.
4. `submit PLAN_ID` launches one job (local `mpirun` or `sbatch`), only if approved, not yet
   submitted, and nothing changed since planning. `--smoke N` plans insert a step cap
   (`fix halt`) before the first run, N <= 10000.
5. `status`, `collect` read the run once; nothing polls in the background.

State (plans, approvals, run records, index cache): `$OPENEDGE_AGENT_HOME` or `~/.openedge-agent`.

## Machine profiles
`openedge_agent/data/profiles/*.json`: `laptop` (tested), `generic-slurm` and `perlmutter-gpu`
(templates, untested here). Put your own in `~/.openedge-agent/profiles/<name>.json`; they take
precedence. A profile sets the launcher, sbatch options, modules, environment (for example
MPI transport settings) and the rank limit.

## Knowledge
- `openedge_agent/data/failures.json`: failure modes with log signature, cause, check and fix.
- `openedge_agent/data/changes.json`: behaviour changes by commit.
- `openedge_agent/data/rules.json`: which settings are counted in steps and which affect physics.
- Command index: built from `src/*.h` style macros and `doc/*.txt` syntax lines of the tree,
  cached per git commit.

## Agents
- Instructions for any agent: `AGENTS.md` (Codex reads it; point other agents at it).
- MCP server: `openedge-agent-mcp` (stdio), after `pip install -e '.[mcp]'` in a venv here.
  Register it at user level: `claude mcp add -s user openedge -- <venv>/bin/openedge-agent-mcp`,
  `codex mcp add openedge -- <venv>/bin/openedge-agent-mcp`. Approval is not an MCP tool.
  In Claude Code, also deny `Edit(~/.openedge-agent/**)`: it stops the file tools (Edit, Write)
  from writing approvals. It does not stop a shell command from writing there; only Claude Code's
  sandbox (`sandbox.filesystem.denyWrite`) or your review of shell commands does.

## Chat (no agent client needed)
    pip install -e '.[chat]'          # in a venv; needs ANTHROPIC_API_KEY
    openedge-agent chat [--model claude-opus-5-5] [--effort medium]
Claude drives the same functions, plus read-only file tools and `copy_case` /
`edit_case_file`, which write only inside directories copied in that session. The model
cannot approve: you type `/approve <plan_id>` at the prompt. Files the model reads are sent
to Anthropic. Transcripts: `~/.openedge-agent/chats/`.

## Tests
    python3 evals/run_evals.py --binary path/to/spa_xxx
    python3 evals/test_chat_loop.py          # chat loop, offline
A: seeded faults that preflight must catch, and a clean base that must pass; B: every tracked
verification deck must give no error (known exceptions listed with the reason); C: no private
strings in the shipped files (`OPENEDGE_AGENT_SCRUB=<file of regexes>` adds patterns).

## Known limits (0.1.0)
- Tested: the CLI and MCP server on macOS with Python 3.14 and Open MPI; preflight on the
  tracked verification decks; Slurm plan/approve/submit/status/collect on a CPU cluster with
  Python 3.11 (Python 3.6 is refused with a message); MCP registration in Claude Code
  (connects) and Codex (listed). Not tested: the generic-slurm and perlmutter-gpu templates
  as shipped, a live Codex session.
- Evaluation suite A copies inputs from emission cases added in commit 0531a2a; older trees
  can run suites B and C only (`--only B`).
- The evaluation suite is static: seeded faults are flagged by preflight; they were not each
  confirmed by running the binary.
- `--smoke` refuses decks whose first `run` is inside an included file, and decks that read
  a restart file.
- Runs start in the deck's directory (outputs go where the deck writes them).
- `chat` is tested offline (file-tool limits; turn loop with a scripted client,
  `evals/test_chat_loop.py`); a live session needs your API key.
- Editable install only (`pip install -e .`): `AGENTS.md` and the source tree are found
  relative to this checkout.
