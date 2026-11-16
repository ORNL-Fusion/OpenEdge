"""Terminal chat over the openedge-agent functions, driven by Claude (Anthropic API).

The model calls the same functions as the MCP server. It cannot approve: the person
types `/approve <plan_id>` at the prompt, which runs the terminal approval. File
writes are limited to case directories this session copied (copy_case).
Needs `pip install anthropic` and ANTHROPIC_API_KEY.
"""
import json
import os
import shutil
import subprocess
import sys
import time

from . import __version__, core, paths

MODEL = "claude-opus-5-5"
MAX_RESULT = 40000          # characters of a tool result sent back to the model
MAX_READ = 200000           # bytes read_file will open
MAX_STEPS = 40              # model turns per user message

CHAT_RULES = """
## This session
You run inside `openedge-agent chat` in the person's terminal. The tools below are the
functions listed above. The person approves a plan by typing `/approve <plan_id>`; tell them
the plan id and what it will run, then wait. You cannot approve.
To set up a case: find the closest example (list_examples, read_file), copy_case it to a new
directory, edit the copy with edit_case_file, preflight, then make_plan. You can only write
inside directories copy_case created in this session.
Keep answers short and give numbers from the tools.
"""


def _schema(props, required=()):
    return {"type": "object", "properties": props, "required": list(required),
            "additionalProperties": False}


S = {"type": "string"}
SL = {"type": "array", "items": {"type": "string"}}
I = {"type": "integer"}
N = {"type": "number"}

TOOLS = [
    ("inspect_deck", "What a deck will do: dt, runs, fixes, settings counted in steps, unresolved parts.",
     _schema({"deck": S, "var": SL}, ["deck"])),
    ("preflight", "Checks before a run: verdict pass/fail/unknown and results with catalog ids and fixes. "
     "reference_var (e.g. [\"dt=2e-9\"]) compares step-counted settings with the deck under those values.",
     _schema({"deck": S, "binary": S, "np": I, "var": SL, "reference_var": SL}, ["deck"])),
    ("explain_log", "Match a log file against the catalog of known failures.", _schema({"log_path": S}, ["log_path"])),
    ("search_catalog", "Known failure modes (cause, check, fix) and behaviour changes by commit.",
     _schema({"query": S})),
    ("style_info", "Syntax and documentation of a style (kind: fix, compute, surf_react, surf_collide, dump, command).",
     _schema({"kind": S, "style": S}, ["kind", "style"])),
    ("log_health", "Progress, steps/s, warnings and catalog matches of a log (read once).",
     _schema({"log_path": S}, ["log_path"])),
    ("list_profiles", "Machine profiles available for plans.", _schema({})),
    ("make_plan", "Make a run plan (runs preflight, refuses on errors). smoke_steps caps a test run (<= 10000).",
     _schema({"deck": S, "binary": S, "profile": S, "np": I, "var": SL, "smoke_steps": I,
              "steps_per_second": N, "reference_var": SL, "label": S}, ["deck", "binary"])),
    ("show_plan", "A plan and whether it is approved.", _schema({"plan_id": S}, ["plan_id"])),
    ("submit_plan", "Launch an approved plan (one job). Refuses if not approved or anything changed.",
     _schema({"plan_id": S}, ["plan_id"])),
    ("run_status", "State of a submitted plan and its log health (read once).", _schema({"plan_id": S}, ["plan_id"])),
    ("collect_run", "Write and return the run record.", _schema({"plan_id": S}, ["plan_id"])),
    ("list_examples", "Tracked example decks (examples/verification and examples/workflows) whose path "
     "contains every word of the query.", _schema({"query": S})),
    ("read_file", "Read a text file (deck, species, recycle, README, log).", _schema({"path": S}, ["path"])),
    ("copy_case", "Copy a case directory to a new directory (must not exist). The copy becomes writable.",
     _schema({"src_dir": S, "dest_dir": S}, ["src_dir", "dest_dir"])),
    ("edit_case_file", "Replace one exact, unique occurrence of old with new in a file inside a copied case, "
     "or create the file when old is empty.", _schema({"path": S, "old": S, "new": S}, ["path", "old", "new"])),
]


class Session:
    def __init__(self, root=None):
        self.root = paths.openedge_root(root)
        self.writable = []          # case directories copied in this session

    # file tools ------------------------------------------------------------
    def _inside_copy(self, path):
        rp = os.path.realpath(path)
        return any(rp == d or rp.startswith(d + os.sep) for d in self.writable)

    def list_examples(self, query=""):
        if not self.root:
            raise RuntimeError("OpenEdge source tree not found; set OPENEDGE_ROOT")
        out = subprocess.run(["git", "-C", self.root, "ls-files", "examples/verification", "examples/workflows"],
                             capture_output=True, text=True, timeout=60).stdout.split()
        words = (query or "").lower().split()
        decks = [f for f in out if os.path.basename(f).startswith("in.") and not f.endswith((".py", ".sh"))
                 and all(w in f.lower() for w in words)]
        return {"root": self.root, "count": len(decks), "decks": decks[:200]}

    def read_file(self, path):
        path = os.path.abspath(os.path.expanduser(path))
        if os.path.getsize(path) > MAX_READ:
            raise ValueError("%s is larger than %d bytes; use log_health/explain_log for logs" % (path, MAX_READ))
        with open(path, "rb") as f:
            data = f.read()
        if b"\0" in data[:4096]:
            raise ValueError("%s is binary" % path)
        return {"path": path, "text": data.decode("utf-8", "replace")}

    def copy_case(self, src_dir, dest_dir):
        src = os.path.abspath(os.path.expanduser(src_dir))
        dest = os.path.abspath(os.path.expanduser(dest_dir))
        if not os.path.isdir(src):
            raise FileNotFoundError("no directory %s" % src)
        if os.path.exists(dest):
            raise PermissionError("%s exists; choose a new directory" % dest)
        shutil.copytree(src, dest, symlinks=True,
                        ignore=shutil.ignore_patterns("output", "output_*", "runs", "log.*", "*.log", "slurm-*.out"))
        self.writable.append(os.path.realpath(dest))
        return {"copied": src, "to": dest, "files": sorted(os.listdir(dest))}

    def edit_case_file(self, path, old, new):
        path = os.path.abspath(os.path.expanduser(path))
        if not self._inside_copy(path):
            raise PermissionError("%s is not inside a case copied in this session (copy_case first)" % path)
        if old == "":
            if os.path.exists(path):
                raise PermissionError("%s exists; give the text to replace" % path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(new)
            return {"created": path}
        with open(path) as f:
            text = f.read()
        n = text.count(old)
        if n != 1:
            raise ValueError("old text occurs %d times in %s; it must occur once" % (n, path))
        with open(path, "w") as f:
            f.write(text.replace(old, new, 1))
        return {"edited": path}

    # dispatch --------------------------------------------------------------
    def call(self, name, a):
        if name == "inspect_deck":
            return core.inspect(a["deck"], a.get("var"))
        if name == "preflight":
            return core.preflight(a["deck"], a.get("var"), None, a.get("binary"), a.get("np", 1),
                                  a.get("reference_var"), self.root)
        if name in ("explain_log", "log_health"):
            if name == "log_health":
                return core.health(a["log_path"])
            with open(a["log_path"], errors="replace") as f:
                return core.explain(f.read())
        if name == "search_catalog":
            return core.catalog(a.get("query") or None)
        if name == "style_info":
            return core.style_info(a["kind"], a["style"], self.root)
        if name == "list_profiles":
            return core.list_profiles()
        if name == "make_plan":
            out = core.plan(a["deck"], a.get("profile", "laptop"), a.get("np", 1), a["binary"], a.get("var"),
                            smoke=a.get("smoke_steps"), steps_per_second=a.get("steps_per_second"),
                            reference_var=a.get("reference_var"), label=a.get("label"), root=self.root)
            if out.get("planned"):
                out["text"] = core.render_plan(out["plan"])
                out["next"] = "The person types /approve %s" % out["plan_id"]
            return out
        if name == "show_plan":
            p = core.get_plan(a["plan_id"])
            return {"plan": p, "text": core.render_plan(p), "approved": core.approved(a["plan_id"])}
        if name == "submit_plan":
            return core.submit(a["plan_id"], self.root)
        if name == "run_status":
            return core.status(a["plan_id"])
        if name == "collect_run":
            return core.collect(a["plan_id"])
        if hasattr(self, name) and name in {t[0] for t in TOOLS}:
            return getattr(self, name)(**a)
        raise ValueError("unknown tool %s" % name)


def _system():
    with open(os.path.join(paths.AGENT_DIR, "AGENTS.md")) as f:
        return f.read() + CHAT_RULES


def _result_text(obj):
    s = json.dumps(obj, indent=1, default=str)
    if len(s) > MAX_RESULT:
        s = s[:MAX_RESULT] + "\n... [truncated %d characters]" % (len(s) - MAX_RESULT)
    return s


def _short(a):
    s = json.dumps(a)
    return s if len(s) < 160 else s[:157] + "..."


def add_user(messages, text):
    """Append the person's text; after tool results it joins that user turn."""
    if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list):
        messages[-1]["content"].append({"type": "text", "text": text})
    else:
        messages.append({"role": "user", "content": text})


def _drop_pending_text(messages):
    """On a failed request, drop the person's unanswered text so the history stays valid."""
    if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], str):
        messages.pop()


def run_turn(client, model, effort, system, tools, messages, sess, record=lambda *a: None,
             api_error=Exception, out=sys.stdout):
    """Model/tool rounds for one message of the person. Every tool_use kept in the history
    gets a tool_result, whatever happens (interrupt, truncation, tool error)."""
    for _ in range(MAX_STEPS):
        try:
            with client.messages.stream(model=model, max_tokens=32000, system=system, tools=tools,
                                        messages=messages, output_config={"effort": effort}) as st:
                started = False
                for text in st.text_stream:
                    if not started:
                        out.write("agent> ")
                        started = True
                    out.write(text)
                    out.flush()
                msg = st.get_final_message()
            if started:
                out.write("\n")
        except KeyboardInterrupt:
            out.write("\n[interrupted]\n")
            _drop_pending_text(messages)
            return "interrupted"
        except api_error as e:
            out.write("\n[API error: %s]\n" % e)
            if getattr(e, "status_code", None) == 401:
                out.write("[authentication failed: use an Anthropic Console API key (sk-ant-api...) in "
                          "ANTHROPIC_API_KEY, and unset ANTHROPIC_AUTH_TOKEN if it is set]\n")
            _drop_pending_text(messages)
            return "api_error"
        content = list(msg.content)
        if msg.stop_reason in ("max_tokens", "refusal"):
            content = [b for b in content if b.type != "tool_use"  # a cut tool call cannot run
                       and not (b.type == "text" and not b.text)]
            if content:
                messages.append({"role": "assistant", "content": content})
            else:
                _drop_pending_text(messages)
            out.write("[stopped: %s]\n" % msg.stop_reason)
            return msg.stop_reason
        messages.append({"role": "assistant", "content": content})
        record("assistant", [b.model_dump() for b in content if b.type in ("text", "tool_use")])
        if msg.stop_reason == "pause_turn":
            continue
        uses = [b for b in content if b.type == "tool_use"]
        if not uses:
            return "done"
        results, interrupted = [], False
        for u in uses:
            if interrupted:
                out_obj, err = {"error": "Interrupted", "message": "the person interrupted before this ran"}, True
            else:
                out.write("  · %s %s\n" % (u.name, _short(u.input)))
                try:
                    out_obj, err = sess.call(u.name, u.input), False
                except KeyboardInterrupt:
                    interrupted = True
                    out_obj, err = {"error": "Interrupted", "message": "the person interrupted this tool"}, True
                except Exception as e:  # tool errors go back to the model
                    out_obj, err = {"error": type(e).__name__, "message": str(e)}, True
            record("tool", {"name": u.name, "input": u.input, "error": err})
            results.append({"type": "tool_result", "tool_use_id": u.id,
                            "content": _result_text(out_obj), "is_error": err})
        messages.append({"role": "user", "content": results})
        if interrupted:
            out.write("[interrupted]\n")
            return "interrupted"
    out.write("[stopped after %d model turns; say 'continue' to go on]\n" % MAX_STEPS)
    return "max_steps"


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="openedge-agent chat", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.environ.get("OPENEDGE_AGENT_MODEL", MODEL))
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--root", help="OpenEdge source tree")
    a = ap.parse_args(argv)
    try:
        import anthropic
    except ImportError:
        sys.exit("openedge-agent chat needs the anthropic package: pip install anthropic")
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        sys.exit("openedge-agent chat is interactive; run it in a terminal")
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit("set ANTHROPIC_API_KEY (an Anthropic API key) to use openedge-agent chat")
    client = anthropic.Anthropic()
    sess = Session(a.root)
    tools = [{"name": n, "description": d, "input_schema": s} for n, d, s in TOOLS]
    system = [{"type": "text", "text": _system(), "cache_control": {"type": "ephemeral"}}]
    log_dir = os.path.join(paths.state_dir(), "chats")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, time.strftime("%Y%m%d-%H%M%S") + ".jsonl")
    messages = []
    print("openedge-agent %s chat (%s). Source tree: %s" % (__version__, a.model, sess.root))
    print("Decks, logs and files the model reads are sent to Anthropic. /approve <plan_id> to approve, "
          "/quit to leave. Transcript: %s\n" % log_path)

    def record(kind, payload):
        with open(log_path, "a") as f:
            f.write(json.dumps({"t": time.strftime("%H:%M:%S"), kind: payload}, default=str) + "\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line.startswith("/approve"):
            parts = line.split()
            if len(parts) != 2:
                print("usage: /approve <plan_id>")
                continue
            try:
                rec = core.approve(parts[1])
                print("approved %s" % parts[1])
                record("approved", rec)
                line = "I approved plan %s." % parts[1]
            except (PermissionError, ValueError) as e:
                print("not approved: %s" % e)
                continue
            except (KeyboardInterrupt, EOFError):
                print("\nnot approved")
                continue
        add_user(messages, line)
        record("user", line)
        run_turn(client, a.model, a.effort, system, tools, messages, sess, record, anthropic.APIError)
    return 0


if __name__ == "__main__":
    sys.exit(main())
