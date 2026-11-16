#!/usr/bin/env python3
"""Offline test of the chat turn loop with a scripted fake client: every tool_use in the
history must keep a tool_result through interrupts, truncation and tool errors. No API calls."""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openedge_agent import chat  # noqa: E402


class B:
    def __init__(self, type, **k):
        self.type = type
        self.__dict__.update(k)

    def model_dump(self):
        return dict(self.__dict__)


class Msg:
    def __init__(self, content, stop):
        self.content, self.stop_reason = content, stop


class Stream:
    def __init__(self, item):
        self.item = item

    def __enter__(self):
        if isinstance(self.item, BaseException):
            raise self.item
        texts = [b.text for b in self.item.content if b.type == "text"]
        self.text_stream = iter(texts)
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.item


class Client:
    def __init__(self, script):
        self.script, self.seen = list(script), []
        self.messages = self

    def stream(self, **kw):
        self.seen.append([dict(m) for m in kw["messages"]])
        return Stream(self.script.pop(0))


class Sess:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    def call(self, name, a):
        b = self.behaviour.get(name, "ok")
        if b == "raise":
            raise ValueError("bad deck")
        if b == "interrupt":
            raise KeyboardInterrupt
        return {"ok": name}


def valid(messages):
    """Each assistant tool_use is answered in the next user turn; roles alternate sensibly."""
    for i, m in enumerate(messages):
        if m["role"] != "assistant" or not isinstance(m["content"], list):
            continue
        ids = {b.id for b in m["content"] if b.type == "tool_use"}
        if not ids:
            continue
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        got = {c["tool_use_id"] for c in (nxt["content"] if nxt and isinstance(nxt["content"], list) else [])
               if isinstance(c, dict) and c.get("type") == "tool_result"}
        if ids - got:
            return False
    return True


def tu(i, name="preflight"):
    return B("tool_use", id="t%d" % i, name=name, input={"deck": "in.x"})


def run(script, behaviour=None, prior=None, text="hi"):
    msgs = list(prior or [])
    chat.add_user(msgs, text)
    c = Client(script)
    r = chat.run_turn(c, "m", "low", [], [], msgs, Sess(behaviour or {}), out=io.StringIO())
    return r, msgs


cases = []
r, m = run([Msg([tu(1)], "tool_use"), Msg([B("text", text="done")], "end_turn")])
cases.append(("tool round then answer", r == "done" and valid(m) and len(m) == 4))
r, m = run([Msg([tu(1)], "tool_use"), KeyboardInterrupt()])
cases.append(("interrupt after tool round keeps results", r == "interrupted" and valid(m) and m[-1]["role"] == "user"))
r2, m2 = run([Msg([B("text", text="ok")], "end_turn")], prior=m, text="next")
cases.append(("next message joins the tool-result turn", r2 == "done" and valid(m2)
              and m2[2]["content"][-1] == {"type": "text", "text": "next"}))
r, m = run([Msg([tu(1), tu(2, "inspect_deck")], "tool_use")], behaviour={"preflight": "interrupt"})
cases.append(("interrupt inside a tool answers all tool_use", r == "interrupted" and valid(m)
              and all(c["is_error"] for c in m[-1]["content"])))
r, m = run([Msg([B("text", text="part"), tu(1)], "max_tokens")])
cases.append(("max_tokens drops the cut tool call", r == "max_tokens" and valid(m)
              and all(b.type != "tool_use" for b in m[-1]["content"])))
r, m = run([Msg([tu(1)], "tool_use"), Msg([B("text", text="sorry")], "end_turn")], behaviour={"preflight": "raise"})
cases.append(("tool error goes back as is_error", r == "done" and valid(m) and m[2]["content"][0]["is_error"]))
r, m = run([KeyboardInterrupt()])
cases.append(("interrupt before any reply drops the text", r == "interrupted" and m == []))
r, m = run([Msg([B("text", text="")], "refusal")])
cases.append(("refusal leaves a valid history", r == "refusal" and valid(m)))

bad = 0
for name, ok in cases:
    print("%-4s %s" % ("ok" if ok else "FAIL", name))
    bad += not ok
print("\n%d cases, %d failed" % (len(cases), bad))
sys.exit(1 if bad else 0)
