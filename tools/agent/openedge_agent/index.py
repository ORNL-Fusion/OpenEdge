"""Command index generated from the source tree: style macros in src/*.h and
syntax lines in doc/*.txt.  Stamped with the git commit; `stale` says whether
the tree moved since the index was built."""
import glob
import json
import os
import re
import subprocess

from . import paths

_STYLE = re.compile(r"^\s*(\w+)Style\(\s*([^,\s)]+)\s*,\s*(\w+)\s*\)", re.M)
_SYNTAX = re.compile(r"^(\w+) ID (\S+) (.*?):pre\s*$", re.M)
STEP_NAMES = {"Nevery", "Nfreq", "N", "Nrepeat"}


def git_head(root):
    try:
        out = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=20)
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--", "src"],
                               capture_output=True, text=True, timeout=60).stdout.strip() != ""
        return out.stdout.strip() or None, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def build(root):
    styles = {}
    for h in sorted(glob.glob(os.path.join(root, "src", "*.h"))):
        with open(h, errors="replace") as f:
            for kind, name, cls in _STYLE.findall(f.read()):
                if "/kk" in name:
                    continue
                styles.setdefault(kind.lower(), {})[name] = {"class": cls,
                                                            "header": os.path.relpath(h, root)}
    syntax = {}
    for t in sorted(glob.glob(os.path.join(root, "doc", "*.txt"))):
        with open(t, errors="replace") as f:
            for cmd, style, rest in _SYNTAX.findall(f.read()):
                args = rest.split()
                syntax.setdefault(cmd, {})[style] = {
                    "doc": os.path.relpath(t, root), "args": args,
                    "step_args": [[a, i] for i, a in enumerate(args) if a in STEP_NAMES]}
    for kind, entries in styles.items():
        for name, e in entries.items():
            s = syntax.get(kind, {}).get(name)
            if s:
                e.update(s)
            else:
                e["doc"] = None
    head, dirty = git_head(root)
    return {"root": root, "git_head": head, "src_dirty": dirty, "styles": styles}


def load(root, rebuild=False):
    cache = os.path.join(paths.state_dir(), "index", re.sub(r"\W", "_", root) + ".json")
    head, _ = git_head(root)
    if not rebuild and os.path.isfile(cache):
        with open(cache) as f:
            idx = json.load(f)
        if idx.get("git_head") == head:
            idx["stale"] = False
            return idx
    idx = build(root)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as f:
        json.dump(idx, f)
    idx["stale"] = False
    return idx


def lookup(idx, kind, style):
    return idx["styles"].get(kind, {}).get(style)
