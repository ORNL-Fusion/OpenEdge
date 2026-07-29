#!/usr/bin/env python3
"""Update OpenEdge's vendored SPARTA base from an upstream clone.

OpenEdge vendors SPARTA in src/ (no git lineage) and layers the OPENEDGE
package plus a set of locally modified base files on top. This tool does a
synthetic three-way update:

  base   = sparta @ OLD tag   (what src/ was vendored from; see SPARTA_BASE)
  ours   = OpenEdge src/      (base + local modifications)
  theirs = sparta @ NEW tag   (the upstream target release)

Per file:
  - upstream-changed, locally untouched  -> copy theirs        (safe)
  - new upstream file                    -> copy theirs        (new)
  - upstream-deleted, locally untouched  -> delete             (del)
  - changed on BOTH sides                -> git merge-file     (merge;
      conflict markers are left in place for files that truly collide —
      the tool lists them and you resolve by hand)
  - locally modified, upstream untouched -> keep ours          (local)

Usage:
  python3 tools/update_sparta_base.py --sparta /path/to/sparta-clone \\
      --old 20Jan2025 --new 24Sep2025 [--apply] [--phase safe|merge|all]

Without --apply it only reports. After a successful update, edit
SPARTA_BASE at the repo root to the new tag and commit.
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git_show(repo: Path, ref: str, path: str) -> bytes | None:
    r = subprocess.run(["git", "-C", str(repo), "show", f"{ref}:{path}"],
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparta", required=True, type=Path)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--phase", choices=["safe", "merge", "all"], default="all")
    args = ap.parse_args()
    up = args.sparta

    # upstream changes old -> new under src/
    r = subprocess.run(["git", "-C", str(up), "diff", "--name-status",
                        args.old, args.new, "--", "src/"],
                       capture_output=True, text=True, check=True)
    u_mod, u_new, u_del = set(), set(), set()
    for line in r.stdout.splitlines():
        st, *rest = line.split("\t")
        f = rest[-1]
        if st.startswith("A"): u_new.add(f)
        elif st.startswith("D"): u_del.add(f)
        else: u_mod.add(f)

    # local modifications vs old base
    base_files = subprocess.run(
        ["git", "-C", str(up), "ls-tree", "-r", "--name-only", args.old, "src/"],
        capture_output=True, text=True, check=True).stdout.split()
    local_mod, local_missing = set(), set()
    for f in base_files:
        oe = ROOT / f
        if not oe.exists():
            local_missing.add(f); continue
        if git_show(up, args.old, f) != oe.read_bytes():
            local_mod.add(f)

    safe = sorted(u_mod - local_mod)
    newf = sorted(u_new)
    dele = sorted(f for f in u_del if f not in local_mod and (ROOT / f).exists())
    conflict = sorted((u_mod | u_del) & local_mod)

    print(f"safe: {len(safe)}  new: {len(newf)}  delete: {len(dele)}  "
          f"three-way merge: {len(conflict)}  "
          f"(locally deleted base files preserved: {len(local_missing)})")

    if not args.apply:
        print("\n(report only; rerun with --apply)")
        for f in conflict: print("  MERGE", f)
        return 0

    if args.phase in ("safe", "all"):
        for f in safe + newf:
            data = git_show(up, args.new, f)
            dst = ROOT / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
        for f in dele:
            (ROOT / f).unlink()
        print(f"applied: {len(safe)} updated, {len(newf)} new, {len(dele)} deleted")

    unresolved = []
    if args.phase in ("merge", "all"):
        for f in conflict:
            if f in u_del:
                unresolved.append((f, "upstream deleted a locally-modified file"))
                continue
            base = git_show(up, args.old, f)
            theirs = git_show(up, args.new, f)
            ours = (ROOT / f).read_bytes()
            with tempfile.TemporaryDirectory() as td:
                pb, po, pt = Path(td, "base"), Path(td, "ours"), Path(td, "theirs")
                pb.write_bytes(base); po.write_bytes(ours); pt.write_bytes(theirs)
                r = subprocess.run(["git", "merge-file", "-L", "openedge",
                                    "-L", f"sparta-{args.old}",
                                    "-L", f"sparta-{args.new}",
                                    str(po), str(pb), str(pt)],
                                   capture_output=True)
                (ROOT / f).write_bytes(po.read_bytes())
                if r.returncode != 0:
                    unresolved.append((f, f"{r.returncode} conflict hunk(s)"))
        print(f"three-way merged: {len(conflict) - len(unresolved)} clean, "
              f"{len(unresolved)} with conflicts:")
        for f, why in unresolved:
            print("  CONFLICT", f, "-", why)

    return 0


if __name__ == "__main__":
    sys.exit(main())
