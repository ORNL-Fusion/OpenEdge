"""Which database the binary will read, and what it holds.

Mirrors src/database_paths.cpp: $OPENEDGE_ROOT/database, else the directory
compiled into the binary, else ./database in the run directory."""
import os
import re
import shutil
import subprocess


def compiled_database_dir(binary):
    if not binary or not os.path.isfile(binary):
        return None
    try:
        out = subprocess.run(["strings", binary], capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    cands = [ln for ln in out.splitlines()
             if ln.startswith("/") and ln.endswith("/database") and " " not in ln and len(ln) > 9]
    return max(cands, key=len) if cands else None


def database_dir(binary=None, run_dir=".", env=None):
    env = os.environ if env is None else env
    if env.get("OPENEDGE_ROOT"):
        return os.path.join(env["OPENEDGE_ROOT"].rstrip("/"), "database"), "OPENEDGE_ROOT"
    comp = compiled_database_dir(binary)
    if comp:
        return comp, "compiled into the binary"
    if binary:
        return os.path.join(run_dir, "database"), "working directory (binary has no compiled default)"
    return None, "unknown (no binary given and OPENEDGE_ROOT not set)"


class H5:
    """Group listing of an HDF5 file via h5py, else the h5ls tool."""

    def __init__(self, path):
        self.path = path
        self.backend = None
        self._cache = {}
        try:
            import h5py  # noqa: F401
            self.backend = "h5py"
        except ImportError:
            if shutil.which("h5ls"):
                self.backend = "h5ls"

    def groups(self, group):
        """Member names of a group, [] if the group is absent, None if unknown."""
        if group in self._cache:
            return self._cache[group]
        res = None
        if self.backend == "h5py":
            import h5py
            try:
                with h5py.File(self.path, "r") as f:
                    res = list(f[group].keys()) if group in f else []
            except OSError:
                res = None
        elif self.backend == "h5ls":
            p = subprocess.run(["h5ls", "%s%s" % (self.path, group)], capture_output=True,
                               text=True, timeout=60)
            if p.returncode == 0:
                res = [ln.split()[0] for ln in p.stdout.splitlines() if ln.strip()]
            elif "not found" in (p.stdout + p.stderr).lower() or "unable" in (p.stdout + p.stderr).lower():
                res = []
        self._cache[group] = res
        return res


RECYCLE_LINE = re.compile(r"^\s*([TS])\s+(\S+)")


def recycle_tables(path):
    """[(kind, name, line)] for T (reflection) and S (sputter) entries of a recycle file."""
    out = []
    with open(path, errors="replace") as f:
        for k, line in enumerate(f, 1):
            line = line.split("#", 1)[0]
            m = RECYCLE_LINE.match(line)
            if m:
                out.append((m.group(1), m.group(2), k))
    return out
