import os

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def state_dir():
    """Plans, approvals, run records: $OPENEDGE_AGENT_HOME or ~/.openedge-agent."""
    d = os.environ.get("OPENEDGE_AGENT_HOME") or os.path.join(os.path.expanduser("~"), ".openedge-agent")
    os.makedirs(d, exist_ok=True)
    return d


def openedge_root(explicit=None):
    """--root, $OPENEDGE_ROOT, or the tree this tool sits in (tools/agent)."""
    for cand in (explicit, os.environ.get("OPENEDGE_ROOT"),
                 os.path.dirname(os.path.dirname(AGENT_DIR))):
        if cand and os.path.isdir(os.path.join(cand, "src")):
            return os.path.abspath(cand)
    return None
