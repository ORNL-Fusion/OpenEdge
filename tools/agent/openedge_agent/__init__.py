"""openedge-agent: deterministic tooling to check, plan and launch OpenEdge runs.

No LLM inside. Agents (Claude Code, Codex, ...) call it through the CLI or the
MCP server; people can use the CLI directly.
"""
import sys

__version__ = "0.1.0"

if sys.version_info < (3, 8):
    raise SystemExit("openedge-agent needs Python 3.8 or newer (found %d.%d)" % sys.version_info[:2])
