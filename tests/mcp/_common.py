"""Shared test helpers for the OC-SpikeLab MCP test suite.

Modeled on oc-ephys-tools/tests/mcp/_common.py — spawns the MCP server as a
subprocess speaking stdio, exercises the tool surface, and asserts on the
structured response.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def server_command(*, dev: bool = False):
    """Build StdioServerParameters for spawning a fresh MCP server.

    Imports lazily so test files that only exercise direct backends can still
    be imported when ``mcp`` isn't installed.
    """
    from mcp import StdioServerParameters  # type: ignore[import-not-found]

    env = os.environ.copy()
    if dev:
        env["OC_SPIKELAB_MCP_DEV"] = "1"
    else:
        env.pop("OC_SPIKELAB_MCP_DEV", None)
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "spikelab.mcp"],
        env=env,
    )


def unwrap(result) -> object:
    """FastMCP returns CallToolResult; pull out the structured content."""
    if getattr(result, "structuredContent", None) is not None:
        sc = result.structuredContent
        if isinstance(sc, dict) and set(sc.keys()) == {"result"}:
            return sc["result"]
        return sc
    if result.content:
        block = result.content[0]
        text = getattr(block, "text", None)
        if text is not None:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return None


async def call(session, name: str, **args):
    result = await session.call_tool(name, args)
    return unwrap(result)


# Tool surface contracts. Tests assert these are exactly registered.
PUBLIC_TOOLS = {
    "ask_spikelab_educator",
    "ask_spikelab_sorter",
    "ask_spikelab_analyzer",
    "ask_spikelab_developer",
    "list_spikelab_tasks",
    "get_spikelab_task_status",
    "get_spikelab_task_progress",
    "kill_spikelab_task",
}

DEV_ONLY_TOOLS = {
    "ask_spikelab_map_updater",
    "run_test_suite",
}
