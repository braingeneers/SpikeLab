"""INFRA-2 — dev-only tools appear when and only when OC_SPIKELAB_MCP_DEV=1.

User mode: 8 tools. Dev mode: 10 tools (adds ask_spikelab_map_updater and
run_test_suite).
"""

from __future__ import annotations

import asyncio
import sys

from _common import DEV_ONLY_TOOLS, server_command


async def _names(dev: bool) -> set[str]:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(server_command(dev=dev)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return {t.name for t in tools.tools}


async def main() -> int:
    user_names = await _names(dev=False)
    dev_names = await _names(dev=True)

    assert DEV_ONLY_TOOLS.isdisjoint(
        user_names
    ), f"dev tools leaked into user mode: {DEV_ONLY_TOOLS & user_names}"
    assert (
        DEV_ONLY_TOOLS <= dev_names
    ), f"dev mode missing tools: {DEV_ONLY_TOOLS - dev_names}"
    assert len(user_names) == 8, f"expected 8 user-mode tools, got {len(user_names)}"
    assert len(dev_names) == 10, f"expected 10 dev-mode tools, got {len(dev_names)}"
    print(f"user mode tools ({len(user_names)}): {sorted(user_names)}")
    print(f"dev mode tools  ({len(dev_names)}): {sorted(dev_names)}")
    print("INFRA-2 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
