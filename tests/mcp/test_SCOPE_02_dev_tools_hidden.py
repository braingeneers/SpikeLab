"""SCOPE-2 — dev-only tools are absent from list_tools() unless OC_SPIKELAB_MCP_DEV=1.

Subset of INFRA-2 framed as a scope assertion: a customer-mode client cannot
see ask_spikelab_map_updater or run_test_suite — they aren't just hidden in
docs, they're structurally absent from the MCP tool list.
"""

from __future__ import annotations

import asyncio
import sys

from _common import DEV_ONLY_TOOLS, server_command


async def main() -> int:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(server_command(dev=False)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}

    leaked = DEV_ONLY_TOOLS & names
    assert not leaked, f"dev-only tools leaked into user mode: {leaked}"
    print(f"user-mode tools (dev-only hidden): {sorted(names)}")
    print("SCOPE-2 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
