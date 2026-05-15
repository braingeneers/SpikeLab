"""INFRA-1 — tool discovery surface.

Verifies the user-mode tool list matches PUBLIC_TOOLS exactly (8 tools), and
that the dev-only tools are absent from list_tools() when OC_SPIKELAB_MCP_DEV
is unset.

Requires the [mcp] extra installed (claude-agent-sdk, mcp).
"""

from __future__ import annotations

import asyncio
import sys

from _common import PUBLIC_TOOLS, DEV_ONLY_TOOLS, server_command


async def main() -> int:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(server_command(dev=False)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {t.name for t in tools.tools}

    missing = PUBLIC_TOOLS - names
    extra = names - PUBLIC_TOOLS
    leaked_dev = DEV_ONLY_TOOLS & names

    print(f"tools: {sorted(names)}")
    assert not missing, f"missing public tools: {missing}"
    assert not leaked_dev, f"dev-only tools visible in user mode: {leaked_dev}"
    assert not extra, f"unexpected extra tools registered: {extra}"
    assert len(names) == 8, f"expected 8 user-mode tools, got {len(names)}"
    print("INFRA-1 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
