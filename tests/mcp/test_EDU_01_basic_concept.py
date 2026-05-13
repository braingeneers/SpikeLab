"""EDU-1 — educator answers a basic concept question coherently.

Ask the educator "what does STTC stand for, in one sentence?" The reply
should mention STTC / spike time tiling coefficient. The educator should
not invoke any tools (no Bash / Read for a simple definition).

This is the cheapest end-to-end smoke that the educator endpoint actually
talks to a live SDK session and returns something the user can read.

Requires: daemon running, valid Claude credentials.
"""

from __future__ import annotations

import asyncio
import sys

from _common import call, server_command

PROMPT = (
    "In one short sentence, what does the acronym STTC stand for in "
    "neuroscience? Do not invoke any tools."
)


async def main() -> int:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    async with stdio_client(server_command(dev=False)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await call(session, "ask_spikelab_educator", prompt=PROMPT)
            print(f"response: {res!r}")
            assert not res.get("is_error"), res
            assert res.get("kind") == "educator"
            response = (res.get("response") or "").lower()
            assert response.strip(), "empty response"
            # Be generous — answer might phrase it as "spike-time tiling
            # coefficient" with or without hyphens; just require the core terms.
            assert (
                "spike" in response and "tiling" in response
            ), f"educator answer doesn't mention spike/tiling: {response!r}"

    print("EDU-1 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
