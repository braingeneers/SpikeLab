"""INFRA-5 — educator and analyzer endpoints run in parallel (lock-free).

Spikelab has no operator concept and no flock between kinds. An educator
turn fired while an analyzer turn is in flight must not be serialised; both
should complete in roughly the time of the slower one, not in series.

Strategy: fire an analyzer turn with a prompt that the agent will spend
non-trivial time on (multi-line reasoning), and ~1 s later fire a short
educator turn from a different MCP server process. The educator should
finish well before the analyzer does.

Uses educator + analyzer (both lightweight; no real Bash tool calls).
Requires Claude credentials.
"""

from __future__ import annotations

import asyncio
import sys
import time

from _common import call, server_command

ANALYZER_PROMPT = (
    "MCP harness test of cross-kind concurrency. Reply with the single "
    "word DONE, but first internally count out loud from 1 to 20 in your "
    "response (one number per line) so this turn takes longer than the "
    "educator turn fired alongside it. Do not invoke any tools."
)
EDUCATOR_PROMPT = (
    "In one short sentence, what does STTC stand for? Do not invoke tools."
)


async def _analyzer() -> tuple[float, dict]:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    t0 = time.time()
    async with stdio_client(server_command(dev=False)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await call(session, "ask_spikelab_analyzer", prompt=ANALYZER_PROMPT)
    return time.time() - t0, res


async def _educator(start_delay: float) -> tuple[float, dict]:
    await asyncio.sleep(start_delay)
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    t0 = time.time()
    async with stdio_client(server_command(dev=False)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await call(session, "ask_spikelab_educator", prompt=EDUCATOR_PROMPT)
    return time.time() - t0, res


async def main() -> int:
    print("=== INFRA-5: educator runs while analyzer turn is in flight ===")
    (an_elapsed, an_res), (ed_elapsed, ed_res) = await asyncio.gather(
        _analyzer(),
        _educator(start_delay=1.0),
    )
    print(f"analyzer: {an_elapsed:.2f} s, kind={an_res.get('kind')}")
    print(f"educator: {ed_elapsed:.2f} s, kind={ed_res.get('kind')}")

    # Neither response should carry _lock — every spikelab kind is lock-free.
    assert "_lock" not in an_res, f"analyzer response has _lock: {an_res}"
    assert "_lock" not in ed_res, f"educator response has _lock: {ed_res}"

    assert an_res.get("kind") == "analyzer", f"wrong kind: {an_res}"
    assert ed_res.get("kind") == "educator", f"wrong kind: {ed_res}"
    assert not an_res.get("is_error"), an_res
    assert not ed_res.get("is_error"), ed_res

    # Educator should have finished well before the analyzer — if they were
    # serialised, the educator would have started only after the analyzer
    # finished, and its elapsed (from its own t0) would be small but its
    # wall-clock end would be after the analyzer's end. With our gather,
    # the gather returns once both complete; ed_elapsed measures only the
    # educator's own t0->t1 — so we want ed_elapsed << an_elapsed.
    assert ed_elapsed < an_elapsed - 0.5, (
        f"educator ({ed_elapsed:.2f}s) didn't measurably outpace analyzer "
        f"({an_elapsed:.2f}s) — concurrency appears broken"
    )

    print("\nINFRA-5 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
