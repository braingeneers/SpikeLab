"""INFRA-7 — multi-turn latency profile proves the daemon amortises the
SDK session spawn cost.

Start a new educator task with a trivially short prompt, then send a second
prompt on the same task_id. The first turn pays the ~10s SDK spawn cost;
the second turn skips it.

We assert a *ratio*, not absolute times — reasoning latency varies by load.
The second turn should be substantially faster than the first; if it isn't,
the daemon isn't actually persisting the child.

Requires: daemon running, valid Claude credentials.
"""

from __future__ import annotations

import asyncio
import sys
import time

from _common import call, server_command

PROMPT_1 = (
    "Multi-turn latency test, turn 1. Reply with the single word PING and "
    "nothing else. Do not invoke any tools."
)
PROMPT_2 = (
    "Multi-turn latency test, turn 2. Reply with the single word PONG and "
    "nothing else. Do not invoke any tools."
)


async def _turn(prompt: str, task_id: str | None) -> tuple[float, dict]:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    t0 = time.time()
    async with stdio_client(server_command(dev=False)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await call(
                session, "ask_spikelab_educator", prompt=prompt, task_id=task_id
            )
    return time.time() - t0, res


async def main() -> int:
    print("=== INFRA-7: multi-turn latency ===")
    t1, res1 = await _turn(PROMPT_1, task_id=None)
    print(f"turn 1: {t1:.2f} s, response={res1.get('response')!r}")
    assert not res1.get("is_error"), f"turn 1 errored: {res1}"
    task_id = res1["task_id"]
    assert task_id

    t2, res2 = await _turn(PROMPT_2, task_id=task_id)
    print(f"turn 2: {t2:.2f} s, response={res2.get('response')!r}")
    assert not res2.get("is_error"), f"turn 2 errored: {res2}"
    assert (
        res2["task_id"] == task_id
    ), f"task_id changed: {res2['task_id']} vs {task_id}"

    ratio = t2 / max(t1, 0.001)
    print(f"ratio t2/t1: {ratio:.2f}")
    # If the daemon were spawning a fresh child on every turn, t2 ≈ t1
    # minus reasoning overhead. We expect t2 < 0.7 × t1 because the SDK
    # session is already warm.
    assert ratio < 0.7, (
        f"second turn was not faster ({ratio:.2f}× first); daemon may "
        f"not be persisting the child"
    )
    print("INFRA-7 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
