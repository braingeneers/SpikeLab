"""INFRA-4 — per-task_id serialization vs cross-task parallelism.

Spikelab is lock-free across kinds and across distinct task_ids — there's
no rig flock, no library mutex, no operator concept. The *only* place
requests serialise is at the daemon's per-``ChildHandle.in_flight``
``asyncio.Lock``: a child has one stdin pipe, so two concurrent calls on
the same task_id must queue.

This test verifies:

a) Two concurrent calls on *different* task_ids complete in roughly the
   time of a single call — they ran in parallel.
b) Two concurrent calls on the *same* task_id complete in roughly
   double-single — they queued behind each other.

Uses educator (lightweight, no GPU). Requires Claude credentials.
"""

from __future__ import annotations

import asyncio
import sys
import time

from _common import call, server_command

PROMPT = "Reply with the single word ACK and nothing else. Do not invoke any tools."


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


async def _single_baseline() -> float:
    """One isolated turn so we have a reference for what 'one turn' costs."""
    t, _ = await _turn(PROMPT, task_id=None)
    return t


async def main() -> int:
    print("=== INFRA-4 baseline: one solo educator turn ===")
    t_solo = await _single_baseline()
    print(f"solo turn: {t_solo:.2f} s")

    # (a) two DIFFERENT task_ids in parallel -> roughly parallel
    print("\n=== INFRA-4a: two new task_ids concurrently (different children) ===")
    t0 = time.time()
    (ta, ra), (tb, rb) = await asyncio.gather(
        _turn(PROMPT, task_id=None),
        _turn(PROMPT, task_id=None),
    )
    wall_parallel = time.time() - t0
    print(f"task A: {ta:.2f} s (id={ra.get('task_id')})")
    print(f"task B: {tb:.2f} s (id={rb.get('task_id')})")
    print(f"wall-clock both: {wall_parallel:.2f} s")
    assert not ra.get("is_error"), ra
    assert not rb.get("is_error"), rb
    assert ra["task_id"] != rb["task_id"], "two new tasks got the same task_id?"
    # Parallel run should be measurably under sum-of-individual times.
    # Allow generous slack — first-turn SDK spawn dominates and varies.
    sum_individual = ta + tb
    assert wall_parallel < sum_individual * 0.85, (
        f"parallel wall {wall_parallel:.2f}s not noticeably under "
        f"sum {sum_individual:.2f}s — different task_ids didn't run in parallel?"
    )

    # (b) reuse one of the task_ids -> two calls on same child serialise
    task_id = ra["task_id"]
    print(f"\n=== INFRA-4b: two calls on same task_id={task_id} concurrently ===")
    t0 = time.time()
    (tc, rc), (td, rd) = await asyncio.gather(
        _turn(PROMPT, task_id=task_id),
        _turn(PROMPT, task_id=task_id),
    )
    wall_serial = time.time() - t0
    print(f"turn C: {tc:.2f} s")
    print(f"turn D: {td:.2f} s")
    print(f"wall-clock both: {wall_serial:.2f} s")
    assert not rc.get("is_error"), rc
    assert not rd.get("is_error"), rd
    assert rc["task_id"] == task_id and rd["task_id"] == task_id
    # Each _turn measures (its own queue wait + its own execution). If the
    # two ran in parallel, tc and td would be ~equal (neither queued). If
    # they serialised, one waited for the other — the difference equals the
    # first turn's execution time. Asymmetry > 1.0 s is solid evidence of
    # queueing.
    asymmetry = abs(tc - td)
    print(f"per-turn asymmetry (queue-wait evidence): {asymmetry:.2f} s")
    assert asymmetry > 1.0, (
        f"same-task tc={tc:.2f}s, td={td:.2f}s nearly equal — the in_flight "
        f"lock didn't queue them (asymmetry only {asymmetry:.2f}s)"
    )

    print("\nINFRA-4 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
