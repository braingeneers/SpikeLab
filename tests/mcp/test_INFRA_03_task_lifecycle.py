"""INFRA-3 — start an ask_spikelab_educator task, list it, resume it from a
fresh server process. Proves session persistence + rehydration across MCP
server restarts.

Uses educator (lock-free, lightweight) — sorter would need real data and a
GPU. No ``_lock`` field is expected in the response: all spikelab kinds are
lock-free.

Requires valid Claude credentials at ``~/.claude/.credentials.json`` and the
daemon running.
"""

from __future__ import annotations

import asyncio
import json
import sys

from _common import call, server_command

PROMPT_1 = (
    "This is an MCP harness test. Reply with the single word PONG and "
    "nothing else. Do not invoke any tools."
)
PROMPT_2 = (
    "What single word did I just ask you to reply with? Reply with that "
    "word only. Do not invoke any tools."
)


async def main() -> int:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    cmd = server_command(dev=False)

    print("=== INFRA-3a: start a new educator task ===")
    async with stdio_client(cmd) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await call(session, "ask_spikelab_educator", prompt=PROMPT_1)
            print(json.dumps(res, indent=2))
            assert isinstance(res, dict)
            assert res.get("task_id"), f"no task_id: {res}"
            assert res.get("kind") == "educator", f"wrong kind: {res}"
            assert "_lock" not in res, "spikelab responses must not carry _lock"
            task_id = res["task_id"]

    print(f"\n=== INFRA-3b: list_spikelab_tasks includes {task_id} ===")
    async with stdio_client(cmd) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tasks = await call(session, "list_spikelab_tasks")
            kinds = {t["task_id"]: t.get("kind") for t in tasks}
            assert task_id in kinds, f"task missing from list: {kinds}"
            assert kinds[task_id] == "educator", f"wrong kind: {kinds[task_id]}"

    print(f"\n=== INFRA-3c: get_spikelab_task_status({task_id}) ===")
    async with stdio_client(cmd) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            status = await call(session, "get_spikelab_task_status", task_id=task_id)
            assert status, "status was empty"
            assert status["task_id"] == task_id
            assert status["kind"] == "educator"
            assert status["turn_count"] >= 1

    print(f"\n=== INFRA-3d: resume from a fresh server process ===")
    async with stdio_client(cmd) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await call(
                session, "ask_spikelab_educator", prompt=PROMPT_2, task_id=task_id
            )
            print(json.dumps(res, indent=2))
            assert (
                res.get("task_id") == task_id
            ), f"task_id changed on resume: {res.get('task_id')} vs {task_id}"

    print("\nINFRA-3 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
