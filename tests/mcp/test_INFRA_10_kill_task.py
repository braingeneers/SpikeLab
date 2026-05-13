"""INFRA-10 — kill_spikelab_task explicitly terminates a live child;
conversation history on disk survives so the next ask_* resumes it.

Strategy:
1. ask_spikelab_educator → record task_id
2. kill_spikelab_task(task_id) → expect {killed: true, was_alive: true}
3. ask_spikelab_educator(prompt, task_id=task_id) → expect success; the
   daemon respawns the child from disk and the educator should remember
   the previous turn's content.
4. kill_spikelab_task(task_id) a second time → the step-3 respawn put a
   live child back in the registry, so this should also return killed=true.

Requires: daemon running, valid Claude credentials.
"""

from __future__ import annotations

import asyncio
import sys

from _common import call, server_command

PROMPT_1 = "Remember the secret word AURORA. Reply OK."
PROMPT_2 = "What was the secret word? Reply with that word only, uppercase."


async def main() -> int:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    cmd = server_command(dev=False)

    async with stdio_client(cmd) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await call(session, "ask_spikelab_educator", prompt=PROMPT_1)
            print(f"step 1: {res!r}")
            assert not res.get("is_error"), res
            task_id = res["task_id"]
            assert task_id

            kill1 = await call(session, "kill_spikelab_task", task_id=task_id)
            print(f"step 2 (kill): {kill1!r}")
            assert kill1.get("killed") is True
            assert kill1.get("was_alive") is True

            # Brief pause so the child fully exits before respawn.
            await asyncio.sleep(0.5)

            res2 = await call(
                session, "ask_spikelab_educator", prompt=PROMPT_2, task_id=task_id
            )
            print(f"step 3 (resume after kill): {res2!r}")
            assert not res2.get("is_error"), res2
            response_text = (res2.get("response") or "").upper()
            assert "AURORA" in response_text, (
                f"resume failed — educator did not recall the secret word: "
                f"{response_text!r}"
            )

            kill2 = await call(session, "kill_spikelab_task", task_id=task_id)
            print(f"step 4 (kill again): {kill2!r}")
            assert kill2.get("killed") is True
            assert kill2.get("was_alive") is True

    print("INFRA-10 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
