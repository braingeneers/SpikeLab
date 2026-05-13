"""INFRA-9 — child process crash → respawn from on-disk SDK state.

Strategy:
1. Start an educator task with a memorable secret word in the prompt.
2. Locate the child PID by walking the daemon's process tree.
3. SIGKILL the child (simulating an unexpected death — different from the
   graceful exit that kill_spikelab_task triggers).
4. Send a second prompt on the same task_id; the daemon must notice the
   dead child, spawn a fresh one, and the new child must resume from disk
   and remember the secret word.

Requires: daemon running, valid Claude credentials, /proc available
(Linux). Skipped on macOS / Windows.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from _common import call, server_command

PROMPT_SETUP = "Remember the secret word ZEPHYR. Reply with the single word OK."
PROMPT_PROBE = "What was the secret word? Reply with that word only, uppercase."


def _find_daemon_pid() -> int:
    """Find the running spikelab daemon PID by inspecting systemctl --user."""
    out = subprocess.run(
        ["systemctl", "--user", "show", "-p", "MainPID", "spikelab-daemon.service"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    # Output looks like "MainPID=379917"
    _, _, pid = out.partition("=")
    return int(pid)


def _find_child_pids(daemon_pid: int) -> list[int]:
    """Return PIDs of immediate children of the daemon — these are our
    per-task_id child agents spawned via ``asyncio.create_subprocess_exec``."""
    out = subprocess.run(
        ["pgrep", "-P", str(daemon_pid)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return [int(line) for line in out.splitlines() if line.strip()]


async def main() -> int:
    if not Path("/proc").is_dir():
        print("INFRA-9 SKIP: /proc not present (need Linux)")
        return 0

    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    cmd = server_command(dev=False)

    print("=== INFRA-9a: start an educator task with a secret word ===")
    async with stdio_client(cmd) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await call(session, "ask_spikelab_educator", prompt=PROMPT_SETUP)
            print(f"setup response: {res!r}")
            assert not res.get("is_error"), res
            task_id = res["task_id"]
            assert task_id

    daemon_pid = _find_daemon_pid()
    print(f"daemon pid = {daemon_pid}")

    # Find the child for this task_id. The daemon may have multiple children
    # from earlier tests; the most recently spawned one is our task.
    children = _find_child_pids(daemon_pid)
    print(f"daemon children: {children}")
    assert children, "daemon has no child processes — task didn't spawn one?"

    # SIGKILL the newest child (highest PID is a reasonable heuristic for
    # "most recently spawned" on Linux PID wrap aside).
    victim = max(children)
    print(f"SIGKILL child pid {victim}")
    try:
        os.kill(victim, signal.SIGKILL)
    except ProcessLookupError:
        print(f"child {victim} already gone")
    # Give the daemon a moment to notice the SIGCHLD and update its registry.
    await asyncio.sleep(2.0)

    print(
        "\n=== INFRA-9b: probe with a fresh server process; expect respawn + resume ==="
    )
    async with stdio_client(cmd) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res2 = await call(
                session, "ask_spikelab_educator", prompt=PROMPT_PROBE, task_id=task_id
            )
            print(f"probe response: {res2!r}")
            assert not res2.get("is_error"), res2
            assert (
                res2.get("task_id") == task_id
            ), f"task_id changed on respawn: {res2.get('task_id')} vs {task_id}"
            response_text = (res2.get("response") or "").upper()
            assert "ZEPHYR" in response_text, (
                f"respawned child didn't recall the secret word from disk: "
                f"{response_text!r}"
            )

    print("\nINFRA-9 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
