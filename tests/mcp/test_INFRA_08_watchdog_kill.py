"""INFRA-8 — daemon reaps a child after its kind's idle-watchdog window.

This test is **disruptive**: it requires modifying ``system_params.json``
to shorten a watchdog window, restarting the daemon to pick up the change,
running the test, then restoring the original window. It is NOT run by
default. Invoke manually when you want to validate the watchdog path:

    python tests/mcp/test_INFRA_08_watchdog_kill.py --confirm

Strategy:
1. Read current ``system_params.json``; back it up.
2. Set ``ask_spikelab_map_updater_idle_watchdog_seconds`` to 30.
3. ``systemctl --user restart spikelab-daemon.service``.
4. Spawn an ``ask_spikelab_map_updater`` task in dev mode (it must be dev
   mode for the endpoint to be visible) and verify the child appears in
   ``list_spikelab_tasks``.
5. Wait ~60 s.
6. Verify the child was reaped: ``daemon ping`` shows ``children: 0`` and
   a fresh ``ask_*`` on the same task_id spawns a new child (transcript
   on disk preserved).
7. Restore ``system_params.json``; restart the daemon again.

Why not run by default: a daemon restart kills any in-flight conversations
on this host. We don't want a routine pytest run to disrupt a developer's
active orchestrator session.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

CONFIG_PATH = Path("/home/mxwbio/work/Habitat-MaxOne-test/system_params.json")
KEY = "ask_spikelab_map_updater_idle_watchdog_seconds"
SHORT_WINDOW_SECONDS = 30


def _read_params() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def _write_params(d: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(d, indent=2) + "\n")


def _restart_daemon() -> None:
    subprocess.run(
        ["systemctl", "--user", "restart", "spikelab-daemon.service"],
        check=True,
    )
    # Wait for the socket to come back.
    sock = Path("/run/user") / str(__import__("os").getuid()) / "spikelab-daemon.sock"
    for _ in range(20):
        if sock.exists():
            return
        time.sleep(0.5)
    raise RuntimeError(f"daemon socket {sock} did not reappear within 10s")


async def _ping_daemon() -> dict:
    """Talk to the daemon directly over its Unix socket."""
    from spikelab.mcp.config import daemon_socket_path  # type: ignore

    r, w = await asyncio.open_unix_connection(str(daemon_socket_path()))
    w.write(b'{"op":"ping"}\n')
    await w.drain()
    line = await r.readline()
    w.close()
    await w.wait_closed()
    return json.loads(line)


async def main(confirm: bool) -> int:
    if not confirm:
        print(
            "INFRA-8 needs --confirm to run (this test restarts the daemon, "
            "which kills any in-flight conversations). Skipped."
        )
        return 0

    from _common import call, server_command
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    # 1+2: back up, write short window
    original = _read_params()
    backup = dict(original)
    modified = dict(original)
    modified[KEY] = SHORT_WINDOW_SECONDS
    _write_params(modified)
    print(f"set {KEY}={SHORT_WINDOW_SECONDS} in {CONFIG_PATH}")

    try:
        # 3: restart daemon
        _restart_daemon()
        print("daemon restarted")

        # 4: spawn a dev-mode map_updater task
        cmd = server_command(dev=True)
        async with stdio_client(cmd) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await call(
                    session,
                    "ask_spikelab_map_updater",
                    prompt="Reply with the single word READY. Do not invoke any tools.",
                )
                print(f"map_updater turn: {res!r}")
                assert not res.get("is_error"), res
                task_id = res["task_id"]

        # Confirm the child is alive in the registry.
        ping = await _ping_daemon()
        print(f"after spawn: ping={ping!r}")
        assert ping.get("children", 0) >= 1, ping

        # 5: wait past the watchdog window plus one daemon-tick (30 s ticker).
        wait_s = SHORT_WINDOW_SECONDS + 35
        print(f"waiting {wait_s} s for the watchdog to fire...")
        await asyncio.sleep(wait_s)

        # 6: child should be gone
        ping_after = await _ping_daemon()
        print(f"after wait: ping={ping_after!r}")
        assert ping_after.get("children", 0) == 0, (
            f"child still alive {wait_s}s after the {SHORT_WINDOW_SECONDS}s "
            f"watchdog should have fired: ping={ping_after}"
        )

        # 7: respawn via resume — transcript on disk should be intact
        async with stdio_client(cmd) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res2 = await call(
                    session,
                    "ask_spikelab_map_updater",
                    prompt="Reply OK.",
                    task_id=task_id,
                )
                assert not res2.get("is_error"), res2
                assert res2.get("task_id") == task_id

        print("INFRA-8 PASS")
        return 0
    finally:
        # Restore original config + restart daemon.
        _write_params(backup)
        try:
            _restart_daemon()
        except Exception as e:
            print(f"WARN: daemon restart on cleanup failed: {e}")
        print(f"restored {KEY} in {CONFIG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required — confirms you've read the disruption note above.",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.confirm)))
