"""OC-SpikeLab MCP daemon — long-lived process owning the per-task_id registry.

Architecture (per host)::

    External caller (orchestrator, bench-companion)
         │   MCP protocol
         ▼
    [MCP server]   ← short-lived per connection (see ``server.py``)
         │   Unix socket (this daemon)
         ▼
    [daemon]       ← long-lived (systemd-managed)
         │   stdin/stdout pipes
         ▼
    [child agent (×N)]   ← one per active task_id (see ``child.py``)

Mirrors the ephys daemon (``OC_ephys_tools.mcp.daemon``) with two structural
simplifications:

- **No operator concept.** All five spikelab kinds are lock-free — there is
  no per-turn flock to coordinate and no ``_lock`` timing dict in responses.
- **No library-lock pre-check.** Spikelab has no hardware-side mutex to race
  against (analogous to the OC_ephys_tools session lock at
  ``/tmp/oc_ephys_session.lock``); the operator-only library-lock branch
  that exists in the ephys daemon is absent here.

The daemon's per-task_id :class:`asyncio.Lock` on each :class:`ChildHandle`
remains — a child serves one turn at a time on its single stdin pipe, so
concurrent requests on the same ``task_id`` must queue at that lock.

JSON-line protocol on the Unix socket
-------------------------------------

Requests (one JSON object per line)::

    {"op": "ask", "kind": "<educator|sorter|analyzer|developer|map_updater>",
                  "task_id": null | "<uuid>", "prompt": "..."}
    {"op": "list_tasks", "kind": null | "<kind>"}
    {"op": "get_task_status", "task_id": "<uuid>"}
    {"op": "kill_task", "task_id": "<uuid>"}
    {"op": "ping"}

Responses (one JSON object per line) match the MCP tool return shapes:

    ask           → child's reply verbatim (task_id, kind, response, ...)
    list_tasks    → list[dict] from sessions.list_tasks(kind=...)
    get_task_status → dict | null from sessions.get_task_status(task_id)
    kill_task     → {"task_id": "...", "killed": bool, "was_alive": bool}
    ping          → {"ok": true, "pid": <int>, "children": <int>}
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from .config import (
    DAEMON_WATCHDOG_TICK_SECONDS,
    daemon_socket_path,
    idle_watchdog_seconds,
)
from .sessions import get_task_status, list_tasks


async def _terminate_child(child: "ChildHandle", grace_seconds: float = 5.0) -> None:
    """SIGTERM the child, also close its stdin so the per-turn ``readline``
    inside the child wakes up with EOF (the SIGTERM handler alone doesn't
    interrupt a blocking ``await reader.readline()``). Escalate to SIGKILL
    if the child hasn't exited after ``grace_seconds``.
    """
    try:
        child.proc.terminate()
    except ProcessLookupError:
        return
    if child.proc.stdin is not None:
        try:
            child.proc.stdin.close()
        except Exception:
            pass
    try:
        await asyncio.wait_for(child.proc.wait(), timeout=grace_seconds)
        return
    except asyncio.TimeoutError:
        pass
    try:
        child.proc.kill()
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(child.proc.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        pass


# ---------------------------------------------------------------------------
# Per-child handle. One ChildHandle per active task_id in the registry.
# ---------------------------------------------------------------------------
@dataclass
class ChildHandle:
    task_id: str | None  # None until the child returns its first result
    kind: str
    proc: asyncio.subprocess.Process
    last_activity: float  # monotonic-ish wall-clock from time.time()
    in_flight: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Reading the child's response is serialised by ``in_flight``: a child
    # serves one turn at a time on its single stdin pipe, so concurrent
    # requests on the same task_id must queue at this lock before sending.

    @property
    def pid(self) -> int:
        return self.proc.pid

    @property
    def alive(self) -> bool:
        return self.proc.returncode is None


# ---------------------------------------------------------------------------
# Daemon state
# ---------------------------------------------------------------------------
class Daemon:
    """Holds the per-task_id child registry and serves the Unix socket."""

    def __init__(self) -> None:
        # ``by_task`` is keyed by the SDK session UUID (known after the first
        # turn responds). New tasks live in ``pending`` until the child reports
        # back its assigned task_id.
        self.by_task: dict[str, ChildHandle] = {}
        self.pending: list[ChildHandle] = []
        self.registry_lock = asyncio.Lock()
        self._reaper_task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()

    # ----- child lifecycle ---------------------------------------------------
    async def spawn_child(self, kind: str, resume_task_id: str | None) -> ChildHandle:
        """Spawn a new child process and register it.

        ``resume_task_id``: if provided, child resumes that SDK session.
        Otherwise it starts a new one and the assigned UUID lands on the
        first response.
        """
        task_arg = resume_task_id if resume_task_id else "none"
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "spikelab.mcp.child",
            "--kind",
            kind,
            "--task-id",
            task_arg,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        handle = ChildHandle(
            task_id=resume_task_id,
            kind=kind,
            proc=proc,
            last_activity=time.time(),
        )
        async with self.registry_lock:
            if resume_task_id:
                self.by_task[resume_task_id] = handle
            else:
                self.pending.append(handle)
        return handle

    async def _send_and_recv(self, child: ChildHandle, prompt: str) -> dict[str, Any]:
        """Send one prompt to a child, await its one response."""
        async with child.in_flight:
            assert child.proc.stdin is not None and child.proc.stdout is not None
            payload = json.dumps({"prompt": prompt}).encode() + b"\n"
            child.proc.stdin.write(payload)
            await child.proc.stdin.drain()
            line = await child.proc.stdout.readline()
            if not line:
                stderr_tail = b""
                try:
                    if child.proc.stderr is not None:
                        stderr_tail = await asyncio.wait_for(
                            child.proc.stderr.read(2000), timeout=0.5
                        )
                except (asyncio.TimeoutError, Exception):
                    pass
                rc = child.proc.returncode
                return {
                    "task_id": child.task_id,
                    "kind": child.kind,
                    "response": "",
                    "is_error": True,
                    "errors": [
                        f"daemon: child exited (returncode={rc}) "
                        f"before responding: {stderr_tail.decode(errors='replace')[-400:]}"
                    ],
                }
            try:
                resp = json.loads(line)
            except json.JSONDecodeError as e:
                return {
                    "task_id": child.task_id,
                    "kind": child.kind,
                    "response": "",
                    "is_error": True,
                    "errors": [f"daemon: bad JSON from child: {e!r}; line={line!r}"],
                }
            child.last_activity = time.time()
            return resp

    async def _promote_pending(self, child: ChildHandle, assigned_task_id: str) -> None:
        """A pending child has reported its assigned task_id — move it into
        ``by_task`` under that key."""
        async with self.registry_lock:
            if child in self.pending:
                self.pending.remove(child)
            child.task_id = assigned_task_id
            self.by_task[assigned_task_id] = child

    # ----- op handlers -------------------------------------------------------
    async def op_ask(
        self, kind: str, task_id: str | None, prompt: str
    ) -> dict[str, Any]:
        # Resume path: re-use the existing child, or spawn a fresh one to
        # rehydrate from disk.
        if task_id:
            existing = self.by_task.get(task_id)
            if existing and existing.alive:
                # Kind mismatch is a hard error: each endpoint is scoped to
                # one sub-skill; resuming an educator task via a sorter
                # endpoint (or vice versa) would silently mix system prompts.
                if existing.kind != kind:
                    return {
                        "task_id": task_id,
                        "kind": existing.kind,
                        "response": "",
                        "is_error": True,
                        "errors": [
                            f"daemon: kind mismatch — task_id {task_id} is "
                            f"kind={existing.kind!r}, but ask was for kind={kind!r}"
                        ],
                    }
                return await self._send_and_recv(existing, prompt)
            # No live child for this task_id, or it died — respawn from disk.
            # Before spawning, verify the on-disk transcript's kind matches.
            status = get_task_status(task_id)
            if status is not None:
                disk_kind = status.get("kind", "unknown")
                if disk_kind not in ("unknown", kind):
                    return {
                        "task_id": task_id,
                        "kind": disk_kind,
                        "response": "",
                        "is_error": True,
                        "errors": [
                            f"daemon: kind mismatch — task_id {task_id} on disk "
                            f"is kind={disk_kind!r}, but ask was for kind={kind!r}"
                        ],
                    }
            child = await self.spawn_child(kind=kind, resume_task_id=task_id)
            return await self._send_and_recv(child, prompt)

        # New task path: spawn, send, latch the assigned task_id when it
        # comes back.
        child = await self.spawn_child(kind=kind, resume_task_id=None)
        result = await self._send_and_recv(child, prompt)
        assigned = result.get("task_id")
        if isinstance(assigned, str) and assigned:
            await self._promote_pending(child, assigned)
        return result

    async def op_list_tasks(self, kind: str | None) -> list[dict[str, Any]]:
        return list_tasks(kind=kind)

    async def op_get_task_status(self, task_id: str) -> dict[str, Any] | None:
        return get_task_status(task_id)

    async def op_kill_task(self, task_id: str) -> dict[str, Any]:
        async with self.registry_lock:
            child = self.by_task.pop(task_id, None)
        if child is None:
            return {"task_id": task_id, "killed": False, "was_alive": False}
        was_alive = child.alive
        if was_alive:
            # Don't await here — kick off the SIGTERM+close-stdin sequence
            # so the daemon serves the response promptly. The child cleans
            # up on its own; SIGKILL escalation happens inside _terminate_child.
            asyncio.create_task(_terminate_child(child))
        return {"task_id": task_id, "killed": True, "was_alive": was_alive}

    async def op_ping(self) -> dict[str, Any]:
        return {
            "ok": True,
            "pid": os.getpid(),
            "children": len(self.by_task) + len(self.pending),
        }

    # ----- reaper / watchdog --------------------------------------------------
    async def _reaper_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(),
                    timeout=DAEMON_WATCHDOG_TICK_SECONDS,
                )
            except asyncio.TimeoutError:
                pass  # tick: scan registry
            now = time.time()
            to_kill: list[ChildHandle] = []
            async with self.registry_lock:
                for child in list(self.by_task.values()) + list(self.pending):
                    if not child.alive:
                        if child.task_id and self.by_task.get(child.task_id) is child:
                            self.by_task.pop(child.task_id, None)
                        if child in self.pending:
                            self.pending.remove(child)
                        continue
                    idle = now - child.last_activity
                    if idle >= idle_watchdog_seconds(child.kind):
                        to_kill.append(child)
                        if child.task_id and self.by_task.get(child.task_id) is child:
                            self.by_task.pop(child.task_id, None)
                        if child in self.pending:
                            self.pending.remove(child)
            for child in to_kill:
                asyncio.create_task(_terminate_child(child))

    # ----- shutdown ----------------------------------------------------------
    async def shutdown(self) -> None:
        self._shutdown.set()
        if self._reaper_task is not None:
            self._reaper_task.cancel()
        async with self.registry_lock:
            all_children = list(self.by_task.values()) + list(self.pending)
            self.by_task.clear()
            self.pending.clear()
        await asyncio.gather(
            *(_terminate_child(c) for c in all_children),
            return_exceptions=True,
        )


# ---------------------------------------------------------------------------
# Per-connection handler
# ---------------------------------------------------------------------------
async def _handle_client(
    daemon: Daemon,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """One MCP-server connection. Reads one request line, returns one reply
    line, closes. Keep it short — the MCP server doesn't multiplex."""
    try:
        line = await reader.readline()
        if not line:
            return
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            writer.write(
                (
                    json.dumps({"is_error": True, "errors": [f"bad JSON: {e!r}"]})
                    + "\n"
                ).encode()
            )
            await writer.drain()
            return

        op = req.get("op")
        try:
            if op == "ask":
                kind = req["kind"]
                task_id = req.get("task_id")
                prompt = req["prompt"]
                result = await daemon.op_ask(kind=kind, task_id=task_id, prompt=prompt)
            elif op == "list_tasks":
                result = await daemon.op_list_tasks(req.get("kind"))
            elif op == "get_task_status":
                result = await daemon.op_get_task_status(req["task_id"])
            elif op == "kill_task":
                result = await daemon.op_kill_task(req["task_id"])
            elif op == "ping":
                result = await daemon.op_ping()
            else:
                result = {"is_error": True, "errors": [f"unknown op: {op!r}"]}
        except KeyError as e:
            result = {"is_error": True, "errors": [f"missing field: {e!r}"]}
        except Exception as e:
            result = {"is_error": True, "errors": [f"daemon: {e!r}"]}

        writer.write((json.dumps(result) + "\n").encode())
        await writer.drain()
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
async def serve() -> int:
    socket_path = daemon_socket_path()
    if socket_path.exists():
        try:
            socket_path.unlink()
        except OSError as e:
            print(
                f"daemon: could not remove stale socket {socket_path}: {e}",
                file=sys.stderr,
            )
            return 1
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    daemon = Daemon()
    daemon._reaper_task = asyncio.create_task(daemon._reaper_loop())

    server = await asyncio.start_unix_server(
        lambda r, w: _handle_client(daemon, r, w),
        path=str(socket_path),
    )
    os.chmod(socket_path, 0o600)
    print(f"daemon: listening on {socket_path}", file=sys.stderr)

    loop = asyncio.get_event_loop()
    shutdown_event = asyncio.Event()

    def _handle_term(*_):
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_term)

    serve_task = asyncio.create_task(server.serve_forever())

    try:
        await shutdown_event.wait()
    finally:
        print("daemon: shutting down", file=sys.stderr)
        server.close()
        try:
            await server.wait_closed()
        except Exception:
            pass
        await daemon.shutdown()
        serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass
        try:
            socket_path.unlink()
        except OSError:
            pass

    return 0


def main() -> int:
    try:
        return asyncio.run(serve())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
