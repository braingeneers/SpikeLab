"""Tiny Unix-socket client used by ``server.py`` to talk to ``daemon.py``.

Each MCP tool call opens a fresh socket connection, sends one JSON-line
request, awaits one JSON-line response, closes. The daemon's per-connection
handler matches this contract (see ``daemon._handle_client``).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from .config import DAEMON_REQUEST_TIMEOUT_SECONDS, daemon_socket_path


def _socket_path() -> str:
    return str(daemon_socket_path())


def _connection_error(detail: str) -> dict[str, Any]:
    return {
        "is_error": True,
        "errors": [f"oc-spikelab daemon not reachable: {detail}"],
        "response": "",
    }


async def call_daemon(request: dict[str, Any]) -> Any:
    """Round-trip one JSON-line request to the local daemon socket.

    Returns whatever the daemon replied — either a dict (for ``ask`` / ``ping``
    / ``get_task_status`` / ``kill_task``) or a list (for ``list_tasks``). If
    the daemon is down or the request times out, returns a
    ``{"is_error": true, ...}`` dict so callers don't have to wrap every call
    in try/except.
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(_socket_path()),
            timeout=2.0,
        )
    except (FileNotFoundError, ConnectionRefusedError) as e:
        return _connection_error(f"{type(e).__name__}: {e}")
    except asyncio.TimeoutError:
        return _connection_error("connect timeout")
    except OSError as e:
        return _connection_error(f"OSError: {e}")

    try:
        writer.write((json.dumps(request) + "\n").encode())
        await writer.drain()
        try:
            line = await asyncio.wait_for(
                reader.readline(),
                timeout=DAEMON_REQUEST_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return _connection_error(
                f"daemon response timeout after {DAEMON_REQUEST_TIMEOUT_SECONDS}s"
            )
        if not line:
            return _connection_error("daemon closed connection without responding")
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            return _connection_error(f"bad JSON from daemon: {e!r}")
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
