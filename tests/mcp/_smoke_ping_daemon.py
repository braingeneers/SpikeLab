"""Ad-hoc smoke: ping the running daemon. Not part of the registered test
suite — just a one-shot helper used during local install verification."""

from __future__ import annotations

import asyncio
import json
import sys

from spikelab.mcp.config import daemon_socket_path


async def main() -> int:
    path = daemon_socket_path()
    print(f"connecting to {path}")
    reader, writer = await asyncio.open_unix_connection(str(path))
    writer.write(json.dumps({"op": "ping"}).encode() + b"\n")
    await writer.drain()
    line = await reader.readline()
    writer.close()
    await writer.wait_closed()
    print("reply:", line.decode().strip())

    # Also list_tasks to verify on-disk SDK session helpers work.
    reader, writer = await asyncio.open_unix_connection(str(path))
    writer.write(json.dumps({"op": "list_tasks", "kind": None}).encode() + b"\n")
    await writer.drain()
    line = await reader.readline()
    writer.close()
    await writer.wait_closed()
    print("list_tasks reply:", line.decode().strip()[:300])
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
