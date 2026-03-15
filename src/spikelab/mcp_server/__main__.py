"""
CLI entry point for the MCP server.

Run with: python -m spikelab.mcp_server
    --transport stdio  (default, for local use)
    --transport sse    (network service)
    --port 8080        (SSE port, default 8080)
    --host 0.0.0.0     (SSE host, default 0.0.0.0)
"""

import asyncio
import sys

from .server import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
