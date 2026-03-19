"""
CLI entry point for the MCP server.

Run with: python -m SpikeLab.mcp_server
"""

import asyncio
import sys

from .server import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
