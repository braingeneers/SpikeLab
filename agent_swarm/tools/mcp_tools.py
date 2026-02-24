import sys
import json
import asyncio
from typing import Dict, Any, Callable, List

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class MCPTools:
    def __init__(self, server_path: str = "mcp_server"):
        # Configure the command to run the local MCP server
        if MCP_AVAILABLE:
            self.server_params = StdioServerParameters(
                command=sys.executable, args=["-m", server_path], env=None
            )
        else:
            self.server_params = None

    def _run_sync(self, coro):
        """Helper to run async code synchronously since AutoGen tool calls are synchronous."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)

    async def _call_mcp_tool_async(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not MCP_AVAILABLE:
            return {
                "status": "error",
                "message": "MCP SDK is not installed or requires Python 3.10+. Tool execution skipped.",
            }

        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Call the specified tool
                    result = await session.call_tool(tool_name, arguments=tool_args)

                    # Note: MCP tool results typically return a list of content blocks
                    # We extract the text content and join it.
                    if result.content:
                        text_results = [
                            block.text
                            for block in result.content
                            if hasattr(block, "text")
                        ]
                        return {"status": "success", "result": "\n".join(text_results)}
                    elif result.isError:
                        return {
                            "status": "error",
                            "message": "The server returned an error but no text.",
                        }
                    else:
                        return {
                            "status": "success",
                            "result": "Tool executed successfully (no output).",
                        }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def _list_mcp_tools_async(self) -> Dict[str, Any]:
        if not MCP_AVAILABLE:
            return {
                "status": "error",
                "message": "MCP SDK is not installed or requires Python 3.10+. Tool listing skipped.",
            }

        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # List tools
                    result = await session.list_tools()

                    tools = []
                    for tool in result.tools:
                        tools.append(
                            {
                                "name": tool.name,
                                "description": tool.description,
                            }
                        )

                    return {"status": "success", "tools": tools}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def call_mcp_tool(self, tool_name: str, tool_args: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for calling an MCP tool. Expects tool_args as a JSON string
        from the LLM to safely decode into a dictionary.
        """
        try:
            # LLMs sometimes send stringified JSON or dicts. Try to parse if it's a string.
            if isinstance(tool_args, str):
                args_dict = json.loads(tool_args)
            elif isinstance(tool_args, dict):
                args_dict = tool_args
            else:
                return {
                    "status": "error",
                    "message": "tool_args must be a JSON string or dict.",
                }

            return self._run_sync(self._call_mcp_tool_async(tool_name, args_dict))
        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "message": f"Failed to parse tool_args JSON: {str(e)}",
            }

    def list_mcp_tools(self) -> Dict[str, Any]:
        """Synchronous wrapper for listing MCP tools."""
        return self._run_sync(self._list_mcp_tools_async())

    def get_tool_map(self) -> Dict[str, Callable]:
        return {
            "call_mcp_tool": self.call_mcp_tool,
            "list_mcp_tools": self.list_mcp_tools,
        }


mcp_tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "call_mcp_tool",
            "description": "Calls a tool on the local MCP server. You must call list_mcp_tools first to know what tools are available and what parameters they require.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "The exact name of the MCP tool to call (e.g., 'run_python_script').",
                    },
                    "tool_args": {
                        "type": "string",
                        "description": 'A valid JSON string containing the arguments required by the MCP tool. Example: \'{"script_path": "foo.py"}\'',
                    },
                },
                "required": ["tool_name", "tool_args"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_mcp_tools",
            "description": "Lists all available tools and their descriptions from the local MCP server.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
