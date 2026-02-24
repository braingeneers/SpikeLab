import inspect
import functools
from typing import Any, Callable, Dict, List
from autogen import ConversableAgent


def register_swarm_tools(
    agent: ConversableAgent,
    executor_agent: ConversableAgent,
    tool_map: Dict[str, Callable],
    tool_definitions: List[Dict[str, Any]],
):
    """
    Registers existing swarm tools to an AutoGen agent and its executor using an explicit map.

    Args:
        agent: The AssistantAgent that will suggest tool calls.
        executor_agent: The UserProxyAgent that will execute the tool calls.
        tool_map: Dict mapping tool names to implementation functions.
        tool_definitions: List of tool definitions (OpenAI format).
    """
    for tool_def in tool_definitions:
        name = tool_def["function"]["name"]
        func = tool_map.get(name)

        if func:
            # Use functools.wraps to preserve signature and annotations
            @functools.wraps(func)
            def wrapper(*args, func=func, **kwargs):
                return func(*args, **kwargs)

            # Register for LLM suggestion
            agent.register_for_llm(
                name=name, description=tool_def["function"]["description"]
            )(wrapper)
            # Register for execution ONLY if not already registered (avoids warnings)
            if name not in executor_agent.function_map:
                executor_agent.register_for_execution(name=name)(wrapper)
        else:
            print(
                f"Warning: Tool function implementation for '{name}' not found in tool_map."
            )
