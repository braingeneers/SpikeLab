<<<<<<< HEAD
import os
from autogen import AssistantAgent, UserProxyAgent

class AgentFactory:
    @staticmethod
    def _read_prompt(path: str) -> str:
        full_path = os.path.join(os.getcwd(), path)
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                return f.read()
        return f"System prompt not found at {path}"

    @staticmethod
    def _is_termination_msg(x):
        content = x.get("content") or ""
        if x.get("name") == "UserProxy":
            return False
        # Strip whitespace and trailing punctuation (.,!;) to handle "TERMINATE_SWARM." or "TERMINATE_SWARM!"
        return content.strip().rstrip(".,!;").endswith("TERMINATE_SWARM")


    @staticmethod
    def create_researcher(llm_config: dict):
        return AssistantAgent(
            name="Researcher",
            system_message=AgentFactory._read_prompt("agent_swarm/agents/prompts/researcher.md"),
            llm_config=llm_config,
            is_termination_msg=AgentFactory._is_termination_msg,
        )

    @staticmethod
    def create_engineer(llm_config: dict):
        return AssistantAgent(
            name="Engineer",
            system_message=AgentFactory._read_prompt("agent_swarm/agents/prompts/engineer.md"),
            llm_config=llm_config,
            is_termination_msg=AgentFactory._is_termination_msg,
        )

    @staticmethod
    def create_validator(llm_config: dict):
        return AssistantAgent(
            name="Validator",
            system_message=AgentFactory._read_prompt("agent_swarm/agents/prompts/validator.md"),
            llm_config=llm_config,
            is_termination_msg=AgentFactory._is_termination_msg,
        )

    @staticmethod
    def create_coordinator(llm_config: dict):
        return AssistantAgent(
            name="Coordinator",
            system_message=AgentFactory._read_prompt("agent_swarm/agents/prompts/coordinator.md"),
            llm_config=llm_config,
            is_termination_msg=AgentFactory._is_termination_msg,
        )

    @staticmethod
    def create_user_proxy(executor):
        return UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            code_execution_config={"executor": executor},
            is_termination_msg=lambda x: "TERMINATE" in (x.get("content") or ""),
            default_auto_reply="Please continue with the objective or signal TERMINATE if finished.",
=======
from agent_swarm.agents.base import BaseAgent
from agent_swarm.tools.research_tools import research_tool_definitions
from agent_swarm.tools.slack_tools import slack_tool_definitions


class AgentFactory:
    @staticmethod
    def create_researcher():
        return BaseAgent(
            name="Researcher", prompt_file="agent_swarm/agents/prompts/researcher.md"
        )

    @staticmethod
    def create_engineer():
        return BaseAgent(
            name="Engineer", prompt_file="agent_swarm/agents/prompts/engineer.md"
        )

    @staticmethod
    def create_validator():
        return BaseAgent(
            name="Validator", prompt_file="agent_swarm/agents/prompts/validator.md"
>>>>>>> ba3f8a9 (initial version)
        )
