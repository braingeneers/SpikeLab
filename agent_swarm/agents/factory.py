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
        )
