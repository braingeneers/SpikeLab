from agent_swarm.agents.base import BaseAgent
from agent_swarm.tools.research_tools import research_tool_definitions
from agent_swarm.tools.slack_tools import slack_tool_definitions
from agent_swarm.tools.artifact_tools import artifact_tool_definitions
from agent_swarm.tools.qmd_tools import qmd_tool_definitions
from agent_swarm.tools.pubmed_tools import pubmed_tool_definitions
from agent_swarm.tools.github_tools import github_tool_definitions
from agent_swarm.tools.terminal_tools import terminal_tool_definitions


class AgentFactory:
    @staticmethod
    def create_researcher():
        return BaseAgent(
            name="Researcher",
            role="Lead Researcher",
            prompt_path="agent_swarm/agents/prompts/researcher.md",
        )

    @staticmethod
    def create_engineer():
        return BaseAgent(
            name="Engineer",
            role="Lead Engineer",
            prompt_path="agent_swarm/agents/prompts/engineer.md",
        )

    @staticmethod
    def create_validator():
        return BaseAgent(
            name="Validator",
            role="Verification Agent",
            prompt_path="agent_swarm/agents/prompts/validator.md",
        )

    @staticmethod
    def create_coordinator():
        return BaseAgent(
            name="Coordinator",
            role="Swarm Coordinator",
            prompt_path="agent_swarm/agents/prompts/coordinator.md",
        )
