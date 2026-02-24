import os
import unittest
from unittest.mock import MagicMock, patch
from agent_swarm.core.swarm import SwarmOrchestrator


class TestPureAutoGen(unittest.TestCase):
    @patch("agent_swarm.core.swarm.SlackTools")
    @patch("agent_swarm.core.swarm.ResearchTools")
    @patch("agent_swarm.core.swarm.PubMedTools")
    @patch("agent_swarm.core.swarm.GithubTools")
    @patch("agent_swarm.core.swarm.TerminalTools")
    @patch("agent_swarm.core.swarm.ArtifactTools")
    @patch("agent_swarm.core.swarm.QmdTools")
    @patch("agent_swarm.core.swarm.FileTools")
    @patch("agent_swarm.core.swarm.MCPTools")
    @patch("agent_swarm.core.swarm.AgentFactory")
    @patch("agent_swarm.core.swarm.LocalCommandLineCodeExecutor")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "dummy-key"})
    def test_orchestrator_initialization(
        self,
        mock_executor,
        mock_agent_factory,
        mock_mcp,
        mock_files,
        mock_qmd,
        mock_artifacts,
        mock_terminal,
        mock_github,
        mock_pubmed,
        mock_research,
        mock_slack,
    ):
        # Setup mocks with annotated functions
        def dummy_search(query: str):
            return []

        def dummy_slack(text: str):
            return {}

        mock_research.return_value.get_tool_map.return_value = {
            "search_arxiv": dummy_search
        }
        mock_slack.return_value.get_tool_map.return_value = {
            "post_slack_message": dummy_slack
        }
        mock_pubmed.return_value.get_tool_map.return_value = {}
        mock_github.return_value.get_tool_map.return_value = {}
        mock_terminal.return_value.get_tool_map.return_value = {}
        mock_artifacts.return_value.get_tool_map.return_value = {}
        mock_qmd.return_value.get_tool_map.return_value = {}
        mock_files.return_value.get_tool_map.return_value = {}
        mock_mcp.return_value.get_tool_map.return_value = {}

        # Mock agents created by factory
        mock_coordinator = MagicMock()
        mock_researcher = MagicMock()
        mock_engineer = MagicMock()
        mock_validator = MagicMock()
        mock_user_proxy = MagicMock()

        mock_agent_factory.create_coordinator.return_value = mock_coordinator
        mock_agent_factory.create_researcher.return_value = mock_researcher
        mock_agent_factory.create_engineer.return_value = mock_engineer
        mock_agent_factory.create_validator.return_value = mock_validator
        mock_agent_factory.create_user_proxy.return_value = mock_user_proxy

        # Test init
        orchestrator = SwarmOrchestrator(objective="Test Objective")

        # Verify initialization
        self.assertEqual(orchestrator.objective, "Test Objective")
        self.assertIsNotNone(orchestrator.llm_config)
        self.assertEqual(
            orchestrator.llm_config["config_list"][0]["model"], "gpt-4-mini"
        )

        # Manually call _init_agents for the test
        orchestrator._init_agents()

        # Verify agents were created
        self.assertTrue(hasattr(orchestrator, "coordinator"))
        self.assertTrue(hasattr(orchestrator, "researcher"))
        self.assertTrue(hasattr(orchestrator, "engineer"))
        self.assertTrue(hasattr(orchestrator, "validator"))
        self.assertTrue(hasattr(orchestrator, "user_proxy"))

        # Verify factory methods were called
        mock_agent_factory.create_coordinator.assert_called()
        mock_agent_factory.create_researcher.assert_called()
        mock_agent_factory.create_engineer.assert_called()
        mock_agent_factory.create_validator.assert_called()
        mock_agent_factory.create_user_proxy.assert_called()


if __name__ == "__main__":
    unittest.main()
