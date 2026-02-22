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
    @patch("agent_swarm.core.swarm.LocalCommandLineCodeExecutor")
    @patch.dict("os.environ", {"OPENAI_API_KEY": "dummy-key"})
    def test_orchestrator_initialization(self, mock_executor, mock_mcp, mock_files, mock_qmd, mock_artifacts, mock_terminal, 
                                        mock_github, mock_pubmed, mock_research, mock_slack):
        # Setup mocks with annotated functions
        def dummy_search(query: str): return []
        def dummy_slack(text: str): return {}
        
        mock_research.return_value.get_tool_map.return_value = {"search_arxiv": dummy_search}
        mock_slack.return_value.get_tool_map.return_value = {"post_to_slack": dummy_slack}
        mock_pubmed.return_value.get_tool_map.return_value = {}
        mock_github.return_value.get_tool_map.return_value = {}
        mock_terminal.return_value.get_tool_map.return_value = {}
        mock_artifacts.return_value.get_tool_map.return_value = {}
        mock_qmd.return_value.get_tool_map.return_value = {}
        mock_files.return_value.get_tool_map.return_value = {}
        mock_mcp.return_value.get_tool_map.return_value = {}
        
        # Test init (will NOT call _init_agents in latest code, only saves objective)
        orchestrator = SwarmOrchestrator(objective="Test Objective")
        
        # Manually call _init_agents for the test since run() does it now
        orchestrator._init_agents()
        
        self.assertEqual(orchestrator.objective, "Test Objective")
        self.assertTrue(hasattr(orchestrator, "coordinator"))
        self.assertTrue(hasattr(orchestrator, "researcher"))
        self.assertTrue(hasattr(orchestrator, "engineer"))
        self.assertTrue(hasattr(orchestrator, "validator"))
        self.assertTrue(hasattr(orchestrator, "user_proxy"))
        
        # Verify tool registration happened
        # (AssistantAgents should have tools registered)
        self.assertIn("search_arxiv", orchestrator.coordinator.llm_config["tools"][0]["function"]["name"])

if __name__ == "__main__":
    unittest.main()
