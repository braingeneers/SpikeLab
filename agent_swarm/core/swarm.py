from agent_swarm.agents.factory import AgentFactory
from agent_swarm.tools.research_tools import ResearchTools, research_tool_definitions
from agent_swarm.tools.slack_tools import SlackTools, slack_tool_definitions
from agent_swarm.tools.artifact_tools import ArtifactTools, artifact_tool_definitions
from agent_swarm.tools.qmd_tools import QmdTools, qmd_tool_definitions
from agent_swarm.tools.pubmed_tools import PubMedTools, pubmed_tool_definitions
from agent_swarm.tools.github_tools import GithubTools, github_tool_definitions
from agent_swarm.tools.terminal_tools import TerminalTools, terminal_tool_definitions
from agent_swarm.tools.file_tools import FileTools, file_tool_definitions
from agent_swarm.tools.mcp_tools import MCPTools, mcp_tool_definitions
from agent_swarm.tools.autogen_tools import register_swarm_tools
import json
import os
import re
import shutil
from autogen import GroupChat, GroupChatManager, Agent
from autogen.coding import LocalCommandLineCodeExecutor


class SwarmOrchestrator:
    def _clean_output(self, text: str) -> str:
        """Remove XML-like tags for cleaner Slack display."""
        if not text:
            return ""
        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", "", text)
        return text.strip()

    def __init__(self, objective: str = None, user_id: str = None):
        self.objective = objective
        self.user_id = user_id
        self.loop_running = False
        self.root_ts = None
        self.artifacts_path = "agent_swarm/artifacts/"
        
        self.llm_config = {
            "config_list": [
                {
                    "model": "gpt-5-mini",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "api_type": "openai"  # <--- Add this
                }
            ],
            "timeout": 60,
            # "stream": False  #<-- 'stream' is often not supported in the root llm_config in older versions
        }
                
        # Tools initialization
        self.slack = SlackTools(message_callback=self.handle_slack_instruction)
        self.research_tools = ResearchTools()
        self.qmd_tools = QmdTools()
        self.pubmed_tools = PubMedTools()
        self.github_tools = GithubTools()
        self.file_tools = FileTools()
        self.mcp_tools = MCPTools()
        self.terminal_tools = TerminalTools(base_dir=".")
        self.artifact_tools = ArtifactTools(self.artifacts_path)
        self.executor = LocalCommandLineCodeExecutor(work_dir="agent_swarm/work")
        
        if self.objective:
            self.root_ts = None # Ensure it starts fresh
            # self._init_agents()  <-- REMOVED: _run_loop will handle this with proper ts

    def _init_agents(self):
        self.coordinator = AgentFactory.create_coordinator(self.llm_config)
        self.researcher = AgentFactory.create_researcher(self.llm_config)
        self.engineer = AgentFactory.create_engineer(self.llm_config)
        self.validator = AgentFactory.create_validator(self.llm_config)
        self.user_proxy = AgentFactory.create_user_proxy(self.executor)

        # Build master tool map (common tools for all agents)
        common_tool_map = {}
        common_tool_map.update(self.research_tools.get_tool_map())
        common_tool_map.update(self.qmd_tools.get_tool_map())
        common_tool_map.update(self.pubmed_tools.get_tool_map())
        common_tool_map.update(self.github_tools.get_tool_map())
        common_tool_map.update(self.terminal_tools.get_tool_map())
        common_tool_map.update(self.artifact_tools.get_tool_map())
        common_tool_map.update(self.file_tools.get_tool_map())
        common_tool_map.update(self.mcp_tools.get_tool_map())

        common_tool_defs = (
            research_tool_definitions + qmd_tool_definitions + 
            pubmed_tool_definitions + github_tool_definitions + 
            terminal_tool_definitions + artifact_tool_definitions +
            file_tool_definitions + mcp_tool_definitions
        )
        
        # Tools specifically for Coordinator (Slack communication)
        coordinator_tool_map = dict(common_tool_map)
        coordinator_tool_map.update(self.slack.get_tool_map(thread_ts=self.root_ts))
        
        coordinator_tool_defs = common_tool_defs + slack_tool_definitions
        
        # Register tools
        register_swarm_tools(self.coordinator, self.user_proxy, coordinator_tool_map, coordinator_tool_defs)
        
        for agent in [self.researcher, self.engineer, self.validator]:
            register_swarm_tools(agent, self.user_proxy, common_tool_map, common_tool_defs)

    def run(self):
        if self.objective:
            self._start_run_thread()
        self.slack.start_listening()

    def _start_run_thread(self):
        import threading
        threading.Thread(target=self._run_loop, daemon=True).start()

    def _run_loop(self):
        self.loop_running = True
        mention = f"<@{self.user_id}> " if self.user_id else ""
        resp = self.slack.post_message(
            f"🚀 *AutoGen Swarm Activated* {mention}🚀\nObjective: {self.objective}"
        )
        self.root_ts = resp.get("ts") if resp else None
        
        self._init_agents()

        groupchat = GroupChat(
            agents=[self.user_proxy, self.coordinator, self.researcher, self.engineer, self.validator],
            messages=[],
            max_round=30,
            speaker_selection_method="auto",
            allow_repeat_speaker=False,
        )
        # Debug: Print masked API key
        key = self.llm_config["config_list"][0].get("api_key", "")
        print(f"DEBUG: Using API key: {key[:5]}...{key[-5:] if len(key) > 10 else ''} (Length: {len(key)})")

        manager = GroupChatManager(groupchat=groupchat, llm_config=self.llm_config)

        try:
            self.user_proxy.initiate_chat(
                manager,
                message=(
                    f"Objective: {self.objective}\n\n"
                    "Workflow:\n"
                    "1. Researcher: Search paper, formalize metrics, draft design.\n"
                    "2. Engineer: Implement PEP8 code with type hints.\n"
                    "3. Validator: Run pytests and ensure coverage.\n"
                    "4. Signal TERMINATE when complete.\n"
                ),
            )
            self._create_pr()
            self.slack.post_message("🏁 *Task complete.*", thread_ts=self.root_ts)
        finally:
            self.loop_running = False

    def _create_pr(self):
        self.slack.post_message("🔨 *Staging Pull Request*...", thread_ts=self.root_ts)
        branch_name = f"swarm-{os.urandom(4).hex()}"
        self.github_tools.create_branch(branch_name)

        # Commit directly from tracked repo changes instead of artifact copying
        self.github_tools.commit_and_push(f"AutoGen Swarm: {self.objective}")
        self.github_tools.create_pr(f"Swarm Fix: {self.objective}", "Details in Slack thread.")

    def handle_slack_instruction(self, text, user_id):
        if not self.loop_running:
            self.objective = text
            self.user_id = user_id
            self._start_run_thread()
