<<<<<<< HEAD
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
=======
from agent_swarm.core.state import SwarmState, DesignDoc
from agent_swarm.agents.factory import AgentFactory
from agent_swarm.tools.research_tools import ResearchTools
from agent_swarm.tools.slack_tools import SlackTools
import json
import os


class SwarmOrchestrator:
    def __init__(self, objective: str = None):
        self.state = SwarmState(objective=objective) if objective else None
        self.researcher = AgentFactory.create_researcher()
        self.engineer = AgentFactory.create_engineer()
        self.validator = AgentFactory.create_validator()
        self.research_tools = ResearchTools()
        self.slack = SlackTools(message_callback=self.handle_slack_instruction)

    def run(self):
        if self.state and self.state.objective:
            self._start_run_thread()

        # Always start listening to keep the process alive and handle approvals/new tasks
>>>>>>> ba3f8a9 (initial version)
        self.slack.start_listening()

    def _start_run_thread(self):
        import threading
<<<<<<< HEAD
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
=======

        thread = threading.Thread(target=self._run_loop, daemon=True)
        thread.start()

    def handle_slack_instruction(self, text: str):
        if not self.state:
            print(f"Received instruction: {text}")
            self.state = SwarmState(objective=text)
            self._start_run_thread()
        else:
            print(f"Swarm already running. Ignoring: {text}")

    def _run_loop(self):
        self.slack.post_message(
            f"🚀 *Swarm Activated* 🚀\nObjective: {self.state.objective}"
        )

        while True:
            if self.state.current_step == "research":
                self.do_research()
            elif self.state.current_step == "design":
                self.do_design()
            elif self.state.current_step == "coding":
                self.do_coding()
            elif self.state.current_step == "testing":
                self.do_testing()
            elif self.state.current_step == "done":
                self.slack.post_message("✅ *Task Completed Successfully* ✅")
                break
            else:
                print(f"Unknown step: {self.state.current_step}")
                break

    def do_research(self):
        self.slack.post_message("🔍 Researcher is looking into the objective...")

        messages = [
            {
                "role": "user",
                "content": f"Research the following objective and fetch any relevant paper details or documentation from provided URLs: {self.state.objective}",
            }
        ]

        # Tool map for researchers
        tool_map = {
            "search_arxiv": self.research_tools.search_arxiv,
            "search_web": self.research_tools.search_web,
            "fetch_url": self.research_tools.fetch_url,
        }

        # Tool-use loop
        max_iterations = 5
        for _ in range(max_iterations):
            response = self.researcher.chat(messages, tools=research_tool_definitions)

            if not response.tool_calls:
                self.state.research_notes = response.content
                break

            # Add assistant message with tool calls to history
            messages.append(response)

            for tool_call in response.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                print(f"DEBUG: Researcher calling {fn_name} with {fn_args}")

                if fn_name in tool_map:
                    result = tool_map[fn_name](**fn_args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": fn_name,
                            "content": json.dumps(result),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": fn_name,
                            "content": f"Error: Tool {fn_name} not found.",
                        }
                    )

        if not self.state.research_notes:
            self.state.research_notes = (
                "Research completed, but no notes were generated."
            )

        self.state.update_step("design")

    def do_design(self):
        self.slack.post_message("📝 Researcher is drafting a Design Doc...")
        # Get design from researcher agent
        messages = [
            {
                "role": "user",
                "content": f"Based on these research notes: {self.state.research_notes}, draft a Design Doc for the objective: {self.state.objective}",
            }
        ]
        resp = self.researcher.chat(messages)

        # In a real run, we'd parse the markdown. For mock, we'll assume it's good.
        self.state.design_doc = DesignDoc(
            title=f"Design for {self.state.objective}",
            content=resp.content,
            milestones=["Implement core logic", "Write tests"],
        )

        # Save to artifacts
        artifact_path = os.path.join(self.state.artifacts_path, "design_doc.md")
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        with open(artifact_path, "w") as f:
            f.write(self.state.design_doc.content)

        # Shared full design doc as a snippet
        self.slack.upload_snippet(
            content=self.state.design_doc.content,
            title="Design Document",
            filename="design_doc.md",
        )

        # HITL Approval with preview
        approved = self.slack.request_approval(
            f"New Design Doc Ready: *{self.state.design_doc.title}*\n\n"
            "*Preview:*\n"
            f"{self.state.design_doc.content[:1000]}...\n\n"
            "Please review the full document above and type 'yes' to approve."
        )
        if approved:
            self.state.design_doc.approved = True
            self.state.update_step("coding")
        else:
            self.state.add_feedback("Design rejected by human.")
            self.state.update_step("research")

    def do_coding(self):
        self.slack.post_message("💻 Coding Agent is implementing the design...")
        # Simplified for mock: user will provide the code or agent will generate it
        messages = [
            {
                "role": "user",
                "content": f"Implement the design: {self.state.design_doc.content}",
            }
        ]
        resp = self.engineer.chat(messages)
        self.state.implementation_status["code"] = resp.content

        # Save implementation to artifacts
        code_artifact_path = os.path.join(
            self.state.artifacts_path, "implementation.py"
        )
        with open(code_artifact_path, "w") as f:
            f.write(resp.content)

        self.state.update_step("testing")

    def do_testing(self):
        self.slack.post_message("🧪 Testing Agent is verifying the implementation...")
        # Simplified for mock
        messages = [
            {
                "role": "user",
                "content": f"Verify this code: {self.state.implementation_status['code']}\nBased on design: {self.state.design_doc.content}",
            }
        ]
        resp = self.validator.chat(messages)
        self.state.test_results = resp.content

        # Final HITL Review
        approved = self.slack.request_approval(
            f"Testing Results Ready:\n{self.state.test_results[:500]}..."
        )
        if approved:
            self.state.update_step("done")
        else:
            self.state.add_feedback("Testing failed or review rejected.")
            self.state.update_step("coding")
>>>>>>> ba3f8a9 (initial version)
