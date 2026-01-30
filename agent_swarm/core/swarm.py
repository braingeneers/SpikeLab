<<<<<<< HEAD
<<<<<<< HEAD
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
=======
>>>>>>> e11f739 (feat(agent_swarm): Implement swarm enhancements for github, mcp, and slack)
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
<<<<<<< HEAD

<<<<<<< HEAD
        # Always start listening to keep the process alive and handle approvals/new tasks
>>>>>>> ba3f8a9 (initial version)
=======
        # Always start listening to keep the process alive
>>>>>>> c98998a (Implement: take a look at this and implement the functional connectivity metric from this paper: <https://pubmed.ncbi.nlm.nih.gov/29024669/>)
=======
>>>>>>> e11f739 (feat(agent_swarm): Implement swarm enhancements for github, mcp, and slack)
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
<<<<<<< HEAD
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
=======
>>>>>>> ba3f8a9 (initial version)

        thread = threading.Thread(target=self._run_loop, daemon=True)
        thread.start()

<<<<<<< HEAD
    def handle_slack_instruction(self, text: str, user_id: str = None):
        if not self.state:
            print(f"Received instruction: {text} from {user_id}")
            self.state = SwarmState(objective=text, user_id=user_id)
            self.artifact_tools = ArtifactTools(
                self.state.artifacts_path, state_callback=self.state.register_artifact
            )
=======
    def handle_slack_instruction(self, text: str):
        if not self.state:
            print(f"Received instruction: {text}")
            self.state = SwarmState(objective=text)
>>>>>>> ba3f8a9 (initial version)
            self._start_run_thread()
        else:
            print(f"Swarm already running. Ignoring: {text}")

<<<<<<< HEAD
    def _get_context(self) -> str:
        """Constructs a context string from the blackboard and artifact registry."""
        context = f"### System Context\n- Project Root: {os.getcwd()}\n"
        context += (
            f"- Active Agents: Coordinator (Lead), Researcher, Engineer, Validator\n"
        )
        context += f"- Current Milestone: {self.state.current_step}\n"
        context += f"- Swarm State: {self.state.dict()}\n"
        context += "\n### Blackboard\n"
        if not self.state.blackboard:
            context += "No shared information yet.\n"
        else:
            for key, value in self.state.blackboard.items():
                context += f"- **{key}**: {value}\n"

        context += "\n### Artifact Registry\n"
        if not self.state.artifact_registry:
            context += "No artifacts saved yet.\n"
        else:
            for artifact in self.state.artifact_registry:
                context += f"- [{artifact['filename']}]: {artifact['description']}\n"

        context += "\n### Human Feedback\n"
        if not self.state.human_feedback:
            context += "No direct human feedback received yet.\n"
        else:
            for fb in self.state.human_feedback:
                context += f"- {fb}\n"

        return context

    def _execute_agent_step(
        self, agent, messages, tool_definitions, tool_map, max_iterations=10
    ):
        """Generic loop for agent execution with tool support."""
        for _ in range(max_iterations):
            context = self._get_context()
            response = agent.chat(messages, tools=tool_definitions, context=context)

            if not response.tool_calls:
                return response.content

=======
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
>>>>>>> ba3f8a9 (initial version)
            messages.append(response)

            for tool_call in response.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

<<<<<<< HEAD
                print(f"DEBUG: {agent.name} calling {fn_name} with args: {fn_args}")

                if fn_name in tool_map:
                    result = tool_map[fn_name](**fn_args)
                else:
                    result = f"Error: Tool {fn_name} not found."

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": fn_name,
                        "content": (
                            json.dumps(result)
                            if isinstance(result, (dict, list))
                            else str(result)
                        ),
                    }
                )
        return "Max iterations reached without a final answer."

    def _create_pr_from_artifacts(self):
        """Moves artifacts to their target locations, commits, and opens a PR."""
        self.slack.post_message(
            "🔨 *Preparing Pull Request*... Staging changes.", thread_ts=self.root_ts
        )

        # 1. Create branch
        branch_name = f"feat-{self.state.objective[:30]}"
        self.github_tools.create_branch(branch_name)

        # 2. Identify and 'deploy' artifacts
        # For simplicity, we'll look for .py and .yaml files in the artifacts dir
        # and assume the agents specified their target path in the description or just root for now.
        # In a real swarm, the Engineer would use a 'deploy' tool.
        artifacts_dir = self.state.artifacts_path
        deployed_files = []

        for root, _, files in os.walk(artifacts_dir):
            for file in files:
                if file.endswith((".py", ".yaml", ".yml")) and not file.startswith(
                    "test_"
                ):
                    src = os.path.join(root, file)
                    # For now, we'll keep them in a specific 'generated' folder or root if appropriate
                    # Ideally, the agent would give the full target path.
                    # We'll just commit them where they are if they are already in the repo?
                    # No, artifacts are in agent_swarm/artifacts/.

                    src = os.path.join(root, file)

                    # Logic: Find the file in the project to overwrite it
                    # If unique match found, use that path.
                    # Else, default to root or 'generated' (simplified).
                    dest = file
                    found_path = None
                    for r, _, fs in os.walk("."):
                        if "agent_swarm/artifacts" in r or ".git" in r:
                            continue
                        if file in fs:
                            found_path = os.path.join(r, file)
                            break

                    if found_path:
                        dest = found_path
                        # Overwrite
                        shutil.copy2(src, dest)
                        deployed_files.append(dest)
                    else:
                        # New file? Put in agent_swarm for now if it looks like agent code
                        # Or just root.
                        shutil.copy2(src, dest)
                        deployed_files.append(dest)

        if not deployed_files:
            self.slack.post_message(
                "⚠️ No code artifacts found to commit.", thread_ts=self.root_ts
            )
            return

        # 3. Commit and Push
        commit_msg = f"Implement: {self.state.objective}"
        self.github_tools.commit_and_push(commit_msg)

        # 4. Create PR
        pr_title = f"Feat: {self.state.objective}"
        pr_body = f"Automatic implementation generated by Agent Swarm.\n\n### Objective\n{self.state.objective}\n\n### Design Doc\nFull trace in artifacts folder."
        pr_res = self.github_tools.create_pr(pr_title, pr_body)

        if pr_res["status"] == "success":
            pr_url = pr_res["url"]
            self.slack.post_message(
                f"❤️ *Pull Request Opened Successfully!* ❤️\nView here: {pr_url}",
                thread_ts=self.root_ts,
            )
        else:
            self.slack.post_message(
                f"❌ *Failed to create PR*: {pr_res.get('stderr')}",
                thread_ts=self.root_ts,
            )

    def do_coordinate(self, user_text: str):
        """Standard interaction loop for the Coordinator Agent."""
        messages = self.state.conversation_history[-5:]  # Last 5 messages

        tool_map = self._get_common_tool_map()
        tool_map.update(
            {
                "qmd_query": self.qmd_tools.query,
                "qmd_get": self.qmd_tools.get,
            }
        )
        tool_defs = (
            artifact_tool_definitions
            + terminal_tool_definitions
            + qmd_tool_definitions
            + github_tool_definitions
        )

        # We don't use _execute_agent_step here because we want a single turn response if it's just a question,
        # OR a multi-tool execution if the coordinator is working on something.
        result = self._execute_agent_step(
            self.coordinator, messages, tool_defs, tool_map
        )
        clean_res = self._clean_output(result)

        # If the result suggests starting a task, we switch state.
        # We look for a keyword in the thought or result, or just move to research
        # if the coordinator says something like "Initiating research phase"
        if "HANDOFF: RESEARCH" in result or "initiating research" in clean_res.lower():
            self.state.update_step("research")
        else:
            # Stay in coordinate mode for the next user message
            # We'll need a way for the loop to wait for the next message
            # For now, we'll just break and let the Slack handler restart it
            pass

        self.slack.post_message(clean_res, thread_ts=self.root_ts)
        self.state.conversation_history.append(
            {"role": "assistant", "content": clean_res}
        )
=======
        threading.Thread(target=self._run_loop, daemon=True).start()
>>>>>>> e11f739 (feat(agent_swarm): Implement swarm enhancements for github, mcp, and slack)

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

<<<<<<< HEAD
    def do_research(self):
        self.slack.post_message(
            "🔍 Researcher is analyzing the objective...", thread_ts=self.root_ts
        )

        messages = [
            {
                "role": "user",
                "content": f"Research the following objective and save your notes as an artifact: {self.state.objective}",
            }
        ]

        tool_map = self._get_common_tool_map()
        tool_map.update(
            {
                "search_arxiv": self.research_tools.search_arxiv,
                "search_web": self.research_tools.search_web,
                "fetch_url": self.research_tools.fetch_url,
                "qmd_search": self.qmd_tools.search,
                "qmd_query": self.qmd_tools.query,
                "qmd_get": self.qmd_tools.get,
                "search_pubmed": self.pubmed_tools.search_pubmed,
                "get_paper_details": self.pubmed_tools.get_paper_by_pmid,
            }
        )

        tool_defs = (
            research_tool_definitions
            + artifact_tool_definitions
            + qmd_tool_definitions
            + pubmed_tool_definitions
        )

        result = self._execute_agent_step(
            self.researcher, messages, tool_defs, tool_map
        )
        self.state.research_notes = result
        self.state.update_blackboard(
            "research_summary", self._clean_output(result)[:500]
        )
        self.state.update_step("design")

    def do_design(self):
        # Notify main thread
        main_msg = self.slack.post_message(
            "📝 Researcher is drafting the Design Doc...", thread_ts=self.root_ts
        )
        phase_ts = main_msg.get("ts") if main_msg else self.root_ts

        messages = [
            {
                "role": "user",
                "content": f"Draft a Design Doc for: {self.state.objective}. Use research notes from artifacts. Save the final document using `save_artifact`.",
            }
        ]

        tool_map = self._get_common_tool_map()
        tool_defs = artifact_tool_definitions

        result = self._execute_agent_step(
            self.researcher, messages, tool_defs, tool_map
        )

        # Save to state
        self.state.design_doc = DesignDoc(
            title=f"Design for {self.state.objective}",
            content=result,
            milestones=["Core implementation", "Verification tests"],
        )

        # Upload full doc to the thread
        self.slack.upload_snippet(
            result,
            title="Design Document",
            filename="design_doc.md",
            thread_ts=phase_ts,
        )

        # Request approval on the main message for this phase
        clean_summary = self._clean_output(result)
        approved, feedback = self.slack.request_approval(
            f"Design Doc Ready. Please review the thread and react with ✅ to approve or ❌ to reject.\n\n"
            f"*High-Level Summary:*\n{clean_summary}...",
            thread_ts=self.root_ts,
        )

        if feedback:
            self.state.add_feedback(feedback)

=======
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
>>>>>>> ba3f8a9 (initial version)
        if approved:
            self.state.design_doc.approved = True
            self.state.update_step("coding")
        else:
<<<<<<< HEAD
            if not feedback:
                self.state.add_feedback("Design rejected by human.")
            self.state.update_step("research")

    def do_coding(self):
        self.slack.post_message(
            "💻 Engineering Agent is implementing the design...", thread_ts=self.root_ts
        )

        messages = [
            {
                "role": "user",
                "content": f"Implement the design found in the artifacts. Save your implementation(s) using `save_artifact`.",
            }
        ]

        tool_map = self._get_common_tool_map()
        tool_defs = artifact_tool_definitions

        result = self._execute_agent_step(self.engineer, messages, tool_defs, tool_map)
        self.state.implementation_status["log"] = result
        self.state.update_step("testing")

    def do_testing(self):
        self.slack.post_message(
            "🧪 Validator Agent is verifying the implementation...",
            thread_ts=self.root_ts,
        )

        messages = [
            {
                "role": "user",
                "content": f"Verify the implementation from artifacts. Write and run tests (pytests). Save test logs/reports as artifacts.",
            }
        ]

        tool_map = self._get_common_tool_map()
        tool_defs = artifact_tool_definitions + terminal_tool_definitions

        result = self._execute_agent_step(self.validator, messages, tool_defs, tool_map)
        self.state.test_results = result

        clean_result = self._clean_output(result)
        approved, feedback = self.slack.request_approval(
            f"Testing Results Ready. Please react with ✅ to finish or ❌ to send back to coding.\n\n"
            f"*Summary:*\n{clean_result}...",
            thread_ts=self.root_ts,
        )

        if feedback:
            self.state.add_feedback(feedback)

        if approved:
            self.state.update_step("done")
        else:
            if not feedback:
                self.state.add_feedback("Testing failed or review rejected.")
            self.state.update_step("coding")
<<<<<<< HEAD
>>>>>>> ba3f8a9 (initial version)
=======

    def handle_slack_instruction(self, text, user_id):
        print(f"Swarm received instruction: {text}")

        if not self.state or self.state.current_step == "done":
            # Start fresh or after completion
            self.state = SwarmState(objective=text, user_id=user_id)
            self.state.conversation_history.append({"role": "user", "content": text})
            # Re-init agents
            self.coordinator = AgentFactory.create_coordinator()
            self.researcher = AgentFactory.create_researcher()
            self.engineer = AgentFactory.create_engineer()
            self.validator = AgentFactory.create_validator()
            
            self.artifact_tools = ArtifactTools(
                self.state.artifacts_path, 
                state_callback=self.state.register_artifact
            )

            # Start by coordinating
            self.state.update_step("coordinate")

            if not self.loop_running:
                import threading

                threading.Thread(target=self._run_loop).start()
        else:
            # If already running, treat as live feedback
            self.state.add_feedback(f"User update: {text}")
            self.state.conversation_history.append({"role": "user", "content": text})
            # Update objective if we are in coordinate mode so the agent sees the latest
            if self.state.current_step == "coordinate":
                self.state.objective = text
                if not self.loop_running:
                    import threading

                    threading.Thread(target=self._run_loop).start()
>>>>>>> c98998a (Implement: take a look at this and implement the functional connectivity metric from this paper: <https://pubmed.ncbi.nlm.nih.gov/29024669/>)
=======
        # Commit directly from tracked repo changes instead of artifact copying
        self.github_tools.commit_and_push(f"AutoGen Swarm: {self.objective}")
        self.github_tools.create_pr(f"Swarm Fix: {self.objective}", "Details in Slack thread.")

    def handle_slack_instruction(self, text, user_id):
        if not self.loop_running:
            self.objective = text
            self.user_id = user_id
            self._start_run_thread()
>>>>>>> e11f739 (feat(agent_swarm): Implement swarm enhancements for github, mcp, and slack)
=======
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
