from agent_swarm.core.state import SwarmState, DesignDoc
from agent_swarm.agents.factory import AgentFactory
from agent_swarm.tools.research_tools import ResearchTools, research_tool_definitions
from agent_swarm.tools.slack_tools import SlackTools, slack_tool_definitions
from agent_swarm.tools.artifact_tools import ArtifactTools, artifact_tool_definitions
from agent_swarm.tools.qmd_tools import QmdTools, qmd_tool_definitions
from agent_swarm.tools.pubmed_tools import PubMedTools, pubmed_tool_definitions
from agent_swarm.tools.github_tools import GithubTools, github_tool_definitions
from agent_swarm.tools.terminal_tools import TerminalTools, terminal_tool_definitions
import json
import os
import re
import shutil


class SwarmOrchestrator:
    def _clean_output(self, text: str) -> str:
        """Remove <thought> tags and other internal XML-like tags for cleaner Slack display."""
        if not text:
            return ""
        # Remove anything between <thought> and </thought>
        text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL)
        # Remove any stray tags
        text = re.sub(r"<[^>]+>", "", text)
        return text.strip()

    def __init__(self, objective: str = None):
        self.state = SwarmState(objective=objective) if objective else None
        self.researcher = AgentFactory.create_researcher()
        self.engineer = AgentFactory.create_engineer()
        self.validator = AgentFactory.create_validator()

        self.research_tools = ResearchTools()
        self.qmd_tools = QmdTools()
        self.pubmed_tools = PubMedTools()
        self.github_tools = GithubTools()
        self.terminal_tools = TerminalTools(base_dir=".")
        self.slack = SlackTools(message_callback=self.handle_slack_instruction)
        self.loop_running = False

        if self.state:
            self.coordinator = AgentFactory.create_coordinator()
            self.artifact_tools = ArtifactTools(
                self.state.artifacts_path, state_callback=self.state.register_artifact
            )

    def run(self):
        if self.state and self.state.objective:
            self._start_run_thread()

        # Always start listening to keep the process alive
        self.slack.start_listening()

    def _start_run_thread(self):
        import threading

        thread = threading.Thread(target=self._run_loop, daemon=True)
        thread.start()

    def handle_slack_instruction(self, text: str, user_id: str = None):
        if not self.state:
            print(f"Received instruction: {text} from {user_id}")
            self.state = SwarmState(objective=text, user_id=user_id)
            self.artifact_tools = ArtifactTools(
                self.state.artifacts_path, state_callback=self.state.register_artifact
            )
            self._start_run_thread()
        else:
            print(f"Swarm already running. Ignoring: {text}")

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

            messages.append(response)

            for tool_call in response.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

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

    def _run_loop(self):
        self.loop_running = True
        mention = f"<@{self.state.user_id}> " if self.state.user_id else ""
        resp = self.slack.post_message(
            f"🚀 *Swarm Activated* {mention}🚀\nObjective: {self.state.objective}"
        )
        self.root_ts = resp.get("ts") if resp else None

        try:
            while True:
                if self.state.current_step == "coordinate":
                    self.do_coordinate(self.state.objective)
                    if self.state.current_step == "coordinate":
                        # Coordination finished but no handoff yet.
                        # End loop and wait for next Slack message.
                        break
                elif self.state.current_step == "research":
                    self.do_research()
                elif self.state.current_step == "design":
                    self.do_design()
                elif self.state.current_step == "coding":
                    self.do_coding()
                elif self.state.current_step == "testing":
                    self.do_testing()
                elif self.state.current_step == "done":
                    # Final wrap up - Create PR before finishing
                    self._create_pr_from_artifacts()
                    self.slack.post_message(
                        "🏁 *Swarm task finished.* Results are in the PR thread.",
                        thread_ts=self.root_ts,
                    )
                    break
                else:
                    break
        finally:
            self.loop_running = False

    def _get_common_tool_map(self):
        return {
            "save_artifact": self.artifact_tools.save_artifact,
            "post_slack_message": lambda text: self.slack.post_message(
                self._clean_output(text), thread_ts=self.root_ts
            ),
            "run_terminal_command": self.terminal_tools.run_command,
            "create_github_issue": self.github_tools.create_issue,
            "create_github_pr": self.github_tools.create_pr,
        }

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

        if approved:
            self.state.design_doc.approved = True
            self.state.update_step("coding")
        else:
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
