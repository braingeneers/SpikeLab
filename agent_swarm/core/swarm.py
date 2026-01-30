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
        self.slack.start_listening()

    def _start_run_thread(self):
        import threading

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
