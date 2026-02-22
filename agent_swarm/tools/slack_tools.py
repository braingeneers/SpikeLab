import os
from typing import Dict, Callable, List, Any
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError


class SlackTools:
    def __init__(self, message_callback=None):
        token = os.getenv("SLACK_BOT_TOKEN")
        self.app = App(token=token) if token else None
        self.channel_id = os.getenv("SLACK_CHANNEL_ID") or "C08NSDULE7M"
        self.app_token = os.getenv("SLACK_APP_TOKEN")
        self.message_callback = message_callback
        self.waiting_for_approval = False
        self.approval_result = None
        self.last_message_ts = None
        self.last_feedback = None
        self.waiting_for_input = False
        self.input_result = None

        if self.app and self.message_callback:

            @self.app.message("")  # Listen to all messages
            def handle_message(event, say, logger):
                channel = event.get("channel")
                text = event.get("text")
                bot_id = event.get("bot_id")
                thread_ts = event.get("thread_ts")
                subtype = event.get("subtype")

                # Ignore bot messages and message_changed events (unfurls, edits)
                if channel == self.channel_id and not bot_id and text and not subtype:
                    user_id = event.get("user")

                    if self.waiting_for_approval:
                        approvals = [
                            "yes",
                            "approve",
                            "ok",
                            "lgtm",
                            "✅",
                            ":white_check_mark:",
                            ":heavy_check_mark:",
                            ":check_mark:",
                        ]
                        rejections = [
                            "no",
                            "reject",
                            "stop",
                            "❌",
                            ":x:",
                            ":no_entry_sign:",
                        ]

                        clean_text = text.strip().lower()
                        found_approval = any(token in clean_text for token in approvals)
                        found_rejection = any(
                            token in clean_text for token in rejections
                        )

                        if found_approval:
                            self.approval_result = True
                            self.last_feedback = text
                            self.waiting_for_approval = False
                        elif found_rejection:
                            self.approval_result = False
                            self.last_feedback = text
                            self.waiting_for_approval = False
                    
                    elif self.waiting_for_input:
                        self.input_result = text
                        self.waiting_for_input = False

                    else:
                        # React with eyes to acknowledge receipt
                        try:
                            self.app.client.reactions_add(
                                name="eyes",
                                channel=channel,
                                timestamp=event["ts"]
                            )
                        except SlackApiError as e:
                            # Ignore if already reacted or other minor error
                            print(f"DEBUG: Could not react with eyes: {e}")

                        self.message_callback(text, user_id)

            @self.app.event("reaction_added")
            def handle_reaction(event, logger):
                if not self.waiting_for_approval:
                    return

                item = event.get("item", {})
                if (
                    item.get("type") == "message"
                    and item.get("ts") == self.last_message_ts
                ):
                    reaction = event.get("reaction")
                    if reaction in [
                        "white_check_mark",
                        "heavy_check_mark",
                        "yes",
                        "approve",
                        "thumbsup",
                    ]:
                        self.approval_result = True
                        self.waiting_for_approval = False
                    elif reaction in ["x", "no_entry_sign", "reject", "thumbsdown"]:
                        self.approval_result = False
                        self.waiting_for_approval = False

    def post_message(self, text: str, blocks: list = None, thread_ts: str = None):
        if not self.app or not self.channel_id:
            print(f"SLACK NOT CONFIGURED: {text}")
            return None

        try:
            print(f"DEBUG: SlackTools.post_message called with text: {text[:50]}...")
            resp = self.app.client.chat_postMessage(
                channel=self.channel_id, text=text, blocks=blocks, thread_ts=thread_ts
            )
            self.last_message_ts = resp.get("ts")
            print("DEBUG: SlackTools.post_message success")
            return resp
        except SlackApiError as e:
            print(f"Error posting to Slack: {e}")
            return None

    def upload_snippet(
        self, content: str, title: str, filename: str = "doc.md", thread_ts: str = None
    ):
        if not self.app or not self.channel_id:
            print(f"SLACK NOT CONFIGURED: Cannot upload {title}")
            return None

        try:
            return self.app.client.files_upload_v2(
                channel=self.channel_id,
                content=content,
                title=title,
                filename=filename,
                initial_comment=f"📄 *Full {title} uploaded for review*",
                thread_ts=thread_ts,
            )
        except SlackApiError as e:
            print(f"Error uploading file to Slack: {e}")
            return None

    def request_approval(self, message: str, thread_ts: str = None):
        resp = self.post_message(
            f"🚨 *APPROVAL REQUIRED* 🚨\n{message}", thread_ts=thread_ts
        )
        # last_message_ts is updated inside post_message

        print(f"\n[SLACK WAIT] Requesting approval for: {message}")

        self.waiting_for_approval = True
        self.approval_result = None
        self.last_feedback = None

        import time

        while self.waiting_for_approval:
            time.sleep(1)

        return self.approval_result, self.last_feedback

    def request_input(self, question: str, thread_ts: str = None):
        resp = self.post_message(
            f"❓ *QUESTION* ❓\n{question}", thread_ts=thread_ts
        )
        
        print(f"\n[SLACK WAIT] Asking user: {question}")
        
        self.waiting_for_input = True
        self.input_result = None
        
        import time 
        
        while self.waiting_for_input:
            time.sleep(1)
            
        return self.input_result

    def start_listening(self):
        if not self.app or not self.app_token:
            print(
                "Missing SLACK_BOT_TOKEN or SLACK_APP_TOKEN. Listening mode disabled."
            )
            return

        self.post_message("📡 *Agent Swarm Online* and listening for instructions...")
        handler = SocketModeHandler(self.app, self.app_token)
        print("⚡️ Slack Swarm is listening for instructions...")
        handler.start()

    def get_tool_map(self, thread_ts: str = None) -> Dict[str, Callable]:
        """Returns a map of tool names to functions, optionally pre-filled with thread_ts."""
        import functools

        def post_to_slack_wrapped(text: str) -> Dict[str, Any]:
            # We don't use 'self' here because we want to use the method on the instance
            res = self.post_message(text, thread_ts=thread_ts)
            return {"status": "success", "ts": res.get("ts")} if res else {"status": "error"}

        def request_approval_wrapped(message: str) -> Dict[str, Any]:
            approved, feedback = self.request_approval(message, thread_ts=thread_ts)
            return {"approved": approved, "feedback": feedback}

        def ask_user_wrapped(question: str) -> Dict[str, Any]:
            response = self.request_input(question, thread_ts=thread_ts)
            return {"response": response}

        return {
            "post_to_slack": post_to_slack_wrapped,
            "request_human_approval": request_approval_wrapped,
            "ask_user": ask_user_wrapped,
        }


slack_tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "post_to_slack",
            "description": "Post a status update or message to the Slack channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The message text."}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_human_approval",
            "description": "Request human approval for a design doc or code change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Description of what needs approval.",
                    }
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask the user a clear question and wait for their text response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the user.",
                    }
                },
                "required": ["question"],
            },
        },
    },
]

