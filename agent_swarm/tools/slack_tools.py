import os
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

        if self.app and self.message_callback:

            @self.app.message("")  # Listen to all messages
            def handle_message(event, say, logger):
                channel = event.get("channel")
                text = event.get("text")
                bot_id = event.get("bot_id")
                thread_ts = event.get("thread_ts")

                # Ignore bot messages
                if channel == self.channel_id and not bot_id and text:
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
                    else:
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
            resp = self.app.client.chat_postMessage(
                channel=self.channel_id, text=text, blocks=blocks, thread_ts=thread_ts
            )
            self.last_message_ts = resp.get("ts")
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
]
