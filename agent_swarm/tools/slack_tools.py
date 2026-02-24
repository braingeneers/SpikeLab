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

        if self.app and self.message_callback:

            @self.app.message("")
            def handle_message(event, say, logger):
                channel = event.get("channel")
                text = event.get("text")
                bot_id = event.get("bot_id")
                subtype = event.get("subtype")

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
                    else:
                        try:
                            self.app.client.reactions_add(
                                name="eyes", channel=channel, timestamp=event.get("ts")
                            )
                        except SlackApiError:
                            pass

                        self.last_message_ts = event.get("ts")
                        if self.message_callback:
                            self.message_callback(text, user_id)

    def post_message(self, text: str, thread_ts: str = None) -> Dict[str, Any]:
        if not self.app:
            return {"error": "Slack bot not configured"}

        try:
            result = self.app.client.chat_postMessage(
                channel=self.channel_id, text=text, thread_ts=thread_ts
            )
            return result
        except SlackApiError as e:
            return {"error": str(e)}

    def upload_snippet(
        self, content: str, title: str, filename: str, thread_ts: str = None
    ) -> Dict[str, Any]:
        if not self.app:
            return {"error": "Slack bot not configured"}

        try:
            result = self.app.client.files_upload(
                channels=self.channel_id,
                content=content,
                title=title,
                filename=filename,
                thread_ts=thread_ts,
            )
            return result
        except SlackApiError as e:
            return {"error": str(e)}

    def request_approval(self, text: str, thread_ts: str = None) -> tuple[bool, str]:
        if not self.app:
            return False, "Slack not configured"

        self.waiting_for_approval = True
        self.approval_result = None
        self.last_feedback = None

        try:
            self.app.client.chat_postMessage(
                channel=self.channel_id,
                text=text,
                blocks=[
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": text},
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": "React with ✅ to approve or ❌ to reject.",
                        },
                    },
                ],
                thread_ts=thread_ts,
            )
        except SlackApiError as e:
            return False, f"Error posting approval request: {str(e)}"

        import time

        timeout = 300
        start = time.time()
        while time.time() - start < timeout:
            if not self.waiting_for_approval:
                return self.approval_result, self.last_feedback or ""
            time.sleep(1)

        self.waiting_for_approval = False
        return False, "Approval request timed out"

    def start_listening(self):
        if not self.app or not self.app_token:
            print("Slack app or app token not configured")
            return

        handler = SocketModeHandler(self.app, self.app_token)
        handler.start()

    def get_tool_map(self, thread_ts: str = None) -> Dict[str, Callable]:
        return {
            "post_slack_message": lambda text: self.post_message(text, thread_ts),
            "upload_snippet": lambda content, title, filename: self.upload_snippet(
                content, title, filename, thread_ts
            ),
        }


slack_tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "post_slack_message",
            "description": "Post a message to Slack.",
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
            "name": "upload_snippet",
            "description": "Upload a snippet/file to Slack.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "File content."},
                    "title": {"type": "string", "description": "Title of the snippet."},
                    "filename": {"type": "string", "description": "Filename."},
                },
                "required": ["content", "title", "filename"],
            },
        },
    },
]
