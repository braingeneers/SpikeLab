import subprocess
import os
import re
from typing import Dict, Any, List


class GithubTools:
    def __init__(self):
        pass

    def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return {"status": "success", "stdout": result.stdout}
        except subprocess.CalledProcessError as e:
            return {"status": "error", "stderr": e.stderr}

    def create_branch(self, branch_name: str) -> Dict[str, Any]:
        # Clean branch name
        branch_name = re.sub(r"[^a-zA-Z0-9-_]", "-", branch_name).lower()
        # Create and checkout
        return self._run_command(["git", "checkout", "-b", branch_name])

    def commit_and_push(self, message: str) -> Dict[str, Any]:
        # Stage all changes
        self._run_command(["git", "add", "."])
        # Commit
        commit_res = self._run_command(["git", "commit", "-m", message])
        if commit_res["status"] == "error":
            return commit_res
        # Push
        branch_res = self._run_command(["git", "branch", "--show-current"])
        branch = branch_res.get("stdout", "").strip()
        return self._run_command(["git", "push", "-u", "origin", branch])

    def create_pr(self, title: str, body: str) -> Dict[str, Any]:
        try:
            # Using gh CLI
            cmd = ["gh", "pr", "create", "--title", title, "--body", body]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Find the PR URL in stdout
            url_match = re.search(r"https://github.com/[^\s]+", result.stdout)
            url = url_match.group(0) if url_match else result.stdout.strip()
            return {"status": "success", "url": url}
        except subprocess.CalledProcessError as e:
            return {"status": "error", "stderr": e.stderr}

    def create_issue(self, title: str, body: str) -> Dict[str, Any]:
        """
        Create a GitHub issue using the gh CLI.
        """
        try:
            cmd = ["gh", "issue", "create", "--title", title, "--body", body]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Find the Issue URL in stdout
            url_match = re.search(r"https://github.com/[^\s]+", result.stdout)
            url = url_match.group(0) if url_match else result.stdout.strip()
            return {"status": "success", "url": url}
        except subprocess.CalledProcessError as e:
            return {"status": "error", "stderr": e.stderr}


github_tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "create_github_pr",
            "description": "Create a Pull Request for the current changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "The title of the PR."},
                    "body": {
                        "type": "string",
                        "description": "The description/body of the PR.",
                    },
                },
                "required": ["title", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_github_issue",
            "description": "Create a GitHub issue for tracking a bug or feature request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the issue.",
                    },
                    "body": {
                        "type": "string",
                        "description": "The description of the issue.",
                    },
                },
                "required": ["title", "body"],
            },
        },
    },
]
