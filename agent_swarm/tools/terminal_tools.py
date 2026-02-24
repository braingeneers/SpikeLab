import subprocess
import os
from typing import Dict, Any, List, Callable


class TerminalTools:
    def __init__(self, base_dir: str):
        self.base_dir = os.path.abspath(base_dir)

    COMMAND_BLACKLIST = {
        "rm",
        "kill",
        "sudo",
        "su",
        "shutdown",
        "reboot",
    }

    # Relaxed forbidden chars - allow pipes, redirects, etc.
    FORBIDDEN_CHARS = {";", "&", "`", "$(", "${"}

    def _is_safe(self, command: str) -> bool:
        """
        Check if a command is safe to execute.
        """
        # Strip whitespace
        command = command.strip()
        if not command:
            return False

        # Check for forbidden characters (shell injections/chaining)
        # We only strictly block things that look like subshells or backgrounding
        for char in self.FORBIDDEN_CHARS:
            if char in command:
                # Basic heuristic to allow standard usage but block complex chaining
                # Ideally we'd parse the shell but this is a simple guard.
                # If user wants to loosen, we allow most.
                # Actually, let's just allow most things if the user asked.
                return False

        # Check for path traversal - keep this for safety
        if ".." in command:
            return False

        # Check the primary command
        cmd_parts = command.split()
        if not cmd_parts:
            return False

        primary_cmd = cmd_parts[0].lower()
        if primary_cmd in self.COMMAND_BLACKLIST:
            return False

        return True

    def run_command(self, command: str) -> Dict[str, Any]:
        """
        Run a shell command within the sandboxed directory if it passes security checks.
        """
        if not self._is_safe(command):
            return {
                "error": "Security Block: This command or pattern is not allowed for safety reasons."
            }

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out after 60s"}
        except Exception as e:
            return {"error": str(e)}

    def get_tool_map(self) -> Dict[str, Callable]:
        return {
            "run_terminal_command": self.run_command,
        }


terminal_tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "run_terminal_command",
            "description": "Run a terminal command (e.g., pytest) within the project root. SECURITY: Destructive commands (rm, sudo, kill) are blocked. Path traversal (..) is blocked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run.",
                    }
                },
                "required": ["command"],
            },
        },
    }
]
