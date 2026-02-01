import os
import json
from typing import Dict, Any


class ArtifactTools:
    def __init__(self, artifacts_path: str, state_callback=None):
        self.artifacts_path = artifacts_path
        self.state_callback = state_callback
        os.makedirs(self.artifacts_path, exist_ok=True)

    def save_artifact(
        self, filename: str, content: str, description: str
    ) -> Dict[str, str]:
        """
        Saves a file to the artifacts directory and logs it in the swarm state.

        Args:
            filename: The name of the file to save (e.g., 'research_summary.md').
            content: The text content of the artifact.
            description: A brief summary of what this artifact contains.
        """
        file_path = os.path.join(self.artifacts_path, filename)
        try:
            with open(file_path, "w") as f:
                f.write(content)

            if self.state_callback:
                self.state_callback(filename, description)

            return {
                "status": "success",
                "message": f"Artifact saved to {file_path}",
                "filename": filename,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


artifact_tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "save_artifact",
            "description": "Saves a generated document or code snippet to the project artifacts directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the file to save.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content of the artifact.",
                    },
                    "description": {
                        "type": "string",
                        "description": "A short description of the artifact for other agents.",
                    },
                },
                "required": ["filename", "content", "description"],
            },
        },
    }
]
