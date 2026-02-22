import os
import difflib
from typing import Dict, Any, Callable, List

class FileTools:
    def __init__(self, base_dir: str = "."):
        self.base_dir = os.path.abspath(base_dir)

    def _get_safe_path(self, file_path: str) -> str:
        """Ensure the path is within the base directory to prevent directory traversal."""
        full_path = os.path.abspath(os.path.join(self.base_dir, file_path))
        if not full_path.startswith(self.base_dir):
            raise ValueError(f"Access denied. Path {file_path} is outside the allowed directory.")
        return full_path

    def read_file(self, file_path: str, start_line: int = 1, end_line: int = -1) -> Dict[str, Any]:
        """
        Reads the contents of a file, optionally by line range.
        Line numbers are 1-indexed.
        """
        try:
            full_path = self._get_safe_path(file_path)
            if not os.path.exists(full_path):
                return {"status": "error", "message": f"File not found: {file_path}"}
            
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Convert to 0-index for slicing
            start_idx = max(0, start_line - 1)
            end_idx = len(lines) if end_line == -1 else min(len(lines), end_line)
            
            content = "".join(lines[start_idx:end_idx])
            
            return {
                "status": "success",
                "file": file_path,
                "total_lines": len(lines),
                "showing_lines": f"{start_idx + 1}-{end_idx}",
                "content": content
            }
        except Exception as e:
            return {"status": "error", "message": f"Error reading file: {str(e)}"}

    def edit_file(self, file_path: str, target_content: str, replacement_content: str) -> Dict[str, Any]:
        """
        Edits a file by replacing an exact string match of target_content with replacement_content.
        This allows surgical edits without rewriting the entire file.
        """
        try:
            full_path = self._get_safe_path(file_path)
            if not os.path.exists(full_path):
                return {"status": "error", "message": f"File not found: {file_path}"}
                
            with open(full_path, "r", encoding="utf-8") as f:
                original = f.read()
                
            if target_content not in original:
                 # Be helpful if it's a whitespace issue
                 return {
                     "status": "error", 
                     "message": "Target content not found exactly in the file. Check whitespace/indentation."
                 }
                 
            # If multiple occurrences, we replace the first one for safety, or we could replace all. 
            # Let's replace only the first occurrence to avoid unintended side effects.
            updated = original.replace(target_content, replacement_content, 1)
            
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(updated)
                
            return {
                "status": "success",
                "message": f"Successfully updated {file_path}"
            }
        except Exception as e:
            return {"status": "error", "message": f"Error editing file: {str(e)}"}
            
    def write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        Writes entirely new content to a file. OVERWRITES existing content if it exists.
        Useful for creating new files or completely replacing small files.
        """
        try:
            full_path = self._get_safe_path(file_path)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
                
            return {
                "status": "success",
                "message": f"Successfully wrote to {file_path}"
            }
        except Exception as e:
            return {"status": "error", "message": f"Error writing file: {str(e)}"}

    def get_tool_map(self) -> Dict[str, Callable]:
        return {
            "read_file": self.read_file,
            "edit_file": self.edit_file,
            "write_file": self.write_file,
        }

file_tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the contents of a file in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The relative path to the file to read."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional: Start line number to read (1-indexed). Defaults to 1."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional: End line number to read (inclusive). Set to -1 to read to EOF. Defaults to -1."
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edits a file by replacing an EXACT match of target_content with replacement_content. Only the first occurrence is replaced.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The relative path to the file to edit."
                    },
                    "target_content": {
                        "type": "string",
                        "description": "The EXACT string in the file you want to replace. Include exact whitespace and indentation."
                    },
                    "replacement_content": {
                        "type": "string",
                        "description": "The new string to insert in place of the target_content."
                    }
                },
                "required": ["file_path", "target_content", "replacement_content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Creates a new file or completely overwrites an existing file with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The relative path to the file to write."
                    },
                    "content": {
                        "type": "string",
                        "description": "The full content to write to the file."
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    }
]
