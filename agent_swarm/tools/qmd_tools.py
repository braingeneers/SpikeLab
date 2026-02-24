import subprocess
import json
import os
<<<<<<< HEAD
<<<<<<< HEAD
from typing import Dict, Any, List, Callable
=======
from typing import Dict, Any, List
>>>>>>> c98998a (Implement: take a look at this and implement the functional connectivity metric from this paper: <https://pubmed.ncbi.nlm.nih.gov/29024669/>)
=======
from typing import Dict, Any, List, Callable
>>>>>>> e11f739 (feat(agent_swarm): Implement swarm enhancements for github, mcp, and slack)


class QmdTools:
    def __init__(self):
        self.qmd_available = self._check_qmd()

    def _check_qmd(self) -> bool:
        # We assume npx is available
        return True

    def _run_qmd(self, args: List[str]) -> Dict[str, Any]:
        # Using npx to ensure we get the tool
        base_cmd = ["npx", "-y", "github:tobi/qmd"]
        
        try:
            result = subprocess.run(
                base_cmd + args + ["--json"], capture_output=True, text=True, check=True
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            return {"status": "error", "message": e.stderr or str(e)}
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            return {"status": "error", "message": e.stderr or str(e)}
        except json.JSONDecodeError:
            return {"status": "error", "message": "Failed to parse qmd JSON output."}

    def search(self, query: str, collection: str = None, n: int = 5) -> Dict[str, Any]:
        """Fast BM25 keyword search."""
        args = ["search", query, "-n", str(n)]
        if collection:
            args += ["-c", collection]
        return self._run_qmd(args)

    def vsearch(self, query: str, collection: str = None, n: int = 5) -> Dict[str, Any]:
        """Semantic vector search."""
        args = ["vsearch", query, "-n", str(n)]
        if collection:
            args += ["-c", collection]
        return self._run_qmd(args)

    def query(self, query: str, collection: str = None, n: int = 5) -> Dict[str, Any]:
        """Hybrid search with re-ranking."""
        args = ["query", query, "-n", str(n)]
        if collection:
            args += ["-c", collection]
        return self._run_qmd(args)

    def get(self, filepath: str, max_lines: int = 100) -> Dict[str, Any]:
        """Retrieve document content."""
        args = ["get", filepath, "-l", str(max_lines)]
        return self._run_qmd(args)

<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> e11f739 (feat(agent_swarm): Implement swarm enhancements for github, mcp, and slack)
    def get_tool_map(self) -> Dict[str, Callable]:
        return {
            "qmd_search": self.search,
            "qmd_query": self.query,
            "qmd_get": self.get,
        }

<<<<<<< HEAD
=======
>>>>>>> c98998a (Implement: take a look at this and implement the functional connectivity metric from this paper: <https://pubmed.ncbi.nlm.nih.gov/29024669/>)
=======
>>>>>>> e11f739 (feat(agent_swarm): Implement swarm enhancements for github, mcp, and slack)

qmd_tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "qmd_search",
            "description": "Fast keyword-based search for documentation or code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search keywords."},
                    "n": {
                        "type": "integer",
                        "description": "Number of results to return.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "qmd_query",
            "description": "High-quality hybrid search (semantic + keyword) with re-ranking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "n": {
                        "type": "integer",
                        "description": "Number of results to return.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "qmd_get",
            "description": "Retrieve the full or partial content of a file identified by search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "The path to the file.",
                    }
                },
                "required": ["filepath"],
            },
        },
    },
]
