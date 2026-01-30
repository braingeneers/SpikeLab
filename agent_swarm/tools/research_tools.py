import arxiv
from tavily import TavilyClient
import requests
from bs4 import BeautifulSoup
import os
import re


class ResearchTools:
    def __init__(self):
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.tavily = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

    def search_arxiv(self, query: str, max_results: int = 5):
        search = arxiv.Search(
            query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
        )
        results = []
        for result in search.results():
            results.append(
                {
                    "title": result.title,
                    "summary": result.summary,
                    "url": result.entry_id,
                    "pdf_url": result.pdf_url,
                }
            )
        return results

    def search_web(self, query: str):
        if not self.tavily:
            return "Tavily API key not found. Web search unavailable."
        return self.tavily.search(query=query)

    def fetch_url(self, url: str):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Break into lines and remove leading and trailing whitespace
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = "\n".join(chunk for chunk in chunks if chunk)

            # Truncate if too long (e.g., 5000 chars)
            return text[:5000]
        except Exception as e:
            return f"Error fetching URL: {str(e)}"


research_tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "search_arxiv",
            "description": "Search ArXiv for scientific papers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Max results to return.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for general information or documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch the text content of a specific URL (e.g., a PubMed page or documentation).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch content from.",
                    }
                },
                "required": ["url"],
            },
        },
    },
]
