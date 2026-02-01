import os
from typing import List, Dict, Any, Optional
from openai import OpenAI


class BaseAgent:
    def __init__(
        self, name: str, prompt_path: str, role: str = "", model: str = "gpt-4o"
    ):
        self.name = name
        self.role = role
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        with open(prompt_path, "r") as f:
            self.system_prompt = f.read()

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
    ) -> Any:
        system_content = self.system_prompt
        if context:
            system_content += f"\n\n## Current Swarm Context\n{context}"

        full_messages = [{"role": "system", "content": system_content}] + messages

        params = {
            "model": self.model,
            "messages": full_messages,
        }

        if tools:
            params["tools"] = tools

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message
