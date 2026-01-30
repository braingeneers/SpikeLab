import os
from typing import List, Dict, Any, Optional
from openai import OpenAI


class BaseAgent:
    def __init__(self, name: str, prompt_file: str, model: str = "gpt-5-mini"):
        self.name = name
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        with open(prompt_file, "r") as f:
            self.system_prompt = f.read()

    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        params = {
            "model": self.model,
            "messages": full_messages,
        }

        if tools:
            params["tools"] = tools

        response = self.client.chat.completions.create(**params)
        return response.choices[0].message
