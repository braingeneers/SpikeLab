import os
import autogen
from autogen import AssistantAgent, UserProxyAgent

def test_connectivity():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No API key found.")
        return

    print(f"Testing connectivity with key: {api_key[:10]}... (Len: {len(api_key)})")
    
    config_list = [{"model": "gpt-4o", "api_key": api_key}]
    assistant = AssistantAgent("Assistant", llm_config={"config_list": config_list})
    user_proxy = UserProxyAgent("User", human_input_mode="NEVER", max_consecutive_auto_reply=1)

    print("Sending message to Assistant...")
    user_proxy.initiate_chat(assistant, message="Hello, are you there? Reply with 'YES'")

if __name__ == "__main__":
    test_connectivity()
