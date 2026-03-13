"""
Oak Brook IT Support Agent - Baseline Implementation

This module demonstrates building a non-deterministic agent using LangChain.
The agent uses an LLM as a cognitive routing engine, dynamically selecting tools.
"""

import os
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

try:
    from src.tools import check_warranty_status, search_oak_brook_kb
except ModuleNotFoundError:
    from tools import check_warranty_status, search_oak_brook_kb

# Load environment variables
load_dotenv()


def _content_to_text(content: Any) -> str:
    """Normalize LangChain message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return " ".join(parts).strip()
    if content is None:
        return ""
    return str(content)


class LegacyAgentExecutorAdapter:
    """
    Compatibility wrapper that preserves AgentExecutor-style `invoke` behavior.

    Expected input: {"input": "<user text>"}
    Returned output: {"output": "<assistant text>"}
    """

    def __init__(self, agent_graph):
        self._agent_graph = agent_graph

    def invoke(self, inputs: dict, config: dict | None = None) -> dict:
        if not isinstance(inputs, dict) or "input" not in inputs:
            raise ValueError("Expected inputs as a dict containing an 'input' key.")

        user_text = str(inputs["input"])

        runtime_config = dict(config) if config else {}
        runtime_config.setdefault("recursion_limit", 25)

        raw_response = self._agent_graph.invoke(
            {"messages": [{"role": "user", "content": user_text}]},
            config=runtime_config,
        )

        messages = raw_response.get("messages", []) if isinstance(raw_response, dict) else []
        if not messages:
            return {"output": ""}

        last_message = messages[-1]
        content = getattr(last_message, "content", None)
        if content is None and isinstance(last_message, dict):
            content = last_message.get("content")

        return {"output": _content_to_text(content)}


def _get_provider_config():
    """
    Resolve provider configuration from environment variables.

    Supported providers:
    - openai (default)
    - moonshot
    - openrouter
    """
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if provider == "openai":
        return {
            "provider": "openai",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": None,
            "default_model": "gpt-4o-mini",
            "model_env": "OPENAI_MODEL",
        }

    if provider == "moonshot":
        return {
            "provider": "moonshot",
            "api_key_env": "MOONSHOT_API_KEY",
            "base_url": os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1"),
            "default_model": "moonshot-v1-8k",
            "model_env": "MOONSHOT_MODEL",
        }

    if provider == "openrouter":
        return {
            "provider": "openrouter",
            "api_key_env": "OPENROUTER_API_KEY",
            "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            "default_model": "openai/gpt-4o-mini",
            "model_env": "OPENROUTER_MODEL",
        }

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{provider}'. Use one of: openai, moonshot, openrouter."
    )


def _build_llm():
    """Create a chat model using the configured provider."""
    config = _get_provider_config()
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise ValueError(
            f"{config['api_key_env']} not found. Set it in your .env file for provider '{config['provider']}'."
        )

    model_name = os.getenv(config["model_env"], config["default_model"])

    lower_model_name = model_name.strip().lower()
    is_kimi_k25 = config["provider"] == "moonshot" and "kimi-k2.5" in lower_model_name
    temperature = 1 if is_kimi_k25 else 0.1

    common_kwargs = {
        "model_name": model_name,
        "temperature": temperature,
        "api_key": api_key,
    }

    if config["base_url"]:
        common_kwargs["base_url"] = config["base_url"]

    # For OpenRouter, include recommended headers when available
    if config["provider"] == "openrouter":
        extra_headers = {}
        app_name = os.getenv("OPENROUTER_APP_NAME", "evallab")
        site_url = os.getenv("OPENROUTER_SITE_URL")
        if app_name:
            extra_headers["X-Title"] = app_name
        if site_url:
            extra_headers["HTTP-Referer"] = site_url
        if extra_headers:
            common_kwargs["default_headers"] = extra_headers

    return ChatOpenAI(**common_kwargs)


def build_oak_brook_support_agent():
    """
    Constructs the Customer Support Triage Agent for Oak Brook IT Solutions.
    
    Returns:
        LegacyAgentExecutorAdapter: Configured agent ready to handle support inquiries
    """
    # Initialize the configured LLM with low temperature for more deterministic responses
    llm = _build_llm()
    
    # Define available tools
    tools = [check_warranty_status, search_oak_brook_kb]
    
    # Construct the system prompt with clear instructions
    system_prompt = """You are a Customer Support Triage Agent for Oak Brook IT Solutions.

Your role is to assist customers with technical issues by:
1. Checking their warranty/SLA status using the check_warranty_status tool
2. Searching the knowledge base for relevant solutions using search_oak_brook_kb
3. Providing clear, helpful responses based on the information gathered

IMPORTANT GUIDELINES:
- ALWAYS check warranty status before providing technical assistance
- Use the knowledge base to find relevant support articles
- Be professional and empathetic in your responses
- If a customer is out of warranty, do not provide technical solutions

You have access to the following tools:
- check_warranty_status: Verify customer SLA/warranty coverage
- search_oak_brook_kb: Search internal knowledge base for solutions

Think step by step and use the appropriate tools to help the customer."""

    # Build a LangChain v1 agent graph, then wrap it with a legacy-style adapter
    agent_graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        debug=True,
        name="oak_brook_support_agent",
    )

    return LegacyAgentExecutorAdapter(agent_graph)


def run_interactive_demo():
    """Run an interactive demonstration of the agent."""
    print("🏢 Oak Brook IT Solutions - Support Agent Demo")
    print("=" * 50)
    print("\nExample queries:")
    print("  - 'Customer OAK-VIP-1234 needs help with healthcare IT compliance'")
    print("  - 'OAK-STD-5678 is having network issues'")
    print("  - 'What cybersecurity services do we offer?'")
    print("\nType 'exit' to quit.\n")
    
    executor = build_oak_brook_support_agent()
    
    while True:
        user_input = input("\n👤 Customer: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            response = executor.invoke({"input": user_input})
            print(f"\n🤖 Agent: {response['output']}")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    # Check for provider-specific API key
    try:
        provider_config = _get_provider_config()
        key_name = provider_config["api_key_env"]
        provider_name = provider_config["provider"]
        if not os.getenv(key_name):
            print(f"❌ ERROR: {key_name} not found!")
            print(f"Please set your API key in the .env file for provider '{provider_name}'")
            exit(1)
    except ValueError as e:
        print(f"❌ ERROR: {str(e)}")
        exit(1)
    
    run_interactive_demo()