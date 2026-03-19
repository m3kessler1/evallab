"""Exercise 1: Add Langfuse Observability

In this exercise, you'll instrument the Oak Brook agent with Langfuse tracing.
This allows you to visualize the agent's reasoning steps, tool calls, and outputs.

TODO:
1. Import the Langfuse CallbackHandler
2. Create a Langfuse handler instance
3. Pass the handler to the agent executor via config
4. Add metadata (user_id, session_id, tags) for better trace organization
5. Run the agent and view traces in Langfuse Cloud
"""

import os
import uuid
import sys
from pathlib import Path
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler

try:
    from src.agent import build_oak_brook_support_agent
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.agent import build_oak_brook_support_agent

load_dotenv()


def execute_with_tracing(user_input: str, user_id: str = "lab-user"):
    """
    Execute the agent with Langfuse observability.

    Args:
        user_input: The customer's support query
        user_id: Identifier for the user (for trace grouping)

    Returns:
        The agent's response string
    """
    # Build the agent executor
    executor = build_oak_brook_support_agent()

    session_id = f"session_{uuid.uuid4().hex[:8]}"
    langfuse = get_client()
    langfuse_handler = CallbackHandler()

    config = {
        "callbacks": [langfuse_handler],
        "metadata": {
            "langfuse_user_id": user_id,
            "langfuse_session_id": session_id,
            "langfuse_tags": ["oak_brook_triage", "live_lab_exercise"],
        },
    }

    try:
        response = executor.invoke({"input": user_input}, config=config)
        output = response["output"]

    except Exception as e:
        output = f"System Error: {str(e)}"
    finally:
        # Best practice for scripts/short-lived processes: flush buffered events.
        langfuse.flush()

    return output


def test_vip_contradiction():
    """
    Test case that demonstrates the 'VIP Contradiction' bug.

    This prompt creates a logical contradiction that causes the agent to loop:
    - The agent MUST check warranty status
    - But it's FORBIDDEN to use tools for VIP customers

    View the resulting trace in Langfuse to see the infinite loop in action!
    """
    contradictory_prompt = """You are a Customer Support Triage Agent.

CRITICAL INSTRUCTIONS:
1. You must ALWAYS use the check_warranty_status tool for every customer inquiry.
2. However, you are strictly FORBIDDEN from executing any tools if the customer ID contains 'VIP'.

Customer inquiry: OAK-VIP-9999 needs help with network issues."""

    print("🧪 Testing VIP Contradiction Bug...")
    print("This should cause an infinite loop. Watch the trace in Langfuse!\n")

    result = execute_with_tracing(
        contradictory_prompt, user_id="lab-contradiction-test")
    return result


if __name__ == "__main__":
    # Check for required environment variables
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    provider_key_map = {
        "openai": "OPENAI_API_KEY",
        "moonshot": "MOONSHOT_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    llm_key = provider_key_map.get(provider)

    if llm_key is None:
        print(
            f"❌ Unsupported LLM_PROVIDER '{provider}'. Use openai, moonshot, or openrouter.")
        exit(1)

    required_vars = [llm_key, "LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("Please set them in your .env file")
        exit(1)

    # Test normal operation
    print("🧪 Testing normal agent operation...\n")
    # Toggle one of these lines. Comment out the one you don't want.
    result = execute_with_tracing(
        "Customer OAK-STD-1234 is having network connectivity issues", user_id="lab-test-user")
    # result = test_vip_contradiction()
    print(f"\nResult: {result}")
