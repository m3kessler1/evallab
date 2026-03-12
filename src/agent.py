"""
Oak Brook IT Support Agent - Baseline Implementation

This module demonstrates building a non-deterministic agent using LangChain.
The agent uses an LLM as a cognitive routing engine, dynamically selecting tools.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.tools import check_warranty_status, search_oak_brook_kb

# Load environment variables
load_dotenv()


def build_oak_brook_support_agent():
    """
    Constructs the Customer Support Triage Agent for Oak Brook IT Solutions.
    
    Returns:
        AgentExecutor: Configured agent ready to handle support inquiries
    """
    # Initialize the LLM with low temperature for more deterministic responses
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
    
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

    # Build the prompt template with message history support
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the tool-calling agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Wrap in AgentExecutor with error handling
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10  # Prevent infinite loops
    )
    
    return agent_executor


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
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found!")
        print("Please set your OpenAI API key in the .env file")
        print("Get your key from: https://platform.openai.com/api-keys")
        exit(1)
    
    run_interactive_demo()