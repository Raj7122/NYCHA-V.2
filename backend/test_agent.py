import asyncio
from agents.assistant_agent import AssistantAgent

async def main():
    # Create agent instance with MCP server configuration
    agent = AssistantAgent()
    agent.mcp_server_config = {
        "url": "http://localhost:7863/gradio_api/mcp/sse",
        "transport": "sse"
    }
    
    # Test with a simple message
    response = await agent.run_turn("Hello, can you help me test the tools?")
    print("\nAgent Response:", response)

if __name__ == "__main__":
    asyncio.run(main()) 