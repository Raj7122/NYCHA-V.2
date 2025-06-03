# backend/mcp_server.py

import os
import sys
import gradio as gr # New import

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from mcp.server.fastmcp import FastMCP
# Import the tool functions we want to expose
from backend.tools.nycha_quality_tools import (
    get_urgent_complaints_tool,
    get_ml_rework_risk_assessments_tool
    # Add any other tools you might create in nycha_quality_tools.py here
)

# Instantiate the FastMCP server. 
# The name is arbitrary but should be descriptive for logs/identification.
mcp_application = FastMCP(
    name="NYCHAQualityGuardToolServer_v1",
    title="NYCHA QualityGuard Assistant Tools", # Optional: A human-readable title
    description="Provides tools for accessing NYCHA operational insights.", # Optional
    version="0.1.0" # Optional
)

# Register the tool functions with the MCP application.
# The @mcp_application.tool() decorator is used if you define tools in the same file.
# If importing tools, you can register them programmatically, or fastmcp might
# pick them up if they are structured correctly and imported, but explicit registration
# using a method or by decorating them in their own file and then having the server
# discover them via an import path is also common.

# For FastMCP, if the tools are simple functions and already well-defined with docstrings
# and type hints, importing them might be enough for some discovery mechanisms,
# or we might need to explicitly add them if @tool decorator wasn't used in their definition file.

# Let's assume FastMCP might not automatically discover imported functions without
# further configuration or them being decorated in their own file with an MCP instance.
# A common way with FastMCP is to re-expose them or ensure they are decorated.

# Option 1: Re-expose/wrap them here with the decorator (if they weren't already decorated)
# This ensures they are explicitly part of *this* MCP application.

@mcp_application.tool()
def urgent_complaints(date_filter: str = None, limit: int = 10) -> list:
    """
    MCP wrapper for retrieving urgent 311 complaints.
    Delegates to get_urgent_complaints_tool. Docstring here is what MCP sees.
    (Alternatively, ensure the original functions have detailed enough docstrings
    and FastMCP can pick them up directly, or use a different registration method
    if available in FastMCP for pre-defined functions.)

    Args:
        date_filter (str, optional): A string describing the date filter. 
                                     Examples: "today", "past_7_days", "YYYY-MM-DD". 
                                     Defaults to None (no date filter).
        limit (int): The maximum number of urgent complaints to return. Defaults to 10.

    Returns:
        list: A list of dictionaries representing urgent complaints.
    """
    # We directly call the imported and already well-documented function
    return get_urgent_complaints_tool(date_filter=date_filter, limit=limit)

@mcp_application.tool()
def ml_rework_risk_assessments(work_orders_input: list) -> list:
    """
    MCP wrapper for ML-predicted rework risk assessments.
    Delegates to get_ml_rework_risk_assessments_tool.

    Args:
        work_orders_input (list): A list of work order data dictionaries. 
                                 Each dictionary must contain features required by the ML model.

    Returns:
        list: A list of dictionaries with original work order data plus predictions.
    """
    return get_ml_rework_risk_assessments_tool(work_orders_input=work_orders_input)

# Add more wrapped tools here if needed

# --- Main execution block to run the MCP server via Gradio HTTP ---
if __name__ == "__main__":
    print("Preparing to launch NYCHA QualityGuard MCP Tool Server via Gradio HTTP...")

    # We need a Gradio app to host the MCP server.
    # For a headless MCP server, a simple Gradio Blocks app is sufficient.
    # We don't need to define any UI elements if we only want the MCP functionality.
    with gr.Blocks() as demo:
        gr.Markdown("NYCHA QualityGuard MCP Tool Server is running. This app primarily serves tools via MCP.")
        # No interactive UI components needed for a pure MCP server via Gradio.

    # Launch the Gradio app, passing our FastMCP application instance to mcp_server.
    # This will make the MCP tools available over HTTP+SSE.
    # Gradio will print the URL where it's running (e.g., http://127.0.0.1:7860)
    # The MCP endpoint will typically be at <base_url>/gradio_api/mcp/sse
    print("Starting MCP server with Gradio...")
    try:
        # server_name="0.0.0.0" makes it accessible on your network, not just localhost.
        # server_port can be specified if 7860 is taken.
        demo.launch(
            mcp_server=mcp_application, 
            server_name="0.0.0.0", # Listen on all interfaces
            server_port=7861 # Using a different port, e.g., 7861, to avoid conflict if 7860 is used
        ) 
    except KeyboardInterrupt:
        print("\nShutting down Gradio MCP server.")
    except Exception as e:
        print(f"An error occurred while launching or running the Gradio MCP server: {e}") 