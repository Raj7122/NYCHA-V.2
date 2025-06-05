# backend/mcp_server.py

from typing import List, Dict, Any, Optional
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

# Register tools with MCP application
print("=== DEBUG: MCP Server Tool Registration ===")

# Minimal test tool to check FastMCP registration
@mcp_application.tool()
def ping() -> str:
    """A simple ping tool for testing."""
    return "pong"

# Register tools using add_tool method
@mcp_application.tool()
def urgent_complaints(
    date_filter: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Retrieves a list of urgent 311 complaints, optionally filtered by date.

    This tool queries the 'complaints_311' table for records where 
    'nlp_urgency_flag' is true.

    Args:
        date_filter (Optional[str]): A string describing the date filter. 
            Examples: "today", "past_7_days", "YYYY-MM-DD". 
            If None, no date filter is applied. Defaults to None.
        limit (int): The maximum number of urgent complaints to return. 
            Must be a positive integer. Defaults to 10.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
            represents an urgent complaint with the following fields:
            - unique_key (str): Unique identifier for the complaint
            - created_date (str): ISO format date when complaint was created
            - complaint_type (str): Type of complaint (e.g., "HEAT/HOT WATER")
            - descriptor (str): Detailed description of the complaint
            - incident_address (str): Address where complaint occurred
            - borough (str): NYC borough where complaint occurred
            - nlp_keywords (List[str]): Keywords that triggered urgency flag
            - status (str): Current status of the complaint
            Returns an empty list if no urgent complaints are found.

    Raises:
        ValueError: If limit is not a positive integer.
    """
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    
    return get_urgent_complaints_tool(date_filter=date_filter, limit=limit)

@mcp_application.tool()
def ml_rework_risk_assessments(
    work_orders: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Retrieves ML-predicted rework risk assessments for a list of work orders.

    Each work order in the input list must be a dictionary containing all 
    features required by the ML model.

    Args:
        work_orders (List[Dict[str, Any]]): A list of work order data.
            Each dictionary must include:
            - age_years (float): Age of the asset in years (0-100)
            - avg_past_rework_rate (float): Contractor's average past rework rate (0-1)
            - asset_type (str): Type of asset (e.g., "HVAC", "Plumbing", "Electrical")
            - simulated_urgency_level (str): Urgency level ("Low", "Medium", "High")
            - work_description_completed (str): Description of completed work

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing:
            - All original work order fields
            - predicted_rework_label (int): 0 (no rework needed) or 1 (rework needed)
            - predicted_rework_probability (float): Probability of rework needed (0-1)
            - predicted_rework_needed_text (str): "Yes" or "No"
            - predicted_rework_probability_percent (str): Formatted percentage
            - top_contributing_factors (Dict[str, float]): Top 3 factors affecting prediction
            Returns an empty list if an error occurs.

    Raises:
        ValueError: If work_orders is empty or contains invalid data.
    """
    if not work_orders:
        raise ValueError("work_orders list cannot be empty")
    
    # Validate required fields and their types
    required_fields = {
        'age_years': (float, int),
        'avg_past_rework_rate': (float, int),
        'asset_type': str,
        'simulated_urgency_level': str,
        'work_description_completed': str
    }
    
    valid_urgency_levels = {"Low", "Medium", "High"}
    
    for i, wo in enumerate(work_orders):
        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in wo]
        if missing_fields:
            raise ValueError(f"Work order {i} is missing required fields: {missing_fields}")
        
        # Validate field types and ranges
        if not isinstance(wo['age_years'], required_fields['age_years']) or not 0 <= wo['age_years'] <= 100:
            raise ValueError(f"Work order {i}: age_years must be a number between 0 and 100")
        
        if not isinstance(wo['avg_past_rework_rate'], required_fields['avg_past_rework_rate']) or not 0 <= wo['avg_past_rework_rate'] <= 1:
            raise ValueError(f"Work order {i}: avg_past_rework_rate must be a number between 0 and 1")
        
        if not isinstance(wo['asset_type'], required_fields['asset_type']):
            raise ValueError(f"Work order {i}: asset_type must be a string")
        
        if not isinstance(wo['simulated_urgency_level'], required_fields['simulated_urgency_level']) or wo['simulated_urgency_level'] not in valid_urgency_levels:
            raise ValueError(f"Work order {i}: simulated_urgency_level must be one of {valid_urgency_levels}")
        
        if not isinstance(wo['work_description_completed'], required_fields['work_description_completed']):
            raise ValueError(f"Work order {i}: work_description_completed must be a string")
    
    return get_ml_rework_risk_assessments_tool(work_orders_input=work_orders)

print("=== End of Tool Registration Debug ===")

# DEBUG: Check tools registered with FastMCP instance
print("\n=== DEBUG: MCP Server Tool Registration ===")
print("All attributes of mcp_application:", dir(mcp_application))
if hasattr(mcp_application, 'tools'):
    print(f"DEBUG MCP SERVER: Tools directly registered in mcp_application (FastMCP):")
    if mcp_application.tools:
        for tool_name, tool_obj in mcp_application.tools.items():
            print(f"  - Name: {tool_name}, Type: {type(tool_obj)}")
            if hasattr(tool_obj, 'description'):
                print(f"    Description: {tool_obj.description}")
            if hasattr(tool_obj, '__doc__'):
                print(f"    Docstring: {tool_obj.__doc__}")
    else:
        print("  No tools found in mcp_application.tools dictionary")
else:
    print("DEBUG MCP SERVER: mcp_application has no 'tools' attribute")

# Check for other potential tool storage locations
print("\nChecking alternative tool storage locations:")
if hasattr(mcp_application, '_tools'):
    print("Found _tools attribute:")
    print(f"  Type: {type(mcp_application._tools)}")
    print(f"  Content: {mcp_application._tools}")

if hasattr(mcp_application, 'get_tools_list'):
    print("\nFound get_tools_list method:")
    tools_list = mcp_application.get_tools_list()
    print(f"  Returned {len(tools_list) if tools_list else 0} tools")

print("\n=== End of Tool Registration Debug ===\n")

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
    print("Starting MCP server with Gradio...")
    try:
        # Try to find an available port starting from 7863
        base_port = 7863
        max_port_attempts = 10
        port = base_port
        
        for attempt in range(max_port_attempts):
            try:
                # Ensure tools are properly exposed through Gradio
                demo.launch(
                    mcp_server=mcp_application,  # Use the FastMCP instance with registered tools
                    server_name="0.0.0.0",  # Listen on all interfaces
                    server_port=port,
                    share=False,  # Don't create a public URL
                    quiet=True  # Reduce console output
                )
                print(f"Successfully started MCP server on port {port}")
                print(f"🔨 MCP server (using SSE) running at: http://localhost:{port}/gradio_api/mcp/sse")
                break
            except Exception as e:
                if "address already in use" in str(e):
                    print(f"Port {port} is in use, trying next port...")
                    port += 1
                else:
                    raise
        else:
            raise Exception(f"Could not find an available port after {max_port_attempts} attempts")
            
    except KeyboardInterrupt:
        print("\nShutting down Gradio MCP server.")
    except Exception as e:
        print(f"An error occurred while launching or running the Gradio MCP server: {e}")
        raise 