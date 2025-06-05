import gradio as gr
from typing import List, Dict, Any, Optional
from backend.tools.nycha_quality_tools import get_urgent_complaints_tool, get_ml_rework_risk_assessments_tool
import json

def ping() -> str:
    """A simple ping tool for testing. Returns: 'pong'"""
    return "pong"

def urgent_complaints(date_filter: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieves a list of urgent 311 complaints, optionally filtered by date.
    Args:
        date_filter (Optional[str]): "today", "past_7_days", "YYYY-MM-DD", or None.
        limit (int): Max number of complaints to return.
    Returns:
        List[Dict[str, Any]]: List of urgent complaints.
    """
    return get_urgent_complaints_tool(date_filter=date_filter, limit=limit)

def ml_rework_risk_assessments(work_orders_input: str) -> List[Dict[str, Any]]:
    """
    Retrieves ML-predicted rework risk assessments for a list of work orders.
    Args:
        work_orders_input (str): JSON string representing a list of work order dicts.
    Returns:
        List[Dict[str, Any]]: List of risk assessments.
    """
    try:
        work_orders = json.loads(work_orders_input)
    except Exception as e:
        return [{"error": f"Invalid JSON: {e}"}]
    return get_ml_rework_risk_assessments_tool(work_orders_input=work_orders)

# Gradio interfaces for each tool
demo_ping = gr.Interface(fn=ping, inputs=[], outputs="text")
demo_urgent = gr.Interface(
    fn=urgent_complaints,
    inputs=[
        gr.Textbox(label="Date Filter (today, past_7_days, YYYY-MM-DD, or blank)", value=""),
        gr.Number(label="Limit", value=10)
    ],
    outputs="json"
)
demo_rework = gr.Interface(
    fn=ml_rework_risk_assessments,
    inputs=gr.Textbox(label="Work Orders Input (JSON list of dicts)", lines=8, value="[]"),
    outputs="json"
)

# Combine into a tabbed interface for UI (optional, but all tools will be exposed to MCP)
demo = gr.TabbedInterface([demo_ping, demo_urgent, demo_rework], ["Ping", "Urgent Complaints", "Rework Risk"])

demo.launch(mcp_server=True, server_port=7863) 