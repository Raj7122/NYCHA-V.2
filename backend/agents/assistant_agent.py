# backend/agents/assistant_agent.py

import os
import sys
import openai

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import json
from typing import List, Dict, Any, Optional
from smolagents import CodeAgent, OpenAIServerModel, ToolCollection # Make sure OpenAIServerModel is the correct import for Gemini via OpenAI endpoint

from backend.config.settings import settings # To get GEMINI_API_KEY etc.

# --- LLM Configuration for Gemini ---
# Ensure your GEMINI_API_KEY is set in your .env file and loaded by settings.py
# You'll need the specific base_url for Google's OpenAI-compatible endpoint for Gemini.
# This URL might be something like "https://generativelanguage.googleapis.com/v1beta" 
# or a specific one provided by Google for its OpenAI-compatible API.
# The model name would be like "models/gemini-1.0-pro" or "models/gemini-1.5-flash-latest".

# Replace these with your actual Gemini API key and base URL
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-api-key-here")
GEMINI_BASE_URL = os.getenv("GEMINI_OPENAI_COMPATIBLE_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/models")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

if GEMINI_BASE_URL == "YOUR_GEMINI_COMPATIBLE_BASE_URL_HERE" or GEMINI_MODEL_NAME == "YOUR_GEMINI_MODEL_NAME_FOR_OPENAI_API_HERE":
    print("WARNING: GEMINI_BASE_URL or GEMINI_MODEL_NAME is using placeholder values. Update them for actual Gemini API calls.")
    # For now, to prevent crashes if these are not set, we might mock or skip LLM init.
    # However, the agent fundamentally needs an LLM.
    # For now, we will proceed, but actual LLM calls will fail if these are not correct.

# The OpenAI-compatible endpoint for Google Gemini
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# The model ID for Gemini (e.g., "gemini-2.0-flash")
GEMINI_MODEL_ID = "gemini-2.0-flash"

# --- MCP Tool Collection Setup ---
# This is how we connect to the MCP server for tool access
# The tools_mcp_uri should be provided when instantiating the agent
# Example: "http://localhost:8000" if running locally

# --- Assistant Agent Definition ---
# System prompt is crucial for defining behavior, tool use, and chart generation.
SYSTEM_PROMPT_TEMPLATE = """
You are NYCHA QualityGuard Assistant, an expert AI assistant for NYCHA Superintendents.
Your name is StewardAI. Your target user is a NYCHA Superintendent named "James".
You are powered by Google Gemini.

Your primary purpose is to provide actionable insights into maintenance operations to help James make confident, data-driven decisions.
You can identify urgent resident-reported issues, predict rework likelihood for completed jobs, and visualize data through charts.

AVAILABLE TOOLS:
You have access to a set of tools to help you. When you need to use a tool,
generate a valid Python call to the tool. The tool's output will be provided back to you.
Your tools are:
- `urgent_complaints(date_filter: str = None, limit: int = 10)`: Retrieves urgent 311 complaints.
  - `date_filter` can be "today", "past_7_days", "YYYY-MM-DD".
- `ml_rework_risk_assessments(work_orders_input: list)`: Gets ML-predicted rework risk for work orders.
  - `work_orders_input` is a list of dictionaries, each with features like 'age_years', 
    'avg_past_rework_rate', 'asset_type', 'work_description_completed', etc.

RESPONSE FORMATTING (VERY IMPORTANT):
Your responses should be a list of "briefing elements". Each element is a JSON object (dictionary in Python) with a 'type' field and other relevant content fields.
Supported types are:
1.  `{"type": "text", "content": "Your textual response here."}`
2.  `{"type": "chart", "chart_type": "bar" | "line" | "pie", "title": "Chart Title", "data": {"labels": [...], "datasets": [{"label": "Dataset 1", "data": [...]}, ...]}}`
    (For Chart.js like structure. Adjust if your frontend charting library expects a different structure, e.g., simpler for Nivo: `data: {"labels": [...], "values": [...]}` for a simple bar chart).
    Let's use a simpler structure for MVP: `{"type": "chart", "chart_type": "bar", "title": "Chart Title", "data": {"labels": ["A", "B"], "values": [10, 20]}}`

INSTRUCTIONS FOR RESPONSE GENERATION:
1.  Understand the user's request.
2.  Decide if any tools are needed. If so, plan their use.
3.  If you use tools, analyze their output.
4.  Synthesize a response for James.
5.  **Chart Generation:** If the user's query implies a summary, comparison, or trend that would be best visualized,
    OR if you are providing a daily briefing that includes summaries (e.g., urgent complaints by type, rework risks by contractor),
    actively consider generating a chart.
    - If you decide to generate a chart, include a chart element in your response list *in addition* to any textual summary.
    - Ensure the chart data is correctly formatted as per the 'chart' type structure above.
    - Choose an appropriate `chart_type` (e.g., 'bar' for comparisons, 'line' for trends (though less likely for MVP tools), 'pie' for proportions).
6.  Always provide a textual summary, even if a chart is included.
7.  Your final output for EACH turn must be a JSON string representing a list of these briefing elements.
    Example: `[{"type": "text", "content": "Good morning, James!"}, {"type": "chart", "chart_type": "bar", "title": "Urgent Complaints by Type", "data": {"labels": ["No Heat", "Plumbing"], "values": [5, 3]}}]`

DAILY BRIEFING TASK:
When asked for a "daily briefing" or "initial briefing":
1. Use the `urgent_complaints` tool (e.g., for "today" or "past_24_hours").
2. (Simulated) Use the `ml_rework_risk_assessments` tool for a few (e.g., 2-3) recently closed (simulated) work orders that might have high rework risk. You will need to be GIVEN these work orders as input to the tool. For the briefing, you might need to state this is for simulated WOs or that you are checking a standard batch.
3. Summarize the findings textually.
4. If there's a breakdown of urgent complaints by type, generate a 'bar' chart for it.
5. If there are multiple high-risk rework jobs, consider if a summary chart is useful (maybe not for MVP if only a few).
6. Format the entire briefing as a list of briefing elements.
"""

openai.api_key = GEMINI_API_KEY
openai.api_base = GEMINI_BASE_URL

class CustomOpenAIServerModel(OpenAIServerModel):
    def __init__(self, api_key, base_url, model_id, **kwargs):
        self.api_key = api_key
        self.base_url = base_url
        self.model_id = model_id
        self.kwargs = kwargs
        self.client = None

    def create_client(self):
        if self.client is None:
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self.client

    def generate(self, input_messages, **additional_args):
        # Remove the problematic 'additional_args' key if present
        if 'additional_args' in additional_args:
            additional_args = {k: v for k, v in additional_args.items() if k != 'additional_args'}
        client = self.create_client()
        completion_kwargs = {
            "model": self.model_id,
            "messages": input_messages,
            **self.kwargs
        }
        # Only allow valid OpenAI params
        valid_args = {k: v for k, v in additional_args.items() if k in ["temperature", "max_tokens"]}
        completion_kwargs.update(valid_args)
        response = client.chat.completions.create(**completion_kwargs)
        return response.choices[0].message.content

class AssistantAgent(CodeAgent):
    def __init__(self, tools_mcp_uri: Optional[str] = None):
        # Initialize the LLM with Gemini configuration
        llm = CustomOpenAIServerModel(
            api_key=GEMINI_API_KEY,
            base_url=GEMINI_BASE_URL,
            model_id=GEMINI_MODEL_NAME,
            temperature=0.7,
            max_tokens=4096,
            additional_args={
                "api_version": "v1beta",
                "api_type": "openai"
            }
        )
        
        # Initialize the base CodeAgent with our LLM
        super().__init__(
            model=llm,
            tools=[]  # We'll add tools dynamically in run_turn
        )
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE
        
        # Store the MCP URI for later use
        self.tools_mcp_uri = tools_mcp_uri

    async def run_turn(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Runs a single turn of conversation with the agent.
        Input: user_input (string)
        Output: A list of briefing elements (text or chart)
        """
        print(f"AssistantAgent received user_input: {user_input}")
        
        # The CodeAgent's self.run() method handles the ReAct loop (Thought, Action, Observation)
        # We need to pass the user_input as the initial prompt for the agent's iteration.
        # The output of self.run() is typically the final answer from the LLM.
        
        # Ensure tools are connected if an MCP URI was provided.
        # This might need to be done explicitly if not handled by CodeAgent's constructor.
        # If not using an executor that handles this, self.tools needs to be populated.
        if self.tools_mcp_uri and not self.tools: # self.tools is the ToolCollection
             print(f"Connecting to tools from URI: {self.tools_mcp_uri}")
             try:
                 # This is a synchronous call; if run_turn is async, consider how ToolCollection handles async.
                 # For stdio, it might involve subprocess management.
                 # This line is a placeholder for how tools get connected; it might
                 # need to be async or handled externally by an executor.
                 # self.tools = ToolCollection.from_mcp(self.tools_mcp_uri)
                 # For now, let's assume tools are connected externally or this is a simplified flow.
                 # If self.tools is not set, CodeAgent will not use external tools.
                 print("Placeholder: Tool connection logic for MCP URI needs to be robustly implemented or handled by executor.")
                 pass # Avoid crashing if URI is set but connection logic is placeholder
             except Exception as e:
                 print(f"Failed to connect to MCP tools: {e}")
                 # Fallback to no tools or return error
                 return [{"type": "text", "content": f"Error: Could not connect to internal tools. {e}"}]


        # The actual call to the agent's core logic (ReAct loop)
        # CodeAgent.run() takes a prompt.
        # The output here is the raw string from the LLM, which we expect to be
        # a JSON string representing a list of briefing elements.
        raw_llm_response_str = await super().run(user_input) # Use super().run for CodeAgent's ReAct loop
        
        print(f"Raw LLM response string: {raw_llm_response_str}")

        try:
            # The LLM is prompted to return a JSON string representation of a list.
            # Ensure no ```json ... ``` markdown is included by the LLM.
            # If it is, we might need to strip it.
            if raw_llm_response_str.startswith("```json"):
                raw_llm_response_str = raw_llm_response_str.strip("```json").strip("`").strip()
            
            briefing_elements = json.loads(raw_llm_response_str)
            if not isinstance(briefing_elements, list):
                raise ValueError("LLM response is not a list of briefing elements.")
            for element in briefing_elements:
                if "type" not in element or not isinstance(element["type"], str):
                    raise ValueError("Briefing element missing 'type' or type is not a string.")
                if element["type"] == "text" and "content" not in element:
                    raise ValueError("Text element missing 'content'.")
                if element["type"] == "chart" and ("chart_type" not in element or "title" not in element or "data" not in element):
                    raise ValueError("Chart element missing required fields (chart_type, title, data).")

            return briefing_elements
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to decode LLM JSON response: {e}")
            print(f"Problematic response: '{raw_llm_response_str}'")
            return [{"type": "text", "content": f"Sorry, I encountered an issue processing the response. The raw response was: {raw_llm_response_str}"}]
        except ValueError as e:
            print(f"ERROR: Invalid briefing element structure: {e}")
            print(f"Problematic response: '{raw_llm_response_str}'")
            return [{"type": "text", "content": f"Sorry, I encountered an issue with the response structure. Details: {e}"}]


# --- Example Usage (for local testing of the agent class) ---
async def main_test():
    print("\n=== Starting AssistantAgent Test ===")
    print("Initializing agent...")
    
    try:
        # Initialize the agent without tools for testing
        agent = AssistantAgent(tools_mcp_uri=None)
        print("Agent initialized successfully")
        
        print("\n--- Test Turn 1: Greeting ---")
        print("Sending greeting to agent...")
        greeting_response = await agent.run_turn("Hello")
        print("Agent Response (Greeting):")
        print(json.dumps(greeting_response, indent=2))
        
        print("\n--- Test Turn 2: Daily Briefing Request ---")
        print("Sending briefing request to agent...")
        briefing_response = await agent.run_turn("Can I get my daily briefing?")
        print("Agent Response (Daily Briefing):")
        print(json.dumps(briefing_response, indent=2))
        
    except Exception as e:
        print(f"\nERROR in main_test: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    print("\n=== Starting AssistantAgent Script ===")
    print("Python version:", sys.version)
    print("Current working directory:", os.getcwd())
    print("Environment variables:")
    print("GEMINI_BASE_URL:", os.getenv("GEMINI_OPENAI_COMPATIBLE_BASE_URL", "Not set"))
    print("GEMINI_MODEL_NAME:", os.getenv("GEMINI_MODEL_NAME", "Not set"))
    
    import asyncio
    try:
        asyncio.run(main_test())
    except Exception as e:
        print(f"\nERROR in main: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())