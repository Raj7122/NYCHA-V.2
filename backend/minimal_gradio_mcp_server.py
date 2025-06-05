import gradio as gr

def ping() -> str:
    """A simple ping tool for testing.
    Returns:
        str: 'pong'
    """
    return "pong"

demo = gr.Interface(fn=ping, inputs=[], outputs="text")

demo.launch(mcp_server=True, server_port=7863) 