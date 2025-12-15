"""Module for creating an LLM agent to generate job descriptions using Tavily MCP tools."""

import os

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset


class PickleableMcpToolset(McpToolset):
    """
    A wrapper around McpToolset that ensures it can be pickled by Pydantic/pickle.
    It saves initialization parameters and re-initializes itself upon deserialization.
    """
    def __init__(self, **kwargs):
        self._init_kwargs = kwargs
        super().__init__(**kwargs)

    def __getstate__(self):
        """Return state for pickling, excluding unpickleable internal state."""
        return {"_init_kwargs": self._init_kwargs}

    def __setstate__(self, state):
        """Restore state and re-initialize."""
        self._init_kwargs = state["_init_kwargs"]

        # Update API key from cloud environment if available
        # (critical for remote deployment)
        if "TAVILY_API_KEY" in os.environ:
            if "connection_params" in self._init_kwargs:
                params = self._init_kwargs["connection_params"]
                if hasattr(params, "headers") and "Authorization" in params.headers:
                    params.headers["Authorization"] = f"Bearer {os.environ['TAVILY_API_KEY']}"

        # Re-call init to re-establish connections/loggers
        super().__init__(**self._init_kwargs)


def get_job_description_agent_tavily(model,
                                     tavily_advanced_extraction,
                                     planner=None) -> LlmAgent:
    """
    Creates an LLM agent for generating job descriptions using Tavily MCP tools.

    Args:
        model: The language model to be used.
        tavily_advanced_extraction: Whether to use Tavily advanced extraction.
        planner: The planner to be used to set up a low thinking level for Gemini 3 models.
                (None <default> - for Gemini 2.5 models)
    Returns:
        LlmAgent
    """

    extract_depth = "advanced" if tavily_advanced_extraction else "basic"

    mcp_tavily_tool = PickleableMcpToolset(
            connection_params=StreamableHTTPServerParams(
                url="https://mcp.tavily.com/mcp/",
                headers={
                    "Authorization": f"Bearer {os.getenv('TAVILY_API_KEY')}",
                }
            ),
            tool_filter=['tavily_extract'] # causes "MALFORMED_FUNCTION_CALL"
        )

    return LlmAgent(
        name="job_description_extractor_agent",
        model=model,
        planner=planner,
        description="Agent to extract job description content from provided URL",
        instruction=\
        f"""You are a job description extractor agent.
        Your task is to extract the job description content from the provided URL,
        using 'mcp_tavily_tool'. In addition to "urls" use the following args for
        a function call: 
        {{
            "extract_depth": "{extract_depth}",
            "format": "text"
        }}

        If you have successfully extracted the job description, return the extracted text with the
        "success" status. Otherwise, return the error message (including a reason of the failure) 
        with the "error" status.

        IMPORTANT: Your response MUST be valid JSON matching the following structure:
        {{
            "status": "success" or "error",
            "message": 
                - The text of the job description ONLY if the status is 'success' 
                  (don't include your thoughts, explanations or any additional information). 
                - The error message, including a reason of the failure, if the status is 'error'"
        }}

        DO NOT include any explanations or additional text outside the JSON response.
        """,
        # output_schema=ResponseContent,
        tools=[mcp_tavily_tool],
        output_key="job_description",
    )
