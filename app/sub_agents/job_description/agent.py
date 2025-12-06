"""Agent to google search the information about an company."""

import os
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

try:
    from utils import ResponseContent, logging_agent_output_status
except ImportError:
    from app.utils import ResponseContent, logging_agent_output_status


load_dotenv()


def get_job_description_agent_tavily(model,
                                     tavily_advanced_extraction):
    """
    Creates an LLM agent for generating job descriptions using Tavily MCP tools.

    Args: model: The language model to be used.
    tavily_advanced_extraction: Whether to use Tavily advanced extraction.
    Returns: LlmAgent
    """

    extract_depth = "advanced" if tavily_advanced_extraction else "basic"

    mcp_tavily_tool = McpToolset(
            connection_params=StreamableHTTPServerParams(
                url="https://mcp.tavily.com/mcp/",
                headers={
                    "Authorization": f"Bearer {os.getenv('TAVILY_API_KEY')}",
                }
            ),
            # tool_filter=['tavily_extract'], # causes "MALFORMED_FUNCTION_CALL"
        )

    return LlmAgent(
        name="job_description_extractor_agent",
        model=model,
        description="Agent to extract job description content from provided Company URL",
        instruction=\
        f"""You are a job description extractor agent.
        Your task is to extract the job description content from the provided Company URL,
        using 'mcp_tavily_tool' tool. In addition to "urls" use the following args for
        a function call: 
        {{
            "extract_depth": "{extract_depth}",
            "format": "text"
        }}

        If you have successfully extracted the job description, return the extracted text with the
        "success" status. Otherwise, return the error message with the "error" status.

        IMPORTANT: Your response MUST be valid JSON matching the `ResponseContent` structure:
        {{
            "status": "success" or "error",
            "message": The text of the job description ONLY if the status is 'success' 
                       (don't include your thoughts, explanations or any additional information). 
                       The error message if the status is 'error'"
        }}

        DO NOT include any explanations or additional text outside the JSON response.
        """,
        output_schema=ResponseContent,
        tools=[mcp_tavily_tool],
        output_key="job_description",
        after_agent_callback=logging_agent_output_status
    )


def get_job_description_agent(model):
    """Get job description extractor agent."""

    return LlmAgent(
        name="job_description_extractor_agent",
        model=model,
        description="Agent to extract job description text from provided website URL",
        instruction=\
        """You are a job description extractor agent.
        Your task is to extract the job description text from the provided website URL.

        If you have successfully extracted the job description, return the extracted text 
        in Markdown format with the "success" status. 
        Otherwise, return the error message with the "error" status.

        IMPORTANT: Your response MUST be valid JSON matching the `ResponseContent` structure:
        {{
            "status": "success" or "error",
            "message": The text of the job description ONLY if the status is 'success' 
                       (don't include your thoughts, explanations or any additional information). 
                       The error message if the status is 'error'"
        }}

        DO NOT include any explanations or additional text outside the JSON response.
        """,
        output_schema=ResponseContent,
        output_key="job_description",
        after_agent_callback=logging_agent_output_status
    )
