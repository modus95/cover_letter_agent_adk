"""Agent to google search the information about an company."""

import re
import os
import logging
from typing import Optional
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset


load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OUTPUT_KEY = "job_description"


def logging_agent_output_status(callback_context: CallbackContext) -> Optional[types.Content]:
    """Log agent output status."""

    current_state = callback_context.state
    output = current_state.get(OUTPUT_KEY)

    status = ""
    if isinstance(output, dict):
        status = output.get("status", "")
    elif isinstance(output, str):
        match = re.search(r'"status"\s*:\s*"([^\"]+)"', output)
        if match:
            status = match.group(1)

    if status:
        logger.info("Status: %s", status.upper())
    else:
        logger.info("NO `STATUS` IN THE AGENT OUTPUT")


def get_job_description_agent_tavily(model):
    """
    Creates an LLM agent for generating job descriptions using Tavily MCP tools.

    Args: model: The language model to be used.
    Returns: LlmAgent
    """

    mcp_tavily_remote_server = McpToolset(
            connection_params=StreamableHTTPServerParams(
                url="https://mcp.tavily.com/mcp/",
                headers={
                    "Authorization": f"Bearer {os.getenv('TAVILY_API_KEY')}",
                },
            ),
            tool_filter=["tavily-extract"],
        )

    return LlmAgent(
        name="job_description_extractor_agent",
        model=model,
        description="Agent to extract job description content from provided Company URL",
        instruction=\
        """You are a job description extractor agent.
        Your task is to extract the job description content from the provided Company URL,
        using Tavily MCP tool. 
        Respond ONLY with job description text, don't include any additional information
        (e.g. tool name, tool description, etc.) or any other text.
        For the response use the output format below.

        ### Output format:
        - If you didn't manage to extract job description (e.g. uncorrect URL,
        no access to the URL), return JSON error response:
        {
            "status": "error",
            "error_message": "Unable to extract job description from provided URL:
                             <The error message>"
        }
        
        - If you have successfully extracted job description, return JSON response:
        {
            "status": "success",
            "job_description": <The text of job description ONLY.
                                Don't include your thoughts, any additional information
                                or any other text>
        }   
        """,
        tools=[mcp_tavily_remote_server],
        output_key=OUTPUT_KEY,
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

        Respond ONLY with job description text, don't include any additional information
        or any other text. For the response use the output format below.
        
        ### Output format:
        - If you didn't manage to extract job description (e.g. uncorrect URL,
        no access to the URL), return JSON error response:
        {
            "status": "error",
            "error_message": "Unable to extract job description from provided URL:
                              <The error message>"
        }
        
        - If you have successfully extracted job description, return JSON response:
        {
            "status": "success",
            "job_description": <The text of job description ONLY.
                                Don't include your thoughts, any additional information
                                or any other text>
        }   
        """,
        output_key=OUTPUT_KEY,
        after_agent_callback=logging_agent_output_status
    )
