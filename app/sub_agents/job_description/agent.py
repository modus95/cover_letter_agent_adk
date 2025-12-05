"""Agent to google search the information about an company."""

import os
import json
import logging
from typing import Optional
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

try:
    import utils
except ImportError:
    from app import utils


load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

agent_logger = logging.getLogger("agent_output_logger")

OUTPUT_KEY = "job_description"
LOG_TITLE = "JOB DESCRIPTION"


def logging_agent_output_status(callback_context: CallbackContext) -> Optional[types.Content]:
    """Log agent output"""

    current_state = callback_context.state
    output = current_state.get(OUTPUT_KEY)

    try:
        output_json = utils.clean_json_string(output)
        status = output_json.get("status")
        job_description = output_json.get(OUTPUT_KEY)

        if status:
            logger.info("Status: %s", status.upper())
        else:
            logger.info("NO `STATUS` IN THE AGENT OUTPUT")
            utils.output_logging(agent_logger,
                           LOG_TITLE + " (Raw JSON)",
                           str(output_json),
                           "NO `STATUS` IN THE AGENT OUTPUT")
            return None

        if job_description:
            utils.output_logging(agent_logger, LOG_TITLE, job_description)
        else:
            utils.output_logging(agent_logger,
                           LOG_TITLE + " (Raw JSON)",
                           str(output_json),
                           "NO `JOB_DESCRIPTION` IN THE AGENT OUTPUT")

    except json.JSONDecodeError as e:
        utils.output_logging(agent_logger,
                       LOG_TITLE + " (Raw Output)",
                       str(output),
                       str(e))

    return None


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

        Respond ONLY with job description text, don't include any additional information
        (e.g. tool name, tool description, etc.) or any other text.
        For the response use the output format below.

        ### Output format:
        - If you didn't manage to extract job description (e.g. uncorrect URL,
        no access to the URL, etc.), return JSON error response:
        ```json
        {{
            "status": "error",
            "error_message": "Unable to extract job description from provided URL:
                             <The error message>"
        }}
        ```
        
        - If you have successfully extracted job description, return JSON response:
        ```json
        {{
            "status": "success",
            "job_description": <The text of job description ONLY.
                                Don't include your thoughts, any additional information
                                or any other text>
        }}
        ```
        """,
        tools=[mcp_tavily_tool],
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
        no access to the URL, etc.), return JSON error response:
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
