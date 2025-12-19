"""Module for creating an LLM agent to generate job descriptions using Tavily MCP tools."""

import os

from typing import Dict
from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext
from tavily import TavilyClient

try:
    from utils import logging_agent_output_status
except ImportError:
    from app.utils import logging_agent_output_status


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def extract_web_content(url: str,
                        extract_depth: str,
                        output_format: str,
                        tool_context: ToolContext) -> Dict[str, str]:
    """
    Extracts web content from a given URL using Tavily's extraction API.

    Args:
        url: The URL to extract content from.
        extract_depth: The depth of extraction ('basic' or 'advanced').
        output_format: The format of the output content ('markdown' or 'text'), default is 'text'.

    Returns:
        Dictionary with status and extracted web content.
        Success: 
            {"status": "success",
            "web_content": "<extracted web content>"}
        Error:
            {"status": "error",
            "error_message": "<error message>"}
    """
    # pylint: disable=broad-exception-caught
    tool_context.actions.skip_summarization = True

    try:
        response = tavily_client.extract(urls=[url],
                                         extract_depth=extract_depth,
                                         format=output_format,
                                         )
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e)
        }

    if response.get("failed_results",[]):
        return {
            "status": "error",
            "error_message": response.get("failed_results")[0].get("error")
        }

    return {
            "status": "success",
            "web_content": response.get("results")[0].get("raw_content")
        }


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
    output_format = "text"

    return LlmAgent(
        name="job_description_extractor_agent",
        model=model,
        planner=planner,
        description="Agent to obtain the job description text from a provided URL",
        instruction=\
        f"""You are a smart assistant who fetches information about a specific job role.
        Your task is to find (and return) the text that aligns with the job role description
        from the content obtained from the provided URL.

        In order to get your task done:
        1. Extract web content using the `extract_web_content` tool with the following arguments:
            {{"url": provided url,
            "extract_depth": {extract_depth},
            "output_format": {output_format}
            }}
        2. Check the "status" field in the tool's response for errors.
        3. If the tool returns status "error", explain the issue to the user clearly.
        4. If the tool returns status "success", process the "web_content" field 
        in the tool's response: 
        You MUST identify and leave a text that aligns with the job role description ONLY. 
        Remove any additional information that is not corresponding to the job role description.
        5. Respond with the following output format.

        <Output>
        IMPORTANT: Your response MUST be valid JSON matching the following structure:
        - Success:
        {{
            "status": "success",
            "message": the text of the job description. 
        }}

        - Error:
        {{
            "status": "error",
            "message": the error message, including a reason of the failure
        }}

        DO NOT include any explanations or additional text outside the JSON response.
        </Output>
        """,
        tools=[extract_web_content],
        output_key="job_description",
        after_agent_callback=logging_agent_output_status
    )
