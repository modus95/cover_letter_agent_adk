"""Module for creating an LLM agent to generate job descriptions using Tavily API."""

import os

from typing import Dict
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool
from tavily import TavilyClient

try:
    from utils import logging_agent_output_status
except ImportError:
    from app.utils import logging_agent_output_status


def get_job_role_agent(model,
                       tavily_advanced_extraction,
                       planner=None) -> LlmAgent:
    """
    Creates an LLM agent for fetching job role information using Tavily API.

    Args:
        model: The language model to be used.
        tavily_advanced_extraction: Whether to use Tavily advanced extraction.
        planner: The planner to be used to set up a low thinking level for Gemini 3 models.
                (None <default> - for Gemini 2.5 models)
    Returns:
        LlmAgent
    """
    # --------- TOOLS ---------

    def extract_web_content(url: str,
                            extract_depth: str,
                            output_format: str,
                            ) -> Dict[str, str]:
        """
        Extracts web content from a given URL using Tavily's extraction API.

        Args:
            url: The URL to extract content from.
            extract_depth: The depth of extraction ('basic' or 'advanced').
            output_format: The format of the output content ('markdown' or 'text').           

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

        tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

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


    extract_depth = "advanced" if tavily_advanced_extraction else "basic"
    output_format = "text"


    job_role_fetch_agent = LlmAgent(
        name="fetch_job_role_information",
        model=model,
        planner=planner,
        description="The agent to fetch information about a job role from text",
        instruction=\
        """You are a dedicated assistant tasked with finding and fetching information 
        about a job role.
        ### Task  
        Your responsibility is to carefully review extracted web content, identify and 
        fetch information about a specific job role, including the job title, description, 
        required skills, qualifications, and any other relevant information.

        ### Constraints  
        1. Don't include your thoughts or explanations in the response. Just copy all relevant 
        information from the web content, without changing sentences.
        
        2. You MUST respond ONLY with information about the job role. Remove any additional 
        information that is not directly related to the role (e.g. any web page elements, 
        copyright notices, names of menu items, buttons, etc).
        """
    )
    # --------- END TOOLS ---------

    return LlmAgent(
        name="job_information_agent",
        model=model,
        planner=planner,
        description="Agent to obtain information about a job role",
        instruction=\
        f"""You are a smart assistant who fetches specific information from web content.
        Your task is to find and fetch text information about a job role in the content 
        obtained from the provided URL.

        <Steps>
        To do this, follow these steps:
        1. Extract web content using the `extract_web_content` tool with the following arguments:
            {{"url": provided url,
            "extract_depth": {extract_depth},
            "output_format": {output_format}
            }}
        2. Check the "status" field in the `extract_web_content` tool's response for errors:
         - If `extract_web_content` returns status "success", use the `fetch_job_role_information` 
         agent tool with "web_content" as input to get information about the job role.
         - If `extract_web_content` returns status "error", explain the issue to the user clearly.
        
        3. Respond in the format, defined in the <Output> section.
        </Steps>

        <Output>
        IMPORTANT: Your response MUST be valid JSON matching the following structure:
        - Success:
        {{
            "status": "success",
            "message": text information about the job role. 
        }}

        - Error:
        {{
            "status": "error",
            "message": error message, including a reason of the failure
        }}

        DO NOT include any explanations or additional text outside the JSON response.
        </Output>
        """,
        tools=[
            extract_web_content,
            AgentTool(agent=job_role_fetch_agent)
        ],
        output_key="job_role_information",
        after_agent_callback=logging_agent_output_status
    )
