"""Agent to google search the information about an company."""

from google.adk.agents import LlmAgent
from google.adk.tools import google_search

try:
    from utils import logging_agent_output_status
except ImportError:
    from app.utils import logging_agent_output_status


def get_web_researcher_agent(model, planner=None):
    """Get web researcher agent."""

    return LlmAgent(
        name="company_web_researcher",
        model=model,
        planner=planner,
        description="Agent to google search the information about an company",
        instruction=\
        """You are a web researcher agent. Your only job is to use the
        `google_search` tool to find information about a company, its culture, values,
        mission and vision based on the provided company official website url.

        If you have successfully found the information about a company, 
        return the information in Markdown format with the "success" status. 
        Otherwise, return the error message with the "error" status.

        IMPORTANT: Your response MUST be valid JSON matching the following structure:
        {
            "status": "success" or "error",
            "message": "The main content of the agent response if the status is 'success'. 
                        The error message if the status is 'error'"
        }

        DO NOT include any explanations or additional text outside the JSON response.
        """,
        # Can't use `output_schema` here due to the conflit with the using tools (`google_search`)
        # output_schema=ResponseContent,
        tools=[google_search],
        output_key="company_info",
        after_agent_callback=logging_agent_output_status
    )
