"""Agent to google search the information about an company."""

import re
import logging
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_KEY = "company_info"


def logging_agent_output_status(callback_context: CallbackContext) -> Optional[types.Content]:
    """Log agent output status."""

    agent = callback_context.agent_name
    current_state = callback_context.state
    output = current_state.get(OUTPUT_KEY)

    status = ""
    if isinstance(output, dict):
        status = output.get("status")

    if isinstance(status, str):
        match = re.search(r'"status"\s*:\s*"([^\"]+)"', output)
        if match:
            status = match.group(1)

    logger.info("Agent %s: Status: %s", agent, status)

    return None


def get_web_researcher_agent(model):
    """Get web researcher agent."""

    return LlmAgent(
        name="company_web_researcher",
        model=model,
        description="Agent to google search the information about an company",
        instruction=\
        """You are a web researcher agent. Your only job is to use the
        google_search tool to find information about a company, its culture, values,
        mission and vision based on provided company official website url.
        
        ### Output format:
        If you didn't manage to retreive any information about the company(e.g. uncorrect url)
        return JSON error response:
        {
            "status": "error",
            "error_message": "Unable to retrieve information about the company: <The error message>"
        }
        
        If you have successfully retrieved information about the company return JSON response:
        {
            "status": "success",
            "company_info": <Information about the company>
        }
        """,
        tools=[google_search],
        output_key=OUTPUT_KEY,
        after_agent_callback=logging_agent_output_status
    )
