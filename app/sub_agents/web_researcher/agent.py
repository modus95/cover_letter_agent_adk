"""Agent to google search the information about an company."""

import json
import logging
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

try:
    import utils
except ImportError:
    from app import utils


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

agent_logger = logging.getLogger("agent_output_logger")


OUTPUT_KEY = "company_info"
LOG_TITLE = "COMPANY INFO"


def logging_agent_output_status(callback_context: CallbackContext) -> Optional[types.Content]:
    """Log agent output"""

    current_state = callback_context.state
    output = current_state.get(OUTPUT_KEY)

    try:
        output_json = utils.clean_json_string(output)
        status = output_json.get("status")
        company_info = output_json.get(OUTPUT_KEY)

        if status:
            logger.info("Status: %s", status.upper())
        else:
            logger.info("NO `STATUS` IN THE AGENT OUTPUT")
            utils.output_logging(agent_logger,
                           LOG_TITLE + " (Raw JSON)",
                           str(output_json),
                           "NO `STATUS` IN THE AGENT OUTPUT")
            return None

        if company_info:
            utils.output_logging(agent_logger, LOG_TITLE, company_info)
        else:
            utils.output_logging(agent_logger,
                           LOG_TITLE + " (Raw JSON)",
                           str(output_json),
                           "NO `COMPANY_INFO` IN THE AGENT OUTPUT")

    except json.JSONDecodeError as e:
        utils.output_logging(agent_logger,
                       LOG_TITLE + " (Raw Output)",
                       str(output),
                       str(e))

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
        - If you didn't manage to retreive any information about the company(e.g. uncorrect url)
        return JSON error response:
        ```json
        {
            "status": "error",
            "error_message": "Unable to retrieve information about the company: <The error message>"
        }
        ```
        
        - If you have successfully retrieved information about the company return JSON response:
        ```json
        {
            "status": "success",
            "company_info": <Information about the company in Markdown format>
        }
        ```
        """,
        tools=[google_search],
        output_key=OUTPUT_KEY,
        after_agent_callback=logging_agent_output_status
    )
