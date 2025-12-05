"""Agent to parse CV information from a PDF file uploaded by the user"""

import json
import logging
from typing import Optional

from google.adk.agents import LlmAgent
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

OUTPUT_KEY = "cv_info"
LOG_TITLE = "CV INFO"


def logging_agent_output_status(callback_context: CallbackContext) -> Optional[types.Content]:
    """Log agent output"""

    current_state = callback_context.state
    output = current_state.get(OUTPUT_KEY)

    try:
        output_json = utils.clean_json_string(output)
        status = output_json.get("status")
        cv_info = output_json.get(OUTPUT_KEY)

        if status:
            logger.info("Status: %s", status.upper())
        else:
            logger.info("NO `STATUS` IN THE AGENT OUTPUT")
            utils.output_logging(agent_logger,
                           LOG_TITLE + " (Raw JSON)",
                           str(output_json),
                           "NO `STATUS` IN THE AGENT OUTPUT")
            return None

        if cv_info:
            utils.output_logging(agent_logger, LOG_TITLE, cv_info)
        else:
            utils.output_logging(agent_logger,
                           LOG_TITLE + " (Raw JSON)",
                           str(output_json),
                           "NO `CV_INFO` IN THE AGENT OUTPUT")

    except json.JSONDecodeError as e:
        utils.output_logging(agent_logger,
                       LOG_TITLE + " (Raw Output)",
                       str(output),
                       str(e))

    return None


def get_cv_parcer_agent(model):
    """Get CV parcer agent."""

    return LlmAgent(
        name="cv_parcer_agent",
        model=model,
        description="Agent to parse CV information from a PDF file uploaded by the user",
        instruction="""You are a CV PDF parser agent.
        Your task is to parse (define the text content of) the uploaded PDF file to extract the following information:
            - Name        
            - Summary
            - Skills
            - Work Experience
            - Education

        ### Output format:
        - If you didn't manage to parse uploaded file (e.g. uncorrect file, no access to the file):
        return JSON error response:
        ```json
        {
            "status": "error",
            "error_message": "Unable to parse uploaded file: <The error message>"
        }
        ```
        
        - If you have successfully parsed uploaded file return JSON response:
        ```json
        {
            "status": "success",
            "cv_info": <The extracted information in Markdown format>
        }
        ```   
        """,
        output_key=OUTPUT_KEY,
        after_agent_callback=logging_agent_output_status
    )
