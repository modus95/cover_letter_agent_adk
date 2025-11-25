"""Agent to parse CV information from a PDF file uploaded by the user"""

import re
import logging
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OUTPUT_KEY = "cv_info"


def logging_agent_output_status(callback_context: CallbackContext) -> Optional[types.Content]:
    """Log agent output status."""

    current_state = callback_context.state
    output = current_state.get(OUTPUT_KEY)

    status = ""
    if isinstance(output, dict):
        status = output.get("status")

    if isinstance(status, str):
        match = re.search(r'"status"\s*:\s*"([^\"]+)"', output)
        if match:
            status = match.group(1)

    logger.info("Status: %s", status.upper())

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
        If you didn't manage to parse uploaded file (e.g. uncorrect file, no access to the file):
        return JSON error response:
        {
            "status": "error",
            "error_message": "Unable to parse uploaded file: <The error message>"
        }
        
        If you have successfully parsed uploaded file return JSON response:
        {   
            "status": "success",
            "cv_info": <The extracted information>
        }   
        """,
        output_key=OUTPUT_KEY,
        after_agent_callback=logging_agent_output_status
    )
