"""Utility functions for agent interactions and asynchronous operations."""

import re
import json

from typing import Optional
from pydantic import BaseModel, Field

from google.genai import types
from google.adk.models.google_llm import Gemini
from google.adk.planners.built_in_planner import BuiltInPlanner


RETRY_CONFIG = types.HttpRetryOptions(
    attempts=3,  # Maximum retry attempts
    exp_base=5,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


class ResponseContent(BaseModel):
    """
    Represents the structured content of an agent's response.

    Attributes:
        status (str): The status of the agent response, either 'success' or 'error'.
        message (str): The main content of the agent response if status is 'success',
                       or the error message if status is 'error'.
    """
    status: str = Field(
        description="The status of the agent response. Should be 'success' or 'error'."
    )
    message: str = Field(
        description=(
            "The main content of the agent response if the status is 'success'."
            " The error message if the status is 'error'."
            )
    )


def load_json(data):
    """Extract and load JSON from a string."""
    try:
        pat = r'\{[^{}]*(?:{[^{}]*}[^{}]*)*\}'
        return json.loads(re.search(pat, data).group(0))

    except json.JSONDecodeError:
        return {}


def define_model(model_name:str) -> Gemini:
    """
    Initializes and returns a Gemini model instance.

    Args:
        model_name (str): The name of the Gemini model to instantiate.
    Returns:
        Gemini: An instance of the Gemini model configured with the specified retry options.
    """
    # Remove parentheses and their contents from the model name
    # example: "gemini-3-pro-preview (Low thinking)" -> "gemini-3-pro-preview"
    model_name = re.sub(r"\s*\([^)]*\)", "", model_name)
    return Gemini(model=model_name, retry_options=RETRY_CONFIG)


def get_planner(md: Gemini) -> Optional[BuiltInPlanner]:
    """
    Determines and returns a BuiltInPlanner based on the model version.

    If the model version (extracted from `md.model`) is 3 or greater,
    it returns a BuiltInPlanner configured with a low thinking level.
    Otherwise, it returns None.

    Args:
        md: An object containing model information, expected to have a 'model'
            attribute (e.g., `md.model = "gemini-3.0-flash"`).

    Returns:
        An instance of BuiltInPlanner if the model version is 3 or higher,
        otherwise None.
    """

    version = float(md.model.split("-")[1])
    if version >= 3:
        return BuiltInPlanner(
            thinking_config=types.ThinkingConfig(thinking_level="low")
            )

    return None
    