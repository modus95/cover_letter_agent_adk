"""Utility functions for agent interactions and asynchronous operations."""

from typing import Optional
from pydantic import BaseModel, Field

from google.genai import types
from google.adk.planners.built_in_planner import BuiltInPlanner


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


def get_planner(md: str) -> Optional[BuiltInPlanner]:
    """
    Determines and returns a BuiltInPlanner based on the model version.

    If the model version is 3 or greater, it returns a BuiltInPlanner 
    configured with a low thinking level. Otherwise, it returns None.

    Args:
        md: A string containing model information, (e.g., `md = "gemini-3.0-flash"`).

    Returns:
        An instance of BuiltInPlanner if the model version is 3 or higher,
        otherwise None.
    """

    version = float(md.split("-")[1])
    if version >= 3:
        return BuiltInPlanner(
            thinking_config=types.ThinkingConfig(thinking_level="low")
            )

    return None
    