"""
Utility functions and data models for interacting with Vertex AI and Gemini models.
Should be deployed in the Vertex AI Agent Engine.
"""


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


def get_planner(md: str, thinking_level: str) -> Optional[BuiltInPlanner]:
    """
    Determines and returns a BuiltInPlanner based on the model version.

    If the model version (extracted from `md.model`) is 3 or greater,
    it returns a BuiltInPlanner configured with a low thinking level.
    Otherwise, it returns None.

    Args:
        md: An object containing model information, expected to have a 'model'
            attribute (e.g., `md.model = "gemini-3.0-flash"`).
        thinking_level: The thinking level to use for the planner 
        ("minimal", "low", "medium", "high").

    Returns:
        An instance of BuiltInPlanner if the model version is 3 or higher,
        otherwise None.
    """
    version = float(md.split("-")[1])
    if version >= 3 and thinking_level in ["minimal", "low", "medium", "high"]:
        return BuiltInPlanner(
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
        )

    return None
    