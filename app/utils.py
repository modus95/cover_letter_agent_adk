"""Utility functions for agent interactions and asynchronous operations."""

import re
import json
import tempfile
import pathlib
from contextlib import suppress
from google.adk.runners import Runner
from google.genai import types
from google.adk.models.google_llm import Gemini


RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


def define_model(model_name:str):
    """
    Initializes and returns a Gemini model instance.

    Args:
        model_name (str): The name of the Gemini model to instantiate.
    Returns:
        Gemini: An instance of the Gemini model configured with the specified retry options.
    """
    return Gemini(model=model_name, retry_options=RETRY_CONFIG)


def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary file and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


async def process_agent_response(event):
    """Process agent response events."""

    final_response = None

    if (event.is_final_response()
        and event.content
        and event.content.parts
        and hasattr(event.content.parts[-1], "text")
        and event.content.parts[-1].text
        ):
        final_response = event.content.parts[-1].text.strip()

    return final_response


async def call_agent_async(
    runner: Runner,
    user_id: str,
    session_id: str,
    prompt: str,
    file_name: str
    ):
    """Call the agent asynchronously with the user's prompt and file."""

    final_response_text = ""

    file_path = pathlib.Path(file_name)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_name}")

    query_content = types.Content(
        role="user",
        parts=[
            types.Part(text=prompt),
            types.Part.from_bytes(
                data=file_path.read_bytes(),
                mime_type="application/pdf"
                )
            ]
        )

    agen = runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=query_content,
    )

    try:
        async for event in agen:
            response = await process_agent_response(event)
            if response:
                final_response_text = response

    finally:
        with suppress(Exception):
            await agen.close()

    return final_response_text


def load_json(data):
    """Extract and load JSON from a string."""
    try:
        pat = r'\{[^{}]*(?:{[^{}]*}[^{}]*)*\}'
        return json.loads(re.search(pat, data).group(0))

    except json.JSONDecodeError:
        return str(data)
    