"""Utility functions for agent interactions and asynchronous operations."""

import pathlib
from google.adk.runners import Runner
from google.genai import types


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

    final_response_text = ""

    try:
        async for event in runner.run_async(
            user_id=user_id, session_id=session_id, new_message=query_content
        ):
            # Process each event and get the final response if available
            response = await process_agent_response(event)
            if response:
                final_response_text = response
    except Exception as e:
        print(f"Error during agent call: {e}")

    return final_response_text
    