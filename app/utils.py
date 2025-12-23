"""Utility functions for agent interactions and asynchronous operations."""

import re
import os
import json
import pathlib
from contextlib import suppress
import logging
import pypdf

logger = logging.getLogger(__name__)


def load_json(data):
    """Extract and load JSON from a string."""
    try:
        pat = r'\{[^{}]*(?:{[^{}]*}[^{}]*)*\}'
        return json.loads(re.search(pat, data).group(0))

    except (json.JSONDecodeError, AttributeError):
        return {}


def get_prompt(file_name: str) -> str:
    """Retrieves company and job description URLs and formats them into a prompt string."""

    file_path = pathlib.Path(file_name)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_name}")

    reader = pypdf.PdfReader(file_path)
    cv_info = "\n".join([page.extract_text() for page in reader.pages]).strip()

    # Use environment variables if provided, otherwise prompt the user
    # (for debugging purposes)
    company_url = os.getenv("COMPANY_URL")
    if not company_url:
        company_url = input("Company URL: ")
    else:
        print(f"Company URL: {company_url}")

    job_role_url = os.getenv("JOB_ROLE_URL")
    if not job_role_url:
        job_role_url = input("Job role URL: ")
    else:
        print(f"Job role URL: {job_role_url}")

    return f"""
### Company:
{company_url}

### Job role url:
{job_role_url}

### User CV:
{cv_info}
"""


def _process_remote_agent_response(event):
    """Process remote agent response events."""
    last_part = event.get('content', {}).get('parts',[None])[-1]
    return last_part.get('text','').strip() if last_part else ""


async def _last_or_default(generator, default=None):
    """Returns the last item from an async generator, or a default value."""
    async for default in generator:
        pass
    return default


async def call_remote_agent_async(
    remote_agent,
    user_id: str,
    prompt: str
    ):
    """Call the agent asynchronously with the user's prompt and file."""

    # Create a new session
    remote_session = await remote_agent.async_create_session(user_id=user_id)
    session_id = remote_session['id']
    logger.info("Session ID: %s", session_id)
    logger.info(
        "See the session details at Vertex AI:\n"
        "https://console.cloud.google.com/vertex-ai/agents/locations/europe-west4/agent-engines/"
        "7880138263619436544/playground?session=%s&project=gen-lang-client-0851419956", session_id)

    events_generator = remote_agent.async_stream_query(
        user_id=user_id,
        session_id=session_id,
        message=prompt
        )

    final_response_text = ""
    try:
        lst_event = await _last_or_default(events_generator)
        final_response_text = _process_remote_agent_response(lst_event) if lst_event else ""

    finally:
        with suppress(Exception):
            await events_generator.close()

    return final_response_text
    