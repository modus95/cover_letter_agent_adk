"""Utility functions for agent interactions and asynchronous operations."""

import re
import os
import json
import pathlib
from typing import Optional
from contextlib import suppress
import logging
import pypdf
import streamlit.components.v1 as components
import cover_letter_agent.agent as agent

logger = logging.getLogger(__name__)


def load_json(data):
    """Extract and load JSON from a string."""
    try:
        pat = r'\{[^{}]*(?:{[^{}]*}[^{}]*)*\}'
        return json.loads(re.search(pat, data).group(0))

    except (json.JSONDecodeError, AttributeError):
        return {}


def get_prompt(cv_file: Optional[str | bytes],
               company_url: str=None,
               job_role_url: str=None) -> str:
    """Retrieves company and job description URLs and formats them into a prompt string."""

    if isinstance(cv_file, str):
        file_ = pathlib.Path(cv_file)
        if not file_.exists():
            raise FileNotFoundError(f"File not found: {cv_file}")
    else:
        file_ = cv_file  # file like BytesIO object

    reader = pypdf.PdfReader(file_)
    cv_info = "\n".join([page.extract_text() for page in reader.pages]).strip()

    if not company_url:
        # Use environment variables if provided, otherwise prompt the user
        # (for debugging purposes)
        company_url = os.getenv("COMPANY_URL")
        if not company_url:
            company_url = input("Company URL: ")
        else:
            print(f"Company URL: {company_url}")

    if not job_role_url:
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

    if isinstance(agent.MODELS, dict):
        # pylint: disable=E1126
        logger.info("Sub-agents models: %s", agent.MODELS["sub_agents_model"])
        logger.info("Main agent model: %s", agent.MODELS["main_agent_model"])
    else:
        logger.info("Agent models: %s", agent.MODELS)
    logger.info("Language level: %s", agent.LANG_LEVEL)
    logger.info("Gemini3 thinking level: %s", agent.G3_THINK)
    logger.info("Tavily advanced extraction: %s", agent.TAVILY_ADVANCE)
    logger.info("\n")

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


def st_copy_to_clipboard_button(text: str):
    """
    Displays a copy-to-clipboard button using a custom HTML component.
    
    Args:
        text (str): The text to be copied to the clipboard.
    """
    # Escape the text for JavaScript
    text_js = json.dumps(text)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .zeroclipboard-container {{
                display: flex;
                justify-content: flex-start; /* Align to left to match potential layout, or center */
                align-items: center;
            }}
            .ClipboardButton {{
                background-color: transparent;
                border: none;
                cursor: pointer;
                padding: 4px;
                border-radius: 6px;
                color: #57606a;
                transition: background-color 0.2s;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .ClipboardButton:hover {{
                background-color: rgba(0,0,0,0.05);
                color: #0969da;
            }}
            .d-none {{
                display: none !important;
            }}
            .color-fg-success {{
                color: #1a7f37 !important;
            }}
        </style>
    </head>
    <body>
        <div class="zeroclipboard-container">
            <button aria-label="Copy" class="ClipboardButton" id="copy-button">
                <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" class="octicon octicon-copy js-clipboard-copy-icon" id="copy-icon">
                    <path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 0 1 0 1.5h-1.5a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-1.5a.75.75 0 0 1 1.5 0v1.5A1.75 1.75 0 0 1 9.25 16h-7.5A1.75 1.75 0 0 1 0 14.25Z"></path>
                    <path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0 1 14.25 11h-7.5A1.75 1.75 0 0 1 5 9.25Zm1.75-.25a.25.25 0 0 0-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 0 0 .25-.25v-7.5a.25.25 0 0 0-.25-.25Z"></path>
                </svg>
                <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" class="octicon octicon-check js-clipboard-check-icon color-fg-success d-none" id="check-icon">
                    <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.751.751 0 0 1 .018-1.042.751.751 0 0 1 1.042-.018L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z"></path>
                </svg>
            </button>
        </div>

        <script>
            const button = document.getElementById('copy-button');
            const copyIcon = document.getElementById('copy-icon');
            const checkIcon = document.getElementById('check-icon');
            const textToCopy = {text_js};

            button.addEventListener('click', () => {{
                navigator.clipboard.writeText(textToCopy).then(() => {{
                    copyIcon.classList.add('d-none');
                    checkIcon.classList.remove('d-none');
                    
                    setTimeout(() => {{
                        checkIcon.classList.add('d-none');
                        copyIcon.classList.remove('d-none');
                    }}, 2000);
                }}).catch(err => {{
                    console.error('Failed to copy text: ', err);
                }});
            }});
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=40)
    