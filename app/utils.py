"""Utility functions for agent interactions and asynchronous operations."""

import os
import re
import json
import tempfile
import pathlib
import logging
import datetime

from typing import Optional
from contextlib import suppress
from pydantic import BaseModel, Field

import streamlit.components.v1 as components

from google.adk.runners import Runner
from google.genai import types
from google.adk.models.google_llm import Gemini
from google.adk.agents.callback_context import CallbackContext


RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
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


def logging_agent_output_status(callback_context: CallbackContext) -> None:
    """
    Logs the output status and message of an agent's operation.

    This function extracts the agent's name and its output from the provided
    `CallbackContext`, determines the status (success or error), and logs
    the relevant information using a dedicated agent output logger. It handles
    different agents by mapping their names to specific output keys.

    Args:
        callback_context (CallbackContext): The context object containing
                                            the agent's state and name
    """

    status_logger = logging.getLogger("agent_status_logger")
    output_logger = logging.getLogger("agent_output_logger")

    output_keys = {
        "cv_parcer_agent": "cv_info",
        "job_description_extractor_agent": "job_description",
        "company_web_researcher": "company_info"
    }

    current_state = callback_context.state
    agent_name = callback_context.agent_name
    agent_output_key = output_keys[agent_name]
    output_dict = current_state.get(agent_output_key)
    if isinstance(output_dict, str):
        output_dict = load_json(output_dict)

    log_title = " ".join(agent_output_key.split("_")).upper()

    try:
        status = output_dict.get("status")
        message = output_dict.get("message")

        status_logger.info("%s: %s", agent_name, status.upper())
        output_logging(output_logger,
                       f"{log_title} / {status.upper()}",
                       message)

    except KeyError as ke:
        output_logging(output_logger,
                       f"{log_title} / (Raw Output)",
                       json.dumps(output_dict, indent=4),
                       str(ke))

    return None


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


def output_logging(logg: logging.Logger,
                   ttl: str,
                   info_str: str,
                   warning: Optional[str] = None) -> None:
    """
    Logs formatted output including a title, separator, optional warning, and information string.

    Args:
        logg (logging.Logger): The logger instance to use for output.
        ttl (str): The title or header string for the log output.
        info_str (str): The main information string to be logged.
        warning (Optional[str]): An optional warning message. 
            If provided, it will be logged as a warning.
    """
    logg.info(ttl)
    logg.info("%s", "-" * len(ttl))
    if warning:
        logg.warning("%s\n", warning)
    logg.info("%s\n\n", str(info_str))


def setup_loggers(logfile_name: str):
    """
    Sets up logging configuration for the application, creating a log directory
    and configuring two specific loggers: one for agent outputs and one for agent status.

    Args:
        logfile_name (str): The name of the log file to be created within the log directory.
    """
    #----- AGENT OUTPUT LOGGER -----
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, logfile_name)

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("agent_output_logger")
    logger.setLevel(logging.INFO)

    # Remove existing handlers to prevent duplicate logging
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.propagate = False

    logger.info("%s", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    #----- AGENT STATUS LOGGER -----
    status_logger = logging.getLogger("agent_status_logger")
    status_logger.setLevel(logging.INFO)
    if not status_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        status_logger.addHandler(console_handler)
    status_logger.propagate = False

    return logger, status_logger


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
    