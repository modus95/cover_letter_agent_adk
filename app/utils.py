"""Utility functions for agent interactions and asynchronous operations."""

import os
import re
import json
import tempfile
import pathlib
import logging
import datetime
import shutil
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Optional
from contextlib import suppress
from pydantic import BaseModel, Field

import pypdf
import streamlit.components.v1 as components

from google.adk.runners import Runner
from google.genai import types
from google.adk.models.google_llm import Gemini
from google.adk.agents.callback_context import CallbackContext
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


@dataclass
class AgentSettings:
    """
    Represents the settings for an agent.

    Attributes:
        models (Optional[str | dict]): The name of the model or a dictionary 
        specifying different models for sub-agents and the main agent.
        g3_thinking_level (str): The thinking level of Gemini3 to use.
        language_level (str): The language level (B1-C2) to use.
        tavily_advanced_extraction (bool): Whether to use Tavily advanced extraction.
    """
    models: Optional[str | dict]
    g3_thinking_level: str
    language_level: str
    tavily_advanced_extraction: bool


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
        "job_information_agent": "job_role_information",
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

    except (KeyError, AttributeError) as err:
        output_logging(output_logger,
                       f"{log_title} / (Raw Output)",
                       json.dumps(output_dict, indent=4),
                       str(err))

    return None


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


def get_planner(md: Gemini, thinking_level: str) -> Optional[BuiltInPlanner]:
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
    version = float(md.model.split("-")[1])
    if version >= 3 and thinking_level in ["minimal", "low", "medium", "high"]:
        return BuiltInPlanner(
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
        )

    return None


def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary file and return the path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def read_pdf(file_name: str) -> str:
    """Reads, extracts, and logs text from a PDF file."""

    file_path = pathlib.Path(file_name)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_name}")

    reader = pypdf.PdfReader(file_path)
    pdf_text = "\n".join([page.extract_text() for page in reader.pages]).strip()

    # Log the extracted text
    output_logging(logging.getLogger("agent_output_logger"),
                   "USER CV",
                   pdf_text)

    return pdf_text


def get_prompt(company_url: str,
               job_role_url: str,
               file_name: str) -> str:
    """
    Returns a formatted prompt string containing company URL, job role URL, and user CV.

    Args:
        company_url (str): The URL of the company.
        job_role_url (str): The URL of the job role.
        file_name (str): The path to the user CV file.
    """
    cv_info = read_pdf(file_name)
    return f"""
### Company url
{company_url}

### Job role url
{job_role_url}

### User CV
{cv_info}
"""


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


def get_domain(url: str) -> str:
    """
    Extracts the domain name from a URL.    
    Example: "https://www.google.com/search" -> "google"
    """
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain.split('.')[0]

    except ValueError:
        return "copy"


def copy_log_file(logfile_name: str, company_url: str):
    """
    Copies the specified log file to a new file named
    "sub_agents_output_<company domain>.log" in the logs folder.
    """
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    source_path = os.path.join(log_dir, logfile_name)

    if not os.path.exists(source_path):
        return

    company_domain = get_domain(company_url)
    dest_filename = f"sub_agents_output_{company_domain}.log"
    dest_path = os.path.join(log_dir, dest_filename)

    shutil.copy2(source_path, dest_path)


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
    ):
    """Call the agent asynchronously with the user's prompt and file."""

    final_response_text = ""

    query_content = types.Content(
        role="user",
        parts=[types.Part(text=prompt)]
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
    # pylint: disable=line-too-long

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
