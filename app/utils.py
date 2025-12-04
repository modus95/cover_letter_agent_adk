"""Utility functions for agent interactions and asynchronous operations."""

import re
import json
import tempfile
import pathlib
from contextlib import suppress
import streamlit.components.v1 as components
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
    