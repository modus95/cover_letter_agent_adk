"""
Streamlit UI module for the Cover Letter AI Agent.

This module provides the Streamlit-based interface components, including page
configuration, sidebar settings, and user interaction elements for generating
cover letters.
"""
import os
import json
import contextlib
import streamlit as st
import streamlit.components.v1 as components
from utils import AgentSettings, get_gemini_model_list


def setup_page() -> None:
    """Configures the page, loads CSS."""
    st.set_page_config(
        page_title="Cover Letter AI Agent",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    default_border_color = st.get_option("theme.borderColor") or "#d6d6d8"

    # Custom CSS for "fancy" look and dynamic border colors
    css_file_path = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_file_path, encoding="utf-8") as f:
        css = f.read()

    st.html(f"""
    <style>
        :root {{
            --border-color: {default_border_color};
        }}
        {css}
    </style>
    """)


def render_sidebar() -> AgentSettings:
    """Renders the sidebar widgets and returns the agent settings."""

    gemini_expander = st.sidebar.expander(":blue[**Gemini model**]", expanded=False)
    language_level_expander = st.sidebar.expander(":blue[**Language level**]", expanded=False)
    tavily_expander = st.sidebar.expander(":blue[**Tavily Extractor settings**]", expanded=False)

    available_models = get_gemini_model_list()

    models = {
        "sub_agents_model": gemini_expander.selectbox(
                            "Sub-agents model",
                            options=available_models,
                            index=0
                        ),
        "main_agent_model": gemini_expander.selectbox(
                            "Main agent model",
                            options=available_models,
                            index=0
                        )
    }

    g3_thinking_level = gemini_expander.selectbox(
                            "Gemini3 thinking level",
                            options=["minimal", "low", "medium", "high"],
                            index=1,
                            help=("The `minimal`/`low` thinking level is preferred "
                                  "for cover letter generation"),
                        )

    top_p = gemini_expander.slider(
                            "Top P",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.95,
                            step=0.05,
                            help=("The lower - more predictable text.\n"
                                  "The higher - more creative text.")
                        )

    language_level = language_level_expander.radio(
        "Language level",
        options=["Intermediate (B1)",
                 "Upper-Intermediate (B2)",
                 "Advanced (C1)",
                #  "Proficient (C2)",
                ],
        index=0,
        label_visibility="collapsed"
    )

    tavily_advanced_extraction = tavily_expander.toggle(
        "Advanced extraction", value=True,
        help="Enable if there is an issue with extracting the job description"
    )

    logging = st.sidebar.toggle("*Logging*", value=False)

    return AgentSettings(
        models=models,
        g3_thinking_level=g3_thinking_level,
        top_p=top_p,
        language_level=language_level,
        tavily_advanced_extraction=tavily_advanced_extraction
        ), logging


def render_main_inputs():
    """Renders the main input areas and the generate button."""

    t1, t2 = st.columns([0.96, 0.04], vertical_alignment="bottom")
    t1.subheader(":blue[*Cover Letter AI Agent*]", divider="blue")
    # Construct absolute path to the logo image
    logo_path = os.path.join(os.path.dirname(__file__), "adk_logo.png")
    t2.image(logo_path, width="content")

    # ----- MAIN PAGE -----
    left, right = st.columns(
        [0.4, 0.6],
        gap="medium",
        vertical_alignment="top",
        border=True
    )

    company_url = left.text_input(
            "**Company Website URL**",
            placeholder="https://www.example.com"
        )

    job_description_url = left.text_input(
            "**Job Description URL**",
            placeholder="https://careers.example.com/job/123"
        )

    uploaded_file = left.file_uploader(
        "**Upload your CV (PDF)**",
        type=["pdf"]
    )

    return left, right, company_url, job_description_url, uploaded_file


def render_generate_button(container, generating_state):
    """Renders the generate button."""
    return container.button("Generate Cover Letter",
                      disabled=generating_state,
                      key="generate_btn")


@contextlib.contextmanager
def render_spinner():
    """Context manager for the loading spinner."""
    with st.spinner(":blue[*Generating cover letter... This may take a minute.*]"):
        yield


def render_processing_status(left, right):
    """Renders invisible status markers for processing state."""
    left.html('<div data-status="pending" style="display:none;"></div>')
    right.html('<div data-status="pending" style="display:none;"></div>')


def render_warning(container, message):
    """Renders a warning message in the specified container."""
    container.warning(message)


def render_page_link(container, page_name, link_text):
    """Renders a link to the logs viewer."""
    container.html(f'''
        <a href="{page_name}" target="_blank"
        style="font-size: 0.9rem; font-style: italic;"
        >{link_text}</a>
    ''')


def render_success(left, right, agent_result):
    """Renders the success message and result."""
    # Add invisible status marker for CSS targeting
    left.html('<div data-status="success" style="display:none;"></div>')
    right.html('<div data-status="success" style="display:none;"></div>')

    left.success("Cover Letter Generated Successfully!", icon="✅")

    right.text_area(
        "Cover Letter",
        value=agent_result.get("message", ""),
        height=450,
        label_visibility="collapsed"
    )

    with right:
        c1, c2 = st.columns([0.85, 0.15], vertical_alignment="center")
        with c1:
            st.markdown("*:red[*Read carefully and make adjustments if needed.]*")
        with c2:
            st_copy_to_clipboard_button(agent_result.get("message", ""))


def render_error(left, right, agent_result=None):
    """Renders error messages."""
    # Add invisible status marker for CSS targeting
    left.html('<div data-status="error" style="display:none;"></div>')
    right.html('<div data-status="error" style="display:none;"></div>')

    left.warning("Cover Letter Generation Failed!", icon="⚠️")

    md = "*:blue[The response from the agent is empty! Check logs for more details.]*"
    if agent_result:
        md = f"*:blue[{agent_result.get('message', '')}]*"

    right.markdown(md)


def render_exception_error(container, message):
    """Renders exception error."""
    container.error(f"An error occurred: {message}", icon="❌")


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
