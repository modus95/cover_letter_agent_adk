"""
Streamlit UI module for the Cover Letter AI Agent.

This module provides the Streamlit-based interface components, including page
configuration, sidebar settings, and user interaction elements for generating
cover letters.
"""
import os
import contextlib
import streamlit as st
from utils import AgentSettings


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

    return None


def render_sidebar() -> AgentSettings:
    """Renders the sidebar widgets and returns the agent settings."""

    gemini_expander = st.sidebar.expander(":blue[**Gemini model**]", expanded=False)
    language_level_expander = st.sidebar.expander(":blue[**Language level**]", expanded=False)
    tavily_expander = st.sidebar.expander(":blue[**Tavily Extractor settings**]", expanded=False)

    models = {
        "sub_agents_model": gemini_expander.selectbox(
                            "Sub-agents model",
                            options=["gemini-2.5-flash-preview-09-2025",
                                    "gemini-3-flash-preview"],
                            index=0
                        ),
        "main_agent_model": gemini_expander.selectbox(
                            "Main agent model",
                            options=["gemini-2.5-flash-preview-09-2025",
                                    "gemini-3-flash-preview"],
                            index=1
                        )
    }

    g3_tl_disabled = all(map(lambda x: float(x.split('-')[1]) != 3, models.values()))
    g3_thinking_level = gemini_expander.selectbox(
                            "Gemini3 thinking level",
                            options=["minimal", "low", "medium", "high"],
                            index=1,
                            help=("The `minimal`/`low` thinking level is preferred "
                                  "for cover letter generation"),
                            disabled=g3_tl_disabled  # enable if any of the models is Gemini3
                        )

    language_level = language_level_expander.radio(
        "Language level",
        options=["Intermediate (B1)",
                 "Upper-Intermediate (B2)",
                 "Advanced (C1)",
                 "Proficient (C2)",
                ],
        index=1,
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


def render_success(left, right, agent_result, copy_callback):
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
            copy_callback(agent_result.get("message", ""))


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
