"""
Streamlit UI module for the Cover Letter AI Agent.

This module provides the Streamlit-based interface components, including page
configuration, sidebar settings, and user interaction elements for generating
cover letters.
"""
import os
import contextlib
import streamlit as st


def setup_page():
    """Configures the page, loads CSS, and renders the sidebar widgets."""
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

    # ----- SIDE BAR -----
    help_text = \
    """
    All controls are disabled since the agent is running 
    on the remote Vertex AI server.
    Set the parameters in the `config.json` file.
    """
    st.sidebar.markdown("", help=help_text)
    gemini_expander = st.sidebar.expander(":grey[**Gemini model**]", expanded=False)
    language_level_expander = st.sidebar.expander(":grey[**Language level**]", expanded=False)
    tavily_expander = st.sidebar.expander(":grey[**Tavily Extractor settings**]", expanded=False)

     # All controls are disabled with using a remote Vertex AI agent
    gemini_expander.selectbox(
        "Sub-agents model",
        options=["gemini-2.5-flash-preview-09-2025",
                "gemini-3-flash-preview"],
        index=0,
        disabled=True
    )
    gemini_expander.selectbox(
        "Main agent model",
        options=["gemini-2.5-flash-preview-09-2025",
                "gemini-3-flash-preview"],
        index=0,
        disabled=True
    )

    gemini_expander.selectbox(
        "Gemini3 thinking level",
        options=["minimal", "low", "medium", "high"],
        index=0,
        help=("The `minimal`/`low` thinking level is preferred "
              "for cover letter generation"),
        disabled=True
    )

    language_level_expander.radio(
        "Language level",
        options=["Intermediate (B1)",
                 "Upper-Intermediate (B2)",
                 "Advanced (C1)",
                 "Proficient (C2)",
                ],
        index=1,
        disabled=True,
        label_visibility="collapsed"
    )

    tavily_expander.toggle(
        "Advanced extraction", value=False, disabled=True,
        help="Enable if there is an issue with extracting the job description"
    )

    st.sidebar.toggle("*Logging*", value=False, disabled=True)


def render_main_inputs():
    """Renders the main input areas and the generate button."""

    t1, t2, t3 = st.columns([0.94, 0.03, 0.03], vertical_alignment="bottom")
    t1.subheader(":orange[*Cover Letter AI Agent (Vertex AI)*]", divider="orange")
    t2.image(os.path.join(os.path.dirname(__file__), "images/Vertex_AI_Logo.svg.png"),
             width="content")
    t3.image(os.path.join(os.path.dirname(__file__), "images/adk_logo.png"),
             width="content")

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
