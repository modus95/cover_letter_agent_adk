"""
Streamlit application for interacting with a Vertex AI-powered Cover Letter Agent.

This module initializes the Vertex AI environment, loads configuration from environment 
variables, and provides a user interface for generating and managing cover letters 
using Google Cloud's Agent Engine.
"""


import asyncio
import os
import logging

import nest_asyncio
import streamlit as st
from dotenv import dotenv_values

import vertexai
from vertexai import agent_engines
import utils


logging.basicConfig(level=logging.INFO)

config = dotenv_values(".env_remote")

project_id = config.get("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
location = config.get("GOOGLE_CLOUD_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION")
bucket = config.get("GOOGLE_CLOUD_STAGING_BUCKET") or os.getenv("GOOGLE_CLOUD_STAGING_BUCKET")
agent_name = config.get("AGENT_NAME") or os.getenv("AGENT_NAME")
user_id = config.get("USER_ID") or os.getenv("USER_ID")

vertexai.init(project=project_id,
              location=location,
              staging_bucket=bucket)

nest_asyncio.apply()

APP_NAME = "Cover Letter Agent"
# LOGFILE_NAME = "sub_agents_output.log"


# Page configuration
st.set_page_config(
    page_title="Cover Letter AI Agent",
    page_icon="📃",
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

st.sidebar.markdown("", help="All controls are disabled since the agent "
                             "is running on the remote Vertex AI server")
gemini_expander = st.sidebar.expander(":grey[Gemini model]", expanded=False)
language_level_expander = st.sidebar.expander(":grey[Language level]", expanded=False)
tavily_expander = st.sidebar.expander(":grey[Tavily Extractor settings]", expanded=False)

# ---- SESSION STATE ----
if "generating" not in st.session_state:
    st.session_state.generating = False

if "generated_cover_letter" not in st.session_state:
    st.session_state.generated_cover_letter = None

if "is_error" not in st.session_state:
    st.session_state.is_error = {
        "error": False,
        "message": ""
    }


def main():
    """Main entry point for the Streamlit app."""

    t1, t2, t3 = st.columns([0.94, 0.03, 0.03], vertical_alignment="bottom")
    t1.subheader(":orange[*Cover Letter AI Agent (Vertex AI)*]", divider="orange")
    t2.image("Vertex_AI_Logo.svg.png", width="content")
    t3.image("adk_logo.png", width="content")

    # ----- SIDE BAR -----
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

    job_role_url = left.text_input(
            "**Job Description URL**",
            placeholder="https://careers.example.com/job/123"
        )

    uploaded_file = left.file_uploader(
        "**Upload your CV (PDF)**",
        type=["pdf"]
    )

    generate_clicked = left.button(
        "Generate Cover Letter",
        disabled=st.session_state.generating,
        key="generate_btn"
    )

    # User clicked button
    if generate_clicked:
        if not company_url or not job_role_url or not uploaded_file:
            left.warning("Please fill in all fields and upload your CV.")
        else:
            st.session_state.generating = True
            st.session_state.is_error = {
                "error": False,
                "message": ""
            }
            left.html('<div data-status="pending" style="display:none;"></div>')
            right.html('<div data-status="pending" style="display:none;"></div>')
            st.rerun()

    # ---- PROCESS GENERATION IF FLAG SET ----
    if st.session_state.generating:
        with st.spinner(":blue[*Generating cover letter... This may take a minute.*]"):
            try:
                # utils.setup_loggers(LOGFILE_NAME)
                prompt = utils.get_prompt(uploaded_file,
                                          company_url,
                                          job_role_url,
                                          )

                existing_agents = list(agent_engines.list(filter=f'display_name={agent_name}'))
                if existing_agents:
                    remote_agent = existing_agents[0]

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(
                        utils.call_remote_agent_async(
                            remote_agent,
                            project_id,
                            user_id,
                            prompt
                        )
                    )

                    # Save the result to session_state (PERSIST)
                    st.session_state.generated_cover_letter = result

                    # Save the log file as `sub_agents_output_<company_domain>.log`
                    # utils.copy_log_file(LOGFILE_NAME, company_url)
                else:
                    raise RuntimeError(f"Vertex AI remote agent '{agent_name}' not found!")

            except RuntimeError as e:
                st.session_state.is_error = {
                    "error": True,
                    "message": str(e)
                }

            finally:
                # ENABLE the button again
                st.session_state.generating = False
                st.rerun()


    # ---- SHOW RESULT IF AVAILABLE ----
    if st.session_state.generated_cover_letter:
        agent_result = st.session_state.generated_cover_letter
        if isinstance(agent_result, str):
            agent_result = utils.load_json(agent_result)

        if agent_result.get("status", "") == "success":

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
                    utils.st_copy_to_clipboard_button(agent_result.get("message", ""))

        if (not agent_result or
            agent_result.get("status", "") == "error"):

            # Add invisible status marker for CSS targeting
            left.html('<div data-status="error" style="display:none;"></div>')
            right.html('<div data-status="error" style="display:none;"></div>')

            left.warning("Cover Letter Generation Failed!", icon="⚠️")

            md = "*:blue[The response from the agent is empty! Check logs for more details.]*"
            if agent_result:
                md = f"*:blue[{agent_result.get('message', '')}]*"

            right.markdown(md)

    # ---- ERROR MESSAGE ----
    if st.session_state.is_error["error"]:
        left.error(f"An error occurred: {st.session_state.is_error['message']}", icon="❌")


if __name__ == "__main__":
    main()
