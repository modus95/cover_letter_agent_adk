"""Streamlit application for the Cover Letter AI Agent."""

import asyncio
import os

import streamlit as st
import nest_asyncio
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.plugins.logging_plugin import LoggingPlugin

import utils
from cover_letter_agent.agent import get_root_agent


load_dotenv()
nest_asyncio.apply()

APP_NAME = "Cover Letter Agent"
USER_ID = "streamlit_user"
LOGFILE_NAME = "sub_agents_output.log"


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

gemini_expander = st.sidebar.expander(":blue[**Gemini model**]", expanded=False)
tavily_expander = st.sidebar.expander(":blue[**Tavily Extractor settings**]", expanded=False)

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

# --------------------------------------------------------------------
async def run_agent(
    company_url: str,
    job_description_url: str,
    file_path: str,
    models: dict,
    logging: bool,
    tavily_advanced_extraction: bool
) -> str:
    """Run the agent asynchronously."""
    session_service = InMemorySessionService()

    runner = Runner(
        agent=get_root_agent(models, tavily_advanced_extraction),
        app_name=APP_NAME,
        session_service=session_service,
        plugins=[LoggingPlugin()] if logging else None
    )

    # Create a new session
    new_session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID)
    session_id = new_session.id

    prompt = f"""
    <company url>
    {company_url}
    </company>

    <job description url>
    {job_description_url}
    </job>
    
    """

    # Process the user query through the agent
    agent_response = await utils.call_agent_async(
        runner,
        USER_ID,
        session_id,
        prompt,
        file_path
    )

    return agent_response


def main():
    """Main entry point for the Streamlit app."""

    t1, t2 = st.columns([0.96, 0.04], vertical_alignment="bottom")
    t1.subheader(":blue[*Cover Letter AI Agent*]", divider="blue")
    t2.image("adk_logo.png", width="content")

    # ----- SIDE BAR -----
    models = {
        "sub_agents_model": gemini_expander.selectbox(
                            "Sub-agents model",
                            options=["gemini-2.5-flash-preview-09-2025",
                                    "gemini-2.5-pro",
                                    "gemini-3-pro-preview (Low thinking)"],
                            index=0
                        ),
        "main_agent_model": gemini_expander.selectbox(
                            "Main agent model",
                            options=["gemini-2.5-flash-preview-09-2025",
                                    "gemini-2.5-pro",
                                    "gemini-3-pro-preview (Low thinking)"],
                            index=2
                        )
    }

    tavily_advanced_extraction = tavily_expander.toggle(
        "Advanced extraction", value=False)

    logging = st.sidebar.toggle("*Logging*", value=False)

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
        "**Upload your CV (PDF, DOC)**",
        type=["pdf", "doc", "docx"]
    )

    generate_clicked = left.button(
        "Generate Cover Letter",
        disabled=st.session_state.generating,
        key="generate_btn"
    )

    # User clicked button
    if generate_clicked:
        if not company_url or not job_description_url or not uploaded_file:
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
                temp_file_path = utils.save_uploaded_file(uploaded_file)

                utils.setup_loggers(LOGFILE_NAME)

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    run_agent(
                        company_url,
                        job_description_url,
                        temp_file_path,
                        models,
                        logging,
                        tavily_advanced_extraction
                    )
                )

                # Save the result to session_state (PERSIST)
                st.session_state.generated_cover_letter = result

            except RuntimeError as e:
                st.session_state.is_error = {
                    "error": True,
                    "message": str(e)
                }

            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

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
