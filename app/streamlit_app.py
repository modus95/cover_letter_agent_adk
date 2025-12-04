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

# Page configuration
st.set_page_config(
    page_title="Cover Letter AI Agent",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

default_border_color = st.get_option("theme.borderColor")

# Custom CSS for "fancy" look and dynamic border colors
st.html("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stButton>button[disabled] {
        background-color: #9e9e9e !important;
        color: #ffffff !important;
        cursor: not-allowed !important;
        opacity: 0.85 !important;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;        
    }
    .stTextInput label p, .stFileUploader label p {
        font-size: 1.1rem !important;
        font-weight: bold;
    }    
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Decrease font size of selectbox options */
    .stSelectbox div[data-baseweb="select"] > div {
        font-size: 0.8rem !important;
    }
    div[data-baseweb="popover"] li, div[data-baseweb="popover"] div {
        font-size: 0.8rem !important;
    }
    
    /* Dynamic border colors based on status */
    [data-testid="stColumn"]:has([data-status="success"]) {
        border: 2px solid #4CAF50 !important;
        border-radius: 10px;
    }
    [data-testid="stColumn"]:has([data-status="error"]) {
        border: 2px solid #FF9800 !important;
        border-radius: 10px;
    }
    [data-testid="stColumn"]:has([data-status="pending"]) {
        border: 2px solid {default_border_color} !important;
        border-radius: 10px;
    }
    
    /* Dark-grey border for the text area with agent output */
    [data-testid="stColumn"] .stTextArea [data-baseweb="textarea"] {
        border: 2px solid #c1c1c1 !important;
        border-radius: 10px !important;
    }    
</style>
""")

# Settings in sidebar
settings_expander = st.sidebar.expander("**Settings**", expanded=False)

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
    model: str,
    logging: bool,
    tavily_advanced_extraction: bool
) -> str:
    """Run the agent asynchronously."""
    session_service = InMemorySessionService()

    runner = Runner(
        agent=get_root_agent(model, tavily_advanced_extraction),
        app_name=APP_NAME,
        session_service=session_service,
        plugins=[LoggingPlugin()] if logging else None
    )

    # Create a new session
    new_session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID)
    session_id = new_session.id

    prompt = f"""
    ### Company:
    {company_url}

    ### Job description:
    {job_description_url}
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

    st.title("📝 Cover Letter AI Agent")
    st.divider()

    # ----- SIDE BAR -----
    model_name = settings_expander.selectbox(
        "Gemini Model",
        options=["gemini-2.5-flash-preview-09-2025",
                 "gemini-2.5-flash-lite"],
        index=0
    )

    tavily_advanced_extraction = settings_expander.toggle(
        "Tavily Advanced Extraction", value=False)
    settings_expander.divider()
    logging = settings_expander.toggle("Logging", value=False)

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

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    run_agent(
                        company_url,
                        job_description_url,
                        temp_file_path,
                        model_name,
                        logging,
                        tavily_advanced_extraction
                    )
                )

                # Save the result to session_state (PERSIST)
                st.session_state.generated_cover_letter = utils.load_json(result)

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

        if (isinstance(agent_result, dict) and
            agent_result.get("status", "") == "success"):

            # Add invisible status marker for CSS targeting
            left.html('<div data-status="success" style="display:none;"></div>')
            right.html('<div data-status="success" style="display:none;"></div>')

            left.success("Cover Letter Generated Successfully!", icon="✅")

            right.text_area(
                "Cover Letter",
                value=agent_result.get("cover_letter", ""),
                height=450,
                label_visibility="collapsed"
            )

            with right:
                c1, c2 = st.columns([0.85, 0.15], vertical_alignment="center")
                with c1:
                    st.markdown("*:red[*Read carefully and make adjustments if needed.]*")
                with c2:
                    utils.st_copy_to_clipboard_button(agent_result.get("cover_letter", ""))

        if (isinstance(agent_result, dict) and
            agent_result.get("status", "") == "error"):

            # Add invisible status marker for CSS targeting
            left.html('<div data-status="error" style="display:none;"></div>')
            right.html('<div data-status="error" style="display:none;"></div>')

            left.warning("Cover Letter Generation Failed!", icon="⚠️")

            md = f"*:blue[{agent_result.get('failure_reason', '')}]*"
            right.markdown(md)

    # ---- ERROR MESSAGE ----
    if st.session_state.is_error["error"]:
        left.error(f"An error occurred: {st.session_state.is_error['message']}", icon="❌")


if __name__ == "__main__":
    main()
