"""Streamlit application for the Cover Letter AI Agent."""

import asyncio
import os

import streamlit as st
import ui

import nest_asyncio
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.plugins.logging_plugin import LoggingPlugin

import utils
from utils import AgentSettings
from cover_letter_agent.agent import get_root_agent

load_dotenv()
nest_asyncio.apply()

APP_NAME = "Cover Letter Agent"
USER_ID = "streamlit_user"
LOGFILE_NAME = "sub_agents_output.log"


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


async def run_agent(
    prompt: str,
    agent_settings: AgentSettings,
    logging: bool,
) -> str:
    """Run the agent asynchronously."""
    session_service = InMemorySessionService()

    runner = Runner(
        agent=get_root_agent(agent_settings),
        app_name=APP_NAME,
        session_service=session_service,
        plugins=[LoggingPlugin()] if logging else None
    )

    # Create a new session
    new_session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID)
    session_id = new_session.id

    # Process the user query through the agent
    agent_response = await utils.call_agent_async(
        runner,
        USER_ID,
        session_id,
        prompt,
    )

    return agent_response


def main():
    """Main entry point for the Streamlit app."""

    ui.setup_page()
    agent_settings, logging = ui.render_sidebar()
    left, right, company_url, job_description_url, uploaded_file = ui.render_main_inputs()
    generate_clicked = ui.render_generate_button(left, st.session_state.generating)

    # A user clicked the generate button
    if generate_clicked:
        if not company_url or not job_description_url or not uploaded_file:
            ui.render_warning(left, "Please fill in all fields and upload your CV.")
        else:
            st.session_state.generating = True
            st.session_state.is_error = {
                "error": False,
                "message": ""
            }
            ui.render_processing_status(left, right)
            st.rerun()

    # ---- PROCESS GENERATION IF FLAG SET ----
    if st.session_state.generating:
        with ui.render_spinner():
            try:
                utils.setup_loggers(LOGFILE_NAME)

                temp_file_path = utils.save_uploaded_file(uploaded_file)
                prompt = utils.get_prompt(company_url,
                                          job_description_url,
                                          temp_file_path)

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    run_agent(
                        prompt,
                        agent_settings,
                        logging,
                    )
                )

                # Save the result to session_state (PERSIST)
                st.session_state.generated_cover_letter = result

                # Save the log file as `sub_agents_output_<company_domain>.log`
                utils.copy_log_file(LOGFILE_NAME, company_url)

            except RuntimeError as e:
                st.session_state.is_error = {
                    "error": True,
                    "message": str(e)
                }

            finally:
                if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
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
            ui.render_success(left, right,
                              agent_result,
                              utils.st_copy_to_clipboard_button)

        if (not agent_result or
            agent_result.get("status", "") == "error"):
            ui.render_error(left, right, agent_result)

        ui.render_page_link(left, "logs_viewer", "subagent logs")

    # ---- ERROR MESSAGE ----
    if st.session_state.is_error["error"]:
        ui.render_exception_error(left, st.session_state.is_error['message'])


if __name__ == "__main__":
    main()
