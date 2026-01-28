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
import ui
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

    ui.setup_page()
    left, right, company_url, job_role_url, uploaded_file = ui.render_main_inputs()
    generate_clicked = ui.render_generate_button(left, st.session_state.generating)

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
            ui.render_processing_status(left, right)
            st.rerun()

    # ---- PROCESS GENERATION IF FLAG SET ----
    if st.session_state.generating:
        with ui.render_spinner():
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
            ui.render_success(left, right,
                              agent_result,
                              utils.st_copy_to_clipboard_button)

        if (not agent_result or
            agent_result.get("status", "") == "error"):
            ui.render_error(left, right, agent_result)

    # ---- ERROR MESSAGE ----
    if st.session_state.is_error["error"]:
        ui.render_exception_error(left, st.session_state.is_error['message'])


if __name__ == "__main__":
    main()
