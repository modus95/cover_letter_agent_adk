"""This module defines the main entry point for the cover letter agent."""

import asyncio
import argparse
import os

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.plugins.logging_plugin import LoggingPlugin

import utils
from utils import AgentSettings
from cover_letter_agent.agent import get_root_agent


load_dotenv()

session_service = InMemorySessionService()

APP_NAME = "Cover Letter Agent"
USER_ID = "slu"
LOGFILE_NAME = "sub_agents_output.log"

utils.setup_loggers(LOGFILE_NAME)


async def main_async(
    file_name: str,
    verbose: bool,
    agnt_settings: AgentSettings,
):
    """Main entry point for the cover letter agent."""

    plugins = [LoggingPlugin()] if verbose else None

    # Initialize the runner
    runner = Runner(
        agent=get_root_agent(agnt_settings),
        app_name=APP_NAME,
        session_service=session_service,
        plugins=plugins
    )

    # Create a new session
    new_session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID)
    session_id = new_session.id

    print("Welcome to the cover letter agent!\n")
    print("Please provide the following information:\n")

    # Use environment variables if provided, otherwise prompt the user
    # (for debugging purposes)
    company_url = os.getenv("COMPANY_URL")
    if not company_url:
        company_url = input("Company URL: ")
    else:
        print(f"Company URL: {company_url}")

    job_role_url = os.getenv("JOB_ROLE_URL")
    if not job_role_url:
        job_role_url = input("Job role URL: ")
    else:
        print(f"Job role URL: {job_role_url}")

    prompt = utils.get_prompt(company_url,
                              job_role_url,
                              file_name)

    print("\nProcessing your request...\n")
    # Process the user query through the agent
    agent_response = await utils.call_agent_async(
        runner,
        USER_ID,
        session_id,
        prompt,
    )

    print("\nTHE AGENT RESPONSE:\n")
    if isinstance(agent_response, str):
        agent_response = utils.load_json(agent_response)
    if agent_response:
        print(agent_response.get("status").upper(),":")
        print(agent_response.get("message"))
    else:
        print("No response from the agent! Check logs for details.")

    # Save the log file as `sub_agents_output_<company_domain>.log`
    utils.copy_log_file(LOGFILE_NAME, company_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cover Letter Agent")
    parser.add_argument("-f", "--file_name", type=str, required=True, help="Path to the PDF file")
    parser.add_argument("-v", "--verbose", default=False, action='store_true',
                        help="Enable verbose logging")
    parser.add_argument("-t", "--tavily", default=False, action='store_true',
                        help="Enable tavily advanced extraction")
    parser.add_argument("-l", "--language_level", type=str, default="b2",
                        choices=["b1", "b2", "c1", "c2"], help="Language level")
    parser.add_argument("-T", "--thinking_level", type=str, default="minimal",
                        choices=["minimal", "low", "medium", "high"], help="Gemini3 thinking level")
    parser.add_argument("-m", "--sa_model", type=str, default="gemini-2.5-flash-preview-09-2025",
                        help="Sub-agents model name")
    parser.add_argument("-M", "--ma_model", type=str, default="gemini-3-flash-preview",
                        help="Main agent model name")
    args = parser.parse_args()

    models = {
        "sub_agents_model": args.sa_model,
        "main_agent_model": args.ma_model
        }

    language_levels = {
        "b1": "Intermediate (B1)",
        "b2": "Upper-Intermediate (B2)",
        "c1": "Advanced (C1)",
        "c2": "Proficient (C2)" 
        }

    agent_settings = AgentSettings(
                    models=models,
                    g3_thinking_level=args.thinking_level,
                    language_level=language_levels[args.language_level],
                    tavily_advanced_extraction=args.tavily
                )

    # Set up and run the asynchronous main function using an event loop
    # This is necessary for the Google ADK to work properly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            main_async(
                args.file_name,
                args.verbose,
                agent_settings
                )
            )
    finally:
        loop.close()
