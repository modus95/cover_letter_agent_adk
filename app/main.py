"""This module defines the main entry point for the cover letter agent."""

import asyncio
import argparse
import os

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.plugins.logging_plugin import LoggingPlugin

import utils
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
    tavily: bool,
    ma_model_name: str,
    sa_model_name: str
):
    """Main entry point for the cover letter agent."""

    plugins = [LoggingPlugin()] if verbose else None

    models = {
        "sub_agents_model": sa_model_name,
        "main_agent_model": ma_model_name
    }

    # Initialize the runner
    runner = Runner(
        agent=get_root_agent(models, tavily),
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

    job_description_url = os.getenv("JOB_DESCRIPTION_URL")
    if not job_description_url:
        job_description_url = input("Job description URL: ")
    else:
        print(f"Job description URL: {job_description_url}")

    prompt = f"""
    <company url>
    {company_url}
    </company>

    <job description url>
    {job_description_url}
    </job>
    
    """

    print("\nProcessing your request...\n")
    # Process the user query through the agent
    agent_response = await utils.call_agent_async(
        runner,
        USER_ID,
        session_id,
        prompt,
        file_name
    )

    print("\nTHE AGENT RESPONSE:\n")
    if isinstance(agent_response, str):
        agent_response = utils.load_json(agent_response)
    print(agent_response["status"].upper(),":")
    print(agent_response["message"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cover Letter Agent")
    parser.add_argument("-f", "--file_name", type=str, required=True, help="Path to the PDF file")
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action='store_true',
        help="Enable verbose logging"
        )
    parser.add_argument(
        "-t",
        "--tavily",
        default=False,
        action='store_true',
        help="Enable tavily advanced extraction"
        )
    parser.add_argument(
        "-m",
        "--sa_model",
        type=str,
        default="gemini-2.5-flash-preview-09-2025",
        help="Sub-agents model name"
    )
    parser.add_argument(
        "-M",
        "--ma_model",
        type=str,
        default="gemini-3-pro-preview",
        help="Main agent model name"
    )
    args = parser.parse_args()

    # Set up and run the asynchronous main function using an event loop
    # This is necessary for the Google ADK to work properly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            main_async(
                args.file_name,
                args.verbose,
                args.tavily,
                args.ma_model,
                args.sa_model
            )
        )
    finally:
        loop.close()
