"""This module defines the main entry point for the cover letter agent."""

import asyncio
import argparse

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from utils import call_agent_async
from cover_letter_agent.agent import root_agent


load_dotenv()

session_service = InMemorySessionService()

APP_NAME = "Cover Letter Agent"
USER_ID = "slu"


async def main_async(file_name: str):
    """Main entry point for the cover letter agent."""

    # Initialize the runner
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
        )

    # Create a new session
    new_session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID)
    session_id = new_session.id

    print("Welcome to the cover letter agent!\n")
    print("Please provide the following information:\n")

    # file_name = "./data/Sergei_Lungullo_DS_m2025.pdf"  #TO DO: make it dynamic
    company_url = input("Company URL: ")
    job_description_url = input("Job description URL: ")

    prompt = f"""
    ### Company
    {company_url}

    ### Job description
    {job_description_url}    
    """

    print("\nProcessing your request...\n")
    # Process the user query through the agent
    agent_response = await call_agent_async(
        runner,
        USER_ID,
        session_id,
        prompt,
        file_name
    )

    print("THE AGENT RESPONSE:\n")
    print(agent_response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cover Letter Agent")
    parser.add_argument("-f", "--file_name", type=str, required=True, help="Path to the PDF file")
    args = parser.parse_args()

    asyncio.run(main_async(args.file_name))
