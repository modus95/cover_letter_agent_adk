"""
Main entry point for the Vertex AI Cover Letter Agent.

This module initializes the Vertex AI environment and provides the main execution
logic for interacting with the cover letter generation agent deployed in the
Google Cloud's Vertex AI Agent Engine.
"""

import asyncio
import argparse
import logging

import vertexai
from vertexai import agent_engines
import utils
from dotenv import dotenv_values


config = dotenv_values(".env_remote")

project_id = config.get("GOOGLE_CLOUD_PROJECT")
location = config.get("GOOGLE_CLOUD_LOCATION")
bucket = config.get("GOOGLE_CLOUD_STAGING_BUCKET")
agent_name = config.get("AGENT_NAME")
user_id = config.get("USER_ID")

vertexai.init(project=project_id,
              location=location,
              staging_bucket=bucket)


async def main_async(file_name: str):
    """Main entry point for the cover letter agent."""

    existing_agents = list(agent_engines.list(filter=f'display_name={agent_name}'))
    if existing_agents:
        remote_agent = existing_agents[0]

        print("Welcome to the cover letter agent!\n")
        print("Please provide the following information:\n")
        prompt = utils.get_prompt(file_name)

        print("\nProcessing your request...\n")
        # Process the user query through the agent
        agent_response = await utils.call_remote_agent_async(
            remote_agent,
            project_id,
            user_id,
            prompt
        )

        print("\nTHE AGENT RESPONSE:\n")
        if isinstance(agent_response, str):
            agent_response = utils.load_json(agent_response)
        if agent_response:
            print(agent_response["status"].upper(),":")
            print(agent_response["message"])
        else:
            print("No response from the agent! Check the session details at Vertex AI.")

    else:
        print(f"Vertex AI remote agent '{agent_name}' not found!")
        return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Cover Letter Agent")
    parser.add_argument("-f", "--file_name", type=str, required=True, help="Path to the PDF file")
    args = parser.parse_args()

    # Set up and run the asynchronous main function using an event loop
    # This is necessary for the Google ADK to work properly
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(
            main_async(args.file_name)
        )
    finally:
        loop.close()
