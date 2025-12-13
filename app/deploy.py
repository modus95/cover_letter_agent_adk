"""
This script handles the deployment and execution of a Vertex AI agent,
specifically a cover letter generation agent. It sets up the necessary
environment variables, initializes Vertex AI, and orchestrates the
interaction with the agent engine.
"""

import os
import argparse

import vertexai
from vertexai import agent_engines
from cover_letter_agent.agent import root_agent
from dotenv import load_dotenv


load_dotenv()


def create(**kwargs) -> None:
    """Creates a new deployment."""
    if not kwargs["agent_name"]:
        print("Please provide an agent name.")
        return

    agent_name = kwargs["agent_name"]

    # Wrap the agent in AdkApp
    adk_app = agent_engines.AdkApp(
        agent=root_agent,
        enable_tracing=True,
    )

    # Deploy to Agent Engine
    remote_agent = agent_engines.create(
        agent_engine=adk_app,
        display_name=agent_name,
        description="Agent to generate a cover letter based on provided information",
        requirements=[
            "google-cloud-aiplatform[adk,agent_engines]",
            "pydantic"
        ],
        extra_packages=[
            "./vertex_utils.py",
            "./sub_agents",
            "./cover_letter_agent",
        ],
        env_vars={
            "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
        }
    )
    print("✅ Agent deployed successfully!")
    print(f"🆔 Agent Engine ID: {remote_agent.resource_name}")


def delete(**kwargs) -> None:
    """Deletes an existing deployment."""

    if not kwargs["resource_id"]:
        print("Please provide a resource ID.")
        return

    resource_id = kwargs["resource_id"]
    remote_app = agent_engines.get(resource_id)
    remote_app.delete(force=True)

    print(f"Deleted remote app: {resource_id}")


def list_deployments(**kwargs) -> None:
    """Lists all deployments."""
    if not kwargs["agent_name"]:
        print("Please provide an agent name.")
        return

    deployments = agent_engines.list()
    if not deployments:
        print("No deployments found.")
        return

    print("Deployments:")
    for deployment in deployments:
        print(f"- {deployment.resource_name}")


def main(args_):
    """Main function"""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    bucket = os.getenv("GOOGLE_CLOUD_STAGING_BUCKET")
    agent_name = os.getenv("AGENT_NAME")

    vertexai.init(
        project=project_id,
        location=location,
        staging_bucket=bucket,
    )

    func = {
        "create": create,   
        "delete": delete,
        "list": list_deployments
    }

    func[args_.mode](
        agent_name=agent_name,
        resource_id=args_.resource_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        required=True,
        choices=["create", "delete", "list"],
        help="Mode to run the script in (create, delete, list)"
    )

    parser.add_argument(
        "-r",
        "--resource_id",
        type=str,
        required=False,
        help="Resource ID of the deployment to delete")

    args = parser.parse_args()
    main(args)
