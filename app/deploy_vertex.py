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


def _get_remote_app(agent_name: str):
    """Retrieves an existing agent engine by name."""
    existing_agents = list(agent_engines.list(filter=f'display_name={agent_name}'))

    if existing_agents:
        return existing_agents[0]
    return None


def create(**kwargs) -> agent_engines.AgentEngine:
    """Creates a new deployment or updates an existing one."""

    remote_app = kwargs["remote_app"]
    agent_name = kwargs["agent_name"]
    agent_config = {
        "agent_engine": remote_app,
        "display_name": agent_name,
        "description": "Agent to generate a cover letter based on provided information",
        "requirements": ["google-cloud-aiplatform[adk,agent_engines]",
                         "pydantic"],
        "extra_packages": ["./vertex_utils.py",
                           "./sub_agents",
                           "./cover_letter_agent"],
        "env_vars": {"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY")}
    }

    if remote_app is None:
        # Create a new agent engine (deployment)
        adk_app = agent_engines.AdkApp(
            agent=root_agent,
            enable_tracing=True,
        )
        agent_config["agent_engine"] = adk_app

        remote_agent = agent_engines.create(**agent_config)
        print("✅ Agent deployed successfully!")
        print(f"🆔 Agent Engine ID: {remote_agent.resource_name}")

    else:
        # Update the existing agent engine (deployment)
        remote_agent = remote_app.update(**agent_config)
        print("✅ Agent updated successfully!")
        print(f"🆔 Agent Engine ID: {remote_agent.resource_name}")

    return remote_agent


def delete(**kwargs) -> None:
    """Deletes an existing deployment."""

    resource_id = kwargs["id"]
    remote_app = agent_engines.get(resource_id)

    if remote_app is None:
        print(f"Remote deployment with ID `{resource_id}` not found.")
        return
    remote_app.delete(force=True)

    print(f"Remote app `{resource_id}` deleted successfully!")


def list_deployments(**kwargs) -> None:
    """Lists all deployments."""

    deployments = agent_engines.list()
    if not deployments:
        print("No deployments found.")
        return

    print("Deployments:")
    for deployment in deployments:
        print(f"- {deployment.resource_name}")


async def create_session(**kwargs):
    """Creates a new remote session."""
    remote_app = kwargs["remote_app"]
    agent_name = kwargs["agent_name"]
    user_id = kwargs["user_id"]

    if remote_app is None:
        print(f"Remote deployment `{agent_name}` not found.")
        return

    remote_session = await remote_app.async_create_session(user_id=user_id)
    session_id = remote_session['id']
    print("✅ Remote session created successfully!")
    print(f"🆔 Remote session ID: {session_id}")

    return remote_session


async def delete_session(**kwargs):
    """Deletes a remote session."""
    remote_app = kwargs["remote_app"]
    agent_name = kwargs["agent_name"]
    session_id = kwargs["id"]
    user_id = kwargs["user_id"]

    if remote_app is None:
        print(f"Remote deployment `{agent_name}` not found.")
        return

    await remote_app.async_delete_session(session_id=session_id,
                                          user_id=user_id)
    print(f"Deleted remote session: {session_id}")


async def list_sessions(**kwargs):
    """Lists all sessions."""
    remote_app = kwargs["remote_app"]
    agent_name = kwargs["agent_name"]
    user_id = kwargs["user_id"]

    if remote_app is None:
        print(f"Remote deployment `{agent_name}` not found.")
        return

    sessions = await remote_app.async_list_sessions(user_id=user_id)
    if not sessions:
        print("No sessions found.")
        return

    print("Sessions:")
    for session in sessions:
        print(f"- {session['id']}")


async def delete_all_sessions(**kwargs) -> None:
    """Deletes all sessions."""
    remote_app = kwargs["remote_app"]
    agent_name = kwargs["agent_name"]
    user_id = kwargs["user_id"]

    if remote_app is None:
        print(f"Remote deployment `{agent_name}` not found.")
        return

    fl = input(f"All remore sessions of the `{agent_name}` will be deleted. Continue? (Y/n)")
    if fl.lower() not in ["y", ""]:
        return

    sessions = await remote_app.async_list_sessions(user_id=user_id)
    if not sessions:
        print("No sessions found.")
        return

    for session in sessions:
        await remote_app.async_delete_session(session_id=session['id'], user_id=user_id)
        print(f"Session {session['id']} is deleted successfully!")


def main(args_):
    """Main function"""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION")
    bucket = os.getenv("GOOGLE_CLOUD_STAGING_BUCKET")
    agent_name = os.getenv("AGENT_NAME")
    user_id = os.getenv("USER_ID", "test_user")

    vertexai.init(
        project=project_id,
        location=location,
        staging_bucket=bucket,
    )

    if not agent_name:
        print("Please provide an agent name in the .env file.")
        return

    remote_app = _get_remote_app(agent_name)

    func = {
        "create": create,   
        "delete": delete,
        "list": list_deployments,
        "create_session": create_session,
        "delete_session": delete_session,
        "list_sessions": list_sessions,
        "delete_all_sessions": delete_all_sessions,
    }

    func[args_.mode](
        remote_app=remote_app,
        agent_name=agent_name,
        id=args_.id,
        user_id=user_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        required=True,
        choices=["create",
                 "delete",
                 "list",
                 "create_session",
                 "delete_session",
                 "list_sessions",
                 "delete_all_sessions"],
        help=("Mode to run the script in (create, delete, list, create_session,"
              " delete_session, list_sessions, delete_all_sessions)")
    )

    parser.add_argument(
        "-i",
        "--id",
        type=str,
        required=False,
        help="Resource ID/ session ID to delete")

    args = parser.parse_args()
    main(args)
