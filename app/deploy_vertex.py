"""
Module for deploying and managing the Cover Letter Agent in 
the Google Cloud Vertex AI Agent Engine.

This script provides functionality to create, update, and retrieve agent engine 
deployments using the Vertex AI SDK, including managing remote sessions.
"""

import argparse
import asyncio
import inspect
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# pylint: disable=wrong-import-position
import vertexai
from vertexai import agent_engines
from cover_letter_agent.agent import root_agent
from dotenv import dotenv_values
from google.api_core.exceptions import FailedPrecondition, NotFound


config = dotenv_values(".env_remote")


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

    adk_app = agent_engines.AdkApp(
        agent=root_agent,
        enable_tracing=True,
    )

    agent_config = {
        "agent_engine": adk_app,
        "display_name": agent_name,
        "description": "Agent to generate a cover letter based on provided information",
        "requirements": ["google-cloud-aiplatform[adk,agent_engines]==1.128.0",
                         "cloudpickle",
                         "pydantic",
                         "tavily-python",
                         ],
        "extra_packages": ["./vertex_utils.py",
                           "./sub_agents",
                           "./cover_letter_agent"],
        "env_vars": {"TAVILY_API_KEY": config["TAVILY_API_KEY"]}
    }

    if remote_app is None:
        # Create a new agent engine (deployment)
        remote_agent = agent_engines.create(**agent_config)
        print("✅ Agent deployed successfully!")
        print(f"🆔 Agent Engine ID: {remote_agent.resource_name}")

    else:
        # Update the existing agent engine (deployment)
        remote_agent = remote_app.update(**agent_config)
        print("✅ Agent updated successfully!")
        print(f"🆔 Agent Engine ID: {remote_agent.resource_name}")

    return remote_agent


def list_deployments(**kwargs) -> None:
    # pylint: disable=W0613
    """Lists all deployments of the project."""

    deployments = agent_engines.list()
    if not deployments:
        print("⚠️: No deployments found.")
        return

    print("✅ Deployments:")
    for deployment in deployments:
        print(f"- {deployment.resource_name}")


def delete(**kwargs) -> None:
    """Deletes an existing deployment by the specific Resource ID."""

    if not kwargs["id"]:
        print("⚠️: Please provide a Resource ID.")
        return

    resource_id = kwargs["id"]
    try:
        remote_app = agent_engines.get(resource_id)
    except NotFound:
        print(f"⚠️: Remote app `{resource_id}` not found.")
        return

    remote_app.delete(force=True)
    print(f"✅ Remote app `{resource_id}` deleted successfully!")

# ----------SESSIONS----------

async def create_session(**kwargs):
    """Creates a new remote session for the user."""
    remote_app = kwargs["remote_app"]
    agent_name = kwargs["agent_name"]
    user_id = kwargs["user_id"]

    if remote_app is None:
        print(f"⚠️: Remote deployment `{agent_name}` not found.")
        return

    remote_session = await remote_app.async_create_session(user_id=user_id)
    session_id = remote_session['id']
    print(f"✅ Remote session for the `{user_id}` created successfully!")
    print(f"🆔 Remote session ID: {session_id}")

    return remote_session


async def delete_session(**kwargs):
    """Deletes a remote session of the user by the specific Session ID."""

    if not kwargs["id"]:
        print("⚠️: Please provide a Session ID.")
        return

    remote_app = kwargs["remote_app"]
    agent_name = kwargs["agent_name"]
    session_id = kwargs["id"]
    user_id = kwargs["user_id"]

    if remote_app is None:
        print(f"⚠️: Remote deployment `{agent_name}` not found.")
        return
    try:
        await remote_app.async_delete_session(session_id=session_id,
                                              user_id=user_id)
    except FailedPrecondition as e:
        print(f"⚠️: Failed to delete remote session {session_id} for the `{user_id}`")
        if "404 NOT_FOUND" in str(e):
            print(f"The session {session_id} not found.")
        else:
            print(e)
        return

    print(f"✅ Remote session {session_id} for the `{user_id}` deleted successfully!")


async def list_sessions(**kwargs):
    """Lists all remote sessions of the user."""
    remote_app = kwargs["remote_app"]
    agent_name = kwargs["agent_name"]
    user_id = kwargs["user_id"]

    if remote_app is None:
        print(f"⚠️: Remote deployment `{agent_name}` not found.")
        return

    resp = await remote_app.async_list_sessions(user_id=user_id)
    sessions = resp.get('sessions', [])
    if not sessions:
        print(f"⚠️: No sessions found for the `{user_id}`.")
        return

    print(f"✅ Remote sessions of the `{user_id}`:")
    for session in sessions:
        print("- ", session['id'])


async def delete_all_sessions(**kwargs) -> None:
    """Deletes all sessions."""
    remote_app = kwargs["remote_app"]
    agent_name = kwargs["agent_name"]
    user_id = kwargs["user_id"]

    if remote_app is None:
        print(f"⚠️: Remote deployment `{agent_name}` not found.")
        return

    fl = input(f"⚠️: All remote sessions of the `{agent_name}` for the `{user_id}`"
               " user will be deleted.\nContinue? [Y/n]")
    if fl.lower() not in ["y", ""]:
        return

    resp = await remote_app.async_list_sessions(user_id=user_id)
    sessions = resp.get('sessions', [])
    if not sessions:
        print(f"⚠️: No sessions found for the `{user_id}`.")
        return

    for session in sessions:
        await remote_app.async_delete_session(session_id=session['id'], user_id=user_id)
        print(f"✅ Session {session['id']} for the `{user_id}` deleted successfully!")


def main(args_):
    """Main function"""
    project_id = config.get("GOOGLE_CLOUD_PROJECT")
    location = config.get("GOOGLE_CLOUD_LOCATION")
    bucket = config.get("GOOGLE_CLOUD_STAGING_BUCKET")
    agent_name = config.get("AGENT_NAME")
    user_id = args_.user_id or config.get("USER_ID", "test_user")

    vertexai.init(
        project=project_id,
        location=location,
        staging_bucket=bucket,
    )

    if not agent_name:
        print("Please provide an agent name in the .env_remote file.")
        return

    remote_app = _get_remote_app(agent_name)

    func = {
        "create": create,   
        "delete": delete,
        "list": list_deployments,
        "create_session": create_session,
        "delete_session": delete_session,
        "list_sessions":  list_sessions,
        "delete_all_sessions": delete_all_sessions,
    }

    f = func[args_.mode]
    kwargs = {
        "remote_app": remote_app,
        "agent_name": agent_name,
        "id": args_.id,
        "user_id": user_id
    }

    if inspect.iscoroutinefunction(f):
        asyncio.run(f(**kwargs))
    else:
        f(**kwargs)


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

    parser.add_argument(
        "-u",
        "--user_id",
        type=str,
        required=False,
        help="User ID to delete")

    args = parser.parse_args()
    main(args)
