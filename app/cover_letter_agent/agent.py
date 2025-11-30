"""This module defines AI agents and models for generating cover letters, utilizing Google ADK."""

from google.adk.agents import ParallelAgent, SequentialAgent
import sub_agents.web_researcher.agent as res
import sub_agents.cv_parcer.agent as cvpa
import sub_agents.job_description.agent as jda
import sub_agents.cl_generator.agent as clg

from utils import define_model


DEFAULT_MODEL_NAME = "gemini-2.5-flash-preview-09-2025"


def get_root_agent(model_name: str):
    """
    Initializes and returns a root agent for cover letter generation.

    This function sets up various AI sub-agents (web researcher, cover letter writer,
    and a sequential agent to orchestrate them) using the specified model.
    It configures retry options for API calls.

    Args:
        model_name: The name of the language model to be used by the agents.

    Returns:
        A SequentialAgent that orchestrates the web research and cover letter
        writing process.
    """

    model = define_model(model_name)

    #SUB-AGENTS:
    web_researcher_agent = res.get_web_researcher_agent(model)
    cv_parcer_agent = cvpa.get_cv_parcer_agent(model)

    # Tavily MCP extractor tool:jda.get_job_description_agent_tavily(model)
    # Currently doesn't work properly, responding with "MALFORMED_FUNCTION_CALL"
    # TO DO: make research and fix
    job_description_agent = jda.get_job_description_agent(model)
    cl_generator_agent = clg.get_cl_generator_agent(model)

    # The ParallelAgent runs all its sub-agents simultaneously.
    parallel_research_team = ParallelAgent(
        name="ParallelResearchTeam",
        sub_agents=[web_researcher_agent, job_description_agent, cv_parcer_agent],
    )

    # This SequentialAgent defines the high-level workflow:
    # run the parallel team first, then run the aggregator (cover letter generator).
    ra = SequentialAgent(
        name="cover_letter_agent",
        sub_agents=[parallel_research_team, cl_generator_agent],
    )

    return ra


root_agent = get_root_agent(DEFAULT_MODEL_NAME)
