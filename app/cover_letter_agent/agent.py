"""This module defines AI agents and models for generating cover letters, utilizing Google ADK."""
from typing import Optional

from google.adk.agents import ParallelAgent, SequentialAgent

import sub_agents.web_researcher.agent as res
import sub_agents.job_description.agent as jda
import sub_agents.cl_generator.agent as clg

import vertex_utils as vu


agent_model_names = {
    "sub_agents_model": "gemini-2.5-flash",
    "main_agent_model": "gemini-2.5-flash"
}


def get_root_agent(models: Optional[str | dict],
                   tavily_advanced_extraction: bool = False):
    """
    Initializes and returns a root agent for cover letter generation.

    This function sets up various AI sub-agents (web researcher, cover letter writer,
    and a sequential agent to orchestrate them) using the specified model.
    It configures retry options for API calls.

    Args:
        models: A string representing the model name to be used by all agents, 
        or a dictionary specifying different models for sub-agents and the main agent 
        (e.g., `{"sub_agents_model": "model_name_1", "main_agent_model": "model_name_2"}`).

        tavily_advanced_extraction: Whether to use Tavily advanced extraction.

    Returns:
        A SequentialAgent that orchestrates the web research and cover letter
        writing process.
    """
    sa_model = models["sub_agents_model"]
    ma_model = models["main_agent_model"]

    sa_planner = vu.get_planner(sa_model)
    ma_planner = vu.get_planner(ma_model)

    #SUB-AGENTS:
    web_researcher_agent = res.get_web_researcher_agent(sa_model, sa_planner)
    job_description_agent = jda.get_job_description_agent_tavily(
                                            sa_model,
                                            tavily_advanced_extraction,
                                            sa_planner
                                            )

    cl_generator_agent = clg.get_cl_generator_agent(ma_model, ma_planner)

    # The ParallelAgent runs all its sub-agents simultaneously.
    parallel_research_team = ParallelAgent(
        name="ParallelResearchTeam",
        sub_agents=[web_researcher_agent, job_description_agent],
    )

    # This SequentialAgent defines the high-level workflow:
    # run the parallel team first, then run the aggregator (cover letter generator).
    ra = SequentialAgent(
        name="cover_letter_agent",
        sub_agents=[parallel_research_team, cl_generator_agent],
    )

    return ra


root_agent = get_root_agent(agent_model_names, tavily_advanced_extraction=False)
