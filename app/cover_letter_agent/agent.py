"""This module defines AI agents and models for generating cover letters, utilizing Google ADK."""
import logging
from google.adk.agents import ParallelAgent, SequentialAgent

import sub_agents.web_researcher.agent as res
import sub_agents.job_info.agent as jda
import sub_agents.cl_generator.agent as clg

from utils import define_model, get_planner, AgentSettings


status_logger = logging.getLogger("agent_status_logger")


def get_root_agent(agent_settings: AgentSettings):
    """
    Constructs and returns the root agent for the cover letter generation workflow.

    This function configures the necessary models and planners based on the provided
    agent settings, initializes sub-agents for web research, job analysis, and
    cover letter generation, and organizes them into a sequential execution pipeline.

    Args:
        agent_settings (AgentSettings): The configuration settings for the agents.

    Returns:
        SequentialAgent: The high-level agent orchestrating the cover letter generation process.
    """

    if isinstance(agent_settings.models, str):
        sa_model = ma_model = define_model(agent_settings.models)
    else:
        sa_model = define_model(agent_settings.models["sub_agents_model"])
        ma_model = define_model(agent_settings.models["main_agent_model"])

    sa_planner = get_planner(sa_model, agent_settings.g3_thinking_level)
    ma_planner = get_planner(ma_model, agent_settings.g3_thinking_level)

    # Logging the models, planners, and language level
    status_logger.info("Sub-agents models: %s", sa_model.model)
    if sa_planner:
        status_logger.info("Sub-agents thinking level: %s",
                           sa_planner.thinking_config.thinking_level)

    status_logger.info("Main agent model: %s", ma_model.model)
    if ma_planner:
        status_logger.info("Main agent thinking level: %s",
                           ma_planner.thinking_config.thinking_level)

    status_logger.info("Language level: %s", agent_settings.language_level)
    status_logger.info("Gemini3 thinking level: %s", agent_settings.g3_thinking_level)

    #SUB-AGENTS:
    web_researcher_agent = res.get_web_researcher_agent(sa_model, sa_planner)

    job_role_agent = jda.get_job_role_agent(sa_model,
                                            agent_settings.tavily_advanced_extraction,
                                            sa_planner
                                            )

    cl_generator_agent = clg.get_cl_generator_agent(ma_model,
                                                    agent_settings.language_level,
                                                    ma_planner
                                                    )

    # The ParallelAgent runs all its sub-agents simultaneously.
    parallel_research_team = ParallelAgent(
        name="ParallelResearchTeam",
        sub_agents=[
            web_researcher_agent,
            job_role_agent,
            ]
    )

    # This SequentialAgent defines the high-level workflow:
    # run the parallel team first, then run the aggregator (cover letter generator).
    ra = SequentialAgent(
        name="cover_letter_agent",
        sub_agents=[parallel_research_team, cl_generator_agent],
    )

    return ra


root_agent = get_root_agent(
    AgentSettings(
        models="gemini-2.5-flash",
        language_level="Intermediate (B1)",
        g3_thinking_level="minimal",
        tavily_advanced_extraction=False
        )
    )
