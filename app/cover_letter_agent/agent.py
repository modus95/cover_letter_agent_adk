"""
This module provides the orchestration logic for the cover letter generation agent.
It defines the `get_root_agent` function, which constructs a multi-agent system
comprising a web researcher, a job role analyzer, and a cover letter generator.

In this module, a user can set the key parameters of the agent to be deployed 
in the Vertex AI Agent Engine (see the Key parameters section below). 
After setting the parameters, the agent deployment in the Vertex AI Agent Engine 
should be updated (`python3 deploy_vertex.py -m create`)
"""


import json
import pathlib
from typing import Optional

from google.adk.agents import ParallelAgent, SequentialAgent

import sub_agents.web_researcher.agent as res
import sub_agents.job_info.agent as jda
import sub_agents.cl_generator.agent as clg

import vertex_utils as vu


def get_root_agent(models: Optional[str | dict],
                   g3_thinking_level: str,
                   language_level: str,
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

        language_level: The language level (B1-C2) to be used by the main agent for 
        cover letter generation.
        thinking_level: The thinking level of Gemini3 ("minimal", "low", "medium", "high").
        For cover letter generation, it is recommended to use "minimal" or "low".
        tavily_advanced_extraction: Whether to use Tavily advanced extraction.

    Returns:
        A SequentialAgent that orchestrates the web research and cover letter
        writing process.
    """
    if isinstance(models, dict):
        sa_model = models["sub_agents_model"]
        ma_model = models["main_agent_model"]
    else:
        sa_model = ma_model = models

    sa_planner = vu.get_planner(sa_model, g3_thinking_level)
    ma_planner = vu.get_planner(ma_model, g3_thinking_level)

    #SUB-AGENTS:
    web_researcher_agent = res.get_web_researcher_agent(sa_model, sa_planner)
    job_role_agent = jda.get_job_role_agent(sa_model,
                                            tavily_advanced_extraction,
                                            sa_planner)

    cl_generator_agent = clg.get_cl_generator_agent(ma_model,
                                                    language_level,
                                                    ma_planner)

    # The ParallelAgent runs all its sub-agents simultaneously.
    parallel_research_team = ParallelAgent(
        name="ParallelResearchTeam",
        sub_agents=[
            web_researcher_agent,
            job_role_agent
        ],
    )

    # This SequentialAgent defines the high-level workflow:
    # run the parallel team first, then run the aggregator (cover letter generator).
    ra = SequentialAgent(
        name="cover_letter_agent",
        sub_agents=[parallel_research_team, cl_generator_agent],
    )

    return ra

# -----------------------------------------------------
# Load configuration from `config.json`
_CONFIG_PATH = pathlib.Path(__file__).parent.parent / "config.json"
with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _config_data = json.load(_f)

MODELS = _config_data["MODELS"]
G3_THINK = _config_data["G3_THINK"]
LANG_LEVEL = _config_data["LANG_LEVEL"]
TAVILY_ADVANCE = _config_data["TAVILY_ADVANCE"]


root_agent = get_root_agent(models=MODELS,
                            g3_thinking_level=G3_THINK,
                            language_level=LANG_LEVEL,
                            tavily_advanced_extraction=TAVILY_ADVANCE
                            )
