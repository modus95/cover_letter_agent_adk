"""This module defines AI agents and models for generating cover letters, utilizing Google ADK."""
import logging

from google.adk.tools.agent_tool import AgentTool
from google.adk.agents import LlmAgent
from google.adk.tools import url_context, google_search

from utils import AgentSettings, get_planner, logging_tool_output_status


status_logger = logging.getLogger("agent_status_logger")


def logging_agent_settings(agent_settings: AgentSettings, planner) -> None:
    """
    Logs the agent settings for debugging and monitoring purposes.

    Args:
        agent_settings (AgentSettings): The configuration settings for the agents.
        planner (BuiltInPlanner): The planner for the agents.
    """
    status_logger.info("Agent model: %s", agent_settings.model)
    if getattr(planner.thinking_config, "thinking_level", None):
        status_logger.info("Agent thinking level: %s",
                           planner.thinking_config.thinking_level)
    else:
        status_logger.info("Agent thinking budget: %s",
                            planner.thinking_config.thinking_budget)

    status_logger.info("Language level: %s", agent_settings.language_level)
    status_logger.info("Gemini3 thinking level: %s", agent_settings.g3_thinking_level)


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
    # pylint: disable=line-too-long

    agent_planner = get_planner(agent_settings)
    logging_agent_settings(agent_settings, agent_planner)

    english_level_instructions = {
        "Intermediate (B1)": (
            "Use clear and straightforward grammar and vocabulary suitable for B1 level. "
            "Prefer short to medium-length sentences, common professional phrases, and simple clause structures."
        ),
        "Upper-Intermediate (B2)": (
            "Use confident grammar and a broader professional vocabulary suitable for B2 level. "
            "Vary sentence patterns with mostly clear complex sentences while keeping wording natural and direct."
        ),
        "Advanced (C1)": (
            "Use advanced and precise grammar and vocabulary suitable for C1 level. "
            "Employ nuanced phrasing and varied sentence structures with strong coherence and polished professional tone."
        )
    }
    selected_english_level_instruction = english_level_instructions[agent_settings.language_level]

    agent_instruction = f"""
Write a highly professional, engaging, and tailored cover letter for a job application.

- Use the 'SearchAgent' agent tool to search for the company at the provided company's URL.
- Use the 'UrlContextAgent' agent tool to extract detailed job description from the provided job role URL.

<Instructions>
1. Tailor the cover letter specifically to the company's mission and culture.
2. Keep it concise, engaging, and professional (up to 300 words).
3. Highlight the most relevant skills and experiences from the provided candidate's CV that match the job description. Do not invent experiences that are not in the CV.
4. Highlight additional values the candidate could bring to the company based on his expertise in bullet points.
5. Don't include any numerical metrics from the CV.
6. Provide the cover letter text clearly. Don't add any personal information (e.g. e-mail, phone number, etc.) at the header or the footer.
7. Write the cover letter at CEFR English level {agent_settings.language_level[-3:-1]}.:
   {selected_english_level_instruction}
8. Present your final answer as a JSON string matching the following schema. Do not include any markdown fences (like ```json), just the raw JSON object string.
 - Success (the cover letter is generated successfully):
 {{
    "status": "success",
    "message": "<generated cover letter text>"
}}
- Error (e.g. if the company information or job description cannot be retrieved, or if the generation fails for any reason):
 {{
    "status": "error",
    "message": "<a clear reason of the failure>"
}}
</Instructions>
"""

    search_agent = LlmAgent(
        name='SearchAgent',
        model=agent_settings.model,
        static_instruction="You're a professional in Google Search, specializing in extracting relevant information about companies from their websites.",
        instruction="""
Provide a comprehensive overview of the company's mission, core values, company culture, and what they do.
Return ONLY the requested information, do not include any conversational filler, introductory phrases, source links, or concluding remarks (e.g., do not say "Here is a comprehensive overview").
""",
        tools=[google_search]
        )

    url_context_agent = LlmAgent(
        name='UrlContextAgent',
        model=agent_settings.model,
        static_instruction="You're a specialist in URL Context, with expertise in extracting detailed information from web pages, particularly job descriptions.",
        instruction="""Extract the full job description, requirements, and responsibilities for the job posting at the provided job role URL. Format it clearly.""",
    tools=[url_context]
    )

    return LlmAgent(
        name="cl_generator_agent",
        description="Agent to generate a tailored cover letter for a job application",
        model=agent_settings.model,
        planner=agent_planner,
        static_instruction="You are an expert career coach and professional copywriter.",
        instruction=agent_instruction,
        tools=[AgentTool(agent=search_agent), AgentTool(agent=url_context_agent)],
        after_tool_callback=logging_tool_output_status,
        # output_schema=ResponseContent
    )


root_agent = get_root_agent(
    AgentSettings(
        model="gemini-3.1-flash-lite",
        language_level="Intermediate (B1)",
        g3_thinking_level="minimal",
        )
    )
