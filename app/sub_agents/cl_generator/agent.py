"""
This module defines the `cl_generator_agent` responsible for generating cover letters
based on provided company information, job description, and user CV details.
It utilizes an LlmAgent to process inputs and adhere to specific constraints
for cover letter generation.
"""

from google.adk.agents import LlmAgent
from google.genai import types
from google.adk.tools import url_context, google_search
from google.adk.tools.google_search_tool import GoogleSearchTool

try:
    from utils import logging_tool_output_status
except ImportError:
    from app.utils import logging_tool_output_status



def get_cl_generator_agent(model,
                           language_level,
                           planner=None,
                           top_p: float = None) -> LlmAgent:
    """Get cover letter generator agent."""
    # pylint: disable=line-too-long

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
    selected_english_level_instruction = english_level_instructions[language_level]

    agent_instruction = f"""
Write a highly professional, engaging, and tailored cover letter for a job application.

- Use the `google_search` tool to search for the company at the provided company's URL. Provide a comprehensive overview of their mission, core values, company culture, and what they do. 
Return ONLY the requested information, do not include any conversational filler, introductory phrases, source links, or concluding remarks (e.g., do not say "Here is a comprehensive overview").

- Use the `url_context` tool to extract the full job description, requirements, and responsibilities for the job posting at the provided job role URL. Format it clearly.


<Instructions>
1. Tailor the cover letter specifically to the company's mission and culture.
2. Keep it concise, engaging, and professional (up to 300 words).
3. Highlight the most relevant skills and experiences from the provided candidate's CV that match the job description. Do not invent experiences that are not in the CV.
4. Highlight additional values the candidate could bring to the company based on his expertise in bullet points.
5. Don't include any numerical metrics from the CV.
6. Provide the cover letter text clearly. Don't add any personal information (e.g. e-mail, phone number, etc.) at the header or the footer.
7. Write the cover letter at CEFR English level {language_level[-3:-1]}.:
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

    return LlmAgent(
        name="cl_generator_agent",
        description="Agent to generate a tailored cover letter for a job application",
        model=model,
        planner=planner,
        generate_content_config=types.GenerateContentConfig(top_p=top_p),
        static_instruction="You are an expert career coach and professional copywriter.",
        instruction=agent_instruction,
        tools=[url_context, google_search],
        after_tool_callback=logging_tool_output_status,
        # output_schema=ResponseContent
    )


# def get_cl_generator_agent(model,
#                            language_level,
#                            planner=None,
#                            top_p: float = None) -> LlmAgent:
#     """Get cover letter generator agent."""

#     return LlmAgent(
#         name="cl_generator_agent",
#         model=model,
#         planner=planner,
#         generate_content_config=types.GenerateContentConfig(top_p=top_p),
#         description="Agent to generate a cover letter based on provided information",
#         instruction=\
#         f"""
#         You are a professional cover letter generator agent.
#         Your task is to generate a professional, well-structured cover letter based on:
#         - `company_web_researcher` sub-agent output: {{company_info}}
#         - `job_information_agent` sub-agent output: {{job_role_information}}
#         - Information about the user's skills and experience from the <User CV>.
        
#         <Style>
#         - Use English grammar and vocabulary appropriate to the {language_level} level.
#         - ALWAYS include the bullet points of values that the user could bring to the company.
#         - Don't include any additional placeholders for date, subject line, company name, 
#           company address, etc. in the beginning. 
#         - Don't include any information about user's e-mail, phone number, job title, etc. 
#           in the closing.
#         </Style>

#         <Constraints>    
#         - Keep the cover letter brief and concise, up to 300 words.
#         - The bullet points of values should be based on the user's skills and experience and 
#           meet the job requirements.  
#         - Don't include any numerical metrics.
#         </Constraints>

#         <Output>
#         **IMPORTANT:**
#         Pay attention to the "status" field of a sub agents' responses:
#         - If ALL "ParallelResearchTeam"'s sub agents have returned the "success" status, 
#         then return the generated cover letter text in Markdown format with the "success" status. 
#         - If ANY of the sub agents has returned the "error" status, don't generate a cover letter, 
#         but return the clear reason of the failure with the "error" status.

#         Your response MUST be valid JSON matching the `ResponseContent` structure:
#         {{
#             "status": "success" or "error",
#             "message": "The generated cover letter if the status is 'success'.
#              The error message with the reason of the failure if the status is 'error'"
#         }}

#         DO NOT include any explanations or additional text outside the JSON response.
#         </Output>
#         """,
#         output_schema=ResponseContent,
#         output_key="cover_letter"
#     )
