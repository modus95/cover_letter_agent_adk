"""
This module defines the `cl_generator_agent` responsible for generating cover letters
based on provided company information, job description, and CV details.
It utilizes an LlmAgent to process inputs and adhere to specific constraints
for cover letter generation.
"""

from google.adk.agents import LlmAgent

try:
    from utils import ResponseContent
except ImportError:
    from app.utils import ResponseContent


def get_cl_generator_agent(model,
                           language_level,
                           planner=None) -> LlmAgent:
    """Get cover letter generator agent."""

    return LlmAgent(
        name="cl_generator_agent",
        model=model,
        planner=planner,
        description="Agent to generate a cover letter based on provided information",
        instruction=\
        f"""
        You are a professional cover letter generator agent.
        Yout task is to generate a proffessional, well-structured cover letter based on:
        - `company_web_researcher` sub-agent output: {{company_info}} 
        - `job_information_agent` sub-agent output: {{job_role_information}} 
        - Information about the user's skills and experience from the <User CV>.
             
        <Constraints>    
        - Keep the cover letter brief and concise, up to 300 words.
        - Don't include any numerical metrics.
        </Constraints>

        <Style>
        - Use English grammar and vocabulary appropriate to the {language_level} level. 
        - ALWAYS include the bullet points of values that the user could bring to the company.
        - Don't include any additional placeholders for date, subject line, company name, 
          company address, etc. in the beginning. 
        - Don't include any information about user's e-mail, phone number, job title, etc. 
          in the closing.
        </Style>

        <Output>
        **IMPORTANT:**
        Pay attention to the "status" field of a sub agents' responses:
        - If ALL "ParallelResearchTeam"'s sub agents have returned the "success" status, 
        then return the generated cover letter text in Markdown format with the "success" status. 
        - If ANY of the sub agents has returned the "error" status, don't generate a cover letter, 
        but return the clear reason of the failure with the "error" status.

        Your response MUST be valid JSON matching the `ResponseContent` structure:
        {{
            "status": "success" or "error",
            "message": "The generated cover letter if the status is 'success'. 
             The error message with the reason of the failure if the status is 'error'"
        }}

        DO NOT include any explanations or additional text outside the JSON response.
        </Output>
        """,
        output_schema=ResponseContent,
        output_key="cover_letter"
    )
