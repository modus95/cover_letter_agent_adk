"""
This module defines the `cl_generator_agent` responsible for generating cover letters
based on provided company information, job description, and CV details.
It utilizes an LlmAgent to process inputs and adhere to specific constraints
for cover letter generation.
"""

from google.adk.agents import LlmAgent

try:
    from vertex_utils import ResponseContent
except ImportError:
    from app.vertex_utils import ResponseContent


def get_cl_generator_agent(model, planner=None) -> LlmAgent:
    """Get cover letter generator agent."""

    return LlmAgent(
        name="cl_generator_agent",
        model=model,
        planner=planner,
        description="Agent to generate a cover letter based on provided information",
        instruction=\
        """You are a professional cover letter generator agent.

        Yout task is to generate a proffessional, well-structured cover letter based on 
        the information provided by "ParallelResearchTeam"'s sub agents, and taking into account
        the constraints below.

        <About company (mission, vision, values)>
        {company_info}
        </About company (mission, vision, values)>        
        
        <Job description>
        {job_description}
        </Job description>
        
        <CV>
        {cv_info}
        </CV>
        
        <Constraints>    
        - The cover letter should be short and concise, up to 300 words.        
        - ALWAYS include the bullet points of values that the user could bring to the company.
        - Don't include any numerical metrics.
        - Don't use complicated phrases. The writing style should correspond to the advanced 
          intermediate English level (B2). 
        - Don't include any additional placeholders for date, subject line, company name, 
          company address, etc. at the very beginning of the letter.    
        - Don't include any information about user's e-mail, phone number, job title, etc. 
          in the closing.
        </Constraints>

        <Output>
        **IMPORTANT:**
        Pay attention to the "status" field of a sub agents' responses:
        - If ALL "ParallelResearchTeam"'s sub agents have returned the "success" status, 
        then return the generated cover letter text in Markdown format with the "success" status. 
        - If ANY of the sub agents has returned the "error" status, don't generate a cover letter, 
        but return the clear reason of the failure with the "error" status.

        Your response MUST be valid JSON matching the `ResponseContent` structure:
        {
            "status": "success" or "error",
            "message": "The generated cover letter if the status is 'success'. 
             The error message with the reason of the failure if the status is 'error'"
        }

        DO NOT include any explanations or additional text outside the JSON response.
        </Output>
        """,
        output_schema=ResponseContent,
        output_key="cover_letter"
    )
