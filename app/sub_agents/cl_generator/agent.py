"""Agent to google search the information about an company."""
from google.adk.agents import LlmAgent

try:
    from utils import ResponseContent
except ImportError:
    from app.utils import ResponseContent


def get_cl_generator_agent(model):
    """Get cover letter generator agent."""

    return LlmAgent(
        name="cl_generator_agent",
        model=model,
        description="Agent to generate a cover letter based on provided information",
        instruction="""You are a professional cover letter generator agent.

        Yout task is to generate a proffessional, well-structured cover letter based on 
        the information provided by "ParallelResearchTeam"'s sub agents, and taking into account
        the important constraints:

        ### About company (mission, vision, values):
        {company_info}
        
        ### Job description:
        {job_description}
        
        ### CV:
        {cv_info}

        ### Constraints:    
        - The cover letter should be short and concise, up to 300 words.        
        - Start the letter with a greeting (e.g. "Dear ..."). Don't include any additional placeholders 
          for date, subject line, company name, company address, etc. at the very beginning of the letter.
        - ALWAYS include the bullet points of values that the user could bring to the company.
        - Don't include any numerical metrics.
        - Don't use complicated phrases. The writing style should correspond to the advanced 
          intermediate English level (B2).     
        - Don't include any information about user's e-mail, phone number, job title, etc. 
          in the closing.

        ### Output:
        **IMPORTANT:**
        - If ALL "ParallelResearchTeam"'s sub agents have returned the "success" status, then return the
        generated cover letter text in Markdown format with the "success" status. 
        - If ANY of the sub_agents has returned the "error" status, don't generate a cover letter, 
        but return the clear reason of the failure with the "error" status.

        Your response MUST be valid JSON matching the `ResponseContent` structure:
        {
            "status": "success" or "error",
            "message": "The generated cover letter if the status is 'success'. 
             The error message with the reason of the failure if the status is 'error'"
        }

        DO NOT include any explanations or additional text outside the JSON response.
        """,
        output_schema=ResponseContent,
        output_key="cover_letter"
    )
