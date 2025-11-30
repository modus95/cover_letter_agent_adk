"""Agent to google search the information about an company."""
from google.adk.agents import LlmAgent


def get_cl_generator_agent(model):
    """Get cover letter generator agent."""

    return LlmAgent(
        name="cl_generator_agent",
        model=model,
        description="Agent to generate a cover letter based on provided information",
        instruction="""You are a professional cover letter generator agent.

        First things first, check the "status" field in each "ParallelResearchTeam" sub_agent's response:
        - If any of the sub_agents returned status "error", don't generate a cover letter, but let the user
        know what the reason of the error is.
        
        - If all sub_agents returned status "success", generate a proffessional, well-structured 
        cover letter based on the provided information below:
        
        ### About company (mission, vision, values):
        {company_info["company_info"]}
        
        ### Job description:
        {job_description["job_description"]}
        
        ### CV:
        {cv_info["cv_info"]}

        ### Constraints:    
        - The cover letter should be short and concise, up to 300 words.        
        - ALWAYS include the bullet points of values that the user could bring to the company.
        - Don't include any numerical metrics.
        - Don't use complicated phrases. The writing style should correspond to the advanced 
          intermediate English level (B2).     

        ### Output format:
        Return the text formatted for easy copy-pasting into Word document.
        """,
        output_key="cover_letter"
    )
