"""This module defines AI agents and models for generating cover letters, utilizing Google ADK."""

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools import google_search
from google.adk.models import Gemini
from google.genai import types


def define_model(model_name:str, retry_options):
    """
    Initializes and returns a Gemini model instance.

    Args:
        model_name (str): The name of the Gemini model to instantiate.
        retry_options: Configuration for retrying HTTP requests to the model API.

    Returns:
        Gemini: An instance of the Gemini model configured with the specified retry options.
    """
    return Gemini(model=model_name, retry_options=retry_options)


# MODEL_NAME = "gemini-2.5-flash-lite"
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

RETRY_CONFIG = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)

model = define_model(MODEL_NAME, RETRY_CONFIG)

# 1. Web researcher agent -------------------------------

web_researcher_agent = LlmAgent(
    name="company_web_researcher",
    model=model,
    description="Agent to google search the information about an company",
    instruction=\
    """You are a web researcher agent. Your only job is to use the
    google_search tool to find information about a company, its culture, values,
    mission and vision based on provided company official website url.
    
    ### Output format:
    If you didn't manage to retreive any information about the company(e.g. uncorrect url)
    return JSON error response:
    {
        "status": "error",
        "error_message": "Unable to retrieve information about the company: <The error message>"
    }
    
    If you have successfully retrieved information about the company return JSON response:
    {
        "status": "success",
        "company_info": <Information about the company>
    }
    """,
    tools=[google_search],
    output_key="company_info"
)


# 2. PDF CV parser agent  -------------------------------
cv_parcer_agent = LlmAgent(
    name="cv_parcer_agent",
    model=model,
    description="Agent to parse CV information from a PDF file uploaded by the user",
    instruction="""You are a CV PDF parser agent.
    Your task is to parse (define the text content of) the uploaded PDF file to extract the following information:
        - Name        
        - Summary
        - Skills
        - Work Experience
        - Education

    ### Output format:
    If you didn't manage to parse uploaded file (e.g. uncorrect file, no access to the file):
    return JSON error response:
    {
        "status": "error",
        "error_message": "Unable to parse uploaded file: <The error message>"
    }
    
    If you have successfully parsed uploaded file return JSON response:
    {   
        "status": "success",
        "cv_info": <The extracted information>
    }   
    """,
    output_key="cv_info"
)

# 3. Job description extractor agent  -------------------------------
job_description_extractor_agent = LlmAgent(
    name="job_description_extractor_agent",
    model=model,
    description="Agent to extract job description text from provided website URL",
    instruction="""You are a job description extractor agent.
    Your task is to extract the job description text from the provided website URL.
    
    ### Output format:
    If you didn't manage to extract job description (e.g. uncorrect URL, no access to the URL):
    return JSON error response:
    {
        "status": "error",
        "error_message": "Unable to extract job description from provided URL: <The error message>"
    }
    
    If you have successfully extracted job description return JSON response:
    {
        "status": "success",
        "job_description": <The text of job description ONLY.
                            Don't include your thoughts, any additional information
                            or any other text>
    }   
    """,
    output_key="job_description"
)


# 4. Cover letter generator agent  -------------------------------
cl_generator_agent = LlmAgent(
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
    - Don't include any numerical metrics.     
    - ALWAYS include the bullet points of values that the user could bring to the company.
    - The style of the letter should NOT be pretentious and pathetic.    

    ### Output format:
    Return the text formatted for easy copy-pasting into Word document.
    """,
    output_key="cover_letter"
)

# The ParallelAgent runs all its sub-agents simultaneously.
parallel_research_team = ParallelAgent(
    name="ParallelResearchTeam",
    sub_agents=[web_researcher_agent, job_description_extractor_agent, cv_parcer_agent],
)

# This SequentialAgent defines the high-level workflow:
# run the parallel team first, then run the aggregator (cover letter generator).
root_agent = SequentialAgent(
    name="cover_letter_agent",
    sub_agents=[parallel_research_team, cl_generator_agent],
)
