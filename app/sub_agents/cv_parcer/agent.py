"""Agent to parse CV information from a PDF file uploaded by the user"""

from google.adk.agents import LlmAgent

try:
    from utils import ResponseContent, logging_agent_output_status
except ImportError:
    from app.utils import ResponseContent, logging_agent_output_status


def get_cv_parcer_agent(model):
    """Get CV parcer agent."""

    return LlmAgent(
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

        If you have successfully parsed the uploaded file, return the extracted information in Markdown format with the
        "success" status. Otherwise, return the error message with the "error" status.

        IMPORTANT: Your response MUST be valid JSON matching the `ResponseContent` structure:
        {
            "status": "success" or "error",
            "message": "The main content of the agent response if the status is 'success'. The error message if the status is 'error'"
        }

        DO NOT include any explanations or additional text outside the JSON response.
        """,
        output_schema=ResponseContent,
        output_key="cv_info",
        after_agent_callback=logging_agent_output_status
    )
