"""Streamlit application for the Cover Letter AI Agent."""

import asyncio
import os

import streamlit as st
import nest_asyncio
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.plugins.logging_plugin import LoggingPlugin

import utils
from cover_letter_agent.agent import get_root_agent


load_dotenv()
nest_asyncio.apply()

APP_NAME = "Cover Letter Agent"
USER_ID = "streamlit_user"

# Page configuration
st.set_page_config(
    page_title="Cover Letter AI Agent",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "fancy" look
st.html("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;        
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .description {
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
</style>
""")


async def run_agent(company_url, job_description_url, file_path, model_name):
    """Run the agent asynchronously."""
    session_service = InMemorySessionService()

    # Initialize the runner
    # We can add LoggingPlugin if we want to see logs in the console
    runner = Runner(
        agent=get_root_agent(model_name),
        app_name=APP_NAME,
        session_service=session_service,
        plugins=[LoggingPlugin()]
    )

    # Create a new session
    new_session = await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID)
    session_id = new_session.id

    prompt = f"""
    ### Company:
    {company_url}

    ### Job description:
    {job_description_url}
    """

    # Process the user query through the agent
    agent_response = await utils.call_agent_async(
        runner,
        USER_ID,
        session_id,
        prompt,
        file_path
    )

    return agent_response


def main():
    """Main entry point for the Streamlit app."""

    st.title("📝 Cover Letter AI Agent")
    st.divider()

    left, right = st.columns(
        2,
        gap="medium",
        vertical_alignment="top",
        border=True
    )

    company_url = left.text_input(
            "**Company Website URL**",
            placeholder="https://www.example.com"
        )

    job_description_url = left.text_input(
            "**Job Description URL**",
            placeholder="https://careers.example.com/job/123"
        )

    uploaded_file = left.file_uploader(
        "**Upload your CV (PDF, DOC)**",
        type=["pdf", "doc", "docx"]
    )

    # Advanced settings (collapsible)
    with left.expander("**Advanced Settings**"):
        model_name = st.text_input("Model Name", value="gemini-2.5-flash-preview-09-2025")

    if left.button("Generate Cover Letter"):
        if not company_url or not job_description_url or not uploaded_file:
            left.warning("Please fill in all fields and upload your CV.")
        else:
            with st.spinner(
                "Generating cover letter... This may take a minute."
            ):
                # Save the uploaded file
                try:
                    temp_file_path = utils.save_uploaded_file(uploaded_file)
                except Exception as e:
                    left.error(f"Error saving file: {e}")

                if temp_file_path:
                    try:
                        # Run the agent with manual loop management to ensure cleanup
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                        result = loop.run_until_complete(
                            run_agent(
                                company_url,
                                job_description_url,
                                temp_file_path,
                                model_name
                            )
                        )

                        left.success("Cover Letter Generated Successfully!", icon="✅")
                        # right.subheader("Your Cover Letter")
                        right.text_area(
                            "Cover Letter",
                            value=result,
                            label_visibility="collapsed",
                            height=500
                            )

                        right.markdown(":red[*Read carefully and make adjustments if needed.]")

                    except Exception as e:
                        left.error(f"An error occurred: {e}", icon="❌")
                    finally:
                        # Clean up the temporary file
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)

if __name__ == "__main__":
    main()
