# Cover Letter Agent (Google ADK)

An intelligent agentic workflow designed to generate tailored, professional cover letters. This project utilizes Google's ADK (Agent Development Kit) and Gemini models to research company information, parse your CV, and analyze job descriptions to craft the perfect cover letter.

## 🚀 Features

- **Automated Company Research**: Scours the web for company culture, values, mission, and vision to ensure the cover letter aligns with the company's ethos.
- **CV Parsing**: Extracts key details (Summary, Skills, Experience, Education) from your PDF CV.
- **Job Description Analysis**: Understands the requirements and nuances of the job posting.
- **Context-Aware Generation**: Synthesizes all gathered data to write a non-pretentious, value-focused cover letter.

## 🛠️ Architecture

The system is built using a **Sequential Agent** that orchestrates a **Parallel Research Team**:

1.  **Parallel Research Team** (Runs simultaneously):
    *   `company_web_researcher`: Uses Google Search to find company insights.
    *   `cv_parcer_agent`: Parses the uploaded PDF CV.
    *   `job_description_extractor_agent`: Extracts text from the job description URL.

2.  **Cover Letter Generator** (`cl_generator_agent`):
    *   Takes the outputs from the research team.
    *   Generates the final cover letter using a Gemini model (e.g., `gemini-2.5-flash-preview`).

## 📦 Requirements

- Python 3.x
- `google-adk`
- `google-genai`
- Access to Google Gemini API and Search tools.

## 🔧 Configuration

Ensure you have your environment variables set up, particularly for API keys required by the Google GenAI and Search tools.

## 🏃 Usage

Import the `root_agent` from `cover_letter_agent.agent` and run it with the necessary inputs (Company URL, Job URL, CV PDF).

```python
from cover_letter_agent.agent import root_agent

# Example usage (conceptual)
# Ensure you provide the inputs expected by the sub-agents
result = root_agent.invoke({
    "company_url": "https://example.com",
    "job_url": "https://example.com/jobs/123",
    "cv_pdf": "path/to/cv.pdf" 
})

print(result['cover_letter'])
```