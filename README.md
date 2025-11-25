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
    *   `web_researcher_agent`: Uses Google Search to find company insights.
    *   `cv_parcer_agent`: Parses the uploaded PDF CV.
    *   `job_description_agent`: Extracts text from the job description URL.

2.  **Cover Letter Generator** (`cl_generator_agent`):
    *   Takes the outputs from the research team.
    *   Generates the final cover letter using a Gemini model.

## 📦 Requirements

- Python 3.x
- `google-adk`
- `google-genai`
- `python-dotenv`
- Access to Google Gemini API and Search tools.

## 🔧 Configuration

1.  Clone the repository.
2.  Install dependencies (e.g., `pip install -r requirements.txt`).
3.  Create a `.env` file in the root directory and add your API keys:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```

## 🏃 Usage

Run the agent from the command line:

```bash
python main.py -f path/to/your_cv.pdf
```

### Arguments

- `-f`, `--file_name` (Required): Path to the PDF CV file.
- `-v`, `--verbose` (Optional): Enable verbose logging to see agent thoughts and actions.
- `-m`, `--model` (Optional): Specify the Gemini model to use. Default is `gemini-2.5-flash-preview-09-2025`.

### Example

```bash
python main.py -f ./my_cv.pdf --verbose --model gemini-1.5-pro
```

When running the script, you will be prompted to enter:
1.  **Company URL**: The website of the company you are applying to.
2.  **Job Description URL**: The link to the job posting.