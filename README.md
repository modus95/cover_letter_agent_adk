# Cover Letter Agent (Google ADK)

An intelligent agentic workflow designed to generate tailored, professional cover letters. This project utilizes **Google's ADK (Agent Development Kit)** and **Gemini models** to research company information, parse your CV, and analyze job descriptions to craft the perfect cover letter.

## 🚀 Features

- **Automated Company Research**: Scours the web for company culture, values, mission, and vision using **Google Search** and **Tavily** (optional advanced extraction).
- **CV Parsing**: Extracts key details (Summary, Skills, Experience, Education) from your PDF CV using the `cv_parser_agent`.
- **Job Description Analysis**: Understands the requirements and nuances of the job posting.
- **Context-Aware Generation**: Synthesizes all gathered data to write a non-pretentious, value-focused cover letter.
- **Multi-Model Support**: Choose different Gemini models for sub-agents and the main generator.

## 📂 Project Structure

The project code is organized within the `app/` directory:

```
app/
├── cover_letter_agent/  # Main agent logic and orchestration
├── sub_agents/          # Individual specialized agents (researcher, parser, etc.)
├── main.py              # CLI entry point for local execution
├── streamlit_app.py     # Streamlit web application
├── utils.py             # Utility functions
└── .env                 # Configuration file
```

## 🛠️ Architecture

The system is built using a **Sequential Agent** that orchestrates a **Parallel Research Team**:

1.  **Parallel Research Team** (Runs simultaneously):
    *   `web_researcher_agent`: Uses Google Search (or Tavily) to find company insights.
    *   `cv_parser_agent`: Parses the uploaded PDF CV to extract professional details.
    *   `job_description_agent`: Extracts and analyzes text from the job description URL.

2.  **Cover Letter Generator** (`cl_generator_agent`):
    *   Takes the aggregated outputs from the research team.
    *   Generates the final cover letter using a Gemini model.

## 📦 Requirements

- Python 3.x
- `google-adk`
- `google-genai`
- `streamlit == 1.51.0`
- `python-dotenv`
- Access to Google Gemini API and Search tools.

## 🔧 Configuration

1.  Clone the repository.
2.  Install dependencies.
3.  Create a `.env` file in the `app/` directory (or root) and add your API keys:
    ```env
    GOOGLE_API_KEY=your_api_key_here
    GOOGLE_CLOUD_PROJECT=your_project_id
    GOOGLE_CLOUD_LOCATION=your_location
    AGENT_NAME=your_agent_name
    # Optional
    TAVILY_API_KEY=your_tavily_key
    ```

## 🏃 Usage

You can run the agent in three different ways depending on your preference.

### 1. Streamlit Web Application

The most user-friendly way to interact with the agent. Provides a graphical interface for uploading your CV and entering URLs.

```bash
streamlit run app/streamlit_app.py
```

**Features:**
- Sidebar for selecting **Sub-agents model** and **Main agent model** (e.g., `gemini-2.5-flash-preview`).
- Toggle for **Tavily Advanced Extraction**.
- Real-time status updates and logging toggle.
- Copy-to-clipboard functionality for the generated letter.

### 2. CLI (Command Line Interface)

Run the agent directly from the terminal using `app/main.py`. This method is useful for quick tests or automation.

```bash
python app/main.py -f path/to/your_cv.pdf [options]
```

#### Arguments

| Argument | Long Flag | Default | Description |
| :--- | :--- | :--- | :--- |
| `-f` | `--file_name` | **Required** | Path to the PDF CV file. |
| `-v` | `--verbose` | `False` | Enable verbose logging to see detailed agent thoughts/actions. |
| `-t` | `--tavily` | `False` | Enable Tavily advanced extraction for web research. |
| `-m` | `--sa_model` | `gemini-2.5-flash-preview-09-2025` | Model name used by sub-agents (researcher, parser, etc.). |
| `-M` | `--ma_model` | `gemini-3-pro-preview` | Model name used by the main agent for final generation. |

#### Example

```bash
python app/main.py -f ./my_cv.pdf --verbose --tavily --ma_model gemini-2.5-pro
```

*Note: You will be prompted to enter the Company URL and Job Description URL after the script starts if they are not set in environment variables.*

### 3. Google ADK Web UI

Launch the agent using the Google Agent Development Kit's standard web interface.

```bash
adk web
```