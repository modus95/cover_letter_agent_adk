# Cover Letter Agent (Google ADK)

## Project Overview
This project is an intelligent agentic workflow designed to generate professional, context-aware cover letters. It utilizes **Google's Agent Development Kit (ADK)** and **Gemini models** to orchestrate a team of AI agents that research company culture, parse user CVs, analyze job descriptions, and synthesize this information into a high-quality cover letter.

## Architecture
The system employs a **Sequential Agent** pattern that orchestrates a **Parallel Research Team**:

1.  **Input Analysis:** The system accepts a PDF CV and URLs for the target company and job description.
2.  **Parallel Research Team (Sub-Agents):**
    *   **`web_researcher_agent`:** Uses Google Search to gather insights on company culture, mission, and values.
    *   **`job_information_agent`:** Uses the Tavily API to extract detailed requirements from the job posting.
3.  **Synthesis & Generation:**
    *   **`cl_generator_agent`:** Aggregates the research data and CV content to generate the final cover letter using a specified Gemini model.

## Key Directories & Files

*   `app/`: Core application source code.
    *   `main.py`: CLI entry point for local execution.
    *   `streamlit_app.py`: Streamlit-based web user interface.
    *   `deploy.py`: Deployment script (likely for GCP).
    *   `cover_letter_agent/`: Main orchestration logic.
    *   `sub_agents/`: Directory containing specialized agents (`web_researcher`, `job_info`, `cl_generator`).
    *   `data/`: Directory for storing processed data or temporary files.
    *   `.env.example`: Template for environment variables.
*   `logs/`: Stores execution logs (e.g., `sub_agents_output_<domain>.log`) useful for debugging agent reasoning.
*   `cl_agent_uv.sh`: specific helper script for running the app with `uv`, handling branch switching for deployment.
*   `Dockerfile` & `run_docker.sh`: Docker configuration for containerized deployment.

## Setup & Installation

### Prerequisites
*   Python >= 3.12
*   API Keys:
    *   **Google Gemini API Key**
    *   **Tavily API Key** (for job description extraction)

### Installation
1.  **Environment Setup:**
    Create a `.env` file in the `app/` directory based on `.env.example`:
    ```env
    GOOGLE_GENAI_USE_VERTEXAI=False
    GOOGLE_API_KEY=<your_google_api_key>
    TAVILY_API_KEY=<your_tavily_api_key>
    ```

2.  **Dependencies:**
    Using `pip`:
    ```bash
    pip install -r app/requirements.txt
    ```
    Or using `uv` (recommended):
    ```bash
    uv sync
    ```

## Running the Application

### 1. Streamlit Web UI (Recommended)
The most feature-rich interface, allowing PDF uploads and model configuration.

**Standard run:**
```bash
streamlit run app/streamlit_app.py
```

**Using `uv` helper script:**
This script handles branch switching for local vs. remote modes.
```bash
./cl_agent_uv.sh        # Local mode (master branch)
./cl_agent_uv.sh -r     # Remote mode (switches to deploy_gcp branch)
```

### 2. CLI Mode
Useful for quick testing or automation.
```bash
python app/main.py -f path/to/your_cv.pdf [options]
```
**Options:**
*   `-v`: Verbose logging.
*   `-t`: Enable Tavily advanced extraction.
*   `-l <level>`: Language level (b1, b2, c1, c2).
*   `-T <level>`: Thinking level (minimal, low, medium, high).

### 3. Google ADK Web UI
Launch the agent utilizing the standard ADK web interface.
```bash
adk web
```

### 4. Docker
Build and run the containerized application.
```bash
# Build
docker build -t cl-agent-streamlit .

# Run (helper script available)
./run_docker.sh
# OR manually
docker run --name cl-agent --rm -it -p 8501:8501 -e GOOGLE_API_KEY=... -e TAVILY_API_KEY=... cl-agent-streamlit
```

## Development Conventions
*   **Agent Framework:** Built on Google ADK.
*   **Logging:** Check `logs/` for detailed agent outputs to understand the reasoning process.
*   **State Management:** Streamlit session state is used for the UI interaction flow.
*   **Testing:** Use the CLI mode for rapid feedback loops during development.
