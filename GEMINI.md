# Cover Letter Agent (Google ADK)

## Project Overview
An intelligent agentic workflow designed to generate tailored, professional cover letters. This project utilizes **Google's ADK (Agent Development Kit)** and **Gemini models** to research company information, parse your CV, and analyze job descriptions to craft the perfect cover letter.

## Architecture
1. The app reads the uploaded CV and builds a prompt containing the company URL, job URL, and CV text.
2. The root `LlmAgent` (`cl_generator_agent`) uses two tools:
   - `SearchAgent` with `google_search` to learn about the company.
   - `UrlContextAgent` with `url_context` to extract the job description.
3. The agent returns a JSON string with the final result:
   - `status: "success"` with the generated letter in `message`
   - `status: "error"` with a clear failure message.

## Key Directories & Files

*   `app/`: Core application source code.
    *   `cover_letter_agent/`: Main agent orchestration.
    *   `main.py`: CLI entry point.
    *   `streamlit_app.py`: Main Streamlit web application.
    *   `pages/logs_viewer.py`: Logs monitoring interface.
    *   `ui.py`: Streamlit UI components.
    *   `style.css`: Custom styling for Streamlit.
    *   `utils.py`: Shared utility functions.
    *   `.env`: Configuration file.
*   `logs/`: Stores execution logs (e.g., `sub_agents_output_<domain>.log`) useful for debugging agent reasoning.
*   `cl_agent_uv.sh`: Specific helper script for running the app with `uv`, handling branch switching for deployment.
*   `Dockerfile` & `run_docker.sh`: Docker configuration for containerized deployment.

## Setup & Installation

### Prerequisites
*   Python >= 3.12
*   API Keys:
    *   **Google Gemini API Key**

### Installation
1.  **Environment Setup:**
    Create a `.env` file in the `app/` directory (or root) based on the template:
    ```env
    GOOGLE_GENAI_USE_VERTEXAI=False
    GOOGLE_API_KEY=<your_google_api_key>
    ```

2.  **Dependencies:**
    Using `uv` (recommended):
    ```bash
    uv sync
    ```
    Or using `pip`:
    ```bash
    pip install -r app/requirements.txt
    ```

## Running the Application

### 1. Streamlit Web UI (Recommended)
The most feature-rich interface, allowing PDF uploads and model configuration.

**Standard run:**
```bash
uv run streamlit run app/streamlit_app.py
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
uv run python app/main.py -f path/to/your_cv.pdf [options]
```
**Options:**
*   `-f`, `--file_name`: Path to the PDF CV file.
*   `-v`, `--verbose`: Enable verbose logging.
*   `-l`, `--language_level`: Language level (b1, b2, c1, c2).
*   `-T`, `--thinking_level`: Gemini 3.0 thinking level (minimal, low, medium, high).
*   `-m`, `--model`: Gemini model used by the root agent.

### 3. Google ADK Web UI
Launch the agent utilizing the standard ADK web interface.
```bash
adk web [options]
```

### 4. Docker
Build and run the containerized application.
```bash
# Build
docker build -t cl-agent-streamlit .

# Run (helper script available)
./run_docker.sh
# OR manually
docker run --name cl-agent --rm -it -p 8501:8501 -e GOOGLE_API_KEY=<your_google_api_key> -v "$(pwd)/logs:/cl_agent/logs" cl-agent-streamlit
```

## Development Conventions
*   **Agent Framework:** Built on Google ADK.
*   **Logging:** Check `logs/` for detailed agent outputs to understand the reasoning process. You can also view logs using the Logs Viewer in the Streamlit UI (`pages/logs_viewer.py`).
*   **State Management:** Streamlit session state is used for the UI interaction flow.
*   **Testing:** Use the CLI mode for rapid feedback loops during development.
