# Use the official Python slim image as base
FROM python:3.10.12-slim

# Set working directory in the container to the project root
WORKDIR /cl_remote_agent

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY app/requirements.txt ./app/
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r app/requirements.txt

# Copy the rest of the application code into the app directory
COPY app/ ./app/

# Expose port for Streamlit
EXPOSE 8501

# Healthcheck to ensure the container is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Change working directory to the app folder so Streamlit finds assets and code
WORKDIR /cl_remote_agent/app

# Command to run the Streamlit application
ENTRYPOINT ["streamlit", "run", "streamlit_vrtx.py", "--server.port=8501", "--server.address=0.0.0.0"]
