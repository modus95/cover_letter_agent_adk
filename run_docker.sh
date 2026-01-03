#!/bin/bash

docker run --name cl-agent \
--rm -it -p 8501:8501 \
-e GOOGLE_API_KEY=$GOOGLE_API_KEY \
-e TAVILY_API_KEY=$TAVILY_API_KEY \
-v "$(pwd)/logs:/cl_agent/logs" \
cl-agent-streamlit