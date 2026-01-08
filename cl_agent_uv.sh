#!/bin/bash

# Store the current branch
current_branch=$(git rev-parse --abbrev-ref HEAD)

if [[ "$1" == "--remote" || "$1" == "-r" ]]; then
    # Remote mode
    if [ "$current_branch" == "master" ]; then
        echo "Switching to deploy_gcp..."
        git checkout deploy_gcp || exit 1
    fi

    cd app
    uv run streamlit run streamlit_vrtx.py
else
    # Local mode
    if [ "$current_branch" == "deploy_gcp" ]; then
        echo "Switching to master..."
        git checkout master || exit 1
    fi

    cd app
    uv run streamlit run streamlit_app.py
fi