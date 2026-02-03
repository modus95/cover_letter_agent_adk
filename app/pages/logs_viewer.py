"""
Logs Viewer Module
This module provides a Streamlit interface to view and monitor subagents logs.
"""

import os
import streamlit as st
try:
    from streamlit_app import LOGFILE_NAME
except ImportError:
    from app.streamlit_app import LOGFILE_NAME

st.set_page_config(page_title="Logs Viewer", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS to reduce font size for the code block
st.markdown("""
    <style>
        code {
            font-size: 0.75rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Path to the log file
log_path = os.path.join(os.path.dirname(__file__), "..", "..", "logs", LOGFILE_NAME)

if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        log_content = f.read()

    st.code(log_content, language="markdown", wrap_lines=True, width=1200)
else:
    st.error("Log file not found.")
