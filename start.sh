#!/bin/bash
# Starts the API (internal port 8000) and the UI (public port 7860).
set -e
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
exec streamlit run ui/streamlit_app.py \
  --server.port 7860 --server.address 0.0.0.0 --server.headless true \
  --server.enableCORS false --server.enableXsrfProtection false
