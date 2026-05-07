"""
run_api.py
----------
Entry point to start the FastAPI server.

Ensure run_training.py has been executed first so that
results/all_states_summary.json exists before starting the API.

Usage:
  python run_api.py

The API will be available at:
  http://localhost:8000           — health check
  http://localhost:8000/docs      — interactive Swagger UI
  http://localhost:8000/states    — list all states
  http://localhost:8000/forecast/California   — forecast for California
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",    # listen on all interfaces (0.0.0.0 = accessible on LAN too)
        port=8000,
        reload=False,      # set True during development for auto-reload on file changes
        log_level="info",
    )
