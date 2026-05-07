"""
api/main.py
-----------
FastAPI REST API for the Sales Forecasting System.

Design principles (production-grade backend service):
  - All 43 states' forecasts are pre-computed during the training phase and
    stored in results/all_states_summary.json. The API loads this file once
    at startup — making every request O(1) with no model inference overhead.
  - Proper Pydantic request/response models for automatic validation + docs.
  - Meaningful HTTP status codes (200, 404, 503) with clear error messages.
  - FastAPI's built-in OpenAPI docs are available at /docs (Swagger UI).
  - CORS enabled so a frontend can call this API from any origin.

Endpoints:
  GET  /                       — Health check + system status
  GET  /states                 — List all 43 available state names
  GET  /forecast/{state}       — 8-week forecast for a specific state
  GET  /model-info/{state}     — Best model selected + all 4 models' metrics
  GET  /all-forecasts          — Combined 8-week forecasts for all 43 states
  POST /forecast               — Same as GET /forecast/{state}, via request body
"""

import sys, os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ── Global in-memory store loaded once at API startup ─────────────────────────
SUMMARY: Dict = {}
RESULTS_PATH = os.path.join(_PROJECT_ROOT, "results", "all_states_summary.json")


# ── Lifespan: loads pre-computed results when the server starts ───────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Code before 'yield' runs at startup; code after 'yield' runs at shutdown.
    We load the pre-computed forecast summary JSON once here so every request
    can serve data in O(1) without re-running any model inference.
    """
    global SUMMARY
    if not os.path.exists(RESULTS_PATH):
        logger.warning(
            f"Summary file not found at {RESULTS_PATH}. "
            "Run `python run_training.py` first to generate results."
        )
    else:
        with open(RESULTS_PATH, "r") as f:
            SUMMARY = json.load(f)
        logger.info(f"Loaded forecast results for {len(SUMMARY)} states ✅")
    yield
    # Shutdown: nothing to clean up for this stateless service
    logger.info("API shutting down.")


# ── FastAPI app definition ────────────────────────────────────────────────────
app = FastAPI(
    title="Sales Forecasting API",
    description=(
        "Production-ready API for 8-week beverage sales forecasting across 43 US states. "
        "Models: ARIMA, Facebook Prophet, XGBoost, LSTM. "
        "Best model per state is automatically selected by lowest validation RMSE."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: allow requests from any origin (needed if a frontend hits this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Response Models ──────────────────────────────────────────────────

class ModelMetrics(BaseModel):
    """Validation metrics for a single forecasting model."""
    rmse: float
    mae:  float
    mape: float


class ForecastResponse(BaseModel):
    """8-week sales forecast for a single state."""
    state:          str
    best_model:     str
    forecast_dates: List[str]
    forecast_values: List[float]
    unit:           str = "USD"


class ModelInfoResponse(BaseModel):
    """Model selection details + all 4 models' validation metrics for a state."""
    state:      str
    best_model: str
    metrics:    Dict[str, ModelMetrics]


class AllForecastsResponse(BaseModel):
    """Aggregated 8-week forecasts for all 43 states."""
    total_states: int
    forecasts:    List[ForecastResponse]


class ForecastRequest(BaseModel):
    """Request body for POST /forecast."""
    state: str


class HealthResponse(BaseModel):
    """Health check response."""
    status:        str
    states_loaded: int
    models_dir_exists: bool
    results_file_exists: bool
    message:       str


# ── Helper ────────────────────────────────────────────────────────────────────

def _normalize_state(state: str) -> str:
    """
    Converts a state name to Title Case so the API is case-insensitive.
    'california' → 'California', 'NEW YORK' → 'New York'
    """
    return state.strip().title()


def _get_state_result(state: str) -> dict:
    """
    Looks up a state's result from the in-memory SUMMARY.
    Raises HTTP 503 if results haven't been loaded yet, or 404 if the state
    is not found in the dataset.
    """
    if not SUMMARY:
        raise HTTPException(
            status_code=503,
            detail=(
                "Forecast results are not loaded. "
                "Please run `python run_training.py` first."
            ),
        )
    normalized = _normalize_state(state)
    if normalized not in SUMMARY:
        available = sorted(SUMMARY.keys())
        raise HTTPException(
            status_code=404,
            detail=(
                f"State '{state}' not found in dataset. "
                f"Available states: {available}"
            ),
        )
    return SUMMARY[normalized]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Returns the health status of the API and whether forecast results are loaded.
    Use this to verify the server is running and training has been completed.
    """
    return HealthResponse(
        status              = "healthy" if SUMMARY else "degraded",
        states_loaded       = len(SUMMARY),
        models_dir_exists   = os.path.isdir(os.path.join(_PROJECT_ROOT, "trained_models")),
        results_file_exists = os.path.exists(RESULTS_PATH),
        message             = (
            f"Forecasting system ready. {len(SUMMARY)} states available."
            if SUMMARY
            else "Training results not found. Run `python run_training.py` first."
        ),
    )


@app.get("/states", response_model=List[str], tags=["Data"])
async def list_states():
    """
    Returns a sorted list of all US state names available in the dataset.
    These are the valid values for the `state` parameter in other endpoints.
    """
    if not SUMMARY:
        raise HTTPException(
            status_code=503,
            detail="Results not loaded. Run `python run_training.py` first.",
        )
    return sorted(SUMMARY.keys())


@app.get("/forecast/{state}", response_model=ForecastResponse, tags=["Forecast"])
async def get_forecast(state: str):
    """
    Returns the 8-week sales forecast for the specified US state.

    The forecast is generated by the best-performing model (selected
    automatically by lowest validation RMSE from ARIMA, Prophet, XGBoost, LSTM).

    - **state**: Full US state name (case-insensitive), e.g. `California`, `New York`
    """
    result = _get_state_result(state)
    return ForecastResponse(
        state            = result["state"],
        best_model       = result["best_model"],
        forecast_dates   = result["forecast_dates"],
        forecast_values  = result["forecast"],
    )


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
async def post_forecast(request: ForecastRequest):
    """
    Same as GET /forecast/{state} but accepts a JSON request body.
    Useful when the state name contains spaces (e.g., `New York`).

    Request body example:
    ```json
    { "state": "New York" }
    ```
    """
    result = _get_state_result(request.state)
    return ForecastResponse(
        state            = result["state"],
        best_model       = result["best_model"],
        forecast_dates   = result["forecast_dates"],
        forecast_values  = result["forecast"],
    )


@app.get("/model-info/{state}", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info(state: str):
    """
    Returns which model was selected as best for this state, along with the
    validation RMSE, MAE, and MAPE for all 4 models that were trained and compared.

    - **state**: Full US state name (case-insensitive)
    """
    result = _get_state_result(state)
    return ModelInfoResponse(
        state      = result["state"],
        best_model = result["best_model"],
        metrics    = {
            m: ModelMetrics(**vals)
            for m, vals in result["metrics"].items()
            if vals["rmse"] != float("inf")    # exclude models that failed
        },
    )


@app.get("/all-forecasts", response_model=AllForecastsResponse, tags=["Forecast"])
async def get_all_forecasts():
    """
    Returns 8-week sales forecasts for ALL 43 states in a single response.
    Each state's forecast uses its individually selected best model.

    This endpoint is useful for dashboards and batch reporting pipelines.
    """
    if not SUMMARY:
        raise HTTPException(
            status_code=503,
            detail="Results not loaded. Run `python run_training.py` first.",
        )

    forecasts = [
        ForecastResponse(
            state           = result["state"],
            best_model      = result["best_model"],
            forecast_dates  = result["forecast_dates"],
            forecast_values = result["forecast"],
        )
        for result in SUMMARY.values()
    ]
    forecasts.sort(key=lambda x: x.state)

    return AllForecastsResponse(
        total_states = len(forecasts),
        forecasts    = forecasts,
    )
