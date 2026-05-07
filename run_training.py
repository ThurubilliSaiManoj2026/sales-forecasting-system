"""
run_training.py
---------------
Master entry point for the training pipeline.

Run this script once to:
  1. Load and preprocess the raw Excel dataset
  2. Train all 4 models (ARIMA, Prophet, XGBoost, LSTM) for all 43 states
  3. Evaluate each model on the 8-week validation set
  4. Select the best model per state (lowest RMSE)
  5. Save all trained models to trained_models/
  6. Save all results + forecasts to results/all_states_summary.json

Usage:
  python run_training.py

Expected runtime: 15-30 minutes for all 43 states (depends on CPU speed).
After this runs successfully, start the API with:
  python run_api.py
"""

import sys
import os
import logging
import time

# Add project root to sys.path so all src.* imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import prepare_all_states
from src.model_selector import run_full_pipeline

# Configure logging — INFO level shows progress, DEBUG would be too noisy
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # Also write logs to a file for post-training review
        logging.FileHandler("training.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    # ── Resolve paths relative to this script's location ─────────────────────
    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH   = os.path.join(BASE_DIR, "data", "Forecasting_Case-_Study.xlsx")
    MODELS_DIR  = os.path.join(BASE_DIR, "trained_models")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    # Verify dataset exists before doing anything
    if not os.path.exists(DATA_PATH):
        logger.error(f"Dataset not found at: {DATA_PATH}")
        logger.error("Please place 'Forecasting_Case-_Study.xlsx' in the data/ folder.")
        sys.exit(1)

    logger.info("=" * 65)
    logger.info("   Sales Forecasting System — Training Pipeline")
    logger.info("=" * 65)
    logger.info(f"Dataset   : {DATA_PATH}")
    logger.info(f"Models dir: {MODELS_DIR}")
    logger.info(f"Results   : {RESULTS_DIR}")
    logger.info("=" * 65)

    # ── Step 1: Load and preprocess all states ────────────────────────────────
    logger.info("\nStep 1/2 — Loading and preprocessing data...")
    start_load = time.time()
    state_data = prepare_all_states(DATA_PATH)
    logger.info(f"Preprocessing complete in {time.time() - start_load:.1f}s")

    # ── Step 2: Train models for all states and select the best ──────────────
    logger.info("\nStep 2/2 — Training models for all states...")
    logger.info("(This will take 15-30 minutes. Progress is logged per state.)\n")
    start_train = time.time()

    all_results = run_full_pipeline(
        state_data  = state_data,
        models_dir  = MODELS_DIR,
        results_dir = RESULTS_DIR,
    )

    elapsed = time.time() - start_train
    logger.info(f"\nTotal training time: {elapsed / 60:.1f} minutes")
    logger.info("\n✅ Training complete! You can now start the API:")
    logger.info("   python run_api.py")


if __name__ == "__main__":
    main()
