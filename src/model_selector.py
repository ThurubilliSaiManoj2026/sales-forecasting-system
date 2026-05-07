"""
model_selector.py
-----------------
Trains all 4 forecasting models for a single state, evaluates each on the
8-week validation set, and selects the best model by RMSE (lowest = best).

This module is the core of the "automated model selection" requirement.
It ensures no manual intervention is needed — the system objectively picks
the best algorithm for every state based on measured performance.

Why RMSE as the selection metric?
  RMSE (Root Mean Squared Error) penalizes large errors more heavily than
  MAE (Mean Absolute Error). In sales forecasting, a single very bad week
  (e.g., missing a demand spike by 50%) is more damaging than many small
  errors. RMSE reflects this priority and is the industry standard for
  regression-based time series evaluation.
"""

import sys, os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import logging
import numpy as np
import pandas as pd

from src.feature_engineering import create_features, train_val_split
from src.models.arima_model    import train_arima, evaluate_arima, forecast_arima, save_arima
from src.models.prophet_model  import train_prophet, evaluate_prophet, forecast_prophet, save_prophet
from src.models.xgboost_model  import train_xgboost, evaluate_xgboost, forecast_xgboost, save_xgboost
from src.models.lstm_model     import train_lstm, evaluate_lstm, forecast_lstm, save_lstm

logger = logging.getLogger(__name__)


def select_best_model_for_state(
    state: str,
    weekly_df: pd.DataFrame,
    models_dir: str,
    val_weeks: int = 8,
) -> dict:
    """
    Trains all 4 models on the state's data, evaluates each on the validation
    set, selects the winner by lowest RMSE, saves all 4 models to disk, and
    returns a comprehensive result dictionary.

    Args:
        state:      Name of the US state (used for model file naming).
        weekly_df:  Clean weekly ['Date', 'Total'] DataFrame for this state.
        models_dir: Directory where trained model files will be saved.
        val_weeks:  Number of weeks to hold out for validation.

    Returns:
        A dict containing:
          - 'state':       state name
          - 'best_model':  name of the winning model ('arima'/'prophet'/'xgboost'/'lstm')
          - 'metrics':     dict of all 4 models' RMSE/MAE/MAPE scores
          - 'forecast':    the 8-week future forecast from the BEST model
          - 'forecast_dates': ISO-format date strings for the 8 forecast weeks
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing state: {state}")
    logger.info(f"{'='*60}")

    # ── SPLIT: feature-engineered data for ML models, raw for ARIMA/Prophet ──
    df_feat              = create_features(weekly_df)
    train_feat, val_feat = train_val_split(df_feat, val_weeks=val_weeks)

    # The raw (pre-feature-engineering) split for ARIMA and Prophet
    # We align the split point to the same date as the feature-engineered split
    split_date = train_feat["Date"].max()
    train_raw  = weekly_df[weekly_df["Date"] <= split_date].copy()
    val_raw    = weekly_df[weekly_df["Date"] >  split_date].copy()

    metrics = {}
    trained_models = {}

    # ── MODEL 1: ARIMA ────────────────────────────────────────────────────────
    try:
        arima_model = train_arima(train_raw["Total"])
        metrics["arima"] = evaluate_arima(arima_model, val_raw["Total"])
        trained_models["arima"] = arima_model
        save_arima(arima_model, state, models_dir)
        logger.info(f"  ARIMA   RMSE: {metrics['arima']['rmse']:>15,.0f}  MAPE: {metrics['arima']['mape']:.2f}%")
    except Exception as e:
        logger.warning(f"  ARIMA failed for {state}: {e}")
        metrics["arima"] = {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf")}

    # ── MODEL 2: PROPHET ──────────────────────────────────────────────────────
    try:
        prophet_model = train_prophet(train_raw)
        metrics["prophet"] = evaluate_prophet(prophet_model, train_raw, val_raw)
        trained_models["prophet"] = prophet_model
        save_prophet(prophet_model, state, models_dir)
        logger.info(f"  Prophet RMSE: {metrics['prophet']['rmse']:>15,.0f}  MAPE: {metrics['prophet']['mape']:.2f}%")
    except Exception as e:
        logger.warning(f"  Prophet failed for {state}: {e}")
        metrics["prophet"] = {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf")}

    # ── MODEL 3: XGBOOST ─────────────────────────────────────────────────────
    try:
        xgb_model = train_xgboost(train_feat)
        metrics["xgboost"] = evaluate_xgboost(xgb_model, train_feat, val_feat)
        trained_models["xgboost"] = xgb_model
        save_xgboost(xgb_model, state, models_dir)
        logger.info(f"  XGBoost RMSE: {metrics['xgboost']['rmse']:>15,.0f}  MAPE: {metrics['xgboost']['mape']:.2f}%")
    except Exception as e:
        logger.warning(f"  XGBoost failed for {state}: {e}")
        metrics["xgboost"] = {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf")}

    # ── MODEL 4: LSTM ─────────────────────────────────────────────────────────
    try:
        lstm_model, lstm_scaler = train_lstm(train_raw["Total"])
        metrics["lstm"] = evaluate_lstm(lstm_model, lstm_scaler, train_raw["Total"], val_raw["Total"])
        trained_models["lstm"] = (lstm_model, lstm_scaler)
        save_lstm(lstm_model, lstm_scaler, state, models_dir)
        logger.info(f"  LSTM    RMSE: {metrics['lstm']['rmse']:>15,.0f}  MAPE: {metrics['lstm']['mape']:.2f}%")
    except Exception as e:
        logger.warning(f"  LSTM failed for {state}: {e}")
        metrics["lstm"] = {"rmse": float("inf"), "mae": float("inf"), "mape": float("inf")}

    # ── SELECT BEST MODEL BY RMSE ─────────────────────────────────────────────
    best_model_name = min(metrics, key=lambda m: metrics[m]["rmse"])
    logger.info(f"\n  ★ Best model for {state}: {best_model_name.upper()} "
                f"(RMSE: {metrics[best_model_name]['rmse']:,.0f})")

    # ── GENERATE FINAL 8-WEEK FORECAST USING BEST MODEL ──────────────────────
    last_date     = pd.Timestamp(weekly_df["Date"].max())
    forecast_dates = [
        (last_date + pd.Timedelta(weeks=i + 1)).strftime("%Y-%m-%d")
        for i in range(8)
    ]

    if best_model_name == "arima" and "arima" in trained_models:
        forecast_values = forecast_arima(trained_models["arima"], n_periods=8).tolist()

    elif best_model_name == "prophet" and "prophet" in trained_models:
        forecast_values = forecast_prophet(
            trained_models["prophet"], last_date, n_periods=8
        ).tolist()

    elif best_model_name == "xgboost" and "xgboost" in trained_models:
        forecast_values = forecast_xgboost(
            trained_models["xgboost"], weekly_df[["Date", "Total"]], n_periods=8
        ).tolist()

    elif best_model_name == "lstm" and "lstm" in trained_models:
        lstm_m, lstm_s = trained_models["lstm"]
        forecast_values = forecast_lstm(lstm_m, lstm_s, weekly_df["Total"], n_periods=8).tolist()

    else:
        forecast_values = [0.0] * 8
        logger.error(f"  All models failed for {state}. Setting forecast to zeros.")

    return {
        "state":          state,
        "best_model":     best_model_name,
        "metrics":        metrics,
        "forecast":       forecast_values,
        "forecast_dates": forecast_dates,
    }


def run_full_pipeline(
    state_data: dict,
    models_dir: str,
    results_dir: str,
) -> dict:
    """
    Runs the full model selection pipeline for ALL states.
    Saves a per-state JSON result and a combined summary JSON.

    Args:
        state_data:  Dict of { state_name -> weekly DataFrame } from preprocessing.
        models_dir:  Directory to save trained model files.
        results_dir: Directory to save JSON result files.

    Returns:
        Dict of { state_name -> result_dict } for all states.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}
    states      = sorted(state_data.keys())
    total       = len(states)

    for i, state in enumerate(states, 1):
        logger.info(f"\n[{i}/{total}] Training models for: {state}")
        result = select_best_model_for_state(
            state       = state,
            weekly_df   = state_data[state],
            models_dir  = models_dir,
        )
        all_results[state] = result

        # Save individual state result
        state_result_path = os.path.join(
            results_dir, f"{state.replace(' ', '_')}_result.json"
        )
        with open(state_result_path, "w") as f:
            json.dump(result, f, indent=2)

    # Save combined summary for API to load at startup
    summary_path = os.path.join(results_dir, "all_states_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE — Model Selection Summary:")
    logger.info(f"{'='*60}")

    from collections import Counter
    winner_counts = Counter(r["best_model"] for r in all_results.values())
    for model_name, count in sorted(winner_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {model_name.upper():12s}: won {count:>2d} / {total} states")

    logger.info(f"\nResults saved to : {results_dir}")
    logger.info(f"Models saved to  : {models_dir}")

    return all_results


def run_batch_pipeline(
    state_data: dict,
    models_dir: str,
    results_dir: str,
    batch_states: list,
) -> dict:
    """
    Trains models for a specific subset of states only.
    Skips any state whose result JSON already exists (resume-safe).
    After processing the batch, rebuilds the combined summary JSON
    from ALL per-state result files found in results_dir.
    """
    import json, os
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    for state in batch_states:
        result_path = os.path.join(results_dir, f"{state.replace(' ', '_')}_result.json")
        if os.path.exists(result_path):
            logger.info(f"SKIP (already done): {state}")
            continue

        result = select_best_model_for_state(
            state=state, weekly_df=state_data[state], models_dir=models_dir
        )
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

    # Rebuild the combined summary from ALL per-state JSONs on disk
    all_results = {}
    for fname in os.listdir(results_dir):
        if fname.endswith("_result.json"):
            with open(os.path.join(results_dir, fname)) as f:
                r = json.load(f)
                all_results[r["state"]] = r

    summary_path = os.path.join(results_dir, "all_states_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Batch done. Summary now covers {len(all_results)} states.")
    return all_results
