"""
arima_model.py
--------------
ARIMA (AutoRegressive Integrated Moving Average) model for sales forecasting.

Design decision:
  Full SARIMA with m=52 (annual weekly seasonality) requires fitting many
  candidate models with large 52-lag seasonal matrices — prohibitively
  memory-intensive when running across 43 states in production. The
  industry-standard alternative is non-seasonal ARIMA with auto-selected
  (p, d, q) via AIC, and let Prophet / XGBoost carry the seasonality signal.
"""

import numpy as np
import pandas as pd
import joblib
import logging
import os
import warnings

import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def train_arima(train_series: pd.Series) -> pm.arima.ARIMA:
    """
    Fits ARIMA using auto_arima with a constrained non-seasonal search.
    d is auto-detected via ADF stationarity test.
    """
    logger.info("  ARIMA: fitting auto_arima...")
    model = pm.auto_arima(
        train_series,
        start_p=0, max_p=3,
        start_q=0, max_q=3,
        d=None,
        seasonal=False,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )
    logger.info(f"  ARIMA: best order = {model.order}")
    return model


def forecast_arima(model: pm.arima.ARIMA, n_periods: int = 8) -> np.ndarray:
    """Returns n_periods future forecasts from a fitted ARIMA model."""
    forecast, _ = model.predict(n_periods=n_periods, return_conf_int=True)
    return np.clip(forecast, 0, None)  # sales cannot be negative


def evaluate_arima(model: pm.arima.ARIMA, val_series: pd.Series) -> dict:
    """Evaluates the ARIMA model on the validation set. Returns RMSE, MAE, MAPE."""
    preds   = forecast_arima(model, n_periods=len(val_series))
    actuals = val_series.values
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae  = mean_absolute_error(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100
    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape)}


def save_arima(model: pm.arima.ARIMA, state: str, save_dir: str) -> str:
    """Saves the ARIMA model to disk."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"arima_{state.replace(' ', '_')}.pkl")
    joblib.dump(model, path)
    return path


def load_arima(state: str, save_dir: str) -> pm.arima.ARIMA:
    """Loads a saved ARIMA model from disk."""
    path = os.path.join(save_dir, f"arima_{state.replace(' ', '_')}.pkl")
    return joblib.load(path)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.preprocessing import prepare_all_states
    from src.feature_engineering import train_val_split

    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "Forecasting_Case-_Study.xlsx"
    )
    state_data = prepare_all_states(DATA_PATH)

    state = "Wyoming"
    df = state_data[state]
    train_df, val_df = train_val_split(df, val_weeks=8)

    model    = train_arima(train_df["Total"])
    metrics  = evaluate_arima(model, val_df["Total"])
    forecast = forecast_arima(model, n_periods=8)

    print(f"\n✅ ARIMA — {state}")
    print(f"   Order         : {model.order}")
    print(f"   Val RMSE      : {metrics['rmse']:,.0f}")
    print(f"   Val MAE       : {metrics['mae']:,.0f}")
    print(f"   Val MAPE      : {metrics['mape']:.2f}%")
    print(f"   8-week forecast: {[f'{v:,.0f}' for v in forecast]}")
