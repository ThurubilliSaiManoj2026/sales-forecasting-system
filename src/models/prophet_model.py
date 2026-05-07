"""
prophet_model.py
----------------
Facebook Prophet model for sales forecasting.

Why Prophet excels here:
  - Natively models trend + yearly seasonality + weekly seasonality
  - Handles missing data and outliers robustly out of the box
  - Requires minimal hyperparameter tuning — very good defaults
  - Produces interpretable decomposition (trend, seasonality, holiday effects)

Prophet expects a DataFrame with exactly two columns: 'ds' (dates) and 'y' (values).
"""

import numpy as np
import pandas as pd
import joblib
import logging
import os
import warnings

from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _to_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts our standard ['Date', 'Total'] DataFrame to Prophet's
    required ['ds', 'y'] format.
    """
    return pd.DataFrame({"ds": df["Date"].values, "y": df["Total"].values})


def train_prophet(train_df: pd.DataFrame) -> Prophet:
    """
    Trains a Prophet model on the provided training DataFrame.

    Config choices:
      - yearly_seasonality=True:  captures the annual sales cycle
      - weekly_seasonality=False: after resampling to weekly, all observations
                                  are Sundays — no intra-week variation exists
      - daily_seasonality=False:  same reason
      - changepoint_prior_scale=0.1: moderate flexibility for trend changepoints
        (higher = more flexible trend, risk of overfitting; 0.05-0.5 is typical)

    Args:
        train_df: DataFrame with ['Date', 'Total'] columns.

    Returns:
        A fitted Prophet model.
    """
    logger.info("  Prophet: training model...")

    prophet_train = _to_prophet_df(train_df)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,    # trend flexibility
        seasonality_prior_scale=10.0,   # seasonality amplitude flexibility
        seasonality_mode="multiplicative",  # better for sales that grow over time
    )

    model.fit(prophet_train)
    logger.info("  Prophet: training complete.")
    return model


def forecast_prophet(
    model: Prophet,
    last_date: pd.Timestamp,
    n_periods: int = 8,
) -> np.ndarray:
    """
    Generates n_periods weekly forecasts starting from the week after last_date.

    Args:
        model:     Fitted Prophet model.
        last_date: The last known date in the training data.
        n_periods: Number of future weekly periods to predict.

    Returns:
        numpy array of forecasted 'yhat' values, shape (n_periods,).
    """
    # Build a future DataFrame with exactly n_periods rows beyond last_date
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=n_periods,
        freq="W",
    )
    future_df = pd.DataFrame({"ds": future_dates})
    forecast  = model.predict(future_df)

    # Clip to zero — sales cannot go negative
    return np.clip(forecast["yhat"].values, 0, None)


def evaluate_prophet(
    model: Prophet,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict:
    """
    Evaluates Prophet on the validation set.
    We pass the full training history's last date so the future periods
    align exactly with the validation dates.

    Returns a dict with RMSE, MAE, MAPE metrics.
    """
    last_train_date = pd.Timestamp(train_df["Date"].max())
    preds   = forecast_prophet(model, last_train_date, n_periods=len(val_df))
    actuals = val_df["Total"].values

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae  = mean_absolute_error(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100

    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape)}


def save_prophet(model: Prophet, state: str, save_dir: str) -> str:
    """Saves the Prophet model to disk using joblib."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"prophet_{state.replace(' ', '_')}.pkl")
    joblib.dump(model, path)
    return path


def load_prophet(state: str, save_dir: str) -> Prophet:
    """Loads a saved Prophet model from disk."""
    path = os.path.join(save_dir, f"prophet_{state.replace(' ', '_')}.pkl")
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

    model    = train_prophet(train_df)
    metrics  = evaluate_prophet(model, train_df, val_df)
    last_dt  = pd.Timestamp(train_df["Date"].max())
    forecast = forecast_prophet(model, last_dt, n_periods=8)

    print(f"\n✅ Prophet — {state}")
    print(f"   Val RMSE      : {metrics['rmse']:,.0f}")
    print(f"   Val MAE       : {metrics['mae']:,.0f}")
    print(f"   Val MAPE      : {metrics['mape']:.2f}%")
    print(f"   8-week forecast: {[f'{v:,.0f}' for v in forecast]}")
