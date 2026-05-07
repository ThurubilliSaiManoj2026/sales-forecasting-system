"""
xgboost_model.py
----------------
XGBoost regressor for sales forecasting using engineered lag/calendar features.

Forecasting strategy — recursive multi-step prediction:
  XGBoost predicts one step ahead at a time. To forecast 8 weeks:
    - Predict week 1 → append prediction → recompute features → predict week 2
    - Repeat for all 8 steps, propagating predictions through the lag structure.
"""

import sys, os
# Ensure the project root is on sys.path so src.* imports work
# whether this file is run directly or imported from the pipeline.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import joblib
import logging

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.feature_engineering import create_features, get_feature_columns

logger = logging.getLogger(__name__)
FEATURE_COLS = get_feature_columns()


def train_xgboost(train_df: pd.DataFrame) -> XGBRegressor:
    """
    Trains XGBoost on feature-engineered training data.

    Args:
        train_df: DataFrame with 'Date', 'Total', and all feature columns.

    Returns:
        Fitted XGBRegressor.
    """
    logger.info("  XGBoost: training model...")

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["Total"].values

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    logger.info("  XGBoost: training complete.")
    return model


def forecast_xgboost(
    model: XGBRegressor,
    history_df: pd.DataFrame,
    n_periods: int = 8,
) -> np.ndarray:
    """
    Recursive multi-step forecast: at each step, predicts t+1, appends that
    prediction to history, recomputes features, then predicts t+2, etc.

    Args:
        model:      Fitted XGBRegressor.
        history_df: Raw ['Date', 'Total'] DataFrame up to the last known date.
        n_periods:  Number of future weekly steps to forecast.

    Returns:
        numpy array of forecasted values, shape (n_periods,).
    """
    running_df = history_df[["Date", "Total"]].copy().sort_values("Date").reset_index(drop=True)
    forecasts  = []
    last_date  = pd.Timestamp(running_df["Date"].max())

    for _ in range(n_periods):
        next_date = last_date + pd.Timedelta(weeks=1)

        # Append a placeholder row with the last known value so lag/rolling
        # computations don't encounter NaN. We replace it with the prediction.
        placeholder = pd.DataFrame({
            "Date":  [next_date],
            "Total": [running_df["Total"].iloc[-1]]
        })
        running_df = pd.concat([running_df, placeholder], ignore_index=True)

        # Recompute features on the growing history
        featured_df = create_features(running_df)

        # Predict the very last row (= next_date)
        last_row  = featured_df.iloc[[-1]][FEATURE_COLS]
        predicted = float(model.predict(last_row)[0])
        predicted = max(predicted, 0)

        # Update the placeholder Total with the actual prediction
        running_df.loc[running_df.index[-1], "Total"] = predicted

        forecasts.append(predicted)
        last_date = next_date

    return np.array(forecasts)


def evaluate_xgboost(
    model: XGBRegressor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict:
    """Evaluates XGBoost recursively on the validation set."""
    raw_history = train_df[["Date", "Total"]].copy()
    preds   = forecast_xgboost(model, raw_history, n_periods=len(val_df))
    actuals = val_df["Total"].values

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae  = mean_absolute_error(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100

    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape)}


def save_xgboost(model: XGBRegressor, state: str, save_dir: str) -> str:
    """Saves the XGBoost model to disk."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"xgboost_{state.replace(' ', '_')}.pkl")
    joblib.dump(model, path)
    return path


def load_xgboost(state: str, save_dir: str) -> XGBRegressor:
    """Loads a saved XGBoost model from disk."""
    path = os.path.join(save_dir, f"xgboost_{state.replace(' ', '_')}.pkl")
    return joblib.load(path)


if __name__ == "__main__":
    from src.preprocessing import prepare_all_states
    from src.feature_engineering import create_features, train_val_split

    DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "Forecasting_Case-_Study.xlsx")
    state_data = prepare_all_states(DATA_PATH)

    state    = "Wyoming"
    df_raw   = state_data[state]
    df_feat  = create_features(df_raw)
    train_df, val_df = train_val_split(df_feat, val_weeks=8)

    model    = train_xgboost(train_df)
    metrics  = evaluate_xgboost(model, train_df, val_df)
    forecast = forecast_xgboost(model, train_df[["Date", "Total"]], n_periods=8)

    print(f"\n✅ XGBoost — {state}")
    print(f"   Val RMSE      : {metrics['rmse']:,.0f}")
    print(f"   Val MAE       : {metrics['mae']:,.0f}")
    print(f"   Val MAPE      : {metrics['mape']:.2f}%")
    print(f"   8-week forecast: {[f'{v:,.0f}' for v in forecast]}")
