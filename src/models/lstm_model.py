"""
lstm_model.py
-------------
LSTM (Long Short-Term Memory) deep learning model for sales forecasting.

Why LSTM for time series:
  LSTM is a special type of Recurrent Neural Network (RNN) that has internal
  memory cells with 'gates' — they can learn WHAT to remember, WHAT to forget,
  and WHAT to output. This makes LSTMs naturally suited for sequences where
  patterns at different time scales matter (e.g., weekly + monthly + quarterly).

Architecture:
  Input → LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)

  Two stacked LSTM layers capture both short-term (first layer) and
  medium-term (second layer) temporal patterns. Dropout prevents overfitting.

Lookback window:
  We use 12 weeks of history as the input sequence to predict the next 1 week.
  For 8-week forecasting we again use recursive prediction (same logic as XGBoost).

Normalization:
  Neural networks require input scaling. We normalize each state's sales
  independently using (x - min) / (max - min) so all values are in [0, 1].
  The inverse transform is applied to convert predictions back to dollar values.
"""

import sys, os
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import joblib
import logging
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TensorFlow startup messages
warnings.filterwarnings("ignore")

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

LOOKBACK = 12   # number of past weeks fed into LSTM as one input sequence


def _create_sequences(values: np.ndarray, lookback: int):
    """
    Converts a 1-D time series array into (X, y) pairs for supervised learning.

    For lookback=12:
      X[0] = values[0:12]  →  y[0] = values[12]
      X[1] = values[1:13]  →  y[1] = values[13]
      ...

    Returns:
      X: shape (n_samples, lookback, 1)  — 3-D as required by LSTM
      y: shape (n_samples,)
    """
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback : i])
        y.append(values[i])
    return np.array(X).reshape(-1, lookback, 1), np.array(y)


def _build_lstm_model(lookback: int) -> Sequential:
    """
    Builds and compiles the two-layer stacked LSTM architecture.

    Args:
        lookback: Number of time steps in each input sequence.

    Returns:
        Compiled but un-fitted Keras Sequential model.
    """
    model = Sequential([
        # First LSTM: return_sequences=True passes the full sequence to the next layer
        LSTM(64, input_shape=(lookback, 1), return_sequences=True),
        Dropout(0.2),               # randomly zero 20% of neurons to reduce overfitting
        LSTM(32, return_sequences=False),   # second layer: output only the last hidden state
        Dropout(0.2),
        Dense(1),                   # single output: the predicted sales value (scaled)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def train_lstm(train_series: pd.Series) -> tuple:
    """
    Normalizes the training data, builds sequences, and fits the LSTM model.

    Args:
        train_series: The training sales values as a pandas Series.

    Returns:
        A tuple of (fitted_keras_model, fitted_MinMaxScaler) — the scaler
        is needed later to inverse-transform predictions back to dollar scale.
    """
    logger.info("  LSTM: training model...")

    values = train_series.values.reshape(-1, 1)

    # Fit scaler ONLY on training data — never on validation or future data.
    # This is essential to prevent data leakage through normalization.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values).flatten()

    # Build (X, y) sequence pairs
    X_train, y_train = _create_sequences(scaled, LOOKBACK)

    model = _build_lstm_model(LOOKBACK)

    # EarlyStopping halts training when val_loss stops improving for 10 epochs,
    # restoring the best weights seen during training.
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=0,
    )

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.1,   # 10% of training for internal Keras validation
        callbacks=[early_stop],
        verbose=0,
    )

    logger.info("  LSTM: training complete.")
    return model, scaler


def forecast_lstm(
    model: Sequential,
    scaler: MinMaxScaler,
    train_series: pd.Series,
    n_periods: int = 8,
) -> np.ndarray:
    """
    Recursive multi-step LSTM forecast.

    We maintain a rolling window of `LOOKBACK` scaled values.
    At each step we predict the next value, append it to the window,
    drop the oldest value, and repeat.

    Args:
        model:        Fitted Keras LSTM model.
        scaler:       MinMaxScaler fitted on training data (for inverse transform).
        train_series: Full training series (we use the last LOOKBACK values as seed).
        n_periods:    Number of future weekly steps to forecast.

    Returns:
        numpy array of forecasted sales values in original dollar scale.
    """
    # Initialize the rolling window with the last LOOKBACK scaled training values
    last_values = train_series.values[-LOOKBACK:].reshape(-1, 1)
    window      = scaler.transform(last_values).flatten().tolist()

    forecasts = []
    for _ in range(n_periods):
        # Shape the window into (1, LOOKBACK, 1) for Keras predict()
        X_input  = np.array(window[-LOOKBACK:]).reshape(1, LOOKBACK, 1)
        pred_scaled = float(model.predict(X_input, verbose=0)[0][0])

        # Append the scaled prediction to the window (for the next step)
        window.append(pred_scaled)

        # Inverse-transform back to dollar scale
        pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]
        forecasts.append(max(pred_real, 0))  # clip to zero

    return np.array(forecasts)


def evaluate_lstm(
    model: Sequential,
    scaler: MinMaxScaler,
    train_series: pd.Series,
    val_series: pd.Series,
) -> dict:
    """Evaluates the LSTM on the validation set using recursive forecasting."""
    preds   = forecast_lstm(model, scaler, train_series, n_periods=len(val_series))
    actuals = val_series.values

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae  = mean_absolute_error(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-8))) * 100

    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape)}


def save_lstm(model: Sequential, scaler: MinMaxScaler, state: str, save_dir: str) -> str:
    """Saves the LSTM model (Keras .h5) and scaler (joblib .pkl) to disk."""
    os.makedirs(save_dir, exist_ok=True)
    safe_state = state.replace(" ", "_")
    model_path  = os.path.join(save_dir, f"lstm_{safe_state}.h5")
    scaler_path = os.path.join(save_dir, f"lstm_scaler_{safe_state}.pkl")
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    return model_path


def load_lstm(state: str, save_dir: str) -> tuple:
    """Loads a saved LSTM model and its scaler from disk."""
    safe_state  = state.replace(" ", "_")
    model_path  = os.path.join(save_dir, f"lstm_{safe_state}.h5")
    scaler_path = os.path.join(save_dir, f"lstm_scaler_{safe_state}.pkl")
    model  = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


if __name__ == "__main__":
    from src.preprocessing import prepare_all_states
    from src.feature_engineering import train_val_split

    DATA_PATH  = os.path.join(_PROJECT_ROOT, "data", "Forecasting_Case-_Study.xlsx")
    state_data = prepare_all_states(DATA_PATH)

    state = "Wyoming"
    df    = state_data[state]
    train_df, val_df = train_val_split(df, val_weeks=8)

    model, scaler = train_lstm(train_df["Total"])
    metrics       = evaluate_lstm(model, scaler, train_df["Total"], val_df["Total"])
    forecast      = forecast_lstm(model, scaler, train_df["Total"], n_periods=8)

    print(f"\n✅ LSTM — {state}")
    print(f"   Val RMSE      : {metrics['rmse']:,.0f}")
    print(f"   Val MAE       : {metrics['mae']:,.0f}")
    print(f"   Val MAPE      : {metrics['mape']:.2f}%")
    print(f"   8-week forecast: {[f'{v:,.0f}' for v in forecast]}")
