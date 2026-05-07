# Sales Forecasting System

An end-to-end, production ready time series forecasting system that trains four algorithms per US state, automatically selects the best performing model, and serves 8 week sales predictions via a REST API.

---

## Problem Statement

Forecast the next 8 weeks of beverage sales for each of 43 US states using ~5 years (2019–2023) of historical sales data. The system handles irregular date spacing, missing value imputation, seasonality, trend, and multi-model comparison — all without manual intervention.

---

## Architecture Overview

```
Raw Excel Data
      │
      ▼
 Preprocessing          ← Parses mixed date formats, resamples irregular data
      │                   to weekly frequency, fills gaps with interpolation
      ▼
Feature Engineering     ← Creates lag features (t-1, t-7, t-30 weeks),
      │                   rolling mean/std, calendar features, holiday flags
      ▼
 4 Model Training       ← Trains ARIMA, Prophet, XGBoost, LSTM per state
      │
      ▼
 Model Selection        ← Evaluates all 4 on 8-week validation split (no leakage)
      │                   Selects winner by lowest RMSE
      ▼
Results JSON            ← Stores forecasts + metrics for all 43 states
      │
      ▼
  FastAPI REST API       ← Serves predictions at localhost:8000
```

---

## Dataset

The raw dataset contains 8,084 rows across 43 US states. Each state has 188 data points recorded on irregular dates spanning January 2019 to December 2023. Since the task requires weekly forecasting, the preprocessing step resamples the irregular series to a clean weekly frequency (summing sales within each calendar week), resulting in 256 weekly periods per state.

---

## Models Implemented

**ARIMA** uses `pmdarima`'s `auto_arima` to automatically select the optimal `(p, d, q)` order via AIC. It captures the linear autocorrelation structure of the sales series. The differencing order `d` is automatically determined using the ADF stationarity test.

**Facebook Prophet** natively decomposes the time series into trend + seasonality components. It uses a multiplicative seasonality mode (appropriate when seasonal fluctuations scale with the trend level, as is the case for growing sales). Yearly seasonality is enabled, weekly and daily are disabled since all data points are weekly Sunday-anchored.

**XGBoost** is a gradient-boosted tree model trained on the full set of engineered features. Since XGBoost has no built-in memory for sequences, the lag and rolling features explicitly encode the "history" the model needs. Forecasting uses a recursive strategy: predict week 1, append to history, recompute features, predict week 2, and so on.

**LSTM** is a two-layer stacked Long Short-Term Memory network (64 → 32 units) trained on a 12-week lookback window. The MinMaxScaler normalizes sales to [0,1] for training stability, and the inverse transform is applied to convert predictions back to dollar values. Forecasting also uses a recursive rolling-window approach.

---

## Feature Engineering Details

All features are engineered on the weekly-resampled data. Lag features at 1 week, 7 weeks, and 30 weeks encode recent momentum, medium term cyclicality, and long-term seasonal patterns respectively. Rolling mean and standard deviation over 4-week and 8-week windows capture the local trend level and volatility. Sine/cosine encoding of week-of-year and month ensures seasonal features are cyclically smooth (avoiding the artificial discontinuity between December and January in raw integer encoding). A US federal holiday flag marks any week containing a major holiday.

**Critical note:** all rolling and lag features are computed using `shift(1)` before the rolling operation. This means every feature value at row `t` is derived exclusively from data before time `t`, ensuring zero data leakage across the train/validation split.

---

## Model Selection

For each state, all four models are trained on the data before the last 8 weeks and evaluated on those final 8 weeks. The model with the lowest RMSE on the validation set is declared the winner and used for the final 8-week future forecast. RMSE is chosen over MAE because it penalises large prediction errors more heavily — in sales forecasting, missing a demand spike by a large margin is far more costly than many small errors.

---

## Project Structure

```
sales_forecasting/
├── data/
│   └── Forecasting_Case-_Study.xlsx   ← Raw dataset
├── src/
│   ├── preprocessing.py               ← Data loading, date parsing, resampling
│   ├── feature_engineering.py         ← Feature creation + train/val split
│   ├── model_selector.py              ← Trains all 4, picks best by RMSE
│   └── models/
│       ├── arima_model.py             ← ARIMA training, evaluation, forecast
│       ├── prophet_model.py           ← Prophet training, evaluation, forecast
│       ├── xgboost_model.py           ← XGBoost training, evaluation, forecast
│       └── lstm_model.py              ← LSTM training, evaluation, forecast
├── api/
│   └── main.py                        ← FastAPI REST API
├── trained_models/                    ← Auto-populated after run_training.py
├── results/
│   └── all_states_summary.json        ← Auto-populated after run_training.py
├── run_training.py                    ← Entry point: trains all models
├── run_api.py                         ← Entry point: starts API server
├── requirements.txt
└── README.md
```

---

## Setup & Installation

**Step 1 — Clone the repository and navigate into it:**
```bash
git clone <repo-url>
cd sales_forecasting
```

**Step 2 — Create a virtual environment and install dependencies:**
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Step 3 — Train all models (runs once, takes ~15–30 minutes):**
```bash
python run_training.py
```

**Step 4 — Start the API server:**
```bash
python run_api.py
```

The API is now live at `http://localhost:8000`.

---

## API Reference

All endpoints are also documented interactively at `http://localhost:8000/docs` (Swagger UI).

**GET /** — Health check. Returns system status and whether forecast results are loaded.

**GET /states** — Returns the sorted list of all 43 state names available in the dataset.

**GET /forecast/{state}** — Returns the 8-week sales forecast for the given state, produced by its best-selected model.

Example request:
```
GET http://localhost:8000/forecast/California
```

Example response:
```json
{
  "state": "California",
  "best_model": "xgboost",
  "forecast_dates": ["2023-12-10", "2023-12-17", ..., "2024-01-28"],
  "forecast_values": [1812345678.0, 1798234512.0, "..."],
  "unit": "USD"
}
```

**POST /forecast** — Same as above but accepts a JSON body (useful for states with spaces in the name).

Example request body:
```json
{ "state": "New York" }
```

**GET /model-info/{state}** — Returns the best model selected and the validation RMSE, MAE, and MAPE for all 4 models trained for this state.

**GET /all-forecasts** — Returns 8-week forecasts for all 43 states in a single response. Suitable for dashboards and batch reporting.

---

## Technology Stack

Python 3.11, FastAPI, Uvicorn, Pydantic, pandas, NumPy, scikit-learn, XGBoost, Facebook Prophet, pmdarima, TensorFlow/Keras, joblib, holidays.
