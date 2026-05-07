"""
feature_engineering.py
-----------------------
Generates all predictive features required by the ML models (XGBoost, LSTM).

The JD explicitly requires:
  - Lag features: t-1, t-7, t-30  (interpreted as 1-week, 7-week, 30-week lags
                                    on the weekly-resampled time series)
  - Rolling mean / std
  - Day of week, month, holiday flag
  - Train / validation split using time series logic (no leakage)

Why these features work for sales forecasting:
  - Lag features: capture momentum and recent trajectory of sales
  - Rolling stats: capture the local trend and volatility window
  - Calendar features: capture seasonality patterns (summer peaks, holiday bumps)
  - Holiday flag: captures demand spikes around major US shopping holidays
"""

import pandas as pd
import numpy as np
import holidays
from typing import Tuple


# U.S. federal holidays — used to flag weeks that contain a major holiday
US_HOLIDAYS = holidays.US()


def _is_holiday_week(date: pd.Timestamp) -> int:
    """
    Returns 1 if any day within the 7-day window ending on 'date'
    (i.e., the current week) is a U.S. federal holiday, else 0.

    We check the entire 7-day window because weekly data means the sales
    for an entire week are lumped together — a holiday on any day of that
    week will influence the weekly total.
    """
    week_start = date - pd.Timedelta(days=6)
    for day_offset in range(7):
        check_date = (week_start + pd.Timedelta(days=day_offset)).date()
        if check_date in US_HOLIDAYS:
            return 1
    return 0


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a single state's weekly DataFrame and adds all engineered features.

    Input columns : ['Date', 'Total']
    Output columns: ['Date', 'Total', + all feature columns listed below]

    Rows with NaN (created by lag operations at the start of the series)
    are dropped to ensure no model receives NaN inputs.

    Args:
        df: Weekly DataFrame for one state, sorted ascending by Date.

    Returns:
        Feature-enriched DataFrame with NaN rows removed.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)

    # ── LAG FEATURES ─────────────────────────────────────────────────────────
    # t-1:  Sales from 1 week ago  → captures short-term momentum
    # t-7:  Sales from 7 weeks ago → captures ~2-month cyclical pattern
    # t-30: Sales from 30 weeks ago → captures ~7-month seasonal pattern
    df["lag_1"]  = df["Total"].shift(1)
    df["lag_7"]  = df["Total"].shift(7)
    df["lag_30"] = df["Total"].shift(30)

    # ── ROLLING STATISTICS ───────────────────────────────────────────────────
    # 4-week rolling mean:  recent local average (short-term trend signal)
    # 8-week rolling mean:  medium-term trend signal (2-month window)
    # 4-week rolling std:   recent volatility (helps model handle noisy weeks)
    # 8-week rolling std:   medium-term volatility
    # min_periods=1 prevents NaN at early rows for the rolling stats
    df["rolling_mean_4"]  = df["Total"].shift(1).rolling(window=4,  min_periods=1).mean()
    df["rolling_mean_8"]  = df["Total"].shift(1).rolling(window=8,  min_periods=1).mean()
    df["rolling_std_4"]   = df["Total"].shift(1).rolling(window=4,  min_periods=1).std().fillna(0)
    df["rolling_std_8"]   = df["Total"].shift(1).rolling(window=8,  min_periods=1).std().fillna(0)

    # ── CALENDAR / SEASONALITY FEATURES ─────────────────────────────────────
    # After weekly resampling, every date is a Sunday (day of week = 6).
    # Instead we use richer calendar signals that do vary across weeks:
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["month"]        = df["Date"].dt.month          # 1–12
    df["quarter"]      = df["Date"].dt.quarter        # 1–4
    df["year"]         = df["Date"].dt.year           # captures long-term trend

    # Sine/cosine encoding of week_of_year makes seasonality cyclically smooth
    # (avoids the discontinuity between week 52 and week 1 in raw integer form)
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    # Same for month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── HOLIDAY FLAG ─────────────────────────────────────────────────────────
    df["is_holiday_week"] = df["Date"].apply(_is_holiday_week)

    # ── DROP NaN ROWS ────────────────────────────────────────────────────────
    # Lag t-30 creates 30 NaN rows at the start — we must drop all of them
    # before any model sees this data. This is safe because we have 256 weeks.
    df = df.dropna().reset_index(drop=True)

    return df


def get_feature_columns() -> list:
    """
    Returns the canonical ordered list of feature column names used by
    XGBoost and LSTM models. Keeping this in one place ensures consistency
    between training and inference.
    """
    return [
        "lag_1", "lag_7", "lag_30",
        "rolling_mean_4", "rolling_mean_8",
        "rolling_std_4", "rolling_std_8",
        "week_of_year", "month", "quarter", "year",
        "week_sin", "week_cos",
        "month_sin", "month_cos",
        "is_holiday_week",
    ]


def train_val_split(
    df: pd.DataFrame,
    val_weeks: int = 8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a feature-engineered DataFrame into train and validation sets
    using strict time-series logic — the validation set is always the LAST
    `val_weeks` rows, and training is everything before it.

    This guarantees ZERO data leakage: the model never trains on future data.

    Args:
        df:        Feature-engineered DataFrame for one state.
        val_weeks: Number of weeks to hold out for validation (default: 8,
                   because we want to evaluate on the same horizon we forecast).

    Returns:
        (train_df, val_df) — both DataFrames with identical columns.
    """
    assert len(df) > val_weeks + 30, (
        f"Not enough data: need >{val_weeks + 30} rows, got {len(df)}"
    )
    train_df = df.iloc[:-val_weeks].copy()
    val_df   = df.iloc[-val_weeks:].copy()
    return train_df, val_df


if __name__ == "__main__":
    # ── Standalone verification ───────────────────────────────────────────────
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.preprocessing import prepare_all_states

    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "Forecasting_Case-_Study.xlsx"
    )
    state_data = prepare_all_states(DATA_PATH)

    # Test on California
    sample = create_features(state_data["California"])
    train, val = train_val_split(sample)

    print(f"\n✅ Feature engineering complete for California")
    print(f"   Total rows after feature creation : {len(sample)}")
    print(f"   Training rows                     : {len(train)}")
    print(f"   Validation rows                   : {len(val)}")
    print(f"   Feature columns                   : {get_feature_columns()}")
    print(f"\n   First 3 rows of engineered data:")
    print(sample[["Date", "Total"] + get_feature_columns()].head(3).to_string(index=False))
    print(f"\n   Holiday weeks in dataset: {sample['is_holiday_week'].sum()}")
