"""
preprocessing.py
----------------
Handles all data loading, cleaning, and transformation for the forecasting system.

Key responsibilities:
  1. Load the raw Excel dataset
  2. Parse the mixed date formats (Excel datetime objects + DD-MM-YYYY strings)
  3. Resample each state's irregular time series to a clean weekly frequency
  4. Fill any gaps created by resampling using linear interpolation
  5. Return a ready-to-use dictionary of { state_name -> weekly DataFrame }
"""

import pandas as pd
import numpy as np
import logging
import os

# Configure a clean logger for this module
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _parse_date(value) -> pd.Timestamp:
    """
    Handles the two date formats present in the dataset:
      - Python datetime objects (produced by Excel's native date cells)
      - String dates in DD-MM-YYYY format (e.g., "31-10-2021")

    Returns a proper pd.Timestamp for every input.
    """
    if isinstance(value, str):
        # dayfirst=True tells pandas that the first number is the day,
        # which correctly handles the DD-MM-YYYY string format
        return pd.to_datetime(value, dayfirst=True)
    # For datetime objects coming directly from openpyxl / Excel
    return pd.to_datetime(value)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads the raw Excel file, applies date parsing, and does basic cleanup.

    Args:
        filepath: Absolute or relative path to the .xlsx file.

    Returns:
        A clean DataFrame with columns: ['State', 'Date', 'Total']
        Date column is fully normalized to pd.Timestamp (weekly Sunday-anchored).
    """
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_excel(filepath, sheet_name="Sheet1")

    # Apply the mixed-format date parser to every row
    df["Date"] = df["Date"].apply(_parse_date)

    # Drop the Category column — it contains only one value ("Beverages")
    # and carries no information for modeling
    df = df.drop(columns=["Category"])

    # Ensure Total is float (handle any edge-case formatting from Excel)
    df["Total"] = pd.to_numeric(df["Total"], errors="coerce")

    logger.info(f"Raw data loaded: {df.shape[0]} rows, {df['State'].nunique()} states")
    logger.info(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    logger.info(f"Missing values: {df.isnull().sum().to_dict()}")

    return df


def resample_to_weekly(df_state: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a single state's irregular time series and resamples it to
    a clean weekly frequency (anchored to every Sunday, 'W' in pandas).

    Why SUM and not mean?
    Because 'Total' represents total sales dollars within a reporting period,
    not an average rate. Summing correctly aggregates multiple observations
    that fall within the same calendar week.

    After resampling, any weeks with no original data (gaps) are filled using
    linear interpolation — which is appropriate for a smooth sales trend signal.

    Args:
        df_state: DataFrame with columns ['Date', 'Total'] for a single state.

    Returns:
        A DataFrame with columns ['Date', 'Total'] at weekly frequency,
        with Date as a proper DatetimeIndex-derived column.
    """
    # Set Date as the index so we can use pandas' resample() function
    df_indexed = df_state.set_index("Date").sort_index()

    # Resample to weekly, summing all sales that fall in the same week
    # 'W' anchors each bin to Sunday (standard calendar week end)
    weekly = df_indexed["Total"].resample("W").sum()

    # Replace any zero-sum weeks (weeks where no data fell) with NaN
    # so interpolation fills them meaningfully instead of treating 0 as real data
    weekly = weekly.replace(0, np.nan)

    # Linear interpolation fills gaps between known values smoothly
    # limit_direction='both' also handles NaNs at the very start/end
    weekly = weekly.interpolate(method="linear", limit_direction="both")

    # Reset index so Date becomes a column again (easier to work with downstream)
    result = weekly.reset_index()
    result.columns = ["Date", "Total"]

    return result


def prepare_all_states(filepath: str) -> dict:
    """
    Master function: loads raw data, then produces a clean weekly time series
    for every state in the dataset.

    Returns:
        A dict where:
          key   = state name (str), e.g. "California"
          value = DataFrame with columns ['Date', 'Total'] at weekly frequency
    """
    df_raw = load_raw_data(filepath)
    states = sorted(df_raw["State"].unique())
    state_data = {}

    for state in states:
        df_state = df_raw[df_raw["State"] == state][["Date", "Total"]].copy()
        weekly_df = resample_to_weekly(df_state)
        state_data[state] = weekly_df

    # Validate: all states should have the same number of weekly periods
    lengths = {s: len(v) for s, v in state_data.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) == 1:
        n_weeks = unique_lengths.pop()
        logger.info(f"All {len(states)} states resampled to {n_weeks} weekly periods ✅")
    else:
        logger.warning(f"States have different week counts: {lengths}")

    # Log one sample to visually verify the output looks correct
    sample_state = "California"
    sample = state_data[sample_state]
    logger.info(
        f"\nSample ({sample_state}) — first 5 weeks:\n"
        f"{sample.head().to_string(index=False)}"
    )
    logger.info(
        f"\nSample ({sample_state}) — last 5 weeks:\n"
        f"{sample.tail().to_string(index=False)}"
    )

    return state_data


if __name__ == "__main__":
    # Quick standalone test — run this file directly to verify preprocessing works
    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "Forecasting_Case-_Study.xlsx"
    )
    state_data = prepare_all_states(DATA_PATH)
    print(f"\n✅ Total states processed: {len(state_data)}")
    print(f"✅ Weeks per state: {len(list(state_data.values())[0])}")
