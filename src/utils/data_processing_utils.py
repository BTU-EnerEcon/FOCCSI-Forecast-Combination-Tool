"""Shared data preparation and validation helpers used by the FOCCSI pipeline."""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def safe_rmse(y_true, y_pred):
    """Compute RMSE while ignoring rows with non-finite values."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    valid_mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if not np.any(valid_mask):
        return np.nan
    return np.sqrt(mean_squared_error(y_true_arr[valid_mask], y_pred_arr[valid_mask]))


def contains_invalid_values(data):
    """Return True if any value is NaN/inf after numeric coercion."""
    if isinstance(data, pd.DataFrame):
        numeric_data = data.apply(pd.to_numeric, errors="coerce")
        values = numeric_data.to_numpy(dtype=float)
    elif isinstance(data, pd.Series):
        numeric_data = pd.to_numeric(data, errors="coerce")
        values = numeric_data.to_numpy(dtype=float)
    else:
        values = np.asarray(data, dtype=float)

    return not np.isfinite(values).all()


def apply_benchmark_threshold_filter(df, benchmark_col, threshold):
    """Set benchmark values below threshold to zero (old-pipeline parity behavior)."""
    if threshold is None:
        return df

    if benchmark_col not in df.columns:
        return df

    out = df.copy()
    mask_below_threshold = pd.to_numeric(out[benchmark_col], errors="coerce") < float(threshold)
    out.loc[mask_below_threshold, benchmark_col] = 0
    return out


def merge_price_data(base_df, day_ahead_df, intraday_df, required=False):
    """Merge day-ahead and intraday prices and append Price_Spread by timestamp."""
    if day_ahead_df is None or intraday_df is None:
        if required:
            raise ValueError(
                "Price data is required but could not be loaded. "
                "Ensure day_ahead_df and intraday_df are available with valid columns."
            )
        return base_df

    da_df = day_ahead_df.copy()
    id1_df = intraday_df.copy()

    if da_df.empty or id1_df.empty:
        raise ValueError("Price data frames must not be empty.")

    if da_df.columns[0] != "time":
        da_df = da_df.rename(columns={da_df.columns[0]: "time"})
    if id1_df.columns[0] != "time":
        id1_df = id1_df.rename(columns={id1_df.columns[0]: "time"})

    if "time" not in da_df.columns or "time" not in id1_df.columns:
        raise ValueError("Price data must contain a time column for merging.")

    da_value_cols = [col for col in da_df.columns if col != "time"]
    id1_value_cols = [col for col in id1_df.columns if col != "time"]

    if not da_value_cols or not id1_value_cols:
        raise ValueError("Price data must contain value columns for day-ahead and intraday prices.")

    da_value_col = da_value_cols[0]
    id1_value_col = id1_value_cols[0]

    merged_prices = pd.merge(
        da_df[["time", da_value_col]],
        id1_df[["time", id1_value_col]],
        on="time",
        how="inner",
    )
    merged_prices["Price_Spread"] = merged_prices[id1_value_col] - merged_prices[da_value_col]

    spread_df = merged_prices[["time", "Price_Spread"]].copy()
    spread_df["time"] = pd.to_datetime(spread_df["time"], errors="coerce")
    spread_df = spread_df.dropna(subset=["time"]).drop_duplicates(subset=["time"], keep="last")
    spread_df = spread_df.set_index("time")

    out = base_df.join(spread_df, how="left")
    return out


def _detect_time_column(df):
    """Detect likely timestamp column name in a dataframe."""
    if df is None or df.empty:
        return None

    candidates = {"time", "timestamp", "datetime", "date", "datum", "zeit"}
    for col in df.columns:
        if str(col).strip().lower() in candidates:
            return col

    # Fallback: use first column if it can be mostly parsed as datetime.
    first_col = df.columns[0]
    parsed = pd.to_datetime(df[first_col], errors="coerce")
    parse_ratio = parsed.notna().mean()
    if parse_ratio >= 0.8:
        return first_col

    return None


def _prepare_df_for_time_merge(df, source_name):
    """Normalize one source dataframe for time-based merging."""
    if df is None or df.empty:
        return None, []

    time_col = _detect_time_column(df)
    if time_col is None:
        raise ValueError(
            f"Could not detect a time column in '{source_name}'. "
            "Provide a timestamp column (first column can also be used)."
        )

    out = df.copy()
    if time_col != "time":
        out = out.rename(columns={time_col: "time"})

    out["time"] = pd.to_datetime(out["time"], errors="coerce")
    out = out.dropna(subset=["time"]).sort_values("time")
    out = out.drop_duplicates(subset=["time"], keep="last")

    value_cols = [col for col in out.columns if col != "time"]
    return out, value_cols


def merge_input_dataframes(input_data_cfg):
    """Merge uploaded input dataframes on time and track source value columns."""
    if not isinstance(input_data_cfg, dict):
        raise ValueError("input_data_cfg must be a dictionary.")

    merged_df = None
    source_column_names = {}

    # Keep the base training/evaluation matrix aligned like the old pipeline:
    # merge only forecasting and benchmark-like timeseries on common timestamps.
    core_sources = [
        "forecasts_df",
        "benchmark_df",
        "secondary_benchmark_df",
    ]

    for source_name in core_sources:
        source_df = input_data_cfg.get(source_name)
        if not isinstance(source_df, pd.DataFrame) or source_df.empty:
            continue

        prepared_df, source_cols = _prepare_df_for_time_merge(source_df, source_name)
        if prepared_df is None:
            continue
        if not source_cols:
            continue

        # Avoid clobbering existing columns by renaming collisions in the incoming frame.
        if merged_df is not None:
            existing_cols = set(merged_df.columns) - {"time"}
            rename_map = {}
            for col in source_cols:
                if col in existing_cols:
                    rename_map[col] = f"{source_name}_{col}"
            if rename_map:
                prepared_df = prepared_df.rename(columns=rename_map)
                source_cols = [rename_map.get(col, col) for col in source_cols]

        source_column_names[source_name] = source_cols

        if merged_df is None:
            merged_df = prepared_df
        else:
            merged_df = pd.merge(merged_df, prepared_df, on="time", how="inner")

    if merged_df is None:
        raise ValueError("No non-empty pandas DataFrames found in input_data config.")

    merged_df = merged_df.sort_values("time").reset_index(drop=True)
    return merged_df, source_column_names, "time"


def get_latest_available_target_date_df(test_date, cutoff_df):
    """Map test_date to the latest allowed benchmark target date using cutoff rules."""
    if cutoff_df is None or cutoff_df.empty:
        return None

    normalized_cutoff_df = cutoff_df.copy()
    if "cutoff_date" not in normalized_cutoff_df.columns:
        if "time" in normalized_cutoff_df.columns:
            normalized_cutoff_df = normalized_cutoff_df.rename(columns={"time": "cutoff_date"})
        else:
            candidate_columns = [col for col in normalized_cutoff_df.columns if "date" in str(col).lower()]
            if candidate_columns:
                normalized_cutoff_df = normalized_cutoff_df.rename(columns={candidate_columns[0]: "cutoff_date"})
            else:
                normalized_cutoff_df = normalized_cutoff_df.rename(
                    columns={normalized_cutoff_df.columns[0]: "cutoff_date"}
                )

    normalized_cutoff_df["cutoff_date"] = pd.to_datetime(
        normalized_cutoff_df["cutoff_date"],
        errors="coerce",
    )
    normalized_cutoff_df = normalized_cutoff_df.dropna(subset=["cutoff_date"]).copy()
    if normalized_cutoff_df.empty:
        return None

    normalized_cutoff_df["year"] = normalized_cutoff_df["cutoff_date"].dt.year
    normalized_cutoff_df["month"] = normalized_cutoff_df["cutoff_date"].dt.month

    # Extract year and month
    year = test_date.year
    month = test_date.month

    # Get cutoff date from DataFrame
    cutoff_row = normalized_cutoff_df[
        (normalized_cutoff_df.year == year) & (normalized_cutoff_df.month == month)
    ]
    if cutoff_row.empty:
        return None # no cutoff data for this month

    cutoff_day = cutoff_row["cutoff_date"].iloc[0]

    # Compare test_date to cutoff_day
    if test_date < cutoff_day:
        available_month = pd.Timestamp(year, month, 1) - pd.DateOffset(months=2)
    else:
        available_month = pd.Timestamp(year, month, 1) - pd.DateOffset(months=1)

    # Return the last calendar day of that available month
    return available_month + pd.offsets.MonthEnd(0)+ pd.Timedelta(days=1)


def preprocess_forecast_column(B, B_hist, criteria):
    """Apply threshold-based cleaning/invalidating to forecast and historical series."""
    # Following-day forecast
    # Threshold keys for B: IV, NN, RN, RZ, RV
    number_impossible_values = 0
    number_missing_wdh = 0
    number_missing_total = 0
    number_repetition_zero = 0
    number_repetition_value = 0
    break_var = 0
    for i in B.index:
        if pd.notna(B[i]):
            if B[i] < 0:
                number_impossible_values += 1
            # IV: too many negative values -> invalidate series
            if number_impossible_values > criteria['IV']:
                B[:] = 0
                break_var = 1
                break
        if i > B.index[0] and pd.notna(B[i]) and pd.notna(B[i-1]):
            if B[i] == 0 and B[i] == B[i-1]:
                number_repetition_zero += 1
                # RZ: too many repeated zeros in a row
                if number_repetition_zero > criteria['RZ']:
                    B[:] = 0
                    break_var = 1
                    break
            else:
                number_repetition_zero = 0
            if B[i] != 0 and B[i] == B[i-1]:
                number_repetition_value += 1
                # RV: too many repeated non-zero values in a row
                if number_repetition_value > criteria['RV']:
                    B[:] = 0
                    break_var = 1
                    break
            else:
                number_repetition_value = 0
        if pd.isna(B[i]):
            number_missing_wdh += 1
            number_missing_total += 1
            # NN or RN: too many missing values (consecutive or total)
            if number_missing_wdh > criteria['NN'] or number_missing_total > criteria['RN']:
                B[:] = 0
                break_var = 1
                break
        else:
            number_missing_wdh = 0
    if break_var == 0:
        # If B passes hard checks, repair negatives/missing values.
        B[B < 0] = np.nan
        B.interpolate(method='linear', inplace=True)
        B.fillna(0, inplace=True)

    # Historical forecast
    # Threshold keys for B_hist: HIV, HNN, HRN, HRZ, HRV
    number_impossible_values_hist = 0
    number_missing_wdh_hist = 0
    number_missing_total_hist = 0
    number_repetition_zero_hist = 0
    number_repetition_value_hist = 0
    break_var_hist = 0
    for i in B_hist.index:
        if pd.notna(B_hist[i]):
            if B_hist[i] < 0:
                number_impossible_values_hist += 1
            # HIV: too many negative historical values
            if number_impossible_values_hist > criteria['HIV']:
                B_hist[:] = 0
                break_var_hist = 1
                break
        if i > B_hist.index[0] and pd.notna(B_hist[i]) and pd.notna(B_hist[i-1]):
            if B_hist[i] == 0 and B_hist[i] == B_hist[i-1]:
                number_repetition_zero_hist += 1
                # HRZ: too many repeated historical zeros in a row
                if number_repetition_zero_hist > criteria['HRZ']:
                    B_hist[:] = 0
                    break_var_hist = 1
                    break
            else:
                number_repetition_zero_hist = 0
            if B_hist[i] != 0 and B_hist[i] == B_hist[i-1]:
                number_repetition_value_hist += 1
                # HRV: too many repeated historical non-zero values in a row
                if number_repetition_value_hist > criteria['HRV']:
                    B_hist[:] = 0
                    break_var_hist = 1
                    break
            else:
                number_repetition_value_hist = 0
        if pd.isna(B_hist[i]):
            number_missing_wdh_hist += 1
            number_missing_total_hist += 1
            # HNN or HRN: too many historical missing values
            if number_missing_wdh_hist > criteria['HNN'] or number_missing_total_hist > criteria['HRN']:
                B_hist[:] = 0
                break_var_hist = 1
                break
        else:
            number_missing_wdh_hist = 0
    if break_var_hist == 0:
        # If B_hist passes hard checks, repair negatives/missing values.
        B_hist[B_hist < 0] = np.nan
        B_hist.interpolate(method='linear', inplace=True)
        B_hist.fillna(0, inplace=True)
    return B, B_hist
