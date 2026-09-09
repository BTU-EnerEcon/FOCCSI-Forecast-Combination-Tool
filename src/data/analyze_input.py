import pandas as pd

def analyze_data(forecast_df):
    """Analyze the given forecast DataFrame and return a summary dictionary."""

    result = {}

    result["columns"] = list(forecast_df.columns)
    result["start"] = str(forecast_df.iloc[:, 0].iloc[0])
    result["end"] = str(forecast_df.iloc[:, 0].iloc[-1])

    # missing values
    result["missing"] = forecast_df.isna().sum().to_dict()

    # time frequency (simple estimate)
    try:
        timestamps = pd.to_datetime(forecast_df.iloc[:, 0], dayfirst=True)
        diff = timestamps.diff().dropna().mode()[0]
        result["frequency"] = str(diff)
    except:
        result["frequency"] = "Unknown"

    return result
def analyze_selected_config(config):
    missing_items = []

    input_data = config.get("input_data", {}) if isinstance(config, dict) else {}
    models = config.get("models", {}) if isinstance(config, dict) else {}
    rolling_window = config.get("rolling_window", {}) if isinstance(config, dict) else {}

    forecasts_df = input_data.get("forecasts_df")
    benchmark_df = input_data.get("benchmark_df")
    secondary_benchmark_df = input_data.get("secondary_benchmark_df")
    secondary_benchmark_cutoff_df = input_data.get("secondary_benchmark_cutoff_df")

    if forecasts_df is None:
        missing_items.append("Input data missing: forecasts_df")

    if benchmark_df is None:
        missing_items.append("Input data missing: benchmark_df")

    if secondary_benchmark_df is not None and secondary_benchmark_cutoff_df is None:
        missing_items.append(
            "Input data missing: secondary_benchmark_cutoff_df is required when secondary_benchmark_df is provided"
        )

    run_delnet = bool(models.get("delnet", False))
    run_pso = bool(models.get("pso", False))
    if not (run_delnet or run_pso):
        missing_items.append("Model selection missing: enable either DELNET or PSO")

    training_days = rolling_window.get("training_days", [])
    if not training_days:
        missing_items.append("Rolling window missing: training_days cannot be empty")

    horizon = rolling_window.get("horizon")
    try:
        horizon_value = float(horizon)
    except (TypeError, ValueError):
        horizon_value = None

    if horizon_value is None or horizon_value <= 0:
        missing_items.append("Rolling window invalid: forecast_horizon must be greater than 0")

    if missing_items:
        message = "Missing or invalid configuration:\n- " + "\n- ".join(missing_items)
        return True, message

    return False, "All required configuration and input data are provided."

def _format_timestamp(ts):
    if pd.isna(ts):
        return "Unknown"
    return ts.strftime("%d.%m.%Y %H:%M")


def _infer_frequency(timestamps):
    diffs = timestamps.sort_values().diff().dropna()
    if diffs.empty:
        return None
    return diffs.mode().iloc[0]


def _find_missing_periods(timestamps, inferred_freq):
    if inferred_freq is None or len(timestamps) < 2:
        return []

    ts_sorted = timestamps.dropna().sort_values().drop_duplicates()
    if ts_sorted.empty:
        return []

    expected = pd.date_range(ts_sorted.min(), ts_sorted.max(), freq=inferred_freq)
    missing = expected.difference(ts_sorted)

    if missing.empty:
        return []

    periods = []
    start = missing[0]
    prev = missing[0]

    for current in missing[1:]:
        if current - prev != inferred_freq:
            periods.append(
                {
                    "start": _format_timestamp(start),
                    "end": _format_timestamp(prev),
                    "missing_steps": int(((prev - start) / inferred_freq) + 1),
                }
            )
            start = current
        prev = current

    periods.append(
        {
            "start": _format_timestamp(start),
            "end": _format_timestamp(prev),
            "missing_steps": int(((prev - start) / inferred_freq) + 1),
        }
    )
    return periods


def _find_missing_periods_for_series(timestamps, missing_mask, inferred_freq):
    valid_ts = timestamps[~timestamps.isna()]
    valid_missing_mask = missing_mask[~timestamps.isna()]

    missing_ts = valid_ts[valid_missing_mask]
    if missing_ts.empty:
        return []

    missing_ts = missing_ts.sort_values().drop_duplicates()

    if inferred_freq is None:
        return [
            {
                "start": _format_timestamp(ts),
                "end": _format_timestamp(ts),
                "missing_steps": 1,
            }
            for ts in missing_ts
        ]

    periods = []
    start = missing_ts.iloc[0]
    prev = missing_ts.iloc[0]

    for current in missing_ts.iloc[1:]:
        if current - prev != inferred_freq:
            periods.append(
                {
                    "start": _format_timestamp(start),
                    "end": _format_timestamp(prev),
                    "missing_steps": int(((prev - start) / inferred_freq) + 1),
                }
            )
            start = current
        prev = current

    periods.append(
        {
            "start": _format_timestamp(start),
            "end": _format_timestamp(prev),
            "missing_steps": int(((prev - start) / inferred_freq) + 1),
        }
    )

    return periods


def analyze_uploaded_data(df):
    result = {}

    if df is None or df.empty:
        return {"error": "Dataset is empty or not loaded."}

    data = df.copy()

    time_col = "time" if "time" in data.columns else data.columns[0]
    timestamps = pd.to_datetime(data[time_col], dayfirst=True, errors="coerce")
    value_columns = [col for col in data.columns if col != time_col]

    inferred_freq = _infer_frequency(timestamps)

    result["columns"] = list(data.columns)
    result["start"] = _format_timestamp(timestamps.min())
    result["end"] = _format_timestamp(timestamps.max())
    result["frequency"] = str(inferred_freq) if inferred_freq is not None else "Unknown"

    summary_rows = []
    for col in value_columns:
        numeric_col = pd.to_numeric(data[col], errors="coerce")
        valid_numeric = numeric_col.dropna()

        if valid_numeric.empty:
            mean_value = None
            max_value = None
            min_value = None
            zero_values = None
        else:
            mean_value = float(valid_numeric.mean())
            max_value = float(valid_numeric.max())
            min_value = float(valid_numeric.min())
            zero_values = int((valid_numeric == 0).sum())

        summary_rows.append(
            {
                "column": col,
                "zero_values": zero_values,
                "mean": mean_value,
                "max": max_value,
                "min": min_value,
            }
        )

    result["column_summary"] = pd.DataFrame(summary_rows)
    result["missing_periods"] = _find_missing_periods(timestamps, inferred_freq)
    result["missing_values"] = data[value_columns].isna().sum().to_dict()

    missing_periods_by_column = {}
    for col in value_columns:
        periods = _find_missing_periods_for_series(
            timestamps,
            data[col].isna(),
            inferred_freq,
        )
        if periods:
            missing_periods_by_column[col] = periods

    result["missing_periods_by_column"] = missing_periods_by_column
    result["value_columns"] = value_columns

    return result