"""Calculate balancing-cost errors from rolling-window forecast results."""

import pandas as pd


_LEVEL_TO_FREQ = {
    "hourly": "h",
    "daily": "D",
    "weekly": "W",
    "monthly": "ME",
    "quarterly": "QE",
    "yearly": "YE",
}


def _price_series(price_df, name):
    if price_df is None or price_df.empty:
        raise ValueError(f"{name} price data is required for economic analysis.")

    time_col = "time" if "time" in price_df.columns else price_df.columns[0]
    value_cols = [col for col in price_df.columns if col != time_col]
    if not value_cols:
        raise ValueError(f"{name} price data must contain a price column.")

    prices = price_df[[time_col, value_cols[0]]].copy()
    prices[time_col] = pd.to_datetime(prices[time_col], errors="coerce")
    prices[value_cols[0]] = pd.to_numeric(prices[value_cols[0]], errors="coerce")
    return prices.dropna(subset=[time_col]).drop_duplicates(time_col, keep="last").set_index(time_col)[value_cols[0]]


def compute_economic_analysis(data_processed, forecast_cols, benchmark_cols, input_data, config):
    """Return interval and calendar-summed balancing costs for each forecast/model."""
    if data_processed is None or data_processed.empty:
        return {}
    if not benchmark_cols:
        raise ValueError("A benchmark column is required for economic analysis.")

    day_ahead_prices = _price_series(input_data.get("day_ahead_df"), "Day-ahead")
    intraday_prices = _price_series(input_data.get("intraday_df"), "Intraday")

    results = data_processed.copy()
    results.index = pd.to_datetime(results.index, errors="coerce")
    results = results[~results.index.isna()].copy()
    if results.empty:
        return {}

    results["Day_Ahead_Price"] = day_ahead_prices.reindex(results.index)
    results["Intraday_Price"] = intraday_prices.reindex(results.index)
    results["Price_Spread"] = results["Intraday_Price"] - results["Day_Ahead_Price"]

    target_col = benchmark_cols[0]
    model_cols = [col for col in forecast_cols if col in results.columns]
    model_cols.extend(col for col in results.columns if col.startswith("Pred_") and col not in model_cols)

    cost_cols = []
    for model_col in model_cols:
        cost_col = f"Balancing_Cost_{model_col}"
        forecast_error = pd.to_numeric(results[model_col], errors="coerce") - pd.to_numeric(results[target_col], errors="coerce")
        results[cost_col] = forecast_error * results["Price_Spread"]
        cost_cols.append(cost_col)

    if not cost_cols:
        return {}

    output_cols = [target_col, "Day_Ahead_Price", "Intraday_Price", "Price_Spread", *cost_cols]
    interval_data = results[output_cols]
    outputs = {"Economic_Interval_Costs": interval_data}

    levels = config.get("postprocessing", {}).get("economic_aggregation_levels", [])
    for level in levels:
        normalized_level = str(level).strip().lower()
        freq = _LEVEL_TO_FREQ.get(normalized_level)
        if freq:
            outputs[f"Economic_Costs_{normalized_level}"] = interval_data[cost_cols].resample(freq).sum(min_count=1)

    outputs["Economic_Costs_Overall"] = interval_data[cost_cols].sum(min_count=1).to_frame().T
    return outputs