"""Aggregate window-level RMSE metrics to calendar periods and overall summaries."""

import pandas as pd


_LEVEL_TO_FREQ = {
    "hourly": "H",
    "daily": "D",
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "Y",
    "overall": None,
    "all": None,
    "total": None,
}


def _normalize_levels(levels):
    """Normalize user-provided aggregation level settings to lowercase list form."""
    if levels is None:
        return []
    if isinstance(levels, str):
        levels = [levels]
    return [str(level).strip().lower() for level in levels if str(level).strip()]


def compute_rmse_aggregations(data_rmse, config):
    """Aggregate window-level RMSE outputs to configured calendar periods.

    Returns a dict of DataFrames keyed by output name, e.g. rmse_agg_daily.
    """
    if data_rmse is None or data_rmse.empty:
        return {}

    post_cfg = config.get("postprocessing", {})
    levels = post_cfg.get("rmse_aggregation_levels", post_cfg.get("aggregation_level", []))
    normalized_levels = _normalize_levels(levels)

    df = data_rmse.copy()
    if "test_date" not in df.columns:
        return {}

    df["test_date"] = pd.to_datetime(df["test_date"], errors="coerce")
    df = df.dropna(subset=["test_date"]).copy()
    if df.empty:
        return {}

    rmse_cols = [col for col in df.columns if col.startswith("RMSE_")]
    if not rmse_cols:
        return {}

    outputs = {}

    # Aggregate configured calendar levels.
    for level in normalized_levels:
        freq = _LEVEL_TO_FREQ.get(level, level.upper())
        if freq is None:
            agg_df = df[rmse_cols].mean(numeric_only=True).to_frame().T
            agg_df.insert(0, "Period", "Overall")
            agg_df.insert(1, "N_windows", len(df))
            outputs["rmse_agg_overall"] = agg_df
            continue

        period_series = df["test_date"].dt.to_period(freq).rename("Period")
        agg_df = df.groupby(period_series)[rmse_cols].mean().reset_index()
        counts = df.groupby(period_series).size().rename("N_windows").reset_index()
        agg_df = agg_df.merge(counts, on="Period", how="left")
        outputs[f"rmse_agg_{level}"] = agg_df

    if "rmse_agg_overall" not in outputs:
        # Always provide an overall benchmark across all windows.
        overall = df[rmse_cols].mean(numeric_only=True).to_frame().T
        overall.insert(0, "Period", "Overall")
        overall.insert(1, "N_windows", len(df))
        outputs["rmse_agg_overall"] = overall

    return outputs
