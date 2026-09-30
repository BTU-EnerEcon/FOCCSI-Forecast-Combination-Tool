import pandas as pd

_TIME_COLUMN_CANDIDATES = {"time", "timestamp", "datetime", "date", "datum", "zeit"}


def _to_datetime(series):
    # helper function to convert time column to datetime, with error handling
    return pd.to_datetime(series, dayfirst=True, format="mixed", errors="coerce")


def _detect_time_column(df):
    """Find the timestamp column by name, falling back to the best-parsing column."""
    for col in df.columns:
        if str(col).strip().lower() in _TIME_COLUMN_CANDIDATES:
            return col

    best_col, best_ratio = None, 0.0
    for col in df.columns:
        ratio = _to_datetime(df[col]).notna().mean()
        if ratio > best_ratio:
            best_col, best_ratio = col, ratio

    if best_col is None or best_ratio < 0.8:
        raise ValueError("Could not detect a timestamp column in the uploaded CSV.")
    return best_col


def load_csv(file_path):
    # Read as strings first: parsing numbers eagerly (thousands=".") would corrupt
    # dd.mm.yyyy dates in single-column files (e.g. "01.05.2025" -> 1052025),
    # which then breaks time-column detection.
    raw_df = pd.read_csv(file_path, sep=";", dtype=str)
    time_column = _detect_time_column(raw_df)

    df = raw_df.rename(columns={time_column: "time"})
    value_columns = [col for col in df.columns if col != "time"]
    for col in value_columns:
        df[col] = pd.to_numeric(
            df[col].str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    df["time"] = _to_datetime(df["time"])
    df.drop_duplicates(subset="time", inplace=True)
    return df

