import pandas as pd



def _to_datetime(series):
    # helper function to convert time column to datetime, with error handling
    return pd.to_datetime(series, dayfirst=True, format="mixed", errors="coerce")


def load_csv(file_path):
    # load csv with appropriate parsing, and convert time column to datetime
    df = pd.read_csv(file_path, sep=";", decimal=",", thousands=".")
    time_column = df.columns[df.dtypes == "object"][0]  # assuming the first object column is the time column
    df.rename(columns={time_column: "time"}, inplace=True)
    df["time"] = _to_datetime(df["time"])
    df.drop_duplicates(subset='time', inplace=True)
    return df

