"""
data_loader.py
--------------
Downloads (once) and loads the hourly Pune weather / solar dataset.

The first call fetches ~7 kB of JSON from the Open-Meteo archive API and caches
it as a CSV inside ``data/``; every later call reads the local CSV, so the
experiment also runs offline.
"""

import json
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

import config


def download_dataset(force: bool = False) -> pd.DataFrame:
    """Fetch the dataset from the Open-Meteo archive API and cache it as CSV."""
    if config.DATA_FILE.exists() and not force:
        return pd.read_csv(config.DATA_FILE, parse_dates=["time"])

    query = urllib.parse.urlencode({
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": config.START_DATE,
        "end_date": config.END_DATE,
        "hourly": ",".join(config.HOURLY_VARIABLES),
        "timezone": config.TIMEZONE,
    })
    with urllib.request.urlopen(f"{config.API_URL}?{query}", timeout=30) as resp:
        payload = json.load(resp)

    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns=config.COLUMN_MAP)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATA_FILE, index=False)
    return df


def load_dataset() -> pd.DataFrame:
    """Return the cleaned dataset with an integer sample index ``n`` (n = 0 at the first hour)."""
    df = download_dataset()
    df = df.sort_values("time").reset_index(drop=True)

    # Reanalysis data has no gaps, but guard against them anyway:
    # linear interpolation in time keeps the uniform sampling grid intact.
    numeric = df.columns.drop("time")
    df[numeric] = df[numeric].interpolate(limit_direction="both")

    df.insert(0, "n", np.arange(len(df)))
    return df


def describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column descriptive statistics used in the report's dataset section."""
    cols = list(config.SIGNAL_INFO)
    stats = df[cols].agg(["count", "mean", "std", "min", "max"]).T
    stats.insert(0, "unit", [config.SIGNAL_INFO[c][1] for c in cols])
    stats.insert(0, "description", [config.SIGNAL_INFO[c][0] for c in cols])
    return stats


if __name__ == "__main__":
    data = load_dataset()
    print(data.head(10).to_string(index=False))
    print(f"\n{len(data)} samples, {data['time'].min()} -> {data['time'].max()}")
    print(describe_dataset(data).round(2).to_string())
