"""
data_loader.py
--------------
Downloads (once) and loads 30 years of daily temperature for New Delhi, and
builds the 365-day climatology (the average for each calendar day).

The first call fetches about 430 kB of CSV from NOAA's NCEI data service and
caches it in ``data/``; every later call reads the local file.
"""

import urllib.request

import numpy as np
import pandas as pd

import config


def download_dataset(force: bool = False):
    if not config.DATA_FILE.exists() or force:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(config.API_URL, headers={"User-Agent": "curl/8.4.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            config.DATA_FILE.write_bytes(resp.read())
    return config.DATA_FILE


def load_daily() -> pd.DataFrame:
    df = pd.read_csv(download_dataset(), parse_dates=["DATE"])
    df = df.rename(columns={"DATE": "date", "TMAX": "tmax", "TMIN": "tmin"})
    df = df[~((df["date"].dt.month == 2) & (df["date"].dt.day == 29))].reset_index(drop=True)
    # calendar-day index 0 ... 364 that ignores leap days
    ref = pd.to_datetime("2001-" + df["date"].dt.strftime("%m-%d"))
    df["k"] = ref.dt.dayofyear - 1
    return df


def climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, count and spread for every calendar day: a 365-sample circular signal."""
    g = df.groupby("k")
    clim = pd.DataFrame({
        "tmax": g["tmax"].mean(), "tmin": g["tmin"].mean(),
        "n_tmax": g["tmax"].count(), "n_tmin": g["tmin"].count(),
        "sd_tmax": g["tmax"].std(),
    }).reindex(range(config.N_DAYS))
    clim["date"] = pd.to_datetime("2001-01-01") + pd.to_timedelta(clim.index, unit="D")
    return clim


def describe(df: pd.DataFrame, clim: pd.DataFrame) -> dict:
    years = df["date"].dt.year
    return {
        "rows": len(df), "tmax_obs": int(df["tmax"].notna().sum()), "tmin_obs": int(df["tmin"].notna().sum()),
        "possible": 30 * config.N_DAYS,
        "complete_years": int((df.groupby(years)["tmax"].count() >= 330).sum()),
        "tmax_max": float(df["tmax"].max()), "tmax_max_date": df.loc[df["tmax"].idxmax(), "date"],
        "tmin_min": float(df["tmin"].min()), "tmin_min_date": df.loc[df["tmin"].idxmin(), "date"],
        "n_min": int(clim["n_tmax"].min()), "n_med": float(clim["n_tmax"].median()),
        "size_kb": config.DATA_FILE.stat().st_size / 1024,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    d = load_daily()
    c = climatology(d)
    print(c.head().to_string())
    print(describe(d, c))
