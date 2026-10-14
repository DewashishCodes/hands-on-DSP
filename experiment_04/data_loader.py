"""
data_loader.py
--------------
Downloads (once) and loads one year of hourly wind reports for Mumbai airport.

The first call fetches about 290 kB of CSV from the Iowa Environmental Mesonet
(IEM) METAR archive and caches it in ``data/``; every later call reads the
local file, so the experiment also runs offline.
"""

import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

import config


def download_dataset(force: bool = False) -> pd.DataFrame:
    if not config.DATA_FILE.exists() or force:
        query = urllib.parse.urlencode({
            "station": config.STATION, "data": ["drct", "sknt"],
            "year1": config.YEAR, "month1": 1, "day1": 1,
            "year2": config.YEAR + 1, "month2": 1, "day2": 1,
            "tz": "Asia/Kolkata", "format": "onlycomma", "latlon": "no",
            "missing": "M", "trace": "T", "direct": "no", "report_type": 3,
        }, doseq=True)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{config.API_URL}?{query}", timeout=120) as resp:
            config.DATA_FILE.write_bytes(resp.read())
    return pd.read_csv(config.DATA_FILE, na_values="M", parse_dates=["valid"])


def load_dataset() -> pd.DataFrame:
    """Clean reports with a direction-bin index k (0 ... 35) and a calm / variable flag."""
    df = download_dataset().rename(columns={"valid": "time", "drct": "dir_deg", "sknt": "speed_kt"})
    df = df.dropna(subset=["speed_kt"]).sort_values("time").reset_index(drop=True)
    df["calm"] = df["speed_kt"] < config.CALM_KT
    df["variable"] = df["dir_deg"].isna() & ~df["calm"]
    ok = ~df["calm"] & ~df["variable"]
    df["k"] = -1
    df.loc[ok, "k"] = (np.round(df.loc[ok, "dir_deg"] / config.BIN_DEG).astype(int)) % config.N_BINS
    df["month"] = df["time"].dt.month
    return df


def describe_dataset(df: pd.DataFrame) -> dict:
    return {
        "reports": len(df),
        "start": df["time"].min(), "end": df["time"].max(),
        "calm": int(df["calm"].sum()), "variable": int(df["variable"].sum()),
        "directional": int((df["k"] >= 0).sum()),
        "speed_mean": float(df["speed_kt"].mean()), "speed_max": float(df["speed_kt"].max()),
        "speed_p95": float(df["speed_kt"].quantile(0.95)),
        "size_kb": config.DATA_FILE.stat().st_size / 1024,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    d = load_dataset()
    print(d.head(8).to_string(index=False))
    print(describe_dataset(d))
