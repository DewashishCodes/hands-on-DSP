"""
data_loader.py
--------------
Downloads (once) and loads the hourly water levels of the five tide gauges.

The first call fetches five CSV files of about 25 kB each from the NOAA CO-OPS
data API and merges them into one cached file in ``data/``; every later call
reads the local file, so the experiment also runs offline.
"""

import io
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

import config


def _fetch_station(station_id):
    query = urllib.parse.urlencode({
        "product": "hourly_height", "station": station_id,
        "begin_date": config.BEGIN, "end_date": config.END,
        "datum": "MSL", "time_zone": "gmt", "units": "metric",
        "format": "csv", "application": "dsp_lab",
    })
    with urllib.request.urlopen(f"{config.API_URL}?{query}", timeout=60) as resp:
        d = pd.read_csv(io.BytesIO(resp.read()))
    d.columns = [c.strip() for c in d.columns]
    return d.set_index(pd.to_datetime(d["Date Time"]))["Water Level"].rename(station_id)


def download_dataset(force: bool = False) -> pd.DataFrame:
    if not config.DATA_FILE.exists() or force:
        df = pd.concat([_fetch_station(s) for s in config.STATIONS], axis=1)
        df.index.name = "time_utc"
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.DATA_FILE)
    return pd.read_csv(config.DATA_FILE, parse_dates=["time_utc"], dtype={s: float for s in config.STATIONS})


def load_dataset() -> pd.DataFrame:
    df = download_dataset()
    cols = list(config.STATIONS)
    df[cols] = df[cols].interpolate(limit_direction="both")   # guard against isolated gaps
    df.insert(0, "n", np.arange(len(df)))
    return df


def describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, (name, lat, lon) in config.STATIONS.items():
        x = df[sid]
        rows.append({"id": sid, "name": name, "lat": lat, "lon": lon, "mean": x.mean(), "std": x.std(),
                     "min": x.min(), "max": x.max(), "range": x.max() - x.min()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    d = load_dataset()
    print(d.head().to_string(index=False))
    print(describe_dataset(d).round(3).to_string(index=False))
