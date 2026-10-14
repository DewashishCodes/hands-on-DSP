"""
data_loader.py
--------------
Downloads (once) and loads the Mauna Loa monthly CO2 record.

The first call fetches about 40 kB of CSV from the NOAA Global Monitoring
Laboratory and caches it in ``data/``; every later call reads the local file.
For the DFT the record is trimmed to start in January and to contain a whole
number of years, so that the annual cycle falls exactly on a DFT bin.
"""

import urllib.request

import numpy as np
import pandas as pd

import config


def download_dataset(force: bool = False):
    if not config.DATA_FILE.exists() or force:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(config.DATA_URL, headers={"User-Agent": "curl/8.4.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            config.DATA_FILE.write_bytes(resp.read())
    return config.DATA_FILE


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(download_dataset(), comment="#")
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"average": "co2", "deseasonalized": "co2_deseason", "decimal_date": "t"})
    df["co2"] = df["co2"].where(df["co2"] > 0).interpolate(limit_direction="both")
    df.insert(0, "n", np.arange(len(df)))
    return df


def whole_years(df: pd.DataFrame) -> pd.DataFrame:
    """Sub-record from the first January with a whole number of years."""
    i0 = int(np.argmax(df["month"].to_numpy() == 1))
    N = (len(df) - i0) // 12 * 12
    out = df.iloc[i0: i0 + N].reset_index(drop=True)
    out["m"] = np.arange(len(out))
    return out


def describe_dataset(df: pd.DataFrame) -> dict:
    w = whole_years(df)
    return {
        "months": len(df), "start": f"{df['year'].iloc[0]}-{df['month'].iloc[0]:02d}",
        "end": f"{df['year'].iloc[-1]}-{df['month'].iloc[-1]:02d}",
        "first": float(df["co2"].iloc[0]), "last": float(df["co2"].iloc[-1]),
        "min": float(df["co2"].min()), "max": float(df["co2"].max()),
        "no_daycount": int((df["ndays"] < 0).sum()),
        "N": len(w), "years": len(w) // 12,
        "w_start": f"{w['year'].iloc[0]}-01", "w_end": f"{w['year'].iloc[-1]}-12",
        "size_kb": config.DATA_FILE.stat().st_size / 1024,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    d = load_dataset()
    print(d.head().to_string(index=False))
    print(describe_dataset(d))
