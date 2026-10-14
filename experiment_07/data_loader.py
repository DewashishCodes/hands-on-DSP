"""
data_loader.py
--------------
Downloads (once) and loads the daily USD/INR exchange rate from FRED.

The first call fetches about 35 kB of CSV from the Federal Reserve Bank of
St. Louis and caches it in ``data/``; every later call reads the local file.
Days without a quotation (US and Indian holidays) are dropped, so the sample
index n counts trading days.
"""

import urllib.request

import numpy as np
import pandas as pd

import config


def download_dataset(force: bool = False):
    if not config.DATA_FILE.exists() or force:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        # FRED's CSV endpoint drops requests from unknown clients; a curl user agent is accepted
        req = urllib.request.Request(config.DATA_URL, headers={"User-Agent": "curl/8.4.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            config.DATA_FILE.write_bytes(resp.read())
    return config.DATA_FILE


def load_dataset() -> pd.DataFrame:
    raw = pd.read_csv(download_dataset(), na_values=["."])
    raw.columns = ["date", "inr"]
    raw["date"] = pd.to_datetime(raw["date"])
    df = raw.dropna().reset_index(drop=True)
    df.insert(0, "n", np.arange(len(df)))
    df["change_pct"] = 100 * df["inr"].pct_change()
    df.attrs["calendar_rows"] = len(raw)
    return df


def describe_dataset(df: pd.DataFrame) -> dict:
    ch = df["change_pct"].dropna()
    return {
        "rows": df.attrs.get("calendar_rows", len(df)), "trading_days": len(df),
        "missing": df.attrs.get("calendar_rows", len(df)) - len(df),
        "start": df["date"].iloc[0], "end": df["date"].iloc[-1],
        "first": float(df["inr"].iloc[0]), "last": float(df["inr"].iloc[-1]),
        "min": float(df["inr"].min()), "max": float(df["inr"].max()),
        "min_date": df.loc[df["inr"].idxmin(), "date"], "max_date": df.loc[df["inr"].idxmax(), "date"],
        "change_std": float(ch.std()), "change_maxabs": float(ch.abs().max()),
        "size_kb": config.DATA_FILE.stat().st_size / 1024,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    d = load_dataset()
    print(d.head().to_string(index=False))
    print(describe_dataset(d))
