"""
data_loader.py
--------------
Downloads (once) and loads the SILSO monthly sunspot-number series.

The first call fetches about 125 kB of semicolon-separated text from the
Royal Observatory of Belgium and caches it in ``data/``; every later call reads
the local file, so the experiment also runs offline.
"""

import urllib.request

import numpy as np
import pandas as pd

import config


def download_dataset(force: bool = False):
    if not config.DATA_FILE.exists() or force:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(config.DATA_URL, headers={"User-Agent": "dsp-lab/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            config.DATA_FILE.write_bytes(resp.read())
    return config.DATA_FILE


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(download_dataset(), sep=";", header=None, names=config.COLUMNS)
    df["sn"] = df["sn"].where(df["sn"] >= 0)               # -1 marks a missing month
    df["sn"] = df["sn"].interpolate(limit_direction="both")
    df.insert(0, "n", np.arange(len(df)))
    return df


def describe_dataset(df: pd.DataFrame) -> dict:
    i_max = int(df["sn"].idxmax())
    return {
        "months": len(df), "start": f"{df['year'].iloc[0]}-{df['month'].iloc[0]:02d}",
        "end": f"{df['year'].iloc[-1]}-{df['month'].iloc[-1]:02d}",
        "mean": float(df["sn"].mean()), "std": float(df["sn"].std()),
        "min": float(df["sn"].min()), "max": float(df["sn"].max()),
        "max_date": f"{df['year'].iloc[i_max]}-{df['month'].iloc[i_max]:02d}",
        "provisional": int((df["definitive"] == 0).sum()),
        "with_obs_count": int((df["n_obs"] > 0).sum()),
        "size_kb": config.DATA_FILE.stat().st_size / 1024,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    d = load_dataset()
    print(d.head().to_string(index=False))
    print(describe_dataset(d))
