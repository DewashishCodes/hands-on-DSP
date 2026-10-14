"""
data_loader.py
--------------
Downloads (once) and loads PhysioNet BIDMC record bidmc01.

The first call fetches about 610 kB (binary signals plus two small CSV files)
and caches them in ``data/``; every later call reads the local copies.  The
WFDB format-16 binary is read directly: interleaved little-endian int16
samples, converted to physical units with the gain and baseline in the header.
"""

import urllib.request

import numpy as np
import pandas as pd

import config


def download(force: bool = False):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for key, name in config.FILES.items():
        path = config.DATA_DIR / name.split("/")[-1]
        if not path.exists() or force:
            req = urllib.request.Request(config.BASE + name, headers={"User-Agent": "curl/8.4.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                path.write_bytes(resp.read())


def _path(key):
    return config.DATA_DIR / config.FILES[key].split("/")[-1]


def read_header():
    """Parse the WFDB header: number of signals, rate, length, and per-signal gain / baseline / name."""
    lines = _path("hea").read_text().splitlines()
    rec = lines[0].split()
    nsig, fs, nsamp = int(rec[1]), float(rec[2]), int(rec[3])
    sigs = []
    for ln in lines[1: 1 + nsig]:
        p = ln.split()
        g = p[2]                                   # e.g. 138229.4494(-63710)/NU
        gain = float(g.split("(")[0])
        base = float(g.split("(")[1].split(")")[0]) if "(" in g else 0.0
        unit = g.split("/")[1] if "/" in g else ""
        sigs.append({"name": " ".join(p[8:]).strip(" ,"), "gain": gain, "baseline": base, "unit": unit})
    return {"nsig": nsig, "fs": fs, "nsamp": nsamp, "signals": sigs,
            "comment": next((ln for ln in lines if ln.startswith("#")), "")}


def load():
    download()
    hdr = read_header()
    raw = np.fromfile(_path("dat"), "<i2").reshape(-1, hdr["nsig"])
    s = hdr["signals"][config.PPG_INDEX]
    ppg = (raw[:, config.PPG_INDEX].astype(float) - s["baseline"]) / s["gain"]
    num = pd.read_csv(_path("numerics"))
    num.columns = [c.strip() for c in num.columns]
    br = pd.read_csv(_path("breaths"))
    breaths = br.iloc[:, 0].dropna().to_numpy(int)
    return {"hdr": hdr, "raw": raw, "ppg": ppg, "fs": hdr["fs"], "numerics": num, "breaths": breaths,
            "size_kb": sum(_path(k).stat().st_size for k in config.FILES) / 1024}


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    D = load()
    print(D["hdr"]["signals"], len(D["ppg"]), D["numerics"].mean().round(1).to_dict(), len(D["breaths"]))
