"""
data_loader.py
--------------
Downloads (once) and loads the three audio files.

The first call fetches about 630 kB of WAV files from GitHub and caches them
in ``data/``; every later call reads the local copies.  Samples are converted
from 16-bit integers to floating point in [-1, 1].
"""

import urllib.request
import warnings

import numpy as np
from scipy.io import wavfile

import config


def download(force: bool = False):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in config.FILES.values():
        path = config.DATA_DIR / name
        if not path.exists() or force:
            with urllib.request.urlopen(config.BASE_URL + name, timeout=60) as resp:
                path.write_bytes(resp.read())


def read(key):
    download()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)   # metadata chunks written by Audacity
        fs, x = wavfile.read(config.DATA_DIR / config.FILES[key])
    if x.ndim > 1:
        x = x[:, 0]
    return fs, x


def load():
    """Return fs and the float signals x (dry), h (impulse response), y_ref (published wet)."""
    out = {}
    fs = None
    for key in config.FILES:
        f, raw = read(key)
        fs = fs or f
        if f != fs:
            raise ValueError("all files must share one sampling rate")
        out[key] = raw.astype(float) / 32768.0
        out[key + "_int"] = raw
    out["fs"] = fs
    return out


def describe(D):
    fs = D["fs"]
    rows = []
    for key, name in config.FILES.items():
        x = D[key]
        rows.append({"key": key, "file": name, "samples": len(x), "seconds": len(x) / fs,
                     "peak": float(np.max(np.abs(x))), "rms": float(np.sqrt(np.mean(x ** 2))),
                     "kb": (config.DATA_DIR / name).stat().st_size / 1024})
    return rows


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    D = load()
    print(D["fs"], describe(D))
