"""
data_loader.py
--------------
Downloads (once) and loads the bird-song recording.

The first call fetches the 441 kB WAV file from the ESC-50 GitHub repository and
caches it in ``data/``; every later call reads the local file, so the
experiment also runs offline.  The clip is converted to floating point in
[-1, 1] and resampled once to 48 kHz, which serves as the "analog" reference
from which every lower-rate version is sampled by an integer factor.
"""

import urllib.request

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

import config


def download_clip(force: bool = False):
    if config.CLIP_FILE.exists() and not force:
        return config.CLIP_FILE
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(config.CLIP_URL, timeout=30) as resp:
        config.CLIP_FILE.write_bytes(resp.read())
    return config.CLIP_FILE


def load_raw():
    """Samples exactly as published: (fs, int16 array)."""
    fs, x = wavfile.read(download_clip())
    if x.ndim > 1:                       # keep one channel if a stereo file is ever used
        x = x[:, 0]
    return fs, x


def load_reference():
    """Float signal on the 48 kHz reference grid, peak-normalised to 0.9."""
    fs, x = load_raw()
    x = x.astype(float) / 32768.0
    x = resample_poly(x, config.RESAMPLE_UP, config.RESAMPLE_DOWN)
    x -= x.mean()
    x *= 0.9 / np.max(np.abs(x))
    return config.FS_REF, x


def describe_clip():
    fs, x = load_raw()
    xf = x.astype(float) / 32768.0
    return {
        "file": config.CLIP_FILE.name, "fs": fs, "samples": len(x), "duration_s": len(x) / fs,
        "dtype": str(x.dtype), "bits": 16, "channels": 1,
        "size_kb": config.CLIP_FILE.stat().st_size / 1024,
        "rms": float(np.sqrt(np.mean(xf ** 2))), "peak": float(np.max(np.abs(xf))),
        "raw": x,
    }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    d = describe_clip()
    print({k: v for k, v in d.items() if k != "raw"})
    fs, x = load_reference()
    print(f"reference: {fs} Hz, {len(x)} samples")
