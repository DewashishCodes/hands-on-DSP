"""
data_loader.py
--------------
Downloads (once) and loads two 1-minute EEG baselines (eyes open, eyes closed)
of subject S001 from the PhysioNet EEG Motor Movement/Imagery Database.

The files (1.25 MB each) are fetched from PhysioNet's open-data mirror on AWS,
which is much faster than the main site, and cached in ``data/``.  The EDF
format is parsed directly: a fixed ASCII header, one block per signal, then
data records holding int16 samples for every signal in turn.
"""

import urllib.request

import numpy as np

import config

MIRROR = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/S001/"


def download(force: bool = False):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in config.RUNS.values():
        path = config.DATA_DIR / name
        if not path.exists() or force:
            for base in (MIRROR, config.BASE):                # mirror first, main site as fallback
                try:
                    with urllib.request.urlopen(base + name, timeout=120) as resp:
                        path.write_bytes(resp.read())
                    break
                except OSError:
                    continue


def read_edf(path):
    """Minimal EDF reader: returns {label: physical signal}, sampling rates and header fields."""
    raw = path.read_bytes()
    h = raw[:256].decode("ascii", "replace")
    nbytes, nrec, dur, ns = int(h[184:192]), int(h[236:244]), float(h[244:252]), int(h[252:256])
    hs = raw[256:nbytes].decode("ascii", "replace")

    def field(offset, width):
        return [hs[offset * ns + i * width: offset * ns + (i + 1) * width].strip() for i in range(ns)]

    # field widths in the signal header: label 16, transducer 80, dimension 8, physmin 8, physmax 8,
    # digmin 8, digmax 8, prefilter 80, samples per record 8, reserved 32
    o = 0
    labels = field(o, 16); o += 16
    o += 80
    units = field(o, 8); o += 8
    pmin = np.array(field(o, 8), float); o += 8
    pmax = np.array(field(o, 8), float); o += 8
    dmin = np.array(field(o, 8), float); o += 8
    dmax = np.array(field(o, 8), float); o += 8
    prefilter = field(o, 80); o += 80
    nsamp = np.array(field(o, 8), int)
    data = np.frombuffer(raw[nbytes:], "<i2")
    rec_len = int(nsamp.sum())
    data = data[: nrec * rec_len].reshape(nrec, rec_len)
    sig, start = {}, 0
    for i, lab in enumerate(labels):
        d = data[:, start: start + nsamp[i]].reshape(-1).astype(float)
        gain = (pmax[i] - pmin[i]) / (dmax[i] - dmin[i])
        sig[lab.rstrip(".")] = (d - dmin[i]) * gain + pmin[i]
        start += nsamp[i]
    return {"signals": sig, "fs": nsamp[0] / dur, "units": units[0], "n_signals": ns, "records": nrec,
            "start": h[168:184].strip(), "prefilter": prefilter[0], "labels": [l.rstrip(".") for l in labels],
            "digital": (dmin[0], dmax[0]), "physical": (pmin[0], pmax[0])}


def load():
    download()
    out = {}
    for key, name in config.RUNS.items():
        e = read_edf(config.DATA_DIR / name)
        out[key] = {"x": e["signals"][config.CHANNEL], "edf": e}
    out["fs"] = out["open"]["edf"]["fs"]
    out["size_kb"] = sum((config.DATA_DIR / n).stat().st_size for n in config.RUNS.values()) / 1024
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    D = load()
    e = D["open"]["edf"]
    print(D["fs"], e["n_signals"], e["records"], e["units"], e["labels"][:10], e["prefilter"], e["digital"], e["physical"])
    for k in ("open", "closed"):
        x = D[k]["x"]
        print(k, len(x), x.mean().round(2), x.std().round(2))
