"""
analysis.py
-----------
Runs the sampling, anti-aliasing and reconstruction study on the bird-song
recording and extracts the numbers quoted in the report.
"""

import numpy as np

import config
import data_loader
import sampling as sp

TRIM = 2000            # reference samples ignored at each edge (filter transients)


def clip_summary(x, fs):
    f, S = sp.spectrum_db(x, fs)
    P = np.abs(np.fft.rfft(x)) ** 2
    return {
        "bandwidth": sp.bandwidth(x, fs),
        "peak_freq": float(np.fft.rfftfreq(len(x), 1 / fs)[np.argmax(P)]),
        "nyquist_rate": 2 * sp.bandwidth(x, fs),
        "above": {r: sp.energy_above(x, fs, r / 2) for r in config.RATES},
        "spectrum": (f, S),
    }


def prefilter_study(x, fs):
    """SNR of (prefilter -> sample -> ideal sinc reconstruction) against the ideal band-limited signal."""
    N = len(x)
    out = {}
    for rate in config.RATES:
        D = fs // rate
        ref = sp.bandlimit(x, fs, rate / 2)
        row = {}
        for kind in config.PREFILTERS:
            xs = sp.sample(x, fs, rate, kind)
            xr = sp.reconstruct(xs, D, N, "Sinc (band-limited)")
            row[kind] = {"snr": sp.snr_db(ref, xr, TRIM), "xs": xs}
        out[rate] = row
    return out


def reconstruction_study(x, fs):
    """Interpolation error alone: FIR prefilter, then each reconstruction method."""
    N = len(x)
    out = {}
    for rate in config.RATES[:2]:
        D = fs // rate
        ref = sp.bandlimit(x, fs, rate / 2)
        xs = sp.sample(x, fs, rate, "FIR (ideal-like)")
        out[rate] = {m: {"snr": sp.snr_db(ref, sp.reconstruct(xs, D, N, m), TRIM),
                         "y": sp.reconstruct(xs, D, N, m)} for m in config.RECON_METHODS}
        out[rate]["ref"] = ref
        out[rate]["xs"] = xs
    return out


def ma_study(x, fs):
    out = {}
    for M in config.MA_LENGTHS:
        rate = fs // M
        y = sp.moving_average(x, M)
        nyq = rate / 2
        out[M] = {
            "f3db": sp.ma_3db_frequency(M, fs),
            "first_null": fs / M,
            "gain_at_nyquist": float(sp.ma_frequency_response(M, nyq, fs)),
            "gain_1k": float(sp.ma_frequency_response(M, 1000, fs)),
            "hf_energy_removed": 1 - sp.energy_above(y, fs, 2000) * np.sum(y ** 2)
                                 / (sp.energy_above(x, fs, 2000) * np.sum(x ** 2)),
            "y": y,
            "rate": rate,
        }
    # first sidelobe of M = 6 (worst-case leakage above the null)
    f = np.linspace(fs / 6, 2 * fs / 6, 4001)
    out["sidelobe_db"] = float(20 * np.log10(sp.ma_frequency_response(6, f, fs).max()))
    return out


def storage(rate, minutes=60):
    """Bytes for one hour of 16-bit mono audio: why a field recorder wants a low rate."""
    return rate * config.BITS_PER_SAMPLE / 8 * 60 * minutes


def run_all():
    fs, x = data_loader.load_reference()
    clip = clip_summary(x, fs)
    pre = prefilter_study(x, fs)
    rec = reconstruction_study(x, fs)
    ma = ma_study(x, fs)
    return {
        "fs": fs, "x": x, "clip": clip, "prefilter": pre, "recon": rec, "ma": ma,
        "storage_mb_per_h": {r: storage(r) / 1e6 for r in [config.FS_REF] + config.RATES},
        "alias_example": {"f": 3100.0, "fs": 4000, "f_alias": float(sp.alias_frequency(3100.0, 4000))},
    }
