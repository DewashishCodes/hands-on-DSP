"""
analysis.py
-----------
Fast-convolution study on the speech and room-impulse-response recordings;
extracts the numbers quoted in the report.
"""

import time

import numpy as np

import config
import data_loader
import fastconv as fc


def sisnr(ref, est):
    """Scale-invariant SNR in dB (the published file is normalised, so the gain is irrelevant)."""
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    k = np.dot(est, ref) / np.dot(est, est)
    return float(10 * np.log10(np.sum(ref ** 2) / np.sum((ref - k * est) ** 2)))


def _time(fn, reps=3):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def accuracy_and_speed(x, h, y_ref, fs):
    L, M = len(x), len(h)
    Nlin = L + M - 1
    Nfast = fc.next_fast_length(Nlin)
    t_exact, y_exact = _time(lambda: fc.conv_dft(x, h, Nlin))
    t_fast, y_fast = _time(lambda: fc.conv_dft(x, h, Nfast)[:Nlin])
    xs = x[: int(config.DIRECT_TIMING_SECONDS * fs)]
    t_dir_short, y_dir_short = _time(lambda: fc.conv_direct(xs, h), reps=1)
    t_direct = t_dir_short * L / len(xs)                      # direct cost is proportional to L
    k = float(np.dot(y_fast, y_ref) / np.dot(y_fast, y_fast))
    return {
        "L": L, "M": M, "Nlin": Nlin, "Nfast": Nfast, "y": y_fast,
        "t_exact": t_exact, "t_fast": t_fast, "t_direct": t_direct, "t_dir_short": t_dir_short,
        "speedup": t_direct / t_fast,
        "err_vs_direct": fc.rel_error(fc.conv_dft(xs, h), y_dir_short),
        "err_vs_published": fc.rel_error(k * y_fast, y_ref), "snr_vs_published": fc.snr_db(y_ref, k * y_fast),
        "scale": k, "mult_direct": fc.mults_direct(L, M), "mult_fft": fc.mults_fft(Nfast),
        "factor_Nlin": _factors(Nlin), "factor_Nfast": _factors(Nfast),
    }


def _factors(n):
    f, d = [], 2
    while d * d <= n:
        while n % d == 0:
            f.append(d)
            n //= d
        d += 1
    if n > 1:
        f.append(n)
    return f


def aliasing_study(x, h, y):
    L, M = len(x), len(h)
    Ns = [L, L + M // 4, L + M // 2, L + 3 * M // 4, L + M - 2, L + M - 1]
    rows = []
    for N in Ns:
        ya = fc.conv_dft(x, h, N)
        rows.append({"N": N, "err": fc.rel_error(ya, y[:N]), "short_by": L + M - 1 - N})
    y_alias = fc.conv_dft(x, h, L)
    return {"rows": rows, "y_alias": y_alias}


def ola_study(x, h, y, fs):
    rows = []
    for B in config.OLA_BLOCKS:
        t, (yo, N) = _time(lambda: fc.overlap_add(x, h, B))
        rows.append({"B": B, "N": N, "err": fc.rel_error(yo, y), "time": t,
                     "latency_ms": 1000 * B / fs, "mult_per_sample": fc.mults_ola(len(x), len(h), B) / len(x)})
    return rows


def room_study(h, fs):
    edc = fc.schroeder_db(h)
    t20, s20, c20, a20, b20 = fc.reverberation_time(edc, fs, -5, -25)
    t30, s30, c30, a30, b30 = fc.reverberation_time(edc, fs, -5, -35)
    edt = fc.reverberation_time(edc, fs, 0, -10)[0]
    drr, k = fc.direct_to_reverberant(h, fs)
    Hf = np.fft.rfft(h)
    f = np.fft.rfftfreq(len(h), 1 / fs)
    return {"edc": edc, "T20": t20, "T30": t30, "EDT": edt, "fit20": (s20, c20, a20, b20),
            "DRR": drr, "peak_idx": k, "peak_ms": 1000 * k / fs, "Hf": Hf, "f": f}


def deconv_study(x, h, y_ref, y):
    rng = np.random.default_rng(config.SEED)
    noise = rng.normal(0, np.std(y) * 10 ** (-config.NOISE_SNR_DB / 20), len(y))
    yn = y + noise
    rows = []
    for eps in config.DECONV_EPS:
        rows.append({"eps": eps, "clean": sisnr(x, fc.deconvolve(y_ref, h, len(x), eps)),
                     "noisy": sisnr(x, fc.deconvolve(yn, h, len(x), eps))})
    best_clean = max(rows, key=lambda r: r["clean"])
    best_noisy = max(rows, key=lambda r: r["noisy"])
    return {"rows": rows, "wet_sisnr": sisnr(x, y_ref[: len(x)]), "best_clean": best_clean,
            "best_noisy": best_noisy,
            "x_hat": fc.deconvolve(y_ref, h, len(x), best_clean["eps"])}


def run_all():
    D = data_loader.load()
    x, h, y_ref, fs = D["dry"], D["rir"], D["wet"], D["fs"]
    A = accuracy_and_speed(x, h, y_ref, fs)
    return {
        "D": D, "fs": fs, "x": x, "h": h, "y_ref": y_ref, "desc": data_loader.describe(D),
        "acc": A, "alias": aliasing_study(x, h, A["y"]), "ola": ola_study(x, h, A["y"], fs),
        "room": room_study(h, fs), "deconv": deconv_study(x, h, y_ref, A["y"]),
    }
