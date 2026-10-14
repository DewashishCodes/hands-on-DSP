"""
analysis.py
-----------
Circular-convolution-via-DFT study of the New Delhi daily climatology; extracts
the numbers quoted in the report.
"""

import time

import numpy as np

import circdft as cd
import config
import data_loader

N = config.N_DAYS


def kernels():
    return {
        f"Boxcar {config.MA_DAYS} d": cd.boxcar(config.MA_DAYS, N),
        f"Gaussian σ = {config.GAUSS_SIGMA:g} d": cd.gaussian(config.GAUSS_SIGMA, N),
        f"{config.HARMONICS} harmonics": cd.harmonic_kernel(config.HARMONICS, N),
    }


def day_label(k):
    import pandas as pd
    return (pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(k))).strftime("%d %b")


def smoothing_study(x, clim):
    """Smooth with each kernel; compare with the expected sampling noise of the raw climatology."""
    noise_expected = float(np.sqrt(np.mean((clim["sd_tmax"] / np.sqrt(clim["n_tmax"])) ** 2)))
    out = {}
    for name, h in kernels().items():
        y = cd.circ_dft(x, h)
        dy = np.roll(y, -1) - y
        out[name] = {
            "h": h, "y": y, "resid_std": float(np.std(x - y)), "roughness": cd.roughness(y),
            "hot": int(np.argmax(y)), "hot_val": float(y.max()), "cold": int(np.argmin(y)), "cold_val": float(y.min()),
            "steepest_fall": int(np.argmin(dy[120:240]) + 120), "steepest_rate": float(dy[120:240].min()),
            "H": cd.frequency_response(h),
        }
    return out, noise_expected


def method_check(x):
    h = cd.boxcar(config.MA_DAYS, N)
    a, b, m = cd.circ_direct(x, h), cd.circ_dft(x, h), cd.circ_dft_matrix(x, h)
    hc = np.ones(config.MA_DAYS) / config.MA_DAYS
    cp = cd.cyclic_prefix_conv(x, hc)
    ref = cd.circ_dft(x, np.r_[hc, np.zeros(N - config.MA_DAYS)])
    return {"direct_vs_fft": float(np.max(np.abs(a - b))), "direct_vs_matrix": float(np.max(np.abs(a - m))),
            "cp_vs_circ": float(np.max(np.abs(cp - ref))), "sum_err": float(abs(b.sum() - x.sum())),
            "cp": cp, "cp_ref": ref, "causal_delay": (config.MA_DAYS - 1) // 2}


def edge_study(x):
    rows = []
    out = {}
    for M in config.KERNEL_LENGTHS:
        h = cd.boxcar(M, N)
        yc = cd.circ_dft(x, h)
        yl = cd.linear_same(x, h, M // 2)
        e = yc - yl
        rows.append({"M": M, "max_err": float(np.max(np.abs(e))), "days_affected": int(np.sum(np.abs(e) > 0.05))})
        out[M] = {"yc": yc, "yl": yl}
    return rows, out


def timing_study(clim):
    rng = np.random.default_rng(10)
    X = clim["tmax"].to_numpy()[None, :] + rng.normal(0, 1, (config.TIMING_SIGNALS, N))
    h = cd.boxcar(config.MA_DAYS, N)
    idx = (np.arange(N)[:, None] - np.arange(N)[None, :]) % N
    Hm = h[idx]
    t0 = time.perf_counter()
    Yd = X @ Hm.T                                      # direct: one N x N matrix product per signal
    t_direct = time.perf_counter() - t0
    t0 = time.perf_counter()
    Yf = np.fft.irfft(np.fft.rfft(X, axis=1) * np.fft.rfft(h), N, axis=1)
    t_fft = time.perf_counter() - t0
    t0 = time.perf_counter()
    for row in X[:50]:
        cd.circ_direct(row, h)
    t_loop = (time.perf_counter() - t0) / 50 * config.TIMING_SIGNALS
    return {"t_direct": t_direct, "t_fft": t_fft, "t_loop": t_loop, "err": float(np.max(np.abs(Yd - Yf))),
            "mult_direct": N * N, "mult_fft": int(3 * N * np.log2(N) + N)}


def scaling_study(sizes=(64, 128, 365, 1024, 2048, 4096)):
    """Time one circular convolution, direct (precomputed circulant matrix) versus FFT, for growing N."""
    rng = np.random.default_rng(11)
    rows = []
    for n in sizes:
        x, h = rng.normal(size=n), rng.normal(size=n)
        Hm = h[(np.arange(n)[:, None] - np.arange(n)[None, :]) % n]
        reps = max(3, 20000 // n)
        t0 = time.perf_counter()
        for _ in range(reps):
            Hm @ x
        t_mat = (time.perf_counter() - t0) / reps
        Hf = np.fft.rfft(h)
        t0 = time.perf_counter()
        for _ in range(reps):
            np.fft.irfft(np.fft.rfft(x) * Hf, n)
        t_fft = (time.perf_counter() - t0) / reps
        rows.append({"N": n, "t_matrix": t_mat, "t_fft": t_fft, "mult_direct": n * n,
                     "mult_fft": float(2 * n * np.log2(n) + n)})
    return rows


def run_all():
    df = data_loader.load_daily()
    clim = data_loader.climatology(df)
    x = cd.fill_circular(clim["tmax"].to_numpy())
    xn = cd.fill_circular(clim["tmin"].to_numpy())
    S, noise = smoothing_study(x, clim)
    rows, edge = edge_study(x)
    h = cd.boxcar(config.MA_DAYS, N)
    return {
        "df": df, "clim": clim, "x": x, "xn": xn, "info": data_loader.describe(df, clim),
        "smooth": S, "noise_expected": noise, "methods": method_check(x),
        "edge_rows": rows, "edge": edge, "timing": timing_study(clim), "scaling": scaling_study(),
        "tmin_s": cd.circ_dft(xn, h), "dtr_s": cd.circ_dft(x - xn, h),
    }
