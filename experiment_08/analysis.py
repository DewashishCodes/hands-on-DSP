"""
analysis.py
-----------
DFT / IDFT study of the Mauna Loa CO2 record; extracts the numbers quoted in
the report.
"""

import time

import numpy as np

import config
import data_loader
import dft as D

FS = config.SAMPLES_PER_YEAR
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def method_check(r):
    X1 = D.dft(r)
    X2 = D.fft(r)
    sub = r[:96]
    X0 = D.dft_direct(sub)
    return {
        "matrix_vs_fft": float(np.max(np.abs(X1 - X2)) / np.max(np.abs(X2))),
        "direct_vs_fft": float(np.max(np.abs(X0 - D.fft(sub))) / np.max(np.abs(D.fft(sub)))),
        "idft_err": float(np.max(np.abs(D.idft(X1).real - r))),
        "idft_imag": float(np.max(np.abs(D.idft(X1).imag))),
        "parseval": D.parseval_error(r, X2), "symmetry": D.symmetry_error(X2),
    }


def spectrum_study(r):
    N = len(r)
    X = D.fft(r)
    A = D.amplitude_spectrum(X)
    f = D.frequencies(N, FS)[: N // 2 + 1]
    k1 = N // FS
    lines = {h: D.bin_amplitude_phase(X, h * k1) for h in (1, 2, 3, 4)}
    noise = np.delete(A[1:], [h * k1 - 1 for h in (1, 2, 3, 4)])
    top = np.argsort(A[1:])[::-1][:3] + 1
    return {"X": X, "A": A, "f": f, "k1": k1, "lines": lines, "noise_median": float(np.median(noise)),
            "top_bins": [int(t) for t in top], "top_freqs": [float(f[t]) for t in top],
            "resolution": FS / N}


def leakage_study(r):
    N = len(r)
    k1 = N // FS
    A_exact = D.amplitude_spectrum(D.fft(r))
    rc = r[: N - config.LEAK_CUT]
    Nc = len(rc)
    A_cut = D.amplitude_spectrum(D.fft(rc))
    w = D.hann(Nc)
    A_hann = D.amplitude_spectrum(D.fft(rc * w)) / np.mean(w)          # coherent-gain correction
    P = config.ZERO_PAD
    A_pad = D.amplitude_spectrum(D.fft(rc * w, P * Nc)) * P / np.mean(w)
    f_cut = D.frequencies(Nc, FS)[: Nc // 2 + 1]
    f_pad = D.frequencies(P * Nc, FS)[: P * Nc // 2 + 1]
    band = (f_cut > 1.5) & (f_cut < 1.9)                            # between the two main lines
    return {
        "f_exact": D.frequencies(N, FS)[: N // 2 + 1], "A_exact": A_exact,
        "f_cut": f_cut, "A_cut": A_cut, "A_hann": A_hann, "f_pad": f_pad, "A_pad": A_pad,
        "peak_exact": float(A_exact[k1]), "peak_cut": float(A_cut.max()), "peak_hann": float(A_hann.max()),
        "peak_pad": float(A_pad.max()), "f_peak_pad": float(f_pad[np.argmax(A_pad)]),
        "leak_rect": float(np.median(A_cut[band])), "leak_hann": float(np.median(A_hann[band])),
        "leak_exact": float(np.median(A_exact[(D.frequencies(N, FS)[: N // 2 + 1] > 1.5) &
                                              (D.frequencies(N, FS)[: N // 2 + 1] < 1.9)])),
        "Nc": Nc, "bin_offset": (Nc / FS) % 1,
    }


def seasonal_study(w, r, trend):
    N = len(r)
    k1 = N // FS
    X = D.fft(r)
    seas = D.idft(D.keep_bins(X, [h * k1 for h in config.HARMONICS])).real
    clim = seas[:12]
    deseason = w["co2"].to_numpy() - seas
    noaa = w["co2_deseason"].to_numpy()
    resid = r - seas
    return {
        "seasonal": seas, "clim": clim, "deseason": deseason, "noaa": noaa,
        "rmse_noaa": float(np.sqrt(np.mean((deseason - noaa) ** 2))),
        "max_month": MONTHS[int(np.argmax(clim))], "min_month": MONTHS[int(np.argmin(clim))],
        "p2p": float(clim.max() - clim.min()),
        "var_explained": float(1 - np.var(resid) / np.var(r)),
        "resid_std": float(np.std(resid)),
    }


def decade_study(w, r):
    out = []
    L = config.DECADE
    for a in range(0, len(r) - L + 1, L):
        seg = r[a: a + L]
        X = D.fft(seg)
        A1, _ = D.bin_amplitude_phase(X, L // FS)
        A2, _ = D.bin_amplitude_phase(X, 2 * L // FS)
        seas = D.idft(D.keep_bins(X, [L // FS, 2 * L // FS])).real[:12]
        out.append({"start": int(w["year"].iloc[a]), "A1": A1, "A2": A2, "p2p": float(seas.max() - seas.min())})
    return out


def growth_study(w, trend_poly):
    years = w["year"].to_numpy()
    dtrend = np.polyval(np.polyder(trend_poly), w["m"].to_numpy()) * FS     # ppm per year
    d = {}
    for y0 in (1960, 1980, 2000, 2020):
        m = (years >= y0) & (years < y0 + 5)
        if m.any():
            d[y0] = float(dtrend[m].mean())
    return {"rate": dtrend, "by_epoch": d}


def timing_study():
    rows = []
    rng = np.random.default_rng(8)
    for N in config.TIMING_N:
        x = rng.normal(size=N)
        t0 = time.perf_counter()
        reps = max(1, 20000 // N)
        for _ in range(reps):
            D.dft(x)
        t_mat = (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        for _ in range(200):
            D.fft(x)
        t_fft = (time.perf_counter() - t0) / 200
        rows.append({"N": N, "matrix_s": t_mat, "fft_s": t_fft, "ops_dft": N * N,
                     "ops_fft": N * np.log2(N)})
    return rows


def run_all():
    df = data_loader.load_dataset()
    w = data_loader.whole_years(df)
    x = w["co2"].to_numpy()
    r, trend, p = D.detrend(x, config.TREND_ORDER)
    return {
        "df": df, "w": w, "x": x, "r": r, "trend": trend, "info": data_loader.describe_dataset(df),
        "methods": method_check(r), "spec": spectrum_study(r), "leak": leakage_study(r),
        "season": seasonal_study(w, r, trend), "decades": decade_study(w, r),
        "growth": growth_study(w, p), "timing": timing_study(),
    }
