"""
analysis.py
-----------
IIR filter design study and its application to occipital EEG: mains notch,
drift removal and alpha-band extraction, eyes open versus eyes closed.
"""

import numpy as np
from scipy import signal, stats

import config
import data_loader
import iir

FS = config.FS


def band_power(x, lo, hi, fs=FS):
    f, P = signal.welch(x - np.mean(x), fs, nperseg=int(config.WELCH_SECONDS * fs))
    m = (f >= lo) & (f <= hi)
    return float(np.trapz(P[m], f[m])), f, P


def designs():
    out = {}
    for fam in config.FAMILIES:
        d = iir.design_bandpass(fam, config.ALPHA_BAND, FS, config.ORDER, config.CHEBY_RIPPLE_DB, config.ELLIP_STOP_DB)
        f, H = iir.freq_response_sos(d["sos"], FS)
        fg, tau = iir.group_delay_sos(d["sos"], FS)
        A = 20 * np.log10(np.abs(H) + 1e-15)
        lo, hi = iir.band_edges(f, H)
        pb = (f >= config.ALPHA_BAND[0] + 0.3) & (f <= config.ALPHA_BAND[1] - 0.3)
        inband = (fg >= config.ALPHA_BAND[0]) & (fg <= config.ALPHA_BAND[1])
        ref = {"butterworth": signal.butter(config.ORDER, config.ALPHA_BAND, "bandpass", fs=FS, output="sos"),
               "chebyshev1": signal.cheby1(config.ORDER, config.CHEBY_RIPPLE_DB, config.ALPHA_BAND, "bandpass", fs=FS, output="sos"),
               "elliptic": signal.ellip(config.ORDER, config.CHEBY_RIPPLE_DB, config.ELLIP_STOP_DB, config.ALPHA_BAND,
                                        "bandpass", fs=FS, output="sos")}[fam]
        _, Href = iir.freq_response_sos(ref, FS)
        out[fam] = {**d, "f": f, "H": H, "fg": fg, "tau": tau, "edges": (lo, hi),
                    "ripple_db": float(A[pb].max() - A[pb].min()),
                    "att5": float(A[np.argmin(np.abs(f - 5))]), "att16": float(A[np.argmin(np.abs(f - 16))]),
                    "att60": float(A[np.argmin(np.abs(f - 60))]),
                    "max_pole": iir.max_pole_radius(d["zpk"][1]), "n_coef": int(len(d["ba"][0]) + len(d["ba"][1]) - 1),
                    "tau_min": float(tau[inband].min() / FS), "tau_max": float(tau[inband].max() / FS),
                    "vs_scipy": float(np.max(np.abs(np.abs(H) - np.abs(Href))))}
    return out


def fir_equivalent(target_db, f_edge=5.0):
    """Shortest Hamming-window FIR band-pass (Experiment 11 method) that is at least as selective at f_edge."""
    from scipy.signal import firwin
    for N in range(21, 2001, 10):
        h = firwin(N, config.ALPHA_BAND, pass_zero=False, fs=FS, window="hamming")
        f, H = signal.freqz(h, 1, 8192, fs=FS)
        if 20 * np.log10(abs(H[np.argmin(np.abs(f - f_edge))]) + 1e-15) <= target_db:
            return {"N": N, "delay_s": (N - 1) / 2 / FS}
    return {"N": None, "delay_s": None}


def prewarp_study():
    """Edges obtained with and without pre-warping, for a band low (alpha) and high (near Nyquist)."""
    rows = []
    for band in (config.ALPHA_BAND, (50.0, 70.0)):
        for warp in (True, False):
            z, p, k = signal.buttap(config.ORDER)
            W = iir.prewarp(band, FS) if warp else 2 * np.pi * np.asarray(band)
            z, p, k = signal.lp2bp_zpk(z, p, k, wo=np.sqrt(W[0] * W[1]), bw=W[1] - W[0])
            zd, pd, kd = iir.bilinear_zpk(z, p, k, FS)
            f, H = signal.sosfreqz(signal.zpk2sos(zd, pd, kd), 8192, fs=FS)
            rows.append({"band": band, "prewarp": warp, "edges": iir.band_edges(f, H)})
    return rows


def numerics_study(D):
    """Order-8 band-pass as one direct-form difference equation versus cascaded biquads."""
    x = D["closed"]["x"] - D["closed"]["x"].mean()
    out = {}
    for fam in ("butterworth", "elliptic"):
        d = iir.design_bandpass(fam, config.ALPHA_BAND, FS, config.ORDER, config.CHEBY_RIPPLE_DB, config.ELLIP_STOP_DB)
        b, a = d["ba"]
        y_df = iir.iir_filter(b, a, x)
        y_sos = iir.sos_filter(d["sos"], x)
        out[fam] = {"rel_err": float(np.linalg.norm(y_df - y_sos) / np.linalg.norm(y_sos)),
                    "lfilter_match": float(np.max(np.abs(y_df - signal.lfilter(b, a, x))))}
    # a high-order narrow design where the direct form fails
    d = iir.design_bandpass("butterworth", (9.0, 11.0), FS, 8)
    b, a = d["ba"]
    y_df = signal.lfilter(b, a, x)
    y_sos = iir.sos_filter(d["sos"], x)
    pa = np.roots(a)
    out["narrow16"] = {"rel_err": float(np.linalg.norm(y_df - y_sos) / np.linalg.norm(y_sos)),
                       "max_pole_tf": float(np.max(np.abs(pa))), "max_pole_true": iir.max_pole_radius(d["zpk"][1])}
    return out


def notch_study(D):
    b, a, r = iir.notch(config.MAINS_HZ, config.NOTCH_Q, FS)
    f, H = iir.freq_response(b, a, FS)
    x = D["open"]["x"] - D["open"]["x"].mean()
    y = iir.iir_filter(b, a, x)
    p_before, ff, Pb = band_power(x, 59, 61)
    p_after, _, Pa = band_power(y, 59, 61)
    a_before = band_power(x, 8, 13)[0]
    a_after = band_power(y, 8, 13)[0]
    lo, hi = None, None
    A = 20 * np.log10(np.abs(H) + 1e-15)
    below = f[A <= -3]
    return {"b": b, "a": a, "r": r, "f": f, "H": H, "f_psd": ff, "P_before": Pb, "P_after": Pa,
            "mains_reduction_db": float(10 * np.log10(p_before / p_after)),
            "p60_before": float(Pb[np.argmin(np.abs(ff - 60))]), "p60_after": float(Pa[np.argmin(np.abs(ff - 60))]),
            "floor": float(np.median(Pb[(ff >= 52) & (ff <= 58)])),
            "alpha_change_db": float(10 * np.log10(a_after / a_before)),
            "bw3": float(below.max() - below.min()), "settle_s": float(np.log(0.01) / np.log(r) / FS),
            "depth_db": float(A[np.argmin(np.abs(f - config.MAINS_HZ))]), "gain10_db": float(A[np.argmin(np.abs(f - 10))])}


def phase_study(D, fam="elliptic"):
    """Causal IIR output versus zero-phase (forward-backward) output: delay and waveform change."""
    sos = iir.design_bandpass(fam, config.ALPHA_BAND, FS, config.ORDER, config.CHEBY_RIPPLE_DB, config.ELLIP_STOP_DB)["sos"]
    x = D["closed"]["x"] - D["closed"]["x"].mean()
    yc = iir.sos_filter(sos, x)
    yz = iir.filtfilt_sos(sos, x)
    lags = np.arange(-40, 41)
    cc = [np.dot(yc[100 + l: len(yc) - 100 + l], yz[100: len(yz) - 100]) for l in lags]
    lag = int(lags[int(np.argmax(cc))])
    return {"yc": yc, "yz": yz, "x": x, "lag_samples": lag, "lag_ms": 1000 * lag / FS}


def alpha_study(D):
    """Alpha-band power per 2-s epoch after drift removal, notch and Butterworth band-pass (zero phase)."""
    bh, ah = iir.dc_blocker(config.HIGHPASS_HZ, FS)
    bn, an, _ = iir.notch(config.MAINS_HZ, config.NOTCH_Q, FS)
    sos = iir.design_bandpass("butterworth", config.ALPHA_BAND, FS, config.ORDER)["sos"]
    L = int(config.EPOCH_SECONDS * FS)
    out = {}
    for k in ("open", "closed"):
        x = D[k]["x"] - D[k]["x"].mean()
        x = iir.iir_filter(bn, an, iir.iir_filter(bh, ah, x))
        y = iir.filtfilt_sos(sos, x)
        env = np.abs(signal.hilbert(y))
        n_ep = len(y) // L
        ep = np.array([np.mean(y[i * L:(i + 1) * L] ** 2) for i in range(n_ep)])
        _, f, P = band_power(x, 1, 40)
        out[k] = {"y": y, "env": env, "x": x, "epochs": ep, "mean_power": float(ep.mean()), "f": f, "P": P,
                  "peak_hz": float(f[(f > 4) & (f < 20)][np.argmax(P[(f > 4) & (f < 20)])]),
                  "rms_uv": float(np.sqrt(ep.mean()))}
    t = stats.ttest_ind(out["closed"]["epochs"], out["open"]["epochs"], equal_var=False)
    out["ratio"] = out["closed"]["mean_power"] / out["open"]["mean_power"]
    out["t"], out["p"] = float(t.statistic), float(t.pvalue)
    out["frac_closed_above"] = float(np.mean(out["closed"]["epochs"] > out["open"]["epochs"].max()))
    return out


def run_all():
    D = data_loader.load()
    Ds = designs()
    return {"D": D, "designs": Ds, "fir_eq": fir_equivalent(Ds["butterworth"]["att5"]),
            "prewarp": prewarp_study(), "numerics": numerics_study(D), "notch": notch_study(D),
            "phase": phase_study(D), "alpha": alpha_study(D)}
