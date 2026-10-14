"""
analysis.py
-----------
FIR window-design study and its application to the BIDMC photoplethysmogram:
heart rate from a band-pass filter, breathing rate from a low-pass filter.
"""

import numpy as np
from scipy.signal import find_peaks

import config
import data_loader
import fir

FS = config.FS


def window_study(fc=config.PULSE_BAND[1], N=config.N_PULSE):
    """Same specification, four windows: ripple, attenuation and transition, against the textbook table."""
    rows = {}
    for w in config.WINDOWS:
        t = fir.WINDOW_TABLE[w]
        df = t["mainlobe"] / 2 * FS / (N - 1)           # approximate transition width (Hz)
        b = fir.design_lowpass(fc, FS, N, w)
        m = fir.measure_lowpass(b, fc, FS, df / 2)
        f, H = fir.freq_response(b, FS)
        rows[w] = {**m, "b": b, "f": f, "H": H, "theory_db": t["stopband_db"], "df_theory": df}
    return rows


def kaiser_study():
    A, tw = config.KAISER_SPEC["atten_db"], config.KAISER_SPEC["transition_hz"]
    N, beta = fir.kaiser_parameters(A, tw, FS)
    b = fir.design_lowpass(config.RESP_CUTOFF + tw / 2, FS, N, "kaiser", beta)
    m = fir.measure_lowpass(b, config.RESP_CUTOFF + tw / 2, FS, tw / 2)
    return {"N": N, "beta": beta, "b": b, **m, "A": A, "tw": tw}


def filters():
    bp = fir.design_bandpass(*config.PULSE_BAND, FS, config.N_PULSE, config.DEFAULT_WINDOW)
    lp = fir.design_lowpass(config.RESP_CUTOFF, FS, config.N_RESP, config.DEFAULT_WINDOW)
    out = {}
    for name, b in (("pulse", bp), ("resp", lp)):
        f, H = fir.freq_response(b, FS)
        fg, tau = fir.group_delay(b, FS)
        out[name] = {"b": b, "f": f, "H": H, "fg": fg, "tau": tau, "N": len(b),
                     "delay_s": (len(b) - 1) / 2 / FS, "linear_phase": fir.is_linear_phase(b)}
    # gains at the frequencies that matter
    def gain(name, f0):
        f, H = out[name]["f"], out[name]["H"]
        return float(20 * np.log10(np.abs(H[np.argmin(np.abs(f - f0))]) + 1e-15))
    out["gains"] = {"pulse@1.5": gain("pulse", 1.5), "pulse@0.25": gain("pulse", 0.25), "pulse@10": gain("pulse", 10),
                    "resp@0.33": gain("resp", 0.33), "resp@1.5": gain("resp", 1.5)}
    return out


def rate_series(peaks, fs, total_s, window_s, per_min):
    """Rate from event times in sliding windows (centre every 1 s): events per minute from mean interval."""
    t = peaks / fs
    centres = np.arange(window_s / 2, total_s - window_s / 2 + 1e-9, 1.0)
    out = np.full(len(centres), np.nan)
    for i, c in enumerate(centres):
        m = (t >= c - window_s / 2) & (t < c + window_s / 2)
        if m.sum() >= 3:
            out[i] = per_min / np.mean(np.diff(t[m]))
    return centres, out


def application(D, F):
    x = D["ppg"]
    N = len(x)
    T = N / FS
    xc = x - x.mean()
    y_p = fir.compensate_delay(fir.fir_filter(F["pulse"]["b"], xc), F["pulse"]["N"])
    y_r = fir.compensate_delay(fir.fir_filter(F["resp"]["b"], xc), F["resp"]["N"])
    guard = int(2 * F["resp"]["delay_s"] * FS)          # ignore filter start-up and the compensated tail
    beats, _ = find_peaks(y_p, distance=int(config.MIN_BEAT_S * FS))
    beats = beats[(beats > guard) & (beats < N - guard)]
    breaths, _ = find_peaks(y_r, distance=int(config.MIN_BREATH_S * FS), prominence=0.3 * np.std(y_r[guard:-guard]))
    breaths = breaths[(breaths > guard) & (breaths < N - guard)]
    tc, hr = rate_series(beats, FS, T, config.HR_WINDOW_S, 60)
    tr, rr = rate_series(breaths, FS, T, config.RR_WINDOW_S, 60)
    ref_b = D["breaths"]
    tr2, rr_ref = rate_series(ref_b, FS, T, config.RR_WINDOW_S, 60)
    num = D["numerics"]
    nm = num.dropna(subset=["PULSE"])
    pulse_ref = np.interp(tc, nm["Time [s]"], nm["PULSE"])
    nr = num.dropna(subset=["RESP"])
    resp_ref_mon = np.interp(tr, nr["Time [s]"], nr["RESP"])
    ok = ~np.isnan(hr)
    okr = ~np.isnan(rr) & ~np.isnan(rr_ref)
    # breath-by-breath matching: an extracted breath "hits" if within 1.5 s of an annotated breath
    tol = 1.5 * FS
    inner = ref_b[(ref_b > guard) & (ref_b < N - guard)]
    hits = sum(np.any(np.abs(breaths - r) <= tol) for r in inner)
    return {
        "y_pulse": y_p, "y_resp": y_r, "beats": beats, "breaths": breaths, "guard": guard,
        "t_hr": tc, "hr": hr, "pulse_ref": pulse_ref, "t_rr": tr, "rr": rr, "rr_ref": rr_ref, "rr_mon": resp_ref_mon,
        "hr_mean": float(np.nanmean(hr)), "hr_ref_mean": float(np.mean(pulse_ref[ok])),
        "hr_mae": float(np.mean(np.abs(hr[ok] - pulse_ref[ok]))),
        "rr_mean": float(np.nanmean(rr)), "rr_ref_mean": float(np.mean(rr_ref[okr])),
        "rr_mae": float(np.mean(np.abs(rr[okr] - rr_ref[okr]))),
        "rr_mon_mean": float(np.mean(resp_ref_mon)),
        "n_beats": len(beats), "n_breaths": len(breaths), "n_ref_breaths": len(inner),
        "breath_sensitivity": float(hits / len(inner)), "pulse_missing": int(num["PULSE"].isna().sum()),
    }


def taps_study(D, sizes=(251, 501, 751, 1001)):
    """Longer band-pass filters reject breathing better; does the heart-rate error follow?"""
    x = D["ppg"] - D["ppg"].mean()
    num = D["numerics"].dropna(subset=["PULSE"])
    rows = []
    for N in sizes:
        b = fir.design_bandpass(*config.PULSE_BAND, FS, N, config.DEFAULT_WINDOW)
        y = fir.compensate_delay(fir.fir_filter(b, x), N)
        g = max(N, int(2 * (config.N_RESP - 1) / 2))
        pk, _ = find_peaks(y, distance=int(config.MIN_BEAT_S * FS))
        pk = pk[(pk > g) & (pk < len(y) - g)]
        tc, hr = rate_series(pk, FS, len(y) / FS, config.HR_WINDOW_S, 60)
        ref = np.interp(tc, num["Time [s]"], num["PULSE"])
        ok = ~np.isnan(hr)
        f, H = fir.freq_response(b, FS)
        rows.append({"N": N, "delay_s": (N - 1) / 2 / FS,
                     "gain_025": float(20 * np.log10(np.abs(H[np.argmin(np.abs(f - 0.25))]))),
                     "mae": float(np.mean(np.abs(hr[ok] - ref[ok]))), "hr_max": float(np.nanmax(hr)),
                     "frac_within_2": float(np.mean(np.abs(hr[ok] - ref[ok]) <= 2))})
    return rows


def spectrum(x, nfft=1 << 15):
    xc = x - x.mean()
    X = np.abs(np.fft.rfft(xc * np.hanning(len(xc)), nfft))
    f = np.fft.rfftfreq(nfft, 1 / FS)
    return f, 20 * np.log10(X / X.max() + 1e-12)


def method_check(D, F):
    from scipy.signal import lfilter
    x = D["ppg"] - D["ppg"].mean()
    b = F["pulse"]["b"]
    return {"vs_lfilter": float(np.max(np.abs(fir.fir_filter(b, x) - lfilter(b, 1, x)))),
            "symmetry": float(np.max(np.abs(b - b[::-1]))),
            "tau_spread": float(np.nanstd(F["pulse"]["tau"][(F["pulse"]["fg"] > 0.7) & (F["pulse"]["fg"] < 3.5)]))}


def run_all():
    D = data_loader.load()
    F = filters()
    A = application(D, F)
    fx, Sx = spectrum(D["ppg"])
    _, Sp = spectrum(A["y_pulse"][A["guard"]:-A["guard"]])
    _, Sr = spectrum(A["y_resp"][A["guard"]:-A["guard"]])
    f0 = fx[np.argmax(np.where(fx > 0.6, Sx, -999))]
    return {"D": D, "F": F, "win": window_study(), "kaiser": kaiser_study(), "app": A,
            "spec": {"f": fx, "x": Sx, "pulse": Sp, "resp": Sr, "f_cardiac": float(f0)},
            "methods": method_check(D, F), "taps": taps_study(D)}
