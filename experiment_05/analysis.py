"""
analysis.py
-----------
Runs the autocorrelation study on the sunspot record and extracts the numbers
quoted in the report.
"""

import numpy as np

import autocorr as ac
import config
import data_loader


def main_acf(x):
    rho = ac.acf(x, config.MAX_LAG)
    k, k_ref, val = ac.dominant_period(rho, config.PEAK_SEARCH)
    peaks = ac.positive_peaks(rho, 60)
    return {
        "rho": rho, "period_lag": k, "period_months": k_ref, "period_years": k_ref / 12,
        "peak_value": val, "peaks": peaks, "peak_values": [float(rho[p]) for p in peaks],
        "zero": ac.first_zero_crossing(rho), "half": ac.lag_below(rho, 0.5),
        "r1": float(rho[1]), "r12": float(rho[12]),
        "bound": ac.confidence_bound(len(x)),
        "min_lag": int(np.argmin(rho[:200])), "min_value": float(rho[:200].min()),
    }


def method_check(x):
    xd = x - x.mean()
    L = 200
    a = ac.autocorr_direct(xd, L)
    b = ac.autocorr_conv(xd)[len(xd) - 1: len(xd) + L]
    c = ac.autocorr_fft(xd, L)
    full = ac.autocorr_conv(xd)
    return {
        "direct_vs_conv": float(np.max(np.abs(a - b)) / a[0]),
        "direct_vs_fft": float(np.max(np.abs(a - c)) / a[0]),
        "symmetry_err": float(np.max(np.abs(full - full[::-1])) / a[0]),
        "max_at_zero": bool(np.all(np.abs(full) <= full[len(xd) - 1] + 1e-9)),
        "r0_equals_energy": float(abs(a[0] - np.sum(xd ** 2)) / a[0]),
        "full": full,
    }


def estimator_study(x):
    L = len(x) - 10
    return {"biased": ac.acf(x, L), "unbiased": ac.acf(x, L, unbiased=True), "L": L}


def noise_study(x):
    """Is what is left after removing the cycle white noise?"""
    sm = ac.running_mean(x, config.SMOOTH_MONTHS)
    h = config.SMOOTH_MONTHS // 2
    resid = (x - sm)[h:-h]
    diff = np.diff(x)
    L = 36
    out = {}
    for name, s in (("residual", resid), ("difference", diff)):
        rho = ac.acf(s, L)
        b = ac.confidence_bound(len(s))
        out[name] = {"rho": rho, "bound": b, "n_outside": int(np.sum(np.abs(rho[1:]) > b)),
                     "r1": float(rho[1]), "std": float(np.std(s))}
    out["smooth"] = sm
    return out


def window_study(df):
    out = []
    for a, b in config.WINDOWS:
        s = df.loc[(df["year"] >= a) & (df["year"] < b), "sn"].to_numpy()
        rho = ac.acf(s, 300)
        k, kr, v = ac.dominant_period(rho, config.PEAK_SEARCH)
        out.append({"span": (a, b), "rho": rho, "period_years": kr / 12, "peak": v, "n": len(s)})
    return out


def noise_robustness(x):
    rng = np.random.default_rng(config.SEED)
    sd = np.std(x)
    out = []
    for lvl in config.NOISE_LEVELS:
        y = x + rng.normal(0, lvl * sd, len(x)) if lvl else x.copy()
        rho = ac.acf(y, config.MAX_LAG)
        k, kr, v = ac.dominant_period(rho, config.PEAK_SEARCH)
        snr_db = np.inf if lvl == 0 else 10 * np.log10(1 / lvl ** 2)
        out.append({"level": lvl, "snr_db": snr_db, "rho": rho, "period_years": kr / 12, "peak": v,
                    "y": y})
    return out


def forecast_note(df, period_months):
    """Rough timing of the next maximum from the ACF period and the last smoothed maximum."""
    sm = ac.running_mean(df["sn"].to_numpy(), config.SMOOTH_MONTHS)
    recent = df.index[df["year"] >= 2019]
    i = int(recent[np.argmax(sm[recent])])
    t_max = df["decimal_year"].iloc[i]
    return {"last_max_year": float(t_max), "last_max_value": float(sm[i]),
            "next_max_year": float(t_max + period_months / 12)}


def run_all():
    df = data_loader.load_dataset()
    x = df["sn"].to_numpy()
    M = main_acf(x)
    return {
        "df": df, "x": x, "info": data_loader.describe_dataset(df),
        "acf": M, "methods": method_check(x), "estimators": estimator_study(x),
        "noise": noise_study(x), "windows": window_study(df),
        "robust": noise_robustness(x), "forecast": forecast_note(df, M["period_months"]),
    }
