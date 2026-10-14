"""
analysis.py
-----------
Applies the difference-equation smoothers to the USD/INR series and extracts
the numbers quoted in the report.
"""

import numpy as np
import pandas as pd
from scipy.signal import lfilter

import config
import data_loader
import smoothing as sm


def filters():
    """Every smoother of the study as (b, a) coefficients."""
    e = sm.ema_coeffs(config.ALPHA)
    return {
        f"MA{config.N_MA} (non-recursive)": sm.ma_coeffs(config.N_MA),
        f"MA{config.N_MA} (recursive)": sm.ma_recursive_coeffs(config.N_MA),
        f"EMA{config.N_EMA}": e,
        f"Double EMA{config.N_EMA}": sm.cascade_coeffs(*e, *e),
        "Holt (level + trend)": sm.holt_coeffs(config.HOLT_ALPHA, config.HOLT_BETA),
    }


def ramp_lag(x, y, i0, i1):
    """Empirical lag on a trending stretch: a filter with lag L trails a ramp of slope s by s*L."""
    n = np.arange(i0, i1 + 1)
    slope = np.polyfit(n, x[i0: i1 + 1], 1)[0]
    return float(np.mean(x[i0: i1 + 1] - y[i0: i1 + 1]) / slope), float(slope)


def filter_table(x, zoom):
    out = {}
    for name, (b, a) in filters().items():
        y = sm.difference_equation(b, a, x, y_init=[x[0]] * (len(a) - 1) if len(a) > 1 else None,
                                   x_init=[x[0]] * (len(b) - 1) if len(b) > 1 else None)
        p, z = sm.poles_zeros(b, a)
        warm = 200
        rough = float(np.std(np.diff(y[warm:])) / np.std(np.diff(x[warm:])))
        track = float(np.sqrt(np.mean((y[warm:] - x[warm:]) ** 2)))
        out[name] = {
            "b": b, "a": a, "y": y, "poles": p, "zeros": z,
            "max_pole": float(np.max(np.abs(p))) if len(p) else 0.0,
            "stable": sm.is_stable(b, a), "noise_gain": sm.noise_gain(b, a), "lag": sm.dc_lag(b, a),
            "lag_measured": ramp_lag(x, y, *zoom)[0], "roughness": rough, "rmse": track,
            "ops": sm.ops_per_sample(b, a),
            "lfilter_err": float(np.max(np.abs(sm.difference_equation(b, a, x) - lfilter(b, a, x)))),
        }
    return out


def tradeoff():
    """Noise gain versus lag for the MA and EMA families (theory curves cross-checked numerically)."""
    rows = []
    for L in config.TRADEOFF_LAGS:
        N = 2 * L + 1
        alpha = 1.0 / (L + 1)
        bm, am = sm.ma_coeffs(N)
        be, ae = sm.ema_coeffs(alpha)
        e2 = sm.cascade_coeffs(*sm.ema_coeffs(2 / (L + 2)), *sm.ema_coeffs(2 / (L + 2)))
        rows.append({"L": L, "N": N, "alpha": alpha,
                     "ma_ng": sm.noise_gain(bm, am), "ma_lag": sm.dc_lag(bm, am),
                     "ema_ng": sm.noise_gain(be, ae), "ema_lag": sm.dc_lag(be, ae),
                     "ema2_ng": sm.noise_gain(*e2), "ema2_lag": sm.dc_lag(*e2)})
    return pd.DataFrame(rows)


def initial_conditions(x, n_show=150):
    b, a = sm.ema_coeffs(config.ALPHA)
    y_rest = sm.difference_equation(b, a, x)                     # y[-1] = 0
    y_full, y_zi, y_zs = sm.zero_input_zero_state(b, a, x, [x[0]])
    tol = config.TRANSIENT_TOL
    n_theory = int(np.ceil(np.log(tol) / np.log(1 - config.ALPHA)))
    rel = np.abs(y_rest - y_full) / y_full
    return {"y_rest": y_rest[:n_show], "y_full": y_full[:n_show], "y_zi": y_zi[:n_show],
            "y_zs": y_zs[:n_show], "x": x[:n_show], "n_theory": n_theory,
            "n_measured": int(np.argmax(rel < tol)), "decomp_err": float(np.max(np.abs(y_full - y_zi - y_zs))),
            "first_err": float(y_full[0] - y_rest[0])}


def macd_study(df):
    x = df["inr"].to_numpy()
    ef, es, line, sig = sm.macd(x, config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL)
    cross = sm.crossings(line - sig)
    w = np.linspace(1e-4, np.pi, 2000)
    bf, af = sm.ema_coeffs(2 / (config.MACD_FAST + 1))
    bs, as_ = sm.ema_coeffs(2 / (config.MACD_SLOW + 1))
    H = sm.frequency_response(bf, af, w) - sm.frequency_response(bs, as_, w)
    k = int(np.argmax(np.abs(H)))
    years = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    return {"ef": ef, "es": es, "line": line, "sig": sig, "cross": cross,
            "n_cross": len(cross), "per_year": len(cross) / years,
            "w": w, "H": H, "peak_period_days": float(2 * np.pi / w[k]), "peak_gain": float(np.abs(H[k]))}


def zoom_mask(df):
    return (df["date"] >= config.ZOOM[0]) & (df["date"] <= config.ZOOM[1])


def run_all():
    df = data_loader.load_dataset()
    x = df["inr"].to_numpy()
    z = zoom_mask(df).to_numpy()
    i0, i1 = int(np.argmax(z)), int(len(z) - np.argmax(z[::-1]) - 1)
    F = filter_table(x, (i0, i1))
    return {
        "df": df, "x": x, "info": data_loader.describe_dataset(df), "filters": F,
        "tradeoff": tradeoff(), "ic": initial_conditions(x), "macd": macd_study(df),
        "zoom": (i0, i1),
        "zoom_change": (float(x[i0]), float(x[i1])),
        "zoom_slope": ramp_lag(x, x, i0, i1)[1],
    }
