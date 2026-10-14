"""
analysis.py
-----------
Applies the black-box LTI tests to the four systems and extracts the
real-world quantities quoted in the report.
"""

import numpy as np
import pandas as pd

import config
import systems as sy


def inputs(df):
    """x1 = PM2.5 and x2 = NO2: two real, differently shaped test inputs."""
    return df["pm25"].to_numpy(), df["no2"].to_numpy()


def test_battery(df):
    """Linearity and time-invariance results for every system."""
    x1, x2 = inputs(df)
    out = {}
    for key, s in sy.default_systems().items():
        res = sy.classify(s, x1, x2)
        res["system"] = s
        res["matches_expectation"] = (res["linear"]["holds"] == s.expected_linear and
                                      res["ti"]["holds"] == s.expected_ti)
        out[key] = res
    return out


def time_responses():
    """Impulse and step responses of the two LTI systems, with response-time metrics."""
    S = sy.default_systems()
    out = {}
    for key in ("S1", "S2"):
        h = sy.impulse_response(S[key])
        s = sy.step_response(S[key])
        out[key] = {
            "h": h, "s": s,
            "h_sum": float(np.sum(sy.impulse_response(S[key], 2000))),     # DC gain
            "t63": int(np.argmax(s >= 1 - np.exp(-1))),                     # 63.2 % rise time
            "t95": int(np.argmax(s >= 0.95)),
        }
    a = config.EWMA_ALPHA
    out["S2"]["tau_theory"] = -1.0 / np.log(1.0 - a)                       # samples
    out["S2"]["h_theory"] = a * (1 - a) ** np.arange(config.RESPONSE_LENGTH)
    return out


def air_quality_application(df):
    """What the LTI systems mean for the real data: standard exceedance, lag and AQI."""
    S = sy.default_systems()
    x = df["pm25"].to_numpy()
    t = df["time"]
    ma = S["S1"](x)
    sm = S["S2"](x)
    aqi = S["S3"](x)

    full = slice(config.MA_LENGTH_M - 1, len(x))          # first complete 24-h window
    exceed = np.zeros(len(x), dtype=bool)
    exceed[full] = ma[full] > config.NAAQS_PM25_24H

    i_raw, i_ma, i_sm = int(np.argmax(x)), int(np.argmax(ma)), int(np.argmax(sm))
    conv = sy.convolution_check(S["S1"], x)
    conv2 = sy.convolution_check(S["S2"], x)

    def category(v):
        return next(name for lim, name in config.AQI_CATEGORIES if v <= lim)

    cats = pd.Series([category(v) for v in aqi])
    return {
        "x": x, "ma": ma, "smooth": sm, "aqi": aqi, "exceed": exceed,
        "peak_raw": float(x[i_raw]), "peak_raw_time": t.iloc[i_raw],
        "peak_ma": float(ma[i_ma]), "peak_ma_time": t.iloc[i_ma],
        "peak_sm": float(sm[i_sm]), "peak_sm_time": t.iloc[i_sm],
        "lag_ma_h": i_ma - i_raw, "lag_sm_h": i_sm - i_raw,
        "exceed_hours": int(exceed.sum()),
        "exceed_start": t[exceed].min() if exceed.any() else None,
        "exceed_end": t[exceed].max() if exceed.any() else None,
        "hourly_over_60": int((x > config.NAAQS_PM25_24H).sum()),
        "peak_aqi": float(aqi[i_raw]), "peak_category": category(aqi[i_raw]),
        "category_hours": cats.value_counts().to_dict(),
        "conv_error_ma": conv["error"], "conv_error_ewma": conv2["error"],
        "pre_mean": float(x[t < pd.Timestamp("2025-10-18")].mean()),
        "post_mean": float(x[t >= pd.Timestamp("2025-10-22")].mean()),
    }


def aqi_nonlinearity_examples():
    """Doubling the concentration does not double the AQI (homogeneity fails)."""
    rows = []
    for c in (20, 45, 75, 110):
        rows.append((c, float(sy.aqi_pm25([c])[0]), float(sy.aqi_pm25([2 * c])[0])))
    return rows


def run_all(df):
    return {
        "battery": test_battery(df),
        "responses": time_responses(),
        "application": air_quality_application(df),
        "aqi_examples": aqi_nonlinearity_examples(),
    }
