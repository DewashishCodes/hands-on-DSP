"""
analysis.py
-----------
Cross-correlates the tide gauges, estimates the tidal travel times and the
wave speed, and extracts the numbers quoted in the report.
"""

import numpy as np

import config
import data_loader
import xcorr as xc

IDS = list(config.STATIONS)


def distances():
    _, la0, lo0 = config.STATIONS[config.REFERENCE]
    return {s: xc.haversine_km(la0, lo0, la, lo) for s, (_, la, lo) in config.STATIONS.items()}


def band(df, sid, which):
    """'raw' water level, 'tidal' band (raw minus 25-h mean) or 'subtidal' band (the 25-h mean)."""
    x = df[sid].to_numpy()
    if which == "raw":
        return x
    low = xc.moving_average(x, config.SUBTIDAL_WINDOW_H)
    return x - low if which == "tidal" else low


def pair_study(df, which="raw"):
    """Delay of every gauge relative to the reference."""
    x = band(df, config.REFERENCE, which)
    d = distances()
    out = {}
    for s in IDS:
        y = band(df, s, which)
        lags, rho = xc.ccf(y, x, config.MAX_LAG_H)
        k, kr, v = xc.peak_lag(lags, rho, config.SEARCH_LAG_H)
        out[s] = {"name": config.STATIONS[s][0], "dist_km": d[s], "lags": lags, "rho": rho,
                  "lag_int": k, "lag_h": kr, "peak": v,
                  "candidates": xc.local_peaks(lags, rho, (-12, 30)),
                  "rho0": float(rho[lags == 0][0])}
    return out


def chain_study(df, which="tidal"):
    """Delays between consecutive gauges: each step is shorter than half a tidal period."""
    steps = []
    for a, b in zip(IDS[:-1], IDS[1:]):
        lags, rho = xc.ccf(band(df, b, which), band(df, a, which), config.MAX_LAG_H)
        k, kr, v = xc.peak_lag(lags, rho, (-6, 6))
        steps.append({"from": config.STATIONS[a][0], "to": config.STATIONS[b][0], "lag_h": kr, "peak": v})
    cum = np.cumsum([0.0] + [s["lag_h"] for s in steps])
    return {"steps": steps, "cumulative": cum}


def speed_study(pairs):
    """Least-squares line distance = c * delay through the origin, then h = c^2 / g."""
    t = np.array([pairs[s]["lag_h"] for s in IDS[1:]])
    dkm = np.array([pairs[s]["dist_km"] for s in IDS[1:]])
    c_kmh = float(np.sum(t * dkm) / np.sum(t * t))
    c_ms = c_kmh / 3.6
    resid = dkm - c_kmh * t
    r2 = 1 - np.sum(resid ** 2) / np.sum((dkm - dkm.mean()) ** 2)
    return {"c_kmh": c_kmh, "c_ms": c_ms, "depth_m": c_ms ** 2 / config.G, "r2": float(r2),
            "c_published": np.sqrt(config.G * config.PUBLISHED_MEAN_DEPTH_M)}


def subtidal_study(df):
    """Remove the tides with a 25-h moving average and correlate what is left (wind-driven setup)."""
    M = config.SUBTIDAL_WINDOW_H
    h = M // 2
    x = xc.moving_average(df[config.REFERENCE].to_numpy(), M)[h:-h]
    out = {"x": x}
    for s in IDS[1:]:
        y = xc.moving_average(df[s].to_numpy(), M)[h:-h]
        lags, rho = xc.ccf(y, x, 72)
        k, kr, v = xc.peak_lag(lags, rho, (-48, 48))
        out[s] = {"y": y, "lags": lags, "rho": rho, "lag_h": kr, "peak": v,
                  "tidal_var_frac": float(1 - np.var(y) / np.var(df[s].to_numpy()))}
    return out


def property_check(df):
    x = df[config.REFERENCE].to_numpy() - df[config.REFERENCE].mean()
    y = df[IDS[-1]].to_numpy() - df[IDS[-1]].mean()
    L = 50
    l1, a = xc.crosscorr_direct(y, x, L)
    l2, b = xc.crosscorr_conv(y, x)
    b = b[(l2 >= -L) & (l2 <= L)]
    l3, c = xc.crosscorr_fft(y, x, L)
    _, ryx = xc.crosscorr_direct(y, x, L)
    _, rxy = xc.crosscorr_direct(x, y, L)
    bound = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))
    return {"direct_vs_conv": float(np.max(np.abs(a - b)) / bound),
            "direct_vs_fft": float(np.max(np.abs(a - c)) / bound),
            "swap_err": float(np.max(np.abs(ryx - rxy[::-1])) / bound),
            "max_abs_over_bound": float(np.max(np.abs(a)) / bound),
            "asymmetry": float(np.max(np.abs(ryx - ryx[::-1])) / bound),
            "ryx": ryx / bound, "rxy": rxy / bound, "lags": l1}


def run_all():
    df = data_loader.load_dataset()
    pairs = pair_study(df, "raw")
    tidal = pair_study(df, "tidal")
    return {
        "df": df, "desc": data_loader.describe_dataset(df), "pairs": pairs, "tidal": tidal,
        "chain": chain_study(df, "tidal"), "chain_raw": chain_study(df, "raw"),
        "speed": speed_study(tidal), "speed_raw": speed_study(pairs),
        "subtidal": subtidal_study(df), "props": property_check(df),
    }
