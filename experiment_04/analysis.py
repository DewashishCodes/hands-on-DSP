"""
analysis.py
-----------
Builds the circular wind-direction signals and runs the circular-convolution
studies quoted in the report.
"""

import numpy as np

import circular as cc
import config
import data_loader


def direction_histogram(df, months=None):
    d = df if months is None else df[df["month"].isin(months)]
    k = d.loc[d["k"] >= 0, "k"].to_numpy()
    return np.bincount(k, minlength=config.N_BINS).astype(float)


def histogram_by_speed(df):
    d = df[df["k"] >= 0]
    speeds = np.arange(1, config.MAX_SPEED_KT + 1)
    H = np.zeros((len(speeds), config.N_BINS))
    s_idx = np.clip(np.round(d["speed_kt"]).astype(int), 1, config.MAX_SPEED_KT) - 1
    np.add.at(H, (s_idx.to_numpy(), d["k"].to_numpy()), 1)
    return H, speeds


def seasonal(df):
    out = {}
    for name, months in config.SEASONS.items():
        p = direction_histogram(df, months)
        mode = int(np.argmax(cc.smooth_circular(p, config.SMOOTH_KERNEL)))
        d = df[df["month"].isin(months)]
        out[name] = {"p": p, "mode_deg": mode * config.BIN_DEG, "n": int(p.sum()),
                     "speed_mean": float(d["speed_kt"].mean())}
    return out


def smoothing_study(p):
    w = config.SMOOTH_KERNEL
    y_c = cc.smooth_circular(p, w)
    y_l = cc.smooth_linear(p, w)
    diff = y_c - y_l
    return {
        "p": p, "circ": y_c, "lin": y_l, "diff": diff,
        "sum_p": float(p.sum()), "sum_circ": float(y_c.sum()), "sum_lin": float(y_l.sum()),
        "seam_err": float(np.max(np.abs(diff))),
        "seam_bins": [int(i) for i in np.nonzero(np.abs(diff) > 1e-9)[0]],
        "interior_err": float(np.max(np.abs(diff[1:-1]))),
    }


def method_agreement(p):
    """The three methods of computing circular convolution give the same answer."""
    h = cc.centred_kernel(config.SMOOTH_KERNEL, len(p))
    a = cc.circular_convolve(p, h)
    b = cc.circular_convolve_matrix(p, h)
    c = cc.circular_via_linear(p, h)
    return {"direct_vs_matrix": float(np.max(np.abs(a - b))),
            "direct_vs_wrap": float(np.max(np.abs(a - c)))}


def linear_vs_circular(p):
    """N = 36 circular vs linear (length 38) and zero-padded N = 38 circular."""
    w = np.asarray(config.SMOOTH_KERNEL)
    lin = np.convolve(p, w)                            # length 36 + 3 - 1 = 38
    circ36 = cc.circular_convolve_matrix(p, w, len(p))
    circ38 = cc.circular_convolve_matrix(p, w, len(p) + len(w) - 1)
    return {"lin": lin, "circ36": circ36, "circ38": circ38,
            "wrap_ok": float(np.max(np.abs(cc.wrap(lin, len(p)) - circ36))),
            "pad_ok": float(np.max(np.abs(circ38 - lin)))}


def runway_study(df):
    H, speeds = histogram_by_speed(df)
    always = int((df["calm"] | df["variable"]).sum())
    total = len(df)
    out = {}
    for lim in config.CROSSWIND_LIMITS_KT:
        cov = cc.runway_coverage(H, speeds, lim, always, total, config.BIN_DEG)
        best = int(np.argmax(cov[:18]))
        out[lim] = {
            "coverage": cov, "best_deg": best * config.BIN_DEG, "best": float(cov[best]),
            "worst_deg": int(np.argmin(cov[:18])) * config.BIN_DEG, "worst": float(cov[:18].min()),
            "runways": {r: float(cov[int(h // config.BIN_DEG)]) for r, h in config.RUNWAYS.items()},
            "ok_headings": [int(i) * config.BIN_DEG for i in np.nonzero(cov[:18] >= config.ICAO_USABILITY)[0]],
        }
    # brute-force check of the convolution result
    lim = config.DEFAULT_LIMIT_KT
    d = df[df["k"] >= 0]
    brute = np.array([
        ((np.abs(np.round(d["speed_kt"]).clip(1, config.MAX_SPEED_KT) *
                 np.sin(np.deg2rad(d["k"] * config.BIN_DEG - h * config.BIN_DEG))) <= lim + 1e-9).sum()
         + always) / total for h in range(config.N_BINS)])
    out["brute_force_err"] = float(np.max(np.abs(brute - out[lim]["coverage"])))
    out["always_ok"] = always
    return out


def run_all():
    df = data_loader.load_dataset()
    p = direction_histogram(df)
    R = {
        "df": df, "info": data_loader.describe_dataset(df), "p": p,
        "seasons": seasonal(df),
        "smoothing": smoothing_study(p),
        "methods": method_agreement(p),
        "lin_circ": linear_vs_circular(p),
        "runway": runway_study(df),
        "towards": cc.circular_shift(p, config.N_BINS // 2),
    }
    return R
