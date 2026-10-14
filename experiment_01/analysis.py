"""
analysis.py
-----------
Real-world analyses built on top of the basic operations in ``signal_ops``.

Each function takes the dataset, applies one (or a chain of) DT operations, and
returns both the resulting signals and the numbers quoted in the report.
"""

import numpy as np
import pandas as pd

import config
import signal_ops as ops


def to_signal(df: pd.DataFrame, column: str, name: str | None = None) -> ops.DTSignal:
    label, unit = config.SIGNAL_INFO[column]
    return ops.DTSignal(df["n"].to_numpy(), df[column].to_numpy(), name or label, unit)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# --------------------------------------------------------------------------
# A. Addition -- irradiance closure:  GHI = direct + diffuse
# --------------------------------------------------------------------------
def irradiance_closure(df):
    direct = to_signal(df, "direct_wm2", "Direct  x₁[n]")
    diffuse = to_signal(df, "diffuse_wm2", "Diffuse  x₂[n]")
    ghi_ref = to_signal(df, "ghi_wm2", "Reported GHI")
    ghi = ops.add(direct, diffuse, "x₁[n] + x₂[n]")
    residual = ops.subtract(ghi, ghi_ref, "Residual")
    daylight = ghi_ref.x > 0
    return {
        "direct": direct, "diffuse": diffuse, "ghi": ghi, "ghi_ref": ghi_ref, "residual": residual,
        "max_abs_error": float(np.max(np.abs(residual.x))),
        "diffuse_fraction_pct": float(100 * diffuse.x[daylight].sum() / ghi_ref.x[daylight].sum()),
        "peak_ghi": float(ghi.x.max()),
        "peak_time": df["time"].iloc[int(np.argmax(ghi.x))],
    }


# --------------------------------------------------------------------------
# B. Subtraction -- dew-point depression (fog / condensation indicator)
# --------------------------------------------------------------------------
def dewpoint_depression(df):
    temp = to_signal(df, "temp_c", "Air temperature  x₁[n]")
    dew = to_signal(df, "dewpoint_c", "Dew point  x₂[n]")
    dpd = ops.subtract(temp, dew, "DPD  x₁[n] − x₂[n]")
    risk = dpd.x < config.FOG_DPD_THRESHOLD
    hours = df["time"].dt.hour.to_numpy()
    return {
        "temp": temp, "dew": dew, "dpd": dpd,
        "min_dpd": float(dpd.x.min()), "max_dpd": float(dpd.x.max()),
        "mean_dpd_night": float(dpd.x[(hours < 6)].mean()),
        "mean_dpd_afternoon": float(dpd.x[(hours >= 13) & (hours <= 16)].mean()),
        "fog_risk_hours": int(risk.sum()),
        "fog_risk_times": df["time"][risk].dt.strftime("%d-%b %H:%M").tolist(),
        "corr_dpd_rh": float(np.corrcoef(dpd.x, df["rh_pct"])[0, 1]),
    }


# --------------------------------------------------------------------------
# C. Multiplication -- temperature-corrected PV power  P[n] = G'[n] · f_T[n]
# --------------------------------------------------------------------------
def pv_power(df):
    ghi = to_signal(df, "ghi_wm2", "Irradiance G[n]")
    temp = to_signal(df, "temp_c")

    # Cell temperature (NOCT model):  Tc = Ta + (NOCT - 20)/800 * G
    t_cell = temp.x + (config.PV_NOCT - 20.0) / 800.0 * ghi.x
    derate = ops.DTSignal(ghi.n, 1.0 + config.PV_GAMMA * (t_cell - config.STC_TEMPERATURE),
                          "Thermal factor f_T[n]", "")
    g_norm = ops.amplitude_scale(ghi, config.PV_RATED_KW / config.STC_IRRADIANCE, "G[n]/1000")
    g_norm.unit = "kW"
    power = ops.multiply(g_norm, derate, "P[n] = G'[n] · f_T[n]", unit="kW")
    ideal = g_norm  # what the panel would give at a constant 25 °C cell temperature

    days = df["time"].dt.date.to_numpy()
    daily = pd.DataFrame({"date": days, "ideal": ideal.x, "actual": power.x}) \
        .groupby("date").sum()            # Ts = 1 h  ->  sum of kW = kWh
    daylight = ghi.x > 0
    return {
        "ghi": ghi, "derate": derate, "g_norm": g_norm, "power": power, "t_cell": t_cell,
        "daily_kwh": daily,
        "mean_daily_kwh": float(daily["actual"].mean()),
        "thermal_loss_pct": float(100 * (1 - power.x.sum() / ideal.x.sum())),
        "min_derate": float(derate.x[daylight].min()),
        "max_t_cell": float(t_cell.max()),
    }


# --------------------------------------------------------------------------
# D. Amplitude scaling -- sizing a rooftop array (1 kWp -> A kWp)
# --------------------------------------------------------------------------
def array_scaling(power: ops.DTSignal, A: float = config.AMP_SCALE_A):
    scaled = ops.amplitude_scale(power, A, f"{A:g}·P[n]  ({A:g} kWp)")
    soiled = ops.amplitude_scale(power, 0.85, "0.85·P[n]  (15 % soiling)")
    return {
        "base": power, "scaled": scaled, "soiled": soiled, "A": A,
        "energy_ratio": ops.energy(scaled) / ops.energy(power),     # should be A^2
        "kwh_ratio": scaled.x.sum() / power.x.sum(),                  # should be A
        "peak_scaled": float(scaled.x.max()),
    }


# --------------------------------------------------------------------------
# E. Time shifting -- persistence forecast  x̂[n] = x[n - 24]
# --------------------------------------------------------------------------
def persistence_forecast(df, column="temp_c", k: int = config.SHIFT_K):
    x = to_signal(df, column, "Observed x[n]")
    forecast = ops.time_shift(x, k, f"Forecast x[n − {k}]")
    advanced = ops.time_shift(x, -k, f"x[n + {k}]")

    # Score only where both observation and forecast exist: n = k ... N-1
    valid = slice(k, len(x))
    obs, fc = x.x[valid], forecast.x[: len(x) - k]
    clim = np.full_like(obs, x.x.mean())       # naive "weekly mean" baseline
    return {
        "x": x, "forecast": forecast, "advanced": advanced, "k": k,
        "error": ops.DTSignal(x.n[valid], obs - fc, "Error x[n] − x[n−k]", x.unit),
        "mae": float(np.mean(np.abs(obs - fc))),
        "rmse": _rmse(obs, fc),
        "rmse_climatology": _rmse(obs, clim),
        "skill_pct": float(100 * (1 - _rmse(obs, fc) / _rmse(obs, clim))),
    }


# --------------------------------------------------------------------------
# F. Time reversal -- folding and even / odd decomposition
# --------------------------------------------------------------------------
def reversal_demo(df, column="temp_c", length: int = 12):
    seg = ops.DTSignal.from_array(df[column].to_numpy()[:length], 0, "x[n]",
                                  config.SIGNAL_INFO[column][1])
    folded = ops.time_reverse(seg)
    xe, xo = ops.even_odd_parts(seg)
    recon = ops.add(xe, xo, "x_e + x_o")
    _, seg_aligned, _ = ops._align(seg, recon)
    return {
        "x": seg, "folded": folded, "even": xe, "odd": xo,
        "recon_error": float(np.max(np.abs(recon.x - seg_aligned))),
        "even_symmetry_error": float(np.max(np.abs(xe.x - xe.x[::-1]))),
        "odd_symmetry_error": float(np.max(np.abs(xo.x + xo.x[::-1]))),
    }


# --------------------------------------------------------------------------
# G. Time scaling -- 3-hourly synoptic reporting and reconstruction
# --------------------------------------------------------------------------
def time_scaling(df, column="temp_c", D: int = config.DECIMATION_D, L: int = config.INTERP_L):
    x = to_signal(df, column, "Hourly x[n]")
    down = ops.downsample(x, D, f"x[{D}n]  ({D}-hourly)")
    up = ops.upsample(down, L, f"x[n/{L}]  (zero-stuffed)")
    recon = ops.linear_interpolate(up, L, "Linear interpolation")

    n_common = min(len(recon), len(x))
    days = df["time"].dt.date.to_numpy()
    true_max = pd.Series(x.x).groupby(days).max()
    down_days = days[x.n % D == 0]
    synoptic_max = pd.Series(down.x).groupby(down_days).max()
    return {
        "x": x, "down": down, "up": up, "recon": recon, "D": D, "L": L,
        "len_x": len(x), "len_down": len(down), "len_up": len(up),
        "recon_rmse": _rmse(recon.x[:n_common], x.x[:n_common]),
        "recon_max_err": float(np.max(np.abs(recon.x[:n_common] - x.x[:n_common]))),
        "daily_max_true": true_max, "daily_max_synoptic": synoptic_max,
        "mean_peak_miss": float((true_max - synoptic_max).mean()),
        "energy_ratio": ops.energy(down) / ops.energy(x),
    }


# --------------------------------------------------------------------------
# H. Convolution -- moving-average smoothing of the diurnal cycle
# --------------------------------------------------------------------------
def moving_average(df, column="temp_c", M: int = config.MA_LENGTH_M, M_short: int = 3):
    x = to_signal(df, column, "x[n]")
    h = ops.moving_average_kernel(M)
    h_short = ops.moving_average_kernel(M_short)
    y = ops.convolve(x, h, f"y[n] = x[n] * h[n],  M = {M}")
    y_short = ops.convolve(x, h_short, f"M = {M_short}")

    # 'Full-overlap' part of the output: no partial windows at the edges
    full = slice(M - 1, len(x))
    days = df["time"].dt.date.to_numpy()
    daily_range = pd.Series(x.x).groupby(days).agg(lambda v: v.max() - v.min())
    return {
        "x": x, "h": h, "y": y, "y_short": y_short, "M": M, "M_short": M_short,
        "len_x": len(x), "len_h": len(h), "len_y": len(y),
        "std_raw": float(np.std(x.x)), "std_smooth": float(np.std(y.x[full])),
        "smooth_min": float(y.x[full].min()), "smooth_max": float(y.x[full].max()),
        "mean_diurnal_range": float(daily_range.mean()),
        "group_delay": (M - 1) / 2,
        "dc_gain": float(h.x.sum()),
    }


def run_all(df):
    """Run every analysis with the default parameters."""
    pv = pv_power(df)
    return {
        "addition": irradiance_closure(df),
        "subtraction": dewpoint_depression(df),
        "multiplication": pv,
        "scaling": array_scaling(pv["power"]),
        "shift": persistence_forecast(df),
        "reversal": reversal_demo(df),
        "time_scaling": time_scaling(df),
        "convolution": moving_average(df),
    }
