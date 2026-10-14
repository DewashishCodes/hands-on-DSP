"""
plots.py
--------
Publication-style figures for Experiment 1.

Every function receives already-computed results (from ``analysis``) and returns
a Matplotlib ``Figure``; saving is done by the caller.  This keeps plotting
independent of processing, and lets the GUI reuse the same drawing helpers.
"""

import matplotlib.pyplot as plt
import numpy as np

import config
import signal_ops as ops

# --------------------------------------------------------------------------
# Visual style: one colour per role, fixed order, colour-blind-validated
# --------------------------------------------------------------------------
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#ffffff", "wash": "#f3f2ee",
}

FULL_W = 6.7        # inches: full text width of an A4 page with 2 cm margins
DPI = 220


def apply_style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8.2,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 7.5,
        "axes.labelcolor": C["ink2"],
        "axes.edgecolor": C["axis"],
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.5,
        "xtick.color": C["muted"],
        "ytick.color": C["muted"],
        "xtick.labelcolor": C["ink2"],
        "ytick.labelcolor": C["ink2"],
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "lines.linewidth": 1.4,
        "figure.facecolor": C["surface"],
        "axes.facecolor": C["surface"],
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


apply_style()


# --------------------------------------------------------------------------
# Small drawing helpers (also used by the GUI)
# --------------------------------------------------------------------------
def stem(ax, n, x, color, label=None, markersize=3.0, lw=0.9):
    """Clean stem plot: thin stems, filled heads, baseline in axis colour."""
    markerline, stemlines, _ = ax.stem(n, x, label=label, basefmt=" ")
    plt.setp(stemlines, color=color, linewidth=lw)
    plt.setp(markerline, color=color, markersize=markersize, markeredgecolor="white",
             markeredgewidth=0.5)
    ax.axhline(0, color=C["axis"], linewidth=0.7, zorder=0)
    return markerline


def day_guides(ax, df, n_lo=None, n_hi=None):
    """Faint vertical lines at midnight with the date written at the top."""
    n_lo = df["n"].min() if n_lo is None else n_lo
    n_hi = df["n"].max() if n_hi is None else n_hi
    for n0 in range(0, len(df), config.SAMPLES_PER_DAY):
        if n_lo <= n0 <= n_hi:
            ax.axvline(n0, color=C["axis"], linewidth=0.6, linestyle=(0, (3, 3)), zorder=0)
            ax.text(n0 + 0.6, 0.985, df["time"].iloc[n0].strftime("%d %b"),
                    transform=ax.get_xaxis_transform(), fontsize=6.2, color=C["muted"],
                    va="top", ha="left")


def top_legend(ax, ncol=3, **kw):
    """Legend on the title row, right-aligned, so it never covers data or the title."""
    return ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=ncol, fontsize=6.8,
                     handlelength=1.5, columnspacing=1.0, borderaxespad=0.1, **kw)


HOURS = "Sample index n  (1 sample = 1 hour)"


# --------------------------------------------------------------------------
# Figure 1: the raw signals
# --------------------------------------------------------------------------
def fig_raw_signals(df):
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.3), sharex=True,
                             gridspec_kw={"hspace": 0.38})
    n = df["n"]
    a = axes[0]
    a.plot(n, df["temp_c"], color=C["blue"], label="Air temperature")
    a.plot(n, df["dewpoint_c"], color=C["orange"], label="Dew point")
    a.set_ylabel("°C")
    a.set_title("(a) Temperature and dew point")
    top_legend(a, ncol=2)

    b = axes[1]
    b.plot(n, df["ghi_wm2"], color=C["ink2"], lw=1.1, label="Global (GHI)")
    b.plot(n, df["direct_wm2"], color=C["blue"], label="Direct")
    b.plot(n, df["diffuse_wm2"], color=C["aqua"], label="Diffuse")
    b.set_ylabel("W/m²")
    b.set_title("(b) Horizontal solar irradiance")
    b.set_xlabel(HOURS)
    top_legend(b, ncol=3)
    for ax in axes:
        day_guides(ax, df)
        ax.set_xlim(-2, len(df) + 1)
    return fig


# --------------------------------------------------------------------------
# Figure 2: the three arithmetic operations
# --------------------------------------------------------------------------
def fig_arithmetic(R, df, n_lo=24, n_hi=71):
    add, sub, mul = R["addition"], R["subtraction"], R["multiplication"]
    fig, axes = plt.subplots(3, 1, figsize=(FULL_W, 5.0), gridspec_kw={"hspace": 0.62})
    sel = slice(n_lo, n_hi + 1)

    a = axes[0]
    a.plot(add["direct"].n[sel], add["direct"].x[sel], color=C["blue"], label="x₁ direct")
    a.plot(add["diffuse"].n[sel], add["diffuse"].x[sel], color=C["aqua"], label="x₂ diffuse")
    a.plot(add["ghi"].n[sel], add["ghi"].x[sel], color=C["orange"], lw=1.9, label="y = x₁ + x₂")
    a.plot(add["ghi_ref"].n[sel], add["ghi_ref"].x[sel], "o", color=C["ink"], ms=2.2,
           label="Reported GHI")
    a.set_ylabel("W/m²")
    a.set_ylim(0, 1000)
    a.set_title("(a) Addition")
    top_legend(a, ncol=4)
    day_guides(a, df, n_lo, n_hi)
    a.set_xlim(n_lo - 1, n_hi + 1)

    b = axes[1]
    n = sub["dpd"].n
    b.plot(n, sub["dpd"].x, color=C["violet"], label="y = T − T_d")
    b.axhspan(0, config.FOG_DPD_THRESHOLD, color=C["red"], alpha=0.10, lw=0)
    b.axhline(config.FOG_DPD_THRESHOLD, color=C["red"], lw=0.8, ls="--")
    risk = sub["dpd"].x < config.FOG_DPD_THRESHOLD
    b.plot(n[risk], sub["dpd"].x[risk], "o", color=C["red"], ms=3.4, mec="white", mew=0.5,
           label=f"Fog risk (< {config.FOG_DPD_THRESHOLD:g} °C)")
    b.set_ylabel("°C")
    b.set_ylim(0, 25)
    b.set_title("(b) Subtraction: dew-point depression")
    top_legend(b, ncol=2)
    day_guides(b, df)
    b.set_xlim(-2, len(df) + 1)

    c = axes[2]
    n = mul["power"].n[sel]
    c.plot(n, mul["g_norm"].x[sel], color=C["orange"], lw=1.0, ls="--", label="x₁ = G/1000 (ideal)")
    c.plot(n, mul["power"].x[sel], color=C["blue"], lw=1.7, label="y = x₁ · x₂")
    c.fill_between(n, mul["power"].x[sel], mul["g_norm"].x[sel], color=C["red"], alpha=0.15, lw=0,
                   label="Thermal loss")
    c.set_ylabel("kW")
    c.set_ylim(0, 1.0)
    c.set_title("(c) Multiplication: PV power")
    c.set_xlabel(HOURS)
    top_legend(c, ncol=3)
    day_guides(c, df, n_lo, n_hi)
    c.set_xlim(n_lo - 1, n_hi + 1)
    return fig


# --------------------------------------------------------------------------
# Figure 3: operations on the independent variable n
# --------------------------------------------------------------------------
def fig_index_ops(R, df, k=3, length=10):
    x = ops.DTSignal.from_array(df["temp_c"].to_numpy()[:length], 0)
    rv, ts = R["reversal"], R["time_scaling"]
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 3.9), gridspec_kw={"hspace": 0.55, "wspace": 0.18})

    a = axes[0, 0]
    d, adv = ops.time_shift(x, k), ops.time_shift(x, -k)
    stem(a, x.n, x.x, C["muted"], "x[n]", markersize=2.4)
    stem(a, d.n + 0.18, d.x, C["orange"], f"x[n − {k}]", markersize=2.4)
    stem(a, adv.n - 0.18, adv.x, C["aqua"], f"x[n + {k}]", markersize=2.4)
    a.axvline(0, color=C["muted"], lw=0.7)
    a.set_xticks(range(-k, length + k, 2))
    a.set_title(f"(a) Time shift, k = ±{k}")
    a.legend(loc="upper left", ncol=3, fontsize=6.6, handlelength=1.0, columnspacing=0.8)

    b = axes[0, 1]
    stem(b, rv["x"].n + 0.15, rv["x"].x, C["blue"], "x[n]", markersize=2.4)
    stem(b, rv["folded"].n - 0.15, rv["folded"].x, C["orange"], "x[−n]", markersize=2.4)
    b.axvline(0, color=C["muted"], lw=0.7)
    b.set_xticks(range(-11, 12, 3))
    b.set_title("(b) Time reversal")
    b.legend(loc="upper center", ncol=2, fontsize=6.6, handlelength=1.0)

    c = axes[1, 0]
    L24 = 24
    keep = (ts["x"].n % ts["D"] == 0)[:L24]
    stem(c, ts["x"].n[:L24], ts["x"].x[:L24], C["axis"], "x[n]", markersize=2.2)
    stem(c, ts["x"].n[:L24][keep], ts["x"].x[:L24][keep], C["blue"], f"kept → x[{ts['D']}n]",
         markersize=2.8)
    c.set_title(f"(c) Down-sampling, D = {ts['D']}")
    c.set_xlabel("n")
    c.legend(loc="upper left", ncol=2, fontsize=6.6, handlelength=1.0)

    dd = axes[1, 1]
    nd = L24 // ts["D"]
    nu = (nd - 1) * ts["D"] + 1
    stem(dd, ts["up"].n[:nu], ts["up"].x[:nu], C["orange"], f"x[n/{ts['L']}] (zeros)", markersize=2.4)
    dd.plot(ts["recon"].n[:nu], ts["recon"].x[:nu], color=C["aqua"], lw=1.1, label="linear interp.")
    dd.set_title(f"(d) Up-sampling, L = {ts['L']}")
    dd.set_xlabel("n")
    dd.legend(loc="upper left", ncol=2, fontsize=6.6, handlelength=1.0)

    for ax in axes.flat:
        ax.set_ylim(0, 45)
        ax.set_ylabel("°C")
    return fig


# --------------------------------------------------------------------------
# Figure 4: persistence forecast (delay by one day)
# --------------------------------------------------------------------------
def fig_persistence(r, df):
    fig, ax = plt.subplots(figsize=(FULL_W, 2.2))
    k = r["k"]
    x, f = r["x"], r["forecast"]
    ax.plot(x.n, x.x, color=C["blue"], label="Observed x[n]")
    keep = f.n <= x.n[-1]
    ax.plot(f.n[keep], f.x[keep], color=C["orange"], ls="--", label=f"Forecast x[n − {k}]")
    ax.axvspan(-2, k - 0.5, color=C["wash"], lw=0)
    ax.text(k / 2, 0.06, "no forecast", transform=ax.get_xaxis_transform(),
            ha="center", fontsize=6.5, color=C["muted"])
    ax.set_ylabel("°C")
    ax.set_title(f"Delay by k = {k}: one-day-ahead persistence forecast")
    ax.set_xlabel(HOURS)
    top_legend(ax, ncol=2)
    day_guides(ax, df)
    ax.set_xlim(-2, len(df) + 1)
    return fig


# --------------------------------------------------------------------------
# Figure 5: convolution
# --------------------------------------------------------------------------
def fig_convolution(r, df):
    fig, ax = plt.subplots(figsize=(FULL_W, 2.5))
    x, y, ys = r["x"], r["y"], r["y_short"]
    M = r["M"]
    ax.axvspan(-1, M - 1.5, color=C["wash"], lw=0)
    ax.axvspan(len(x) - 0.5, y.n[-1] + 1, color=C["wash"], lw=0)
    for xc in [(M - 2) / 2, (len(x) + y.n[-1]) / 2]:
        ax.text(xc, 0.05, "edge", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=6.5, color=C["muted"])
    ax.plot(x.n, x.x, color=C["blue"], lw=0.9, label="x[n]")
    ax.plot(ys.n, ys.x, color=C["aqua"], lw=1.1, label=f"M = {r['M_short']}")
    ax.plot(y.n, y.x, color=C["orange"], lw=2.0, label=f"M = {M}")
    ax.set_ylim(0, 37)
    ax.set_ylabel("°C")
    ax.set_title(f"y[n] = x[n] * h[n],  h[n] = 1/M   (output length {r['len_x']} + {M} − 1 = {r['len_y']})")
    ax.set_xlabel(HOURS)
    ax.legend(loc="lower center", ncol=3)
    day_guides(ax, df)
    ax.set_xlim(-2, y.n[-1] + 2)
    return fig


# --------------------------------------------------------------------------
# Generate everything
# --------------------------------------------------------------------------
def make_all_figures(df, results):
    """Return {filename: Figure} for every figure in the report."""
    return {
        "fig1_raw_signals.png": fig_raw_signals(df),
        "fig2_arithmetic.png": fig_arithmetic(results, df),
        "fig3_index_ops.png": fig_index_ops(results, df),
        "fig4_persistence.png": fig_persistence(results["shift"], df),
        "fig5_convolution.png": fig_convolution(results["convolution"], df),
    }
