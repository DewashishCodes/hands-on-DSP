"""
plots.py
--------
Publication-style figures for Experiment 10.

Every function receives already-computed results (from ``analysis``) and returns
a Matplotlib ``Figure``; saving is done by the caller.
"""

import matplotlib.pyplot as plt
import numpy as np

import config

# --------------------------------------------------------------------------
# Visual style (identical to Experiments 1 to 9)
# --------------------------------------------------------------------------
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#ffffff", "wash": "#f3f2ee",
}
KCOL = [C["blue"], C["orange"], C["aqua"]]
FULL_W = 6.7
DPI = 220
MONTH_START = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def apply_style():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 8.2, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.labelsize": 7.5, "axes.labelcolor": C["ink2"],
        "axes.edgecolor": C["axis"], "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.5,
        "xtick.color": C["muted"], "ytick.color": C["muted"],
        "xtick.labelcolor": C["ink2"], "ytick.labelcolor": C["ink2"],
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.frameon": False, "legend.fontsize": 7, "lines.linewidth": 1.4,
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
        "savefig.dpi": DPI, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    })


apply_style()


def stem(ax, n, x, color, label=None, markersize=3.0, lw=0.9):
    markerline, stemlines, _ = ax.stem(n, x, label=label, basefmt=" ")
    plt.setp(stemlines, color=color, linewidth=lw)
    plt.setp(markerline, color=color, markersize=markersize, markeredgecolor="white",
             markeredgewidth=0.5)
    ax.axhline(0, color=C["axis"], linewidth=0.7, zorder=0)
    return markerline


def top_legend(ax, ncol=3, **kw):
    return ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=ncol, fontsize=6.8,
                     handlelength=1.5, columnspacing=1.0, borderaxespad=0.1, **kw)


def month_axis(ax):
    ax.set_xticks(MONTH_START, MONTHS)
    ax.set_xlim(-3, 367)


# --------------------------------------------------------------------------
# Figure 1: raw data and the raw climatology
# --------------------------------------------------------------------------
def fig_raw(R):
    df, clim, x = R["df"], R["clim"], R["x"]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.6), sharex=True,
                             gridspec_kw={"hspace": 0.3, "height_ratios": [2, 1]})
    a = axes[0]
    a.plot(df["k"], df["tmax"], ".", color=C["axis"], ms=1.0, alpha=0.6, label="daily maxima, 1991 to 2020")
    a.plot(np.arange(len(x)), x, color=C["blue"], lw=0.9, label="mean for each calendar day x[k]")
    a.set_ylabel("°C")
    a.set_title("(a) Maximum temperature by calendar day")
    top_legend(a, ncol=2)
    b = axes[1]
    b.bar(np.arange(len(x)), clim["n_tmax"], width=1.0, color=C["violet"])
    b.set_ylabel("Years")
    b.set_ylim(0, 31)
    b.set_title("(b) Number of years with an observation")
    month_axis(b)
    return fig


# --------------------------------------------------------------------------
# Figure 2: circular versus zero-padded linear smoothing
# --------------------------------------------------------------------------
def fig_edge(R):
    x, E, rows = R["x"], R["edge"][config.MA_DAYS], R["edge_rows"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.7), gridspec_kw={"wspace": 0.25, "width_ratios": [1.7, 1]})
    a = axes[0]
    k = np.arange(len(x))
    a.plot(k, x, color=C["axis"], lw=1.2, label="raw x[k]")
    a.plot(k, E["yl"], color=C["red"], lw=1.1, ls="--", label="linear (zero-padded)")
    a.plot(k, E["yc"], color=C["blue"], lw=1.4, label="circular via DFT")
    a.set_ylim(8, 44)
    a.set_ylabel("°C")
    a.set_title(f"(a) {config.MA_DAYS}-day smoothing")
    month_axis(a)
    a.legend(loc="lower center", fontsize=6.4)
    b = axes[1]
    M = [r["M"] for r in rows]
    b.plot(M, [r["max_err"] for r in rows], "o-", color=C["red"], ms=4)
    for r in rows:
        b.annotate(f"{r['days_affected']} days", (r["M"], r["max_err"]), xytext=(4, -10),
                   textcoords="offset points", fontsize=6.3, color=C["ink2"])
    b.set_xlabel("Kernel length M (days)")
    b.set_ylabel("Largest error (°C)")
    b.set_ylim(0, 13)
    b.set_title("(b) Error at the year ends")
    return fig


# --------------------------------------------------------------------------
# Figure 3: kernels in time and frequency
# --------------------------------------------------------------------------
def fig_kernels(R):
    S = R["smooth"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), gridspec_kw={"wspace": 0.25})
    a = axes[0]
    n = np.arange(-45, 46)
    for (name, d), col in zip(S.items(), KCOL):
        a.plot(n, d["h"][n % len(d["h"])], color=col, lw=1.2, label=name)
    a.axhline(0, color=C["axis"], lw=0.7)
    a.set_xlabel("n (days, circularly centred)")
    a.set_ylabel("h[n]")
    a.set_title("(a) Kernels h[n]")
    a.legend(loc="upper right", fontsize=6.3)
    b = fig.axes[1] if len(fig.axes) > 1 else axes[1]
    k = np.arange(0, 31)
    for i, ((name, d), col) in enumerate(zip(S.items(), KCOL)):
        b.plot(k + (i - 1) * 0.2, d["H"][k], "o-", color=col, ms=2.6, lw=0.9, label=name)
    b.axhline(0, color=C["axis"], lw=0.7)
    b.set_xlabel("Harmonic k (cycles per year)")
    b.set_ylabel("|H[k]|")
    b.set_title("(b) Gain on each annual harmonic")
    b.legend(loc="upper right", fontsize=6.3)
    return fig


# --------------------------------------------------------------------------
# Figure 4: the smoothed climatology and its residual
# --------------------------------------------------------------------------
def fig_climate(R):
    S, x, clim = R["smooth"], R["x"], R["clim"]
    box = S[f"Boxcar {config.MA_DAYS} d"]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.9), sharex=True,
                             gridspec_kw={"hspace": 0.3, "height_ratios": [1.8, 1]})
    a = axes[0]
    k = np.arange(len(x))
    a.plot(k, box["y"], color=C["orange"], lw=1.6, label="normal maximum")
    a.plot(k, R["tmin_s"], color=C["blue"], lw=1.6, label="normal minimum")
    a.fill_between(k, R["tmin_s"], box["y"], color=C["wash"], lw=0)
    for d, col, txt, dy in ((box["hot"], C["orange"], "hottest", 1.2), (box["cold"], C["blue"], "coldest", -3.0)):
        v = box["y"][d] if txt == "hottest" else R["tmin_s"][int(np.argmin(R["tmin_s"]))]
        dd = d if txt == "hottest" else int(np.argmin(R["tmin_s"]))
        a.plot([dd], [v], "o", color=col, ms=4.5, mec="white", mew=0.6)
        a.annotate(txt, (dd, v), xytext=(5, 3 if dy > 0 else -9), textcoords="offset points", fontsize=6.4, color=C["ink2"])
    a.axvline(box["steepest_fall"], color=C["muted"], lw=0.7, ls=":")
    a.text(box["steepest_fall"] + 3, 43, "monsoon\nonset", fontsize=6.3, color=C["muted"], va="top")
    a.set_ylim(0, 45)
    a.set_ylabel("°C")
    a.set_title(f"(a) Daily normals, {config.MA_DAYS}-day circular smoothing")
    top_legend(a, ncol=2)
    b = axes[1]
    noise = clim["sd_tmax"] / np.sqrt(clim["n_tmax"])
    b.fill_between(k, -noise, noise, color=C["blue"], alpha=0.12, lw=0, label="± sampling error σ/√n")
    b.plot(k, x - box["y"], color=C["violet"], lw=0.7, label="residual x − y")
    b.axhline(0, color=C["axis"], lw=0.7)
    b.set_ylim(-2.5, 2.5)
    b.set_ylabel("°C")
    b.set_title("(b) What the smoothing removed")
    top_legend(b, ncol=2)
    month_axis(b)
    return fig


# --------------------------------------------------------------------------
# Figure 5: cyclic prefix and computational cost
# --------------------------------------------------------------------------
def fig_prefix_timing(R):
    Mt, Sc = R["methods"], R["scaling"]
    x = R["x"]
    M = config.MA_DAYS
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), gridspec_kw={"wspace": 0.28, "width_ratios": [1.5, 1]})
    a = axes[0]
    xp = np.r_[x[-(M - 1):], x]
    n = np.arange(-(M - 1), len(x))
    a.axvspan(-(M - 1), -0.5, color=C["yellow"], alpha=0.18, lw=0)
    a.text(-(M - 1) / 2, 0.06, "cyclic\nprefix", transform=a.get_xaxis_transform(), ha="center", fontsize=6.3,
           color=C["ink2"])
    a.plot(n, xp, color=C["axis"], lw=1.2, label="x with prefix (last 30 days copied)")
    a.plot(np.arange(len(x)), Mt["cp"], color=C["blue"], lw=1.5, label="linear conv., prefix dropped")
    a.plot(np.arange(len(x)), Mt["cp_ref"], color=C["orange"], lw=0.8, ls="--", label="circular conv. (DFT)")
    a.set_xlim(-40, 120)
    a.set_ylim(15, 42)
    a.set_xlabel("n (day of year)")
    a.set_ylabel("°C")
    a.set_title("(a) Cyclic prefix")
    a.legend(loc="upper left", fontsize=6.0)
    b = axes[1]
    Ns = [r["N"] for r in Sc]
    b.loglog(Ns, [r["t_matrix"] * 1e3 for r in Sc], "o-", color=C["red"], ms=3.5, label="direct (matrix)")
    b.loglog(Ns, [r["t_fft"] * 1e3 for r in Sc], "s-", color=C["aqua"], ms=3.5, label="via DFT (FFT)")
    b.axvline(365, color=C["muted"], lw=0.7, ls=":")
    b.text(380, 3e-3, "N = 365", fontsize=6.3, color=C["muted"])
    b.set_xlabel("N")
    b.set_ylabel("ms per convolution")
    b.set_title("(b) Time against N")
    b.legend(loc="upper left", fontsize=6.2)
    return fig


def make_all_figures(R):
    return {
        "fig1_raw.png": fig_raw(R),
        "fig2_edge.png": fig_edge(R),
        "fig3_kernels.png": fig_kernels(R),
        "fig4_climate.png": fig_climate(R),
        "fig5_prefix_timing.png": fig_prefix_timing(R),
    }
