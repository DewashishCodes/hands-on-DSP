"""
plots.py
--------
Publication-style figures for Experiment 4.

Every function receives already-computed results (from ``analysis``) and returns
a Matplotlib ``Figure``; saving is done by the caller.
"""

import matplotlib.pyplot as plt
import numpy as np

import config

# --------------------------------------------------------------------------
# Visual style (identical to Experiments 1 to 3)
# --------------------------------------------------------------------------
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#ffffff", "wash": "#f3f2ee",
}
FULL_W = 6.7
DPI = 220
DEG = np.arange(config.N_BINS) * config.BIN_DEG


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


def deg_axis(ax, upto=360):
    ax.set_xticks(range(0, upto, 30))
    ax.set_xlim(-8, upto - 2)
    ax.set_xlabel("Wind direction (degrees, from)")


def rose(ax, p, color, title):
    """Polar bar chart (meteorological convention: north up, clockwise)."""
    th = np.deg2rad(DEG)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(th, 100 * p / p.sum(), width=np.deg2rad(config.BIN_DEG) * 0.88, color=color,
           edgecolor="white", linewidth=0.4)
    ax.set_title(title, fontsize=8, pad=10, loc="center")
    ax.tick_params(labelsize=6.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g} %"))
    ax.tick_params(axis="y", labelsize=5.8, labelcolor=C["muted"])
    ax.grid(color=C["grid"], linewidth=0.5)


# --------------------------------------------------------------------------
# Figure 1: raw data
# --------------------------------------------------------------------------
def fig_raw(R):
    df, p = R["df"], R["p"]
    fig = plt.figure(figsize=(FULL_W, 2.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.25)
    a = fig.add_subplot(gs[0])
    m = df.groupby("month")["speed_kt"].mean()
    a.bar(m.index, m.values, color=C["blue"], width=0.75)
    a.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    a.set_ylabel("Mean speed (kt)")
    a.set_title("(a) Monthly mean wind speed")
    a.grid(axis="x", visible=False)
    b = fig.add_subplot(gs[1])
    b.bar(DEG, p, width=8, color=C["blue"])
    b.set_ylabel("Reports")
    b.set_title("(b) Direction histogram p[k], 36 bins")
    deg_axis(b)
    b.grid(axis="x", visible=False)
    return fig


# --------------------------------------------------------------------------
# Figure 2: seasonal wind roses
# --------------------------------------------------------------------------
def fig_roses(R):
    S = R["seasons"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.6), subplot_kw={"projection": "polar"},
                             gridspec_kw={"wspace": 0.45})
    for ax, (name, col) in zip(axes, (("Annual", C["blue"]), ("Winter (Dec to Feb)", C["aqua"]),
                                      ("Monsoon (Jun to Sep)", C["orange"]))):
        rose(ax, S[name]["p"], col, f"{name}\nmode {S[name]['mode_deg']}°")
    return fig


# --------------------------------------------------------------------------
# Figure 3: circular versus linear smoothing
# --------------------------------------------------------------------------
def fig_smoothing(R):
    s = R["smoothing"]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.32})
    a = axes[0]
    a.bar(DEG, s["p"], width=8, color=C["axis"], label="p[k]")
    a.plot(DEG, s["circ"], "o-", color=C["blue"], ms=3, label="circular  p ⊛ h")
    a.plot(DEG, s["lin"], "s--", color=C["orange"], ms=2.6, lw=1.1, label="linear  p * h (zero-padded)")
    for k in s["seam_bins"]:
        a.annotate("", xy=(DEG[k], s["lin"][k]), xytext=(DEG[k], s["circ"][k]),
                   arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.0))
    a.set_ylabel("Reports")
    a.set_title("(a) Smoothing with h = [¼, ½, ¼]")
    top_legend(a, ncol=3)
    b = axes[1]
    stem(b, DEG, s["diff"], C["red"], markersize=3)
    b.set_ylabel("Difference")
    b.set_title("(b) Circular minus linear")
    deg_axis(b)
    return fig


# --------------------------------------------------------------------------
# Figure 4: relation between linear and circular convolution
# --------------------------------------------------------------------------
def fig_lin_circ(R):
    lc = R["lin_circ"]
    N = config.N_BINS
    fig, axes = plt.subplots(3, 1, figsize=(FULL_W, 3.8), sharex=True, gridspec_kw={"hspace": 0.5})
    n38 = np.arange(len(lc["lin"]))
    a = axes[0]
    stem(a, n38, lc["lin"], C["ink2"], markersize=2.4)
    a.axvspan(N - 0.5, len(lc["lin"]) - 0.5, color=C["red"], alpha=0.12, lw=0)
    a.text(N + 0.5, 0.82, "n = 36, 37\nwrap to 0, 1", transform=a.get_xaxis_transform(), fontsize=6.3,
           color=C["red"], va="top")
    a.set_title(f"(a) Linear convolution, length 36 + 3 − 1 = {len(lc['lin'])}")
    b = axes[1]
    stem(b, np.arange(N), lc["circ36"], C["blue"], markersize=2.4)
    b.set_title("(b) 36-point circular convolution = (a) wrapped modulo 36")
    c = axes[2]
    stem(c, n38, lc["circ38"], C["aqua"], markersize=2.4)
    c.set_title("(c) 38-point circular convolution of the zero-padded inputs = (a)")
    c.set_xlabel("Index n (bin)")
    c.set_xticks(range(0, 39, 3))
    for ax in axes:
        ax.set_ylabel("Reports")
        ax.set_xlim(-1, 38.5)
    return fig


# --------------------------------------------------------------------------
# Figure 5: runway wind coverage
# --------------------------------------------------------------------------
def fig_runway(R):
    rw = R["runway"]
    fig, ax = plt.subplots(figsize=(FULL_W, 2.6))
    h = DEG[:18]
    for lim, col in ((10.0, C["blue"]), (13.0, C["aqua"])):
        ax.plot(h, 100 * rw[lim]["coverage"][:18], "o-", color=col, ms=3, label=f"crosswind limit {lim:g} kt")
    ax.axhline(100 * config.ICAO_USABILITY, color=C["red"], ls="--", lw=0.9, label="ICAO minimum 95 %")
    for name, hd in config.RUNWAYS.items():
        v = 100 * rw[10.0]["runways"][name]
        ax.axvline(hd, color=C["muted"], lw=0.7, ls=":")
        ax.text(hd + 1.5, 93.2, f"runway {name}\n{v:.2f} %", fontsize=6.4, color=C["ink2"])
    b = rw[10.0]["best_deg"]
    ax.plot([b], [100 * rw[10.0]["best"]], "*", color=C["orange"], ms=9, mec="white", mew=0.5,
            label=f"best heading {b:03d}°")
    ax.set_xticks(range(0, 180, 10), [f"{v:03d}" for v in range(0, 180, 10)], rotation=0, fontsize=6.4)
    ax.set_xlim(-3, 173)
    ax.set_ylim(92.8, 100.4)
    ax.set_xlabel("Runway heading (degrees; the reciprocal end is +180°)")
    ax.set_ylabel("Usability (%)")
    ax.set_title("Runway usability")
    top_legend(ax, ncol=2)
    return fig


def make_all_figures(R):
    return {
        "fig1_raw.png": fig_raw(R),
        "fig2_roses.png": fig_roses(R),
        "fig3_smoothing.png": fig_smoothing(R),
        "fig4_lin_circ.png": fig_lin_circ(R),
        "fig5_runway.png": fig_runway(R),
    }
