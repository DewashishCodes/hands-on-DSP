"""
plots.py
--------
Publication-style figures for Experiment 2.

Every function receives already-computed results (from ``analysis``) and returns
a Matplotlib ``Figure``; saving is done by the caller.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

# --------------------------------------------------------------------------
# Visual style (identical to Experiment 1)
# --------------------------------------------------------------------------
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#ffffff", "wash": "#f3f2ee",
}
FULL_W = 6.7
DPI = 220
HOURS = "Sample index n  (1 sample = 1 hour)"


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


def day_guides(ax, df, n_lo=None, n_hi=None, every=1):
    n_lo = df["n"].min() if n_lo is None else n_lo
    n_hi = df["n"].max() if n_hi is None else n_hi
    for i, n0 in enumerate(range(0, len(df), 24)):
        if n_lo <= n0 <= n_hi:
            ax.axvline(n0, color=C["axis"], linewidth=0.6, linestyle=(0, (3, 3)), zorder=0)
            if i % every == 0:
                ax.text(n0 + 0.6, 0.985, df["time"].iloc[n0].strftime("%d %b"),
                        transform=ax.get_xaxis_transform(), fontsize=6.2, color=C["muted"],
                        va="top", ha="left")


def top_legend(ax, ncol=3, **kw):
    return ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=ncol, fontsize=6.8,
                     handlelength=1.5, columnspacing=1.0, borderaxespad=0.1, **kw)


def festival_marker(ax, df):
    n0 = int(df.index[df["time"] == pd.Timestamp(config.FESTIVAL_DATE)][0]) + 18   # evening of Diwali
    ax.axvline(n0, color=C["yellow"], lw=1.2, zorder=0)
    ax.text(n0 + 1.5, 0.80, "Diwali\n(20 Oct)", transform=ax.get_xaxis_transform(), ha="left",
            fontsize=6.3, color=C["ink2"], linespacing=1.1)


# --------------------------------------------------------------------------
# Figure 1: raw signals
# --------------------------------------------------------------------------
def fig_raw_signals(df):
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.2), sharex=True, gridspec_kw={"hspace": 0.38})
    n = df["n"]
    a = axes[0]
    a.plot(n, df["pm25"], color=C["blue"], label="x₁[n]  PM2.5")
    a.axhline(config.NAAQS_PM25_24H, color=C["red"], lw=0.8, ls="--", label="NAAQS 24-h limit (60)")
    a.set_ylabel("µg/m³")
    a.set_ylim(0, 95)
    a.set_title("(a) PM2.5")
    top_legend(a, ncol=2)
    b = axes[1]
    b.plot(n, df["no2"], color=C["orange"], label="x₂[n]  NO₂")
    b.set_ylabel("µg/m³")
    b.set_ylim(0, 52)
    b.set_title("(b) Nitrogen dioxide")
    b.set_xlabel(HOURS)
    top_legend(b, ncol=1)
    for ax in axes:
        day_guides(ax, df, every=2)
        festival_marker(ax, df)
        ax.set_xlim(-3, len(df) + 2)
    return fig


# --------------------------------------------------------------------------
# Figure 2: the two failing tests, shown as overlays
# --------------------------------------------------------------------------
def fig_failures(R, df, n_lo=96, n_hi=191):
    B = R["battery"]
    lin, ti = B["S3"]["linear"], B["S4"]["ti"]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.6), gridspec_kw={"hspace": 0.5})
    n = np.arange(len(lin["lhs"]))
    sel = (n >= n_lo) & (n <= n_hi)
    a = axes[0]
    a.plot(n[sel], lin["lhs"][sel], color=C["blue"], lw=1.8, label="T{a x₁ + b x₂}")
    a.plot(n[sel], lin["rhs"][sel], color=C["orange"], ls="--", label="a T{x₁} + b T{x₂}")
    a.set_ylabel("AQI")
    a.set_title(f"(a) S3 AQI, superposition  (ε = {lin['error']:.3f})")
    top_legend(a, ncol=2)

    k = ti["k"]
    m = np.arange(len(ti["lhs"]))
    sel2 = (m >= n_lo) & (m <= n_hi)
    b = axes[1]
    b.plot(m[sel2], ti["lhs"][sel2], color=C["blue"], lw=1.8, label=f"T{{x[n − {k}]}}")
    b.plot(m[sel2], ti["rhs"][sel2], color=C["orange"], ls="--", label=f"y[n − {k}]")
    b.fill_between(m[sel2], ti["lhs"][sel2], ti["rhs"][sel2], color=C["red"], alpha=0.25, lw=0)
    b.set_ylabel("µg/m³")
    b.set_title(f"(b) S4 drifting sensor, shift test  (ε = {ti['error']:.4f})")
    b.set_xlabel(HOURS)
    top_legend(b, ncol=2)
    for ax in axes:
        day_guides(ax, df, n_lo, n_hi)
        ax.set_xlim(n_lo - 1, n_hi + 1)
    return fig


# --------------------------------------------------------------------------
# Figure 3: error signals of both tests for all four systems
# --------------------------------------------------------------------------
def fig_error_grid(R, df):
    B = R["battery"]
    fig, axes = plt.subplots(4, 2, figsize=(FULL_W, 4.6), sharex=True, sharey="col",
                             gridspec_kw={"hspace": 0.55, "wspace": 0.16})
    for row, key in enumerate(("S1", "S2", "S3", "S4")):
        res = B[key]
        for col, test in enumerate(("linear", "ti")):
            ax = axes[row, col]
            e = res[test]["lhs"] - res[test]["rhs"]
            holds = res[test]["holds"]
            ax.plot(np.arange(len(e)), e, color=C["aqua"] if holds else C["red"], lw=1.0)
            verdict = "holds" if holds else "FAILS"
            name = "superposition" if test == "linear" else "shift"
            ax.set_title(f"{key} {res['system'].name}: {name} {verdict} (ε = {res[test]['error']:.1e})",
                         fontsize=7.2)
            ax.set_xlim(-3, len(e) + 2)
    axes[0, 0].text(0, 1.55, "Linearity error  e_L[n]", transform=axes[0, 0].transAxes,
                    fontsize=8.2, weight="semibold")
    axes[0, 1].text(0, 1.55, "Time-invariance error  e_T[n]", transform=axes[0, 1].transAxes,
                    fontsize=8.2, weight="semibold")
    for ax in axes[-1]:
        ax.set_xlabel("n")
    return fig


# --------------------------------------------------------------------------
# Figure 4: impulse and step responses
# --------------------------------------------------------------------------
def fig_responses(R):
    rs = R["responses"]
    L = config.RESPONSE_LENGTH
    n = np.arange(L)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4), gridspec_kw={"wspace": 0.22})
    a = axes[0]
    stem(a, n - 0.15, rs["S1"]["h"], C["blue"], f"S1, M = {config.MA_LENGTH_M}", markersize=2.2)
    stem(a, n + 0.15, rs["S2"]["h"], C["orange"], f"S2, α = {config.EWMA_ALPHA:g}", markersize=2.2)
    a.plot(n, rs["S2"]["h_theory"], color=C["ink2"], lw=0.8, ls=":", label="α(1−α)ⁿ")
    a.set_title("(a) Impulse response h[n]")
    a.set_ylim(0, 0.24)
    a.set_xlabel("n")
    a.legend(loc="upper right", fontsize=6.6)

    b = axes[1]
    b.plot(n, rs["S1"]["s"], color=C["blue"], label="S1")
    b.plot(n, rs["S2"]["s"], color=C["orange"], label="S2")
    for key, col in (("S1", C["blue"]), ("S2", C["orange"])):
        t = rs[key]["t63"]
        b.plot([t], [rs[key]["s"][t]], "o", color=col, ms=4, mec="white", mew=0.6)
    b.axhline(1 - np.exp(-1), color=C["muted"], lw=0.7, ls="--")
    b.text(L - 1, 1 - np.exp(-1) + 0.02, "63.2 %", ha="right", fontsize=6.5, color=C["muted"])
    b.set_ylim(0, 1.1)
    b.set_title("(b) Step response s[n]")
    b.set_xlabel("n")
    b.legend(loc="lower right", fontsize=6.6)
    return fig


# --------------------------------------------------------------------------
# Figure 5: the LTI systems applied to the real data
# --------------------------------------------------------------------------
def fig_application(R, df, n_lo=72, n_hi=215):
    A = R["application"]
    n = df["n"].to_numpy()
    sel = (n >= n_lo) & (n <= n_hi)
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.9), sharex=True, gridspec_kw={"hspace": 0.4})
    a = axes[0]
    a.plot(n[sel], A["x"][sel], color=C["ink2"], lw=0.9, label="Hourly PM2.5")
    a.plot(n[sel], A["smooth"][sel], color=C["orange"], lw=1.4, label="S2 smoother")
    a.plot(n[sel], A["ma"][sel], color=C["blue"], lw=2.0, label="S1 24-h average")
    a.axhline(config.NAAQS_PM25_24H, color=C["red"], lw=0.8, ls="--")
    ex = A["exceed"] & sel
    a.fill_between(n[sel], 0, 1, where=ex[sel], transform=a.get_xaxis_transform(),
                   color=C["red"], alpha=0.10, lw=0)
    a.set_ylim(0, 95)
    a.set_ylabel("µg/m³")
    a.set_title("(a) LTI outputs and the 24-h standard")
    top_legend(a, ncol=3)

    b = axes[1]
    bands = [(0, 50, "#e8f6ef"), (50, 100, "#f3f7e4"), (100, 200, "#fdf3dc"), (200, 300, "#fbe5d6")]
    for lo, hi, col in bands:
        b.axhspan(lo, hi, color=col, lw=0, zorder=0)
    for (lo, hi, _), name in zip(bands, ("Good", "Satisfactory", "Moderate", "Poor")):
        b.text(n_hi - 1, (lo + hi) / 2, name, ha="right", va="center", fontsize=6.4, color=C["ink2"])
    b.plot(n[sel], A["aqi"][sel], color=C["violet"], lw=1.4)
    b.set_ylim(0, 230)
    b.set_ylabel("AQI")
    b.set_title("(b) S3 output: hourly PM2.5 sub-index with CPCB categories")
    b.set_xlabel(HOURS)
    for ax in axes:
        day_guides(ax, df, n_lo, n_hi)
        ax.set_xlim(n_lo - 1, n_hi + 1)
    return fig


def make_all_figures(df, results):
    return {
        "fig1_raw_signals.png": fig_raw_signals(df),
        "fig2_failures.png": fig_failures(results, df),
        "fig3_error_grid.png": fig_error_grid(results, df),
        "fig4_responses.png": fig_responses(results),
        "fig5_application.png": fig_application(results, df),
    }
