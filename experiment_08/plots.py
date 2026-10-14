"""
plots.py
--------
Publication-style figures for Experiment 8.

Every function receives already-computed results (from ``analysis``) and returns
a Matplotlib ``Figure``; saving is done by the caller.
"""

import matplotlib.pyplot as plt
import numpy as np

import config

# --------------------------------------------------------------------------
# Visual style (identical to Experiments 1 to 7)
# --------------------------------------------------------------------------
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#ffffff", "wash": "#f3f2ee",
}
FULL_W = 6.7
DPI = 220
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


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


# --------------------------------------------------------------------------
# Figure 1: the Keeling curve, its trend and the detrended residual
# --------------------------------------------------------------------------
def fig_raw(R):
    df, w, r, trend = R["df"], R["w"], R["r"], R["trend"]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.5), sharex=True,
                             gridspec_kw={"hspace": 0.32, "height_ratios": [1.5, 1]})
    a = axes[0]
    a.plot(df["t"], df["co2"], color=C["blue"], lw=0.8, label="monthly mean x[n]")
    a.plot(w["t"], trend, color=C["orange"], lw=1.2, label=f"order-{config.TREND_ORDER} trend")
    a.set_ylabel("CO₂ (ppm)")
    a.set_title("(a) Mauna Loa CO₂")
    top_legend(a, ncol=2)
    b = axes[1]
    b.plot(w["t"], r, color=C["violet"], lw=0.7)
    b.axhline(0, color=C["axis"], lw=0.7)
    b.set_ylabel("ppm")
    b.set_xlabel("Year")
    b.set_title("(b) Detrended signal r[n] = x[n] − trend, the input of the DFT")
    return fig


# --------------------------------------------------------------------------
# Figure 2: the amplitude spectrum
# --------------------------------------------------------------------------
def fig_spectrum(R):
    s = R["spec"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), gridspec_kw={"wspace": 0.25, "width_ratios": [1.6, 1]})
    a = axes[0]
    a.plot(s["f"], s["A"], color=C["blue"], lw=0.8)
    for h in (1, 2, 3, 4):
        A, _ = s["lines"][h]
        a.plot([h], [A], "o", color=C["orange"], ms=4, mec="white", mew=0.5)
        a.annotate(f"{h} cy/yr\n{A:.2f} ppm", (h, A), xytext=(4, 2), textcoords="offset points",
                   fontsize=6.3, color=C["ink2"])
    a.set_xlim(0, 6)
    a.set_ylim(0, 3.3)
    a.set_xlabel("Frequency (cycles per year)")
    a.set_ylabel("Amplitude (ppm)")
    a.set_title(f"(a) Amplitude spectrum 2|X[k]|/N, N = {len(R['r'])}")
    b = axes[1]
    k1 = s["k1"]
    k = np.arange(k1 - 8, k1 + 9)
    stem(b, k, s["A"][k], C["blue"], markersize=3)
    b.set_xlabel("Bin k")
    b.set_title(f"(b) Bins around k = {k1} (1 cy/yr)")
    return fig


# --------------------------------------------------------------------------
# Figure 3: leakage, windowing and zero padding
# --------------------------------------------------------------------------
def fig_leakage(R):
    L = R["leak"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), gridspec_kw={"wspace": 0.25})
    a = axes[0]
    a.plot(L["f_exact"], L["A_exact"], "o-", color=C["blue"], ms=2.5, lw=0.8, label=f"N = {len(R['r'])} (whole years)")
    a.plot(L["f_cut"], L["A_cut"], "s-", color=C["red"], ms=2.5, lw=0.8, label=f"N = {L['Nc']} (rectangular)")
    a.plot(L["f_cut"], L["A_hann"], "^-", color=C["aqua"], ms=2.5, lw=0.8, label=f"N = {L['Nc']}, Hann window")
    a.set_xlim(0.85, 1.15)
    a.set_ylim(0, 3.2)
    a.set_xlabel("Frequency (cycles per year)")
    a.set_ylabel("Amplitude (ppm)")
    a.set_title("(a) Leakage and windowing")
    a.legend(loc="upper left", fontsize=6.2)
    b = axes[1]
    b.plot(L["f_pad"], L["A_pad"], color=C["violet"], lw=1.0, label=f"Hann, zero-padded ×{config.ZERO_PAD}")
    b.plot(L["f_cut"], L["A_hann"], "^", color=C["aqua"], ms=3.5, label="Hann, N bins")
    b.set_xlim(0.85, 1.15)
    b.set_ylim(0, 3.2)
    b.set_xlabel("Frequency (cycles per year)")
    b.set_title("(b) Zero padding interpolates the spectrum")
    b.legend(loc="upper left", fontsize=6.2)
    return fig


# --------------------------------------------------------------------------
# Figure 4: IDFT synthesis of the seasonal cycle
# --------------------------------------------------------------------------
def fig_idft(R):
    w, r, S = R["w"], R["r"], R["season"]
    fig = plt.figure(figsize=(FULL_W, 4.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], hspace=0.5, wspace=0.25)
    a = fig.add_subplot(gs[0, 0])
    sel = slice(len(r) - 48, len(r))
    a.plot(w["t"].iloc[sel], r[sel], color=C["axis"], lw=1.4, label="r[n]")
    a.plot(w["t"].iloc[sel], S["seasonal"][sel], color=C["orange"], lw=1.3, label="IDFT of bins 1 and 2 cy/yr")
    a.set_ylabel("ppm")
    a.set_title("(a) Last 4 years")
    top_legend(a, ncol=2)
    b = fig.add_subplot(gs[0, 1])
    b.bar(range(12), S["clim"], color=[C["blue"] if v >= 0 else C["orange"] for v in S["clim"]], width=0.75)
    b.set_xticks(range(12), MONTHS)
    b.axhline(0, color=C["axis"], lw=0.7)
    b.set_title("(b) One year of the model")
    b.grid(axis="x", visible=False)
    c = fig.add_subplot(gs[1, :])
    c.plot(w["t"], S["noaa"], color=C["axis"], lw=2.2, label="NOAA deseasonalized")
    c.plot(w["t"], S["deseason"], color=C["blue"], lw=0.8, label="x[n] − IDFT seasonal model")
    c.set_ylabel("CO₂ (ppm)")
    c.set_xlabel("Year")
    c.set_title("(c) Deseasonalized series")
    top_legend(c, ncol=2)
    return fig


# --------------------------------------------------------------------------
# Figure 5: change over the decades, and the FFT speed-up
# --------------------------------------------------------------------------
def fig_decades_timing(R):
    Dd, G, T = R["decades"], R["growth"], R["timing"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.4), gridspec_kw={"wspace": 0.42})
    a = axes[0]
    x = np.arange(len(Dd))
    a.plot(x, [d["p2p"] for d in Dd], "o-", color=C["blue"], ms=4)
    a.set_xticks(x, [f"{d['start']}" for d in Dd], rotation=45, fontsize=6.3)
    a.set_xlabel("Decade starting")
    a.set_ylim(5.5, 7)
    a.set_ylabel("ppm")
    a.set_title("(a) Seasonal swing (peak to trough)", fontsize=7.4)
    b = axes[1]
    b.plot(R["w"]["t"], G["rate"], color=C["orange"], lw=1.2)
    b.set_ylabel("ppm per year")
    b.set_title("(b) Growth rate of trend", fontsize=7.8)
    c = axes[2]
    N = [t["N"] for t in T]
    c.loglog(N, [t["matrix_s"] * 1e3 for t in T], "o-", color=C["red"], ms=3, label="DFT matrix")
    c.loglog(N, [t["fft_s"] * 1e3 for t in T], "s-", color=C["aqua"], ms=3, label="FFT")
    c.set_xlabel("N")
    c.set_ylabel("ms")
    c.set_title("(c) Computation time", fontsize=7.8)
    c.legend(loc="upper left", fontsize=6.2)
    return fig


def make_all_figures(R):
    return {
        "fig1_raw.png": fig_raw(R),
        "fig2_spectrum.png": fig_spectrum(R),
        "fig3_leakage.png": fig_leakage(R),
        "fig4_idft.png": fig_idft(R),
        "fig5_decades_timing.png": fig_decades_timing(R),
    }
