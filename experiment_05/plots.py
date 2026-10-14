"""
plots.py
--------
Publication-style figures for Experiment 5.

Every function receives already-computed results (from ``analysis``) and returns
a Matplotlib ``Figure``; saving is done by the caller.
"""

import matplotlib.pyplot as plt
import numpy as np

import config

# --------------------------------------------------------------------------
# Visual style (identical to Experiments 1 to 4)
# --------------------------------------------------------------------------
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#ffffff", "wash": "#f3f2ee",
}
SERIES = [C["blue"], C["orange"], C["aqua"], C["violet"]]
FULL_W = 6.7
DPI = 220


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


def bounds(ax, b, x0, x1):
    ax.fill_between([x0, x1], -b, b, color=C["blue"], alpha=0.10, lw=0, label="95 % white-noise band")


# --------------------------------------------------------------------------
# Figure 1: the record
# --------------------------------------------------------------------------
def fig_raw(R):
    df, sm = R["df"], R["noise"]["smooth"]
    fig, ax = plt.subplots(figsize=(FULL_W, 2.4))
    t = df["decimal_year"]
    ax.plot(t, df["sn"], color=C["axis"], lw=0.5, label="monthly mean")
    ax.plot(t, sm, color=C["blue"], lw=1.3, label=f"{config.SMOOTH_MONTHS}-month running mean")
    ax.set_xlim(t.iloc[0], t.iloc[-1] + 1)
    ax.set_ylim(0, 420)
    ax.set_xlabel("Year")
    ax.set_ylabel("Sunspot number")
    ax.set_title(f"Monthly sunspot number, {R['info']['start'][:4]} to {R['info']['end'][:4]}")
    top_legend(ax, ncol=2)
    return fig


# --------------------------------------------------------------------------
# Figure 2: the ACF
# --------------------------------------------------------------------------
def fig_acf(R):
    A = R["acf"]
    rho = A["rho"]
    lags = np.arange(len(rho))
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.9), gridspec_kw={"hspace": 0.5,
                                                                        "height_ratios": [1.6, 1]})
    a = axes[0]
    bounds(a, A["bound"], 0, len(rho))
    a.plot(lags / 12, rho, color=C["blue"], lw=1.3, label="ρ[l]")
    for p, v in zip(A["peaks"], A["peak_values"]):
        a.plot([p / 12], [v], "o", color=C["orange"], ms=4.5, mec="white", mew=0.6)
        a.text(p / 12, v + 0.07, f"{p / 12:.1f} y\n{v:.2f}", ha="center", fontsize=6.3, color=C["ink2"])
    a.axhline(0, color=C["axis"], lw=0.7)
    a.set_ylim(-0.6, 1.12)
    a.set_xlim(0, config.MAX_LAG / 12)
    a.set_xlabel("Lag (years)")
    a.set_ylabel("ρ[l]")
    a.set_title("(a) Normalised autocorrelation, lags 0 to 50 years")
    top_legend(a, ncol=2)

    b = axes[1]
    L = 48
    bounds(b, A["bound"], -1, L + 1)
    stem(b, lags[: L + 1], rho[: L + 1], C["blue"], markersize=2.6)
    b.axvline(A["zero"], color=C["red"], lw=0.8, ls="--")
    b.text(A["zero"] + 0.6, 0.75, f"first zero at {A['zero']} months", fontsize=6.4, color=C["red"])
    b.set_xlim(-1, L + 1)
    b.set_xticks(range(0, L + 1, 6))
    b.set_xlabel("Lag l (months)")
    b.set_ylabel("ρ[l]")
    b.set_title("(b) First four years in detail")
    return fig


# --------------------------------------------------------------------------
# Figure 3: properties and estimators
# --------------------------------------------------------------------------
def fig_properties(R):
    M, E, x = R["methods"], R["estimators"], R["x"]
    N = len(x)
    full = M["full"] / M["full"][N - 1]
    lags = np.arange(-(N - 1), N)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.5), gridspec_kw={"wspace": 0.22})
    a = axes[0]
    a.plot(lags / 12, full, color=C["blue"], lw=0.9)
    a.axvline(0, color=C["muted"], lw=0.7)
    a.set_xlabel("Lag (years)")
    a.set_ylabel("r[l] / r[0]")
    a.set_title("(a) Full sequence r[l], −(N−1) ≤ l ≤ N−1")
    b = axes[1]
    l = np.arange(E["L"] + 1) / 12
    b.plot(l, E["unbiased"], color=C["orange"], lw=0.7, label="unbiased  r[l]/(N − l)")
    b.plot(l, E["biased"], color=C["blue"], lw=1.2, label="biased  r[l]/N")
    b.set_ylim(-1.6, 1.6)
    b.set_xlabel("Lag (years)")
    b.set_title("(b) Two estimators at large lags")
    b.legend(loc="lower left", fontsize=6.6)
    return fig


# --------------------------------------------------------------------------
# Figure 4: is the leftover white?
# --------------------------------------------------------------------------
def fig_noise(R):
    nz = R["noise"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.4), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, key, col, title in ((axes[0], "residual", C["violet"], "(a) Residual x − 13-month mean"),
                                (axes[1], "difference", C["aqua"], "(b) First difference x[n] − x[n−1]")):
        d = nz[key]
        L = len(d["rho"]) - 1
        bounds(ax, d["bound"], 0.5, L + 0.5)
        stem(ax, np.arange(1, L + 1), d["rho"][1:], col, markersize=2.6)
        ax.set_xlim(0, L + 1)
        ax.set_ylim(-0.45, 0.45)
        ax.set_xlabel("Lag (months)")
        ax.set_title(f"{title}: {d['n_outside']} of {L} outside", fontsize=7.6)
    axes[0].set_ylabel("ρ[l]")
    return fig


# --------------------------------------------------------------------------
# Figure 5: epochs and noise robustness
# --------------------------------------------------------------------------
def fig_windows_noise(R):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), gridspec_kw={"wspace": 0.22})
    a = axes[0]
    for w, col in zip(R["windows"], SERIES):
        rho = w["rho"]
        a.plot(np.arange(len(rho)) / 12, rho, color=col, lw=1.1,
               label=f"{w['span'][0]} to {w['span'][1]}: {w['period_years']:.1f} y")
    a.axhline(0, color=C["axis"], lw=0.7)
    a.set_xlim(0, 25)
    a.set_ylim(-0.8, 1.05)
    a.set_xlabel("Lag (years)")
    a.set_ylabel("ρ[l]")
    a.set_title("(a) ACF of four 70-year epochs")
    a.legend(loc="upper right", fontsize=6.2)
    b = axes[1]
    for r, col in zip(R["robust"], SERIES):
        rho = r["rho"]
        snr = "no noise" if r["level"] == 0 else f"SNR {r['snr_db']:.0f} dB"
        b.plot(np.arange(len(rho)) / 12, rho, color=col, lw=1.0, label=f"{snr}: {r['period_years']:.1f} y")
    b.axhline(0, color=C["axis"], lw=0.7)
    b.set_xlim(0, 25)
    b.set_ylim(-0.8, 1.05)
    b.set_xlabel("Lag (years)")
    b.set_title("(b) Same record buried in white noise")
    b.legend(loc="upper right", fontsize=6.2)
    return fig


def make_all_figures(R):
    return {
        "fig1_raw.png": fig_raw(R),
        "fig2_acf.png": fig_acf(R),
        "fig3_properties.png": fig_properties(R),
        "fig4_noise.png": fig_noise(R),
        "fig5_windows_noise.png": fig_windows_noise(R),
    }
