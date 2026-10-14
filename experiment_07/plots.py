"""
plots.py
--------
Publication-style figures for Experiment 7.

Every function receives already-computed results (from ``analysis``) and returns
a Matplotlib ``Figure``; saving is done by the caller.
"""

import matplotlib.pyplot as plt
import numpy as np

import config
import smoothing as sm

# --------------------------------------------------------------------------
# Visual style (identical to Experiments 1 to 6)
# --------------------------------------------------------------------------
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#ffffff", "wash": "#f3f2ee",
}
FULL_W = 6.7
DPI = 220
MA = f"MA{config.N_MA} (non-recursive)"
EMA = f"EMA{config.N_EMA}"
EMA2 = f"Double EMA{config.N_EMA}"
HOLT = "Holt (level + trend)"
FCOL = {MA: C["blue"], EMA: C["orange"], EMA2: C["aqua"], HOLT: C["violet"]}


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
# Figure 1: the raw series
# --------------------------------------------------------------------------
def fig_raw(R):
    df = R["df"]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.4), sharex=True,
                             gridspec_kw={"hspace": 0.35, "height_ratios": [1.6, 1]})
    a = axes[0]
    a.plot(df["date"], df["inr"], color=C["blue"], lw=0.9)
    i0, i1 = R["zoom"]
    a.axvspan(df["date"].iloc[i0], df["date"].iloc[i1], color=C["wash"], lw=0)
    a.text(df["date"].iloc[(i0 + i1) // 2], 0.92, "2022 (zoom)", transform=a.get_xaxis_transform(),
           ha="center", fontsize=6.6, color=C["ink2"])
    a.set_ylabel("₹ per US$")
    a.set_title("(a) Exchange rate x[n]")
    b = axes[1]
    b.plot(df["date"], df["change_pct"], color=C["ink2"], lw=0.5)
    b.set_ylabel("%")
    b.set_title("(b) Daily change 100·(x[n] − x[n−1]) / x[n−1]")
    return fig


# --------------------------------------------------------------------------
# Figure 2: the smoothers on the 2022 fall of the rupee
# --------------------------------------------------------------------------
def fig_smoothers(R):
    df, x, F = R["df"], R["x"], R["filters"]
    i0, i1 = R["zoom"]
    sel = slice(i0, i1 + 1)
    d = df["date"].iloc[sel]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.9), sharex=True,
                             gridspec_kw={"hspace": 0.32, "height_ratios": [1.6, 1]})
    a = axes[0]
    a.plot(d, x[sel], color=C["axis"], lw=1.6, label="x[n]")
    for name in (MA, EMA, EMA2, HOLT):
        a.plot(d, F[name]["y"][sel], color=FCOL[name], lw=1.2, label=name.split(" (")[0])
    a.set_ylabel("₹ per US$")
    a.set_title("(a) Smoothed outputs, 2022")
    top_legend(a, ncol=5)
    b = axes[1]
    for name in (MA, EMA, EMA2, HOLT):
        b.plot(d, x[sel] - F[name]["y"][sel], color=FCOL[name], lw=1.0)
    b.axhline(0, color=C["axis"], lw=0.7)
    b.set_ylabel("x − y (₹)")
    b.set_title("(b) Tracking error: positive = smoother lags behind a rising rate")
    return fig


# --------------------------------------------------------------------------
# Figure 3: impulse responses and the z-plane
# --------------------------------------------------------------------------
def fig_responses(R):
    F = R["filters"]
    L = 45
    fig = plt.figure(figsize=(FULL_W, 2.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.28)
    a = fig.add_subplot(gs[0])
    for i, name in enumerate((MA, EMA, EMA2, HOLT)):
        h = sm.impulse_response(F[name]["b"], F[name]["a"], L)
        a.plot(np.arange(L), h, "o-", color=FCOL[name], ms=2.2, lw=1.0, label=name.split(" (")[0])
    a.axhline(0, color=C["axis"], lw=0.7)
    a.set_xlabel("n (trading days)")
    a.set_ylabel("h[n]")
    a.set_title("(a) Impulse responses")
    a.legend(loc="upper right", fontsize=6.4)
    b = fig.add_subplot(gs[1])
    t = np.linspace(0, 2 * np.pi, 400)
    b.plot(np.cos(t), np.sin(t), color=C["axis"], lw=0.8)
    for name, mk in ((MA, "o"), (EMA, "x"), (HOLT, "x")):
        p, z = F[name]["poles"], F[name]["zeros"]
        if name == MA:
            b.plot(z.real, z.imag, "o", mfc="none", mec=FCOL[name], ms=4, label="MA20 zeros")
        else:
            b.plot(p.real, p.imag, "x", color=FCOL[name], ms=6, mew=1.4, label=f"{name.split(' (')[0]} poles")
            b.plot(z.real, z.imag, "o", mfc="none", mec=FCOL[name], ms=5)
    b.set_aspect("equal")
    b.set_xlim(-1.25, 1.25)
    b.set_ylim(-1.25, 1.25)
    b.set_title("(b) z-plane")
    b.legend(loc="lower left", fontsize=5.8)
    return fig


# --------------------------------------------------------------------------
# Figure 4: initial conditions
# --------------------------------------------------------------------------
def fig_initial(R):
    ic = R["ic"]
    n = np.arange(len(ic["x"]))
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.5), gridspec_kw={"wspace": 0.25})
    a = axes[0]
    a.plot(n, ic["x"], color=C["axis"], lw=1.6, label="x[n]")
    a.plot(n, ic["y_rest"], color=C["red"], lw=1.2, label="y[−1] = 0 (at rest)")
    a.plot(n, ic["y_full"], color=C["blue"], lw=1.2, label="y[−1] = x[0]")
    a.set_xlabel("n (trading days from 2 Jan 2019)")
    a.set_ylabel("₹ per US$")
    a.set_title("(a) EMA20 with two initial conditions")
    a.legend(loc="lower right", fontsize=6.4)
    b = axes[1]
    b.semilogy(n, ic["y_zi"], color=C["violet"], lw=1.3, label="zero-input response y_zi[n]")
    b.axhline(config.TRANSIENT_TOL * ic["x"][0], color=C["muted"], lw=0.8, ls="--", label="1 % of x[0]")
    b.axvline(ic["n_theory"], color=C["red"], lw=0.8, ls=":")
    b.text(ic["n_theory"] + 2, 1e-3, f"1 % reached at\nn = {ic['n_theory']}", fontsize=6.4, color=C["red"])
    b.set_xlabel("n")
    b.set_ylabel("₹ (log scale)")
    b.set_title("(b) Decay of the initial condition")
    b.legend(loc="upper right", fontsize=6.4)
    return fig


# --------------------------------------------------------------------------
# Figure 5: trade-off and frequency response
# --------------------------------------------------------------------------
def fig_tradeoff(R):
    T, F, M = R["tradeoff"], R["filters"], R["macd"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), gridspec_kw={"wspace": 0.28})
    a = axes[0]
    Ls = np.linspace(0.5, 21, 200)
    a.plot(Ls, 1 / (2 * Ls + 1), color=C["muted"], lw=0.9, ls="--", label="theory 1/(2L+1)")
    a.plot(T["ma_lag"], T["ma_ng"], "o", color=C["blue"], ms=4.5, label="moving average")
    a.plot(T["ema_lag"], T["ema_ng"], "x", color=C["orange"], ms=5.5, mew=1.3, label="EMA")
    a.plot(T["ema2_lag"], T["ema2_ng"], "+", color=C["aqua"], ms=6, mew=1.3, label="double EMA")
    a.set_yscale("log")
    a.set_xlabel("Lag L (days)")
    a.set_ylabel("Noise gain Σh²")
    a.set_title("(a) Smoothing versus lag")
    a.legend(loc="upper right", fontsize=6.3)
    b = axes[1]
    w = np.linspace(1e-3, np.pi, 1500)
    for name in (MA, EMA, HOLT):
        H = sm.frequency_response(F[name]["b"], F[name]["a"], w)
        b.plot(w / (2 * np.pi), 20 * np.log10(np.abs(H) + 1e-9), color=FCOL[name], lw=1.1, label=name.split(" (")[0])
    b.plot(M["w"] / (2 * np.pi), 20 * np.log10(np.abs(M["H"]) + 1e-9), color=C["ink2"], lw=1.1, ls="--", label="MACD 12/26")
    b.set_ylim(-50, 5)
    b.set_xlim(0, 0.5)
    b.set_xlabel("Frequency (cycles per day)")
    b.set_ylabel("|H| (dB)")
    b.set_title("(b) Frequency responses")
    b.legend(loc="upper right", fontsize=6.3)
    return fig


# --------------------------------------------------------------------------
# Figure 6: MACD application
# --------------------------------------------------------------------------
def fig_macd(R):
    df, M = R["df"], R["macd"]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.3), sharex=True, gridspec_kw={"hspace": 0.3})
    a = axes[0]
    a.plot(df["date"], df["inr"], color=C["axis"], lw=1.2, label="x[n]")
    a.plot(df["date"], M["ef"], color=C["orange"], lw=0.9, label=f"EMA{config.MACD_FAST}")
    a.plot(df["date"], M["es"], color=C["blue"], lw=0.9, label=f"EMA{config.MACD_SLOW}")
    a.set_ylabel("₹ per US$")
    a.set_title("(a) Fast and slow EMAs")
    top_legend(a, ncol=3)
    b = axes[1]
    b.plot(df["date"], M["line"], color=C["violet"], lw=0.9, label="MACD = EMA12 − EMA26")
    b.plot(df["date"], M["sig"], color=C["orange"], lw=0.9, label="signal = EMA9 of MACD")
    b.axhline(0, color=C["axis"], lw=0.7)
    b.set_ylabel("₹")
    b.set_title(f"(b) MACD: {M['n_cross']} crossovers ({M['per_year']:.0f} per year)")
    top_legend(b, ncol=2)
    return fig


def make_all_figures(R):
    return {
        "fig1_raw.png": fig_raw(R),
        "fig2_smoothers.png": fig_smoothers(R),
        "fig3_responses.png": fig_responses(R),
        "fig4_initial.png": fig_initial(R),
        "fig5_tradeoff.png": fig_tradeoff(R),
        "fig6_macd.png": fig_macd(R),
    }
