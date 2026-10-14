"""
plots.py
--------
Publication-style figures for Experiment 6.

Every function receives already-computed results (from ``analysis``) and returns
a Matplotlib ``Figure``; saving is done by the caller.
"""

import matplotlib.pyplot as plt
import numpy as np

import config

# --------------------------------------------------------------------------
# Visual style (identical to Experiments 1 to 5)
# --------------------------------------------------------------------------
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#ffffff", "wash": "#f3f2ee",
}
IDS = list(config.STATIONS)
STATION_COLOURS = dict(zip(IDS, [C["ink2"], C["blue"], C["orange"], C["aqua"], C["violet"]]))
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


def top_legend(ax, ncol=3, **kw):
    return ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=ncol, fontsize=6.8,
                     handlelength=1.5, columnspacing=1.0, borderaxespad=0.1, **kw)


def short(sid):
    return config.STATIONS[sid][0].replace(" Island", "")


# --------------------------------------------------------------------------
# Figure 1: the five water-level records
# --------------------------------------------------------------------------
def fig_raw(R, n0=240, hours=72):
    df = R["df"]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.9), gridspec_kw={"hspace": 0.5})
    a = axes[0]
    sel = slice(n0, n0 + hours)
    t = df["n"].to_numpy()[sel] - n0
    for s in IDS:
        a.plot(t, df[s].to_numpy()[sel], color=STATION_COLOURS[s], lw=1.2, label=short(s))
    a.set_xlim(0, hours)
    a.set_xticks(range(0, hours + 1, 6))
    a.set_xlabel(f"Hours from {df['time_utc'].iloc[n0]:%d %b %Y %H:%M} UTC")
    a.set_ylabel("Water level (m, MSL)")
    a.set_title("(a) Three days")
    top_legend(a, ncol=5)

    b = axes[1]
    st = R["subtidal"]
    h = config.SUBTIDAL_WINDOW_H // 2
    tt = df["time_utc"]
    for s, lw in ((IDS[0], 0.5), (IDS[-1], 0.5)):
        b.plot(tt, df[s], color=STATION_COLOURS[s], lw=lw, alpha=0.6)
    b.plot(tt.iloc[h:-h], st["x"], color=STATION_COLOURS[IDS[0]], lw=1.6, label=f"{short(IDS[0])} 25-h mean")
    b.plot(tt.iloc[h:-h], st[IDS[-1]]["y"], color=STATION_COLOURS[IDS[-1]], lw=1.6, label=f"{short(IDS[-1])} 25-h mean")
    b.set_ylabel("m, MSL")
    b.set_title("(b) Whole month")
    b.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d %b"))
    b.tick_params(axis="x", labelsize=6.5)
    top_legend(b, ncol=2)
    return fig


# --------------------------------------------------------------------------
# Figure 2: cross-correlation functions
# --------------------------------------------------------------------------
def fig_ccf(R):
    T, P = R["tidal"], R["pairs"]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 4.0), gridspec_kw={"hspace": 0.5})
    a = axes[0]
    for s in IDS[1:]:
        d = T[s]
        a.plot(d["lags"], d["rho"], color=STATION_COLOURS[s], lw=1.3, label=f"{short(s)}: {d['lag_h']:.2f} h")
        a.plot([d["lag_h"]], [d["peak"]], "o", color=STATION_COLOURS[s], ms=4.5, mec="white", mew=0.6)
    a.axhline(0, color=C["axis"], lw=0.7)
    a.axvline(0, color=C["muted"], lw=0.7)
    a.set_xlim(-12, config.MAX_LAG_H)
    a.set_ylim(-1.05, 1.15)
    a.set_xlabel("Lag l (hours)")
    a.set_ylabel("ρ_yx[l]")
    a.set_title("(a) Tidal band")
    top_legend(a, ncol=4)

    b = axes[1]
    s = IDS[-1]
    b.plot(P[s]["lags"], P[s]["rho"], color=C["axis"], lw=1.1, label="raw water level")
    b.plot(T[s]["lags"], T[s]["rho"], color=STATION_COLOURS[s], lw=1.4, label="tidal band")
    k = T[s]["lag_h"]
    for m in (-1, 0, 1, 2):
        cand = k + m * config.M2_PERIOD_H
        if -12 <= cand <= config.MAX_LAG_H:
            b.axvline(cand, color=C["red"] if m else C["green"], lw=0.8, ls="--")
            b.text(cand + 0.3, 0.92, f"{cand:.1f} h", fontsize=6.3, color=C["red"] if m else C["green"])
    b.axhline(0, color=C["axis"], lw=0.7)
    b.set_xlim(-12, config.MAX_LAG_H)
    b.set_ylim(-1.05, 1.15)
    b.set_xlabel("Lag l (hours)")
    b.set_ylabel("ρ_yx[l]")
    b.set_title("(b) Baltimore: candidate delays 12.42 h apart")
    top_legend(b, ncol=2)
    return fig


# --------------------------------------------------------------------------
# Figure 3: properties and alignment
# --------------------------------------------------------------------------
def fig_properties(R, n0=240, hours=60):
    Pp, df, T = R["props"], R["df"], R["tidal"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.5), gridspec_kw={"wspace": 0.22})
    a = axes[0]
    a.plot(Pp["lags"], Pp["ryx"], color=C["violet"], lw=1.3, label="r_yx[l]")
    a.plot(Pp["lags"], Pp["rxy"], color=C["orange"], lw=1.1, ls="--", label="r_xy[l]")
    a.axvline(0, color=C["muted"], lw=0.7)
    a.axhline(0, color=C["axis"], lw=0.7)
    a.set_xlim(-50, 50)
    a.set_ylim(-0.9, 0.9)
    a.set_xlabel("Lag l (hours)")
    a.set_title("(a) Swapping signals mirrors the lag axis")
    a.legend(loc="lower left", fontsize=6.4)

    b = axes[1]
    s = IDS[-1]
    lag = T[s]["lag_h"]
    t = np.arange(len(df))
    sel = slice(n0, n0 + hours)
    b.plot(t[sel] - n0, df[IDS[0]].to_numpy()[sel], color=STATION_COLOURS[IDS[0]], lw=1.1, label="x: Sewells Point")
    b.plot(t[sel] - n0, df[s].to_numpy()[sel], color=STATION_COLOURS[s], lw=0.9, ls=":", label="y: Baltimore")
    b.plot(t[sel] - n0 - lag, df[s].to_numpy()[sel], color=STATION_COLOURS[s], lw=1.4,
           label=f"y advanced by {lag:.2f} h")
    b.set_xlim(0, hours - 12)
    b.set_xlabel("Hours")
    b.set_ylabel("m")
    b.set_title("(b) Undoing the delay aligns the tides")
    b.legend(loc="lower left", fontsize=6.0)
    return fig


# --------------------------------------------------------------------------
# Figure 4: delay versus distance
# --------------------------------------------------------------------------
def fig_speed(R):
    T, sp, ch = R["tidal"], R["speed"], R["chain"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.7), gridspec_kw={"wspace": 0.28, "width_ratios": [1.25, 1]})
    a = axes[0]
    dk = np.array([T[s]["dist_km"] for s in IDS])
    lg = np.array([T[s]["lag_h"] for s in IDS])
    tt = np.linspace(0, 11, 50)
    a.plot(tt, sp["c_kmh"] * tt, color=C["blue"], lw=1.2, label=f"fit: c = {sp['c_ms']:.2f} m/s")
    a.plot(tt, sp["c_published"] * 3.6 * tt, color=C["muted"], lw=1.0, ls="--",
           label=f"√(g·{config.PUBLISHED_MEAN_DEPTH_M} m) = {sp['c_published']:.2f} m/s")
    for s, d_, l_ in zip(IDS, dk, lg):
        a.plot([l_], [d_], "o", color=STATION_COLOURS[s], ms=5, mec="white", mew=0.6)
        a.annotate(short(s), (l_, d_), xytext=(4, -9 if s == IDS[-1] else 3), textcoords="offset points",
                   fontsize=6.4, color=C["ink2"])
    a.set_xlim(-0.3, 11)
    a.set_ylim(-10, 300)
    a.set_xlabel("Delay from cross-correlation (hours)")
    a.set_ylabel("Distance from Sewells Point (km)")
    a.set_title("(a) Distance versus delay")
    a.legend(loc="upper left", fontsize=6.4)

    b = axes[1]
    names = [f"{st['from'][:4]}→{st['to'][:4]}" for st in ch["steps"]]
    vals = [st["lag_h"] for st in ch["steps"]]
    b.bar(range(len(vals)), vals, color=[STATION_COLOURS[s] for s in IDS[1:]], width=0.65)
    for i, (v, st) in enumerate(zip(vals, ch["steps"])):
        b.text(i, v + 0.1, f"{v:.2f} h\nρ={st['peak']:.2f}", ha="center", fontsize=6.2, color=C["ink2"])
    b.set_xticks(range(len(vals)), names, fontsize=6.4)
    b.set_ylim(0, 6)
    b.set_ylabel("Delay (hours)")
    b.set_title("(b) Gauge-to-gauge delays")
    b.grid(axis="x", visible=False)
    return fig


# --------------------------------------------------------------------------
# Figure 5: the wind-driven (subtidal) component
# --------------------------------------------------------------------------
def fig_subtidal(R):
    st, df = R["subtidal"], R["df"]
    h = config.SUBTIDAL_WINDOW_H // 2
    tt = df["time_utc"].iloc[h:-h]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), gridspec_kw={"wspace": 0.25, "width_ratios": [1.3, 1]})
    a = axes[0]
    a.plot(tt, st["x"], color=STATION_COLOURS[IDS[0]], lw=1.3, label=short(IDS[0]))
    for s in IDS[1:]:
        a.plot(tt, st[s]["y"], color=STATION_COLOURS[s], lw=1.1, label=short(s))
    a.set_ylabel("25-h mean level (m)")
    a.set_title("(a) Subtidal water level")
    a.tick_params(axis="x", labelsize=6.2, rotation=0)
    a.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%d"))
    a.set_xlabel("Day of January 2024")
    a.legend(loc="upper left", fontsize=6.0, ncol=2)
    b = axes[1]
    for s in IDS[1:]:
        d = st[s]
        b.plot(d["lags"], d["rho"], color=STATION_COLOURS[s], lw=1.2, label=f"{short(s)}: {d['lag_h']:.1f} h")
    b.axvline(0, color=C["muted"], lw=0.7)
    b.axhline(0, color=C["axis"], lw=0.7)
    b.set_xlim(-48, 48)
    b.set_xlabel("Lag (hours)")
    b.set_title("(b) Cross-correlation of setup")
    b.legend(loc="lower left", fontsize=6.0)
    return fig


def make_all_figures(R):
    return {
        "fig1_raw.png": fig_raw(R),
        "fig2_ccf.png": fig_ccf(R),
        "fig3_properties.png": fig_properties(R),
        "fig4_speed.png": fig_speed(R),
        "fig5_subtidal.png": fig_subtidal(R),
    }
