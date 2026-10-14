"""
plots.py
--------
Publication-style figures for Experiment 11.

Every function receives already-computed results (from ``analysis``) and returns
a Matplotlib ``Figure``; saving is done by the caller.
"""

import matplotlib.pyplot as plt
import numpy as np

import config
import fir

# --------------------------------------------------------------------------
# Visual style (identical to Experiments 1 to 10)
# --------------------------------------------------------------------------
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#ffffff", "wash": "#f3f2ee",
}
WCOL = {"rectangular": C["red"], "hann": C["orange"], "hamming": C["blue"], "blackman": C["aqua"]}
FULL_W = 6.7
DPI = 220
FS = config.FS


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


def db(H):
    return 20 * np.log10(np.abs(H) + 1e-15)


# --------------------------------------------------------------------------
# Figure 1: the raw PPG and its spectrum
# --------------------------------------------------------------------------
def fig_raw(R):
    x, Sp = R["D"]["ppg"], R["spec"]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.5), gridspec_kw={"hspace": 0.5})
    a = axes[0]
    t = np.arange(len(x)) / FS
    m = (t >= 100) & (t < 130)
    a.plot(t[m], x[m], color=C["violet"], lw=0.8)
    a.set_xlabel("Time (s)")
    a.set_ylabel("PLETH (NU)")
    a.set_title("(a) 30 s of the fingertip photoplethysmogram")
    b = axes[1]
    b.plot(Sp["f"], Sp["x"], color=C["violet"], lw=0.7)
    b.axvspan(*config.PULSE_BAND, color=C["blue"], alpha=0.08, lw=0)
    b.axvspan(0.1, config.RESP_CUTOFF, color=C["orange"], alpha=0.12, lw=0)
    b.text(0.12, 0.92, "breathing", transform=b.get_xaxis_transform(), fontsize=6.4, color=C["orange"])
    b.text(1.6, 0.92, "pulse and harmonics", transform=b.get_xaxis_transform(), fontsize=6.4, color=C["blue"])
    b.set_xscale("log")
    b.set_xlim(0.05, 60)
    b.set_ylim(-100, 3)
    b.set_xlabel("Frequency (Hz)")
    b.set_ylabel("dB")
    b.set_title("(b) Spectrum")
    return fig


# --------------------------------------------------------------------------
# Figure 2: window comparison
# --------------------------------------------------------------------------
def fig_windows(R):
    W = R["win"]
    N = config.N_PULSE
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.7), gridspec_kw={"wspace": 0.25, "width_ratios": [1, 1.5]})
    a = axes[0]
    n = np.arange(N)
    for w in config.WINDOWS:
        a.plot(n, fir.window(w, N), color=WCOL[w], lw=1.2, label=w.capitalize())
    a.set_xlabel("n")
    a.set_ylabel("w[n]")
    a.set_ylim(0, 1.15)
    a.set_title(f"(a) Windows, N = {N}")
    a.legend(loc="lower center", fontsize=6.3)
    b = axes[1]
    for w in config.WINDOWS:
        d = W[w]
        b.plot(d["f"], db(d["H"]), color=WCOL[w], lw=0.9, label=f"{w.capitalize()}: {d['stop_db']:.0f} dB")
    b.set_xlim(0, 15)
    b.set_ylim(-110, 5)
    b.axvline(config.PULSE_BAND[1], color=C["muted"], lw=0.7, ls=":")
    b.set_xlabel("Frequency (Hz)")
    b.set_ylabel("|H| (dB)")
    b.set_title(f"(b) Low-pass at {config.PULSE_BAND[1]:g} Hz")
    b.legend(loc="upper right", fontsize=6.3)
    return fig


# --------------------------------------------------------------------------
# Figure 3: the two designed filters
# --------------------------------------------------------------------------
def fig_filters(R):
    F, K = R["F"], R["kaiser"]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.5), gridspec_kw={"wspace": 0.4, "width_ratios": [1, 1.25, 1]})
    a = axes[0]
    b = F["pulse"]["b"]
    n = np.arange(len(b)) - (len(b) - 1) / 2
    a.plot(n / FS, b, color=C["blue"], lw=1.0)
    a.axhline(0, color=C["axis"], lw=0.6)
    a.set_xlabel("(n − M/2)/fs (s)")
    a.set_title("(a) Band-pass h[n]", fontsize=7.8)
    c = axes[1]
    for name, col, lab in (("pulse", C["blue"], "band-pass 0.5 to 4 Hz"), ("resp", C["orange"], "low-pass 0.5 Hz")):
        c.plot(F[name]["f"], db(F[name]["H"]), color=col, lw=1.0, label=lab)
    fk, Hk = np.fft.rfftfreq(1 << 16, 1 / FS), np.fft.rfft(K["b"], 1 << 16)
    c.plot(fk, db(Hk), color=C["aqua"], lw=0.8, ls="--", label=f"Kaiser N = {K['N']}")
    c.set_xscale("log")
    c.set_xlim(0.05, 30)
    c.set_ylim(-100, 5)
    c.set_xlabel("Frequency (Hz)")
    c.set_ylabel("|H| (dB)")
    c.set_title("(b) |H(f)|", fontsize=7.8)
    c.legend(loc="lower right", bbox_to_anchor=(1.0, 1.08), ncol=1, fontsize=5.6, borderaxespad=0.1)
    d = axes[2]
    for name, col in (("pulse", C["blue"]), ("resp", C["orange"])):
        fg, tau = F[name]["fg"], F[name]["tau"]
        d.plot(fg[fg < 6], np.clip(tau[fg < 6] / FS, 0, 2.5), color=col, lw=1.2)
    d.set_ylim(0, 2.5)
    d.set_xlabel("Frequency (Hz)")
    d.set_ylabel("Group delay (s)")
    d.set_title("(c) Constant delay", fontsize=7.8)
    return fig


# --------------------------------------------------------------------------
# Figure 4: the filtered signals
# --------------------------------------------------------------------------
def fig_filtered(R, t0=100, dur=30):
    x, A = R["D"]["ppg"], R["app"]
    t = np.arange(len(x)) / FS
    m = (t >= t0) & (t < t0 + dur)
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.6), sharex=True, gridspec_kw={"hspace": 0.3})
    a = axes[0]
    a.plot(t[m], x[m] - x.mean(), color=C["axis"], lw=1.0, label="raw (mean removed)")
    a.plot(t[m], A["y_pulse"][m], color=C["blue"], lw=1.1, label="band-pass output")
    bt = A["beats"][(A["beats"] / FS >= t0) & (A["beats"] / FS < t0 + dur)]
    a.plot(bt / FS, A["y_pulse"][bt], "v", color=C["red"], ms=3.5, label="detected beats")
    a.set_title("(a) Pulse")
    top_legend(a, ncol=3)
    b = axes[1]
    b.plot(t[m], x[m] - x.mean(), color=C["axis"], lw=0.8, label="raw")
    b.plot(t[m], A["y_resp"][m], color=C["orange"], lw=1.6, label="low-pass output")
    ref = R["D"]["breaths"]
    for r in ref[(ref / FS >= t0) & (ref / FS < t0 + dur)]:
        b.axvline(r / FS, color=C["green"], lw=0.6, ls=":")
    b.plot([], [], color=C["green"], lw=0.6, ls=":", label="annotated breaths")
    br = A["breaths"][(A["breaths"] / FS >= t0) & (A["breaths"] / FS < t0 + dur)]
    b.plot(br / FS, A["y_resp"][br], "o", color=C["orange"], ms=3.5, mec="white", label="detected")
    b.set_xlabel("Time (s)")
    b.set_title("(b) Breathing")
    top_legend(b, ncol=4)
    return fig


# --------------------------------------------------------------------------
# Figure 5: rates against the references
# --------------------------------------------------------------------------
def fig_rates(R):
    A = R["app"]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.4), sharex=True, gridspec_kw={"hspace": 0.32})
    a = axes[0]
    a.plot(A["t_hr"], A["pulse_ref"], color=C["axis"], lw=2.0, label="monitor PULSE")
    a.plot(A["t_hr"], A["hr"], color=C["blue"], lw=1.0, label="FIR band-pass + peaks")
    a.set_ylim(80, 100)
    a.set_ylabel("beats / min")
    a.set_title(f"(a) Heart rate (MAE {A['hr_mae']:.1f} bpm)")
    top_legend(a, ncol=2)
    b = axes[1]
    b.plot(A["t_rr"], A["rr_ref"], color=C["axis"], lw=2.0, label="manual annotations")
    b.plot(A["t_rr"], A["rr_mon"], color=C["muted"], lw=0.8, ls="--", label="monitor RESP")
    b.plot(A["t_rr"], A["rr"], color=C["orange"], lw=1.1, label="FIR low-pass + peaks")
    b.set_ylim(10, 30)
    b.set_ylabel("breaths / min")
    b.set_xlabel("Time (s)")
    b.set_title(f"(b) Breathing rate (MAE {A['rr_mae']:.1f})")
    top_legend(b, ncol=3)
    return fig


def make_all_figures(R):
    return {
        "fig1_raw.png": fig_raw(R),
        "fig2_windows.png": fig_windows(R),
        "fig3_filters.png": fig_filters(R),
        "fig4_filtered.png": fig_filtered(R),
        "fig5_rates.png": fig_rates(R),
    }
