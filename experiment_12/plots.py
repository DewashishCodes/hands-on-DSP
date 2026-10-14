"""
plots.py
--------
Publication-style figures for Experiment 12.

Every function receives already-computed results (from ``analysis``) and returns
a Matplotlib ``Figure``; saving is done by the caller.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import config
import iir

# --------------------------------------------------------------------------
# Visual style (identical to Experiments 1 to 11)
# --------------------------------------------------------------------------
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#ffffff", "wash": "#f3f2ee",
}
FCOL = {"butterworth": C["blue"], "chebyshev1": C["orange"], "elliptic": C["aqua"]}
FNAME = {"butterworth": "Butterworth", "chebyshev1": "Chebyshev I", "elliptic": "Elliptic"}
OPEN, CLOSED = C["orange"], C["blue"]
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
# Figure 1: the raw EEG
# --------------------------------------------------------------------------
def fig_raw(R):
    D, A = R["D"], R["alpha"]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.7), gridspec_kw={"hspace": 0.5})
    a = axes[0]
    t = np.arange(len(D["open"]["x"])) / FS
    m = (t >= 20) & (t < 25)
    a.plot(t[m], D["open"]["x"][m] - D["open"]["x"].mean() + 150, color=OPEN, lw=0.7, label="eyes open")
    a.plot(t[m], D["closed"]["x"][m] - D["closed"]["x"].mean() - 150, color=CLOSED, lw=0.7, label="eyes closed")
    a.set_yticks([])
    a.set_xlabel("Time (s)")
    a.set_title(f"(a) Channel {config.CHANNEL}, 5 s of each recording (offset for clarity)")
    top_legend(a, ncol=2)
    b = axes[1]
    for k, col, lab in (("open", OPEN, "eyes open"), ("closed", CLOSED, "eyes closed")):
        f, P = signal.welch(D[k]["x"] - D[k]["x"].mean(), FS, nperseg=int(config.WELCH_SECONDS * FS))
        b.semilogy(f, P, color=col, lw=1.0, label=lab)
    b.axvspan(*config.ALPHA_BAND, color=C["blue"], alpha=0.08, lw=0)
    b.text(10.5, 0.92, "alpha", transform=b.get_xaxis_transform(), ha="center", fontsize=6.4, color=C["blue"])
    b.axvline(config.MAINS_HZ, color=C["red"], lw=0.7, ls=":")
    b.text(config.MAINS_HZ - 1, 0.92, "60 Hz mains", transform=b.get_xaxis_transform(), ha="right", fontsize=6.4,
           color=C["red"])
    b.set_xlim(0, 80)
    b.set_xlabel("Frequency (Hz)")
    b.set_ylabel("PSD (µV²/Hz)")
    b.set_title("(b) Power spectral density")
    top_legend(b, ncol=2)
    return fig


# --------------------------------------------------------------------------
# Figure 2: the three band-pass designs
# --------------------------------------------------------------------------
def fig_designs(R):
    Ds = R["designs"]
    fig = plt.figure(figsize=(FULL_W, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.1, 0.9], wspace=0.38)
    a = fig.add_subplot(gs[0])
    for fam in config.FAMILIES:
        a.plot(Ds[fam]["f"], db(Ds[fam]["H"]), color=FCOL[fam], lw=1.1, label=FNAME[fam])
    a.set_xlim(0, 40)
    a.set_ylim(-90, 5)
    a.set_xlabel("Frequency (Hz)")
    a.set_ylabel("|H| (dB)")
    a.set_title("(a) Magnitude, order 4 prototypes", fontsize=7.8)
    a.legend(loc="upper right", fontsize=6.0)
    b = fig.add_subplot(gs[1])
    for fam in config.FAMILIES:
        m = Ds[fam]["fg"] < 25
        b.plot(Ds[fam]["fg"][m], Ds[fam]["tau"][m] / FS, color=FCOL[fam], lw=1.1)
    b.axvspan(*config.ALPHA_BAND, color=C["wash"], lw=0)
    b.set_ylim(0, 1.0)
    b.set_xlabel("Frequency (Hz)")
    b.set_ylabel("Group delay (s)")
    b.set_title("(b) Group delay", fontsize=7.8)
    c = fig.add_subplot(gs[2])
    t = np.linspace(0, 2 * np.pi, 300)
    c.plot(np.cos(t), np.sin(t), color=C["axis"], lw=0.8)
    z, p, _ = Ds["elliptic"]["zpk"]
    c.plot(z.real, z.imag, "o", mfc="none", mec=C["aqua"], ms=4.5)
    c.plot(p.real, p.imag, "x", color=C["red"], ms=5, mew=1.2)
    c.set_aspect("equal")
    c.set_xlim(-1.15, 1.15)
    c.set_ylim(-1.15, 1.15)
    c.set_title("(c) Elliptic z-plane", fontsize=7.8)
    return fig


# --------------------------------------------------------------------------
# Figure 3: pre-warping and numerical precision
# --------------------------------------------------------------------------
def fig_bilinear(R, D):
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), gridspec_kw={"wspace": 0.28})
    a = axes[0]
    from scipy.signal import buttap, lp2bp_zpk, zpk2sos, sosfreqz
    for warp, col, ls in ((True, C["blue"], "-"), (False, C["red"], "--")):
        z, p, k = buttap(config.ORDER)
        W = iir.prewarp((50.0, 70.0), FS) if warp else 2 * np.pi * np.array([50.0, 70.0])
        z, p, k = lp2bp_zpk(z, p, k, wo=np.sqrt(W[0] * W[1]), bw=W[1] - W[0])
        f, H = sosfreqz(zpk2sos(*iir.bilinear_zpk(z, p, k, FS)), 8192, fs=FS)
        a.plot(f, db(H), color=col, ls=ls, lw=1.2, label="with pre-warping" if warp else "without")
    a.axvspan(50, 70, color=C["wash"], lw=0)
    a.set_ylim(-60, 5)
    a.set_xlim(20, 80)
    a.set_xlabel("Frequency (Hz)")
    a.set_ylabel("|H| (dB)")
    a.set_title("(a) Specified band 50 to 70 Hz")
    a.legend(loc="lower center", fontsize=6.4)
    b = axes[1]
    x = D["closed"]["x"] - D["closed"]["x"].mean()
    d = iir.design_bandpass("butterworth", (9.0, 11.0), FS, 8)
    y_tf = signal.lfilter(*d["ba"], x)
    y_sos = iir.sos_filter(d["sos"], x)
    n = np.arange(len(x)) / FS
    b.semilogy(n, np.abs(y_tf) + 1e-12, color=C["red"], lw=0.6, label="one direct-form equation")
    b.semilogy(n, np.abs(y_sos) + 1e-12, color=C["blue"], lw=0.6, label="cascade of 8 biquads")
    b.set_xlabel("Time (s)")
    b.set_ylabel("|y[n]| (µV)")
    b.set_title("(b) Order-16 band-pass, 9 to 11 Hz")
    b.legend(loc="upper left", fontsize=6.4)
    return fig


# --------------------------------------------------------------------------
# Figure 4: the notch filter
# --------------------------------------------------------------------------
def fig_notch(R):
    N = R["notch"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), gridspec_kw={"wspace": 0.28})
    a = axes[0]
    a.plot(N["f"], db(N["H"]), color=C["violet"], lw=1.2)
    a.set_ylim(-60, 3)
    a.set_xlim(0, 80)
    a.set_xlabel("Frequency (Hz)")
    a.set_ylabel("|H| (dB)")
    a.set_title(f"(a) Notch, r = {N['r']:.3f}, Q = {config.NOTCH_Q:g}")
    ins = a.inset_axes([0.08, 0.15, 0.42, 0.4])
    t = np.linspace(0, 2 * np.pi, 200)
    ins.plot(np.cos(t), np.sin(t), color=C["axis"], lw=0.6)
    ins.plot(np.roots(N["b"]).real, np.roots(N["b"]).imag, "o", mfc="none", mec=C["violet"], ms=3.5)
    ins.plot(np.roots(N["a"]).real, np.roots(N["a"]).imag, "x", color=C["red"], ms=3.5)
    ins.set_aspect("equal")
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_title("z-plane", fontsize=6)
    b = axes[1]
    f = N["f_psd"]
    m = (f >= 40) & (f <= 75)
    b.semilogy(f[m], N["P_before"][m], color=C["axis"], lw=1.6, label="before")
    b.semilogy(f[m], N["P_after"][m], color=C["violet"], lw=1.0, label="after notch")
    b.set_xlabel("Frequency (Hz)")
    b.set_ylabel("PSD (µV²/Hz)")
    b.set_title("(b) Eyes-open EEG around 60 Hz")
    b.legend(loc="upper right", fontsize=6.4)
    return fig


# --------------------------------------------------------------------------
# Figure 5: causal versus zero-phase filtering
# --------------------------------------------------------------------------
def fig_phase(R, t0=30.0, dur=1.5):
    P = R["phase"]
    t = np.arange(len(P["x"])) / FS
    m = (t >= t0) & (t < t0 + dur)
    fig, ax = plt.subplots(figsize=(FULL_W, 2.4))
    ax.plot(t[m], P["x"][m], color=C["axis"], lw=1.0, label="raw (eyes closed)")
    ax.plot(t[m], P["yz"][m], color=C["blue"], lw=1.5, label="zero-phase (forward-backward)")
    ax.plot(t[m], P["yc"][m], color=C["aqua"], lw=1.2, ls="--", label="causal elliptic")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("µV")
    ax.set_title(f"Causal lag {P['lag_ms']:.0f} ms")
    top_legend(ax, ncol=3)
    return fig


# --------------------------------------------------------------------------
# Figure 6: the Berger effect
# --------------------------------------------------------------------------
def fig_alpha(R, t0=20.0, dur=8.0):
    A = R["alpha"]
    fig = plt.figure(figsize=(FULL_W, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.25)
    a = fig.add_subplot(gs[0])
    t = np.arange(len(A["open"]["y"])) / FS
    m = (t >= t0) & (t < t0 + dur)
    for k, col, off, lab in (("open", OPEN, 100, "eyes open"), ("closed", CLOSED, -100, "eyes closed")):
        a.plot(t[m], A[k]["y"][m] + off, color=col, lw=0.7, label=lab)
        a.plot(t[m], A[k]["env"][m] + off, color=C["ink2"], lw=0.6)
    a.set_yticks([])
    a.set_xlabel("Time (s)")
    a.set_title("(a) Alpha band")
    top_legend(a, ncol=2)
    b = fig.add_subplot(gs[1])
    rng = np.random.default_rng(12)
    for i, (k, col) in enumerate((("open", OPEN), ("closed", CLOSED))):
        e = A[k]["epochs"]
        b.plot(i + rng.uniform(-0.12, 0.12, len(e)), e, "o", color=col, ms=3.2, mec="white", mew=0.4)
        b.plot([i - 0.25, i + 0.25], [e.mean(), e.mean()], color=C["ink"], lw=1.4)
    b.set_yscale("log")
    b.set_xticks([0, 1], ["eyes open", "eyes closed"])
    b.set_ylabel("Alpha power per 2 s (µV²)")
    b.set_title(f"(b) {A['ratio']:.1f}× more alpha")
    b.grid(axis="x", visible=False)
    return fig


def make_all_figures(R):
    return {
        "fig1_raw.png": fig_raw(R),
        "fig2_designs.png": fig_designs(R),
        "fig3_bilinear.png": fig_bilinear(R, R["D"]),
        "fig4_notch.png": fig_notch(R),
        "fig5_phase.png": fig_phase(R),
        "fig6_alpha.png": fig_alpha(R),
    }
