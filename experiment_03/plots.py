"""
plots.py
--------
Publication-style figures for Experiment 3.

Every function receives already-computed results (from ``analysis``) and returns
a Matplotlib ``Figure``; saving is done by the caller.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

import config
import sampling as sp

# --------------------------------------------------------------------------
# Visual style (identical to Experiments 1 and 2)
# --------------------------------------------------------------------------
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#ffffff", "wash": "#f3f2ee",
}
FULL_W = 6.7
DPI = 220
PREFILTER_COLOURS = {"None": C["red"], "Moving average": C["orange"], "FIR (ideal-like)": C["blue"]}
RECON_COLOURS = {"Zero-order hold": C["orange"], "Linear": C["aqua"], "Sinc (band-limited)": C["blue"]}


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


def khz(ax, axis="x"):
    fmt = plt.FuncFormatter(lambda v, _: f"{v / 1000:g}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


# --------------------------------------------------------------------------
# Figure 1: the recording in time and frequency
# --------------------------------------------------------------------------
def fig_raw(R):
    fs, x, clip = R["fs"], R["x"], R["clip"]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.5), gridspec_kw={"hspace": 0.5})
    a = axes[0]
    t = np.arange(len(x)) / fs
    a.plot(t, x, color=C["blue"], lw=0.4)
    a.set_xlim(0, t[-1])
    a.set_ylim(-1, 1)
    a.set_xlabel("Time (s)")
    a.set_ylabel("Amplitude")
    a.set_title("(a) Waveform, 5 s at 48 kHz")

    b = axes[1]
    f, S = clip["spectrum"]
    step = 8
    b.plot(f[::step], S[::step], color=C["blue"], lw=0.6)
    for rate, col in zip(config.RATES, (C["aqua"], C["orange"], C["red"])):
        b.axvline(rate / 2, color=col, lw=1.0, ls="--", label=f"Nyquist of {rate // 1000} kHz")
    b.axvline(clip["bandwidth"], color=C["violet"], lw=1.0, label="99 % bandwidth")
    b.set_xlim(0, 12000)
    b.set_ylim(-90, 3)
    khz(b)
    b.set_xlabel("Frequency (kHz)")
    b.set_ylabel("dB")
    b.set_title("(b) Spectrum")
    top_legend(b, ncol=4)
    return fig


# --------------------------------------------------------------------------
# Figure 2: sampling and the three reconstructions (zoom on one chirp)
# --------------------------------------------------------------------------
def fig_sampling_concept(R, rate=8000, t0=None, dur=0.004):
    fs, x = R["fs"], R["x"]
    rec = R["recon"][rate]
    D = fs // rate
    i0 = int(np.argmax(np.abs(rec["ref"][4000:-4000]))) + 4000 if t0 is None else int(t0 * fs)
    i0 -= int(dur * fs / 2)
    i0 -= i0 % D
    sel = slice(i0, i0 + int(dur * fs))
    t = (np.arange(len(x)) / fs)[sel] * 1000
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.7), sharex=True, gridspec_kw={"hspace": 0.35})
    a = axes[0]
    a.plot(t, rec["ref"][sel], color=C["ink2"], lw=1.0, label="x_a(t)  band-limited reference")
    ks = np.arange(i0 // D, (i0 + int(dur * fs)) // D)
    stem(a, ks * D / fs * 1000, rec["xs"][ks], C["violet"], f"samples x[n], fs = {rate // 1000} kHz",
         markersize=3.2)
    a.set_ylabel("Amplitude")
    a.set_title("(a) Sampling")
    top_legend(a, ncol=2)

    b = axes[1]
    b.plot(t, rec["ref"][sel], color=C["axis"], lw=2.4, label="reference")
    for m in config.RECON_METHODS:
        b.plot(t, rec[m]["y"][sel], color=RECON_COLOURS[m], lw=1.1,
               drawstyle="steps-post" if m.startswith("Zero") else "default",
               label=f"{m.split(' (')[0]} ({rec[m]['snr']:.1f} dB)")
    b.set_xlabel("Time (ms)")
    b.set_ylabel("Amplitude")
    b.set_title("(b) Reconstruction")
    top_legend(b, ncol=4)
    return fig


# --------------------------------------------------------------------------
# Figure 3: the moving-average filter
# --------------------------------------------------------------------------
def fig_moving_average(R):
    fs, x, ma = R["fs"], R["x"], R["ma"]
    f = np.linspace(1, 24000, 4000)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), gridspec_kw={"wspace": 0.25})
    a = axes[0]
    for M, col in zip(config.MA_LENGTHS, (C["aqua"], C["orange"], C["red"])):
        H = sp.ma_frequency_response(M, f, fs)
        a.plot(f, 20 * np.log10(H + 1e-6), color=col, label=f"M = {M}")
    a.axhline(-3, color=C["muted"], lw=0.7, ls="--")
    a.set_ylim(-45, 3)
    a.set_xlim(0, 24000)
    khz(a)
    a.set_xlabel("Frequency (kHz)")
    a.set_ylabel("|H| (dB)")
    a.set_title("(a) Frequency response at 48 kHz")
    a.legend(loc="lower left", fontsize=6.6)

    b = axes[1]
    fr, Sx = sp.spectrum_db(x, fs)
    _, Sy = sp.spectrum_db(ma[6]["y"], fs)
    step = 8
    b.plot(fr[::step], Sx[::step], color=C["ink2"], lw=0.5, label="input")
    b.plot(fr[::step], Sy[::step] + (Sy.max() - Sx.max()), color=C["orange"], lw=0.6, label="after M = 6")
    b.set_xlim(0, 12000)
    b.set_ylim(-100, 3)
    khz(b)
    b.set_xlabel("Frequency (kHz)")
    b.set_title("(b) Bird-song spectrum, M = 6")
    b.legend(loc="upper right", fontsize=6.6)
    return fig


# --------------------------------------------------------------------------
# Figure 4: aliasing seen in spectrograms
# --------------------------------------------------------------------------
def _spec(ax, x, fs, title, fmax):
    f, t, Sxx = spectrogram(x, fs, nperseg=512 if fs > 10000 else 256, noverlap=None)
    S = 10 * np.log10(Sxx + 1e-14)
    vmax = S.max()
    ax.pcolormesh(t, f, S, cmap="Blues", vmin=vmax - 70, vmax=vmax, shading="auto", rasterized=True)
    ax.set_ylim(0, fmax)
    khz(ax, "y")
    ax.set_title(title, fontsize=7.6)
    ax.grid(False)


def fig_aliasing(R):
    fs, x, pre = R["fs"], R["x"], R["prefilter"]
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 3.9), sharex=True,
                             gridspec_kw={"hspace": 0.42, "wspace": 0.15})
    _spec(axes[0, 0], x, fs, "(a) Original, 48 kHz (0 to 8 kHz shown)", 8000)
    _spec(axes[0, 1], pre[8000]["None"]["xs"], 8000,
          f"(b) 8 kHz, no prefilter (SNR {pre[8000]['None']['snr']:.1f} dB)", 4000)
    _spec(axes[1, 0], pre[4000]["None"]["xs"], 4000,
          f"(c) 4 kHz, no prefilter (SNR {pre[4000]['None']['snr']:.1f} dB)", 2000)
    _spec(axes[1, 1], pre[4000]["FIR (ideal-like)"]["xs"], 4000,
          f"(d) 4 kHz, FIR prefilter (SNR {pre[4000]['FIR (ideal-like)']['snr']:.1f} dB)", 2000)
    for ax in axes[:, 0]:
        ax.set_ylabel("kHz")
    for ax in axes[1]:
        ax.set_xlabel("Time (s)")
    return fig


# --------------------------------------------------------------------------
# Figure 5: SNR summary
# --------------------------------------------------------------------------
def fig_snr(R):
    pre, rec = R["prefilter"], R["recon"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.5), gridspec_kw={"wspace": 0.25})
    a = axes[0]
    rates = config.RATES
    xpos = np.arange(len(rates))
    w = 0.26
    for i, kind in enumerate(config.PREFILTERS):
        vals = [pre[r][kind]["snr"] for r in rates]
        a.bar(xpos + (i - 1) * (w + 0.02), vals, w, color=PREFILTER_COLOURS[kind], label=kind.split(" (")[0])
    a.axhline(0, color=C["axis"], lw=0.8)
    a.set_xticks(xpos, [f"{r // 1000} kHz" for r in rates])
    a.set_ylabel("SNR (dB)")
    a.set_ylim(-12, 48)
    a.set_title("(a) Anti-aliasing prefilter")
    a.grid(axis="x", visible=False)
    a.legend(loc="upper right", fontsize=6.6)

    b = axes[1]
    rates2 = list(rec)
    xpos = np.arange(len(rates2))
    for i, m in enumerate(config.RECON_METHODS):
        vals = [rec[r][m]["snr"] for r in rates2]
        b.bar(xpos + (i - 1) * (w + 0.02), vals, w, color=RECON_COLOURS[m], label=m.split(" (")[0])
        for xp, v in zip(xpos + (i - 1) * (w + 0.02), vals):
            b.text(xp, v + 0.6, f"{v:.0f}", ha="center", fontsize=6.4, color=C["ink2"])
    b.set_xticks(xpos, [f"{r // 1000} kHz" for r in rates2])
    b.set_ylim(0, 48)
    b.set_title("(b) Reconstruction method")
    b.grid(axis="x", visible=False)
    b.legend(loc="upper right", fontsize=6.6)
    return fig


def make_all_figures(R):
    return {
        "fig1_recording.png": fig_raw(R),
        "fig2_sampling.png": fig_sampling_concept(R),
        "fig3_moving_average.png": fig_moving_average(R),
        "fig4_aliasing.png": fig_aliasing(R),
        "fig5_snr.png": fig_snr(R),
    }
