"""
plots.py
--------
Publication-style figures for Experiment 9.

Every function receives already-computed results (from ``analysis``) and returns
a Matplotlib ``Figure``; saving is done by the caller.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

import config

# --------------------------------------------------------------------------
# Visual style (identical to Experiments 1 to 8)
# --------------------------------------------------------------------------
C = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#ffffff", "wash": "#f3f2ee",
}
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


def _db(v, floor=-80):
    return np.maximum(20 * np.log10(np.abs(v) / np.max(np.abs(v)) + 1e-12), floor)


# --------------------------------------------------------------------------
# Figure 1: the two inputs
# --------------------------------------------------------------------------
def fig_inputs(R):
    x, h, fs, rm = R["x"], R["h"], R["fs"], R["room"]
    fig, axes = plt.subplots(2, 1, figsize=(FULL_W, 3.4), gridspec_kw={"hspace": 0.5})
    a = axes[0]
    a.plot(np.arange(len(x)) / fs, x, color=C["blue"], lw=0.4)
    a.set_xlim(0, len(x) / fs)
    a.set_ylim(-1.05, 1.05)
    a.set_xlabel("Time (s)")
    a.set_title(f"(a) Dry speech x[n]: L = {len(x):,} samples")
    b = axes[1]
    t = np.arange(len(h)) / fs * 1000
    b.plot(t, _db(h), color=C["violet"], lw=0.4)
    b.axvline(rm["peak_ms"], color=C["orange"], lw=0.9, ls="--")
    b.text(rm["peak_ms"] + 15, -12, "direct sound", fontsize=6.4, color=C["orange"])
    b.set_xlim(0, 1000)
    b.set_ylim(-80, 3)
    b.set_xlabel("Time (ms)")
    b.set_ylabel("20 log|h[n]| (dB)")
    b.set_title(f"(b) Room impulse response h[n]: M = {len(h):,} samples")
    return fig


# --------------------------------------------------------------------------
# Figure 2: the result
# --------------------------------------------------------------------------
def _spec(ax, s, fs, title):
    f, t, S = spectrogram(s, fs, nperseg=1024, noverlap=768)
    S = 10 * np.log10(S + 1e-14)
    ax.pcolormesh(t, f / 1000, S, cmap="Blues", vmin=S.max() - 75, vmax=S.max(), shading="auto", rasterized=True)
    ax.set_ylim(0, 8)
    ax.set_title(title, fontsize=7.6)
    ax.set_xlabel("Time (s)")
    ax.grid(False)


def fig_result(R):
    A, fs, y_ref, x = R["acc"], R["fs"], R["y_ref"], R["x"]
    fig = plt.figure(figsize=(FULL_W, 4.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.55, wspace=0.12)
    a = fig.add_subplot(gs[0, :])
    t = np.arange(len(y_ref)) / fs
    a.plot(t, y_ref, color=C["axis"], lw=1.6, label="published xmas_clap.wav")
    a.plot(t, A["scale"] * A["y"], color=C["orange"], lw=0.4, label="IDFT{X·H} (this work)")
    a.set_xlim(0, t[-1])
    a.set_xlabel("Time (s)")
    a.set_title(f"(a) Output y[n], N = {A['Nlin']:,} = L + M − 1")
    top_legend(a, ncol=2)
    b = fig.add_subplot(gs[1, 0])
    _spec(b, x, fs, "(b) Dry speech")
    b.set_ylabel("kHz")
    c = fig.add_subplot(gs[1, 1], sharey=b)
    _spec(c, A["y"], fs, "(c) Speech in the seminar room")
    plt.setp(c.get_yticklabels(), visible=False)
    return fig


# --------------------------------------------------------------------------
# Figure 3: insufficient zero padding
# --------------------------------------------------------------------------
def fig_aliasing(R):
    A, Al, fs = R["acc"], R["alias"], R["fs"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.6), gridspec_kw={"wspace": 0.25, "width_ratios": [1.6, 1]})
    a = axes[0]
    e = Al["y_alias"] - A["y"][: A["L"]]
    t = np.arange(len(e)) / fs
    a.plot(t, A["y"][: A["L"]], color=C["axis"], lw=0.5, label="linear y[n]")
    a.plot(t, e, color=C["red"], lw=0.5, label=f"error, {A['L']:,}-point circular")
    a.axvspan(0, A["M"] / fs, color=C["red"], alpha=0.07, lw=0)
    a.text(A["M"] / fs / 2, 0.92, "first M − 1 samples:\nthe tail wraps here", transform=a.get_xaxis_transform(),
           ha="center", va="top", fontsize=6.3, color=C["red"])
    a.set_xlim(0, t[-1])
    a.set_xlabel("Time (s)")
    a.set_title("(a) Time aliasing, N = L")
    a.legend(loc="lower right", fontsize=6.2)
    b = axes[1]
    rows = Al["rows"]
    b.semilogy([r["short_by"] / fs * 1000 for r in rows[:-1]], [r["err"] for r in rows[:-1]], "o-",
               color=C["red"], ms=4)
    b.set_xlabel("Padding missing (ms)")
    b.set_ylabel("Relative error")
    b.set_title("(b) Error versus N")
    return fig


# --------------------------------------------------------------------------
# Figure 4: overlap-add and cost
# --------------------------------------------------------------------------
def fig_ola(R):
    O, A = R["ola"], R["acc"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.5), gridspec_kw={"wspace": 0.32})
    a = axes[0]
    B = [o["B"] for o in O]
    a.loglog(B, [o["mult_per_sample"] for o in O], "o-", color=C["blue"], ms=4, label="overlap-add")
    a.axhline(A["mult_direct"] / A["L"], color=C["red"], lw=1.0, ls="--", label="direct (= M)")
    a.axhline(A["mult_fft"] / A["L"], color=C["aqua"], lw=1.0, ls="--", label="one big FFT")
    for o in O:
        a.annotate(f"{o['latency_ms']:.0f} ms", (o["B"], o["mult_per_sample"]), xytext=(3, 4),
                   textcoords="offset points", fontsize=6.2, color=C["ink2"])
    a.set_xlabel("Block length B (samples)")
    a.set_ylabel("Multiplications per output sample")
    a.set_title("(a) Cost and latency of overlap-add")
    a.legend(loc="upper right", fontsize=6.2)
    b = axes[1]
    names = ["direct", "DFT, N = L+M−1", f"DFT, N = {A['Nfast']:,}", f"OLA, B = {config.DEFAULT_BLOCK}"]
    ola = next(o for o in O if o["B"] == config.DEFAULT_BLOCK)
    vals = [A["t_direct"], A["t_exact"], A["t_fast"], ola["time"]]
    cols = [C["red"], C["orange"], C["aqua"], C["blue"]]
    b.barh(range(4), vals, color=cols, height=0.6)
    b.set_xscale("log")
    b.set_yticks(range(4), names, fontsize=6.6)
    b.invert_yaxis()
    for i, v in enumerate(vals):
        b.text(v * 1.15, i, f"{v * 1000:.0f} ms" if v < 1 else f"{v:.1f} s", va="center", fontsize=6.4, color=C["ink2"])
    b.set_xlim(1e-3, 300)
    b.set_xlabel("Computation time (s)")
    b.set_title("(b) Measured time")
    b.grid(axis="y", visible=False)
    return fig


# --------------------------------------------------------------------------
# Figure 5: room acoustics read from h[n]
# --------------------------------------------------------------------------
def fig_room(R):
    rm, fs = R["room"], R["fs"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.5), gridspec_kw={"wspace": 0.25})
    a = axes[0]
    t = np.arange(len(rm["edc"])) / fs
    a.plot(t, rm["edc"], color=C["violet"], lw=1.2, label="Schroeder decay")
    s, c0, i0, i1 = rm["fit20"]
    tt = np.linspace(0, 1, 50)
    a.plot(tt, s * tt + c0, color=C["orange"], lw=0.9, ls="--", label=f"T20 fit: RT60 = {rm['T20']:.2f} s")
    a.axhline(-60, color=C["muted"], lw=0.6, ls=":")
    a.set_ylim(-70, 2)
    a.set_xlim(0, 1)
    a.set_xlabel("Time (s)")
    a.set_ylabel("Energy decay (dB)")
    a.set_title("(a) Reverberation time")
    a.legend(loc="upper right", fontsize=6.4)
    b = axes[1]
    f, H = rm["f"], np.abs(rm["Hf"])
    Hs = np.convolve(H ** 2, np.ones(41) / 41, mode="same") ** 0.5
    b.semilogx(f[1:], 20 * np.log10(H[1:] / Hs.max() + 1e-9), color=C["axis"], lw=0.3)
    b.semilogx(f[1:], 20 * np.log10(Hs[1:] / Hs.max() + 1e-9), color=C["blue"], lw=1.2)
    b.set_xlim(50, 20000)
    b.set_ylim(-60, 8)
    b.set_xlabel("Frequency (Hz)")
    b.set_ylabel("|H(f)| (dB)")
    b.set_title("(b) Frequency response of the room")
    return fig


# --------------------------------------------------------------------------
# Figure 6: deconvolution
# --------------------------------------------------------------------------
def fig_deconv(R):
    Dc, x, fs = R["deconv"], R["x"], R["fs"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.5), gridspec_kw={"wspace": 0.25, "width_ratios": [1, 1.4]})
    a = axes[0]
    e = [r["eps"] for r in Dc["rows"]]
    a.semilogx(e, [r["clean"] for r in Dc["rows"]], "o-", color=C["blue"], ms=4, label="published (16-bit)")
    a.semilogx(e, [r["noisy"] for r in Dc["rows"]], "s-", color=C["orange"], ms=4,
               label=f"with {config.NOISE_SNR_DB:.0f} dB noise")
    a.axhline(Dc["wet_sisnr"], color=C["muted"], lw=0.8, ls=":")
    a.text(e[0], Dc["wet_sisnr"] + 1.5, "reverberant input", fontsize=6.2, color=C["muted"])
    a.invert_xaxis()
    a.set_xlabel("Regularisation ε")
    a.set_ylabel("SNR of recovered speech (dB)")
    a.set_title("(a) Removing the room")
    a.legend(loc="upper left", fontsize=6.2)
    b = axes[1]
    i0, n = int(1.0 * fs), int(0.006 * fs)
    t = (np.arange(i0, i0 + n) / fs - 1.0) * 1000
    xh = Dc["x_hat"]
    k = np.dot(xh, x) / np.dot(xh, xh)
    yw = R["y_ref"][: len(x)]
    kw = np.dot(yw, x) / np.dot(yw, yw)
    b.plot(t, kw * yw[i0: i0 + n], color=C["axis"], lw=1.0, label="reverberant y")
    b.plot(t, x[i0: i0 + n], color=C["ink2"], lw=2.4, label="dry x")
    b.plot(t, k * xh[i0: i0 + n], color=C["aqua"], lw=0.9, ls="--", label=f"recovered (ε = {Dc['best_clean']['eps']:g})")
    b.set_xlabel("Time from 1.00 s (ms)")
    b.set_title("(b) 6 ms of speech")
    b.legend(loc="lower right", fontsize=6.0)
    return fig


def make_all_figures(R):
    return {
        "fig1_inputs.png": fig_inputs(R),
        "fig2_result.png": fig_result(R),
        "fig3_aliasing.png": fig_aliasing(R),
        "fig4_ola.png": fig_ola(R),
        "fig5_room.png": fig_room(R),
        "fig6_deconv.png": fig_deconv(R),
    }
