"""
sampling.py
-----------
Pure functions for the moving-average filter, sampling (decimation of the
48 kHz reference) and reconstruction back onto the reference grid.

No function here does any I/O or plotting, so the same code is used by the
batch pipeline, the GUI and the unit tests.
"""

import numpy as np
from scipy.signal import firwin, oaconvolve

import config


# ==========================================================================
# 1. Moving-average filter
# ==========================================================================
def moving_average(x, M, centred=True):
    """y[n] = (1/M) sum_{k=0}^{M-1} x[n-k].

    ``centred=True`` removes the (M-1)/2-sample delay so the output lines up
    with the input; this is how the filter is used before sampling.
    """
    h = np.full(int(M), 1.0 / int(M))
    y = np.convolve(x, h)
    if not centred:
        return y[: len(x)]
    start = (int(M) - 1) // 2
    return y[start: start + len(x)]


def ma_frequency_response(M, f, fs):
    """|H(e^{jω})| = |sin(Mω/2) / (M sin(ω/2))|,  ω = 2π f / fs."""
    w = 2 * np.pi * np.asarray(f, dtype=float) / fs
    with np.errstate(invalid="ignore", divide="ignore"):
        H = np.sin(M * w / 2) / (M * np.sin(w / 2))
    H = np.where(np.isclose(np.sin(w / 2), 0), 1.0, H)
    return np.abs(H)


def ma_3db_frequency(M, fs):
    """Frequency where |H| first falls to 1/sqrt(2), found numerically."""
    f = np.linspace(0, fs / M, 20001)
    H = ma_frequency_response(M, f, fs)
    return float(f[np.argmax(H <= 1 / np.sqrt(2))])


# ==========================================================================
# 2. Sampling
# ==========================================================================
def fir_antialias(fs_ref, fs_new, taps=config.FIR_TAPS, edge=config.FIR_EDGE):
    """Windowed-sinc low-pass with cut-off just below the new Nyquist frequency."""
    return firwin(taps, edge * fs_new / 2, fs=fs_ref, window=("kaiser", 8.0))


def prefilter(x, fs_ref, fs_new, kind):
    D = fs_ref // fs_new
    if kind == "None":
        return np.asarray(x, dtype=float)
    if kind == "Moving average":
        return moving_average(x, D)
    if kind.startswith("FIR"):
        h = fir_antialias(fs_ref, fs_new)
        return oaconvolve(x, h, mode="same")
    raise ValueError(kind)


def sample(x, fs_ref, fs_new, kind="None"):
    """x_s[n] = x_f(n T_s): keep every D-th sample of the (optionally pre-filtered) reference."""
    D = fs_ref // fs_new
    if fs_ref % fs_new:
        raise ValueError("fs_ref must be an integer multiple of fs_new")
    return prefilter(x, fs_ref, fs_new, kind)[::D]


def alias_frequency(f, fs):
    """Apparent frequency of a tone f after sampling at fs: |f - k fs| folded into [0, fs/2]."""
    r = np.mod(f, fs)
    return np.minimum(r, fs - r)


# ==========================================================================
# 3. Reconstruction onto the reference grid
# ==========================================================================
def reconstruct(xs, D, N, method):
    """Rebuild N reference-rate samples from x_s (rate fs_ref / D)."""
    t = np.arange(N) / D                                # reference instants in units of T_s
    if method == "Zero-order hold":
        return np.repeat(xs, D)[:N]
    if method == "Linear":
        return np.interp(t, np.arange(len(xs)), xs)
    if method.startswith("Sinc"):
        return sinc_interpolate(xs, D, N)
    raise ValueError(method)


def sinc_kernel(D, K=config.SINC_HALF_WIDTH, beta=config.SINC_KAISER_BETA):
    """Truncated Whittaker-Shannon kernel sinc(m/D), |m| <= K D, Kaiser-windowed."""
    m = np.arange(-K * D, K * D + 1)
    return np.sinc(m / D) * np.kaiser(len(m), beta)


def sinc_interpolate(xs, D, N):
    """x_r(t) = sum_n x_s[n] sinc((t - nT_s)/T_s), evaluated on the reference grid."""
    up = np.zeros(len(xs) * D)
    up[::D] = xs
    y = oaconvolve(up, sinc_kernel(D), mode="same")
    return y[:N]


# ==========================================================================
# 4. Measures
# ==========================================================================
def bandlimit(x, fs, f_cut):
    """Ideal (brick-wall) low-pass via the FFT: the best any system can do at that rate."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / fs)
    X[f > f_cut] = 0
    return np.fft.irfft(X, len(x))


def snr_db(reference, estimate, trim=0):
    """10 log10( sum r^2 / sum (r - e)^2 ), ignoring ``trim`` samples at each edge."""
    r = np.asarray(reference)[trim: len(reference) - trim]
    e = np.asarray(estimate)[trim: len(estimate) - trim]
    return float(10 * np.log10(np.sum(r ** 2) / np.sum((r - e) ** 2)))


def spectrum_db(x, fs):
    """One-sided magnitude spectrum in dB (relative to its own maximum)."""
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return f, 20 * np.log10(X / X.max() + 1e-12)


def energy_above(x, fs, f_cut):
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return float(P[f > f_cut].sum() / P.sum())


def bandwidth(x, fs, fraction=config.BANDWIDTH_FRACTION):
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return float(f[np.searchsorted(np.cumsum(P) / P.sum(), fraction)])
