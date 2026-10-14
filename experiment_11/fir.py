"""
fir.py
------
FIR filter design by the window method, implemented from first principles,
and the difference-equation implementation of the filters.

No function here does any I/O or plotting.
"""

import numpy as np


# ==========================================================================
# 1. Windows  w[n], 0 <= n <= N-1
# ==========================================================================
def window(name, N, beta=None):
    n = np.arange(N)
    M = N - 1
    if name == "rectangular":
        return np.ones(N)
    if name == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * n / M)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / M)
    if name == "blackman":
        return 0.42 - 0.5 * np.cos(2 * np.pi * n / M) + 0.08 * np.cos(4 * np.pi * n / M)
    if name == "kaiser":
        return np.i0(beta * np.sqrt(1 - (2 * n / M - 1) ** 2)) / np.i0(beta)
    raise ValueError(name)


# textbook properties (Oppenheim and Schafer, Table 7.2): peak sidelobe of the window,
# minimum stopband attenuation of the designed filter, approximate main-lobe width
WINDOW_TABLE = {
    "rectangular": {"sidelobe_db": -13, "stopband_db": -21, "mainlobe": 4.0},
    "hann":        {"sidelobe_db": -31, "stopband_db": -44, "mainlobe": 8.0},
    "hamming":     {"sidelobe_db": -41, "stopband_db": -53, "mainlobe": 8.0},
    "blackman":    {"sidelobe_db": -57, "stopband_db": -74, "mainlobe": 12.0},
}


# ==========================================================================
# 2. Ideal impulse responses and windowed designs
# ==========================================================================
def ideal_lowpass(fc, fs, N):
    """h_d[n] = 2 fc/fs · sinc(2 fc/fs (n - M/2)): the ideal low-pass, delayed by M/2 to be causal."""
    n = np.arange(N) - (N - 1) / 2
    return 2 * fc / fs * np.sinc(2 * fc / fs * n)


def ideal_bandpass(f1, f2, fs, N):
    """Difference of two ideal low-passes."""
    return ideal_lowpass(f2, fs, N) - ideal_lowpass(f1, fs, N)


def design_lowpass(fc, fs, N, win="hamming", beta=None):
    h = ideal_lowpass(fc, fs, N) * window(win, N, beta)
    return h / h.sum()                                   # unit gain at DC


def design_bandpass(f1, f2, fs, N, win="hamming", beta=None):
    h = ideal_bandpass(f1, f2, fs, N) * window(win, N, beta)
    fc = np.sqrt(f1 * f2)                                # normalise the gain at the band centre
    H = np.abs(np.sum(h * np.exp(-2j * np.pi * fc / fs * np.arange(N))))
    return h / H


def kaiser_parameters(atten_db, transition_hz, fs):
    """Kaiser's formulas: beta from the attenuation, length from attenuation and transition width."""
    A = atten_db
    beta = 0.1102 * (A - 8.7) if A > 50 else (0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21) if A >= 21 else 0.0)
    dw = 2 * np.pi * transition_hz / fs
    N = int(np.ceil((A - 8) / (2.285 * dw))) + 1
    return N + (N % 2 == 0), beta                        # odd length (type I filter)


# ==========================================================================
# 3. The filter as a difference equation
# ==========================================================================
def fir_filter(b, x):
    """y[n] = sum_{k=0}^{N-1} b_k x[n-k]   (non-recursive difference equation, zero initial state)."""
    b, x = np.asarray(b, float), np.asarray(x, float)
    y = np.zeros(len(x))
    for k, bk in enumerate(b):                           # one shifted, scaled copy of x per tap
        if bk:
            y[k:] += bk * x[: len(x) - k]
    return y


def compensate_delay(y, N):
    """Advance the output by the constant group delay (N-1)/2 of a linear-phase filter."""
    d = (N - 1) // 2
    return np.r_[y[d:], np.zeros(d)]


# ==========================================================================
# 4. Frequency-domain analysis
# ==========================================================================
def freq_response(b, fs, nfft=1 << 16):
    H = np.fft.rfft(b, nfft)
    return np.fft.rfftfreq(nfft, 1 / fs), H


def group_delay(b, fs, nfft=1 << 14):
    """tau(w) = -d phase / d w, from the DFT of n·b[n] (exact for FIR filters), in samples."""
    n = np.arange(len(b))
    B = np.fft.rfft(b, nfft)
    Bn = np.fft.rfft(n * b, nfft)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = np.real(Bn / B)
    return np.fft.rfftfreq(nfft, 1 / fs), tau


def measure_lowpass(b, fc, fs, guard):
    """Passband ripple (dB), stopband attenuation (dB) and the -3 dB / -40 dB transition of a low-pass."""
    f, H = freq_response(b, fs)
    A = 20 * np.log10(np.abs(H) + 1e-15)
    pb = f <= fc - guard
    sb = f >= fc + guard
    i3 = int(np.argmax(A <= -3))
    i40 = int(np.argmax(A <= -40)) if np.any(A <= -40) else len(A) - 1
    return {"ripple_db": float(A[pb].max() - A[pb].min()), "stop_db": float(A[sb].max()),
            "f3": float(f[i3]), "f40": float(f[i40]), "transition": float(f[i40] - f[i3])}


def is_linear_phase(b):
    return bool(np.allclose(b, b[::-1]))
