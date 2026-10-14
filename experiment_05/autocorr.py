"""
autocorr.py
-----------
Pure implementations of the autocorrelation sequence and the tools built on it.

Three independent routes are provided so the report can show that they agree:
the direct definition, convolution of x with its time-reversed copy, and the
Wiener-Khinchin route through the power spectrum.  No function here does any
I/O or plotting.
"""

import numpy as np


# ==========================================================================
# 1. Autocorrelation: three methods
# ==========================================================================
def autocorr_direct(x, max_lag):
    """r_xx[l] = sum_n x[n] x[n - l],  l = 0 ... max_lag   (definition, O(N L))."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    return np.array([np.dot(x[l:], x[: N - l]) for l in range(max_lag + 1)])


def autocorr_conv(x):
    """r_xx[l] = x[l] * x[-l]: convolution with the folded sequence, lags -(N-1) ... N-1."""
    x = np.asarray(x, dtype=float)
    return np.convolve(x, x[::-1])


def autocorr_fft(x, max_lag):
    """Wiener-Khinchin: r_xx = IDFT{|X|^2}, zero-padded to avoid circular wrap-around."""
    x = np.asarray(x, dtype=float)
    nfft = 1 << int(np.ceil(np.log2(2 * len(x) - 1)))
    X = np.fft.rfft(x, nfft)
    return np.fft.irfft(np.abs(X) ** 2, nfft)[: max_lag + 1]


# ==========================================================================
# 2. Normalised autocorrelation (ACF) estimators
# ==========================================================================
def acf(x, max_lag, unbiased=False, demean=True):
    """rho[l] = r[l] / r[0] of the mean-removed series.

    biased:    r[l] / N           (always a valid, decaying estimate; standard choice)
    unbiased:  r[l] / (N - l)     (no bias, but very noisy at large lags)
    """
    x = np.asarray(x, dtype=float)
    if demean:
        x = x - x.mean()
    N = len(x)
    r = autocorr_fft(x, max_lag)
    r = r / (N - np.arange(max_lag + 1)) if unbiased else r / N
    return r / r[0]


def confidence_bound(N, z=1.96):
    """95 % limits of the sample ACF of white noise: ±z / sqrt(N)."""
    return z / np.sqrt(N)


# ==========================================================================
# 3. Features read from the ACF
# ==========================================================================
def first_zero_crossing(rho):
    idx = np.nonzero(rho < 0)[0]
    return int(idx[0]) if len(idx) else None


def lag_below(rho, level):
    idx = np.nonzero(rho < level)[0]
    return int(idx[0]) if len(idx) else None


def dominant_period(rho, search=(1, None)):
    """Lag of the largest ACF value in the search window (the fundamental period)."""
    lo, hi = search
    hi = hi or len(rho) - 1
    seg = rho[lo: hi + 1]
    k = int(np.argmax(seg)) + lo
    if 0 < k < len(rho) - 1:          # parabolic refinement for a sub-sample estimate
        a, b, c = rho[k - 1], rho[k], rho[k + 1]
        den = a - 2 * b + c
        frac = 0.5 * (a - c) / den if den else 0.0
    else:
        frac = 0.0
    return k, float(k + frac), float(rho[k])


def positive_peaks(rho, min_sep):
    """Successive local maxima above zero, at least ``min_sep`` lags apart."""
    peaks = []
    for k in range(1, len(rho) - 1):
        if rho[k] > 0 and rho[k] >= rho[k - 1] and rho[k] >= rho[k + 1]:
            if peaks and k - peaks[-1] < min_sep:
                if rho[k] > rho[peaks[-1]]:
                    peaks[-1] = k
            else:
                peaks.append(k)
    return peaks


def running_mean(x, M):
    """Centred M-point moving average (edges use the available samples)."""
    x = np.asarray(x, dtype=float)
    k = np.ones(M)
    num = np.convolve(x, k, mode="same")
    den = np.convolve(np.ones(len(x)), k, mode="same")
    return num / den
