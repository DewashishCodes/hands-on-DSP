"""
xcorr.py
--------
Pure implementations of cross-correlation and time-delay estimation.

Convention used throughout:  r_yx[l] = sum_n y[n] x[n - l].
If y is a delayed copy of x, y[n] = x[n - D], then r_yx peaks at l = D > 0.
No function here does any I/O or plotting.
"""

import numpy as np


# ==========================================================================
# 1. Cross-correlation: three methods
# ==========================================================================
def crosscorr_direct(y, x, max_lag):
    """r_yx[l] = sum_n y[n] x[n - l],  l = -max_lag ... max_lag   (definition)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    N = len(x)
    out = []
    for l in range(-max_lag, max_lag + 1):
        if l >= 0:
            out.append(np.dot(y[l:], x[: N - l]))
        else:
            out.append(np.dot(y[: N + l], x[-l:]))
    return np.arange(-max_lag, max_lag + 1), np.array(out)


def crosscorr_conv(y, x):
    """r_yx[l] = y[l] * x[-l]: convolution with the folded reference, lags -(N-1) ... N-1."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    return np.arange(-(len(x) - 1), len(y)), np.convolve(y, x[::-1])


def crosscorr_fft(y, x, max_lag):
    """r_yx = IDFT{ Y X* }, zero-padded so that no circular wrap-around occurs."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    N = len(x)
    nfft = 1 << int(np.ceil(np.log2(2 * N - 1)))
    r = np.fft.irfft(np.fft.rfft(y, nfft) * np.conj(np.fft.rfft(x, nfft)), nfft)
    r = np.r_[r[-max_lag:], r[: max_lag + 1]]
    return np.arange(-max_lag, max_lag + 1), r


# ==========================================================================
# 2. Normalised cross-correlation coefficient
# ==========================================================================
def ccf(y, x, max_lag):
    """rho_yx[l] = r_yx[l] / sqrt(r_xx[0] r_yy[0]) of the mean-removed signals, in [-1, 1]."""
    y = np.asarray(y, float) - np.mean(y)
    x = np.asarray(x, float) - np.mean(x)
    lags, r = crosscorr_fft(y, x, max_lag)
    return lags, r / np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))


# ==========================================================================
# 3. Time-delay estimation
# ==========================================================================
def peak_lag(lags, rho, search=None):
    """Lag of the largest correlation, refined to a fraction of a sample by a parabola."""
    lags = np.asarray(lags)
    m = np.ones(len(lags), bool) if search is None else (lags >= search[0]) & (lags <= search[1])
    idx = np.nonzero(m)[0]
    k = idx[np.argmax(rho[idx])]
    frac = 0.0
    if 0 < k < len(rho) - 1:
        a, b, c = rho[k - 1], rho[k], rho[k + 1]
        den = a - 2 * b + c
        frac = 0.5 * (a - c) / den if den else 0.0
    return int(lags[k]), float(lags[k] + frac), float(rho[k])


def local_peaks(lags, rho, search=None):
    """All local maxima (candidate delays) inside the search window, strongest first."""
    lags = np.asarray(lags)
    out = []
    for k in range(1, len(rho) - 1):
        if rho[k] > rho[k - 1] and rho[k] >= rho[k + 1] and rho[k] > 0:
            if search is None or search[0] <= lags[k] <= search[1]:
                out.append((int(lags[k]), float(rho[k])))
    return sorted(out, key=lambda t: -t[1])


def moving_average(x, M):
    """Centred M-point moving average with edge correction (removes the tides when M = 25 h)."""
    x = np.asarray(x, float)
    k = np.ones(M)
    return np.convolve(x, k, "same") / np.convolve(np.ones(len(x)), k, "same")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))
